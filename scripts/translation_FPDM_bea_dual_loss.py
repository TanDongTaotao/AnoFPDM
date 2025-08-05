import argparse
import os
import pathlib
import sys
import time

sys.path.append(os.path.realpath("./"))

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from common import get_latest_model_path_in_directory, set_seed_for_reproducibility
from data import get_data_iter
from obtain_hyperpara import obtain_hyperpara, get_mask_batch_FPDM
from evaluate import get_stats, evaluate, logging_metrics
from sample import sample
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    set_seed_for_reproducibility(args.seed)
    logger.configure()
    logger.log(f"args: {args}")
    logger.log(f"Using BEA Dual Loss UNet with use_bea={args.use_bea}")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        logger.log(f"num_classes: {args.num_classes}")
        args.multi_class = True if args.num_classes > 2 else False
    else:
        args.multi_class = False

    # Force BEA dual loss UNet version
    args.unet_ver = "bea_dual_loss"
    args.use_bea = True
    # Force class_cond=False for inference to avoid classifier layers
    args.class_cond = False
    
    # Read model using dual loss version
    model_path, _ = get_latest_model_path_in_directory(
        args.model_dir, model_number=args.model_num, ema=args.ema
    )
    logger.log(f"Model path: {model_path}")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"), strict=False)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if dist.get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )

    logger.log("loading data...")
    data_test = get_data_iter(
        args.name,
        args.data_dir,
        mixed=True,
        batch_size=args.batch_size,
        split="test",
        seed=args.seed,
        logger=logger,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    logger.log(f"Validation: starting to get threshold and abe range ...")

    if args.num_batches_val != 0:
        data_val = get_data_iter(
            args.name,
            args.data_dir,
            mixed=True,
            batch_size=args.batch_size_val,
            split="val",
            seed=args.seed,
            logger=logger,
            use_weighted_sampler=args.use_weighted_sampler,
        )

        thr_01, diff_min, diff_max, n_min = obtain_hyperpara(
            data_val, diffusion, model, args, dist_util.dev()
        )
        logger.log(f"diff_min: {diff_min}, diff_max: {diff_max}, thr_01: {thr_01}, n_min: {n_min}")
    else:
        logger.log(f"loading hyperparameters for {args.name} with forward_steps {args.forward_steps}...")
        if args.name == "brats":
            # model 210000; w = 2; forward_steps = 600
            if args.forward_steps == 999:
                thr_01 = 0.9993147253990173
                diff_min = torch.tensor([0.0022, 0.0010], device=dist_util.dev())
                diff_max = torch.tensor([0.0551, 0.0388], device=dist_util.dev())
            elif args.forward_steps == 600:
                thr_01 = 0.9948798418045044
                diff_min = torch.tensor([5.5484e-05, 3.4732e-05], device=dist_util.dev())
                diff_max = torch.tensor([0.0509, 0.0397], device=dist_util.dev())
            
        elif args.name == "atlas":
            # model 290000; w = 20; forward_steps = 600; unweighted
            thr_01 = 0.7285396456718445
            diff_min = torch.tensor([0.0392], device=dist_util.dev())
            diff_max = torch.tensor([0.8555], device=dist_util.dev())
            
        logger.log(f"diff_min: {diff_min}, diff_max: {diff_max}, thr_01: {thr_01}")

    logger.log(f"starting to inference ...")

    logging = logging_metrics(logger)
    Y = [[] for _ in range(len(args.t_e_ratio))]
    PRED_Y = [[] for _ in range(len(args.t_e_ratio))]
    
    k = 0
    while k < args.num_batches:
        all_sources = []
        all_masks = []
        all_pred_maps = []
        all_terms = {"xstart_null": [], "xstart": []}
        all_pred_masks_all = []

        k += 1

        source, mask, lab = data_test.__iter__().__next__()

        logger.log(
            f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}..."
        )
        logger.log(f"device: {torch.cuda.current_device()}")

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(
            f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}"
        )

        y0 = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 for healthy
        y0 = y0.reshape(-1, 1).squeeze().to(dist_util.dev())

        model_kwargs_reverse = {"threshold": -1, "clf_free": True, "null": args.null}
        model_kwargs0 = {"y": y0, "threshold": -1, "clf_free": True}

        # inference

        # obtain xstart and xstart_null
        xstarts = diffusion.calc_pred_xstart_loop(
            model,
            source,
            args.w,
            modality=args.modality,
            d_reverse=args.d_reverse,
            sample_steps=args.forward_steps,
            model_kwargs=model_kwargs0,
            model_kwargs_reverse=model_kwargs_reverse,
            dynamic_clip=args.dynamic_clip,
        )

        # collect metrics
        for n, ratio in enumerate(args.t_e_ratio):
            pred_mask, pred_mask_all, pred_lab, pred_map, _ = get_mask_batch_FPDM(
                xstarts,
                source,
                args.modality,
                thr_01,
                diff_min,
                diff_max,
                args.image_size,
                median_filter=args.median_filter,
                device=dist_util.dev(),
                t_e_ratio=ratio,
                last_only=args.last_only,
                interval=args.subset_interval,
                use_gradient_sam=args.use_gradient_sam,
                use_gradient_para_sam=args.use_gradient_para_sam,
                forward_steps=args.forward_steps,
                diffusion_steps=args.diffusion_steps,
                w=args.w,
            )
            
            Y[n].append(lab)
            PRED_Y[n].append(pred_lab)
            eval_metrics = evaluate(mask, pred_mask, source, pred_map)
            eval_metrics_ano = evaluate(mask, pred_mask_all, source, pred_map, lab)
            cls_metrics = get_stats(Y[n], PRED_Y[n])
            logger.log(f"ratio: {ratio}")
            logging.logging(eval_metrics, eval_metrics_ano, cls_metrics, k)

        if args.save_data:
            logger.log("collecting metrics...")
            for key in all_terms.keys():
                gathered_terms = [
                    torch.zeros_like(xstarts[key]) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_terms, xstarts[key])
                all_terms[key].extend(
                    [term.cpu().numpy() for term in gathered_terms]
                )

            gathered_source = [
                torch.zeros_like(source) for _ in range(dist.get_world_size())
            ]
            gathered_mask = [
                torch.zeros_like(mask) for _ in range(dist.get_world_size())
            ]
            gathered_pred_map = [
                torch.zeros_like(pred_map) for _ in range(dist.get_world_size())
            ]
            gathered_pred_masks_all = [
                torch.zeros_like(pred_mask_all)
                for _ in range(dist.get_world_size())
            ]
            
            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_mask, mask)
            dist.all_gather(gathered_pred_map, pred_map)
            dist.all_gather(gathered_pred_masks_all, pred_mask_all)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            all_pred_maps.extend(
                [pred_map.cpu().numpy() for pred_map in gathered_pred_map]
            )
            all_pred_masks_all.extend(
                [pred_mask_all.cpu().numpy() for pred_mask_all in gathered_pred_masks_all]
            )

            all_sources = np.concatenate(all_sources, axis=0)
            all_sources_path = os.path.join(image_subfolder, f"source_{k}.npy")
            np.save(all_sources_path, all_sources)

            all_masks = np.concatenate(all_masks, axis=0)
            all_masks_path = os.path.join(image_subfolder, f"mask_{k}.npy")
            np.save(all_masks_path, all_masks)

            
            all_pred_maps = np.concatenate(all_pred_maps, axis=0)
            all_pred_maps_path = os.path.join(image_subfolder, f"pred_map_{k}.npy")
            np.save(all_pred_maps_path, all_pred_maps)

            all_pred_masks_all = np.concatenate(all_pred_masks_all, axis=0)
            all_pred_masks_all_path = os.path.join(image_subfolder, f"pred_mask_all_{k}.npy")
            np.save(all_pred_masks_all_path, all_pred_masks_all)

            for key in all_terms.keys():
                all_terms[key] = np.concatenate(all_terms[key], axis=0)
                all_terms_path = os.path.join(image_subfolder, f"{key}_{k}.npy")
                np.save(all_terms_path, all_terms[key])

    dist.barrier()
    logger.log(f"evaluation complete")


def create_argparser():
    defaults = dict(
        name="",
        data_dir="",
        image_dir="",
        model_dir="",
        batch_size=32,
        forward_steps=600,
        model_num=None,
        ema=False,
        null=False,
        save_data=False,
        num_batches_val=2,
        batch_size_val=100,
        d_reverse=True, # deterministic encoding or not
        median_filter=True,
        dynamic_clip=False, 
        last_only=False,
        subset_interval=-1,
        seed=0,  # reproduce
        use_weighted_sampler=False,
        use_gradient_sam=False,
        use_gradient_para_sam=False,
        unet_ver="bea_dual_loss",  # Force BEA dual loss UNet
        use_bea=True,  # Enable BEA by default
        num_samples=-1,  # Number of samples to generate, -1 for all
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=[0, 3],  # flair as default
    )
    parser.add_argument(
        "--t_e_ratio",
        type=float,
        nargs="+",
        default=[1],
    )
    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1,  # disabled in default
    )
    
    parser.add_argument(
        "--num_batches",
        type=int,
        help="number of batches to process",
        default=1,
    )
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()