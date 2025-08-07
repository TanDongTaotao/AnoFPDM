"""Synthetic domain translation from a source 2D domain to a target using Heterogeneous Autoencoder (HAE) UNet V2."""

import argparse
import os
import pathlib

import numpy as np
import torch.distributed as dist
import torch

from common import read_model_and_diffusion, set_seed_for_reproducibility
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)

from data import get_data_iter
from obtain_hyperpara import obtain_hyperpara, get_mask_batch_FPDM
from evaluate import get_stats, evaluate, logging_metrics

from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    set_seed_for_reproducibility(args.seed)
    logger.configure()
    logger.log(f"args: {args}")
    logger.log(f"Using Heterogeneous Autoencoder (HAE) UNet V2 with bottleneck MLP layers")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True
    args.multi_class = True if args.num_classes > 2 else False
    
    # Force HAE V2 UNet version
    args.unet_ver = "hae_v2"

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )

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

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
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
            elif args.forward_steps == 300:
                thr_01 = 0.9948798418045044
                diff_min = torch.tensor([5.5484e-05, 3.4732e-05], device=dist_util.dev())
                diff_max = torch.tensor([0.0509, 0.0397], device=dist_util.dev())
            else:
                raise ValueError(f"Unsupported forward_steps: {args.forward_steps}")
        elif args.name == "isles":
            # model 210000; w = 2; forward_steps = 600
            if args.forward_steps == 999:
                thr_01 = 0.9993147253990173
                diff_min = torch.tensor([0.0022, 0.0010], device=dist_util.dev())
                diff_max = torch.tensor([0.0551, 0.0388], device=dist_util.dev())
            elif args.forward_steps == 600:
                thr_01 = 0.9948798418045044
                diff_min = torch.tensor([5.5484e-05, 3.4732e-05], device=dist_util.dev())
                diff_max = torch.tensor([0.0509, 0.0397], device=dist_util.dev())
            elif args.forward_steps == 300:
                thr_01 = 0.9948798418045044
                diff_min = torch.tensor([5.5484e-05, 3.4732e-05], device=dist_util.dev())
                diff_max = torch.tensor([0.0509, 0.0397], device=dist_util.dev())
            else:
                raise ValueError(f"Unsupported forward_steps: {args.forward_steps}")
        else:
            raise ValueError(f"Unsupported dataset: {args.name}")

    logger.log(f"Test: starting to translate ...")

    all_pred_masks_all = []
    all_terms = {}

    for k in range(args.num_batches):
        logger.log(f"batch {k}")
        batch, cond = next(data_test)
        batch = batch.to(dist_util.dev())
        cond = cond.to(dist_util.dev())

        logger.log(f"batch shape: {batch.shape}")
        logger.log(f"cond shape: {cond.shape}")

        if args.subset_interval > 0:
            batch = batch[::args.subset_interval]
            cond = cond[::args.subset_interval]

        logger.log(f"batch shape after subset: {batch.shape}")
        logger.log(f"cond shape after subset: {cond.shape}")

        # get mask
        pred_masks_all, terms = get_mask_batch_FPDM(
            diffusion,
            model,
            batch,
            cond,
            args,
            thr_01,
            diff_min,
            diff_max,
            n_min,
            dist_util.dev(),
        )

        all_pred_masks_all.append(pred_masks_all)

        for key in terms.keys():
            if key not in all_terms:
                all_terms[key] = []
            all_terms[key].append(terms[key])

        if args.save_data:
            if dist.get_rank() == 0:
                all_pred_masks_all_path = os.path.join(
                    logger.get_dir(), f"pred_masks_all_{k}.npy"
                )
                np.save(all_pred_masks_all_path, pred_masks_all)

                for key in terms.keys():
                    terms_path = os.path.join(
                        logger.get_dir(), f"{key}_terms_{k}.npy"
                    )
                    np.save(terms_path, terms[key])

    # Concatenate all results
    all_pred_masks_all = np.concatenate(all_pred_masks_all, axis=0)
    for key in all_terms.keys():
        all_terms[key] = np.concatenate(all_terms[key], axis=0)

    logger.log(f"all_pred_masks_all shape: {all_pred_masks_all.shape}")

    # Evaluation
    if dist.get_rank() == 0:
        logger.log(f"starting evaluation ...")
        
        # Get statistics
        stats = get_stats(
            all_pred_masks_all,
            all_terms,
            args.modality,
            args.t_e_ratio,
            logger,
        )
        
        # Evaluate and log metrics
        metrics = evaluate(
            all_pred_masks_all,
            all_terms,
            args.modality,
            args.t_e_ratio,
            logger,
        )
        
        logging_metrics(metrics, logger)
        
        if args.save_data:
            # Save final results
            all_pred_masks_all_path = os.path.join(
                logger.get_dir(), "pred_masks_all_final.npy"
            )
            np.save(all_pred_masks_all_path, all_pred_masks_all)

            for key in all_terms.keys():
                all_terms_path = os.path.join(
                    logger.get_dir(), f"{key}_terms_final.npy"
                )
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
        unet_ver="hae_v2",  # Force HAE V2 UNet
        use_hae=True,  # Enable HAE by default
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
        help="weight for clf-free samples",
        default=1,  # disabled in default
    )
    


    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()