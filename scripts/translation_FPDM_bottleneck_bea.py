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
    logger.log(f"Using Bottleneck Boundary-Enhanced Attention (Bottleneck BEA) UNet with use_bottleneck_bea={args.use_bottleneck_bea}")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True
    args.multi_class = True if args.num_classes > 2 else False
    
    # Force Bottleneck BEA UNet version
    args.unet_ver = "bottleneck_bea"

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )

    data_test = get_data_iter(
        args.name,
        args.data_dir,
        mixed=False,
        batch_size=args.batch_size,
        split="test",
        ret_lab=True,
        logger=logger,
    )

    logger.log(f"obtaining hyperparameters ...")
    hyperpara = obtain_hyperpara(
        args,
        model,
        diffusion,
        data_test,
        args.num_batches_val,
        args.batch_size_val,
        args.modality,
        args.t_e_ratio,
        args.w,
    )

    logger.log(f"sampling ...")
    all_images = []
    all_labels = []
    all_pred_maps = []
    all_pred_masks = []
    all_sources = []
    all_masks = []
    all_terms = {}
    model_kwargs = {}
    if args.class_cond:
        model_kwargs["y"] = None

    if dist.get_rank() == 0:
        logger.log(f"hyperpara: {hyperpara}")

    for k, batch in enumerate(data_test):
        if k >= args.num_batches:
            break
        logger.log(f"batch {k}")
        batch_size = batch[0].shape[0]
        model_kwargs["y"] = batch[1] if args.class_cond else None

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                torch.zeros_like(batch[1]) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, batch[1])
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # get pred_map and pred_mask
        pred_map, pred_mask_all = get_mask_batch_FPDM(
            batch[0],
            model,
            diffusion,
            hyperpara,
            args.modality,
            args.t_e_ratio,
            args.w,
            args.median_filter,
            args.dynamic_clip,
            args.last_only,
        )

        gathered_pred_maps = [
            torch.zeros_like(pred_map) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_pred_maps, pred_map)
        all_pred_maps.extend([pred_map.cpu().numpy() for pred_map in gathered_pred_maps])

        gathered_pred_masks = [
            torch.zeros_like(pred_mask_all) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_pred_masks, pred_mask_all)
        all_pred_masks.extend(
            [pred_mask.cpu().numpy() for pred_mask in gathered_pred_masks]
        )

        gathered_sources = [
            torch.zeros_like(batch[0]) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_sources, batch[0])
        all_sources.extend([source.cpu().numpy() for source in gathered_sources])

        gathered_masks = [
            torch.zeros_like(batch[2]) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_masks, batch[2])
        all_masks.extend([mask.cpu().numpy() for mask in gathered_masks])

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    # save pred_map and pred_mask
    pred_map_arr = np.concatenate(all_pred_maps, axis=0)
    pred_map_arr = pred_map_arr[: args.num_samples]
    pred_mask_arr = np.concatenate(all_pred_masks, axis=0)
    pred_mask_arr = pred_mask_arr[: args.num_samples]
    source_arr = np.concatenate(all_sources, axis=0)
    source_arr = source_arr[: args.num_samples]
    mask_arr = np.concatenate(all_masks, axis=0)
    mask_arr = mask_arr[: args.num_samples]

    if dist.get_rank() == 0:
        if args.save_data:
            source_path = os.path.join(logger.get_dir(), f"source.npy")
            logger.log(f"saving to {source_path}")
            np.save(source_path, source_arr)

            mask_path = os.path.join(logger.get_dir(), f"mask.npy")
            logger.log(f"saving to {mask_path}")
            np.save(mask_path, mask_arr)

            pred_map_path = os.path.join(logger.get_dir(), f"pred_map.npy")
            logger.log(f"saving to {pred_map_path}")
            np.save(pred_map_path, pred_map_arr)

            pred_mask_path = os.path.join(logger.get_dir(), f"pred_mask_all.npy")
            logger.log(f"saving to {pred_mask_path}")
            np.save(pred_mask_path, pred_mask_arr)

        # evaluation
        logger.log(f"evaluation ...")
        stats = get_stats(pred_mask_arr, mask_arr, args.multi_class)
        evaluate(stats, logger)
        logging_metrics(stats, logger)

        if len(all_terms) > 0:
            for key in all_terms.keys():
                for k in range(len(all_terms[key])):
                    all_terms_path = os.path.join(
                        logger.get_dir(), f"{key}_terms_{k}.npy"
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
        unet_ver="bottleneck_bea",  # Force Bottleneck BEA UNet
        use_bottleneck_bea=True,  # Enable Bottleneck BEA by default
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
        help="ratio of t_e",
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