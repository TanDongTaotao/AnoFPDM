"""Synthetic domain translation with domain adaptation using dilated UNet."""

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
from obtain_hyperpara import obtain_hyperpara, get_mask_batch_FPDM, get_mask_batch_FPDM_dual_threshold, get_mask_batch_FPDM_with_snr_weighting
from evaluate import get_stats, evaluate, logging_metrics

from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    set_seed_for_reproducibility(args.seed)
    logger.configure()
    logger.log(f"args: {args}")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True
    args.multi_class = True if args.num_classes > 2 else False

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )

    # Set domain adaptation mode for inference
    if hasattr(model, 'enable_domain_adaptation'):
        model.enable_domain_adaptation = True
        logger.log("Domain adaptation enabled for inference")
    elif hasattr(model, 'module') and hasattr(model.module, 'enable_domain_adaptation'):
        model.module.enable_domain_adaptation = True
        logger.log("Domain adaptation enabled for inference (DDP)")

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
            
        elif args.name == "atlas":
            # model 290000; w = 30; forward_steps = 600; unweighted
            # thr_01 = 0.6869481205940247
            # diff_min = torch.tensor([0.0708], device=dist_util.dev())
            # diff_max = torch.tensor([1.0770], device=dist_util.dev())
            # model 290000; w = 20; forward_steps = 600; unweighted
            thr_01 = 0.7285396456718445
            diff_min = torch.tensor([0.0392], device=dist_util.dev())
            diff_max = torch.tensor([0.8555], device=dist_util.dev())
            
            
        logger.log(f"diff_min: {diff_min}, diff_max: {diff_max}, thr_01: {thr_01}")


    logger.log(f"starting to inference with domain adaptation...")

    
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

        # Domain label for inference (target domain = 1)
        domain_label = torch.ones(source.shape[0], dtype=torch.long, device=dist_util.dev()) * args.target_domain

        model_kwargs_reverse = {
            "threshold": -1, 
            "clf_free": True, 
            "null": args.null,
            "domain_label": domain_label
        }
        model_kwargs0 = {
            "y": y0, 
            "threshold": -1, 
            "clf_free": True,
            "domain_label": domain_label
        }

        # inference with domain adaptation

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
            # 根据参数选择使用SNR权重或双阈值策略
            if getattr(args, 'enable_snr_weighting', False):
                # 使用SNR权重聚合
                pred_mask, pred_mask_all, pred_lab, pred_map, _ = get_mask_batch_FPDM_with_snr_weighting(
                    xstarts,
                    source,
                    args.modality,
                    thr_01,
                    diff_min,
                    diff_max,
                    args.image_size,
                    device=dist_util.dev(),
                    # SNR权重参数
                    enable_snr_weighting=True,
                    snr_smoothing=getattr(args, 'snr_smoothing', 0.05),
                    temporal_decay=getattr(args, 'temporal_decay', 0.98),
                    min_weight=getattr(args, 'min_weight', 0.3),
                    max_weight=getattr(args, 'max_weight', 1.5),
                    consistency_weight=getattr(args, 'consistency_weight', 0.6),
                    sensitivity_weight=getattr(args, 'sensitivity_weight', 0.4),
                    aggregation_mode=getattr(args, 'aggregation_mode', 'robust_weighted'),
                    # 原有参数
                    median_filter=args.median_filter,
                    t_e_ratio=ratio,
                    last_only=args.last_only,
                    interval=args.subset_interval,
                    use_gradient_sam=args.use_gradient_sam,
                    use_gradient_para_sam=args.use_gradient_para_sam,
                    forward_steps=args.forward_steps,
                )
            elif getattr(args, 'enable_dual_threshold', False):
                # 使用双阈值策略
                pred_mask, pred_mask_all, pred_lab, pred_map, _ = get_mask_batch_FPDM_dual_threshold(
                    xstarts,
                    source,
                    args.modality,
                    thr_01,
                    diff_min,
                    diff_max,
                    args.image_size,
                    device=dist_util.dev(),
                    # 双阈值参数
                    low_quant_offset=getattr(args, 'low_quant_offset', -0.05),
                    high_quant_offset=getattr(args, 'high_quant_offset', 0.05),
                    entropy_weight=getattr(args, 'entropy_weight', 0.3),
                    entropy_threshold=getattr(args, 'entropy_threshold', 0.5),
                    # 原有参数
                    median_filter=args.median_filter,
                    t_e_ratio=ratio,
                    last_only=args.last_only,
                    interval=args.subset_interval,
                    use_gradient_sam=args.use_gradient_sam,
                    use_gradient_para_sam=args.use_gradient_para_sam,
                    forward_steps=args.forward_steps,
                )
            else:
                # 使用原始方法
                pred_mask, pred_mask_all, pred_lab, pred_map, _ = get_mask_batch_FPDM(
                    xstarts,
                    source,
                    args.modality,
                    thr_01,
                    diff_min,
                    diff_max,
                    args.image_size,
                    device=dist_util.dev(),
                    median_filter=args.median_filter,
                    t_e_ratio=ratio,
                    last_only=args.last_only,
                    interval=args.subset_interval,
                    use_gradient_sam=args.use_gradient_sam,
                    use_gradient_para_sam=args.use_gradient_para_sam,
                    forward_steps=args.forward_steps,
                )

            all_sources.append(source.cpu())
            all_masks.append(mask.cpu())
            all_pred_maps.append(pred_map.cpu())
            all_pred_masks_all.append(pred_mask_all.cpu())

            Y[n].append(mask.cpu())
            PRED_Y[n].append(pred_mask.cpu())

        # save images
        if args.save_images:
            for i in range(source.shape[0]):
                source_i = source[i].cpu().numpy()
                mask_i = mask[i].cpu().numpy()
                pred_map_i = pred_map[i].cpu().numpy()

                # save source
                np.save(
                    os.path.join(image_subfolder, f"source_{k}_{i}.npy"),
                    source_i,
                )
                # save mask
                np.save(
                    os.path.join(image_subfolder, f"mask_{k}_{i}.npy"),
                    mask_i,
                )
                # save pred_map
                np.save(
                    os.path.join(image_subfolder, f"pred_map_{k}_{i}.npy"),
                    pred_map_i,
                )

    # evaluate
    for n, ratio in enumerate(args.t_e_ratio):
        y_true = torch.cat(Y[n], dim=0).numpy()
        y_pred = torch.cat(PRED_Y[n], dim=0).numpy()

        logger.log(f"evaluating at ratio {ratio} ...")
        stats = get_stats(y_true, y_pred)
        logging.log_metrics(stats, ratio)

    logger.log("evaluation done.")


def create_argparser():
    defaults = dict(
        data_dir="",
        name="",
        model_dir="",
        model_num="",
        ema=True,
        image_dir="",
        batch_size=1,
        batch_size_val=1,
        num_batches_val=0,
        seed=42,
        modality="t1",
        d_reverse=False,
        forward_steps=600,
        dynamic_clip=False,
        null=False,
        median_filter=False,
        last_only=False,
        subset_interval=1,
        use_gradient_sam=False,
        use_gradient_para_sam=False,
        save_images=False,
        use_weighted_sampler=False,
        t_e_ratio=[0.5],
        # Domain adaptation specific parameters
        target_domain=1,  # 0 for source domain, 1 for target domain
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

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
    
    # Domain adaptation parameters
    parser.add_argument(
        "--target_domain",
        type=int,
        default=1,
        help="Target domain label for inference (0=source, 1=target)",
    )
    
    # 双阈值策略参数
    parser.add_argument(
        "--enable_dual_threshold",
        action="store_true",
        help="Enable dual threshold strategy for enhanced anomaly detection",
    )
    parser.add_argument(
        "--low_quant_offset",
        type=float,
        default=-0.05,
        help="Low threshold quantile offset (relative to original quantile point)",
    )
    parser.add_argument(
        "--high_quant_offset",
        type=float,
        default=0.05,
        help="High threshold quantile offset (relative to original quantile point)",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.3,
        help="Weight for local entropy in final mask fusion",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Threshold for local entropy binarization",
    )
    
    # SNR权重策略参数
    parser.add_argument(
        "--enable_snr_weighting",
        action="store_true",
        help="Enable SNR proxy weighting for sub-anomaly map aggregation",
    )
    parser.add_argument(
        "--snr_smoothing",
        type=float,
        default=0.02,
        help="SNR smoothing parameter to avoid division by zero",
    )
    parser.add_argument(
        "--temporal_decay",
        type=float,
        default=0.95,
        help="Temporal decay factor for time-step weighting",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.7,
        help="Minimum weight value for SNR weighting",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=1.2,
        help="Maximum weight value for SNR weighting",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.8,
        help="Weight for reconstruction consistency in SNR calculation",
    )
    parser.add_argument(
        "--sensitivity_weight",
        type=float,
        default=0.2,
        help="Weight for anomaly sensitivity in SNR calculation",
    )
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="robust_weighted",
        choices=["weighted_mean", "weighted_sum", "robust_weighted"],
        help="Aggregation mode for SNR weighted sub-anomaly maps",
    )

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()