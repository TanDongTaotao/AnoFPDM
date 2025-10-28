"""Synthetic domain translation from a source 2D domain to a target using dilated UNet with visualization."""

import argparse
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch.distributed as dist
import torch
import torch.nn.functional as F

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


def aggregate_reconstructions(xstarts, source, modality, t_e_ratio=1, last_only=False, interval=-1):
    """
    聚合重建图像，类似于异常检测中的聚合方式
    
    Args:
        xstarts: 包含 'xstart' 和 'xstart_null' 的字典
        source: 原始图像
        modality: 模态索引
        t_e_ratio: 时间步比例
        last_only: 是否只使用最后一步
        interval: 间隔采样
    
    Returns:
        guided_recon: 引导重建图像
        unguided_recon: 无引导重建图像
    """
    device = source.device
    batch_size = source.shape[0]
    
    # 计算差异来确定最佳时间步
    diff = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
    diff_flat = torch.mean(diff, dim=(3, 4))  # batch_size x sample_steps x n_modality
    
    guided_recons = []
    unguided_recons = []
    
    for sample_num in range(batch_size):
        # 获取每个模态的最佳时间步
        diff_i = diff_flat[sample_num, ...]  # sample_steps x n_modality
        t_e_i = torch.argmax(diff_i, dim=0)  # n_modality
        
        if t_e_ratio != 1:
            t_e_i = torch.round(t_e_i * t_e_ratio).to(torch.int64)
        
        t_s_i = torch.tensor([0, 0], device=device)
        if last_only:
            t_s_i = t_e_i - 1
        
        # 为每个模态聚合重建
        guided_recon_sample = torch.zeros_like(source[sample_num])
        unguided_recon_sample = torch.zeros_like(source[sample_num])
        
        for mod_idx, mod in enumerate(modality):
            # 获取时间步范围
            start_step = t_s_i[mod_idx]
            end_step = t_e_i[mod_idx]
            
            # 提取重建序列
            guided_subset = xstarts["xstart"][sample_num, start_step:end_step, mod_idx, ...]
            unguided_subset = xstarts["xstart_null"][sample_num, start_step:end_step, mod_idx, ...]
            
            # 间隔采样
            if interval != -1 and interval > 0:
                guided_subset = guided_subset[::interval, ...]
                unguided_subset = unguided_subset[::interval, ...]
                
                # 确保包含最后一步
                if end_step > start_step:
                    guided_subset = torch.cat([
                        guided_subset,
                        xstarts["xstart"][sample_num, end_step-1:end_step, mod_idx, ...]
                    ], dim=0)
                    unguided_subset = torch.cat([
                        unguided_subset,
                        xstarts["xstart_null"][sample_num, end_step-1:end_step, mod_idx, ...]
                    ], dim=0)
            
            # 聚合（平均）
            if guided_subset.shape[0] > 0:
                guided_recon_sample[mod] = torch.mean(guided_subset, dim=0)
                unguided_recon_sample[mod] = torch.mean(unguided_subset, dim=0)
            else:
                # 如果没有有效步骤，使用原始图像
                guided_recon_sample[mod] = source[sample_num, mod]
                unguided_recon_sample[mod] = source[sample_num, mod]
        
        guided_recons.append(guided_recon_sample.unsqueeze(0))
        unguided_recons.append(unguided_recon_sample.unsqueeze(0))
    
    guided_recon = torch.cat(guided_recons, dim=0)
    unguided_recon = torch.cat(unguided_recons, dim=0)
    
    return guided_recon, unguided_recon


def create_heatmap_overlay(source_img, anomaly_map, alpha=0.6):
    """
    在原图上叠加异常热力图
    
    Args:
        source_img: 原始图像 (H, W)
        anomaly_map: 异常热力图 (H, W)
        alpha: 热力图透明度
    
    Returns:
        overlay: 叠加后的图像
    """
    # 归一化图像到 [0, 1]
    source_norm = (source_img - source_img.min()) / (source_img.max() - source_img.min() + 1e-8)
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # 创建热力图颜色映射
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('anomaly', colors, N=n_bins)
    
    # 将异常图转换为RGB
    anomaly_rgb = cmap(anomaly_norm.cpu().numpy())[:, :, :3]  # 去掉alpha通道
    
    # 将原图转换为RGB（灰度图复制到三个通道）
    source_rgb = np.stack([source_norm.cpu().numpy()] * 3, axis=-1)
    
    # 叠加
    overlay = (1 - alpha) * source_rgb + alpha * anomaly_rgb
    
    return overlay


def visualize_sample(source, guided_recon, unguided_recon, anomaly_map, pred_mask, true_mask, 
                    sample_idx, modality, save_path):
    """
    可视化单个样本的2x3布局图像
    
    Args:
        source: 原始图像 (C, H, W)
        guided_recon: 引导重建 (C, H, W)
        unguided_recon: 无引导重建 (C, H, W)
        anomaly_map: 异常热力图 (1, H, W)
        pred_mask: 预测掩码 (1, H, W)
        true_mask: 真实掩码 (1, H, W)
        sample_idx: 样本索引
        modality: 模态索引列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sample {sample_idx} - Anomaly Detection Results', fontsize=16)
    
    # 选择主要模态进行显示（通常是第一个模态）
    main_modality = modality[0] if len(modality) > 0 else 0
    
    # 转换为numpy并移到CPU
    source_np = source[main_modality].cpu().numpy()
    guided_recon_np = guided_recon[main_modality].cpu().numpy()
    unguided_recon_np = unguided_recon[main_modality].cpu().numpy()
    anomaly_map_np = anomaly_map[0].cpu().numpy()
    pred_mask_np = pred_mask[0].cpu().numpy()
    true_mask_np = true_mask[0].cpu().numpy()
    
    # 第一行
    # 原始图像
    axes[0, 0].imshow(source_np, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 引导重建
    axes[0, 1].imshow(guided_recon_np, cmap='gray')
    axes[0, 1].set_title('Guided Reconstruction')
    axes[0, 1].axis('off')
    
    # 无引导重建
    axes[0, 2].imshow(unguided_recon_np, cmap='gray')
    axes[0, 2].set_title('Unguided Reconstruction')
    axes[0, 2].axis('off')
    
    # 第二行
    # 异常热力图叠加
    overlay = create_heatmap_overlay(torch.tensor(source_np), torch.tensor(anomaly_map_np))
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Anomaly Heatmap Overlay')
    axes[1, 0].axis('off')
    
    # 预测掩码
    axes[1, 1].imshow(pred_mask_np, cmap='Reds', alpha=0.8)
    axes[1, 1].imshow(source_np, cmap='gray', alpha=0.3)
    axes[1, 1].set_title('Predicted Mask')
    axes[1, 1].axis('off')
    
    # 真实掩码
    axes[1, 2].imshow(true_mask_np, cmap='Greens', alpha=0.8)
    axes[1, 2].imshow(source_np, cmap='gray', alpha=0.3)
    axes[1, 2].set_title('Ground Truth Mask')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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

        # 聚合重建图像
        guided_recon, unguided_recon = aggregate_reconstructions(
            xstarts, source, args.modality, 
            t_e_ratio=args.t_e_ratio[0] if args.t_e_ratio else 1,
            last_only=args.last_only,
            interval=args.subset_interval
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
                    diffusion_steps=args.diffusion_steps,
                    w=args.w,
                )
            else:
                # 使用原有的双阈值策略
                pred_mask, pred_mask_all, pred_lab, pred_map, _ = get_mask_batch_FPDM_dual_threshold(
                    xstarts,
                    source,
                    args.modality,
                    thr_01,
                    diff_min,
                    diff_max,
                    args.image_size,
                    device=dist_util.dev(),
                    # 双阈值策略参数
                    enable_dual_threshold=getattr(args, 'enable_dual_threshold', False),
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

            # 可视化每个样本
            if dist.get_rank() == 0:  # 只在主进程中保存图像
                for sample_idx in range(source.shape[0]):
                    save_path = os.path.join(
                        image_subfolder, 
                        f"visualization_batch_{k}_sample_{sample_idx}_ratio_{ratio:.2f}.png"
                    )
                    
                    visualize_sample(
                        source[sample_idx],
                        guided_recon[sample_idx],
                        unguided_recon[sample_idx],
                        pred_map[sample_idx],
                        pred_mask_all[sample_idx],
                        mask[sample_idx],
                        sample_idx,
                        args.modality,
                        save_path
                    )
                    
                    logger.log(f"Saved visualization: {save_path}")

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
        unet_ver="dilated",  # Use dilated UNet by default
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
    
    # 可视化相关参数
    parser.add_argument(
        "--visualization_output_dir",
        type=str,
        default="./visualization_outputs",
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--num_samples_to_visualize",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()