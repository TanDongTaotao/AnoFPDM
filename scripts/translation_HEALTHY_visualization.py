"""针对 anoddpm 的可视化推理脚本：
从源域 2D 图像生成目标健康图像，并可视化原图、重建图、异常热力图、预测掩码与真实掩码。

参照 dilated 的可视化脚本风格实现，支持 Simplex/Gaussian 噪声与固定/自动阈值。
"""

import argparse
import os
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
from evaluate import get_stats, evaluate, logging_metrics
from sample import sample
from obtain_hyperpara import obtain_optimal_threshold, get_mask_batch

from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def create_heatmap_overlay(source_img, anomaly_map, alpha=0.6):
    """
    在原图上叠加异常热力图

    Args:
        source_img: 原始图像 (H, W)
        anomaly_map: 异常热力图 (H, W)
        alpha: 热力图透明度

    Returns:
        overlay: 叠加后的 RGB 图像 (H, W, 3)
    """
    # 归一化到 [0, 1]
    source_norm = (source_img - source_img.min()) / (source_img.max() - source_img.min() + 1e-8)
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # 颜色映射（蓝→青→黄→红）
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('anomaly', colors, N=n_bins)

    # 异常图转换为 RGB（去掉 alpha 通道）
    anomaly_rgb = cmap(anomaly_norm.cpu().numpy())[:, :, :3]

    # 原图转为 RGB（灰度复制到三个通道）
    source_rgb = np.stack([source_norm.cpu().numpy()] * 3, axis=-1)

    # 叠加
    overlay = (1 - alpha) * source_rgb + alpha * anomaly_rgb
    return overlay


def visualize_sample(source, target, anomaly_map, pred_mask, true_mask, sample_idx, modality, save_path):
    """
    可视化单个样本的 2x3 布局：
    第一行：原始图像、重建（目标健康图像）、异常热力图叠加
    第二行：预测掩码、真实掩码、异常热力图（灰度）

    Args:
        source: 原始图像 (C, H, W)
        target: 目标健康图像 (C, H, W)
        anomaly_map: 异常热力图 (1, H, W)
        pred_mask: 预测掩码 (1, H, W)
        true_mask: 真实掩码 (1, H, W)
        sample_idx: 样本索引
        modality: 模态索引列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sample {sample_idx} - Anoddpm Visualization', fontsize=16)

    # 选择主要模态进行显示（通常是第一个模态）
    main_modality = modality[0] if isinstance(modality, (list, tuple)) and len(modality) > 0 else (modality if isinstance(modality, int) else 0)

    # 转换为 numpy 并移到 CPU
    source_np = source[main_modality].cpu().numpy()
    target_np = target[main_modality].cpu().numpy()
    anomaly_map_np = anomaly_map[0].cpu().numpy()
    pred_mask_np = pred_mask[0].cpu().numpy()
    true_mask_np = true_mask[0].cpu().numpy()

    # 第一行：原图、重建、热力图叠加
    axes[0, 0].imshow(source_np, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_np, cmap='gray')
    axes[0, 1].set_title('Translated Healthy')
    axes[0, 1].axis('off')

    overlay = create_heatmap_overlay(torch.tensor(source_np), torch.tensor(anomaly_map_np))
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Anomaly Heatmap Overlay')
    axes[0, 2].axis('off')

    # 第二行：预测掩码、真实掩码、异常图（灰度）
    axes[1, 0].imshow(pred_mask_np, cmap='Reds', alpha=0.8)
    axes[1, 0].imshow(source_np, cmap='gray', alpha=0.3)
    axes[1, 0].set_title('Predicted Mask')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(true_mask_np, cmap='Greens', alpha=0.8)
    axes[1, 1].imshow(source_np, cmap='gray', alpha=0.3)
    axes[1, 1].set_title('Ground Truth Mask')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(anomaly_map_np, cmap='magma')
    axes[1, 2].set_title('Anomaly Map (MSE)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = create_argparser().parse_args()

    # 分布式与随机种子
    dist_util.setup_dist()
    set_seed_for_reproducibility(args.seed)
    logger.configure()
    logger.log(f"args: {args}")

    # 输出目录
    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    vis_output_dir = args.visualization_output_dir if args.visualization_output_dir else image_subfolder
    pathlib.Path(vis_output_dir).mkdir(parents=True, exist_ok=True)

    # 读取模型与扩散过程
    logger.log("reading models ...")
    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
    if args.num_classes:
        args.class_cond = True

    # 噪声类型
    if args.noise_type == "simplex":
        from noise import generate_simplex_noise
        from simplex import Simplex_CLASS

        simplex = Simplex_CLASS()
        noise_fn = lambda x, t: generate_simplex_noise(
            simplex,
            x,
            t,
            False,
            in_channels=args.in_channels,
            octave=6,
            persistence=0.8,
            frequency=64,
        )
    elif args.noise_type == "gaussian":
        noise_fn = None
    else:
        raise ValueError(f"Unknown noise type: {args.noise_type}")

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

    # 选择阈值：验证集自动搜索或使用预设
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

        opt_thr, dice_max_val = obtain_optimal_threshold(
            data_val,
            diffusion,
            model,
            args,
            dist_util.dev(),
            guided=False,
            ddib=False,
            noise_fn=noise_fn,
            use_ddpm=args.use_ddpm,
        )
        logger.log(f"optimal threshold: {opt_thr}, dice_max_val: {dice_max_val}")
    else:
        # 使用固定阈值（与原 anoddpm 脚本保持一致）
        # brats ddpm 200 simplex 下的经验阈值
        opt_thr = args.opt_thr if args.opt_thr > 0 else 0.08

    logging = logging_metrics(logger)
    Y = []
    PRED_Y = []

    k = 0
    while k < args.num_batches:
        k += 1

        source, mask, lab = data_test.__iter__().__next__()
        logger.log(
            f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}..."
        )

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(
            f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}"
        )

        # 前向加噪（ddpm：t = sample_steps - 1），并采样目标健康图像
        t = torch.tensor([args.sample_steps - 1] * source.shape[0], device=dist_util.dev())
        ep = noise_fn(source, t) if args.noise_type == "simplex" else None
        noise = diffusion.q_sample(source, t=t, noise=ep)

        Y.append(lab)

        target, _ = sample(
            model,
            diffusion,
            noise=noise,
            w=args.w,
            sample_shape=source.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
            noise_fn=noise_fn,
            ddpm=args.use_ddpm,
        )

        pred_mask, pred_map, pred_lab = get_mask_batch(source, target, opt_thr, args.modality)
        PRED_Y.append(pred_lab)

        # 评估与日志
        eval_metrics = evaluate(mask, pred_mask, source, pred_map)
        eval_metrics_ano = evaluate(mask, pred_mask, source, pred_map, lab)
        cls_metrics = get_stats(Y, PRED_Y)
        logging.logging(eval_metrics, eval_metrics_ano, cls_metrics, k)

        # 可视化（仅在主进程保存）
        if dist.get_rank() == 0:
            num_to_vis = min(args.num_samples_to_visualize, source.shape[0])
            for sample_idx in range(num_to_vis):
                save_path = os.path.join(
                    vis_output_dir,
                    f"visualization_batch_{k}_sample_{sample_idx}_steps_{args.sample_steps}.png",
                )
                visualize_sample(
                    source[sample_idx],
                    target[sample_idx],
                    pred_map[sample_idx],
                    pred_mask[sample_idx],
                    mask[sample_idx],
                    sample_idx,
                    args.modality,
                    save_path,
                )
                logger.log(f"Saved visualization: {save_path}")

    dist.barrier()
    logger.log("anoddpm visualization complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        name="",
        image_dir="",
        model_dir="",
        seed=0,
        batch_size=32,
        sample_steps=1000,
        use_ddpm=True,
        model_num=None,
        ema=False,
        dynamic_clip=False,
        save_data=False,
        num_batches_val=2,
        batch_size_val=100,
        use_weighted_sampler=False,
        noise_type="gaussian",
        opt_thr=-1,
        visualization_output_dir="",
        num_samples_to_visualize=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=0,  # flair as default
    )

    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1.0,  # disabled in default
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        help="number of batches to visualize",
        default=1,
    )

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()