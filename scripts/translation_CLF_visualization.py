import argparse
import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from common import read_classifier, read_model_and_diffusion
from data import get_data_iter
from evaluate import evaluate, get_stats, logging_metrics
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    classifier_defaults,
    model_and_diffusion_defaults,
)
from obtain_hyperpara import get_mask_batch, obtain_optimal_threshold
from sample import sample


def create_heatmap_overlay(source_img, anomaly_map, alpha=0.6):
    source_norm = (source_img - source_img.min()) / (source_img.max() - source_img.min() + 1e-8)
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    cmap = LinearSegmentedColormap.from_list("anomaly", ["blue", "cyan", "yellow", "red"], N=256)
    anomaly_rgb = cmap(anomaly_norm.cpu().numpy())[:, :, :3]
    source_rgb = np.stack([source_norm.cpu().numpy()] * 3, axis=-1)
    overlay = (1 - alpha) * source_rgb + alpha * anomaly_rgb
    return overlay


def visualize_sample(source, target, anomaly_map, pred_mask, true_mask, sample_idx, modality, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} - CLF Guided Visualization", fontsize=16)

    main_modality = modality[0] if isinstance(modality, (list, tuple)) and len(modality) > 0 else (modality if isinstance(modality, int) else 0)

    source_np = source[main_modality].cpu().numpy()
    target_np = target[main_modality].cpu().numpy()
    anomaly_map_np = anomaly_map[0].cpu().numpy()
    pred_mask_np = pred_mask[0].cpu().numpy()
    true_mask_np = true_mask[0].cpu().numpy()

    axes[0, 0].imshow(source_np, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_np, cmap="gray")
    axes[0, 1].set_title("Translated Healthy")
    axes[0, 1].axis("off")

    overlay = create_heatmap_overlay(torch.tensor(source_np), torch.tensor(anomaly_map_np))
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("Anomaly Heatmap Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pred_mask_np, cmap="Reds", alpha=0.8)
    axes[1, 0].imshow(source_np, cmap="gray", alpha=0.3)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(true_mask_np, cmap="Greens", alpha=0.8)
    axes[1, 1].imshow(source_np, cmap="gray", alpha=0.3)
    axes[1, 1].set_title("Ground Truth Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(anomaly_map_np, cmap="magma")
    axes[1, 2].set_title("Anomaly Map (MSE)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"args: {args}")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    vis_output_dir = args.visualization_output_dir if args.visualization_output_dir else image_subfolder
    pathlib.Path(vis_output_dir).mkdir(parents=True, exist_ok=True)

    logger.log("reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True
    if (
        hasattr(args, "timestep_respacing")
        and (args.timestep_respacing is None or args.timestep_respacing == "")
        and hasattr(args, "diffusion_steps")
        and hasattr(args, "sample_steps")
        and args.sample_steps is not None
        and args.diffusion_steps is not None
        and int(args.sample_steps) != int(args.diffusion_steps)
    ):
        args.timestep_respacing = f"ddim{int(args.sample_steps)}"

    model, diffusion = read_model_and_diffusion(args, args.model_dir, args.model_num, args.ema)
    model.eval()

    classifier = read_classifier(args, args.clf_dir, args.clf_num)
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a = torch.autograd.grad(selected.sum(), x_in)[0]
            return a * args.classifier_scale

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
            ddib=False,
            cond_fn=cond_fn,
        )
        logger.log(f"optimal threshold: {opt_thr}, dice_max_val: {dice_max_val}")
    else:
        opt_thr = 0.63

    logging = logging_metrics(logger)
    Y = []
    PRED_Y = []

    k = 0
    data_test_iter = iter(data_test)
    while k < args.num_batches:
        k += 1

        all_sources = []
        all_latents = []
        all_targets = []
        all_masks = []
        all_pred_maps = []

        try:
            source, mask, lab = next(data_test_iter)
        except StopIteration:
            data_test_iter = iter(data_test)
            source, mask, lab = next(data_test_iter)
        logger.log(f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}...")

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}")

        Y.append(lab)

        y = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(start=0, end=1).reshape(-1, 1)
        y = y.reshape(-1, 1).squeeze().to(dist_util.dev())

        t = torch.tensor([args.sample_steps - 1] * source.shape[0], device=dist_util.dev())
        noise = diffusion.q_sample(source, t=t)

        target, _ = sample(
            model,
            diffusion,
            y=y,
            cond_fn=cond_fn,
            noise=noise,
            w=args.w,
            sample_shape=source.shape,
            sample_steps=args.sample_steps,
            normalize_img=False,
            ddpm=False,
        )

        if args.num_batches_val == 0:
            opt_thr = 0.24

        pred_mask, pred_map, pred_lab = get_mask_batch(source, target, opt_thr, args.modality, median_filter=True)
        PRED_Y.append(pred_lab)

        if dist.get_rank() == 0:
            gt_pixels = mask.view(mask.shape[0], -1).sum(dim=1)
            pred_pixels = pred_mask.view(pred_mask.shape[0], -1).sum(dim=1)
            pred_map_flat = pred_map.view(pred_map.shape[0], -1)
            overlap_pixels = torch.logical_and(mask > 0, pred_mask > 0).view(mask.shape[0], -1).sum(dim=1)
            lab_cpu = lab.detach().cpu() if torch.is_tensor(lab) else lab
            overlap_cpu = overlap_pixels.detach().cpu()
            gt_cpu = gt_pixels.detach().cpu()
            pred_cpu = pred_pixels.detach().cpu()
            logger.log(
                f"thr={float(opt_thr):.6f} pred_map[min={float(pred_map_flat.min()):.6f}, "
                f"mean={float(pred_map_flat.mean()):.6f}, max={float(pred_map_flat.max()):.6f}] "
                f"gt_pixels[min={float(gt_pixels.min()):.0f}, mean={float(gt_pixels.float().mean()):.1f}, max={float(gt_pixels.max()):.0f}] "
                f"pred_pixels[min={float(pred_pixels.min()):.0f}, mean={float(pred_pixels.float().mean()):.1f}, max={float(pred_pixels.max()):.0f}]"
            )
            logger.log(
                f"overlap_pixels[min={float(overlap_cpu.min()):.0f}, mean={float(overlap_cpu.float().mean()):.1f}, max={float(overlap_cpu.max()):.0f}] "
                f"overlap/gt_mean={float((overlap_cpu.float() / (gt_cpu.float() + 1e-6)).mean()):.4f} "
                f"overlap/pred_mean={float((overlap_cpu.float() / (pred_cpu.float() + 1e-6)).mean()):.4f}"
            )
            if torch.is_tensor(lab_cpu) and (lab_cpu == 1).any():
                overlap_ano = overlap_cpu[lab_cpu == 1]
                gt_ano = gt_cpu[lab_cpu == 1]
                pred_ano = pred_cpu[lab_cpu == 1]
                logger.log(
                    f"overlap_ano[min={float(overlap_ano.min()):.0f}, mean={float(overlap_ano.float().mean()):.1f}, max={float(overlap_ano.max()):.0f}] "
                    f"overlap_ano/gt_mean={float((overlap_ano.float() / (gt_ano.float() + 1e-6)).mean()):.4f} "
                    f"overlap_ano/pred_mean={float((overlap_ano.float() / (pred_ano.float() + 1e-6)).mean()):.4f}"
                )

        eval_metrics = evaluate(mask, pred_mask, source, pred_map, cc_filter=args.cc_filter, cc_thr=args.cc_thr)
        eval_metrics_ano = evaluate(mask, pred_mask, source, pred_map, lab, cc_filter=args.cc_filter, cc_thr=args.cc_thr)
        if dist.get_rank() == 0:
            post_pixels = eval_metrics["recon_mask"].view(eval_metrics["recon_mask"].shape[0], -1).sum(dim=1)
            overlap_post = torch.logical_and(mask > 0, eval_metrics["recon_mask"] > 0).view(mask.shape[0], -1).sum(dim=1)
            logger.log(
                f"pred_pixels_postcc[min={float(post_pixels.min()):.0f}, mean={float(post_pixels.float().mean()):.1f}, max={float(post_pixels.max()):.0f}] cc_filter={args.cc_filter}"
            )
            logger.log(
                f"overlap_postcc[min={float(overlap_post.min()):.0f}, mean={float(overlap_post.float().mean()):.1f}, max={float(overlap_post.max()):.0f}]"
            )
        cls_metrics = get_stats(Y, PRED_Y)
        logging.logging(eval_metrics, eval_metrics_ano, cls_metrics, k)

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

        if args.save_data:
            logger.log("collecting metrics...")
            gathered_source = [torch.zeros_like(source) for _ in range(dist.get_world_size())]
            gathered_latent = [torch.zeros_like(noise) for _ in range(dist.get_world_size())]
            gathered_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            gathered_mask = [torch.zeros_like(mask) for _ in range(dist.get_world_size())]
            gathered_pred_map = [torch.zeros_like(pred_map) for _ in range(dist.get_world_size())]

            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_latent, noise)
            dist.all_gather(gathered_target, target)
            dist.all_gather(gathered_mask, mask)
            dist.all_gather(gathered_pred_map, pred_map)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_latents.extend([noise.cpu().numpy() for noise in gathered_latent])
            all_targets.extend([target.cpu().numpy() for target in gathered_target])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            all_pred_maps.extend([pred_map.cpu().numpy() for pred_map in gathered_pred_map])

            all_sources = np.concatenate(all_sources, axis=0)
            np.save(os.path.join(image_subfolder, f"source_{k}.npy"), all_sources)

            all_latents = np.concatenate(all_latents, axis=0)
            np.save(os.path.join(image_subfolder, f"latent_{k}.npy"), all_latents)

            all_targets = np.concatenate(all_targets, axis=0)
            np.save(os.path.join(image_subfolder, f"target_{k}.npy"), all_targets)

            all_masks = np.concatenate(all_masks, axis=0)
            np.save(os.path.join(image_subfolder, f"mask_{k}.npy"), all_masks)

            all_pred_maps = np.concatenate(all_pred_maps, axis=0)
            np.save(os.path.join(image_subfolder, f"pred_map_{k}.npy"), all_pred_maps)

    dist.barrier()
    logger.log("clf guided visualization complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        name="",
        image_dir="",
        model_dir="",
        clf_dir="",
        seed=0,
        batch_size=32,
        sample_steps=1000,
        model_num=None,
        clf_num=None,
        ema=True,
        save_data=False,
        dynamic_clip=False,
        num_batches_val=2,
        batch_size_val=100,
        classifier_scale=100,
        unet_ver="v1",
        use_weighted_sampler=False,
        visualization_output_dir="",
        num_samples_to_visualize=5,
        cc_filter=True,
        cc_thr=40,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument("--modality", type=int, nargs="+", default=0)
    parser.add_argument("--w", type=float, default=-1.0)
    parser.add_argument("--num_batches", type=int, default=1)

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

