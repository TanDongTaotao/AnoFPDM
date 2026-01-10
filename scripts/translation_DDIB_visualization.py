import argparse
import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from common import read_model_and_diffusion, set_seed_for_reproducibility
from data import get_data_iter
from evaluate import evaluate, get_stats, logging_metrics
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults
from obtain_hyperpara import get_mask_batch, obtain_optimal_threshold
from sample import sample


def _safe_get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _safe_get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _safe_barrier():
    if dist.is_initialized():
        dist.barrier()


def _safe_all_gather(output_list, tensor):
    if dist.is_initialized():
        dist.all_gather(output_list, tensor)
    else:
        output_list[:] = [tensor]


def create_heatmap_overlay(source_img, anomaly_map, alpha=0.6):
    source_norm = (source_img - source_img.min()) / (source_img.max() - source_img.min() + 1e-8)
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    cmap = LinearSegmentedColormap.from_list("anomaly", ["blue", "cyan", "yellow", "red"], N=256)
    anomaly_rgb = cmap(anomaly_norm.cpu().numpy())[:, :, :3]
    source_rgb = np.stack([source_norm.cpu().numpy()] * 3, axis=-1)
    overlay = (1 - alpha) * source_rgb + alpha * anomaly_rgb
    return overlay


def visualize_sample(
    source,
    latent,
    target,
    anomaly_map,
    pred_mask,
    true_mask,
    sample_idx,
    modality,
    save_path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} - DDIB Visualization", fontsize=16)

    main_modality = (
        modality[0]
        if isinstance(modality, (list, tuple)) and len(modality) > 0
        else (modality if isinstance(modality, int) else 0)
    )

    source_np = source[main_modality].cpu().numpy()
    latent_np = latent[main_modality].cpu().numpy()
    target_np = target[main_modality].cpu().numpy()
    anomaly_map_np = anomaly_map[0].cpu().numpy()
    pred_mask_np = pred_mask[0].cpu().numpy()
    true_mask_np = true_mask[0].cpu().numpy()

    axes[0, 0].imshow(source_np, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(latent_np, cmap="gray")
    axes[0, 1].set_title("DDIM Reverse Latent")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(target_np, cmap="gray")
    axes[0, 2].set_title("Translated Healthy")
    axes[0, 2].axis("off")

    overlay = create_heatmap_overlay(torch.tensor(source_np), torch.tensor(anomaly_map_np))
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Anomaly Heatmap Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_mask_np, cmap="Reds", alpha=0.8)
    axes[1, 1].imshow(source_np, cmap="gray", alpha=0.3)
    axes[1, 1].set_title("Predicted Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(true_mask_np, cmap="Greens", alpha=0.8)
    axes[1, 2].imshow(source_np, cmap="gray", alpha=0.3)
    axes[1, 2].set_title("Ground Truth Mask")
    axes[1, 2].axis("off")

    plt.tight_layout()
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    set_seed_for_reproducibility(args.seed)
    logger.configure()
    logger.log(f"args: {args}")
    logger.log("starting to sample.")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    vis_output_dir = args.visualization_output_dir if args.visualization_output_dir else image_subfolder
    pathlib.Path(vis_output_dir).mkdir(parents=True, exist_ok=True)

    logger.log("reading models ...")
    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
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

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
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
            guided=True,
            ddib=True,
            use_ddpm=False,
        )
        logger.log(f"optimal threshold: {opt_thr}, dice_max_val: {dice_max_val}")
    else:
        if args.name == "brats":
            opt_thr = 0.22
        elif args.name == "atlas":
            opt_thr = 0.52
        else:
            opt_thr = 0.1
        logger.log(f"optimal threshold: {opt_thr} for {args.name}")

    logging = logging_metrics(logger)
    Y = []
    PRED_Y = []

    k = 0
    while k < args.num_batches:
        k += 1

        all_sources = []
        all_latents = []
        all_targets = []
        all_masks = []
        all_pred_maps = []

        source, mask, lab = data_test.__iter__().__next__()
        Y.append(lab)

        logger.log(f"translating at batch {k} on rank {_safe_get_rank()}, shape {source.shape}...")

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        latent, _ = sample(
            model,
            diffusion,
            noise=source,
            reverse=True,
            null=True,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            ddpm=False,
            normalize_img=False,
        )

        y0 = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(start=0, end=1).reshape(-1, 1)
        y0 = y0.reshape(-1, 1).squeeze().to(dist_util.dev())

        target, _ = sample(
            model,
            diffusion,
            y=y0,
            noise=latent,
            w=args.w,
            sample_shape=source.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
            ddpm=False,
        )

        pred_mask, pred_map, pred_lab = get_mask_batch(source, target, opt_thr, args.modality)
        PRED_Y.append(pred_lab)

        eval_metrics = evaluate(
            mask,
            pred_mask,
            source,
            pred_map,
            cc_filter=args.cc_filter,
            cc_thr=args.cc_thr,
        )
        eval_metrics_ano = evaluate(
            mask,
            pred_mask,
            source,
            pred_map,
            lab,
            cc_filter=args.cc_filter,
            cc_thr=args.cc_thr,
        )
        cls_metrics = get_stats(Y, PRED_Y)
        logging.logging(eval_metrics, eval_metrics_ano, cls_metrics, k)

        if _safe_get_rank() == 0:
            num_to_vis = min(args.num_samples_to_visualize, source.shape[0])
            for sample_idx in range(num_to_vis):
                save_path = os.path.join(
                    vis_output_dir,
                    f"visualization_batch_{k}_sample_{sample_idx}_steps_{args.sample_steps}.png",
                )
                visualize_sample(
                    source[sample_idx],
                    latent[sample_idx],
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
            world_size = _safe_get_world_size()
            gathered_source = [torch.zeros_like(source) for _ in range(world_size)]
            gathered_latent = [torch.zeros_like(latent) for _ in range(world_size)]
            gathered_target = [torch.zeros_like(target) for _ in range(world_size)]
            gathered_mask = [torch.zeros_like(mask) for _ in range(world_size)]
            gathered_pred_maps = [torch.zeros_like(pred_map) for _ in range(world_size)]

            _safe_all_gather(gathered_source, source)
            _safe_all_gather(gathered_latent, latent)
            _safe_all_gather(gathered_target, target)
            _safe_all_gather(gathered_mask, mask)
            _safe_all_gather(gathered_pred_maps, pred_map)

            all_sources.extend([t.cpu().numpy() for t in gathered_source])
            all_latents.extend([t.cpu().numpy() for t in gathered_latent])
            all_targets.extend([t.cpu().numpy() for t in gathered_target])
            all_masks.extend([t.cpu().numpy() for t in gathered_mask])
            all_pred_maps.extend([t.cpu().numpy() for t in gathered_pred_maps])

            np.save(os.path.join(image_subfolder, f"source_{k}.npy"), np.concatenate(all_sources, axis=0))
            np.save(os.path.join(image_subfolder, f"latent_{k}.npy"), np.concatenate(all_latents, axis=0))
            np.save(os.path.join(image_subfolder, f"target_{k}.npy"), np.concatenate(all_targets, axis=0))
            np.save(os.path.join(image_subfolder, f"mask_{k}.npy"), np.concatenate(all_masks, axis=0))
            np.save(os.path.join(image_subfolder, f"pred_map_{k}.npy"), np.concatenate(all_pred_maps, axis=0))

    _safe_barrier()
    logger.log("ddib visualization complete")


def create_argparser():
    defaults = dict(
        name="",
        data_dir="",
        image_dir="",
        model_dir="",
        unet_ver="v2",
        seed=0,
        batch_size=32,
        sample_steps=1000,
        use_ddpm=False,
        model_num=None,
        ema=False,
        dynamic_clip=False,
        save_data=False,
        num_batches_val=2,
        batch_size_val=100,
        cc_filter=True,
        cc_thr=40,
        use_weighted_sampler=False,
        visualization_output_dir="",
        num_samples_to_visualize=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument("--modality", type=int, nargs="+", help="0:flair, 1:t1, 2:t1ce, 3:t2", default=0)
    parser.add_argument("--w", type=float, help="weight for clf-free samples", default=-1.0)
    parser.add_argument("--num_batches", type=int, help="number of batches to run", default=1)

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
