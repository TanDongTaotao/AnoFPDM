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
from obtain_hyperpara import get_mask_batch_FPDM, obtain_hyperpara


def aggregate_reconstructions(xstarts, source, modality, t_e_ratio=1, last_only=False, interval=-1):
    device = source.device
    batch_size = source.shape[0]

    diff = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
    diff_flat = torch.mean(diff, dim=(3, 4))

    guided_recons = []
    unguided_recons = []

    modality_list = modality if isinstance(modality, (list, tuple)) else [modality]

    for sample_num in range(batch_size):
        diff_i = diff_flat[sample_num, ...]
        t_e_i = torch.argmax(diff_i, dim=0)

        if t_e_ratio != 1:
            t_e_i = torch.round(t_e_i * t_e_ratio).to(torch.int64)

        t_s_i = torch.zeros_like(t_e_i, device=device)
        if last_only:
            t_s_i = torch.clamp(t_e_i - 1, min=0)

        guided_recon_sample = torch.zeros_like(source[sample_num])
        unguided_recon_sample = torch.zeros_like(source[sample_num])

        for mod_idx, mod in enumerate(modality_list):
            start_step = int(t_s_i[mod_idx].item())
            end_step = int(t_e_i[mod_idx].item())

            if end_step <= start_step:
                guided_recon_sample[mod] = source[sample_num, mod]
                unguided_recon_sample[mod] = source[sample_num, mod]
                continue

            guided_subset = xstarts["xstart"][sample_num, start_step:end_step, mod_idx, ...]
            unguided_subset = xstarts["xstart_null"][sample_num, start_step:end_step, mod_idx, ...]

            if interval != -1 and interval > 0:
                guided_subset = guided_subset[::interval, ...]
                unguided_subset = unguided_subset[::interval, ...]
                guided_subset = torch.cat(
                    [guided_subset, xstarts["xstart"][sample_num, end_step - 1 : end_step, mod_idx, ...]], dim=0
                )
                unguided_subset = torch.cat(
                    [unguided_subset, xstarts["xstart_null"][sample_num, end_step - 1 : end_step, mod_idx, ...]], dim=0
                )

            guided_recon_sample[mod] = torch.mean(guided_subset, dim=0)
            unguided_recon_sample[mod] = torch.mean(unguided_subset, dim=0)

        guided_recons.append(guided_recon_sample.unsqueeze(0))
        unguided_recons.append(unguided_recon_sample.unsqueeze(0))

    guided_recon = torch.cat(guided_recons, dim=0)
    unguided_recon = torch.cat(unguided_recons, dim=0)
    return guided_recon, unguided_recon


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
    guided_recon,
    unguided_recon,
    anomaly_map,
    pred_mask,
    true_mask,
    sample_idx,
    modality,
    save_path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} - FPDM Visualization", fontsize=16)

    main_modality = (
        modality[0]
        if isinstance(modality, (list, tuple)) and len(modality) > 0
        else (modality if isinstance(modality, int) else 0)
    )

    source_np = source[main_modality].cpu().numpy()
    guided_recon_np = guided_recon[main_modality].cpu().numpy()
    unguided_recon_np = unguided_recon[main_modality].cpu().numpy()
    anomaly_map_np = anomaly_map[0].cpu().numpy()
    pred_mask_np = pred_mask[0].cpu().numpy()
    true_mask_np = true_mask[0].cpu().numpy()

    axes[0, 0].imshow(source_np, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(guided_recon_np, cmap="gray")
    axes[0, 1].set_title("Guided Reconstruction")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(unguided_recon_np, cmap="gray")
    axes[0, 2].set_title("Unguided Reconstruction")
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

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    vis_output_dir = args.visualization_output_dir if args.visualization_output_dir else image_subfolder
    pathlib.Path(vis_output_dir).mkdir(parents=True, exist_ok=True)

    logger.log("reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True
    args.multi_class = True if args.num_classes and args.num_classes > 2 else False

    model, diffusion = read_model_and_diffusion(args, args.model_dir, args.model_num, args.ema)

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

    logger.log("Validation: starting to get threshold and abe range ...")
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
        thr_01, diff_min, diff_max, n_min = obtain_hyperpara(data_val, diffusion, model, args, dist_util.dev())
        logger.log(f"diff_min: {diff_min}, diff_max: {diff_max}, thr_01: {thr_01}, n_min: {n_min}")
    else:
        logger.log(f"loading hyperparameters for {args.name} with forward_steps {args.forward_steps}...")
        if args.name == "brats":
            if args.forward_steps == 999:
                thr_01 = 0.9993147253990173
                diff_min = torch.tensor([0.0022, 0.0010], device=dist_util.dev())
                diff_max = torch.tensor([0.0551, 0.0388], device=dist_util.dev())
            elif args.forward_steps == 600:
                thr_01 = 0.9948798418045044
                diff_min = torch.tensor([5.5484e-05, 3.4732e-05], device=dist_util.dev())
                diff_max = torch.tensor([0.0509, 0.0397], device=dist_util.dev())
        elif args.name == "atlas":
            thr_01 = 0.7285396456718445
            diff_min = torch.tensor([0.0392], device=dist_util.dev())
            diff_max = torch.tensor([0.8555], device=dist_util.dev())

        logger.log(f"diff_min: {diff_min}, diff_max: {diff_max}, thr_01: {thr_01}")

    logger.log("starting to inference ...")

    logging = logging_metrics(logger)
    Y = [[] for _ in range(len(args.t_e_ratio))]
    PRED_Y = [[] for _ in range(len(args.t_e_ratio))]

    k = 0
    while k < args.num_batches:
        k += 1

        source, mask, lab = data_test.__iter__().__next__()
        logger.log(f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}...")

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        y0 = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(start=0, end=1).reshape(-1, 1)
        y0 = y0.reshape(-1, 1).squeeze().to(dist_util.dev())

        model_kwargs_reverse = {"threshold": -1, "clf_free": True, "null": args.null}
        model_kwargs0 = {"y": y0, "threshold": -1, "clf_free": True}

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

        guided_recon, unguided_recon = aggregate_reconstructions(
            xstarts,
            source,
            args.modality,
            t_e_ratio=args.t_e_ratio[0] if args.t_e_ratio else 1,
            last_only=args.last_only,
            interval=args.subset_interval,
        )

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

            if dist.get_rank() == 0:
                num_to_vis = min(args.num_samples_to_visualize, source.shape[0])
                for sample_idx in range(num_to_vis):
                    save_path = os.path.join(
                        vis_output_dir,
                        f"visualization_batch_{k}_sample_{sample_idx}_ratio_{ratio:.2f}.png",
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
                        save_path,
                    )
                    logger.log(f"Saved visualization: {save_path}")

    dist.barrier()
    logger.log("fpdm visualization complete")


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
        d_reverse=True,
        median_filter=True,
        dynamic_clip=False,
        last_only=False,
        subset_interval=-1,
        seed=0,
        use_weighted_sampler=False,
        use_gradient_sam=False,
        use_gradient_para_sam=False,
        visualization_output_dir="",
        num_samples_to_visualize=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument("--modality", type=int, nargs="+", help="0:flair, 1:t1, 2:t1ce, 3:t2", default=[0, 3])
    parser.add_argument("--t_e_ratio", type=float, nargs="+", default=[1])
    parser.add_argument("--w", type=float, help="weight for clf-free samples", default=-1)
    parser.add_argument("--num_batches", type=int, help="number of batches to run", default=1)

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
