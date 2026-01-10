import argparse
import os
import pathlib
import random
import numpy as np
import torch.distributed as dist
import torch

from common import read_model_and_diffusion, read_classifier
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    classifier_defaults,
)

from data import get_data_iter

from evaluate import get_stats, evaluate, logging_metrics
from sample import sample

import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from obtain_hyperpara import get_mask_batch, obtain_optimal_threshold


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"args: {args}")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
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

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )
    model.eval()

    classifier = read_classifier(args, args.clf_dir, args.clf_num)
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)  # 100x2
            selected = log_probs[range(len(logits)), y.view(-1)]  # 100
            a = torch.autograd.grad(selected.sum(), x_in)[0]  # 100x4x128x128
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
        opt_thr = 0.63  # atlas 200 500

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
        logger.log(
            f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}..."
        )

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(
            f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}"
        )

        Y.append(lab)

        y = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 only for healthy
        y = y.reshape(-1, 1).squeeze().to(dist_util.dev())

        t = torch.tensor(
            [args.sample_steps - 1] * source.shape[0], device=dist_util.dev()
        )
        noise = diffusion.q_sample(source, t=t)
        # model_kwargs_reverse = {"uncond": True}
        # noise = diffusion.ddim_reverse_sample(
        #     model, source, t=t, model_kwargs=model_kwargs_reverse
        # )

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
            opt_thr = 0.24  # for optimal setup

        pred_mask, pred_map, pred_lab = get_mask_batch(
            source, target, opt_thr, args.modality, median_filter=True
        )
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

        if args.save_data:
            logger.log("collecting metrics...")
            gathered_source = [
                torch.zeros_like(source) for _ in range(dist.get_world_size())
            ]
            gathered_latent = [
                torch.zeros_like(noise) for _ in range(dist.get_world_size())
            ]
            gathered_target = [
                torch.zeros_like(target) for _ in range(dist.get_world_size())
            ]
            gathered_mask = [
                torch.zeros_like(mask) for _ in range(dist.get_world_size())
            ]
            gathered_pred_map = [
                torch.zeros_like(pred_map) for _ in range(dist.get_world_size())
            ]

            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_latent, noise)
            dist.all_gather(gathered_target, target)
            dist.all_gather(gathered_mask, mask)
            dist.all_gather(gathered_pred_map, pred_map)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_latents.extend([noise.cpu().numpy() for noise in gathered_latent])
            all_targets.extend([target.cpu().numpy() for target in gathered_target])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            all_pred_maps.extend(
                [pred_map.cpu().numpy() for pred_map in gathered_pred_map]
            )

            all_sources = np.concatenate(all_sources, axis=0)
            all_sources_path = os.path.join(image_subfolder, f"source_{k}.npy")
            np.save(all_sources_path, all_sources)

            all_latents = np.concatenate(all_latents, axis=0)
            all_latents_path = os.path.join(image_subfolder, f"latent_{k}.npy")
            np.save(all_latents_path, all_latents)

            all_targets = np.concatenate(all_targets, axis=0)
            all_targets_path = os.path.join(image_subfolder, f"target_{k}.npy")
            np.save(all_targets_path, all_targets)

            all_masks = np.concatenate(all_masks, axis=0)
            all_masks_path = os.path.join(image_subfolder, f"mask_{k}.npy")
            np.save(all_masks_path, all_masks)

            all_pred_maps = np.concatenate(all_pred_maps, axis=0)
            all_pred_maps_path = os.path.join(image_subfolder, f"pred_map_{k}.npy")
            np.save(all_pred_maps_path, all_pred_maps)

    dist.barrier()
    logger.log(f"synthetic data translation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        name="",
        image_dir="",
        model_dir="",  # model directory,
        clf_dir="",  # classifier directory
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
        cc_filter=True,
        cc_thr=40,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=0,  # first modality as default
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
        help="weight for clf-free samples",
        default=1,  # disabled in default
    )

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
