import argparse
import os
import sys
import time

sys.path.append(os.path.realpath("./"))

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gaussian_diffusion_dual_loss import (
    GaussianDiffusionDualLoss,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from data import get_data_iter
from obtain_hyperpara import obtain_hyperpara
from sample import sample
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def create_model_and_diffusion_dual_loss(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_ddim,
    num_classes,
    unet_ver="bea_dual_loss",
    use_bea=True,
    boundary_loss_weight=0.3,
    **kwargs
):
    """
    Create a model and diffusion process with dual loss support.
    """
    # Import the dual loss UNet model
    from guided_diffusion.unet_bea_dual_loss import BEADualLossUNetModel
    
    model = BEADualLossUNetModel(
        image_size=image_size,
        in_channels=num_channels,
        model_channels=128,
        out_channels=(num_channels if not learn_sigma else num_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_resolutions),
        dropout=dropout,
        channel_mult=tuple(channel_mult),
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        use_bea=use_bea,
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.DUAL_LOSS  # Use dual loss
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    
    diffusion = GaussianDiffusionDualLoss(
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (ModelVarType.FIXED_LARGE if not learn_sigma else ModelVarType.LEARNED_RANGE)
            if not rescale_learned_sigmas
            else ModelVarType.LEARNED
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        boundary_loss_weight=boundary_loss_weight,
    )
    
    return model, diffusion


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("loading hyperparameters...")
    hyperparameters = obtain_hyperpara(args.name)
    logger.log(f"hyperparameters: {hyperparameters}")

    logger.log("creating model and diffusion...")
    
    # Force BEA dual loss UNet version
    args.unet_ver = "bea_dual_loss"
    args.use_bea = True
    
    model, diffusion = create_model_and_diffusion_dual_loss(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        boundary_loss_weight=args.boundary_loss_weight,
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
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
    data = get_data_iter(
        args.name,
        args.data_dir,
        mixed=args.mixed,
        batch_size=args.batch_size,
        split=args.split,
        ret_lab=args.ret_lab,
        n_unhealthy_patients=args.n_unhealthy_patients,
        n_healthy_patients=args.n_healthy_patients,
        logger=logger,
    )

    logging_metrics = []
    Y = []
    PRED_Y = []
    all_sources = []
    all_masks = []
    all_pred_maps = []
    all_terms = []
    all_pred_masks_all = []

    logger.log("sampling...")
    all_images = []
    all_labels = []
    
    for k, (batch, cond) in enumerate(data):
        logger.log(f"sampling batch {k}...")
        
        if args.num_samples > 0 and k * args.batch_size >= args.num_samples:
            break
            
        batch = batch.to(dist_util.dev())
        if args.class_cond:
            cond = {key: val.to(dist_util.dev()) for key, val in cond.items()}
        else:
            cond = {}

        # Sample from the model
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        sample_shape = batch.shape
        samples = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=cond,
        )

        # Compute reconstruction error
        mse_null_flat = ((batch - samples) ** 2).view(batch.shape[0], -1).mean(dim=1)
        
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)
        gathered_source = [torch.zeros_like(batch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_source, batch)
        gathered_mse = [torch.zeros_like(mse_null_flat) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_mse, mse_null_flat)
        
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_labels.extend([source.cpu().numpy() for source in gathered_source])
        
        # Collect metrics
        if args.ret_lab:
            labels = cond.get('y', torch.zeros(batch.shape[0], dtype=torch.long, device=batch.device))
            gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels)
            
            for i, (mse_batch, label_batch) in enumerate(zip(gathered_mse, gathered_labels)):
                Y.extend(label_batch.cpu().numpy())
                PRED_Y.extend(mse_batch.cpu().numpy())

        # Collect data for saving
        if args.save_data:
            logger.log("collecting metrics...")
            
            # Collect terms for analysis
            terms = {}
            for term_name in ['mse', 'bea', 'loss']:
                if hasattr(diffusion, 'training_losses'):
                    # Get training losses for analysis
                    t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=batch.device)
                    loss_dict = diffusion.training_losses(model, batch, t, model_kwargs=cond)
                    if term_name in loss_dict:
                        terms[term_name] = loss_dict[term_name].cpu().numpy()
                    else:
                        terms[term_name] = np.zeros(batch.shape[0])
                else:
                    terms[term_name] = np.zeros(batch.shape[0])
            
            all_terms.append(terms)
            
            # Collect source and prediction data
            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_pred_maps.extend([pred.cpu().numpy() for pred in gathered_mse])
            
            # Collect masks if available
            if 'mask' in cond:
                mask = cond['mask']
                gathered_mask = [torch.zeros_like(mask) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_mask, mask)
                all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            else:
                all_masks.extend([np.zeros_like(source.cpu().numpy()) for source in gathered_source])
                
            # Collect prediction masks
            pred_masks = (mse_null_flat > mse_null_flat.median()).float()
            gathered_pred_masks = [torch.zeros_like(pred_masks) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_pred_masks, pred_masks)
            all_pred_masks_all.extend([mask.cpu().numpy() for mask in gathered_pred_masks])

        logger.log(f"created {len(all_images)} samples")

    # Compute and log metrics
    if args.ret_lab and len(Y) > 0 and len(PRED_Y) > 0:
        Y = np.array(Y)
        PRED_Y = np.array(PRED_Y)
        
        # Compute AUC and AUPRC
        auc = roc_auc_score(Y, PRED_Y)
        auprc = average_precision_score(Y, PRED_Y)
        
        logger.log(f"AUC: {auc:.4f}")
        logger.log(f"AUPRC: {auprc:.4f}")
        
        eval_metrics = {'auc': auc, 'auprc': auprc}
        eval_metrics_ano = eval_metrics  # For compatibility
        cls_metrics = eval_metrics  # For compatibility
        
        logging_metrics.append((eval_metrics, eval_metrics_ano, cls_metrics, k))

    # Save data if requested
    if args.save_data:
        logger.log("collecting metrics...")
        
        # Concatenate all collected data
        all_sources = np.concatenate(all_sources, axis=0)
        all_masks = np.concatenate(all_masks, axis=0) if all_masks else np.array([])
        all_pred_maps = np.concatenate(all_pred_maps, axis=0)
        all_pred_masks_all = np.concatenate(all_pred_masks_all, axis=0)
        
        # Save data
        if dist.get_rank() == 0:
            save_path = os.path.join(args.results_path, f"results_{args.name}_{args.split}.npz")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            np.savez(
                save_path,
                sources=all_sources,
                masks=all_masks,
                pred_maps=all_pred_maps,
                pred_masks=all_pred_masks_all,
                terms=all_terms,
                Y=Y if len(Y) > 0 else np.array([]),
                PRED_Y=PRED_Y if len(PRED_Y) > 0 else np.array([]),
            )
            logger.log(f"saved results to {save_path}")

    dist.barrier()
    logger.log(f"evaluation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        name="",
        split="test",
        mixed=True,
        ret_lab=True,
        n_unhealthy_patients=-1,
        n_healthy_patients=-1,
        results_path="./results",
        save_data=True,
        unet_ver="bea_dual_loss",  # Force BEA dual loss UNet
        use_bea=True,  # Enable BEA by default
        boundary_loss_weight=0.3,  # λ parameter for boundary-aware consistency loss
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boundary_loss_weight",
        type=float,
        help="weight for boundary-aware consistency loss (λ parameter)",
        default=0.3,
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()