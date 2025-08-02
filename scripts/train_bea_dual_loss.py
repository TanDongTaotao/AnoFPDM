"""Train a diffusion model with Boundary-Enhanced Attention (BEA) UNet and dual loss on images."""

import sys
import os

sys.path.append(os.path.realpath("./"))

import argparse
import pathlib
from guided_diffusion import dist_util, logger
from data import get_data_iter, check_data
from guided_diffusion.resample import create_named_schedule_sampler

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_dual_loss import TrainLoopDualLoss
from guided_diffusion.gaussian_diffusion_dual_loss import (
    GaussianDiffusionDualLoss,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from sample import sample


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

    args.w = args.w if isinstance(args.w, list) else [args.w]

    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
    if args.num_classes:
        args.class_cond = True

    # Force BEA dual loss UNet version
    args.unet_ver = "bea_dual_loss"
    
    logger.log(f"args: {args}")
    logger.log(f"Using Boundary-Enhanced Attention (BEA) UNet with dual loss, use_bea={args.use_bea}, boundary_loss_weight={args.boundary_loss_weight}")

    model, diffusion = create_model_and_diffusion_dual_loss(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        boundary_loss_weight=args.boundary_loss_weight,
    )

    # get model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    logger.log("Model params: %.2f M" % (model_size / 1024 / 1024))

    pathlib.Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

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

    logger.log("creating data loader...")

    data = get_data_iter(
        args.name,
        args.data_dir,
        mixed=args.mixed,
        batch_size=args.batch_size, # global batch size, for each device it will be batch_size // num_devices
        split=args.split,
        ret_lab=args.ret_lab,
        n_unhealthy_patients=args.n_unhealthy_patients,
        n_healthy_patients=args.n_healthy_patients,
        logger=logger,
    )

    check_data(data[0], args.image_dir, name=args.name, split=args.split)

    logger.log("training...")

    TrainLoopDualLoss(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        name="",
        split="train",
        training=True,
        mixed=True,
        ret_lab=True,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        n_unhealthy_patients=-1,
        n_healthy_patients=-1,
        noise_type="gaussian",
        ddpm_sampling=False,
        unet_ver="bea_dual_loss",  # Force BEA dual loss UNet
        total_epochs=1000,
        use_bea=True,  # Enable BEA by default
        boundary_loss_weight=0.3,  # λ parameter for boundary-aware consistency loss
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_shape", type=int, nargs="+", help="sample shape")

    parser.add_argument(
        "--w",
        type=float,
        nargs="+",
        help="weight for clf-free samples",
        default=-1.0,  # disabled in default
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold for clf-free training",
        default=-1.0,  # disabled in default
    )
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