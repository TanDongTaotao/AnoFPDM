import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """

    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
        in_channels=3,
        out_channels=1000,
        unet_ver="v2",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=3,
        num_classes=2,
        unet_ver="v2",
        clf_free=True,
        use_bea=False,  # Enable/disable boundary-aware attention
        use_multi_bea=False,  # Enable/disable multi-layer boundary-aware attention
        use_bottleneck_bea=False,  # Enable/disable bottleneck-only boundary-aware attention
        use_hae=False,  # Enable/disable heterogeneous autoencoder
        bottleneck_ratio=0.25,  # Bottleneck ratio for HAE V2
        boundary_loss_weight=0.1,  # Weight for boundary-aware consistency loss (λ parameter)
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    in_channels,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    num_classes,
    unet_ver,
    clf_free,
    use_bea=False,
    use_multi_bea=False,
    use_bottleneck_bea=False,
    use_hae=False,
    bottleneck_ratio=0.25,
    boundary_loss_weight=0.1,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        in_channels=in_channels,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        num_classes=num_classes,
        unet_ver=unet_ver,
        clf_free=clf_free,
        use_bea=use_bea,
        use_multi_bea=use_multi_bea,
        use_bottleneck_bea=use_bottleneck_bea,
        use_hae=use_hae,
        bottleneck_ratio=bottleneck_ratio,
    )
    
    # Use dual loss diffusion for bea_dual_loss UNet version
    if unet_ver == "bea_dual_loss":
        from .gaussian_diffusion_dual_loss import (
            GaussianDiffusionDualLoss,
            ModelMeanType,
            ModelVarType,
            LossType,
            get_named_beta_schedule,
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
    else:
        diffusion = create_gaussian_diffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
        )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    in_channels=3,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_classes=2,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    unet_ver="v2",
    clf_free=True,
    use_bea=False,
    use_multi_bea=False,
    use_bottleneck_bea=False,
    use_hae=False,
    bottleneck_ratio=0.25,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if in_channels == 1 or in_channels == 4:
        out_channels = in_channels if not learn_sigma else in_channels * 2
    else:
        out_channels = 3 if not learn_sigma else 6

    if unet_ver == "v2":
        from .unet_v2 import UNetModel
    elif unet_ver == "v1":
        from .unet_v1 import UNetModel
    elif unet_ver == "bea":
        from .unet_bea import BEAUNetModel as UNetModel
    elif unet_ver == "bea_dual_loss":
        from .unet_bea_dual_loss import BEADualLossUNetModel as UNetModel
    elif unet_ver == "multi_bea":
        from .unet_multi_bea import MultiBEAUNetModel as UNetModel
    elif unet_ver == "bottleneck_bea":
        from .unet_bottleneck_bea import BottleneckBEAUNetModel as UNetModel
    elif unet_ver == "hae":
        from .unet_hae import HAEUNetModel as UNetModel
    elif unet_ver == "hae_lite":
        from .unet_hae_lite import HAEUNetModelLite as UNetModel
    elif unet_ver == "hae_v2":
        from .unet_hae_v2 import HAEUNetModelV2 as UNetModel
    else:
        raise ValueError(f"unsupported unet version: {unet_ver}")
    
    # Create model with appropriate parameters
    model_kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        clf_free=clf_free,
    )
    
    # Add specific parameters for different UNet versions
    if unet_ver == "bea":
        model_kwargs["use_bea"] = use_bea
    elif unet_ver == "bea_dual_loss":
        model_kwargs["use_bea"] = use_bea
        model_kwargs["return_features"] = True  # Enable feature return for dual loss
    elif unet_ver == "multi_bea":
        model_kwargs["use_multi_bea"] = use_multi_bea
    elif unet_ver == "bottleneck_bea":
        model_kwargs["use_bottleneck_bea"] = use_bottleneck_bea
    elif unet_ver == "hae":
        model_kwargs["use_hae"] = use_hae
    elif unet_ver == "hae_lite":
        model_kwargs["use_hae"] = use_hae
    elif unet_ver == "hae_v2":
        model_kwargs["use_hae"] = use_hae
        model_kwargs["bottleneck_ratio"] = bottleneck_ratio
    
    return UNetModel(**model_kwargs)


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    in_channels=1,
    out_channels=1000,
    unet_ver="v2",
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        in_channels=in_channels,
        out_channels=out_channels,
        unet_ver=unet_ver,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    in_channels=3,
    out_channels=1000,
    unet_ver="v2",
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if unet_ver == "v2":
        from .unet_v2 import EncoderUNetModel
    elif unet_ver == "v1":
        from .unet_v1 import EncoderUNetModel
    else:
        raise ValueError(f"unsupported unet version: {unet_ver}")

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


# def sr_model_and_diffusion_defaults():
#     res = model_and_diffusion_defaults()
#     res["large_size"] = 256
#     res["small_size"] = 64
#     arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
#     for k in res.copy().keys():
#         if k not in arg_names:
#             del res[k]
#     return res


# def sr_create_model_and_diffusion(
#         large_size,
#         small_size,
#         class_cond,
#         learn_sigma,
#         num_channels,
#         num_res_blocks,
#         num_heads,
#         num_head_channels,
#         num_heads_upsample,
#         attention_resolutions,
#         dropout,
#         diffusion_steps,
#         noise_schedule,
#         timestep_respacing,
#         use_kl,
#         predict_xstart,
#         rescale_timesteps,
#         rescale_learned_sigmas,
#         use_checkpoint,
#         use_scale_shift_norm,
#         resblock_updown,
#         use_fp16,
# ):
#     model = sr_create_model(
#         large_size,
#         small_size,
#         num_channels,
#         num_res_blocks,
#         learn_sigma=learn_sigma,
#         class_cond=class_cond,
#         use_checkpoint=use_checkpoint,
#         attention_resolutions=attention_resolutions,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         dropout=dropout,
#         resblock_updown=resblock_updown,
#         use_fp16=use_fp16,
#     )
#     diffusion = create_gaussian_diffusion(
#         steps=diffusion_steps,
#         learn_sigma=learn_sigma,
#         noise_schedule=noise_schedule,
#         use_kl=use_kl,
#         predict_xstart=predict_xstart,
#         rescale_timesteps=rescale_timesteps,
#         rescale_learned_sigmas=rescale_learned_sigmas,
#         timestep_respacing=timestep_respacing,
#     )
#     return model, diffusion


# def sr_create_model(
#         large_size,
#         small_size,
#         num_channels,
#         num_res_blocks,
#         learn_sigma,
#         class_cond,
#         use_checkpoint,
#         attention_resolutions,
#         num_heads,
#         num_head_channels,
#         num_heads_upsample,
#         use_scale_shift_norm,
#         dropout,
#         resblock_updown,
#         use_fp16,
#         num_classes=1000
# ):
#     _ = small_size  # hack to prevent unused variable

#     if large_size == 512:
#         channel_mult = (1, 1, 2, 2, 4, 4)
#     elif large_size == 256:
#         channel_mult = (1, 1, 2, 2, 4, 4)
#     elif large_size == 64:
#         channel_mult = (1, 2, 3, 4)
#     else:
#         raise ValueError(f"unsupported large size: {large_size}")

#     attention_ds = []
#     for res in attention_resolutions.split(","):
#         attention_ds.append(large_size // int(res))

#     return SuperResModel(
#         image_size=large_size,
#         in_channels=3,
#         model_channels=num_channels,
#         out_channels=(3 if not learn_sigma else 6),
#         num_res_blocks=num_res_blocks,
#         attention_resolutions=tuple(attention_ds),
#         dropout=dropout,
#         channel_mult=channel_mult,
#         num_classes=(num_classes if class_cond else None),
#         use_checkpoint=use_checkpoint,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=resblock_updown,
#         use_fp16=use_fp16,
#     )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
