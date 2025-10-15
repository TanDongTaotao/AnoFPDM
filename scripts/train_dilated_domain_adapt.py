"""Train a diffusion model on images with dilated UNet for domain adaptation."""

import sys
import os

sys.path.append(os.path.realpath("./"))

import argparse
import pathlib
import functools
import torch as th
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from data import get_data_iter, check_data
from guided_diffusion.resample import create_named_schedule_sampler, LossAwareSampler

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop, log_loss_dict
from sample import sample


class DomainAdaptTrainLoop(TrainLoop):
    """
    Extended TrainLoop for domain adaptation training.
    """
    
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        sample_shape,
        img_dir,
        threshold=-1,
        w=-1,
        num_classes=None,
        sample_fn=None,
        noise_fn=None,
        ddpm_sampling=False,
        total_epochs=1000,
        # Domain adaptation specific parameters
        domain_adapt_weight=1.0,
        consistency_weight=1.0,
        entropy_weight=0.1,
        feature_align_weight=0.5,
    ):
        super().__init__(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            sample_shape=sample_shape,
            img_dir=img_dir,
            threshold=threshold,
            w=w,
            num_classes=num_classes,
            sample_fn=sample_fn,
            noise_fn=noise_fn,
            ddpm_sampling=ddpm_sampling,
            total_epochs=total_epochs,
        )
        
        # Domain adaptation specific attributes
        self.domain_adapt_weight = domain_adapt_weight
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        self.feature_align_weight = feature_align_weight
    
    def run_step(self, batch, cond):
        """
        Target-only domain adaptation training.
        Only uses target domain data for self-supervised adaptation.
        """
        # Target domain self-supervised training
        # Use domain_label=1 for target domain (Atlas)
        target_loss = self.forward_backward_target(batch, cond, domain_label=1)
        
        # Log adaptation loss
        if self.step % self.log_interval == 0:
            logger.log(f"target_adaptation_loss: {target_loss:.6f}")
        
        self.mp_trainer.optimize(self.opt)
        if self.ema_rate > 0:
            self.ema.update()
        
        self.log_step()
    
    def forward_backward(self, batch, cond, domain_label=0):
        """
        Standard forward-backward pass with domain label.
        """
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            micro_cond = (
                {k: v[i : i + self.microbatch] for k, v in cond.items()}
                if cond is not None
                else {}
            )
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                domain_label=th.full((micro.shape[0],), domain_label, device=dist_util.dev(), dtype=th.long),
                threshold=self.threshold,
                noise_fn=self.noise_fn,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            
        return loss.item()
    
    def forward_backward_target(self, batch, cond, domain_label=1):
        """
        Target domain adaptation forward-backward pass.
        """
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            micro_cond = (
                {k: v[i : i + self.microbatch] for k, v in cond.items()}
                if cond is not None
                else {}
            )
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # Self-supervised target domain loss
            target_loss = self.compute_target_domain_loss(
                micro, t, micro_cond, domain_label
            )
            
            # Apply domain adaptation weight
            weighted_loss = target_loss * self.domain_adapt_weight
            
            if last_batch or not self.use_ddp:
                self.mp_trainer.backward(weighted_loss)
            else:
                with self.ddp_model.no_sync():
                    self.mp_trainer.backward(weighted_loss)
                    
        return target_loss.item()
    
    def compute_target_domain_loss(self, x, t, model_kwargs, domain_label):
        """
        Compute self-supervised loss for target domain.
        """
        # Consistency loss: predict same output for augmented versions
        x_aug1 = self.apply_augmentation(x)
        x_aug2 = self.apply_augmentation(x)
        
        domain_tensor = th.full((x.shape[0],), domain_label, device=x.device, dtype=th.long)
        
        # Get model predictions for both augmented versions
        with th.no_grad():
            pred1 = self.ddp_model(x_aug1, t, domain_label=domain_tensor, **model_kwargs)
        pred2 = self.ddp_model(x_aug2, t, domain_label=domain_tensor, **model_kwargs)
        
        # Consistency loss
        consistency_loss = F.mse_loss(pred1, pred2)
        
        # Entropy minimization loss (encourage confident predictions)
        pred_softmax = F.softmax(pred2, dim=1)
        entropy_loss = -th.sum(pred_softmax * th.log(pred_softmax + 1e-8), dim=1).mean()
        
        # Total target domain loss
        total_loss = (
            self.consistency_weight * consistency_loss +
            self.entropy_weight * entropy_loss
        )
        
        return total_loss
    
    def apply_augmentation(self, x):
        """
        Apply simple augmentations for consistency training.
        """
        # Random noise injection
        noise_scale = 0.01
        noise = th.randn_like(x) * noise_scale
        x_aug = x + noise
        
        # Random intensity scaling
        intensity_scale = 0.1
        scale_factor = 1.0 + (th.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * intensity_scale
        x_aug = x_aug * scale_factor
        
        return th.clamp(x_aug, -1, 1)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    args.w = args.w if isinstance(args.w, list) else [args.w]

    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
    if args.num_classes:
        args.class_cond = True

    logger.log(f"args: {args}")

    # Override unet_ver to use domain adaptation version
    args.unet_ver = "domain_adapt"

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load pretrained weights if specified
    if args.pretrained_model_path:
        logger.log(f"Loading pretrained weights from {args.pretrained_model_path}")
        pretrained_state_dict = th.load(args.pretrained_model_path, map_location="cpu")
        if "model" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["model"]
        
        missing_keys, unexpected_keys = model.load_pretrained_weights(
            pretrained_state_dict, strict=False
        )
        logger.log(f"Loaded pretrained weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    # get model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    logger.log("Model params: %.2f M" % (model_size / 1024 / 1024))

    # Count domain adaptation parameters
    if hasattr(model, 'enable_domain_adaptation') and model.enable_domain_adaptation:
        domain_params = 0
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ['domain_', 'film_', 'adapter']):
                domain_params += param.data.nelement()
        logger.log("Domain adaptation params: %.2f M (%.2f%%)" % (
            domain_params / 1024 / 1024, 
            100.0 * domain_params / model_size
        ))

    # Freeze main network parameters if specified
    if args.freeze_main_network:
        logger.log("Freezing main network parameters, only training domain adaptation modules...")
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # Keep domain adaptation modules trainable
            if any(keyword in name for keyword in ['domain_', 'film_', 'adapter']):
                param.requires_grad = True
                trainable_params += param.data.nelement()
            else:
                # Freeze main network parameters
                param.requires_grad = False
                frozen_params += param.data.nelement()
        
        logger.log("Frozen params: %.2f M (%.2f%%)" % (
            frozen_params / 1024 / 1024, 
            100.0 * frozen_params / model_size
        ))
        logger.log("Trainable params: %.2f M (%.2f%%)" % (
            trainable_params / 1024 / 1024, 
            100.0 * trainable_params / model_size
        ))

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

    # Load target domain data (e.g., ATLAS) for self-supervised adaptation
    logger.log("Loading target domain data for adaptation...")
    data = get_data_iter(
        args.name,
        args.data_dir,
        mixed=args.mixed,
        batch_size=args.batch_size,
        split=args.split,
        ret_lab=False,  # Target domain is unlabeled for self-supervised learning
        n_unhealthy_patients=args.n_unhealthy_patients,
        n_healthy_patients=args.n_healthy_patients,
        logger=logger,
    )

    check_data(data[0], args.image_dir, name=args.name, split=args.split)

    logger.log("training...")

    # Use domain adaptation training loop for target-only adaptation
    DomainAdaptTrainLoop(
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
        sample_shape=tuple(args.sample_shape),
        img_dir=args.image_dir,
        threshold=args.threshold,
        w=args.w,
        num_classes=args.num_classes,
        sample_fn=sample,
        noise_fn=noise_fn,
        ddpm_sampling=args.ddpm_sampling,
        total_epochs=args.total_epochs,
        # Domain adaptation specific parameters
        domain_adapt_weight=args.domain_adapt_weight,
        consistency_weight=args.consistency_weight,
        entropy_weight=args.entropy_weight,
        feature_align_weight=args.feature_align_weight,
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
        unet_ver="domain_adapt",  # Use domain adaptation UNet
        total_epochs=1000,
        # Domain adaptation specific parameters
        source_name="brats",
        source_data_dir="",
        target_name="atlas",
        target_data_dir="",
        pretrained_model_path="",
        domain_adapt_weight=1.0,
        consistency_weight=1.0,
        entropy_weight=0.1,
        feature_align_weight=0.5,
        adaptation_start_epoch=100,
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
    
    # Domain adaptation specific arguments
    parser.add_argument("--source_name", type=str, help="source domain dataset name")
    parser.add_argument("--source_data_dir", type=str, help="source domain data directory")
    parser.add_argument("--target_name", type=str, help="target domain dataset name")
    parser.add_argument("--target_data_dir", type=str, help="target domain data directory")
    parser.add_argument("--pretrained_model_path", type=str, help="path to pretrained model weights")
    parser.add_argument("--domain_adapt_weight", type=float, help="weight for domain adaptation loss")
    parser.add_argument("--consistency_weight", type=float, help="weight for consistency loss")
    parser.add_argument("--entropy_weight", type=float, help="weight for entropy minimization loss")
    parser.add_argument("--feature_align_weight", type=float, help="weight for feature alignment loss")
    parser.add_argument("--adaptation_start_epoch", type=int, help="epoch to start domain adaptation")
    parser.add_argument("--freeze_main_network", type=bool, default=False, help="freeze main network parameters, only train domain adaptation modules")
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()