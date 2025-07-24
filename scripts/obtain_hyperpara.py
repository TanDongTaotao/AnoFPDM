import torch
import numpy as np
from kornia.filters import sobel
import torch.nn.functional as F
from sample import sample
from evaluate import evaluate, median_pool


# %% This block is for the proposed method
def cal_cos_and_abe_range(mse_flat, mse_null_flat, diff_flat, lab):
    """
    mse_flat: N_val x sample_steps x n_modality
    mse_null_flat: N_val x sample_steps x n_modality
    diff_flat: N_val x sample_steps x n_modality
    lab: N_val

    get the cosine similarity threshold to differentiate healthy and tumour slices
    get the abe diff range for tumour slices to determine the quantile threshold for predicted mask
    """
    # Check if we have both healthy (0) and unhealthy (1) samples
    healthy_indices = torch.where(lab == 0)[0]
    unhealthy_indices = torch.where(lab == 1)[0]
    
    if len(healthy_indices) == 0:
        print("Warning: No healthy samples (label=0) found in validation data")
        # Use a default threshold when no healthy samples are available
        thr_01 = torch.tensor(0.5, device=mse_flat.device)
        n_min = 0
    else:
        mse_0_flat = mse_flat[healthy_indices]
        mse_0_null_flat = mse_null_flat[healthy_indices]
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output_0 = cos(
            torch.mean(mse_0_flat, dim=2), torch.mean(mse_0_null_flat, dim=2)
        )  # N_val
        
        if len(unhealthy_indices) == 0:
            print("Warning: No unhealthy samples (label=1) found in validation data")
            # Use a default threshold when no unhealthy samples are available
            thr_01 = torch.quantile(output_0, 0.05)
            n_min = 0
        else:
            mse_1_flat = mse_flat[unhealthy_indices]
            mse_1_null_flat = mse_null_flat[unhealthy_indices]
            
            output_1 = cos(
                torch.mean(mse_1_flat, dim=2), torch.mean(mse_1_null_flat, dim=2)
            )  # N_val
            
            n_min = 1e6
            for q in np.linspace(0.005, 0.095, 100):
                n = torch.sum(output_1 > torch.quantile(output_0, q)) + torch.sum(
                    output_0 < torch.quantile(output_0, q)
                )
                if n < n_min:
                    n_min = n
                    thr_01 = torch.quantile(output_0, q)
    
    # Handle diff_max calculation
    if len(unhealthy_indices) == 0:
        print("Warning: No unhealthy samples for diff calculation, using default values")
        # Use default values when no unhealthy samples are available
        diff_min = torch.zeros(mse_flat.shape[2], device=mse_flat.device)
        diff_max = torch.ones(mse_flat.shape[2], device=mse_flat.device)
    else:
        diff_1 = diff_flat[unhealthy_indices]  # N_val_1 x sample_steps x n_modality
        diff_max_vals = torch.max(diff_1, dim=1)[0]  # N_val_1 x n_modality
        diff_min = diff_max_vals.min(dim=0)[0]
        diff_max = diff_max_vals.max(dim=0)[0]
    
    return thr_01, diff_min, diff_max, n_min


def obtain_hyperpara(data_val, diffusion, model, args, device, edge_loss=False, edge_weight=1.0, attention_edge_weight=1.0):
    """
    return the optimal threshold for cosine similarity for classification
            and the range of abe diff for quantile threshold for predicted mask
    """
    MSE = []
    MSE_NULL = []
    DIFF = []
    LAB = []

    # TODO: make it memory efficient!
    for i in range(args.num_batches_val):
        source_val, _, lab_val = data_val.__iter__().__next__()
        source_val = source_val.to(device)

        y0 = torch.ones(source_val.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 for healthy
        y0 = y0.reshape(-1, 1).squeeze().to(device)

        model_kwargs_reverse = {"threshold": -1, "clf_free": True, "null": args.null}
        model_kwargs = {"y": y0, "threshold": -1, "clf_free": True}
        xstarts = diffusion.calc_pred_xstart_loop(
            model,
            source_val,
            args.w,
            modality=args.modality,
            d_reverse=args.d_reverse,
            sample_steps=args.forward_steps,
            model_kwargs=model_kwargs,
            model_kwargs_reverse=model_kwargs_reverse,
        )

        # for cosine similarity
        mse_flat = torch.mean(
            (xstarts["xstart"] - source_val[:, args.modality, ...].unsqueeze(1)) ** 2,
            dim=(3, 4),
        )  # batch_size x sample_steps x n_modality
        mse_null_flat = torch.mean(
            (xstarts["xstart_null"] - source_val[:, args.modality, ...].unsqueeze(1)) ** 2,
            dim=(3, 4),
        )
        # for scaled M_t (threshold selection)
        diff = (
            (xstarts["xstart"] - xstarts["xstart_null"])
        ) ** 2  # batch_size x sample_steps x n_modality x 128 x 128

        if edge_loss:
            edge_losses = []
            for t in range(len(xstarts['xstart'])):
                xstart_null_t = xstarts['xstart_null'][t]
                xstart_t = xstarts['xstart'][t]

                edge_null = sobel(xstart_null_t)
                edge_healthy = sobel(xstart_t)

                edge_null_flat = edge_null.view(edge_null.size(0), -1)
                edge_healthy_flat = edge_healthy.view(edge_healthy.size(0), -1)

                cos_sim_loss = 1 - F.cosine_similarity(edge_null_flat, edge_healthy_flat, dim=-1).mean()
                edge_losses.append(cos_sim_loss.item())
            
            avg_edge_loss = np.mean(edge_losses)
            edge_loss_tensor = torch.tensor(avg_edge_loss, device=diff.device, dtype=diff.dtype)
            diff = diff * torch.exp(edge_weight * edge_loss_tensor)

        diff_flat = torch.mean(diff, dim=(3, 4))
        
        MSE.append(mse_flat)
        MSE_NULL.append(mse_null_flat)
        DIFF.append(diff_flat)
        LAB.append(lab_val)

    MSE = torch.cat(MSE, dim=0)
    MSE_NULL = torch.cat(MSE_NULL, dim=0)
    DIFF = torch.cat(DIFF, dim=0)
    LAB = torch.cat(LAB, dim=0)
    thr_01, diff_min, diff_max, n_min = cal_cos_and_abe_range(MSE, MSE_NULL, DIFF, LAB)
    return thr_01, diff_min, diff_max, n_min


def get_mask_batch_FPDM(
    xstarts,
    source,
    modality,
    thr_01,
    diff_min,
    diff_max,
    shape,
    device,
    thr=None,
    t_e=None,
    t_e_ratio=1,
    median_filter=True,
    edge_loss=None,
    edge_weight=1.0,
    attention_edge_weight=1.0,  # 注意力级边缘权重
    # for ablation study
    last_only=False,
    use_gradient_sam=False,
    use_gradient_para_sam=False,
    interval=-1,
    forward_steps=None,
    diffusion_steps=None,
    w=None,
):
    """
    thr_01: threshold for cosine similarity (healthy or unhealthy)
    thr: threshold for predicted mask (if not provided, it will be calculated)
    t_e: steps (noise scale) for predicted mask (if not provided, it will be calculated)
    mse: batch_size x sample_steps x n_modality x H x W 
    mse_null: batch_size x sample_steps x n_modality x H x W 
    """
    # sub-anomaly maps for aggregation
    if not use_gradient_sam:
        mse = (
            xstarts["xstart"] - source[:, modality, ...].unsqueeze(1)
        ) ** 2  # batch_size x sample_steps x n_modality x 128 x 128
    else: # for ablation study
        assert forward_steps is not None
        assert diffusion_steps is not None
        assert w is not None
        beta_start = 0.0001
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        Bt = (sqrt_one_minus_alphas_cumprod)**2 / sqrt_alphas_cumprod 
        Bt = Bt[:forward_steps]
        Bt = torch.tensor(Bt, dtype=torch.float32).to(device)
        mse = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
        if not use_gradient_para_sam:
            mse = mse / (Bt**2)[None, :, None, None, None] / (1+w)**2

    # for cosine similarity
    mse_flat = torch.mean(
        (xstarts["xstart"] - source[:, modality, ...].unsqueeze(1)) ** 2, dim=(2, 3, 4)
    )  # batch_size x sample_steps

    mse_null_flat = torch.mean(
        (xstarts["xstart_null"] - source[:, modality, ...].unsqueeze(1)) ** 2, dim=(2, 3, 4),
    )
    
    # for t_e selection
    diff = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
    if edge_loss is not None:
        # 线性增强
        # diff = diff * (1 + edge_weight * edge_loss)
        #指数增强
        # Convert the scalar edge_loss to a tensor before applying torch.exp
        edge_loss_tensor = torch.tensor(edge_loss, device=diff.device, dtype=diff.dtype)
        diff = diff * torch.exp(edge_weight * edge_loss_tensor)

    diff_flat = torch.mean(diff, dim=(3, 4))
     # batch_size x sample_steps x n_modality

    batch_mask = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    batch_mask_all = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    batch_map = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    quant_range = torch.flip(torch.linspace(0.90, 0.98, 101), dims=(0,)).to(device)

    pred_lab = []
    end_steps = []
    for sample_num in range(diff_flat.shape[0]):
        # get the cosine similarity between mse and mse_null
        sim = cos(
            mse_flat[sample_num : sample_num + 1, ...],
            mse_null_flat[sample_num : sample_num + 1, ...],
        )

        # get the quantile threshold for predicted mask
        diff_i = diff_flat[sample_num, ...]  # sample_steps x n_modality
        diff_max_i = diff_i.max(dim=0)[0]  # n_modality
        diff_max_i = torch.clamp((diff_max_i / diff_max), 0, 1)
        diff_max_i = torch.round(diff_max_i, decimals=2) * 100
        index = diff_max_i.to(torch.int64)  # n_modality
        quant = quant_range[index]

        # get the steps for predicted mask
        t_s_i = torch.tensor([0, 0], device=device)
        t_e_i = torch.argmax(diff_i, dim=0) if t_e is None else t_e  # n_modality

        # for ablation study
        if t_e_ratio != 1:
            t_e_i = torch.round(t_e_i * t_e_ratio).to(torch.int64)
            end_steps.append(t_e_i)
        # for ablation study
        if last_only:
            t_s_i = t_e_i - 1
            assert interval == -1  # no interval for last_only
            
        # for each modality
        thr_i = 0
        mapp = torch.zeros(1, 1, shape, shape).to(device)
        for mod in range(mse.shape[2]):
            mse_subset = mse[
                sample_num, t_s_i[mod] : t_e_i[mod], [mod], ...
            ]  # sample_steps x 1 x 128 x 128

            ############################################################
            # jumping interval for ablation study and memory efficiency
            if interval != -1:
                assert interval > 0
                mse_subset = mse_subset[::interval, ...]
                # make sure add the last step
                mse_subset = torch.cat(
                    [
                        mse_subset,
                        mse[sample_num, t_e_i[mod] : t_e_i[mod] + 1, [mod], ...],
                    ],
                    dim=0,
                )
            ############################################################
            
            mask_mod = torch.mean(
                mse_subset, axis=[0, 1], keepdim=True
            )  # 1 x 1 x 128 x 128

            thr_i += torch.quantile(mask_mod.reshape(-1), quant[mod])
            mapp += mask_mod

        # collect the predicted mask and map
        mapp /= mse.shape[2]  # average over n_modality
        mapp = (
            median_pool(mapp, kernel_size=5, stride=1, padding=2)
            if median_filter
            else mapp
        )
        batch_map[sample_num] = mapp

        thr_i = (thr_i / mse.shape[2]) if thr is None else thr
        mask = mapp >= thr_i
        batch_mask_all[sample_num] = mask.float() # for the unhealthy setup

        if sim <= thr_01:
            batch_mask[sample_num] = mask.float() # for the mixed setup
            pred_lab.append(1)
        else:
            pred_lab.append(0)

    return batch_mask, batch_mask_all, torch.tensor(pred_lab), batch_map, torch.tensor(end_steps)


# %% For non-dynamical threshold to obtain pred_mask (other comparison methods)
def get_mask_batch(source, target, threshold, mod, median_filter=True):
    mse = (
        ((source[:, mod, ...] - target[:, mod, ...])**2).mean(dim=1, keepdims=True)
    )  # nx1x128x128

    mse = (
        median_pool(mse, kernel_size=5, stride=1, padding=2)
        if median_filter
        else mse
    )

    mse_mask = mse >= threshold
    mse_mask = mse_mask.float()
    pred_lab = (torch.sum(mse_mask, dim=(1, 2, 3)) > 0).float().cpu()
    return mse_mask, mse, pred_lab


def obtain_optimal_threshold(
    data_val,
    diffusion,
    model,
    args,
    device,
    ddib=True,
    guided=True,
    cond_fn=None,
    noise_fn=None,
    use_ddpm=False,
):

    TARGET = []
    SOURCE = []
    MASK = []
    for i in range(args.num_batches_val):
        source_val, mask_val, _ = data_val.__iter__().__next__()
        source_val = source_val.to(device)
        mask_val = mask_val.to(device)

        # Forward process
        # if ddib, image will be encoded by DDIM forward process
        # ddib is from the paper "DUAL DIFFUSION IMPLICIT BRIDGES FOR IMAGE-TO-IMAGE TRANSLATION"
        if ddib:
            noise, _ = sample(
                model,
                diffusion,
                noise=source_val,
                sample_steps=args.sample_steps,
                reverse=True,
                null=True,
                dynamic_clip=args.dynamic_clip,
                normalize_img=False,
                ddpm=False,
            )
        else:
            t = torch.tensor(
                [args.sample_steps - 1] * source_val.shape[0], device=device
            )
            ep = noise_fn(source_val, t) if noise_fn else None

            noise = diffusion.q_sample(source_val, t=t, noise=ep)

        # if guided, y will be used to guide the sampling process
        if guided:
            y = torch.ones(source_val.shape[0], dtype=torch.long) * torch.arange(
                start=0, end=1
            ).reshape(
                -1, 1
            )  # 0 for healthy
            y = y.reshape(-1, 1).squeeze().to(device)
        else:
            y = None

        # sampling process
        target, _ = sample(
            model,
            diffusion,
            y=y,
            noise=noise,
            w=args.w,
            noise_fn=noise_fn,
            cond_fn=cond_fn,
            sample_shape=source_val.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
            ddpm=use_ddpm,
        )
        TARGET.append(target)
        SOURCE.append(source_val)
        MASK.append(mask_val)

    TARGET = torch.cat(TARGET, dim=0)
    SOURCE = torch.cat(SOURCE, dim=0)
    MASK = torch.cat(MASK, dim=0)

    dice_max = 0
    thr_opt = 0
    # range of threshold, select the best one
    threshold_range = np.arange(0.01, 0.7, 0.01)
    for thr in threshold_range:
        PRED_MASK, PRED_MAP, _ = get_mask_batch(
            SOURCE, TARGET, thr, args.modality, median_filter=True
        )
        eval_metrics = evaluate(MASK, PRED_MASK, SOURCE, PRED_MAP)

        if eval_metrics["dice"] > dice_max:
            dice_max = eval_metrics["dice"]
            thr_opt = thr

    return thr_opt, dice_max
