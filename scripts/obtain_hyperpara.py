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

    get the 99th percentile difference threshold to differentiate healthy and tumour slices
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
        
        # 计算健康样本的99分位数差异
        diff_0 = torch.abs(torch.mean(mse_0_flat, dim=2) - torch.mean(mse_0_null_flat, dim=2))
        output_0 = torch.quantile(diff_0, 0.99, dim=1)  # N_val
        
        if len(unhealthy_indices) == 0:
            print("Warning: No unhealthy samples (label=1) found in validation data")
            # Use a default threshold when no unhealthy samples are available (99分位数差异)
            thr_01 = torch.quantile(output_0, 0.99)
            n_min = 0
        else:
            mse_1_flat = mse_flat[unhealthy_indices]
            mse_1_null_flat = mse_null_flat[unhealthy_indices]
            
            # 计算异常样本的99分位数差异
            diff_1 = torch.abs(torch.mean(mse_1_flat, dim=2) - torch.mean(mse_1_null_flat, dim=2))
            output_1 = torch.quantile(diff_1, 0.99, dim=1)  # N_val
            
            n_min = 1e6
            # 注意：95分位数差异越大越异常，所以阈值选择逻辑需要调整
            for q in np.linspace(0.905, 0.995, 100):  # 使用高分位数作为阈值候选
                thr_candidate = torch.quantile(output_0, q)
                n = torch.sum(output_1 < thr_candidate) + torch.sum(
                    output_0 > thr_candidate
                )
                if n < n_min:
                    n_min = n
                    thr_01 = thr_candidate
    
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
    use_timestep_weights=False,  # 独立的时间步权重参数
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
        # get the 99th percentile difference between mse and mse_null (替换余弦相似度为99分位数差异)
        diff_seq = torch.abs(mse_flat[sample_num, :] - mse_null_flat[sample_num, :])
        sim = torch.quantile(diff_seq, 0.99)  # 使用99分位数作为异常分数

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
            
            # 计算温和的平方根衰减权重（独立于梯度SAM）
            if use_timestep_weights and mse_subset.shape[1] > 1:
                T = mse_subset.shape[1]  # 时间步总数
                # 为每个时间步计算平方根衰减权重（温和衰减）
                time_weights = torch.sqrt(torch.arange(T, 0, -1, dtype=torch.float32, device=device) / T)
                time_weights = time_weights.view(1, -1, 1, 1, 1)  # [1, T, 1, 1, 1]
                
                # 应用时间步权重进行加权平均
                weighted_mse = mse_subset * time_weights
                mask_mod = torch.sum(weighted_mse, axis=[0, 1], keepdim=True) / torch.sum(time_weights)
            else:
                # 默认等权重平均
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

        # 注意：99分位数差异越大越异常，所以判断逻辑与余弦相似度相反
        if sim >= thr_01:
            batch_mask[sample_num] = mask.float() # for the mixed setup
            pred_lab.append(1)
        else:
            pred_lab.append(0)

    return batch_mask, batch_mask_all, torch.tensor(pred_lab), batch_map, torch.tensor(end_steps)


def calculate_adaptive_local_entropy(feature_map, window_size=5, complexity_threshold=0.1):
    """
    计算自适应局部熵
    Args:
        feature_map: 输入特征图 (batch_size, 1, H, W)
        window_size: 滑动窗口大小
        complexity_threshold: 复杂度阈值，用于自适应调整
    Returns:
        entropy_map: 局部熵图 (batch_size, 1, H, W)
    """
    batch_size, channels, height, width = feature_map.shape
    device = feature_map.device
    
    # 计算全局复杂度（标准差）
    global_complexity = torch.std(feature_map, dim=(2, 3), keepdim=True)
    
    # 自适应调整窗口大小
    adaptive_window = torch.where(
        global_complexity > complexity_threshold,
        window_size + 2,  # 复杂区域使用更大窗口
        window_size
    ).int().item()
    
    pad = adaptive_window // 2
    padded_map = F.pad(feature_map, (pad, pad, pad, pad), mode='reflect')
    
    entropy_map = torch.zeros_like(feature_map)
    
    for i in range(height):
        for j in range(width):
            # 提取局部窗口
            window = padded_map[:, :, i:i+adaptive_window, j:j+adaptive_window]
            
            # 计算直方图（简化为8个bins）
            window_flat = window.flatten(start_dim=2)
            min_val = window_flat.min(dim=2, keepdim=True)[0]
            max_val = window_flat.max(dim=2, keepdim=True)[0]
            
            # 避免除零
            range_val = max_val - min_val + 1e-8
            normalized = (window_flat - min_val) / range_val
            
            # 计算直方图
            bins = torch.linspace(0, 1, 9, device=device)
            hist = torch.zeros(batch_size, channels, 8, device=device)
            
            for b in range(8):
                mask = (normalized >= bins[b]) & (normalized < bins[b+1])
                hist[:, :, b] = mask.float().sum(dim=2)
            
            # 归一化直方图
            hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-8)
            
            # 计算熵
            entropy = -torch.sum(hist * torch.log(hist + 1e-8), dim=2)
            entropy_map[:, :, i, j] = entropy.squeeze()
    
    return entropy_map


def get_mask_batch_FPDM_dual_threshold(
    xstarts,
    source,
    modality,
    thr_01,
    diff_min,
    diff_max,
    shape,
    device,
    # 双阈值策略参数
    enable_dual_threshold=False,
    low_quant_offset=-0.05,  # 低阈值偏移（相对于原始量化点）
    high_quant_offset=0.05,  # 高阈值偏移（相对于原始量化点）
    entropy_weight=0.3,      # 局部熵权重
    entropy_threshold=0.5,   # 局部熵阈值
    # 原有参数
    thr=None,
    t_e=None,
    t_e_ratio=1,
    median_filter=True,
    edge_loss=None,
    edge_weight=1.0,
    attention_edge_weight=1.0,
    last_only=False,
    use_gradient_sam=False,
    use_gradient_para_sam=False,
    interval=-1,
    forward_steps=None,
    diffusion_steps=None,
    w=None,
):
    """
    双阈值策略的FPDM掩码生成函数
    
    Args:
        enable_dual_threshold: 是否启用双阈值策略
        low_quant_offset: 低阈值量化点偏移
        high_quant_offset: 高阈值量化点偏移
        entropy_weight: 局部熵在最终掩码中的权重
        entropy_threshold: 局部熵二值化阈值
        其他参数与原函数相同
    
    Returns:
        如果启用双阈值策略，返回融合后的掩码；否则返回原始结果
    """
    # 如果未启用双阈值策略，直接调用原函数
    if not enable_dual_threshold:
        return get_mask_batch_FPDM(
            xstarts, source, modality, thr_01, diff_min, diff_max, shape, device,
            thr, t_e, t_e_ratio, median_filter, edge_loss, edge_weight, 
            attention_edge_weight, last_only, use_gradient_sam, use_gradient_para_sam,
            interval, forward_steps, diffusion_steps, w
        )
    
    # 获取原始结果作为基础
    batch_mask, batch_mask_all, pred_lab, batch_map, end_steps = get_mask_batch_FPDM(
        xstarts, source, modality, thr_01, diff_min, diff_max, shape, device,
        thr, t_e, t_e_ratio, median_filter, edge_loss, edge_weight,
        attention_edge_weight, last_only, use_gradient_sam, use_gradient_para_sam,
        interval, forward_steps, diffusion_steps, w
    )
    
    # 双阈值策略处理
    if not use_gradient_sam:
        mse = (
            xstarts["xstart"] - source[:, modality, ...].unsqueeze(1)
        ) ** 2
    else:
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
    
    # 处理边缘损失
    diff = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
    if edge_loss is not None:
        edge_loss_tensor = torch.tensor(edge_loss, device=diff.device, dtype=diff.dtype)
        diff = diff * torch.exp(edge_weight * edge_loss_tensor)
    
    diff_flat = torch.mean(diff, dim=(3, 4))
    
    # 动态量化范围
    quant_range = torch.flip(torch.linspace(0.90, 0.98, 101), dims=(0,)).to(device)
    
    # 为每个样本生成双阈值掩码
    batch_mask_dual = torch.zeros_like(batch_mask)
    
    for sample_num in range(diff_flat.shape[0]):
        # 跳过健康样本（pred_lab[sample_num] == 0）
        if pred_lab[sample_num] == 0:
            batch_mask_dual[sample_num] = batch_mask[sample_num]
            continue
        
        # 计算动态量化点
        diff_i = diff_flat[sample_num, ...]
        diff_max_i = diff_i.max(dim=0)[0]
        diff_max_i = torch.clamp((diff_max_i / diff_max), 0, 1)
        diff_max_i = torch.round(diff_max_i, decimals=2) * 100
        index = diff_max_i.to(torch.int64)
        
        # 原始量化点
        base_quant = quant_range[index]
        
        # 计算低阈值和高阈值的量化点
        low_index = torch.clamp(index + int(low_quant_offset * 100), 0, 100)
        high_index = torch.clamp(index + int(high_quant_offset * 100), 0, 100)
        
        low_quant = quant_range[low_index]
        high_quant = quant_range[high_index]
        
        # 获取时间步
        t_s_i = torch.tensor([0, 0], device=device)
        t_e_i = torch.argmax(diff_i, dim=0) if t_e is None else t_e
        
        if t_e_ratio != 1:
            t_e_i = torch.round(t_e_i * t_e_ratio).to(torch.int64)
        if last_only:
            t_s_i = t_e_i - 1
        
        # 为每个模态计算双阈值掩码
        low_thr_i = 0
        high_thr_i = 0
        mapp = torch.zeros(1, 1, shape, shape).to(device)
        
        for mod in range(mse.shape[2]):
            mse_subset = mse[sample_num, t_s_i[mod]:t_e_i[mod], [mod], ...]
            
            if interval != -1:
                assert interval > 0
                mse_subset = mse_subset[::interval, ...]
                mse_subset = torch.cat([
                    mse_subset,
                    mse[sample_num, t_e_i[mod]:t_e_i[mod] + 1, [mod], ...],
                ], dim=0)
            
            mask_mod = torch.mean(mse_subset, axis=[0, 1], keepdim=True)
            mapp += mask_mod
            
            # 计算低阈值和高阈值
            low_thr_i += torch.quantile(mask_mod.reshape(-1), low_quant[mod])
            high_thr_i += torch.quantile(mask_mod.reshape(-1), high_quant[mod])
        
        # 平均化
        mapp /= mse.shape[2]
        mapp = (
            median_pool(mapp, kernel_size=5, stride=1, padding=2)
            if median_filter else mapp
        )
        
        low_thr_i /= mse.shape[2]
        high_thr_i /= mse.shape[2]
        
        # 生成候选掩码（低阈值，高召回）和高置信掩码（高阈值，高精度）
        candidate_mask = (mapp >= low_thr_i).float()
        confident_mask = (mapp >= high_thr_i).float()
        
        # 计算自适应局部熵
        entropy_map = calculate_adaptive_local_entropy(mapp)
        entropy_mask = (entropy_map >= entropy_threshold).float()
        
        # 融合策略：高置信区域 + (候选区域 ∩ 高熵区域)
        final_mask = confident_mask + entropy_weight * (candidate_mask * entropy_mask)
        final_mask = torch.clamp(final_mask, 0, 1)
        
        batch_mask_dual[sample_num] = final_mask
    
    return batch_mask_dual, batch_mask_all, pred_lab, batch_map, end_steps


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
