"""
SNR权重计算模块
实现SNR代理权重方案，用于Dilated UNet异常检测的异常子图聚合优化

作者：基于之前的分析和推荐方案实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_snr_proxy_weights(
    xstarts: Dict[str, torch.Tensor],
    source: torch.Tensor,
    modality: int,
    device: torch.device,
    snr_smoothing: float = 0.1,
    temporal_decay: float = 0.95,
    min_weight: float = 0.1,
    max_weight: float = 2.0
) -> torch.Tensor:
    """
    计算SNR代理权重，用于异常子图的加权聚合
    
    Args:
        xstarts: 包含'xstart'和'xstart_null'的字典
                xstart: [batch_size, sample_steps, H, W] - 健康引导重建
                xstart_null: [batch_size, sample_steps, H, W] - 无引导重建
        source: [batch_size, n_modality, H, W] - 原始输入图像
        modality: int - 模态索引
        device: torch.device - 计算设备
        snr_smoothing: float - SNR平滑参数，避免除零
        temporal_decay: float - 时间衰减因子
        min_weight: float - 最小权重值
        max_weight: float - 最大权重值
    
    Returns:
        weights: [batch_size, sample_steps] - SNR代理权重
    """
    
    # 提取张量
    xstart = xstarts["xstart"]  # [batch_size, sample_steps, H, W]
    xstart_null = xstarts["xstart_null"]  # [batch_size, sample_steps, H, W]
    source_mod = source[:, modality, ...].unsqueeze(1)  # [batch_size, 1, H, W]
    
    batch_size, sample_steps, H, W = xstart.shape
    
    # 1. 计算信号强度 (Signal)
    # 使用健康引导重建与原始图像的相似性作为信号强度
    signal = torch.mean((xstart - source_mod) ** 2, dim=(2, 3))  # [batch_size, sample_steps]
    
    # 2. 计算噪声强度 (Noise)
    # 使用健康引导与无引导重建的差异作为噪声强度
    noise = torch.mean((xstart - xstart_null) ** 2, dim=(2, 3))  # [batch_size, sample_steps]
    
    # 3. 计算SNR代理
    # SNR = Signal / (Noise + smoothing)
    snr_proxy = signal / (noise + snr_smoothing)  # [batch_size, sample_steps]
    
    # 4. 应用时间衰减
    # 早期时间步通常包含更多噪声，应用衰减权重
    time_weights = torch.pow(temporal_decay, torch.arange(sample_steps, device=device))
    time_weights = time_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, sample_steps]
    
    # 5. 结合SNR和时间权重
    raw_weights = snr_proxy * time_weights
    
    # 6. 归一化权重到指定范围
    # 对每个样本独立归一化
    for b in range(batch_size):
        sample_weights = raw_weights[b, :]
        min_val = sample_weights.min()
        max_val = sample_weights.max()
        
        if max_val > min_val:
            # 归一化到[0, 1]然后缩放到[min_weight, max_weight]
            normalized = (sample_weights - min_val) / (max_val - min_val)
            raw_weights[b, :] = min_weight + normalized * (max_weight - min_weight)
        else:
            # 如果所有权重相同，设置为平均值
            raw_weights[b, :] = (min_weight + max_weight) / 2
    
    return raw_weights


def apply_snr_weighted_aggregation(
    mse_subset: torch.Tensor,
    snr_weights: torch.Tensor,
    sample_indices: torch.Tensor,
    aggregation_mode: str = "weighted_mean"
) -> torch.Tensor:
    """
    应用SNR权重进行异常子图聚合
    
    Args:
        mse_subset: [sample_steps, 1, H, W] - 异常子图MSE
        snr_weights: [sample_steps] - 对应的SNR权重
        sample_indices: [sample_steps] - 时间步索引
        aggregation_mode: str - 聚合模式 ("weighted_mean", "weighted_sum")
    
    Returns:
        aggregated_map: [1, 1, H, W] - 聚合后的异常图
    """
    
    # 确保权重维度匹配
    if snr_weights.dim() == 1:
        weights = snr_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [sample_steps, 1, 1, 1]
    else:
        weights = snr_weights
    
    # 应用权重
    weighted_mse = mse_subset * weights  # [sample_steps, 1, H, W]
    
    if aggregation_mode == "weighted_mean":
        # 加权平均
        weight_sum = torch.sum(weights, dim=0, keepdim=True)  # [1, 1, 1, 1]
        aggregated_map = torch.sum(weighted_mse, dim=0, keepdim=True) / (weight_sum + 1e-8)
    elif aggregation_mode == "weighted_sum":
        # 加权求和
        aggregated_map = torch.sum(weighted_mse, dim=0, keepdim=True)
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")
    
    return aggregated_map


def get_mask_batch_FPDM_with_snr_weighting(
    xstarts: Dict[str, torch.Tensor],
    source: torch.Tensor,
    modality: int,
    thr_01: float,
    diff_min: float,
    diff_max: float,
    shape: int,
    device: torch.device,
    # SNR权重参数
    enable_snr_weighting: bool = True,
    snr_smoothing: float = 0.1,
    temporal_decay: float = 0.95,
    min_weight: float = 0.1,
    max_weight: float = 2.0,
    aggregation_mode: str = "weighted_mean",
    # 原有参数
    thr: Optional[float] = None,
    t_e: Optional[torch.Tensor] = None,
    t_e_ratio: float = 1,
    median_filter: bool = True,
    edge_loss: Optional[float] = None,
    edge_weight: float = 1.0,
    attention_edge_weight: float = 1.0,
    last_only: bool = False,
    use_gradient_sam: bool = False,
    use_gradient_para_sam: bool = False,
    interval: int = -1,
    forward_steps: Optional[int] = None,
    diffusion_steps: Optional[int] = None,
    w: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    带SNR权重的FPDM掩码生成函数
    
    这是对原始get_mask_batch_FPDM函数的增强版本，集成了SNR权重聚合
    
    Returns:
        batch_mask: [batch_size, 1, H, W] - 混合设置的预测掩码
        batch_mask_all: [batch_size, 1, H, W] - 不健康设置的预测掩码
        pred_lab: [batch_size] - 预测标签
        batch_map: [batch_size, 1, H, W] - 异常图
        end_steps: [batch_size, n_modality] - 结束时间步
    """
    
    # 计算SNR权重（如果启用）
    snr_weights = None
    if enable_snr_weighting:
        snr_weights = calculate_snr_proxy_weights(
            xstarts, source, modality, device,
            snr_smoothing, temporal_decay, min_weight, max_weight
        )
    
    # 计算异常子图MSE
    if not use_gradient_sam:
        mse = (
            xstarts["xstart"] - source[:, modality, ...].unsqueeze(1)
        ) ** 2  # batch_size x sample_steps x H x W
    else:
        # 梯度SAM的处理逻辑
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
            mse = mse / (Bt**2)[None, :, None, None] / (1+w)**2

    # 计算余弦相似度用的平坦MSE
    mse_flat = torch.mean(
        (xstarts["xstart"] - source[:, modality, ...].unsqueeze(1)) ** 2, dim=(2, 3)
    )  # batch_size x sample_steps

    mse_null_flat = torch.mean(
        (xstarts["xstart_null"] - source[:, modality, ...].unsqueeze(1)) ** 2, dim=(2, 3),
    )
    
    # 计算差异用于t_e选择
    diff = (xstarts["xstart"] - xstarts["xstart_null"]) ** 2
    if edge_loss is not None:
        edge_loss_tensor = torch.tensor(edge_loss, device=diff.device, dtype=diff.dtype)
        diff = diff * torch.exp(edge_weight * edge_loss_tensor)

    diff_flat = torch.mean(diff, dim=(2, 3))  # batch_size x sample_steps

    batch_mask = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    batch_mask_all = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    batch_map = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)

    quant_range = torch.flip(torch.linspace(0.90, 0.98, 101), dims=(0,)).to(device)

    pred_lab = []
    end_steps = []
    
    for sample_num in range(diff_flat.shape[0]):
        # 计算99分位数差异作为异常分数
        diff_seq = torch.abs(mse_flat[sample_num, :] - mse_null_flat[sample_num, :])
        sim = torch.quantile(diff_seq, 0.99)

        # 计算量化阈值
        diff_i = diff_flat[sample_num, ...]  # sample_steps
        diff_max_i = diff_i.max(dim=0)[0]  # scalar
        diff_max_i = torch.clamp((diff_max_i / diff_max), 0, 1)
        diff_max_i = torch.round(diff_max_i, decimals=2) * 100
        index = diff_max_i.to(torch.int64)
        quant = quant_range[index]

        # 计算时间步
        t_s_i = torch.tensor(0, device=device)
        t_e_i = torch.argmax(diff_i, dim=0) if t_e is None else t_e

        if t_e_ratio != 1:
            t_e_i = torch.round(t_e_i * t_e_ratio).to(torch.int64)
            end_steps.append(t_e_i)

        if last_only:
            t_s_i = t_e_i - 1
            assert interval == -1

        # 提取MSE子集
        mse_subset = mse[sample_num, t_s_i:t_e_i, ...]  # sample_steps x H x W

        # 处理跳跃间隔
        if interval != -1:
            assert interval > 0
            mse_subset = mse_subset[::interval, ...]
            mse_subset = torch.cat([
                mse_subset,
                mse[sample_num, t_e_i:t_e_i + 1, ...]
            ], dim=0)

        # 应用SNR权重聚合
        if enable_snr_weighting and snr_weights is not None:
            # 获取对应的SNR权重
            sample_snr_weights = snr_weights[sample_num, t_s_i:t_e_i]
            if interval != -1:
                sample_snr_weights = sample_snr_weights[::interval]
                sample_snr_weights = torch.cat([
                    sample_snr_weights,
                    snr_weights[sample_num, t_e_i:t_e_i + 1]
                ], dim=0)
            
            # 应用SNR加权聚合
            mse_subset_expanded = mse_subset.unsqueeze(1)  # [sample_steps, 1, H, W]
            sample_indices = torch.arange(t_s_i, t_e_i, device=device)
            if interval != -1:
                sample_indices = sample_indices[::interval]
                sample_indices = torch.cat([
                    sample_indices,
                    torch.tensor([t_e_i], device=device)
                ], dim=0)
            
            mask_mod = apply_snr_weighted_aggregation(
                mse_subset_expanded, sample_snr_weights, sample_indices, aggregation_mode
            )  # [1, 1, H, W]
        else:
            # 原始平均聚合
            mask_mod = torch.mean(mse_subset, axis=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]

        # 应用中值滤波
        if median_filter:
            from .obtain_hyperpara import median_pool
            mask_mod = median_pool(mask_mod, kernel_size=5, stride=1, padding=2)

        batch_map[sample_num] = mask_mod.squeeze(0)

        # 计算阈值和掩码
        thr_i = torch.quantile(mask_mod.reshape(-1), quant) if thr is None else thr
        mask = mask_mod >= thr_i
        batch_mask_all[sample_num] = mask.float().squeeze(0)

        # 预测标签
        if sim >= thr_01:
            batch_mask[sample_num] = mask.float().squeeze(0)
            pred_lab.append(1)
        else:
            pred_lab.append(0)

    return batch_mask, batch_mask_all, torch.tensor(pred_lab), batch_map, torch.tensor(end_steps)


def median_pool(input_tensor, kernel_size=3, stride=1, padding=1):
    """
    中值池化函数（如果原模块中没有的话）
    """
    # 使用unfold实现中值池化
    batch_size, channels, height, width = input_tensor.shape
    
    # 展开张量
    unfolded = F.unfold(input_tensor, kernel_size, padding=padding, stride=stride)
    # unfolded shape: [batch_size, channels * kernel_size^2, num_patches]
    
    # 重塑为 [batch_size, channels, kernel_size^2, num_patches]
    unfolded = unfolded.view(batch_size, channels, kernel_size * kernel_size, -1)
    
    # 计算中值
    median_values, _ = torch.median(unfolded, dim=2)
    
    # 计算输出尺寸
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 重塑回原始形状
    output = median_values.view(batch_size, channels, out_height, out_width)
    
    return output