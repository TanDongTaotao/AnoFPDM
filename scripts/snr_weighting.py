"""
SNR权重计算模块
实现SNR代理权重方案，用于Dilated UNet异常检测的异常子图聚合优化

作者：基于之前的分析和推荐方案实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, List


def calculate_snr_proxy_weights(
    xstarts: Dict[str, torch.Tensor],
    source: torch.Tensor,
    modality,
    device: torch.device,
    snr_smoothing: float = 1.0,
    temporal_decay: float = 0.98,
    min_weight: float = 0.3,
    max_weight: float = 1.5,
    consistency_weight: float = 0.7,
    sensitivity_weight: float = 0.3
) -> torch.Tensor:
    """
    计算改进的SNR代理权重用于异常检测
    
    基于重建一致性和异常敏感性的新方法：
    - 重建一致性：健康引导重建的稳定性（越稳定权重越高）
    - 异常敏感性：引导差异的显著性（差异越大权重越高）
    
    Args:
        xstarts: 包含'xstart'和'xstart_null'的字典
                xstart: [batch_size, sample_steps, H, W] - 健康引导重建
                xstart_null: [batch_size, sample_steps, H, W] - 无引导重建
        source: [batch_size, n_modality, H, W] - 原始输入图像
        modality: int or list - 模态索引（如果是列表则使用第一个）
        device: torch.device - 计算设备
        snr_smoothing: float - SNR平滑参数，避免除零
        temporal_decay: float - 时间衰减因子
        min_weight: float - 最小权重值
        max_weight: float - 最大权重值
        consistency_weight: float - 重建一致性权重
        sensitivity_weight: float - 异常敏感性权重
    
    Returns:
        weights: [batch_size, sample_steps] - 改进的SNR权重
    """
    
    # 处理modality参数（可能是列表）
    if isinstance(modality, list):
        if len(modality) == 0:
            raise ValueError("modality list cannot be empty")
        modality_idx = modality[0]  # 使用第一个模态
    else:
        modality_idx = modality
    
    # 提取张量
    xstart = xstarts["xstart"]  # 可能是 [batch_size, sample_steps, H, W] 或 [batch_size, sample_steps, C, H, W]
    xstart_null = xstarts["xstart_null"]  # 可能是 [batch_size, sample_steps, H, W] 或 [batch_size, sample_steps, C, H, W]
    
    # 检查张量维度并适配
    if len(xstart.shape) == 5:  # [batch_size, sample_steps, C, H, W]
        # 检查模态索引是否有效
        if modality_idx < 0 or modality_idx >= xstart.shape[2]:
            raise ValueError(f"modality index {modality_idx} is out of bounds for tensor with {xstart.shape[2]} channels")
        
        # 选择指定模态
        xstart = xstart[:, :, modality_idx, :, :]  # [batch_size, sample_steps, H, W]
        xstart_null = xstart_null[:, :, modality_idx, :, :]  # [batch_size, sample_steps, H, W]
        
        # 检查source张量的模态维度
        if modality_idx >= source.shape[1]:
            raise ValueError(f"modality index {modality_idx} is out of bounds for source tensor with {source.shape[1]} channels")
        source_mod = source[:, modality_idx, ...].unsqueeze(1)  # [batch_size, 1, H, W]
    elif len(xstart.shape) == 4:  # [batch_size, sample_steps, H, W]
        # 已经是正确的维度，检查source张量的模态维度
        if modality_idx >= source.shape[1]:
            raise ValueError(f"modality index {modality_idx} is out of bounds for source tensor with {source.shape[1]} channels")
        source_mod = source[:, modality_idx, ...].unsqueeze(1)  # [batch_size, 1, H, W]
    else:
        raise ValueError(f"Unexpected xstart shape: {xstart.shape}. Expected 4D or 5D tensor.")
    
    batch_size, sample_steps, H, W = xstart.shape
    
    # 1. 计算重建一致性 (Reconstruction Consistency)
    # 健康引导重建的稳定性：重建误差越小，一致性越高
    reconstruction_error = torch.mean((xstart - source_mod) ** 2, dim=(2, 3))  # [batch_size, sample_steps]
    consistency = 1.0 / (reconstruction_error + snr_smoothing)  # 误差越小，一致性越高
    
    # 2. 计算异常敏感性 (Anomaly Sensitivity)
    # 引导差异的显著性：差异越大，对异常越敏感
    guidance_difference = torch.mean((xstart - xstart_null) ** 2, dim=(2, 3))  # [batch_size, sample_steps]
    sensitivity = guidance_difference  # 差异越大，敏感性越高
    
    # 3. 组合权重计算
    # 结合重建一致性和异常敏感性
    combined_weights = consistency_weight * consistency + sensitivity_weight * sensitivity
    
    # 4. 应用时间衰减
    # 早期时间步通常包含更多噪声，应用衰减权重
    time_weights = torch.pow(temporal_decay, torch.arange(sample_steps, device=device))
    time_weights = time_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, sample_steps]
    
    # 5. 结合组合权重和时间权重
    raw_weights = combined_weights * time_weights
    
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
    aggregation_mode: str = "robust_weighted"
) -> torch.Tensor:
    """
    应用改进的SNR权重进行异常子图聚合
    
    Args:
        mse_subset: [sample_steps, 1, H, W] - 异常子图MSE
        snr_weights: [sample_steps] - 对应的SNR权重
        sample_indices: [sample_steps] - 时间步索引
        aggregation_mode: str - 聚合模式 ("robust_weighted", "weighted_mean", "weighted_sum")
    
    Returns:
        aggregated_map: [1, 1, H, W] - 聚合后的异常图
    """
    
    # 确保权重维度匹配
    if snr_weights.dim() == 1:
        weights = snr_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [sample_steps, 1, 1, 1]
    else:
        weights = snr_weights
    
    if aggregation_mode == "robust_weighted":
        # 稳健加权方法：限制权重范围，避免极端值
        # 1. 限制权重范围，避免极端值
        clipped_weights = torch.clamp(weights, 0.5, 1.5)
        
        # 2. 应用权重
        weighted_mse = mse_subset * clipped_weights  # [sample_steps, 1, H, W]
        
        # 3. 稳健平均（使用中位数减少极值影响）
        aggregated_map = torch.median(weighted_mse, dim=0, keepdim=True)[0]
        
    elif aggregation_mode == "weighted_mean":
        # 传统加权平均
        weighted_mse = mse_subset * weights  # [sample_steps, 1, H, W]
        weight_sum = torch.sum(weights, dim=0, keepdim=True)  # [1, 1, 1, 1]
        aggregated_map = torch.sum(weighted_mse, dim=0, keepdim=True) / (weight_sum + 1e-8)
        
    elif aggregation_mode == "weighted_sum":
        # 加权求和
        weighted_mse = mse_subset * weights  # [sample_steps, 1, H, W]
        aggregated_map = torch.sum(weighted_mse, dim=0, keepdim=True)
        
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation_mode}. Supported: 'robust_weighted', 'weighted_mean', 'weighted_sum'")
    
    return aggregated_map


def get_mask_batch_FPDM_with_snr_weighting(
    xstarts: Dict[str, torch.Tensor],
    source: torch.Tensor,
    modality: Union[int, List[int]],
    thr_01: float,
    diff_min: Union[float, torch.Tensor],
    diff_max: Union[float, torch.Tensor],
    shape: int,
    device: torch.device,
    # SNR权重参数
    enable_snr_weighting: bool = True,
    snr_smoothing: float = 1.0,
    temporal_decay: float = 0.98,
    min_weight: float = 0.3,
    max_weight: float = 1.5,
    consistency_weight: float = 0.7,
    sensitivity_weight: float = 0.3,
    aggregation_mode: str = "robust_weighted",
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
    
    # 处理modality参数（可能是列表）
    if isinstance(modality, list):
        if len(modality) == 0:
            raise ValueError("modality list cannot be empty")
        modality_idx = modality[0]  # 使用第一个模态
    else:
        modality_idx = modality
    
    # 处理多模态的diff_min和diff_max
    if isinstance(diff_min, torch.Tensor) and diff_min.numel() > 1:
        # 多模态情况，选择对应模态的值
        if modality_idx >= diff_min.numel():
            raise ValueError(f"modality index {modality_idx} is out of bounds for diff_min tensor with {diff_min.numel()} elements")
        diff_min_scalar = diff_min[modality_idx].item()
    else:
        # 单模态或标量情况
        diff_min_scalar = diff_min.item() if isinstance(diff_min, torch.Tensor) else diff_min
    
    if isinstance(diff_max, torch.Tensor) and diff_max.numel() > 1:
        # 多模态情况，选择对应模态的值
        if modality_idx >= diff_max.numel():
            raise ValueError(f"modality index {modality_idx} is out of bounds for diff_max tensor with {diff_max.numel()} elements")
        diff_max_scalar = diff_max[modality_idx].item()
    else:
        # 单模态或标量情况
        diff_max_scalar = diff_max.item() if isinstance(diff_max, torch.Tensor) else diff_max
    
    # 计算SNR权重（如果启用）
    snr_weights = None
    if enable_snr_weighting:
        snr_weights = calculate_snr_proxy_weights(
            xstarts, source, modality, device,
            snr_smoothing, temporal_decay, min_weight, max_weight,
            consistency_weight, sensitivity_weight
        )
    
    # 提取并适配xstart张量
    xstart = xstarts["xstart"]
    xstart_null = xstarts["xstart_null"]
    
    # 检查张量维度并适配
    if len(xstart.shape) == 5:  # [batch_size, sample_steps, C, H, W]
        # 检查模态索引是否有效
        if modality_idx < 0 or modality_idx >= xstart.shape[2]:
            raise ValueError(f"modality index {modality_idx} is out of bounds for tensor with {xstart.shape[2]} channels")
        
        # 选择指定模态
        xstart = xstart[:, :, modality_idx, :, :]  # [batch_size, sample_steps, H, W]
        xstart_null = xstart_null[:, :, modality_idx, :, :]  # [batch_size, sample_steps, H, W]
    elif len(xstart.shape) == 4:  # [batch_size, sample_steps, H, W]
        # 已经是正确的维度
        pass
    else:
        raise ValueError(f"Unexpected xstart shape: {xstart.shape}. Expected 4D or 5D tensor.")
    
    # 检查source张量的模态维度
    if modality_idx >= source.shape[1]:
        raise ValueError(f"modality index {modality_idx} is out of bounds for source tensor with {source.shape[1]} channels")
    
    # 计算异常子图MSE
    if not use_gradient_sam:
        mse = (
            xstart - source[:, modality_idx, ...].unsqueeze(1)
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
        mse = (xstart - xstart_null) ** 2
        if not use_gradient_para_sam:
            mse = mse / (Bt**2)[None, :, None, None] / (1+w)**2

    # 计算余弦相似度用的平坦MSE
    mse_flat = torch.mean(
        (xstart - source[:, modality_idx, ...].unsqueeze(1)) ** 2, dim=(2, 3)
    )  # batch_size x sample_steps

    mse_null_flat = torch.mean(
        (xstart_null - source[:, modality_idx, ...].unsqueeze(1)) ** 2, dim=(2, 3),
    )
    
    # 计算差异用于t_e选择
    diff = (xstart - xstart_null) ** 2
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
        diff_i = diff_flat[sample_num, :]  # [sample_steps]
        diff_max_i = diff_i.max()  # 确保返回标量
        diff_max_i = torch.clamp((diff_max_i / diff_max_scalar), 0, 1)
        diff_max_i = torch.round(diff_max_i, decimals=2) * 100
        index = diff_max_i.to(torch.int64)
        # 确保 index 在有效范围内
        index = torch.clamp(index, 0, len(quant_range) - 1)
        quant = quant_range[index]

        # 计算时间步
        t_s_i = torch.tensor(0, device=device)
        t_e_i = torch.argmax(diff_i).item() if t_e is None else t_e

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


def analyze_snr_weights(
    snr_weights: torch.Tensor,
    mse_subset: torch.Tensor,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    分析SNR权重分布和影响
    
    Args:
        snr_weights: [batch_size, sample_steps] - SNR权重
        mse_subset: [sample_steps, 1, H, W] - 异常子图MSE
        save_path: 可选的保存路径
        verbose: 是否打印详细信息
    
    Returns:
        analysis_results: 分析结果字典
    """
    
    # 权重统计
    weight_stats = {
        "min_weight": snr_weights.min().item(),
        "max_weight": snr_weights.max().item(),
        "mean_weight": snr_weights.mean().item(),
        "std_weight": snr_weights.std().item(),
        "weight_range": (snr_weights.max() - snr_weights.min()).item()
    }
    
    if verbose:
        print("=== SNR权重分析报告 ===")
        print(f"权重统计:")
        print(f"  最小值: {weight_stats['min_weight']:.4f}")
        print(f"  最大值: {weight_stats['max_weight']:.4f}")
        print(f"  平均值: {weight_stats['mean_weight']:.4f}")
        print(f"  标准差: {weight_stats['std_weight']:.4f}")
        print(f"  权重范围: {weight_stats['weight_range']:.4f}")
    
    # 比较加权前后的异常图
    if mse_subset is not None:
        # 原始平均
        original_map = torch.mean(mse_subset, dim=0)
        
        # 加权聚合（使用第一个样本的权重）
        sample_weights = snr_weights[0] if snr_weights.dim() > 1 else snr_weights
        weighted_map = apply_snr_weighted_aggregation(
            mse_subset, sample_weights, None, "robust_weighted"
        )
        
        # 计算相关性
        original_flat = original_map.flatten()
        weighted_flat = weighted_map.flatten()
        
        # 计算皮尔逊相关系数
        correlation = torch.corrcoef(torch.stack([original_flat, weighted_flat]))[0, 1].item()
        
        # 计算差异统计
        diff_map = weighted_map - original_map
        diff_stats = {
            "correlation": correlation,
            "mean_diff": diff_map.mean().item(),
            "std_diff": diff_map.std().item(),
            "max_abs_diff": diff_map.abs().max().item()
        }
        
        weight_stats.update(diff_stats)
        
        if verbose:
            print(f"\n加权效果分析:")
            print(f"  加权前后相关性: {correlation:.4f}")
            print(f"  平均差异: {diff_stats['mean_diff']:.4f}")
            print(f"  差异标准差: {diff_stats['std_diff']:.4f}")
            print(f"  最大绝对差异: {diff_stats['max_abs_diff']:.4f}")
    
    # 权重分布分析
    if snr_weights.numel() > 1:
        # 计算权重分布的均匀性
        uniform_weights = torch.ones_like(snr_weights) / snr_weights.numel()
        kl_divergence = torch.nn.functional.kl_div(
            torch.log(snr_weights / snr_weights.sum() + 1e-8),
            uniform_weights,
            reduction='sum'
        ).item()
        
        weight_stats["kl_divergence"] = kl_divergence
        
        if verbose:
            print(f"\n权重分布分析:")
            print(f"  与均匀分布的KL散度: {kl_divergence:.4f}")
            print(f"  (KL散度越小，权重分布越均匀)")
    
    if verbose:
        print("=" * 30)
    
    return weight_stats


def compare_aggregation_methods(
    mse_subset: torch.Tensor,
    snr_weights: torch.Tensor,
    methods: List[str] = ["robust_weighted", "weighted_mean", "weighted_sum"],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    比较不同聚合方法的效果
    
    Args:
        mse_subset: [sample_steps, 1, H, W] - 异常子图MSE
        snr_weights: [sample_steps] - SNR权重
        methods: 要比较的聚合方法列表
        verbose: 是否打印详细信息
    
    Returns:
        results: 不同方法的聚合结果
    """
    
    results = {}
    
    # 原始平均作为基准
    original_map = torch.mean(mse_subset, dim=0, keepdim=True)
    results["original_mean"] = original_map
    
    if verbose:
        print("=== 聚合方法比较 ===")
        print(f"原始平均 - 异常值范围: [{original_map.min():.4f}, {original_map.max():.4f}]")
    
    # 测试不同聚合方法
    for method in methods:
        try:
            aggregated_map = apply_snr_weighted_aggregation(
                mse_subset, snr_weights, None, method
            )
            results[method] = aggregated_map
            
            # 计算与原始方法的相关性
            correlation = torch.corrcoef(torch.stack([
                original_map.flatten(),
                aggregated_map.flatten()
            ]))[0, 1].item()
            
            if verbose:
                print(f"{method} - 异常值范围: [{aggregated_map.min():.4f}, {aggregated_map.max():.4f}], "
                      f"与原始相关性: {correlation:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"{method} - 错误: {str(e)}")
            results[method] = None
    
    if verbose:
        print("=" * 30)
    
    return results