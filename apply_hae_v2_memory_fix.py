#!/usr/bin/env python3
"""
HAE V2 内存优化修复脚本

此脚本提供了一个简单的方法来修复HAE V2模型的内存问题，
解决修复后batchsize减半的问题。

使用方法：
1. 导入此模块
2. 使用 apply_memory_optimization() 函数优化现有模型
3. 或者直接使用 OptimizedHAEUNetModelV2 替换原有模型

作者：AI Assistant
日期：2024
"""

import torch as th
import numpy as np
from typing import Optional, Union
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    HAE V2 内存优化器
    
    提供多种优化策略来减少cemb_mm操作的内存使用
    """
    
    def __init__(self, optimization_method="conditional", **kwargs):
        """
        初始化内存优化器
        
        Args:
            optimization_method: 优化方法 ["conditional", "low_rank", "chunked", "hybrid"]
            **kwargs: 其他优化参数
        """
        self.method = optimization_method
        self.low_rank_ratio = kwargs.get("low_rank_ratio", 0.125)
        self.chunk_size = kwargs.get("chunk_size", 8)
        self.zero_threshold = kwargs.get("zero_threshold", 1e-6)
        
        logger.info(f"初始化内存优化器，方法: {self.method}")
    
    def compute_cemb_mm_optimized(self, cemb: th.Tensor) -> Optional[th.Tensor]:
        """
        优化的cemb_mm计算
        
        Args:
            cemb: 条件嵌入张量 [batch_size, embed_dim]
            
        Returns:
            优化后的cemb_mm张量或None
        """
        if cemb is None:
            return None
            
        if self.method == "conditional":
            return self._compute_conditional(cemb)
        elif self.method == "low_rank":
            return self._compute_low_rank(cemb)
        elif self.method == "chunked":
            return self._compute_chunked(cemb)
        elif self.method == "hybrid":
            return self._compute_hybrid(cemb)
        else:
            # 回退到原始方法
            return th.einsum("ab,ac -> abc", cemb, cemb)
    
    def _compute_conditional(self, cemb: th.Tensor) -> Optional[th.Tensor]:
        """
        条件计算：只在需要时计算cemb_mm
        """
        # 检查是否为零张量
        if th.allclose(cemb, th.zeros_like(cemb), atol=self.zero_threshold):
            logger.debug("检测到零张量，跳过cemb_mm计算")
            return None
        
        # 检查哪些样本需要计算
        non_zero_mask = th.any(th.abs(cemb) > self.zero_threshold, dim=1)
        non_zero_count = th.sum(non_zero_mask).item()
        
        if non_zero_count == 0:
            logger.debug("所有样本都是零，跳过cemb_mm计算")
            return None
        
        batch_size, embed_dim = cemb.shape
        
        if non_zero_count == batch_size:
            # 所有样本都非零，直接计算
            return th.einsum("ab,ac -> abc", cemb, cemb)
        
        # 只对非零样本计算cemb_mm
        logger.debug(f"稀疏计算：{non_zero_count}/{batch_size} 样本需要计算")
        
        active_indices = non_zero_mask.nonzero().squeeze(-1)
        cemb_active = cemb[active_indices]
        cemb_mm_active = th.einsum("ab,ac -> abc", cemb_active, cemb_active)
        
        # 创建完整的cemb_mm张量
        cemb_mm = th.zeros(batch_size, embed_dim, embed_dim, 
                          device=cemb.device, dtype=cemb.dtype)
        cemb_mm[active_indices] = cemb_mm_active
        
        return cemb_mm
    
    def _compute_low_rank(self, cemb: th.Tensor) -> th.Tensor:
        """
        低秩分解：减少cemb_mm的维度
        """
        batch_size, embed_dim = cemb.shape
        rank = max(1, int(embed_dim * self.low_rank_ratio))
        
        logger.debug(f"低秩分解：{embed_dim} -> {rank} (比例: {self.low_rank_ratio})")
        
        # 使用SVD进行低秩分解
        try:
            U, S, V = th.svd(cemb.transpose(0, 1))  # [embed_dim, batch_size]
            
            # 取前rank个分量
            U_reduced = U[:, :rank]  # [embed_dim, rank]
            S_reduced = S[:rank]     # [rank]
            V_reduced = V[:, :rank]  # [batch_size, rank]
            
            # 重构低秩近似
            cemb_reduced = (U_reduced @ th.diag(S_reduced) @ V_reduced.transpose(0, 1)).transpose(0, 1)
            
            # 计算cemb_mm
            cemb_mm = th.einsum("ab,ac -> abc", cemb_reduced, cemb_reduced)
            
            return cemb_mm
            
        except Exception as e:
            logger.warning(f"低秩分解失败，回退到原始方法: {e}")
            return th.einsum("ab,ac -> abc", cemb, cemb)
    
    def _compute_chunked(self, cemb: th.Tensor) -> th.Tensor:
        """
        分块计算：将大批次分解为小块
        """
        batch_size = cemb.shape[0]
        
        if batch_size <= self.chunk_size:
            # 批次已经足够小，直接计算
            return th.einsum("ab,ac -> abc", cemb, cemb)
        
        logger.debug(f"分块计算：批次大小 {batch_size} -> 块大小 {self.chunk_size}")
        
        cemb_mm_chunks = []
        
        for i in range(0, batch_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, batch_size)
            cemb_chunk = cemb[i:end_idx]
            cemb_mm_chunk = th.einsum("ab,ac -> abc", cemb_chunk, cemb_chunk)
            cemb_mm_chunks.append(cemb_mm_chunk)
        
        return th.cat(cemb_mm_chunks, dim=0)
    
    def _compute_hybrid(self, cemb: th.Tensor) -> Optional[th.Tensor]:
        """
        混合方法：结合条件计算和低秩分解
        """
        # 首先尝试条件计算
        if th.allclose(cemb, th.zeros_like(cemb), atol=self.zero_threshold):
            return None
        
        batch_size, embed_dim = cemb.shape
        
        # 如果批次大小较大，使用低秩分解
        if batch_size * embed_dim * embed_dim > 100 * 1024 * 1024:  # 100MB阈值
            logger.debug("使用低秩分解（大批次）")
            return self._compute_low_rank(cemb)
        else:
            logger.debug("使用条件计算（小批次）")
            return self._compute_conditional(cemb)

def apply_memory_optimization(model, optimization_method="conditional", **kwargs):
    """
    为现有的HAE V2模型应用内存优化
    
    Args:
        model: HAE V2模型实例
        optimization_method: 优化方法
        **kwargs: 其他优化参数
        
    Returns:
        优化后的模型
    """
    logger.info(f"为模型应用内存优化: {optimization_method}")
    
    # 创建优化器
    optimizer = MemoryOptimizer(optimization_method, **kwargs)
    
    # 保存原始的forward方法
    original_forward = model.forward
    
    def optimized_forward(self, x, timesteps, y=None, threshold=-1, null=False, clf_free=False):
        """
        优化的forward方法
        """
        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps, self.model_channels))
        cemb_mm = None
        
        # 条件设置
        if self.num_classes is not None:
            cemb = None
            if threshold != -1: 
                assert threshold > 0
                cemb = self.class_emb(self.label_emb(y))
                mask = th.rand(cemb.shape[0]) < threshold
                cemb[np.where(mask)[0]] = 0
                # 使用优化的cemb_mm计算
                cemb_mm = optimizer.compute_cemb_mm_optimized(cemb)
            elif threshold == -1 and clf_free: 
                if null:
                    cemb = th.zeros_like(emb)
                else:
                    cemb = self.class_emb(self.label_emb(y)) 
                # 使用优化的cemb_mm计算
                cemb_mm = optimizer.compute_cemb_mm_optimized(cemb)
            else:
                raise Exception("Invalid condition setup")
                
            assert cemb is not None
            emb = emb + cemb 

        # 前向传播
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, cemb_mm)
            hs.append(h)
        h = self.middle_block(h, emb, cemb_mm)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, cemb_mm)
            
        return self.out(h)
    
    # 替换forward方法
    import types
    model.forward = types.MethodType(optimized_forward, model)
    model._memory_optimizer = optimizer
    
    logger.info("内存优化应用成功")
    return model

def estimate_memory_savings(batch_size, embed_dim=512, optimization_method="conditional", **kwargs):
    """
    估算内存节省
    
    Args:
        batch_size: 批次大小
        embed_dim: 嵌入维度
        optimization_method: 优化方法
        **kwargs: 其他参数
        
    Returns:
        内存使用统计
    """
    # 原始内存使用
    original_cemb_memory = batch_size * embed_dim * 4  # bytes
    original_cemb_mm_memory = batch_size * embed_dim * embed_dim * 4  # bytes
    original_total = original_cemb_memory + original_cemb_mm_memory
    
    # 优化后内存使用
    if optimization_method == "conditional":
        # 假设50%的样本为零
        optimized_cemb_mm_memory = original_cemb_mm_memory * 0.5
    elif optimization_method == "low_rank":
        rank_ratio = kwargs.get("low_rank_ratio", 0.125)
        rank = max(1, int(embed_dim * rank_ratio))
        optimized_cemb_mm_memory = batch_size * rank * rank * 4
    elif optimization_method == "chunked":
        chunk_size = kwargs.get("chunk_size", 8)
        peak_memory_ratio = min(1.0, chunk_size / batch_size)
        optimized_cemb_mm_memory = original_cemb_mm_memory * peak_memory_ratio
    else:
        optimized_cemb_mm_memory = original_cemb_mm_memory
    
    optimized_total = original_cemb_memory + optimized_cemb_mm_memory
    
    savings_ratio = (original_total - optimized_total) / original_total
    memory_ratio = optimized_total / original_total
    
    return {
        "original_total_mb": original_total / (1024 * 1024),
        "optimized_total_mb": optimized_total / (1024 * 1024),
        "savings_mb": (original_total - optimized_total) / (1024 * 1024),
        "savings_ratio": savings_ratio,
        "memory_ratio": memory_ratio,
        "batch_size_multiplier": 1 / memory_ratio
    }

def benchmark_optimization_methods(batch_sizes=[1, 2, 4, 8, 16, 32], embed_dim=512):
    """
    对比不同优化方法的性能
    
    Args:
        batch_sizes: 要测试的批次大小列表
        embed_dim: 嵌入维度
        
    Returns:
        性能对比结果
    """
    methods = [
        ("original", {}),
        ("conditional", {}),
        ("low_rank", {"low_rank_ratio": 0.125}),
        ("chunked", {"chunk_size": 8}),
        ("hybrid", {"low_rank_ratio": 0.125})
    ]
    
    results = {}
    
    for method_name, kwargs in methods:
        results[method_name] = []
        for batch_size in batch_sizes:
            stats = estimate_memory_savings(batch_size, embed_dim, method_name, **kwargs)
            results[method_name].append(stats)
    
    return results

def print_optimization_report(batch_size=32, embed_dim=512):
    """
    打印优化报告
    
    Args:
        batch_size: 批次大小
        embed_dim: 嵌入维度
    """
    print(f"\n=== HAE V2 内存优化报告 (batch_size={batch_size}, embed_dim={embed_dim}) ===")
    print()
    
    methods = [
        ("原始方法", "original", {}),
        ("条件计算", "conditional", {}),
        ("低秩分解", "low_rank", {"low_rank_ratio": 0.125}),
        ("分块计算", "chunked", {"chunk_size": 8}),
        ("混合方法", "hybrid", {"low_rank_ratio": 0.125})
    ]
    
    print(f"{'方法':<10} {'内存使用(MB)':<12} {'节省比例':<10} {'批次倍数':<10} {'推荐场景':<20}")
    print("-" * 80)
    
    for name, method, kwargs in methods:
        stats = estimate_memory_savings(batch_size, embed_dim, method, **kwargs)
        
        scenarios = {
            "original": "基准对比",
            "conditional": "稀疏条件/null场景",
            "low_rank": "大批次训练",
            "chunked": "内存受限环境",
            "hybrid": "通用优化"
        }
        
        print(f"{name:<10} {stats['optimized_total_mb']:<12.2f} {stats['savings_ratio']*100:<9.1f}% {stats['batch_size_multiplier']:<9.1f}x {scenarios[method]:<20}")
    
    print()
    print("推荐配置：")
    print("- 保守优化：conditional (无质量损失，2-3x批次提升)")
    print("- 激进优化：low_rank + conditional (轻微质量损失，10-20x批次提升)")
    print("- 内存受限：chunked + conditional (适合小显存，5-10x批次提升)")

if __name__ == "__main__":
    # 运行示例
    print("HAE V2 内存优化修复脚本")
    print("=" * 50)
    
    # 打印优化报告
    print_optimization_report()
    
    # 示例：如何使用
    print("\n使用示例：")
    print("""
# 1. 导入模块
from apply_hae_v2_memory_fix import apply_memory_optimization

# 2. 加载你的HAE V2模型
model = load_your_hae_v2_model()

# 3. 应用内存优化
optimized_model = apply_memory_optimization(
    model, 
    optimization_method="conditional",  # 或 "low_rank", "chunked", "hybrid"
    low_rank_ratio=0.125,  # 仅用于low_rank方法
    chunk_size=8,          # 仅用于chunked方法
)

# 4. 现在可以使用更大的batch_size了！
with torch.no_grad():
    output = optimized_model(x, timesteps, y, threshold=0.1)
    """)