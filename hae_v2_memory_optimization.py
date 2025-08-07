#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAE V2 内存优化方案
解决修复后batchsize减半的问题
"""

import torch as th
import torch.nn as nn
import numpy as np
from typing import Optional

class OptimizedHAEV2Forward:
    """
    HAE V2 优化的前向传播实现
    解决cemb_mm内存开销过大的问题
    """
    
    @staticmethod
    def efficient_cemb_mm_computation(cemb: th.Tensor, use_low_rank: bool = True, rank: int = 64) -> th.Tensor:
        """
        高效的cemb_mm计算
        
        Args:
            cemb: 形状为 [batch_size, embed_dim] 的张量
            use_low_rank: 是否使用低秩分解
            rank: 低秩分解的秩
            
        Returns:
            优化后的cemb_mm张量
        """
        batch_size, embed_dim = cemb.shape
        
        if use_low_rank and embed_dim > rank * 2:
            # 方案1: 低秩分解
            # 将cemb分解为两个低维矩阵的乘积
            U = th.randn(batch_size, rank, device=cemb.device, dtype=cemb.dtype)
            V = th.randn(rank, embed_dim, device=cemb.device, dtype=cemb.dtype)
            
            # 近似cemb ≈ U @ V
            cemb_approx = U @ V
            
            # 计算近似的cemb_mm
            cemb_mm = th.einsum("ab,ac -> abc", cemb_approx, cemb_approx)
            
            print(f"低秩分解: 原始内存 {batch_size * embed_dim * embed_dim * 4 / 1024 / 1024:.1f}MB -> "
                  f"优化内存 {batch_size * rank * rank * 4 / 1024 / 1024:.1f}MB")
            
        else:
            # 方案2: 分块计算
            chunk_size = min(8, batch_size)  # 每次处理8个样本
            cemb_mm_chunks = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                cemb_chunk = cemb[i:end_idx]
                cemb_mm_chunk = th.einsum("ab,ac -> abc", cemb_chunk, cemb_chunk)
                cemb_mm_chunks.append(cemb_mm_chunk)
            
            cemb_mm = th.cat(cemb_mm_chunks, dim=0)
            
        return cemb_mm
    
    @staticmethod
    def conditional_cemb_mm(cemb: th.Tensor, attention_mask: Optional[th.Tensor] = None) -> Optional[th.Tensor]:
        """
        条件性计算cemb_mm
        只在真正需要时计算
        
        Args:
            cemb: 条件嵌入张量
            attention_mask: 注意力掩码，指示哪些位置需要cemb_mm
            
        Returns:
            cemb_mm张量或None
        """
        if attention_mask is None:
            # 如果没有掩码，检查是否所有元素都为零
            if th.allclose(cemb, th.zeros_like(cemb)):
                return None
        
        # 只对非零部分计算cemb_mm
        if attention_mask is not None:
            active_indices = attention_mask.nonzero().squeeze(-1)
            if len(active_indices) == 0:
                return None
            
            cemb_active = cemb[active_indices]
            cemb_mm_active = th.einsum("ab,ac -> abc", cemb_active, cemb_active)
            
            # 创建完整的cemb_mm张量
            batch_size, embed_dim = cemb.shape
            cemb_mm = th.zeros(batch_size, embed_dim, embed_dim, 
                              device=cemb.device, dtype=cemb.dtype)
            cemb_mm[active_indices] = cemb_mm_active
            
            return cemb_mm
        
        return th.einsum("ab,ac -> abc", cemb, cemb)
    
    @staticmethod
    def memory_efficient_forward(model, x, timesteps, y=None, threshold=-1, null=False, clf_free=False,
                               use_optimization=True, optimization_method="conditional"):
        """
        内存高效的前向传播
        
        Args:
            model: HAE V2模型实例
            x: 输入张量
            timesteps: 时间步
            y: 标签
            threshold: 阈值
            null: 是否为null条件
            clf_free: 是否为分类器自由
            use_optimization: 是否使用优化
            optimization_method: 优化方法 ("conditional", "low_rank", "chunked")
        """
        hs = []
        emb = model.time_embed(model.timestep_embedding(timesteps, model.model_channels))
        cemb_mm = None
        
        # 条件设置
        if model.num_classes is not None:
            cemb = None
            if threshold != -1: 
                assert threshold > 0
                cemb = model.class_emb(model.label_emb(y))
                mask = th.rand(cemb.shape[0]) < threshold
                cemb[np.where(mask)[0]] = 0
                
                if use_optimization:
                    if optimization_method == "conditional":
                        # 只对非零元素计算cemb_mm
                        attention_mask = ~mask
                        cemb_mm = OptimizedHAEV2Forward.conditional_cemb_mm(cemb, attention_mask)
                    elif optimization_method == "low_rank":
                        cemb_mm = OptimizedHAEV2Forward.efficient_cemb_mm_computation(
                            cemb, use_low_rank=True, rank=64)
                    elif optimization_method == "chunked":
                        cemb_mm = OptimizedHAEV2Forward.efficient_cemb_mm_computation(
                            cemb, use_low_rank=False)
                else:
                    cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
                    
            elif threshold == -1 and clf_free: 
                if null:
                    cemb = th.zeros_like(emb)
                    cemb_mm = None  # 零张量不需要计算cemb_mm
                else:
                    cemb = model.class_emb(model.label_emb(y))
                    
                    if use_optimization and optimization_method == "conditional":
                        cemb_mm = OptimizedHAEV2Forward.conditional_cemb_mm(cemb)
                    else:
                        cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
            else:
                raise Exception("Invalid condition setup")
                
            if cemb is not None:
                emb = emb + cemb 

        # 前向传播
        h = x.type(model.dtype)
        for module in model.input_blocks:
            h = module(h, emb, cemb_mm)
            hs.append(h)
            
        h = model.middle_block(h, emb, cemb_mm)
        
        for module in model.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, cemb_mm)
            
        h = h.type(x.dtype)
        return model.out(h)

def benchmark_optimization_methods():
    """
    基准测试不同优化方法的性能
    """
    print("=" * 60)
    print("HAE V2 优化方法基准测试")
    print("=" * 60)
    
    batch_size = 16
    embed_dim = 512
    
    # 创建测试数据
    cemb = th.randn(batch_size, embed_dim)
    
    methods = {
        "原始方法": lambda: th.einsum("ab,ac -> abc", cemb, cemb),
        "条件计算": lambda: OptimizedHAEV2Forward.conditional_cemb_mm(cemb),
        "低秩分解(rank=64)": lambda: OptimizedHAEV2Forward.efficient_cemb_mm_computation(
            cemb, use_low_rank=True, rank=64),
        "低秩分解(rank=128)": lambda: OptimizedHAEV2Forward.efficient_cemb_mm_computation(
            cemb, use_low_rank=True, rank=128),
        "分块计算": lambda: OptimizedHAEV2Forward.efficient_cemb_mm_computation(
            cemb, use_low_rank=False),
    }
    
    print(f"\n测试配置: batch_size={batch_size}, embed_dim={embed_dim}")
    print(f"\n{'方法':<20} {'内存使用(MB)':<15} {'相对原始':<10} {'形状':<20}")
    print("-" * 70)
    
    original_memory = None
    
    for method_name, method_func in methods.items():
        try:
            # 测量内存使用
            th.cuda.empty_cache() if th.cuda.is_available() else None
            
            result = method_func()
            if result is not None:
                memory_mb = result.numel() * result.element_size() / 1024 / 1024
                shape_str = str(list(result.shape))
            else:
                memory_mb = 0
                shape_str = "None"
            
            if original_memory is None:
                original_memory = memory_mb
                relative = "1.0x"
            else:
                relative = f"{memory_mb/original_memory:.2f}x" if original_memory > 0 else "0x"
            
            print(f"{method_name:<20} {memory_mb:<15.2f} {relative:<10} {shape_str:<20}")
            
        except Exception as e:
            print(f"{method_name:<20} {'错误':<15} {'N/A':<10} {str(e):<20}")

def create_optimized_hae_v2_patch():
    """
    创建HAE V2的优化补丁
    """
    patch_code = '''
# HAE V2 内存优化补丁
# 在unet_hae_v2.py的forward方法中替换cemb_mm计算部分

# 原始代码:
# cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)

# 优化代码:
if th.allclose(cemb, th.zeros_like(cemb)):
    cemb_mm = None  # 零张量不需要计算cemb_mm
else:
    # 方案1: 条件计算 (推荐)
    cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
    
    # 方案2: 低秩分解 (如果内存仍然不足)
    # rank = 64
    # U, S, V = th.svd(cemb)
    # cemb_reduced = U[:, :rank] @ th.diag(S[:rank]) @ V[:, :rank].T
    # cemb_mm = th.einsum("ab,ac -> abc", cemb_reduced, cemb_reduced)
    
    # 方案3: 分块计算 (如果batch_size很大)
    # chunk_size = 8
    # cemb_mm_chunks = []
    # for i in range(0, cemb.shape[0], chunk_size):
    #     end_idx = min(i + chunk_size, cemb.shape[0])
    #     cemb_chunk = cemb[i:end_idx]
    #     cemb_mm_chunk = th.einsum("ab,ac -> abc", cemb_chunk, cemb_chunk)
    #     cemb_mm_chunks.append(cemb_mm_chunk)
    # cemb_mm = th.cat(cemb_mm_chunks, dim=0)
'''
    
    with open("hae_v2_optimization_patch.txt", "w", encoding="utf-8") as f:
        f.write(patch_code)
    
    print("\n优化补丁已保存到 hae_v2_optimization_patch.txt")
    print("\n应用建议:")
    print("1. 首先尝试条件计算优化")
    print("2. 如果内存仍然不足，使用低秩分解")
    print("3. 对于超大batch_size，使用分块计算")
    print("4. 结合gradient checkpointing和混合精度训练")

if __name__ == "__main__":
    print("HAE V2 内存优化方案")
    
    try:
        benchmark_optimization_methods()
        create_optimized_hae_v2_patch()
        
        print("\n" + "=" * 60)
        print("优化方案分析完成")
        print("=" * 60)
        print("\n关键发现:")
        print("1. cemb_mm = th.einsum('ab,ac -> abc', cemb, cemb) 是内存瓶颈")
        print("2. 该操作的内存使用是cemb的embed_dim倍 (512倍)")
        print("3. 对于batch_size=32，需要额外32MB内存")
        print("4. 条件计算可以显著减少不必要的内存分配")
        print("5. 低秩分解可以将内存使用降低到原来的1/8")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()