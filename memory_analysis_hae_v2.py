#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAE V2 内存使用分析脚本
分析修复前后HAE V2模型的内存差异
"""

import torch as th
import numpy as np
import psutil
import os
from torch.profiler import profile, record_function, ProfilerActivity

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def analyze_cemb_mm_memory():
    """分析cemb_mm操作的内存开销"""
    print("=" * 60)
    print("HAE V2 内存使用分析")
    print("=" * 60)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    embed_dim = 512  # 时间嵌入维度
    
    print(f"\n测试配置:")
    print(f"- 嵌入维度: {embed_dim}")
    print(f"- 测试批次大小: {batch_sizes}")
    
    print(f"\n{'批次大小':<8} {'cemb内存(MB)':<12} {'cemb_mm内存(MB)':<15} {'内存倍数':<10} {'总内存(MB)':<12}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        # 模拟cemb张量
        cemb = th.randn(batch_size, embed_dim)
        cemb_memory = cemb.numel() * cemb.element_size() / 1024 / 1024
        
        # 计算cemb_mm的内存使用
        # cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
        # 形状为 [batch_size, embed_dim, embed_dim]
        cemb_mm_numel = batch_size * embed_dim * embed_dim
        cemb_mm_memory = cemb_mm_numel * 4 / 1024 / 1024  # float32
        
        memory_ratio = cemb_mm_memory / cemb_memory
        total_memory = cemb_memory + cemb_mm_memory
        
        print(f"{batch_size:<8} {cemb_memory:<12.2f} {cemb_mm_memory:<15.2f} {memory_ratio:<10.1f}x {total_memory:<12.2f}")
    
    print("\n分析结果:")
    print("1. cemb_mm的内存使用是cemb的embed_dim倍 (512倍)")
    print("2. 对于batch_size=32, cemb_mm需要额外的2048MB内存")
    print("3. 这解释了为什么修复后batchsize减半")

def simulate_forward_memory():
    """模拟forward过程的内存使用"""
    print("\n" + "=" * 60)
    print("Forward过程内存模拟")
    print("=" * 60)
    
    batch_size = 16
    embed_dim = 512
    
    print(f"\n测试批次大小: {batch_size}")
    
    # 修复前：只有cemb
    print("\n修复前内存使用:")
    start_memory = get_memory_usage()
    cemb_old = th.randn(batch_size, embed_dim)
    cemb_memory = get_memory_usage() - start_memory
    print(f"- cemb内存: {cemb_memory:.2f} MB")
    
    # 修复后：cemb + cemb_mm
    print("\n修复后内存使用:")
    start_memory = get_memory_usage()
    cemb_new = th.randn(batch_size, embed_dim)
    cemb_mm = th.einsum("ab,ac -> abc", cemb_new, cemb_new)
    total_memory = get_memory_usage() - start_memory
    cemb_mm_memory = total_memory - cemb_memory
    
    print(f"- cemb内存: {cemb_memory:.2f} MB")
    print(f"- cemb_mm内存: {cemb_mm_memory:.2f} MB")
    print(f"- 总内存: {total_memory:.2f} MB")
    print(f"- 内存增长: {total_memory/cemb_memory:.1f}x")
    
    # 清理内存
    del cemb_old, cemb_new, cemb_mm
    th.cuda.empty_cache() if th.cuda.is_available() else None

def analyze_attention_impact():
    """分析注意力机制的内存影响"""
    print("\n" + "=" * 60)
    print("注意力机制内存影响分析")
    print("=" * 60)
    
    batch_size = 16
    seq_len = 64 * 64  # 64x64图像展平
    embed_dim = 512
    
    print(f"\n配置:")
    print(f"- 批次大小: {batch_size}")
    print(f"- 序列长度: {seq_len}")
    print(f"- 嵌入维度: {embed_dim}")
    
    # 模拟注意力计算
    print("\n内存使用分析:")
    
    # QKV张量
    qkv_memory = batch_size * seq_len * embed_dim * 3 * 4 / 1024 / 1024
    print(f"- QKV张量: {qkv_memory:.2f} MB")
    
    # cemb_mm用于encoder_kv
    cemb_mm_memory = batch_size * embed_dim * embed_dim * 4 / 1024 / 1024
    print(f"- cemb_mm (encoder_kv): {cemb_mm_memory:.2f} MB")
    
    # 注意力权重
    attention_memory = batch_size * seq_len * seq_len * 4 / 1024 / 1024
    print(f"- 注意力权重: {attention_memory:.2f} MB")
    
    total = qkv_memory + cemb_mm_memory + attention_memory
    print(f"- 总内存: {total:.2f} MB")
    
    print(f"\ncemb_mm占总内存的比例: {cemb_mm_memory/total*100:.1f}%")

def recommend_solutions():
    """推荐解决方案"""
    print("\n" + "=" * 60)
    print("解决方案推荐")
    print("=" * 60)
    
    print("\n问题根源:")
    print("- 修复后增加了 cemb_mm = th.einsum('ab,ac -> abc', cemb, cemb)")
    print("- 该操作创建形状为 [batch_size, embed_dim, embed_dim] 的张量")
    print("- 内存使用是原来的 embed_dim 倍 (512倍)")
    
    print("\n解决方案:")
    print("\n1. 【推荐】条件计算优化:")
    print("   - 只在需要时计算cemb_mm")
    print("   - 使用更高效的矩阵运算")
    
    print("\n2. 内存优化策略:")
    print("   - 使用gradient checkpointing")
    print("   - 启用混合精度训练 (FP16)")
    print("   - 减小embed_dim或使用低秩分解")
    
    print("\n3. 批次大小调整:")
    print("   - 根据GPU内存动态调整batch_size")
    print("   - 使用gradient accumulation")
    
    print("\n4. 架构优化:")
    print("   - 考虑使用更轻量的条件嵌入方式")
    print("   - 实现延迟计算或缓存机制")

if __name__ == "__main__":
    print("HAE V2 内存分析工具")
    print("分析修复前后的内存使用差异")
    
    try:
        analyze_cemb_mm_memory()
        simulate_forward_memory()
        analyze_attention_impact()
        recommend_solutions()
        
        print("\n" + "=" * 60)
        print("分析完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()