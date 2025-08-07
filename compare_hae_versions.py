#!/usr/bin/env python3
"""
比较HAE UNet Lite和HAE UNet V2（带MLP瓶颈）的参数量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from guided_diffusion.unet_hae_lite import HAEUNetModelLite
from guided_diffusion.unet_hae_v2 import HAEUNetModelV2

def count_parameters(model):
    """计算模型的总参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num_params):
    """格式化参数数量显示"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.1f}K"
    else:
        return str(num_params)

def create_model_configs():
    """创建模型配置"""
    config = {
        'image_size': 128,
        'in_channels': 4,
        'model_channels': 128,
        'out_channels': 4,
        'num_res_blocks': 2,
        'attention_resolutions': "32,16,8",
        'dropout': 0.1,
        'channel_mult': (1, 2, 4, 8),
        'conv_resample': True,
        'dims': 2,
        'num_classes': None,
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': 8,
        'num_head_channels': -1,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': False,
        'resblock_updown': False,
        'use_new_attention_order': False,
        'use_hae': True,
        'clf_free': True,
    }
    
    attention_resolutions = []
    for res in config['attention_resolutions'].split(","):
        attention_resolutions.append(config['image_size'] // int(res))
    config['attention_resolutions'] = attention_resolutions
    
    return config

def main():
    print("=" * 60)
    print("HAE UNet Lite vs V2 (MLP Bottleneck) 参数对比")
    print("=" * 60)
    
    config = create_model_configs()
    
    # 1. HAE UNet Lite
    print("\n创建 HAE UNet Lite...")
    hae_lite_model = HAEUNetModelLite(**config)
    lite_params = count_parameters(hae_lite_model)
    print(f"  HAE UNet Lite 参数量: {format_params(lite_params)} ({lite_params:,})")

    # 2. HAE UNet V2 (MLP Bottleneck)
    print("\n创建 HAE UNet V2 (MLP Bottleneck)...")
    hae_v2_model = HAEUNetModelV2(**config)
    v2_params = count_parameters(hae_v2_model)
    print(f"  HAE UNet V2 参数量: {format_params(v2_params)} ({v2_params:,})")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    
    reduction = lite_params - v2_params
    reduction_percent = (reduction / lite_params) * 100
    
    print(f"\n📊 参数对比:")
    print(f"  HAE UNet Lite: {format_params(lite_params)}")
    print(f"  HAE UNet V2:   {format_params(v2_params)}")
    print(f"  参数减少量:    {format_params(reduction)} ({reduction_percent:.2f}%)")
    
    print(f"\n✅ 分析完成! 引入MLP瓶颈结构成功将参数从 {format_params(lite_params)} 减少到 {format_params(v2_params)}.")

if __name__ == "__main__":
    main()