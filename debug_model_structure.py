#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试HAE模型结构
"""

import torch
import torch.nn as nn
from guided_diffusion.unet_hae_lite import HAEUNetModelLite
from guided_diffusion.unet_hae_v2 import HAEUNetModelV2
from guided_diffusion.unet_hae_lite import HybridCNNTransformerBlockLite
from guided_diffusion.unet_hae_v2 import HybridCNNTransformerBlockV2

def debug_model_structure():
    print("=" * 60)
    print("调试HAE模型结构")
    print("=" * 60)
    
    # 模型配置
    model_config = {
        'image_size': 64,
        'in_channels': 3,
        'model_channels': 128,
        'out_channels': 3,
        'num_res_blocks': 2,
        'attention_resolutions': (16, 32),
        'dropout': 0.0,
        'channel_mult': (1, 2, 4, 8),
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': 4,
        'num_head_channels': 32,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'clf_free': True,
        'use_hae': True,
    }
    
    # 测试HAE UNet Lite
    print("\n🔍 HAE UNet Lite 结构分析:")
    try:
        model_lite = HAEUNetModelLite(**model_config)
        
        print(f"\n📥 编码器 (input_blocks): {len(model_lite.input_blocks)} 个块")
        for i, block in enumerate(model_lite.input_blocks):
            print(f"  Block {i}: {len(block)} 层")
            for j, layer in enumerate(block):
                print(f"    Layer {j}: {type(layer).__name__}")
        
        print(f"\n🔄 中间块 (middle_block): {len(model_lite.middle_block)} 层")
        for i, layer in enumerate(model_lite.middle_block):
            print(f"  Layer {i}: {type(layer).__name__}")
        
        print(f"\n📤 解码器 (output_blocks): {len(model_lite.output_blocks)} 个块")
        for i, block in enumerate(model_lite.output_blocks):
            print(f"  Block {i}: {len(block)} 层")
            for j, layer in enumerate(block):
                print(f"    Layer {j}: {type(layer).__name__}")
                if isinstance(layer, HybridCNNTransformerBlockLite):
                    print(f"      ✅ 发现异构块!")
        
    except Exception as e:
        print(f"❌ HAE UNet Lite 创建失败: {e}")
    
    # 测试HAE UNet V2
    print("\n🔍 HAE UNet V2 结构分析:")
    try:
        model_v2 = HAEUNetModelV2(**{**model_config, 'bottleneck_ratio': 0.25})
        
        print(f"\n📥 编码器 (input_blocks): {len(model_v2.input_blocks)} 个块")
        for i, block in enumerate(model_v2.input_blocks):
            print(f"  Block {i}: {len(block)} 层")
            for j, layer in enumerate(block):
                print(f"    Layer {j}: {type(layer).__name__}")
        
        print(f"\n🔄 中间块 (middle_block): {len(model_v2.middle_block)} 层")
        for i, layer in enumerate(model_v2.middle_block):
            print(f"  Layer {i}: {type(layer).__name__}")
        
        print(f"\n📤 解码器 (output_blocks): {len(model_v2.output_blocks)} 个块")
        for i, block in enumerate(model_v2.output_blocks):
            print(f"  Block {i}: {len(block)} 层")
            for j, layer in enumerate(block):
                print(f"    Layer {j}: {type(layer).__name__}")
                if isinstance(layer, HybridCNNTransformerBlockV2):
                    print(f"      ✅ 发现异构块!")
        
    except Exception as e:
        print(f"❌ HAE UNet V2 创建失败: {e}")

if __name__ == "__main__":
    debug_model_structure()