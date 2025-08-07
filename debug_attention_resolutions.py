#!/usr/bin/env python3
"""
调试HAE模型的attention_resolutions问题
"""

import torch
import torch.nn as nn
from guided_diffusion.unet_hae_lite import HAEUNetModelLite
from guided_diffusion.unet_hae_v2 import HAEUNetModelV2
from guided_diffusion.unet_hae import HAEUNetModel

def debug_model_construction(model_class, model_name):
    """
    调试模型构建过程中的分辨率计算
    """
    print(f"\n🔍 调试 {model_name} 的分辨率计算...")
    
    # 模型配置
    config = {
        'image_size': 64,
        'in_channels': 3,
        'model_channels': 128,
        'out_channels': 3,
        'num_res_blocks': 2,
        'attention_resolutions': (1, 2, 4, 8),
        'dropout': 0.0,
        'channel_mult': (1, 2, 4, 8),
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': 4,
        'num_head_channels': 32,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'use_new_attention_order': False,
        'clf_free': True,
        'use_hae': True,
    }
    
    # 添加V2特有的参数
    if 'V2' in model_name:
        config['bottleneck_ratio'] = 0.25
    
    print(f"   配置的attention_resolutions: {config['attention_resolutions']}")
    print(f"   配置的channel_mult: {config['channel_mult']}")
    print(f"   配置的num_res_blocks: {config['num_res_blocks']}")
    
    # 手动计算分辨率变化
    print("\n   📐 手动计算分辨率变化:")
    image_size = config['image_size']
    channel_mult = config['channel_mult']
    num_res_blocks = config['num_res_blocks']
    
    ds = 1  # 当前下采样倍数
    current_resolution = image_size
    
    print(f"   初始分辨率: {current_resolution}, ds={ds}")
    
    # 编码器分辨率计算
    for level, mult in enumerate(channel_mult):
        print(f"\n   Level {level} (mult={mult}):")
        
        # 每个level有num_res_blocks个ResBlock
        for block_idx in range(num_res_blocks):
            print(f"     ResBlock {block_idx}: 分辨率={current_resolution}, ds={ds}")
            if ds in config['attention_resolutions']:
                print(f"       -> 应该添加AttentionBlock (ds={ds} in {config['attention_resolutions']})")
            else:
                print(f"       -> 不添加AttentionBlock (ds={ds} not in {config['attention_resolutions']})")
        
        # 除了最后一个level，都有下采样
        if level != len(channel_mult) - 1:
            ds *= 2
            current_resolution //= 2
            print(f"     下采样后: 分辨率={current_resolution}, ds={ds}")
    
    print(f"\n   中间块: 分辨率={current_resolution}, ds={ds}")
    print(f"   -> 中间块总是有AttentionBlock")
    
    # 创建模型并检查实际结构
    print(f"\n   🏗️  创建 {model_name} 模型...")
    try:
        model = model_class(**config)
        print(f"   ✅ 模型创建成功")
        
        # 检查编码器中的注意力块
        encoder_attention_count = 0
        print(f"\n   📥 检查编码器 (input_blocks):")
        for i, block in enumerate(model.input_blocks):
            has_attention = False
            for layer in block:
                if hasattr(layer, '__class__') and 'AttentionBlock' in layer.__class__.__name__:
                    has_attention = True
                    encoder_attention_count += 1
                    break
            print(f"     Block {i}: {'有AttentionBlock' if has_attention else '无AttentionBlock'}")
        
        # 检查中间块
        middle_attention_count = 0
        print(f"\n   🔄 检查中间块 (middle_block):")
        for i, layer in enumerate(model.middle_block):
            if hasattr(layer, '__class__') and 'AttentionBlock' in layer.__class__.__name__:
                middle_attention_count += 1
                print(f"     Layer {i}: AttentionBlock")
            else:
                print(f"     Layer {i}: {layer.__class__.__name__}")
        
        # 检查解码器中的注意力块
        decoder_attention_count = 0
        print(f"\n   📤 检查解码器 (output_blocks):")
        for i, block in enumerate(model.output_blocks):
            has_attention = False
            for layer in block:
                if hasattr(layer, '__class__') and 'AttentionBlock' in layer.__class__.__name__:
                    has_attention = True
                    decoder_attention_count += 1
                    break
            print(f"     Block {i}: {'有AttentionBlock' if has_attention else '无AttentionBlock'}")
        
        print(f"\n   📊 统计结果:")
        print(f"     编码器AttentionBlock数量: {encoder_attention_count}")
        print(f"     中间块AttentionBlock数量: {middle_attention_count}")
        print(f"     解码器AttentionBlock数量: {decoder_attention_count}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return False

def main():
    print("🔍 调试HAE模型的attention_resolutions问题")
    print("="*60)
    
    models_to_test = [
        (HAEUNetModel, "HAE UNet (原版)"),
        (HAEUNetModelLite, "HAE UNet Lite"),
        (HAEUNetModelV2, "HAE UNet V2"),
    ]
    
    for model_class, model_name in models_to_test:
        debug_model_construction(model_class, model_name)
        print("="*60)

if __name__ == "__main__":
    main()