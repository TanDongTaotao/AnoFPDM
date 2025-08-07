#!/usr/bin/env python3
"""
计算原始FPDM UNet和HAE UNet的参数量对比
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from guided_diffusion.unet_v2 import UNetModel  # 原始FPDM UNet
from guided_diffusion.unet_hae import HAEUNetModel  # HAE UNet
from guided_diffusion.unet_hae_lite import HAEUNetModelLite  # HAE UNet Lite

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
    # 标准配置，基于BraTS数据集
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
        'clf_free': True,
    }
    
    # 处理attention_resolutions
    attention_resolutions = []
    for res in config['attention_resolutions'].split(","):
        attention_resolutions.append(config['image_size'] // int(res))
    config['attention_resolutions'] = attention_resolutions
    
    return config

def test_model_functionality(model, model_name):
    """测试模型功能"""
    print(f"\n测试 {model_name} 功能...")
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    try:
        with torch.no_grad():
            if 'HAE' in model_name:
                output = model(x, timesteps, clf_free=True)
            else:
                output = model(x, timesteps, clf_free=True)
        
        print(f"✅ {model_name} 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 前向传播失败: {e}")
        return False

def main():
    print("=" * 60)
    print("原始FPDM UNet vs HAE UNet 参数量对比分析")
    print("=" * 60)
    
    # 获取模型配置
    config = create_model_configs()
    print(f"\n模型配置:")
    print(f"  图像尺寸: {config['image_size']}x{config['image_size']}")
    print(f"  输入通道: {config['in_channels']}")
    print(f"  模型通道: {config['model_channels']}")
    print(f"  注意力分辨率: {config['attention_resolutions']}")
    print(f"  通道倍数: {config['channel_mult']}")
    
    models = {}
    
    try:
        # 1. 原始FPDM UNet (unet_v2.py)
        print(f"\n创建原始FPDM UNet...")
        original_config = config.copy()
        models['Original FPDM UNet'] = UNetModel(**original_config)
        
        # 2. HAE UNet
        print(f"创建HAE UNet...")
        hae_config = config.copy()
        hae_config['use_hae'] = True
        models['HAE UNet'] = HAEUNetModel(**hae_config)
        
        # 3. HAE UNet Lite
        print(f"创建HAE UNet Lite...")
        hae_lite_config = config.copy()
        hae_lite_config['use_hae'] = True
        models['HAE UNet Lite'] = HAEUNetModelLite(**hae_lite_config)
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    print(f"\n" + "=" * 60)
    print("参数量统计")
    print("=" * 60)
    
    results = {}
    
    for name, model in models.items():
        try:
            param_count = count_parameters(model)
            results[name] = param_count
            
            print(f"\n{name}:")
            print(f"  总参数量: {format_params(param_count)} ({param_count:,})")
            
            # 测试模型功能
            test_model_functionality(model, name)
            
        except Exception as e:
            print(f"❌ {name} 参数计算失败: {e}")
    
    # 对比分析
    if len(results) >= 2:
        print(f"\n" + "=" * 60)
        print("对比分析")
        print("=" * 60)
        
        original_params = results.get('Original FPDM UNet', 0)
        hae_params = results.get('HAE UNet', 0)
        hae_lite_params = results.get('HAE UNet Lite', 0)
        
        if original_params > 0:
            print(f"\n📊 参数量对比 (以原始FPDM UNet为基准):")
            print(f"  原始FPDM UNet:     {format_params(original_params)} (基准)")
            
            if hae_params > 0:
                hae_ratio = hae_params / original_params
                hae_diff = hae_params - original_params
                print(f"  HAE UNet:          {format_params(hae_params)} ({hae_ratio:.2f}x, {'+' if hae_diff > 0 else ''}{format_params(abs(hae_diff))})")
            
            if hae_lite_params > 0:
                lite_ratio = hae_lite_params / original_params
                lite_diff = hae_lite_params - original_params
                print(f"  HAE UNet Lite:     {format_params(hae_lite_params)} ({lite_ratio:.2f}x, {'+' if lite_diff > 0 else ''}{format_params(abs(lite_diff))})")
        
        if hae_params > 0 and hae_lite_params > 0:
            lite_vs_hae_ratio = hae_lite_params / hae_params
            lite_vs_hae_diff = hae_lite_params - hae_params
            print(f"\n🔍 HAE Lite vs HAE 对比:")
            print(f"  参数减少: {format_params(abs(lite_vs_hae_diff))} ({(1-lite_vs_hae_ratio)*100:.1f}% 减少)")
            print(f"  压缩比: {1/lite_vs_hae_ratio:.2f}x")
    
    print(f"\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    print(f"\n✅ 分析完成!")
    print(f"\n📋 关键发现:")
    if original_params > 0:
        print(f"  • 原始FPDM UNet是基础模型，参数量为 {format_params(original_params)}")
    if hae_params > 0 and original_params > 0:
        increase_pct = ((hae_params - original_params) / original_params) * 100
        print(f"  • HAE UNet通过添加Transformer增加了 {increase_pct:.1f}% 的参数")
    if hae_lite_params > 0 and hae_params > 0:
        reduction_pct = ((hae_params - hae_lite_params) / hae_params) * 100
        print(f"  • HAE UNet Lite通过共享投影层减少了 {reduction_pct:.1f}% 的HAE参数")
    
    print(f"\n🎯 建议:")
    print(f"  • 如需基础扩散模型功能，使用原始FPDM UNet")
    print(f"  • 如需更强特征提取能力，使用HAE UNet")
    print(f"  • 如需平衡性能和效率，使用HAE UNet Lite")

if __name__ == "__main__":
    main()