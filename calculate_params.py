import torch
import sys
import os
sys.path.append(os.path.realpath("./"))

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_transformer_params(model):
    """分析Transformer相关参数"""
    transformer_params = 0
    projection_params = 0
    attention_params = 0
    mlp_params = 0
    
    for name, module in model.named_modules():
        if 'MultiScaleSparseTransformerBlock' in str(type(module)):
            print(f"Found Transformer block: {name}")
            block_params = sum(p.numel() for p in module.parameters())
            transformer_params += block_params
            print(f"  Block parameters: {block_params:,}")
            
            # 分析子模块
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"    {sub_name}: {sub_params:,}")
                
                if 'patch_projections' in sub_name:
                    projection_params += sub_params
                elif 'multihead_attn' in sub_name:
                    attention_params += sub_params
                elif 'mlp' in sub_name:
                    mlp_params += sub_params
    
    return transformer_params, projection_params, attention_params, mlp_params

def main():
    # 使用训练脚本中的默认参数
    args = {
        'image_size': 256,
        'in_channels': 1,
        'num_channels': 128,
        'num_res_blocks': 2,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'num_head_channels': -1,
        'attention_resolutions': '16,8',
        'channel_mult': '1,1,2,2,4,4',
        'dropout': 0.0,
        'class_cond': False,
        'num_classes': None,
        'clf_free': True,
        'use_checkpoint': False,
        'use_scale_shift_norm': True,
        'resblock_updown': True,
        'use_fp16': False,
        'use_new_attention_order': True,
        'learn_sigma': False,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'unet_ver': 'hae',  # 使用HAE版本
        'use_hae': True  # 启用HAE
    }
    
    print("Creating HAE UNet model...")
    model, diffusion = create_model_and_diffusion(**args)
    
    # 计算总参数量
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # 分析Transformer参数
    print("\nAnalyzing Transformer components...")
    transformer_params, projection_params, attention_params, mlp_params = analyze_transformer_params(model)
    
    print(f"\nTransformer Summary:")
    print(f"Total Transformer parameters: {transformer_params:,} ({transformer_params/1e6:.1f}M)")
    print(f"Projection layers: {projection_params:,} ({projection_params/1e6:.1f}M)")
    print(f"Attention layers: {attention_params:,} ({attention_params/1e6:.1f}M)")
    print(f"MLP layers: {mlp_params:,} ({mlp_params/1e6:.1f}M)")
    print(f"Transformer ratio: {transformer_params/total_params*100:.1f}%")
    
    # 计算投影层的具体参数
    print("\nProjection layer details:")
    patch_sizes = [4, 8]  # 默认patch_sizes[1:]
    channels = [128, 128, 256, 256, 512, 512]  # 根据channel_mult计算
    
    total_projection_calc = 0
    for i, patch_size in enumerate(patch_sizes):
        for ch in channels:
            input_dim = ch * patch_size * patch_size
            output_dim = ch
            params = input_dim * output_dim + output_dim  # 权重 + 偏置
            total_projection_calc += params
            print(f"  Patch size {patch_size}, channels {ch}: {input_dim} -> {output_dim} = {params:,} params")
    
    print(f"\nCalculated projection parameters: {total_projection_calc:,} ({total_projection_calc/1e6:.1f}M)")

if __name__ == "__main__":
    main()