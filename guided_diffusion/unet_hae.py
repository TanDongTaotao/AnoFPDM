from abc import abstractmethod
from typing import List
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

# 复用原有的基础模块
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cemb_mm=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, cemb_mm)
            elif isinstance(layer, HybridCNNTransformerBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cemb_mm=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if cemb_mm is not None:
            cemb_mm_expand = self.encoder_kv(cemb_mm)
            h = self.attention(qkv, cemb_mm_expand)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


# 新增：多尺度稀疏Transformer块（MSTB）
class MultiScaleSparseTransformerBlock(nn.Module):
    """
    Multi-Scale Sparse Transformer Block (MSTB)
    实现论文中的多尺度稀疏Transformer，包含局部信息和区域信息处理
    """
    
    def __init__(self, channels, num_heads=8, dropout=0.1, patch_sizes=[1, 4, 8]):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        
        # 局部信息处理
        self.local_norm = nn.LayerNorm(channels)
        self.local_pos_embed = nn.Parameter(th.randn(1, channels) * 0.02)
        
        # 区域信息处理 - 多尺度
        self.regional_norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in patch_sizes[1:]
        ])
        self.regional_pos_embeds = nn.ParameterList([
            nn.Parameter(th.randn(1, channels) * 0.02) for _ in patch_sizes[1:]
        ])
        
        # 稀疏多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.norm_out = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
    def create_sparse_mask(self, seq_len, sparsity_ratio=0.9):
        """
        创建稀疏注意力掩码，减少90%的计算量
        """
        mask = th.ones(seq_len, seq_len, dtype=th.bool)
        # 保留对角线和部分随机位置
        keep_indices = max(1, int(seq_len * (1 - sparsity_ratio)))
        for i in range(seq_len):
            # 保留对角线附近
            start = max(0, i - keep_indices // 2)
            end = min(seq_len, i + keep_indices // 2 + 1)
            mask[i, start:end] = False
            
            # 随机保留一些远距离连接
            if keep_indices > 4:  # 确保有足够的索引可以选择
                num_random = max(1, keep_indices // 4)
                random_indices = th.randperm(seq_len)[:num_random]
                mask[i, random_indices] = False
                
        return mask
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 局部信息：直接展平
        x_flat = x.view(B, C, -1).transpose(1, 2)  # B, HW, C
        # 确保位置嵌入维度匹配
        pos_embed = self.local_pos_embed.expand(B, -1)  # B, C
        # 确保x_flat的最后一个维度与pos_embed匹配
        if x_flat.size(-1) != pos_embed.size(-1):
            # 使用线性层来调整维度
            if not hasattr(self, 'local_dim_adjust'):
                setattr(self, 'local_dim_adjust', nn.Linear(x_flat.size(-1), pos_embed.size(-1)).to(x_flat.device))
            local_dim_adjust_layer = getattr(self, 'local_dim_adjust')
            x_flat = local_dim_adjust_layer(x_flat)
        # 将pos_embed扩展到匹配x_flat的形状: (B, HW, C)
        seq_len = x_flat.size(1)
        pos_embed_expanded = pos_embed.unsqueeze(1).expand(B, seq_len, -1)
        local_features = self.local_norm(x_flat + pos_embed_expanded)  # B, HW, C
        
        # 区域信息：多尺度块划分
        regional_features = []
        for i, patch_size in enumerate(self.patch_sizes[1:]):
            if H % patch_size == 0 and W % patch_size == 0:
                # 重塑为块
                x_patches = x.view(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
                x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                x_patches = x_patches.view(B, (H//patch_size)*(W//patch_size), C*patch_size*patch_size)
                
                # 空间缩减到正确的通道数
                x_patches = F.adaptive_avg_pool1d(x_patches.transpose(1, 2), C).transpose(1, 2)
                # 确保区域位置嵌入维度匹配
                regional_pos = self.regional_pos_embeds[i].expand(B, -1)  # B, C
                
                # 确保x_patches的最后一个维度与regional_pos匹配
                if x_patches.size(-1) != regional_pos.size(-1):
                    # 使用线性层来调整维度而不是adaptive_avg_pool1d
                    x_patches = x_patches.view(x_patches.size(0), x_patches.size(1), -1)
                    if not hasattr(self, f'dim_adjust_{i}'):
                        setattr(self, f'dim_adjust_{i}', nn.Linear(x_patches.size(-1), regional_pos.size(-1)).to(x_patches.device))
                    dim_adjust_layer = getattr(self, f'dim_adjust_{i}')
                    x_patches = dim_adjust_layer(x_patches)
                
                # 将regional_pos扩展到匹配x_patches的形状: (B, num_patches, C)
                num_patches = x_patches.size(1)
                regional_pos_expanded = regional_pos.unsqueeze(1).expand(B, num_patches, -1)
                x_patches = self.regional_norms[i](x_patches + regional_pos_expanded)
                regional_features.append(x_patches)
        
        # 合并所有特征
        if regional_features:
            all_features = th.cat([local_features] + regional_features, dim=1)
        else:
            all_features = local_features
            
        # 创建稀疏掩码
        seq_len = all_features.size(1)
        attn_mask = self.create_sparse_mask(seq_len, sparsity_ratio=0.9).to(x.device)
        
        # 稀疏多头注意力
        attn_out, _ = self.multihead_attn(
            all_features, all_features, all_features,
            attn_mask=attn_mask
        )
        
        # 只取局部特征部分用于输出
        local_out = attn_out[:, :H*W, :]
        
        # MLP和残差连接
        local_out = local_out + self.mlp(self.norm_out(local_out))
        
        # 重塑回原始形状
        output = local_out.transpose(1, 2).view(B, C, H, W)
        
        return output + x  # 残差连接


# 新增：混合CNN-Transformer块
class HybridCNNTransformerBlock(TimestepBlock):
    """
    混合CNN-Transformer块，结合CNN的局部建模和Transformer的长距离依赖建模
    """
    
    def __init__(self, channels, emb_channels, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.use_checkpoint = use_checkpoint
        
        # CNN分支
        self.conv_branch = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(2, channels, channels, 3, padding=1),
            normalization(channels),
            nn.SiLU(),
            conv_nd(2, channels, channels, 3, padding=1)
        )
        
        # Transformer分支（使用MSTB）
        self.transformer_branch = MultiScaleSparseTransformerBlock(
            channels=channels,
            dropout=dropout
        )
        
        # 时间步嵌入
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, channels),
        )
        
        # 特征融合
        self.fusion_conv = conv_nd(2, channels * 2, channels, 1)
        
    def forward(self, x, emb):
        # 时间步嵌入
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
            
        # 添加时间步嵌入
        x_with_emb = x + emb_out
            
        # CNN分支
        conv_out = self.conv_branch(x_with_emb)
        
        # Transformer分支
        trans_out = self.transformer_branch(x_with_emb)
        
        # 确保两个分支输出维度一致
        assert conv_out.shape == trans_out.shape, f"CNN output shape {conv_out.shape} != Transformer output shape {trans_out.shape}"
        
        # 特征融合
        fused = th.cat([conv_out, trans_out], dim=1)
        output = self.fusion_conv(fused)
        
        # 残差连接
        return output + x


# 异构UNet模型
class HAEUNetModel(nn.Module):
    """
    异构自动编码器UNet模型
    编码器：传统CNN
    解码器：混合CNN-Transformer网络
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        clf_free=True,
        use_hae=True,  # Enable/disable heterogeneous autoencoder
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels 
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_hae = use_hae

        time_embed_dim = model_channels * 4
        encoder_channels = time_embed_dim

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None and clf_free:
            self.label_emb = nn.Embedding(self.num_classes, model_channels)
            self.class_emb = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        elif self.num_classes is not None and not clf_free:
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)

        # 编码器：传统CNN架构
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 中间块：保持原有设计
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 解码器：混合CNN-Transformer架构
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                
                # 在解码器中使用混合CNN-Transformer块
                if ds in attention_resolutions:
                    layers.append(
                        HybridCNNTransformerBlock(
                            ch,
                            time_embed_dim,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, threshold=-1, null=False, clf_free=False):
        """
        Apply the model to an input batch.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        cemb_mm = None
        
        # 条件设置（与原版保持一致）
        if self.num_classes is not None:
            cemb = None
            if threshold != -1: 
                assert threshold > 0
                cemb = self.class_emb(self.label_emb(y))
                mask = th.rand(cemb.shape[0])<threshold
                cemb[np.where(mask)[0]] = 0
                cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
            elif threshold == -1 and clf_free: 
                if null:
                    cemb = th.zeros_like(emb)
                else:
                    cemb = self.class_emb(self.label_emb(y)) 
                cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb) 
            else:
                raise Exception("Invalid condition setup")
                
            assert cemb is not None
            assert cemb_mm is not None
            emb = emb + cemb 

        # 编码器前向传播
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, cemb_mm)
            hs.append(h)
            
        # 中间块
        h = self.middle_block(h, emb, cemb_mm)
        
        # 解码器前向传播（使用混合CNN-Transformer）
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, cemb_mm)
            
        h = h.type(x.dtype)
        return self.out(h)