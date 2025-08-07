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
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
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
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
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
            normalization(channels),
            nn.SiLU(),
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
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        """
        if self.use_checkpoint:
            return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
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

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116dcc91b1c5b68b1e0b8e/diffusion_tf/models/unet.py#L66.
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
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cemb_mm=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, cemb_mm), self.parameters(), True)
        else:
            return self._forward(x, cemb_mm)

    def _forward(self, x, cemb_mm):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        if cemb_mm is not None:
            encoder_out = self.encoder_kv(cemb_mm)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
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
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


# 新增：瓶颈MLP结构 - HAE V2的核心优化
class BottleneckMLP(nn.Module):
    """
    瓶颈MLP结构，通过降维-升维的方式大幅减少参数量
    """
    
    def __init__(self, channels, bottleneck_ratio=0.25, dropout=0.1):
        super().__init__()
        self.channels = channels
        bottleneck_dim = max(1, int(channels * bottleneck_ratio))
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, bottleneck_dim),  # 降维
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, channels),  # 升维
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.mlp(x)


class MultiScaleSparseTransformerBlockV2(nn.Module):
    """
    Multi-Scale Sparse Transformer Block (MSTB) - V2 Version
    在Lite版本基础上，使用瓶颈MLP结构进一步减少参数量
    """
    
    def __init__(self, channels, num_heads=8, dropout=0.1, patch_sizes=[1, 4, 8], bottleneck_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        
        # 局部信息处理
        self.local_norm = nn.LayerNorm(channels)
        
        # 区域信息处理 - 多尺度，使用共享的卷积投影层（借鉴ViT）
        self.regional_norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in patch_sizes[1:]
        ])
        
        # 共享的卷积投影层 - 这是关键的参数精简部分
        # 使用单个卷积层来处理所有尺度的patch，而不是为每个尺度创建独立的Linear层
        self.shared_patch_embed = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        
        # 稀疏多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影 - 使用瓶颈MLP替代原来的大型MLP
        self.norm_out = nn.LayerNorm(channels)
        self.mlp = BottleneckMLP(channels, bottleneck_ratio=bottleneck_ratio, dropout=dropout)
        
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
        
        # 使用共享的卷积投影层处理输入特征
        x_projected = self.shared_patch_embed(x)  # B, C, H, W
        
        # 局部信息：直接展平
        x_flat = x_projected.view(B, C, -1).transpose(1, 2)  # B, HW, C
        local_features = self.local_norm(x_flat)  # B, HW, C
        
        # 区域信息：多尺度块划分，但使用相同的投影特征
        regional_features = []
        for i, patch_size in enumerate(self.patch_sizes[1:]):
            if H % patch_size == 0 and W % patch_size == 0:
                # 对投影后的特征进行块划分
                x_patches = x_projected.view(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
                x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                # 平均池化来聚合patch内的信息，而不是展平后线性投影
                x_patches = x_patches.mean(dim=(4, 5))  # B, H//patch_size, W//patch_size, C
                x_patches = x_patches.view(B, -1, C)  # B, (H//patch_size)*(W//patch_size), C
                
                x_patches = self.regional_norms[i](x_patches)
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
        
        # 瓶颈MLP和残差连接
        local_out = local_out + self.mlp(self.norm_out(local_out))
        
        # 重塑回原始形状
        output = local_out.transpose(1, 2).view(B, C, H, W)
        
        return output + x  # 残差连接


# 新增：混合CNN-Transformer块 - V2版本
class HybridCNNTransformerBlockV2(TimestepBlock):
    """
    混合CNN-Transformer块，结合CNN的局部建模和Transformer的长距离依赖建模 - V2版本
    使用瓶颈MLP结构进一步优化参数量
    """
    
    def __init__(self, channels, emb_channels, dropout=0.1, use_checkpoint=False, bottleneck_ratio=0.25):
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
        
        # Transformer分支（使用V2版本MSTB）
        self.transformer_branch = MultiScaleSparseTransformerBlockV2(
            channels=channels,
            dropout=dropout,
            bottleneck_ratio=bottleneck_ratio
        )
        
        # 时间步嵌入
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, channels),
        )
        
        # 特征融合 - 使用1x1卷积而不是2倍通道的卷积
        self.fusion_conv = conv_nd(2, channels, channels, 1)
        
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, emb), self.parameters(), True)
        else:
            return self._forward(x, emb)
            
    def _forward(self, x, emb):
        # 时间步嵌入
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        
        # CNN分支
        conv_out = self.conv_branch(x + emb_out)
        
        # Transformer分支
        trans_out = self.transformer_branch(x + emb_out)
        
        # 特征融合：加权平均而不是拼接
        fused = (conv_out + trans_out) / 2
        output = self.fusion_conv(fused)
        
        return output + x  # 残差连接


class HAEUNetModelV2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    HAE (Heterogeneous AutoEncoder) UNet Model - V2 Version
    在Lite版本基础上，使用瓶颈MLP结构进一步减少参数量

    :param image_size: the size of the input images.
    :param in_channels: the number of channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: the number of channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param clf_free: whether to use classifier-free guidance.
    :param use_hae: whether to use heterogeneous autoencoder blocks.
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
        bottleneck_ratio=0.25,  # V2新增：瓶颈比例
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
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
        self.clf_free = clf_free
        self.use_hae = use_hae
        self.bottleneck_ratio = bottleneck_ratio  # V2新增

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

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
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
                    # 编码器始终使用标准的AttentionBlock，不使用HAE异构结构
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
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 中间块：使用标准注意力块，确保编码器和中间块都有标准注意力
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
                
                # 标准注意力块（与UNet-V2保持一致）
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                    
                # 在上采样前添加异构块（根据图示，每个上采样前都有双分支结构）
                if level and i == num_res_blocks:
                    # 添加异构块在上采样前
                    if use_hae:
                        layers.append(
                            HybridCNNTransformerBlockV2(
                                ch,
                                time_embed_dim,
                                dropout=dropout,
                                use_checkpoint=use_checkpoint,
                                bottleneck_ratio=bottleneck_ratio
                            )
                        )
                    
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
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
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
        
        # 条件设置（与HAE原版保持一致）
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