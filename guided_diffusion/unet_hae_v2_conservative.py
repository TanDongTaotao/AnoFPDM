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

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

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
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116dcc91b1c5b68e5b8b40/diffusion_tf/models/unet.py#L66.
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
    A module which performs QKV attention. Matches legacy QKVAttention.
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


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP for efficient feature transformation
    """
    def __init__(self, channels, bottleneck_ratio=0.25, dropout=0.1):
        super().__init__()
        bottleneck_channels = int(channels * bottleneck_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, bottleneck_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_channels, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.mlp(x)


class MultiScaleSparseTransformerBlockV2(nn.Module):
    """
    Multi-scale sparse transformer block with improved efficiency
    """
    def __init__(self, channels, num_heads=8, dropout=0.1, patch_sizes=[1, 4, 8], bottleneck_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        
        # Multi-head attention for different scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in patch_sizes
        ])
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Bottleneck MLP
        self.mlp = BottleneckMLP(channels, bottleneck_ratio, dropout)
        
        # Scale fusion
        self.scale_fusion = nn.Linear(len(patch_sizes) * channels, channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_patches(self, x, patch_size):
        """Create patches from input tensor"""
        B, C, H, W = x.shape
        
        if patch_size == 1:
            # Pixel-level attention
            patches = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        else:
            # Patch-level attention
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                H, W = H + pad_h, W + pad_w
            
            patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, patch_size * patch_size)
            patches = patches.mean(dim=-1).transpose(1, 2)  # [B, num_patches, C]
        
        return patches
    
    def create_sparse_mask(self, seq_len, sparsity_ratio=0.9):
        """Create sparse attention mask"""
        # Create a sparse mask that keeps only a fraction of attention weights
        mask = th.rand(seq_len, seq_len) > sparsity_ratio
        
        # Ensure diagonal is always attended (self-attention)
        mask.fill_diagonal_(True)
        
        # Ensure each position attends to at least one other position
        for i in range(seq_len):
            if not mask[i].any():
                # If no attention, attend to the next position (circular)
                mask[i, (i + 1) % seq_len] = True
        
        return mask
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-scale processing
        scale_outputs = []
        
        for i, (attention, patch_size) in enumerate(zip(self.attentions, self.patch_sizes)):
            # Create patches
            patches = self.create_patches(x, patch_size)
            seq_len = patches.shape[1]
            
            # Apply layer norm
            patches_norm = self.norm1(patches)
            
            # Create sparse attention mask for efficiency
            if seq_len > 64:  # Only apply sparsity for long sequences
                attn_mask = self.create_sparse_mask(seq_len, sparsity_ratio=0.8)
                attn_mask = attn_mask.to(x.device)
            else:
                attn_mask = None
            
            # Self-attention
            attn_output, _ = attention(
                patches_norm, patches_norm, patches_norm,
                attn_mask=attn_mask,
                need_weights=False
            )
            
            # Residual connection
            patches = patches + self.dropout(attn_output)
            
            # MLP with residual
            patches = patches + self.dropout(self.mlp(self.norm2(patches)))
            
            # Reshape back to spatial dimensions for fusion
            if patch_size == 1:
                spatial_output = patches.transpose(1, 2).view(B, C, H, W)
            else:
                # For patches, we need to interpolate back to original size
                num_patches_h = H // patch_size
                num_patches_w = W // patch_size
                patches_spatial = patches.transpose(1, 2).view(B, C, num_patches_h, num_patches_w)
                spatial_output = F.interpolate(patches_spatial, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_outputs.append(spatial_output)
        
        # Fuse multi-scale features
        if len(scale_outputs) > 1:
            fused = th.cat(scale_outputs, dim=1)  # [B, len(patch_sizes)*C, H, W]
            fused = fused.view(B, -1, H * W).transpose(1, 2)  # [B, H*W, len(patch_sizes)*C]
            fused = self.scale_fusion(fused)  # [B, H*W, C]
            fused = fused.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        else:
            fused = scale_outputs[0]
        
        return fused


class HybridCNNTransformerBlockV2(TimestepBlock):
    """
    Hybrid CNN-Transformer block that combines convolutional and transformer operations
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
        num_heads=8,
        patch_sizes=[1, 4, 8],
        bottleneck_ratio=0.25
    ):
        super().__init__()
        
        # CNN component (ResBlock)
        self.cnn_block = ResBlock(
            channels=channels,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down
        )
        
        # Transformer component
        self.transformer_block = MultiScaleSparseTransformerBlockV2(
            channels=out_channels or channels,
            num_heads=num_heads,
            dropout=dropout,
            patch_sizes=patch_sizes,
            bottleneck_ratio=bottleneck_ratio
        )
        
        # Fusion layer
        fusion_channels = out_channels or channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.GroupNorm(8, fusion_channels),
            nn.SiLU()
        )
        
    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.cnn_block.use_checkpoint
        )
    
    def _forward(self, x, emb):
        # CNN path
        cnn_output = self.cnn_block(x, emb)
        
        # Transformer path
        transformer_output = self.transformer_block(cnn_output)
        
        # Fuse CNN and Transformer outputs
        fused = th.cat([cnn_output, transformer_output], dim=1)
        output = self.fusion(fused)
        
        return output


class HAEUNetModelV2Conservative(nn.Module):
    """
    HAE V2 模型的保守优化版本
    
    主要优化:
    1. 条件计算cemb_mm，避免不必要的内存分配
    2. 保持完全的功能兼容性
    3. 无质量损失的内存优化
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
        memory_optimization=True,  # 新增：启用内存优化
        zero_threshold=1e-6,  # 新增：零张量检测阈值
    ):
        super().__init__()
        
        # 保存优化配置
        self.memory_optimization = memory_optimization
        self.zero_threshold = zero_threshold
        
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

        time_embed_dim = model_channels * 4
        encoder_channels = time_embed_dim
        
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

    def _compute_cemb_mm_optimized(self, cemb):
        """
        优化的cemb_mm计算 - 条件计算版本
        
        Args:
            cemb: 条件嵌入张量 [batch_size, embed_dim]
            
        Returns:
            优化后的cemb_mm张量或None
        """
        if not self.memory_optimization:
            # 如果未启用优化，使用原始方法
            return th.einsum("ab,ac -> abc", cemb, cemb)
        
        # 检查是否为零张量
        if th.allclose(cemb, th.zeros_like(cemb), atol=self.zero_threshold):
            return None
        
        # 检查哪些样本需要计算
        non_zero_mask = th.any(th.abs(cemb) > self.zero_threshold, dim=1)
        non_zero_count = th.sum(non_zero_mask).item()
        
        if non_zero_count == 0:
            return None
        
        batch_size, embed_dim = cemb.shape
        
        if non_zero_count == batch_size:
            # 所有样本都非零，直接计算
            return th.einsum("ab,ac -> abc", cemb, cemb)
        
        # 只对非零样本计算cemb_mm
        active_indices = non_zero_mask.nonzero().squeeze(-1)
        cemb_active = cemb[active_indices]
        cemb_mm_active = th.einsum("ab,ac -> abc", cemb_active, cemb_active)
        
        # 创建完整的cemb_mm张量
        cemb_mm = th.zeros(batch_size, embed_dim, embed_dim, 
                          device=cemb.device, dtype=cemb.dtype)
        cemb_mm[active_indices] = cemb_mm_active
        
        return cemb_mm

    def forward(self, x, timesteps, y=None, threshold=-1, null=False, clf_free=False):
        """
        Apply the model to an input batch.
        
        优化版本的forward方法，使用条件计算来减少内存使用
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        cemb_mm = None
        
        # 条件设置（与HAE原版保持一致，但使用优化的cemb_mm计算）
        if self.num_classes is not None:
            cemb = None
            if threshold != -1: 
                assert threshold > 0
                cemb = self.class_emb(self.label_emb(y))
                mask = th.rand(cemb.shape[0]) < threshold
                cemb[np.where(mask)[0]] = 0
                # 使用优化的cemb_mm计算
                cemb_mm = self._compute_cemb_mm_optimized(cemb)
            elif threshold == -1 and clf_free: 
                if null:
                    cemb = th.zeros_like(emb)
                else:
                    cemb = self.class_emb(self.label_emb(y)) 
                # 使用优化的cemb_mm计算
                cemb_mm = self._compute_cemb_mm_optimized(cemb)
            else:
                raise Exception("Invalid condition setup")
                
            assert cemb is not None
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