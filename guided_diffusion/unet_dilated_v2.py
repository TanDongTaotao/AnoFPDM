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

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
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
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
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
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
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


class SEBlock(nn.Module):

    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DirectionalSEBlock(nn.Module):
    """
    Directional Squeeze-and-Excitation block for enhanced directional feature attention.
    
    This block applies SE attention separately to different directional features from
    multiple dilation rates, then performs global fusion with another SE attention.
    
    Key innovations:
    1. Direction-wise SE attention: Applies SE to horizontal, vertical, and diagonal features separately
    2. Multi-scale directional fusion: Combines features from different dilation rates per direction
    3. Global SE attention: Final attention mechanism on the fused directional features
    
    :param channels_per_direction: List of channel numbers for each direction [horizontal, vertical, diagonal1, diagonal2]
    :param num_dilation_branches: Number of dilation rate branches (e.g., 3 for dilation rates 1, 2, 4)
    :param reduction: Reduction ratio for SE attention
    """
    def __init__(self, channels_per_direction, num_dilation_branches=3, reduction=16, extra_channels=0):
        super().__init__()
        self.channels_per_direction = channels_per_direction
        self.num_dilation_branches = num_dilation_branches
        self.num_directions = len(channels_per_direction)
        self.extra_channels = extra_channels
        
        # Calculate total channels per direction across all dilation branches
        # The last direction gets the extra channels
        self.total_channels_per_dir = []
        for i, ch in enumerate(channels_per_direction):
            if i == len(channels_per_direction) - 1:
                # Last direction gets extra channels from the last branch
                total_ch = ch * num_dilation_branches + extra_channels
            else:
                total_ch = ch * num_dilation_branches
            self.total_channels_per_dir.append(total_ch)
        
        self.total_channels = sum(self.total_channels_per_dir)
        
        # Direction-wise SE blocks
        self.directional_se_blocks = nn.ModuleList([
            SEBlock(total_ch, reduction) for total_ch in self.total_channels_per_dir
        ])
        
        # Global SE block for final fusion
        self.global_se_block = SEBlock(self.total_channels, reduction)
        
        # Optional: Cross-directional attention for enhanced feature interaction
        # Use appropriate number of groups for normalization
        num_groups = min(32, self.total_channels)
        while self.total_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.cross_directional_fusion = nn.Sequential(
            conv_nd(2, self.total_channels, self.total_channels, 1),
            nn.GroupNorm(num_groups, self.total_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, directional_features_list):
        """
        Apply directional SE attention to multi-scale directional features.
        
        :param directional_features_list: List of feature tensors from different dilation branches
                                        Each tensor contains concatenated directional features
                                        Expected format: [dilation1_features, dilation2_features]
                                        Each dilation_features: [batch, channels, height, width]
                                        where channels = sum(channels_per_direction) for each branch
        :return: Enhanced feature tensor with directional SE attention applied
        """
        batch_size = directional_features_list[0].size(0)
        height, width = directional_features_list[0].size(2), directional_features_list[0].size(3)
        
        # Split each dilation branch into directional components
        direction_wise_features = [[] for _ in range(self.num_directions)]
        
        for branch_idx, dilation_features in enumerate(directional_features_list):
            # For the last branch, it might have extra channels
            if branch_idx == len(directional_features_list) - 1:
                # Last branch might have extra channels
                total_expected = sum(self.channels_per_direction)
                actual_channels = dilation_features.size(1)
                extra_channels = actual_channels - total_expected
                
                # Split the concatenated features into directional components
                start_idx = 0
                for dir_idx, ch_per_dir in enumerate(self.channels_per_direction):
                    if dir_idx == len(self.channels_per_direction) - 1:
                        # Add extra channels to the last direction
                        end_idx = start_idx + ch_per_dir + extra_channels
                    else:
                        end_idx = start_idx + ch_per_dir
                    
                    direction_wise_features[dir_idx].append(
                        dilation_features[:, start_idx:end_idx, :, :]
                    )
                    start_idx = end_idx
            else:
                # Regular branches
                start_idx = 0
                for dir_idx, ch_per_dir in enumerate(self.channels_per_direction):
                    end_idx = start_idx + ch_per_dir
                    direction_wise_features[dir_idx].append(
                        dilation_features[:, start_idx:end_idx, :, :]
                    )
                    start_idx = end_idx
        
        # Apply SE attention to each direction across all dilation scales
        enhanced_directional_features = []
        for dir_idx in range(self.num_directions):
            # Concatenate features from all dilation rates for this direction
            dir_features = th.cat(direction_wise_features[dir_idx], dim=1)  # [batch, total_ch_per_dir, H, W]
            
            # Apply directional SE attention
            enhanced_dir_features = self.directional_se_blocks[dir_idx](dir_features)
            enhanced_directional_features.append(enhanced_dir_features)
        
        # Concatenate all enhanced directional features
        fused_features = th.cat(enhanced_directional_features, dim=1)  # [batch, total_channels, H, W]
        
        # Apply cross-directional fusion
        fused_features = self.cross_directional_fusion(fused_features)
        
        # Apply global SE attention for final feature refinement
        final_features = self.global_se_block(fused_features)
        
        return final_features


class DirectionalDilatedConv(nn.Module):
    """
    Directional dilated convolutions for enhanced boundary preservation.
    Implements horizontal, vertical, and diagonal directional filters.
    """
    def __init__(self, in_channels, out_channels, dilation, dims=2):
        super().__init__()
        self.dilation = dilation
        self.dims = dims
        
        # Calculate output channels per direction
        channels_per_dir = out_channels // 4  # 4 directions: horizontal, vertical, diagonal1, diagonal2
        remaining_channels = out_channels - 4 * channels_per_dir
        
        # Store channel information for DirectionalSEBlock
        self.channels_per_dir = channels_per_dir
        self.remaining_channels = remaining_channels
        self.channels_per_direction = [channels_per_dir, channels_per_dir, channels_per_dir, channels_per_dir]
        if remaining_channels > 0:
            self.channels_per_direction.append(remaining_channels)
        
        # Horizontal directional convolution (1x3 kernel)
        self.horizontal_conv = conv_nd(dims, in_channels, channels_per_dir, 
                                     kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation))
        
        # Vertical directional convolution (3x1 kernel)
        self.vertical_conv = conv_nd(dims, in_channels, channels_per_dir,
                                   kernel_size=(3, 1), padding=(dilation, 0), dilation=(dilation, 1))
        
        # Diagonal convolutions (3x3 with asymmetric dilation)
        self.diagonal1_conv = conv_nd(dims, in_channels, channels_per_dir,
                                    kernel_size=3, padding=dilation, dilation=dilation)
        
        self.diagonal2_conv = conv_nd(dims, in_channels, channels_per_dir,
                                    kernel_size=3, padding=dilation, dilation=dilation)
        
        # Handle remaining channels with standard convolution
        if remaining_channels > 0:
            self.extra_conv = conv_nd(dims, in_channels, remaining_channels, 3, padding=dilation, dilation=dilation)
        else:
            self.extra_conv = None
            
    def forward(self, x):
        """
        Standard forward pass that concatenates all directional features.
        Maintains backward compatibility with existing code.
        """
        # Apply directional convolutions
        h_horizontal = self.horizontal_conv(x)
        h_vertical = self.vertical_conv(x)
        h_diagonal1 = self.diagonal1_conv(x)
        h_diagonal2 = self.diagonal2_conv(x)
        
        # Concatenate all directional features
        if self.extra_conv is not None:
            h_extra = self.extra_conv(x)
            return th.cat([h_horizontal, h_vertical, h_diagonal1, h_diagonal2, h_extra], dim=1)
        else:
            return th.cat([h_horizontal, h_vertical, h_diagonal1, h_diagonal2], dim=1)
    
    def forward_directional(self, x):
        """
        Forward pass that returns directional features separately for DirectionalSEBlock.
        
        :param x: Input tensor [batch, in_channels, height, width]
        :return: Concatenated directional features (same as forward) and 
                 channels_per_direction list for DirectionalSEBlock
        """
        # Apply directional convolutions
        h_horizontal = self.horizontal_conv(x)
        h_vertical = self.vertical_conv(x)
        h_diagonal1 = self.diagonal1_conv(x)
        h_diagonal2 = self.diagonal2_conv(x)
        
        # Concatenate all directional features
        if self.extra_conv is not None:
            h_extra = self.extra_conv(x)
            concatenated_features = th.cat([h_horizontal, h_vertical, h_diagonal1, h_diagonal2, h_extra], dim=1)
        else:
            concatenated_features = th.cat([h_horizontal, h_vertical, h_diagonal1, h_diagonal2], dim=1)
        
        return concatenated_features, self.channels_per_direction


class DilatedResBlock(TimestepBlock):
    """
    Enhanced dilated residual block with directional atrous convolutions and SE attention.
    
    Key innovations:
    1. Directional Atrous Convolutions: Separate horizontal, vertical, and diagonal filters
       for better boundary preservation in medical images
    2. SE Attention: Channel-wise attention for adaptive feature selection
    3. Multi-scale fusion: Combines features from three dilation rates (1, 2, 4)
    
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param use_directional: if True, use directional dilated convolutions.
    :param use_se_attention: if True, use SE attention mechanism.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_directional=True,
        use_se_attention=True,
        use_directional_se=True,  # New parameter for directional SE attention
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_directional = use_directional
        self.use_se_attention = use_se_attention
        self.use_directional_se = use_directional_se

        # Input normalization and initial convolution
        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.use_directional:
            # Three-branch dilated convolutions with dilation rates [1, 2, 4]
            # Optimized for 32x32 resolution with perfect receptive field matching
            
            # Calculate channel distribution for three branches
            base_channels = self.out_channels // 3  # Split into 3 branches
            remaining_channels = self.out_channels - 2 * base_channels
            
            # Branch 1: Dilation rate 1 (3x3 receptive field)
            self.dilated_conv_1 = DirectionalDilatedConv(
                self.out_channels, base_channels, dilation=1, dims=dims
            )
            
            # Branch 2: Dilation rate 2 (5x5 receptive field)
            self.dilated_conv_2 = DirectionalDilatedConv(
                self.out_channels, base_channels, dilation=2, dims=dims
            )
            
            # Branch 3: Dilation rate 4 (9x9 receptive field) + remaining channels
            last_branch_channels = base_channels + remaining_channels
            self.dilated_conv_3 = DirectionalDilatedConv(
                self.out_channels, last_branch_channels, dilation=4, dims=dims
            )
            
            # Store channel information for DirectionalSEBlock
            self.channels_per_direction = self.dilated_conv_1.channels_per_direction
            
            if self.use_directional_se:
                # Enhanced directional SE attention for multi-scale directional features
                self.directional_se_block = DirectionalSEBlock(
                    channels_per_direction=self.channels_per_direction,
                    num_dilation_branches=3,  # 3 for dilation rates 1, 2, 4
                    reduction=16,
                    extra_channels=remaining_channels
                )
        else:
            # Standard convolution fallback
            self.standard_conv = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)

        # SE attention for channel-wise feature enhancement
        if self.use_se_attention and not self.use_directional_se:
            self.se_block = SEBlock(self.out_channels, reduction=16)

        # Output layers
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        # Skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the dilated residual block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        # Initial processing
        h = self.in_layers(x)
        
        # Apply timestep embedding
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
            
        if self.use_directional:
            # Apply three-branch directional dilated convolutions
            h1 = self.dilated_conv_1(h)  # Dilation rate 1
            h2 = self.dilated_conv_2(h)  # Dilation rate 2
            h3 = self.dilated_conv_3(h)  # Dilation rate 4
            
            # Concatenate multi-scale directional features
            h = th.cat([h1, h2, h3], dim=1)
            
            if self.use_directional_se:
                # Apply enhanced directional SE attention
                h = self.directional_se_block([h1, h2, h3])
        else:
            # Standard convolution path
            h = self.standard_conv(h)
            
        # Apply SE attention if enabled and not using directional SE
        if self.use_se_attention and not self.use_directional_se:
            h = self.se_block(h)
            
        # Final output processing
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116dcc91b1c5b68b1e8b3f/diffusion_tf/models/unet.py#L66.
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
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        if num_head_channels == -1:
            self.num_head_channels = channels
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_head_channels = num_head_channels
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1) if encoder_channels else None
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        if encoder_out is not None:
            encoder_out = encoder_out.reshape(b, -1, spatial[0] * spatial[1])
            encoder_kv = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_kv)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] % (2 * self.n_heads) == 0
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


class UNetModelV2(nn.Module):
    """
    Enhanced UNet model with dilated convolutions optimized for 32x32 resolution layers.
    
    This version moves DilatedResBlock from bottleneck (8x8) to 32x32 resolution layers
    for optimal receptive field utilization and computational efficiency.
    
    Key improvements:
    1. DilatedResBlock applied at 32x32 resolution for optimal feature-computation balance
    2. Bottleneck layer uses standard ResBlock to avoid over-dilated receptive fields
    3. Dual-branch dilated convolutions [1,2] perfectly matched to 32x32 feature maps
    4. Enhanced directional SE attention for multi-scale directional features
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
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        # Calculate target resolution for DilatedResBlock (32x32)
        target_resolution = 32
        current_resolution = image_size
        target_ds = image_size // target_resolution  # ds value when resolution becomes 32x32
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # Determine if we should use DilatedResBlock at this resolution
                use_dilated = (ds == target_ds)  # Use DilatedResBlock when resolution is 32x32
                
                if use_dilated:
                    layers = [
                        DilatedResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                else:
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

        # Bottleneck layer: Use standard ResBlock instead of DilatedResBlock
        # This avoids over-dilated receptive fields at 8x8 resolution
        self.middle_block = TimestepEmbedSequential(
            ResBlock(  # Standard ResBlock for bottleneck
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
            ResBlock(  # Standard ResBlock for bottleneck
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
                
                # Determine if we should use DilatedResBlock at this resolution
                use_dilated = (ds == target_ds)  # Use DilatedResBlock when resolution is 32x32
                
                if use_dilated:
                    layers = [
                        DilatedResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                else:
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

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param threshold: a float threshold for clf-free training (portion of samples to be masked)
                            also indicating if the model is training in clf-free mode
        :param clf_free: a bool indicating if the model is sampled in clf-free mode
        :param null: a bool indicating if the null embedding should be used in sampling
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"
         
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        cemb_mm = None
        
        #-------------------------------- Condition Setup --------------------------
        '''
        For clf-free training, set threshold > 0
        For clf-free sampling, set threshold = -1, and clf_free = True
        For clf training, set threshold = -1, and clf_free = False
        '''
        if self.num_classes is not None:
            # assert y.shape == (x.shape[0],)
            cemb = None
            # for clf-free training
            if threshold != -1: 
                assert threshold > 0
                cemb = self.class_emb(self.label_emb(y))
                mask = th.rand(cemb.shape[0])<threshold
                cemb[np.where(mask)[0]] = 0
                cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb)
            # for clf-free sampling
            elif threshold == -1 and clf_free: 
                if null: # null embedding
                    cemb = th.zeros_like(emb)
                else: # class condition embedding
                    cemb = self.class_emb(self.label_emb(y)) 
                cemb_mm = th.einsum("ab,ac -> abc", cemb, cemb) 
            # for non-clf-free condition embedding, e.g., classifier guided sampling
            # elif threshold == -1 and not clf_free:
            #     cemb = self.label_emb(y)
            else:
                raise Exception("Invalid condition setup")
                
            assert cemb is not None
            assert cemb_mm is not None
            emb = emb + cemb 
        #-------------------------------- Condition Setup --------------------------
            

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, cemb_mm)
            hs.append(h)
        h = self.middle_block(h, emb, cemb_mm)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, cemb_mm)
        h = h.type(x.dtype)
        return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
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
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
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
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
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

        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch, swish=1.0),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch, swish=1.0),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048, swish=1.0),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, dim=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)