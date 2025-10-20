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
    :param num_dilation_branches: Number of dilation rate branches (e.g., 3 for dilation rates 2, 4, 8)
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
                                        Expected format: [dilation2_features, dilation4_features, dilation8_features]
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
    3. Multi-scale fusion: Combines features from different dilation rates
    
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

        # Input normalization and projection
        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
        )

        # Enhanced parallel dilated convolutions with directional support
        # Calculate channels per branch to ensure exact division
        base_channels = self.out_channels // 3
        remaining_channels = self.out_channels - 3 * base_channels
        
        if self.use_directional:
            # Use directional dilated convolutions
            self.dilated_conv_2 = DirectionalDilatedConv(channels, base_channels, dilation=2, dims=dims)
            self.dilated_conv_4 = DirectionalDilatedConv(channels, base_channels, dilation=4, dims=dims)
            
            # Add remaining channels to the last branch
            last_branch_channels = base_channels + remaining_channels
            self.dilated_conv_8 = DirectionalDilatedConv(channels, last_branch_channels, dilation=8, dims=dims)
            
            # Get channel information for DirectionalSEBlock directly from the conv layer
            self.channels_per_direction = self.dilated_conv_2.channels_per_direction
        else:
            # Use standard dilated convolutions (fallback)
            self.dilated_conv_2 = conv_nd(dims, channels, base_channels, 3, padding=2, dilation=2)
            self.dilated_conv_4 = conv_nd(dims, channels, base_channels, 3, padding=4, dilation=4)
            
            # Add remaining channels to the last branch
            last_branch_channels = base_channels + remaining_channels
            self.dilated_conv_8 = conv_nd(dims, channels, last_branch_channels, 3, padding=8, dilation=8)
            self.channels_per_direction = None
        
        # No extra conv needed since we handle remaining channels in the last branch
        self.dilated_conv_extra = None

        # Fusion layer with batch normalization
        self.fusion = nn.Sequential(
            conv_nd(dims, self.out_channels, self.out_channels, 1),
            normalization(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # SE Attention mechanism - choose between directional and standard
        if self.use_se_attention:
            if self.use_directional and self.use_directional_se and self.channels_per_direction is not None:
                # Use DirectionalSEBlock for enhanced directional attention
                self.se_block = DirectionalSEBlock(
                    channels_per_direction=self.channels_per_direction,
                    num_dilation_branches=3,  # 3 dilation rates: 2, 4, 8
                    reduction=16,
                    extra_channels=remaining_channels  # Pass the extra channels to the last branch
                )
                self.use_directional_se_block = True
            else:
                # Use standard SEBlock
                self.se_block = SEBlock(self.out_channels, reduction=16)
                self.use_directional_se_block = False
        else:
            self.se_block = nn.Identity()
            self.use_directional_se_block = False

        # Timestep embedding processing
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
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
        Apply the enhanced dilated block to a Tensor, conditioned on a timestep embedding.
        
        Features:
        1. Directional dilated convolutions for boundary preservation
        2. Enhanced directional SE attention or standard SE attention
        3. Multi-scale feature fusion

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Input processing
        h = self.in_layers(x)
        
        # Parallel dilated convolutions (directional or standard)
        if self.use_directional and self.use_directional_se_block:
            # Use directional convolutions with separate outputs for DirectionalSEBlock
            h2, _ = self.dilated_conv_2.forward_directional(h)
            h4, _ = self.dilated_conv_4.forward_directional(h)
            h8, _ = self.dilated_conv_8.forward_directional(h)
            
            # Prepare features list for DirectionalSEBlock
            directional_features_list = [h2, h4, h8]
            
            # Apply DirectionalSEBlock before fusion
            h = self.se_block(directional_features_list)
            
        else:
            # Standard approach: concatenate first, then apply SE
            h2 = self.dilated_conv_2(h)
            h4 = self.dilated_conv_4(h)
            h8 = self.dilated_conv_8(h)
            
            # Concatenate dilated features (h8 already includes remaining channels)
            h = th.cat([h2, h4, h8], dim=1)
            
            # Multi-scale fusion with normalization
            h = self.fusion(h)
            
            # Apply standard SE attention for adaptive feature selection
            h = self.se_block(h)
        
        # For directional SE, apply fusion after SE attention
        if self.use_directional and self.use_directional_se_block:
            h = self.fusion(h)
        
        # Process timestep embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        # Apply timestep conditioning
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h


class PyramidFusionBlock(nn.Module):
    """
    多尺度特征金字塔融合模块
    
    通过并行的多膨胀率卷积构建特征金字塔，实现多尺度信息的高效融合。
    设计用于增强跳跃连接，提升异常检测的精度和边界定位能力。
    
    核心特性：
    1. 多尺度感受野：3×3, 5×5, 9×9 三种感受野并行处理
    2. 特征分组：减少计算量，提高效率
    3. 残差连接：保证梯度流动和特征复用
    4. 自适应融合：通过注意力机制智能融合不同尺度特征
    """
    
    def __init__(
        self,
        skip_channels,      # 跳跃连接的通道数
        current_channels,   # 当前层的通道数
        dims=2,            # 空间维度
        dilation_rates=(1, 3, 5),  # 膨胀率组合
        reduction=8,       # 注意力机制的降维比例
        use_checkpoint=False
    ):
        super().__init__()
        
        self.skip_channels = skip_channels
        self.current_channels = current_channels
        self.dims = dims
        self.dilation_rates = dilation_rates
        self.use_checkpoint = use_checkpoint
        
        # 计算每个分组的通道数
        self.group_channels = skip_channels // len(dilation_rates)
        assert skip_channels % len(dilation_rates) == 0, \
            f"skip_channels ({skip_channels}) must be divisible by number of dilation rates ({len(dilation_rates)})"
        
        # 多尺度膨胀卷积分支
        self.pyramid_convs = nn.ModuleList()
        for dilation in dilation_rates:
            conv_branch = nn.Sequential(
                conv_nd(dims, self.group_channels, self.group_channels, 
                       kernel_size=3, padding=dilation, dilation=dilation),
                normalization(self.group_channels),
                nn.SiLU(),
                conv_nd(dims, self.group_channels, self.group_channels, 
                       kernel_size=1)  # 1x1卷积用于特征精炼
            )
            self.pyramid_convs.append(conv_branch)
        
        # 特征融合层
        fused_channels = skip_channels
        self.fusion_conv = nn.Sequential(
            conv_nd(dims, fused_channels, fused_channels, kernel_size=1),
            normalization(fused_channels),
            nn.SiLU()
        )
        
        # 自适应注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool3d(1),
            conv_nd(dims, fused_channels, fused_channels // reduction, kernel_size=1),
            nn.SiLU(),
            conv_nd(dims, fused_channels // reduction, fused_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出投影层（保持跳跃连接的通道数不变）
        self.output_proj = conv_nd(dims, fused_channels, skip_channels, kernel_size=1)
        
        # 残差连接保持原始通道数
        self.skip_proj = nn.Identity()
    
    def forward(self, skip_features):
        """
        前向传播
        
        Args:
            skip_features: 来自编码器的跳跃连接特征 [B, skip_channels, H, W]
            
        Returns:
            enhanced_features: 增强后的特征 [B, current_channels, H, W]
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, skip_features)
        else:
            return self._forward(skip_features)
    
    def _forward(self, skip_features):
        B, C, *spatial_dims = skip_features.shape
        
        # 1. 特征分组
        # 将跳跃连接特征按通道分组，每组对应一个膨胀率
        grouped_features = th.chunk(skip_features, len(self.dilation_rates), dim=1)
        
        # 2. 多尺度膨胀卷积
        pyramid_outputs = []
        for i, (group_feat, conv_branch) in enumerate(zip(grouped_features, self.pyramid_convs)):
            # 对每组特征应用对应的膨胀卷积
            enhanced_feat = conv_branch(group_feat)
            pyramid_outputs.append(enhanced_feat)
        
        # 3. 特征拼接
        fused_features = th.cat(pyramid_outputs, dim=1)
        
        # 4. 特征融合
        fused_features = self.fusion_conv(fused_features)
        
        # 5. 自适应注意力加权
        attention_weights = self.attention(fused_features)
        attended_features = fused_features * attention_weights
        
        # 6. 输出投影
        output = self.output_proj(attended_features)
        
        # 7. 残差连接
        residual = self.skip_proj(skip_features)
        enhanced_features = output + residual
        
        return enhanced_features


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
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

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out_expand = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out_expand)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


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


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    This version uses DilatedResBlock in the middle_block for improved receptive field.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
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
        use_pyramid_fusion=False,  # 新增：是否启用多尺度金字塔融合
        pyramid_fusion_levels=None,  # 新增：指定在哪些层级启用金字塔融合
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

        # Modified middle_block with DilatedResBlock
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
            DilatedResBlock(  # Replace second ResBlock with DilatedResBlock
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 保存 input_block_chans 的副本，因为在构建 output_blocks 时会被 pop() 清空
        input_block_chans_copy = input_block_chans.copy()

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
        
        # 初始化多尺度金字塔融合模块
        self.use_pyramid_fusion = use_pyramid_fusion
        self.pyramid_fusion_blocks = nn.ModuleDict()
        
        # 更新 input_block_chans 为最终的完整列表（使用副本）
        self.input_block_chans = input_block_chans_copy
        
        if use_pyramid_fusion:
            # 如果未指定层级，默认先在16×16分辨率启用（保守策略）
            if pyramid_fusion_levels is None:
                # 根据 channel_mult 确定关键层级
                # 通常 16×16 对应 level 2，先从这个分辨率开始验证
                pyramid_fusion_levels = [2] if len(channel_mult) > 2 else [1]
            
            # 为每个指定的层级创建金字塔融合模块
            output_block_idx = 0
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    if level in pyramid_fusion_levels and output_block_idx < len(input_block_chans_copy):
                        # 获取对应的跳跃连接通道数和当前层通道数
                        skip_ch = input_block_chans_copy[-(output_block_idx + 1)]  # 对应的跳跃连接通道数
                        current_ch = int(model_channels * mult)  # 当前层通道数
                        
                        # 确保跳跃连接通道数能被膨胀率数量整除
                        if skip_ch % 2 == 0:  # 使用2个膨胀率 (1, 3) 以兼容更多通道数
                            fusion_key = f"level_{level}_block_{i}"
                            self.pyramid_fusion_blocks[fusion_key] = PyramidFusionBlock(
                                skip_channels=skip_ch,
                                current_channels=current_ch,
                                dims=dims,
                                dilation_rates=(1, 3),  # 使用2个膨胀率
                                reduction=8,
                                use_checkpoint=use_checkpoint
                            )
                    output_block_idx += 1
        
        self.use_fp16 = use_fp16

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
        
        # 处理输出块，集成金字塔融合
        output_block_idx = 0
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                skip_features = hs.pop()
                
                # 检查是否需要应用金字塔融合
                if self.use_pyramid_fusion:
                    fusion_key = f"level_{level}_block_{i}"
                    if fusion_key in self.pyramid_fusion_blocks:
                        # 应用金字塔融合增强跳跃连接
                        enhanced_skip = self.pyramid_fusion_blocks[fusion_key](skip_features)
                        h = th.cat([h, enhanced_skip], dim=1)
                    else:
                        # 使用原始跳跃连接
                        h = th.cat([h, skip_features], dim=1)
                else:
                    # 使用原始跳跃连接
                    h = th.cat([h, skip_features], dim=1)
                
                # 应用对应的输出块
                h = self.output_blocks[output_block_idx](h, emb, cemb_mm)
                output_block_idx += 1
        
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
        # encoder_channels = time_embed_dim
        encoder_channels = None

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
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
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
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
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
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
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "adaptive_v1":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(ch, self.out_channels),
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
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) if timesteps is not None else None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)