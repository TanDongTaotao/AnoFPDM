import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums


def linear(in_features, out_features, bias=True):
    """
    Create a linear module.
    """
    return nn.Linear(in_features, out_features, bias)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GlobalExtraction(nn.Module):
  def __init__(self, dim=None):
    super().__init__()
    self.dim = dim
    # Use spatial pooling instead of channel pooling to preserve channel dimensions
    self.spatial_pool = nn.AdaptiveAvgPool2d(1)
    
  def forward(self, x):
    # Apply spatial pooling and then broadcast back to original spatial size
    B, C, H, W = x.shape
    pooled = self.spatial_pool(x)  # [B, C, 1, 1]
    # Broadcast back to original spatial dimensions
    global_features = pooled.expand(B, C, H, W)
    return global_features

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g):
    # Process local features (x)
    x_local = self.local(x)
    
    # Process global features (g) and ensure channel compatibility
    g_global = self.global_(g)
    
    # If g has different channels than x, we need to adapt it
    if g_global.shape[1] != x_local.shape[1]:
        # Use adaptive pooling to match spatial dimensions, then use conv to match channels
        B, C_g, H, W = g_global.shape
        B, C_x, H_x, W_x = x_local.shape
        
        # Resize g to match x's spatial dimensions if needed
        if H != H_x or W != W_x:
            g_global = F.interpolate(g_global, size=(H_x, W_x), mode='bilinear', align_corners=False)
        
        # Use 1x1 conv to match channel dimensions
        if not hasattr(self, 'channel_adapter'):
            self.channel_adapter = nn.Conv2d(C_g, C_x, 1, bias=False).to(g_global.device)
        g_global = self.channel_adapter(g_global)

    fuse = self.bn(x_local + g_global)
    return fuse


class DilatedUNetSkipAdapter(nn.Module):
    """
    Adapter module to integrate MSA²Net skip connections into Dilated UNet.
    Handles channel dimension matching and timestep conditioning.
    """
    def __init__(self, skip_channels, decoder_channels, timestep_emb_dim, dims=2):
        super().__init__()
        self.skip_channels = skip_channels
        self.decoder_channels = decoder_channels
        self.dims = dims
        
        # Channel adaptation for decoder features to match skip channels
        if decoder_channels != skip_channels:
            self.decoder_adapter = conv_nd(dims, decoder_channels, skip_channels, 1)
        else:
            self.decoder_adapter = nn.Identity()
        
        # MSA²Net attention module (operates on skip channels)
        self.masag_attention = MultiScaleGatedAttn(
            dim=skip_channels, 
            timestep_emb_dim=timestep_emb_dim,
            dims=dims
        )
        
    def forward(self, skip_features, decoder_features, timestep_emb):
        """
        Apply MSA²Net attention to skip connections.
        
        Args:
            skip_features: Features from encoder skip connection [B, skip_channels, H, W]
            decoder_features: Features from decoder [B, decoder_channels, H, W]
            timestep_emb: Timestep embedding [B, timestep_emb_dim]
        
        Returns:
            Enhanced skip features [B, skip_channels, H, W]
        """
        # Adapt decoder features to match skip channel dimensions
        adapted_decoder = self.decoder_adapter(decoder_features)
        
        # Apply MSA²Net attention to skip features
        enhanced_skip = self.masag_attention(
            x=skip_features,
            g=adapted_decoder, 
            timestep_emb=timestep_emb
        )
        
        return enhanced_skip


class MultiScaleGatedAttn(nn.Module):
    """
    Multi-scale Gated Attention module with timestep conditioning for diffusion models.
    Enhanced version of MSA²Net skip connections with temporal conditioning.
    """
    def __init__(self, dim, timestep_emb_dim=None, dims=2):
        super().__init__()
        self.dim = dim
        self.dims = dims
        self.timestep_emb_dim = timestep_emb_dim
        
        # Core multi-scale fusion
        self.multi = MultiscaleFusion(dim)
        self.selection = conv_nd(dims, dim, 2, 1)
        self.proj = conv_nd(dims, dim, dim, 1)
        
        # Normalization layers
        if dims == 2:
            self.bn = nn.BatchNorm2d(dim)
            self.bn_2 = nn.BatchNorm2d(dim)
        elif dims == 3:
            self.bn = nn.BatchNorm3d(dim)
            self.bn_2 = nn.BatchNorm3d(dim)
        else:
            # Use GroupNorm for 1D or as fallback
            self.bn = nn.GroupNorm(min(32, dim), dim)
            self.bn_2 = nn.GroupNorm(min(32, dim), dim)
        
        # Timestep conditioning layers
        if timestep_emb_dim is not None:
            self.timestep_proj = nn.Sequential(
                nn.SiLU(),
                linear(timestep_emb_dim, dim * 2),  # For scale and shift
            )
            self.use_timestep_conditioning = True
        else:
            self.use_timestep_conditioning = False
        
        # Final convolution block
        self.conv_block = nn.Sequential(
            conv_nd(dims, dim, dim, kernel_size=1, stride=1)
        )

    def forward(self, x, g, timestep_emb=None):
        """
        Forward pass with optional timestep conditioning.
        
        Args:
            x: Skip connection features from encoder [B, C, H, W]
            g: Gate signal features from decoder [B, C, H, W] 
            timestep_emb: Timestep embedding [B, timestep_emb_dim] (optional)
        
        Returns:
            Enhanced skip connection features [B, C, H, W]
        """
        x_ = x.clone()
        g_ = g.clone()

        # Multi-scale fusion
        multi = self.multi(x, g)  # B, C, H, W

        # Attention weight generation
        multi = self.selection(multi)  # B, 2, H, W
        attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
        A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

        # Apply attention weights
        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_

        # Residual connections
        x_att = x_att + x_
        g_att = g_att + g_
        
        # Ensure x_att and g_att have the same channel dimensions for interaction
        if x_att.shape[1] != g_att.shape[1]:
            # Adapt g_att to match x_att's channel dimension
            if not hasattr(self, 'g_channel_adapter'):
                self.g_channel_adapter = conv_nd(self.dims, g_att.shape[1], x_att.shape[1], 1).to(g_att.device)
            g_att = self.g_channel_adapter(g_att)
        
        # Bidirectional interaction
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att

        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        # Project interaction features
        projected = self.proj(interaction)
        
        # Apply timestep conditioning if available
        if self.use_timestep_conditioning and timestep_emb is not None:
            timestep_out = self.timestep_proj(timestep_emb)
            while len(timestep_out.shape) < len(projected.shape):
                timestep_out = timestep_out[..., None]
            
            # Split into scale and shift
            scale, shift = torch.chunk(timestep_out, 2, dim=1)
            projected = projected * (1 + scale) + shift
        
        # Apply normalization and activation
        projected = torch.sigmoid(self.bn(projected))
        weighted = projected * x_

        # Final convolution
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        
        return y

if __name__ == "__main__":
    xi = torch.randn(1, 192, 28, 28).cuda()
    #xi_1 = torch.randn(1, 384, 14, 14)
    g = torch.randn(1, 192, 28, 28).cuda()
    #ff = ContextBridge(dim=192)

    attn = MultiScaleGatedAttn(dim = xi.shape[1]).cuda()

    print(attn(xi, g).shape)
