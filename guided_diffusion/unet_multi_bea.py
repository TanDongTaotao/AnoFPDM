import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from .unet_v2 import AttentionBlock, ResBlock, TimestepEmbedSequential, Upsample, Downsample


def sobel(x):
    """
    Apply Sobel operator to compute gradient magnitude.
    
    :param x: input tensor [N x C x H x W]
    :return: gradient magnitude [N x 1 x H x W]
    """
    # Sobel kernels
    sobel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    # Compute gradients
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_mag = th.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    return grad_mag


class MultiBoundaryAwareAttention(nn.Module):
    """
    Multi-layer Boundary-Aware Attention (Multi-BEA) module that computes gradient maps using Sobel operator
    and generates channel attention weights to enhance boundary sensitivity across multiple layers.
    """
    def __init__(self, channels, dims=2):
        super().__init__()
        self.channels = channels
        self.dims = dims
        # 1x1 convolution to generate channel attention weights from gradient map
        self.channel_attention = conv_nd(dims, 1, channels, 1)
        
    def forward(self, x, x0):
        """
        Apply boundary-aware attention to feature map.
        
        :param x: feature map [N x C x H x W]
        :param x0: input image [N x C x H x W] for gradient computation
        :return: boundary-aware weighted feature map
        """
        # Compute gradient map using Sobel operator
        # Convert to grayscale if multi-channel
        if x0.shape[1] > 1:
            x0_gray = th.mean(x0, dim=1, keepdim=True)
        else:
            x0_gray = x0
            
        # Compute gradient magnitude using Sobel filter
        grad_map = sobel(x0_gray)  # [N x 1 x H x W]
        
        # Resize gradient map to match feature map size if needed
        if grad_map.shape[-2:] != x.shape[-2:]:
            grad_map = F.interpolate(grad_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Generate channel attention weights
        attention_weights = th.sigmoid(self.channel_attention(grad_map))  # [N x C x H x W]
        
        # Apply channel-wise attention
        enhanced_features = x * attention_weights
        
        return enhanced_features


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
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class MultiBEAUNetModel(nn.Module):
    """
    The Multi-layer Boundary-Enhanced Attention UNet model with attention and timestep embedding.
    Based on UNetV2 with added Multi-layer Boundary-Aware Attention (Multi-BEA) mechanism.
    BEA is applied to: middle block, last encoder block, and first decoder block.
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
        use_multi_bea=True,  # Enable/disable multi-layer boundary-aware attention
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
        self.use_multi_bea = use_multi_bea

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
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
        
        # Track which blocks need BEA
        self.encoder_bea_block_idx = None  # Will be set to the last encoder block
        
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

        # Set the encoder BEA block index (last encoder block before middle)
        self.encoder_bea_block_idx = len(self.input_blocks) - 1

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

        # Add BEA to middle block
        if self.use_multi_bea:
            self.middle_bea_module = MultiBoundaryAwareAttention(ch, dims=dims)

        # Store encoder channel info before decoder construction modifies input_block_chans
        encoder_ch_for_bea = None
        if self.use_multi_bea and self.encoder_bea_block_idx < len(input_block_chans):
            encoder_ch_for_bea = input_block_chans[self.encoder_bea_block_idx]
        
        # Create BEA modules before decoder construction
        if self.use_multi_bea:
            # BEA for last encoder block
            encoder_ch = encoder_ch_for_bea if encoder_ch_for_bea is not None else ch
            self.encoder_bea_module = MultiBoundaryAwareAttention(encoder_ch, dims=dims)
            
            # BEA for first decoder block (will use first decoder block's output channel count)
            self.decoder_bea_block_idx = 0  # First decoder block

        self.output_blocks = nn.ModuleList([])
        decoder_bea_ch = None  # Store first decoder block's output channels
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
                
                # Store the first decoder block's output channels for BEA
                if self.use_multi_bea and decoder_bea_ch is None:
                    decoder_bea_ch = ch
                    
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
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

        # Create decoder BEA module with correct channel count
        if self.use_multi_bea:
            self.decoder_bea_module = MultiBoundaryAwareAttention(decoder_bea_ch, dims=dims)

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def forward(self, x, timesteps, y=None, threshold=-1, null=False, clf_free=False, **kwargs):
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

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        cemb_mm = None
        
        #-------------------------------- Condition Setup --------------------------
        if self.num_classes is not None:
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
            else:
                raise Exception("Invalid condition setup")
                
            assert cemb is not None
            assert cemb_mm is not None
            emb = emb + cemb 
        #-------------------------------- Condition Setup --------------------------

        h = x.type(self.dtype)
        x_original = x  # Store original input for BEA
        
        # Encoder blocks
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, cemb_mm)
            # Apply BEA to the last encoder block (closest to middle)
            if self.use_multi_bea and i == self.encoder_bea_block_idx:
                h = self.encoder_bea_module(h, x_original)
            hs.append(h)
            
        # Middle block with BEA
        h = self.middle_block(h, emb, cemb_mm)
        if self.use_multi_bea:
            h = self.middle_bea_module(h, x_original)
            
        # Decoder blocks
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, cemb_mm)
            # Apply BEA to the first decoder block (closest to middle)
            if self.use_multi_bea and i == self.decoder_bea_block_idx:
                h = self.decoder_bea_module(h, x_original)
                
        h = h.type(x.dtype)
        return self.out(h)