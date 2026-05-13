"""
models/attention.py  –  Attention building blocks

Classes
-------
ChannelAttention        : SE-style avg+max pool → shared MLP → sigmoid gate
SpatialAttention        : avg+max pool across channels → 7×7 conv → sigmoid gate
GCSA                    : Global Channel-Spatial Attention
                          (ChannelAttention → channel shuffle → SpatialAttention)
DepthwiseSeparableConv  : depthwise conv + pointwise conv + BN + ReLU
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    SE-style channel attention.
    Both average-pool and max-pool descriptors are computed independently,
    passed through a shared MLP, summed, then squashed with sigmoid.

    Input  : (B, C, H, W)
    Output : (B, C, H, W)  – channel-reweighted feature map
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=[2, 3])               # (B, C)
        mx  = x.amax(dim=[2, 3])               # (B, C)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Attention
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Spatial attention using channel-wise avg and max descriptors.
    A single conv layer maps the 2-channel descriptor to a 1-channel
    spatial weight map which is then sigmoid-gated.

    Input  : (B, C, H, W)
    Output : (B, C, H, W)  – spatially-reweighted feature map
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)      # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)      # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * attn


# ─────────────────────────────────────────────────────────────────────────────
# GCSA
# ─────────────────────────────────────────────────────────────────────────────

class GCSA(nn.Module):
    """
    Global Channel-Spatial Attention (GCSA).

    Pipeline
    --------
    x  →  ChannelAttention  →  channel_shuffle  →  SpatialAttention  →  out

    The channel shuffle step between the two attention phases improves
    cross-group feature mixing and prevents the model from over-relying
    on fixed channel groups.

    Parameters
    ----------
    channels  : number of input/output channels (C)
    groups    : number of channel-shuffle groups (must divide C evenly)
    reduction : reduction ratio for the channel attention MLP
    """

    def __init__(self, channels: int, groups: int = 8, reduction: int = 16):
        super().__init__()
        assert channels % groups == 0, (
            f"channels ({channels}) must be divisible by groups ({groups})"
        )
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
        self.groups       = groups

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self._channel_shuffle(x, self.groups)
        x = self.spatial_attn(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise Separable Convolution
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution  (depthwise + pointwise).

    Standard convolution filters spatial and channel dimensions jointly.
    DSC decouples these into two steps:
      1. Depthwise conv  – one spatial filter per input channel.
      2. Pointwise conv  – 1×1 conv to mix channel information.

    This reduces parameter count and FLOPs significantly while keeping
    feature quality high for medical CT images.

    Input  : (B, in_ch, H, W)
    Output : (B, out_ch, H', W')
    """

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        kernel:  int = 3,
        stride:  int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.dw  = nn.Conv2d(
            in_ch, in_ch, kernel,
            stride=stride, padding=padding,
            groups=in_ch, bias=False,
        )
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))
