"""
models/cnn_encoder.py  –  CNN Image Encoder

Architecture
------------
ResNet50 (ImageNet pre-trained, adapted to single-channel input)
  → DepthwiseSeparableConv   (2048 → 2048, channel-efficient refinement)
  → GCSA                     (Global Channel-Spatial Attention)
  → AdaptiveAvgPool2d(1)
  → Linear projection        (2048 → CNN_OUT_DIM)

Key adaptations for grayscale CT images
----------------------------------------
• First conv layer changed from 3-channel to 1-channel.
  Pre-trained weights are preserved by summing across the colour dimension,
  so low-level edge/texture detectors learned on ImageNet are retained.
• Depthwise separable convolution replaces one standard conv to cut FLOPs.
• GCSA guides attention to diagnostically relevant regions (GGOs,
  consolidations, upper-lobe lesions) without extra supervision.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from config import CNN_OUT_DIM
from models.attention import DepthwiseSeparableConv, GCSA


class CNNImageEncoder(nn.Module):
    """
    Modified ResNet50 for single-channel (grayscale) CT images.

    Parameters
    ----------
    out_dim    : output feature dimension after linear projection
    pretrained : whether to load ImageNet-1K pre-trained weights

    Forward
    -------
    x : (B, 1, H, W)   →   (B, out_dim)
    """

    def __init__(self, out_dim: int = CNN_OUT_DIM, pretrained: bool = True):
        super().__init__()

        # ── Load backbone ─────────────────────────────────────────────────
        weights  = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        def disable_inplace_relu(module):
            for child in module.children():
                if isinstance(child, nn.ReLU):
                    child.inplace = False
                disable_inplace_relu(child)

        disable_inplace_relu(backbone)

        # ── Adapt first conv: 3-channel → 1-channel ───────────────────────
        # Sum the RGB weights so low-level feature detectors are preserved.
        orig_conv = backbone.conv1                    # weight: (64, 3, 7, 7)
        new_conv  = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            new_conv.weight.copy_(
                orig_conv.weight.sum(dim=1, keepdim=True)  # (64, 1, 7, 7)
            )
        backbone.conv1 = new_conv

        # ── Stem + ResNet stages ──────────────────────────────────────────
        self.stem   = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # output: (B, 256,  H/4,  W/4)
        self.layer2 = backbone.layer2   # output: (B, 512,  H/8,  W/8)
        self.layer3 = backbone.layer3   # output: (B, 1024, H/16, W/16)
        self.layer4 = backbone.layer4   # output: (B, 2048, H/32, W/32)

        # ── Depthwise Separable Conv refinement ───────────────────────────
        # Refines the 2048-channel feature maps with fewer FLOPs.
        self.dsc = DepthwiseSeparableConv(2048, 2048)

        # ── GCSA ─────────────────────────────────────────────────────────
        # Highlights discriminative channels and spatial regions.
        self.gcsa = GCSA(channels=2048, groups=8)

        # ── Global pool + projection ──────────────────────────────────────
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
        )

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  – grayscale CT image tensor

        Returns
        -------
        (B, out_dim)  – image feature vector
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dsc(x)
        x = self.gcsa(x)
        x = self.pool(x)
        x = self.fc_proj(x)
        return x

    # ── Fine-tuning helpers ───────────────────────────────────────────────

    def freeze_backbone(self):
        """
        Freeze stem, layer1, layer2 during the first training phase
        so that only the head and task-specific modules are updated.
        """
        freeze_prefixes = ("stem", "layer1", "layer2")
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in freeze_prefixes):
                param.requires_grad = False

    def unfreeze_all(self):
        """Un-freeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
