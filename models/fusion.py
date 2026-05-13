"""
models/fusion.py  –  Feature Fusion Layer

Architecture
------------
[cnn_feat | text_feat | num_feat]  (concatenation along feature dim)
  → Linear(total → fusion_dim * 2)
  → LayerNorm + ReLU + Dropout
  → Linear(fusion_dim * 2 → fusion_dim)
  → LayerNorm + ReLU

Purpose
-------
After each modality is independently encoded into its own feature space,
this layer concatenates all three representations into a single vector
and projects it into a shared FUSION_DIM space.

The shared space serves as the input to the Cross-Attention Transformer,
where each modality token is a FUSION_DIM-dimensional vector:
  • cnn_feat   (B, CNN_OUT_DIM)   →  projected to (B, FUSION_DIM)
  • text_feat  (B, TEXT_OUT_DIM)  →  projected to (B, FUSION_DIM)
  • num_feat   (B, NUM_OUT_DIM)   →  projected to (B, FUSION_DIM)

Note: the concat → project path here produces a *combined* representation,
while the per-modality projections (cnn_proj, text_proj, num_proj) in
PneumoFusionNet produce the individual tokens fed to the transformer.
Both projections share the same FUSION_DIM target.
"""

import torch
import torch.nn as nn

from config import CNN_OUT_DIM, TEXT_OUT_DIM, NUM_OUT_DIM, FUSION_DIM


class FeatureFusion(nn.Module):
    """
    Concatenate the three modality embeddings and project to FUSION_DIM.

    Parameters
    ----------
    cnn_dim    : dimension of the CNN image features
    text_dim   : dimension of the text features
    num_dim    : dimension of the numerical features
    fusion_dim : target dimension of the fused representation
    dropout    : dropout probability in the projection MLP

    Forward
    -------
    cnn_feat  : (B, cnn_dim)
    text_feat : (B, text_dim)
    num_feat  : (B, num_dim)
    → fused   : (B, fusion_dim)
    """

    def __init__(
        self,
        cnn_dim:    int   = CNN_OUT_DIM,
        text_dim:   int   = TEXT_OUT_DIM,
        num_dim:    int   = NUM_OUT_DIM,
        fusion_dim: int   = FUSION_DIM,
        dropout:    float = 0.3,
    ):
        super().__init__()
        total = cnn_dim + text_dim + num_dim
        self.proj = nn.Sequential(
            nn.Linear(total, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        cnn_feat:  torch.Tensor,
        text_feat: torch.Tensor,
        num_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cnn_feat  : (B, cnn_dim)
        text_feat : (B, text_dim)
        num_feat  : (B, num_dim)

        Returns
        -------
        (B, fusion_dim)  – fused multimodal feature vector
        """
        x = torch.cat([cnn_feat, text_feat, num_feat], dim=-1)
        return self.proj(x)
