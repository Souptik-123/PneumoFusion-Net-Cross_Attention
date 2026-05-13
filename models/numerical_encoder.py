"""
models/numerical_encoder.py  –  MLP + Residual Numerical Feature Encoder

Architecture
------------
standardised_numerics
  → ResidualBlock(in_dim  → hidden[0])
  → ResidualBlock(hidden[0] → hidden[1])
  → ResidualBlock(hidden[1] → out_dim)

Each ResidualBlock
  Linear → BatchNorm1d → ReLU → Dropout
  + skip connection (projected if dimensions differ)

Design rationale
----------------
• Residual connections provide a direct gradient path for backpropagation,
  which stabilises training and preserves original feature information
  (e.g. an elevated WBC count does not get "forgotten" through deep layers).
• Standardisation is applied upstream (in the Dataset), so every indicator
  (WBC, CRP, PCT, …) is treated on the same scale.
• Dropout (p=0.3) prevents over-reliance on any single biomarker.
• 15 input features: 7 numerical lab values + one-hot sex (2 dims) = 9
  (configurable via config.NUM_NUMERICAL_FEATURES).
"""

import torch
import torch.nn as nn

from config import NUM_NUMERICAL_FEATURES, MLP_HIDDEN_DIMS, NUM_OUT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    One fully-connected residual block.

    Forward path  : x → Linear → BatchNorm1d → ReLU → Dropout → out
    Skip path     : x → (Linear projection if dims differ) → skip
    Output        : ReLU(out + skip)

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    dropout : dropout probability
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        # 1×1 linear projection for the residual when dimensions differ
        self.proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.drop(self.bn(self.fc(x)))
        skip = self.proj(x)
        return self.act(out + skip)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Numerical Encoder
# ─────────────────────────────────────────────────────────────────────────────

class MLPNumericalEncoder(nn.Module):
    """
    Stack of ResidualBlocks that encodes standardised laboratory /
    numerical features into a compact representation.

    Parameters
    ----------
    in_dim      : number of input features  (default: NUM_NUMERICAL_FEATURES)
    hidden_dims : sequence of hidden layer widths
    out_dim     : output feature dimension  (default: NUM_OUT_DIM)
    dropout     : dropout probability for every residual block

    Forward
    -------
    x : (B, in_dim)  →  (B, out_dim)
    """

    def __init__(
        self,
        in_dim:      int   = NUM_NUMERICAL_FEATURES,
        hidden_dims        = MLP_HIDDEN_DIMS,
        out_dim:     int   = NUM_OUT_DIM,
        dropout:     float = 0.3,
    ):
        super().__init__()
        dims   = [in_dim] + list(hidden_dims) + [out_dim]
        blocks = [
            ResidualBlock(dims[i], dims[i + 1], dropout)
            for i in range(len(dims) - 1)
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_dim)  – standardised numerical feature vector

        Returns
        -------
        (B, out_dim)  – encoded numerical representation
        """
        return self.net(x)
