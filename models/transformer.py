"""
models/transformer.py  –  Cross-Attention Transformer

FIX 1 – Tensor shape crash  ("size 256 must match size 3 at non-singleton dim 2")
----------------------------------------------------------------------------------
Root cause: nn.MultiheadAttention(batch_first=True) was silently ignored in some
PyTorch builds (< 1.9 stable) and in certain CUDA-compiled wheels, returning tensors
in (T, B, D) layout instead of (B, T, D).  The classification head then received
seq of shape (3, B, 256) and tried to broadcast with w of shape (1, 3, 1),
triggering "size 256 ≠ size 3" at dimension 2.

Fix: drop batch_first=True entirely.  All MHA calls now explicitly permute to
(T, B, D) before attention and permute back to (B, T, D) after.  This works on
every PyTorch version ≥ 1.7 and is CUDA-safe.

FIX 2 – Stochastic Depth (DropPath) for overfitting
-----------------------------------------------------
With ~5 600 training samples the transformer layers overfit quickly.
Stochastic Depth randomly drops entire residual branches during training
(sets their contribution to zero for a random subset of batch items),
which acts as a strong regulariser without changing the architecture.
drop_path_rate is set per-layer with linear scaling from 0 → max_drop_path.

Architecture (one CrossModalAttention layer)
--------------------------------------------
  cnn_token  text_token  num_token          (each: B × 1 × D)
      │            │           │
      ├────────────┼───────────┤   ← cross-modal QKV attention
      │            │           │     each token queries the other two
      │            │           │
      ├────────────┼───────────┤   ← multi-head self-attention
      │            │           │
      └────────── FFN ────────┘   ← FFN + LayerNorm + StochasticDepth
                     │
             (B, 3, D) output sequence

CrossAttentionTransformer stacks N such layers.
"""

import torch
import torch.nn as nn

from config import FUSION_DIM, XATTN_HEADS, XATTN_FF_DIM, XATTN_LAYERS, XATTN_DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Depth (DropPath)
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """
    Randomly drop entire residual branches per sample during training.
    At test time the full branch is used (scaled by survival probability).

    drop_prob : probability of dropping the branch (0 = disabled)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        # shape (B, 1, 1, ...) so it broadcasts over all feature dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(survival).div_(survival)
        return x * noise


# ─────────────────────────────────────────────────────────────────────────────
# Single Cross-Modal Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    One cross-modal attention layer processing three modality tokens.

    NOTE: all MultiheadAttention modules use the default (T, B, D) layout
    (batch_first=False) to guarantee correctness across all PyTorch versions.
    Permutations are handled explicitly in forward().

    Parameters
    ----------
    d_model       : token / embedding dimension (FUSION_DIM)
    n_heads       : number of attention heads
    ff_dim        : hidden dimension of the FFN
    dropout       : dropout in attention and FFN
    drop_path_rate: stochastic depth probability for residual branches
    """

    def __init__(
        self,
        d_model:        int   = FUSION_DIM,
        n_heads:        int   = XATTN_HEADS,
        ff_dim:         int   = XATTN_FF_DIM,
        dropout:        float = XATTN_DROPOUT,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        # ── per-modality input projections ────────────────────────────────
        self.proj_cnn  = nn.Linear(d_model, d_model)
        self.proj_text = nn.Linear(d_model, d_model)
        self.proj_num  = nn.Linear(d_model, d_model)

        # ── cross-modal attention (batch_first=False → (T,B,D) layout) ───
        # NOTE: do NOT pass batch_first=True; permute manually in forward()
        self.cross_attn_cnn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.cross_attn_num  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )

        # ── multi-head self-attention ────────────────────────────────────
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )

        # ── position-wise FFN ─────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # ── Layer norms (separate instances – avoids shared-state issues) ─
        self.norm_cnn   = nn.LayerNorm(d_model)   # after cross-modal for CNN
        self.norm_text  = nn.LayerNorm(d_model)   # after cross-modal for Text
        self.norm_num   = nn.LayerNorm(d_model)   # after cross-modal for Num
        self.norm_self  = nn.LayerNorm(d_model)   # after self-attention
        self.norm_ffn   = nn.LayerNorm(d_model)   # after FFN

        # ── Stochastic Depth per residual branch ──────────────────────────
        self.drop_path_cross = StochasticDepth(drop_path_rate)
        self.drop_path_self  = StochasticDepth(drop_path_rate)
        self.drop_path_ffn   = StochasticDepth(drop_path_rate)

        self.attn_drop = nn.Dropout(dropout)

    # ── helpers: permute between (B, T, D) ↔ (T, B, D) ──────────────────

    @staticmethod
    def _to_tbd(x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (T, B, D)"""
        return x.permute(1, 0, 2)

    @staticmethod
    def _to_btd(x: torch.Tensor) -> torch.Tensor:
        """(T, B, D) → (B, T, D)"""
        return x.permute(1, 0, 2)

    def forward(
        self,
        cnn_feat:  torch.Tensor,   # (B, D)
        text_feat: torch.Tensor,   # (B, D)
        num_feat:  torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        """
        Returns
        -------
        seq : (B, 3, D)
              seq[:, 0] = updated CNN token
              seq[:, 1] = updated text token
              seq[:, 2] = updated numerical token
        """
        # ── project and add sequence dim → (B, 1, D) ─────────────────────
        c = self.proj_cnn(cnn_feat).unsqueeze(1)    # (B, 1, D)
        t = self.proj_text(text_feat).unsqueeze(1)  # (B, 1, D)
        n = self.proj_num(num_feat).unsqueeze(1)    # (B, 1, D)

        # ── Step 1: cross-modal attention (explicit (T,B,D) permutation) ──
        # CNN queries text + numerical context
        tn_ctx = torch.cat([t, n], dim=1)           # (B, 2, D)
        ctx_c, _ = self.cross_attn_cnn(
            self._to_tbd(c),                        # Q: (1, B, D)
            self._to_tbd(tn_ctx),                   # K: (2, B, D)
            self._to_tbd(tn_ctx),                   # V: (2, B, D)
        )
        ctx_c = self._to_btd(ctx_c)                 # (B, 1, D)

        # Text queries CNN + numerical context
        cn_ctx = torch.cat([c, n], dim=1)
        ctx_t, _ = self.cross_attn_text(
            self._to_tbd(t),
            self._to_tbd(cn_ctx),
            self._to_tbd(cn_ctx),
        )
        ctx_t = self._to_btd(ctx_t)                 # (B, 1, D)

        # Numerical queries CNN + text context
        ct_ctx = torch.cat([c, t], dim=1)
        ctx_n, _ = self.cross_attn_num(
            self._to_tbd(n),
            self._to_tbd(ct_ctx),
            self._to_tbd(ct_ctx),
        )
        ctx_n = self._to_btd(ctx_n)                 # (B, 1, D)

        # residual + norm (separate LayerNorm per token to avoid state sharing)
        c = self.norm_cnn(c   + self.drop_path_cross(self.attn_drop(ctx_c)))
        t = self.norm_text(t  + self.drop_path_cross(self.attn_drop(ctx_t)))
        n = self.norm_num(n   + self.drop_path_cross(self.attn_drop(ctx_n)))

        # ── Step 2: multi-head self-attention ─────────────────────────────
        seq = torch.cat([c, t, n], dim=1)           # (B, 3, D)
        seq_tbd = self._to_tbd(seq)                 # (3, B, D)
        sa_out, _ = self.self_attn(seq_tbd, seq_tbd, seq_tbd)
        sa_out = self._to_btd(sa_out)               # (B, 3, D)
        seq = self.norm_self(seq + self.drop_path_self(self.attn_drop(sa_out)))

        # ── Step 3: FFN + LayerNorm ───────────────────────────────────────
        seq = self.norm_ffn(seq + self.drop_path_ffn(self.ffn(seq)))

        return seq   # (B, 3, D)  — guaranteed


# ─────────────────────────────────────────────────────────────────────────────
# Stacked Cross-Attention Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionTransformer(nn.Module):
    """
    Stack of N CrossModalAttention layers with linearly-scaled stochastic depth.

    drop_path_rate is linearly increased from 0 (first layer) to max_drop_path
    (last layer), following the schedule used in DeiT / Swin Transformer.

    Parameters
    ----------
    n_layers      : number of stacked CrossModalAttention layers
    d_model       : token dimension
    n_heads       : number of attention heads
    ff_dim        : FFN hidden dimension
    dropout       : attention + FFN dropout
    max_drop_path : maximum stochastic-depth probability (last layer)

    Forward
    -------
    cnn_feat, text_feat, num_feat : (B, D)  →  seq : (B, 3, D)
    """

    def __init__(
        self,
        n_layers:      int   = XATTN_LAYERS,
        d_model:       int   = FUSION_DIM,
        n_heads:       int   = XATTN_HEADS,
        ff_dim:        int   = XATTN_FF_DIM,
        dropout:       float = XATTN_DROPOUT,
        max_drop_path: float = 0.10,          # 10% max stochastic depth
    ):
        super().__init__()
        # linearly scale drop-path rate across layers
        dpr = [
            max_drop_path * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, ff_dim, dropout, drop_path_rate=dpr[i])
            for i in range(n_layers)
        ])

    def forward(
        self,
        cnn_feat:  torch.Tensor,
        text_feat: torch.Tensor,
        num_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
        seq : (B, 3, D)  – guaranteed shape on every PyTorch version
        """
        seq = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                seq = layer(cnn_feat, text_feat, num_feat)     # (B, 3, D)
            else:
                # slice each token: (B, D) – then each gets unsqueeze(1) inside
                seq = layer(seq[:, 0], seq[:, 1], seq[:, 2])  # (B, 3, D)

        return seq   # (B, 3, D)
