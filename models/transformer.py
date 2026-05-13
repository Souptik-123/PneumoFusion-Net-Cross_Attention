"""
models/transformer.py  –  Cross-Attention Transformer

Architecture (one CrossModalAttention layer)
--------------------------------------------

  cnn_token  text_token  num_token          (each: B × 1 × D)
      │            │           │
      ├────────────┼───────────┤   ← cross-modal QKV attention
      │            │           │     each token queries the other two
      │            │           │
      ├────────────┼───────────┤   ← multi-head self-attention
      │            │           │     all three tokens attend to each other
      │                            (intra-modal refinement)
      └────────── FFN ────────┘   ← position-wise feed-forward + LayerNorm
                     │
             (B, 3, D) output sequence

CrossAttentionTransformer stacks N such layers.

Design rationale
----------------
• Cross-modal attention models interactions between different data sources
  explicitly: when CT findings are ambiguous, text cues ("lymphocyte surge")
  or lab values ("CRP > 100") can shift attention weights accordingly.
• Multi-head self-attention after cross-attention provides intra-sequence
  refinement – the three updated tokens can still interact freely.
• GELU activation in the FFN is smoother than ReLU for transformer blocks.
• Layer normalisation + residual connections stabilise training.
"""

import torch
import torch.nn as nn

from config import FUSION_DIM, XATTN_HEADS, XATTN_FF_DIM, XATTN_LAYERS, XATTN_DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
# Single Cross-Modal Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    One cross-modal attention layer processing three modality tokens.

    Step 1 – Cross-modal QKV attention
        Each token acts as the query; the other two tokens form the key-value
        context. This lets every modality "read" complementary information
        from the remaining modalities.

    Step 2 – Multi-head self-attention
        All three tokens are concatenated into a 3-token sequence and
        processed with standard multi-head self-attention. This provides
        intra-sequence refinement after the cross-modal exchange.

    Step 3 – FFN + LayerNorm (pre-norm style)
        Position-wise feed-forward network with GELU, followed by
        add-and-norm residual.

    Parameters
    ----------
    d_model : token / embedding dimension (FUSION_DIM)
    n_heads : number of attention heads
    ff_dim  : hidden dimension of the FFN
    dropout : dropout in attention and FFN
    """

    def __init__(
        self,
        d_model: int   = FUSION_DIM,
        n_heads: int   = XATTN_HEADS,
        ff_dim:  int   = XATTN_FF_DIM,
        dropout: float = XATTN_DROPOUT,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        # ── per-modality input projections ────────────────────────────────
        # each modality vector is already FUSION_DIM, but a learned
        # projection allows the layer to reparameterise freely.
        self.proj_cnn  = nn.Linear(d_model, d_model)
        self.proj_text = nn.Linear(d_model, d_model)
        self.proj_num  = nn.Linear(d_model, d_model)

        # ── cross-modal attention (one per modality as query) ────────────
        self.cross_attn_cnn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_num  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ── multi-head self-attention (all 3 tokens) ─────────────────────
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ── position-wise FFN ─────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # ── Layer normalisation ───────────────────────────────────────────
        self.norm_cross = nn.LayerNorm(d_model)  # after cross-modal step
        self.norm_self  = nn.LayerNorm(d_model)  # after self-attention step
        self.norm_ffn   = nn.LayerNorm(d_model)  # after FFN step

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        cnn_feat:  torch.Tensor,
        text_feat: torch.Tensor,
        num_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cnn_feat, text_feat, num_feat : (B, D)  – one vector per modality

        Returns
        -------
        seq : (B, 3, D)  – updated three-token sequence
              seq[:, 0] = updated CNN token
              seq[:, 1] = updated text token
              seq[:, 2] = updated numerical token
        """
        # project and unsqueeze → (B, 1, D) tokens
        c = self.proj_cnn(cnn_feat).unsqueeze(1)
        t = self.proj_text(text_feat).unsqueeze(1)
        n = self.proj_num(num_feat).unsqueeze(1)

        # ── Step 1: cross-modal attention ─────────────────────────────────
        # CNN token queries text + numerical context
        ctx_c, _ = self.cross_attn_cnn(
            c,
            torch.cat([t, n], dim=1),
            torch.cat([t, n], dim=1),
        )
        # Text token queries CNN + numerical context
        ctx_t, _ = self.cross_attn_text(
            t,
            torch.cat([c, n], dim=1),
            torch.cat([c, n], dim=1),
        )
        # Numerical token queries CNN + text context
        ctx_n, _ = self.cross_attn_num(
            n,
            torch.cat([c, t], dim=1),
            torch.cat([c, t], dim=1),
        )

        # residual + norm
        c = self.norm_cross(c + self.drop(ctx_c))
        t = self.norm_cross(t + self.drop(ctx_t))
        n = self.norm_cross(n + self.drop(ctx_n))

        # ── Step 2: multi-head self-attention ─────────────────────────────
        seq = torch.cat([c, t, n], dim=1)         # (B, 3, D)
        sa_out, _ = self.self_attn(seq, seq, seq)
        seq = self.norm_self(seq + self.drop(sa_out))

        # ── Step 3: FFN + LayerNorm ───────────────────────────────────────
        seq = self.norm_ffn(seq + self.ffn(seq))

        return seq   # (B, 3, D)


# ─────────────────────────────────────────────────────────────────────────────
# Stacked Cross-Attention Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionTransformer(nn.Module):
    """
    Stack of N CrossModalAttention layers.

    The output of each layer is a 3-token sequence (B, 3, D).
    Each subsequent layer reads the three refined tokens from the
    previous layer, deepening the cross-modal interaction.

    Parameters
    ----------
    n_layers : number of stacked CrossModalAttention layers
    d_model  : token dimension
    n_heads  : number of attention heads
    ff_dim   : FFN hidden dimension
    dropout  : dropout probability

    Forward
    -------
    cnn_feat, text_feat, num_feat : (B, D)  →  seq : (B, 3, D)
    """

    def __init__(
        self,
        n_layers: int   = XATTN_LAYERS,
        d_model:  int   = FUSION_DIM,
        n_heads:  int   = XATTN_HEADS,
        ff_dim:   int   = XATTN_FF_DIM,
        dropout:  float = XATTN_DROPOUT,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        cnn_feat:  torch.Tensor,
        text_feat: torch.Tensor,
        num_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cnn_feat, text_feat, num_feat : (B, D) – projected modality features

        Returns
        -------
        seq : (B, 3, D)  – final refined three-token sequence
        """
        seq = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                # first layer: consume the raw projected modality vectors
                seq = layer(cnn_feat, text_feat, num_feat)    # (B, 3, D)
            else:
                # subsequent layers: consume the updated tokens
                seq = layer(seq[:, 0], seq[:, 1], seq[:, 2])
        return seq   # (B, 3, D)
