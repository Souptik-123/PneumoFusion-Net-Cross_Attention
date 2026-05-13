"""
models/text_encoder.py  –  BiLSTM + Additive Attention Text Encoder

Architecture
------------
token_ids
  → Embedding (vocab_size × embed_dim)
  → PositionalEncoding   (sinusoidal, fixed)
  → BiLSTM               (bidirectional, num_layers)
  → AdditiveAttention    (Bahdanau-style, collapses sequence to context vector)
  → FC projection        (2*hidden_dim → TEXT_OUT_DIM)

Design rationale
----------------
• Bidirectional LSTM captures both past (prefix) and future (suffix) context,
  which is critical for medical text where findings mentioned late in a report
  can disambiguate earlier observations.
• Sinusoidal positional encoding injects word-order information without
  learning extra parameters, keeping the encoder lightweight.
• Additive attention lets the model dynamically focus on symptom keywords
  (e.g. "bilateral GGOs", "elevated WBC") regardless of their position.
• The same architecture is used for both clinical observation text and
  radiology report text; they are encoded by separate model instances.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, BILSTM_HIDDEN, BILSTM_LAYERS,
    TEXT_OUT_DIM, MAX_SEQ_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters
    ----------
    d_model : embedding / model dimension
    max_len : maximum sequence length to pre-compute
    dropout : applied to the sum of embedding + positional encoding

    Input  : (B, T, d_model)
    Output : (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int   = MAX_SEQ_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # register as buffer so it moves with the model but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
# Additive (Bahdanau) Attention
# ─────────────────────────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    Collapses a variable-length sequence of hidden states (B, T, H)
    into a single context vector (B, H) by computing a weighted sum
    where weights reflect the importance of each time step.

    α = softmax(v · tanh(W · H^T))
    context = α · H

    Parameters
    ----------
    hidden_dim : dimension of each hidden state vector (H)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        hidden_states : (B, T, H)
        mask          : (B, T) – 1 for real tokens, 0 for padding (optional)

        Returns
        -------
        context : (B, H)   – weighted sum of hidden states
        weights : (B, T)   – attention weights
        """
        scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)                              # (B, T)
        context = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)    # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Text Encoder
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMTextEncoder(nn.Module):
    """
    Full text encoder: Embedding → PositionalEncoding → BiLSTM
                       → AdditiveAttention → FC projection.

    Parameters
    ----------
    vocab_size  : size of the vocabulary (including PAD=0, UNK=1)
    embed_dim   : word embedding dimension
    hidden_dim  : per-direction LSTM hidden size
                  (output size = 2 × hidden_dim due to bidirectionality)
    num_layers  : number of stacked BiLSTM layers
    out_dim     : final projected feature dimension (TEXT_OUT_DIM)
    max_seq_len : maximum token sequence length
    dropout     : dropout rate applied after FC and between LSTM layers

    Forward
    -------
    token_ids : (B, T)  →  (B, out_dim)
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        embed_dim:   int   = EMBED_DIM,
        hidden_dim:  int   = BILSTM_HIDDEN,
        num_layers:  int   = BILSTM_LAYERS,
        out_dim:     int   = TEXT_OUT_DIM,
        max_seq_len: int   = MAX_SEQ_LEN,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=0.1)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AdditiveAttention(hidden_dim * 2)

        self.fc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (B, T)  – integer token indices (0 = PAD)

        Returns
        -------
        (B, out_dim)  – text feature vector
        """
        mask = (token_ids != 0).long()        # (B, T)  — padding mask

        x = self.embedding(token_ids)         # (B, T, E)
        x = self.pos_enc(x)                   # (B, T, E) + positional info

        out, _ = self.bilstm(x)               # (B, T, 2*H)
        context, _ = self.attention(out, mask) # (B, 2*H)

        return self.fc_proj(context)           # (B, out_dim)
