"""
models/classification_head.py  –  Classification Head
                                   with Learnable Modality Weights

DATA-LEAKAGE FIX
================
FIX 3 — token_weights initialised with paper-derived non-uniform priors [MEDIUM]
---------------------------------------------------------------------------------
Previous bug: token_weights was initialised as:
    nn.Parameter(torch.tensor([0.45, 0.22, 0.33], dtype=torch.float32))

These values came directly from the paper's Figure 5A result
(CT ≈ 45%, Lab ≈ 33%, Text ≈ 22%).  Baking them in as the starting
point means:
  1. The model begins training with an implicit prior injected from
     prior-knowledge / held-out results, not from the data itself.
  2. If the class balance or modality contributions differ in YOUR
     dataset, the biased init slows convergence or produces a local
     minimum close to the paper's regime rather than the data's truth.
  3. It constitutes a subtle form of information leakage from the
     paper's test set into the model's parameter initialisation.

Fix: token_weights initialised to ones (→ uniform softmax = 0.333 each).
The model now learns the contribution ratio entirely from the training
data, with no external prior baked in.  This is the standard approach
and adds only 3 scalar parameters of overhead.

Architecture
------------
transformer_output  (B, 3, D)  – three modality tokens [CNN | Text | Numerical]
  → learnable scalar weight per token  (softmax-normalised)
  → weighted sum across token dimension  →  (B, D)
  → Linear(D → hidden_dim)
  → LayerNorm + ReLU + Dropout
  → Linear(hidden_dim → num_classes)
  → raw logits  (B, num_classes)

Token order convention (matches transformer.py output)
-------------------------------------------------------
    seq[:, 0, :]  →  CNN   image token
    seq[:, 1, :]  →  Text  token
    seq[:, 2, :]  →  Numerical token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FUSION_DIM, CLS_HIDDEN_DIM, NUM_CLASSES

MODALITY_NAMES = ["CT image (CNN)", "Clinical text", "Lab numerics"]


class ClassificationHead(nn.Module):
    """
    Learnable-weighted pool of the transformer's 3-token output, then classify.

    Parameters
    ----------
    d_model     : dimension of each token (FUSION_DIM)
    hidden_dim  : intermediate FC hidden dimension
    num_classes : number of output classes
    dropout     : dropout probability applied before final linear

    Attributes
    ----------
    token_weights : nn.Parameter  shape (3,)
        Raw (pre-softmax) scalar importance score for each modality token.
        Initialised to ones → uniform 0.333 each.
        Inspectable via self.get_modality_weights() after training.

    Forward
    -------
    seq : (B, 3, d_model)  →  logits : (B, num_classes)
    """

    def __init__(
        self,
        d_model:     int   = FUSION_DIM,
        hidden_dim:  int   = CLS_HIDDEN_DIM,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # FIX 3: initialised to ones (uniform) — no paper-derived prior baked in.
        # After training, these will naturally converge toward the true
        # contribution ratio for this dataset.
        self.token_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))

        # Classification MLP
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : (B, 3, d_model)
              Axis-1 order: [CNN token | Text token | Numerical token]

        Returns
        -------
        logits : (B, num_classes)  – raw (pre-softmax) classification scores
        """
        # softmax over the 3 raw scalars → weights that sum to 1
        w = F.softmax(self.token_weights, dim=0).view(1, 3, 1)   # (1, 3, 1)

        # weighted sum across the token dimension
        pooled = (seq * w).sum(dim=1)                             # (B, d_model)

        return self.net(pooled)                                   # (B, num_classes)

    def get_modality_weights(self) -> dict:
        """
        Return the learned modality contribution as a plain dict.

        Usage (after training)
        ----------------------
            weights = model.cls_head.get_modality_weights()
            # e.g. {'CT image (CNN)': 0.45, 'Clinical text': 0.12, 'Lab numerics': 0.43}

        Returns
        -------
        dict  {modality_name: float}  – values sum to 1.0
        """
        with torch.no_grad():
            w = F.softmax(self.token_weights, dim=0).cpu().tolist()
        return {name: round(weight, 4) for name, weight in zip(MODALITY_NAMES, w)}
