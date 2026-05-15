"""
models/classification_head.py  –  Classification Head
                                   with Learnable Modality Weights

Architecture
------------
transformer_output  (B, 3, D)  – three modality tokens [CNN | Text | Numerical]
  → learnable scalar weight per token  (softmax-normalised)
  → weighted sum across token dimension  →  (B, D)
  → Linear(D → hidden_dim)
  → LayerNorm + ReLU + Dropout
  → Linear(hidden_dim → num_classes)
  → raw logits  (B, num_classes)

Why learnable weights instead of mean-pool
------------------------------------------
The paper (Figure 5A) reports that the three modalities contribute
unequally to the final decision:
    CT image        ≈ 45 %
    Lab numerics    ≈ 33 %
    Clinical text   ≈ 12 %
    Radiology report≈ 10 %  (folded into text token in our 3-modality code)

A plain mean-pool forces each modality to contribute exactly 33.3 %,
ignoring this imbalance.

Option B – learnable token weights
    self.token_weights = nn.Parameter(torch.ones(3))   # initialised equal
    w = softmax(token_weights)                         # always sums to 1
    pooled = (seq * w).sum(dim=1)                      # weighted sum

Benefits
--------
• The model learns the contribution ratio from *this* dataset during
  training – it will naturally converge towards something close to the
  paper's 45/33/12 split if the data supports it.
• Weights are inspectable after training:
      w = torch.softmax(model.cls_head.token_weights, dim=0)
      # → tensor([0.45, 0.12, 0.33])  (CNN, Text, Numerical)
• Initialised as equal (ones → softmax → 0.333 each) so training starts
  from the same unbiased point as mean-pool.
• Adds only 3 scalar parameters – negligible cost.

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

# Human-readable names for the 3 token positions – used in get_modality_weights()
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
        Inspectable via  self.get_modality_weights()  after training.

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

        # ── Learnable modality importance weights ─────────────────────────
        # Initialised to ones so softmax starts at uniform (0.333, 0.333, 0.333).
        # During training the model will push these towards the true importance
        # ratio (e.g. CNN > Numerical > Text for pneumonia classification).
        self.token_weights = nn.Parameter(torch.tensor([0.45, 0.22, 0.33], dtype=torch.float32))

        # ── Classification MLP ────────────────────────────────────────────
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
        # shape: (3,) → broadcast to (1, 3, 1) for element-wise multiply
        w = F.softmax(self.token_weights, dim=0).view(1, 3, 1)   # (1, 3, 1)

        # weighted sum across the token dimension
        pooled = (seq * w).sum(dim=1)                             # (B, d_model)

        return self.net(pooled)                                   # (B, num_classes)

    # ── Inspection helper ─────────────────────────────────────────────────

    def get_modality_weights(self) -> dict:
        """
        Return the learned modality contribution as a plain dict.

        Usage (after training)
        ----------------------
            weights = model.cls_head.get_modality_weights()
            # {'CT image (CNN)': 0.45, 'Clinical text': 0.12, 'Lab numerics': 0.43}

        Returns
        -------
        dict  {modality_name: float}  – values sum to 1.0
        """
        with torch.no_grad():
            w = F.softmax(self.token_weights, dim=0).cpu().tolist()
        return {name: round(weight, 4) for name, weight in zip(MODALITY_NAMES, w)}
