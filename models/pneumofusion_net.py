"""
models/pneumofusion_net.py  –  Full PneumoFusion-Net

Changes vs previous version
----------------------------
• CrossAttentionTransformer now receives max_drop_path from config so
  stochastic depth regularisation is active during training.
• DROPOUT_RATE from config is now threaded into all sub-encoders so
  the single config knob controls all dropout.
"""

import torch
import torch.nn as nn

from config import (
    NUM_CLASSES, VOCAB_SIZE,
    CNN_OUT_DIM, TEXT_OUT_DIM, NUM_OUT_DIM, FUSION_DIM,
    XATTN_LAYERS, XATTN_HEADS, XATTN_FF_DIM, XATTN_DROPOUT,
    CLS_HIDDEN_DIM, NUM_NUMERICAL_FEATURES,
    IMAGE_SIZE, MAX_SEQ_LEN, DROPOUT_RATE,
    MAX_DROP_PATH,
)
from models.cnn_encoder        import CNNImageEncoder
from models.text_encoder       import BiLSTMTextEncoder
from models.numerical_encoder  import MLPNumericalEncoder
from models.fusion             import FeatureFusion
from models.transformer        import CrossAttentionTransformer
from models.classification_head import ClassificationHead


class PneumoFusionNet(nn.Module):
    """
    Full PneumoFusion-Net: multimodal deep learning framework for
    pneumonia classification.

    Parameters
    ----------
    num_classes    : number of output classes
    vocab_size     : vocabulary size for the text encoder
    pretrained_cnn : load ImageNet-1K weights for ResNet50
    dropout        : dropout rate applied throughout all sub-modules
    """

    def __init__(
        self,
        num_classes:    int   = NUM_CLASSES,
        vocab_size:     int   = VOCAB_SIZE,
        pretrained_cnn: bool  = True,
        dropout:        float = DROPOUT_RATE,
    ):
        super().__init__()

        # ── Per-modality encoders ─────────────────────────────────────────
        self.cnn_encoder  = CNNImageEncoder(
            out_dim=CNN_OUT_DIM, pretrained=pretrained_cnn
        )
        self.text_encoder = BiLSTMTextEncoder(
            vocab_size=vocab_size, out_dim=TEXT_OUT_DIM, dropout=dropout
        )
        self.num_encoder  = MLPNumericalEncoder(
            in_dim=NUM_NUMERICAL_FEATURES, out_dim=NUM_OUT_DIM, dropout=dropout
        )

        # ── Per-modality projections → shared FUSION_DIM space ───────────
        self.cnn_proj  = nn.Linear(CNN_OUT_DIM,  FUSION_DIM)
        self.text_proj = nn.Linear(TEXT_OUT_DIM, FUSION_DIM)
        self.num_proj  = nn.Linear(NUM_OUT_DIM,  FUSION_DIM)

        # ── Feature Fusion (combined representation) ──────────────────────
        self.fusion = FeatureFusion(
            cnn_dim=CNN_OUT_DIM, text_dim=TEXT_OUT_DIM,
            num_dim=NUM_OUT_DIM, fusion_dim=FUSION_DIM,
            dropout=dropout,
        )

        # ── Cross-Attention Transformer ───────────────────────────────────
        # max_drop_path enables stochastic depth regularisation
        self.transformer = CrossAttentionTransformer(
            n_layers=XATTN_LAYERS,
            d_model=FUSION_DIM,
            n_heads=XATTN_HEADS,
            ff_dim=XATTN_FF_DIM,
            dropout=XATTN_DROPOUT,
            max_drop_path=MAX_DROP_PATH,
        )

        # ── Classification Head ───────────────────────────────────────────
        self.cls_head = ClassificationHead(
            d_model=FUSION_DIM,
            hidden_dim=CLS_HIDDEN_DIM,
            num_classes=num_classes,
            dropout=dropout,
        )

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        image:     torch.Tensor,
        text_ids:  torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        # 1. encode each modality independently
        cnn_feat  = self.cnn_encoder(image)        # (B, CNN_OUT_DIM)
        text_feat = self.text_encoder(text_ids)    # (B, TEXT_OUT_DIM)
        num_feat  = self.num_encoder(num_feats)    # (B, NUM_OUT_DIM)

        # 2. project each to FUSION_DIM
        c = self.cnn_proj(cnn_feat)                # (B, FUSION_DIM)
        t = self.text_proj(text_feat)              # (B, FUSION_DIM)
        n = self.num_proj(num_feat)                # (B, FUSION_DIM)

        # 3. cross-attention transformer  → always (B, 3, FUSION_DIM)
        seq    = self.transformer(c, t, n)

        # 4. classification head
        logits = self.cls_head(seq)                # (B, num_classes)
        return logits

    # ── Fine-tuning helpers ───────────────────────────────────────────────

    def freeze_cnn_backbone(self):
        self.cnn_encoder.freeze_backbone()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_parameter_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-3):
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if "cnn_encoder.stem"   in name or \
               "cnn_encoder.layer1" in name or \
               "cnn_encoder.layer2" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check  –  python models/pneumofusion_net.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    B = 2
    model = PneumoFusionNet(pretrained_cnn=False)
    model.eval()

    with torch.no_grad():
        imgs     = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE)
        tok_ids  = torch.randint(0, VOCAB_SIZE, (B, MAX_SEQ_LEN))
        num_feat = torch.randn(B, NUM_NUMERICAL_FEATURES)
        out      = model(imgs, tok_ids, num_feat)

    print(f"Output shape  : {out.shape}")   # must be (2, NUM_CLASSES)
    assert out.shape == (B, NUM_CLASSES), f"Shape error: {out.shape}"
    print("Shape check passed ✓")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params  : {total:,}")
    print(f"Trainable     : {trainable:,}")
