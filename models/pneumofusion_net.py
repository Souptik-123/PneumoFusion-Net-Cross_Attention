"""
models/pneumofusion_net.py  –  Full PneumoFusion-Net

Assembles all sub-modules into a single nn.Module:

    CNNImageEncoder        (models/cnn_encoder.py)
    BiLSTMTextEncoder      (models/text_encoder.py)
    MLPNumericalEncoder    (models/numerical_encoder.py)
    FeatureFusion          (models/fusion.py)           ← optional combined view
    CrossAttentionTransformer  (models/transformer.py)
    ClassificationHead     (models/classification_head.py)

Forward signature
-----------------
    image     : (B, 1, H, W)
    text_ids  : (B, MAX_SEQ_LEN)
    num_feats : (B, NUM_NUMERICAL_FEATURES)
    → logits  : (B, NUM_CLASSES)

Fine-tuning helpers
-------------------
    freeze_cnn_backbone()   – freeze ResNet50 stem/layer1/layer2
    unfreeze_all()          – enable all gradients
    get_parameter_groups()  – differential LRs for backbone vs head
"""

import torch
import torch.nn as nn

from config import (
    NUM_CLASSES, VOCAB_SIZE,
    CNN_OUT_DIM, TEXT_OUT_DIM, NUM_OUT_DIM, FUSION_DIM,
    XATTN_LAYERS, XATTN_HEADS, XATTN_FF_DIM, XATTN_DROPOUT,
    CLS_HIDDEN_DIM, NUM_NUMERICAL_FEATURES,
    IMAGE_SIZE, MAX_SEQ_LEN,
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

    Forward
    -------
    image     : (B, 1, H, W)              – grayscale CT image
    text_ids  : (B, MAX_SEQ_LEN)          – tokenised clinical text
    num_feats : (B, NUM_NUMERICAL_FEATURES) – standardised lab values

    Returns
    -------
    logits : (B, num_classes)  – raw (pre-softmax) classification scores
    """

    def __init__(
        self,
        num_classes:    int  = NUM_CLASSES,
        vocab_size:     int  = VOCAB_SIZE,
        pretrained_cnn: bool = True,
    ):
        super().__init__()

        # ── Per-modality encoders ─────────────────────────────────────────
        self.cnn_encoder  = CNNImageEncoder(
            out_dim=CNN_OUT_DIM, pretrained=pretrained_cnn
        )
        self.text_encoder = BiLSTMTextEncoder(
            vocab_size=vocab_size, out_dim=TEXT_OUT_DIM
        )
        self.num_encoder  = MLPNumericalEncoder(
            in_dim=NUM_NUMERICAL_FEATURES, out_dim=NUM_OUT_DIM
        )

        # ── Per-modality projections → shared FUSION_DIM space ───────────
        # These project each modality's feature vector into the shared
        # dimension that the Cross-Attention Transformer operates in.
        self.cnn_proj  = nn.Linear(CNN_OUT_DIM,  FUSION_DIM)
        self.text_proj = nn.Linear(TEXT_OUT_DIM, FUSION_DIM)
        self.num_proj  = nn.Linear(NUM_OUT_DIM,  FUSION_DIM)

        # ── Feature Fusion (combined representation, for reference) ───────
        # Used to produce a single fused vector (not fed to transformer here,
        # but available for auxiliary losses or future extensions).
        self.fusion = FeatureFusion(
            cnn_dim=CNN_OUT_DIM, text_dim=TEXT_OUT_DIM,
            num_dim=NUM_OUT_DIM, fusion_dim=FUSION_DIM,
        )

        # ── Cross-Attention Transformer ───────────────────────────────────
        self.transformer = CrossAttentionTransformer(
            n_layers=XATTN_LAYERS,
            d_model=FUSION_DIM,
            n_heads=XATTN_HEADS,
            ff_dim=XATTN_FF_DIM,
            dropout=XATTN_DROPOUT,
        )

        # ── Classification Head ───────────────────────────────────────────
        self.cls_head = ClassificationHead(
            d_model=FUSION_DIM,
            hidden_dim=CLS_HIDDEN_DIM,
            num_classes=num_classes,
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

        # 2. project each to FUSION_DIM (one token per modality)
        c = self.cnn_proj(cnn_feat)                # (B, FUSION_DIM)
        t = self.text_proj(text_feat)              # (B, FUSION_DIM)
        n = self.num_proj(num_feat)                # (B, FUSION_DIM)

        # 3. cross-attention transformer
        seq    = self.transformer(c, t, n)         # (B, 3, FUSION_DIM)

        # 4. classification head
        logits = self.cls_head(seq)                # (B, num_classes)
        return logits

    # ── Fine-tuning helpers ───────────────────────────────────────────────

    def freeze_cnn_backbone(self):
        """Freeze ResNet50 stem + layer1 + layer2; leave GCSA/DSC/head free."""
        self.cnn_encoder.freeze_backbone()

    def unfreeze_all(self):
        """Un-freeze all parameters (call before fine-tuning)."""
        for p in self.parameters():
            p.requires_grad = True

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_head:     float = 1e-3,
    ):
        """
        Return two parameter groups with different learning rates for AdamW.

        Groups
        ------
        backbone  : ResNet50 stem, layer1, layer2  → lr_backbone (lower)
        head      : everything else                → lr_head     (higher)

        Usage
        -----
        optimizer = AdamW(model.get_parameter_groups(1e-5, 1e-4), ...)
        """
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

    print(f"Output shape  : {out.shape}")           # (2, NUM_CLASSES)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params  : {total:,}")
    print(f"Trainable     : {trainable:,}")
