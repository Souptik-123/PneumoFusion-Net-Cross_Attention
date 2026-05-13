"""
models/__init__.py  –  Public API for the PneumoFusion-Net model package

Importing from this package
---------------------------
    from models import PneumoFusionNet                    # full model
    from models import CNNImageEncoder                    # image sub-module
    from models import BiLSTMTextEncoder                  # text sub-module
    from models import MLPNumericalEncoder                # numerical sub-module
    from models import FeatureFusion                      # fusion layer
    from models import CrossAttentionTransformer          # transformer
    from models import ClassificationHead                 # cls head
    from models import GCSA, DepthwiseSeparableConv       # attention blocks
    from models import ChannelAttention, SpatialAttention # fine-grained blocks
    from models import ResidualBlock                      # MLP building block
    from models import PositionalEncoding, AdditiveAttention  # text helpers
    from models import CrossModalAttention                # single xattn layer

Module map
----------
    models/
    ├── __init__.py              ← you are here
    ├── attention.py             ← ChannelAttention, SpatialAttention,
    │                               GCSA, DepthwiseSeparableConv
    ├── cnn_encoder.py           ← CNNImageEncoder
    ├── text_encoder.py          ← PositionalEncoding, AdditiveAttention,
    │                               BiLSTMTextEncoder
    ├── numerical_encoder.py     ← ResidualBlock, MLPNumericalEncoder
    ├── fusion.py                ← FeatureFusion
    ├── transformer.py           ← CrossModalAttention, CrossAttentionTransformer
    ├── classification_head.py   ← ClassificationHead
    └── pneumofusion_net.py      ← PneumoFusionNet  (full assembled model)
"""

from models.attention           import (
    ChannelAttention,
    SpatialAttention,
    GCSA,
    DepthwiseSeparableConv,
)
from models.cnn_encoder         import CNNImageEncoder
from models.text_encoder        import (
    PositionalEncoding,
    AdditiveAttention,
    BiLSTMTextEncoder,
)
from models.numerical_encoder   import ResidualBlock, MLPNumericalEncoder
from models.fusion              import FeatureFusion
from models.transformer         import CrossModalAttention, CrossAttentionTransformer
from models.classification_head import ClassificationHead, MODALITY_NAMES
from models.pneumofusion_net    import PneumoFusionNet

__all__ = [
    # building blocks
    "ChannelAttention",
    "SpatialAttention",
    "GCSA",
    "DepthwiseSeparableConv",
    # encoders
    "CNNImageEncoder",
    "PositionalEncoding",
    "AdditiveAttention",
    "BiLSTMTextEncoder",
    "ResidualBlock",
    "MLPNumericalEncoder",
    # fusion
    "FeatureFusion",
    # transformer
    "CrossModalAttention",
    "CrossAttentionTransformer",
    # head
    "ClassificationHead",
    "MODALITY_NAMES",
    # full model
    "PneumoFusionNet",
]
