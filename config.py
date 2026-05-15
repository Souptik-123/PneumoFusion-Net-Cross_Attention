"""
config.py  –  Central configuration for PneumoFusion-Net
All hyper-parameters and paths are defined here so that
every other module imports from a single source of truth.

OVERFITTING FIX SUMMARY (val acc ~100% from epoch 2)
=====================================================
Root causes diagnosed from the Fold-5 curves:
  1. Val accuracy hits ~99% by epoch 2 → model memorises the small dataset
     (~5 600 samples split 80/20 ≈ 4 480 train / 1 120 val per fold).
  2. Train loss stays ~0.40 while val loss is ~0.39 → near-perfect fit,
     no generalisation gap at epoch 26 but the gap could grow later.

Fixes applied (each one independently reduces overfitting):

A. DROPOUT_RATE 0.3 → 0.5
   Applied in all encoders + classification head.
   Prevents co-adaptation of neurons that causes fast memorisation.

B. WEIGHT_DECAY 1e-4 → 1e-3
   Stronger L2 regularisation; penalises large weight norms harder.

C. LABEL_SMOOTHING 0.1 → 0.25
   Prevents the model becoming over-confident (soft targets discourage
   the logit gap from growing without bound).

D. MIXUP_ALPHA 0.3 → 0.4
   Slightly stronger mixup; combined with label smoothing it makes
   the decision boundary smoother.

E. EARLY_STOP_PAT 12 → 8
   Stop sooner once val loss plateaus; the curves show no improvement
   after epoch 5–6 in any fold.

F. WARMUP_EPOCHS 5 → 3
   With only ~140 training batches per fold, 5 warmup epochs is too
   slow; 3 epochs still stabilises training without wasting capacity.

G. LEARNING_RATE 2e-4 → 1e-4
   Lower base LR → slower convergence → less risk of jumping into a
   memorisation regime in the first few epochs.

H. STOCHASTIC DEPTH (max_drop_path=0.10) added to CrossAttentionTransformer
   Randomly zeroes entire residual branches; very effective for small
   datasets in transformer-based models (see transformer.py).

I. AUGMENTATION (data_pipeline.py)
   Random Erasing probability raised 0.3 → 0.5.
   RandomAffine shear range widened.
   These are already in data_pipeline.py; no change needed here.
"""

import os
import torch

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_ROOT   = "."
CSV_PATH    = "unified_dataset_new1.csv"
OUTPUT_DIR  = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")

for _d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
IMAGE_SIZE    = 224
CLASS_NAMES = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
NUM_CLASSES = 5
NUM_WORKERS   = 4

# ─────────────────────────────────────────────
# NUMERICAL FEATURES
# ─────────────────────────────────────────────
NUMERICAL_COLS = [
    "Patient_Age",
    "WBC (x10^9/L)",
    "NEUT%",
    "LYMP%",
    "NLR",
    "CRP (mg/L)",
    "PCT (ng/mL)",
]
CATEGORICAL_COLS = ["Patient_Sex"]
NUM_NUMERICAL_FEATURES = len(NUMERICAL_COLS) + 2   # 9

TEXT_COL    = "Clinical_Observation"
MAX_SEQ_LEN = 128
VOCAB_SIZE  = 5000
EMBED_DIM   = 100

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
CNN_OUT_DIM     = 512
BILSTM_HIDDEN   = 256
BILSTM_LAYERS   = 2
TEXT_OUT_DIM    = 512
MLP_HIDDEN_DIMS = [128, 64]
NUM_OUT_DIM     = 64
FUSION_DIM      = 256

XATTN_HEADS      = 8
XATTN_FF_DIM     = 512
XATTN_LAYERS     = 2
XATTN_DROPOUT    = 0.15          # slightly higher than before (was 0.1)
MAX_DROP_PATH    = 0.10          # stochastic depth for transformer layers

CLS_HIDDEN_DIM  = 128

# ─────────────────────────────────────────────
# TRAINING  (overfitting-corrected values)
# ─────────────────────────────────────────────
K_FOLDS        = 5
EPOCHS         = 80
BATCH_SIZE     = 32

# Fix A – lower LR (less aggressive memorisation in early epochs)
LEARNING_RATE  = 1e-4            # was 2e-4

# Fix B – stronger weight decay
WEIGHT_DECAY   = 1e-3            # was 5e-4

LR_ETA_MIN     = 1e-6

# Fix F – shorter warmup (dataset too small for 5-epoch warmup)
WARMUP_EPOCHS  = 3               # was 5

GRAD_CLIP_NORM = 0.5

# Fix E – shorter patience (curves plateau fast)
EARLY_STOP_PAT = 8               # was 12

# Fix C – stronger label smoothing
LABEL_SMOOTHING = 0.25           # was 0.2

# Fix A – higher dropout everywhere
DROPOUT_RATE   = 0.5             # was 0.3

# Fix D – slightly stronger mixup
MIXUP_ALPHA    = 0.4             # was 0.3

SEED           = 42
MIXED_PRECISION = False

# Fine-tuning phase
FINETUNE_EPOCHS = 20
FINETUNE_LR     = 2e-5           # was 3e-5, lowered to match stronger regularisation
FINETUNE_UNFREEZE_LAYERS = ["layer3", "layer4", "gcsa", "fc_proj"]
