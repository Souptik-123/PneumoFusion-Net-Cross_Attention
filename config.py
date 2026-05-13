"""
config.py  –  Central configuration for PneumoFusion-Net
All hyper-parameters and paths are defined here so that
every other module imports from a single source of truth.
"""

import os
import torch

# ─────────────────────────────────────────────
# PATHS  (edit DATA_ROOT to where your images live)
# ─────────────────────────────────────────────
DATA_ROOT   = "."          # folder that contains the "images/" subdirectory
CSV_PATH    = "unified_dataset_new.csv"
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
IMAGE_SIZE    = 224          # resize both H and W to this
NUM_CLASSES   = 5           # Bacterial, Covid-19, Normal, Tuberculosis, Viral  → set 5 if Covid present
#CLASS_NAMES   = ["Bacterial Pneumonia", "Normal", "Tuberculosis", "Viral Pneumonia"]
# If your CSV also has Covid-19:
CLASS_NAMES = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
# NUM_CLASSES = 5

NUM_WORKERS   = 4

# ─────────────────────────────────────────────
# NUMERICAL FEATURES  (columns in CSV)
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
CATEGORICAL_COLS = ["Patient_Sex"]   # will be one-hot encoded → 2 dims
# total numerical feature dim = len(NUMERICAL_COLS) + 2
NUM_NUMERICAL_FEATURES = len(NUMERICAL_COLS) + 2   # 9

TEXT_COL  = "Clinical_Observation"
MAX_SEQ_LEN = 128        # max token length for BiLSTM
VOCAB_SIZE  = 5000       # built from training set
EMBED_DIM   = 100        # word embedding dimension

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
# --- CNN (ResNet50 + GCSA) ---
CNN_OUT_DIM     = 512    # projected CNN feature dimension

# --- BiLSTM text encoder ---
BILSTM_HIDDEN   = 256    # per-direction hidden size  → output = 2*256 = 512
BILSTM_LAYERS   = 2
TEXT_OUT_DIM    = 512    # projected text feature dimension

# --- MLP numerical encoder ---
MLP_HIDDEN_DIMS = [128, 64]
NUM_OUT_DIM     = 64     # projected numerical feature dimension

# --- Feature fusion (concat → linear projection) ---
FUSION_DIM = 256         # unified projection dim D fed into the transformer

# --- Cross-Attention Transformer ---
XATTN_HEADS      = 8
XATTN_FF_DIM     = 512
XATTN_LAYERS     = 2
XATTN_DROPOUT    = 0.1

# --- Classification head ---
CLS_HIDDEN_DIM  = 128

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
K_FOLDS        = 5
EPOCHS         = 80
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
LR_T0          = 10     # CosineAnnealingWarmRestarts T_0
LR_ETA_MIN     = 1e-6
EARLY_STOP_PAT = 10      # patience epochs
SEED           = 42
MIXED_PRECISION = False   # AMP  (fp16/fp32)

# fine-tuning phase (run after full training on best fold)
FINETUNE_EPOCHS = 20
FINETUNE_LR     = 1e-4
FINETUNE_UNFREEZE_LAYERS = ["layer3", "layer4", "gcsa", "fc_proj"]
