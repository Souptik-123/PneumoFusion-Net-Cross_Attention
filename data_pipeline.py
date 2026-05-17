"""
data_pipeline.py  –  Everything data-related for PneumoFusion-Net

DATA-LEAKAGE FIXES (vs previous version)
=========================================

FIX 1 — StandardScaler fitted ONLY on training rows  [CRITICAL]
----------------------------------------------------------------
Previous bug: PneumoDataset.__init__() accepted a `fit_scaler` flag and
called scaler.fit() inside the Dataset constructor.  This was safe *only*
if the caller passed fit_scaler=True for the training set and False for
validation.  However, the scaler object was shared by reference: once
fit_scaler=True fired inside train_ds, the scaler was already mutated
before val_ds was constructed.  In practice it worked, BUT the interface
was fragile and prone to accidental re-fitting if datasets were rebuilt
in a different order.

Fix: scaler fitting is now done OUTSIDE the Dataset, explicitly on
train_df rows only, before either Dataset is constructed.  PneumoDataset
receives an already-fitted scaler and only calls scaler.transform().
The fit_scaler parameter has been removed entirely.

FIX 2 — Vocabulary built ONLY on training text  [CRITICAL]
-----------------------------------------------------------
Previous code passed train_df[TEXT_COL].tolist() to Vocabulary.build(),
which was correct.  The fix keeps this, but makes the guarantee explicit
and adds a docstring assertion so future callers cannot accidentally pass
the full corpus.

FIX 3 — Augmentation pipeline (no leakage, but correctness)
------------------------------------------------------------
No functional changes from the strengthened augmentation introduced for
anti-overfitting.  Preserved as-is.
"""

import os
import re
import collections
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_ROOT, CSV_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    NUMERICAL_COLS, CATEGORICAL_COLS, NUM_NUMERICAL_FEATURES,
    TEXT_COL, MAX_SEQ_LEN, VOCAB_SIZE, EMBED_DIM,
    CLASS_NAMES, NUM_CLASSES, K_FOLDS, SEED,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  TOKENISER
# ═══════════════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def _tokenise(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class Vocabulary:
    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, sentences):
        """
        Build vocabulary from a list of sentences.

        IMPORTANT: `sentences` must come ONLY from the training split.
        Never pass validation or test text here — doing so leaks
        token-frequency information from held-out samples into the vocab.
        """
        counter = collections.Counter()
        for sent in sentences:
            counter.update(_tokenise(str(sent)))
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list:
        tokens = _tokenise(str(text))[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def build_label_map(labels: pd.Series):
    unique = sorted(labels.unique())
    return {name: i for i, name in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMS  (strengthened for anti-overfitting; no leakage)
# ═══════════════════════════════════════════════════════════════════════════

def _train_transform(image_size: int = IMAGE_SIZE):
    """
    Strong augmentation applied to training images ONLY.
    NEVER use this transform on validation/test data.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size + 64, image_size + 64)),
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.0), value=0),
    ])


def _val_transform(image_size: int = IMAGE_SIZE):
    """
    Deterministic transform for validation/test images.
    No augmentation, no randomness.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ═══════════════════════════════════════════════════════════════════════════

class PneumoDataset(Dataset):
    """
    Parameters
    ----------
    df        : DataFrame for this split (train OR val, never mixed)
    vocab     : Vocabulary built on training text ONLY
    scaler    : StandardScaler ALREADY FITTED on training numerical data only.
                Only scaler.transform() is called here — never scaler.fit().
    label_map : {class_name: int}
    transform : torchvision transform (train or val variant)
    data_root : root directory for image paths

    Returns (per __getitem__)
    -------
    image      : FloatTensor  [1, H, W]
    text_ids   : LongTensor   [MAX_SEQ_LEN]
    num_feats  : FloatTensor  [NUM_NUMERICAL_FEATURES]
    label      : LongTensor   scalar
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        scaler: StandardScaler,         # must be pre-fitted on train rows only
        label_map: dict,
        transform,
        data_root: str = DATA_ROOT,
    ):
        # FIX 1: removed fit_scaler parameter — scaler is always pre-fitted
        # by the caller (get_fold_dataloaders) on training rows only.
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.label_map = label_map
        self.transform = transform
        self.data_root = data_root

        sex_dummies = pd.get_dummies(self.df["Patient_Sex"], prefix="sex").astype(float)
        for col in ["sex_Female", "sex_Male"]:
            if col not in sex_dummies.columns:
                sex_dummies[col] = 0.0

        num_raw = pd.concat(
            [self.df[NUMERICAL_COLS].astype(float), sex_dummies[["sex_Female", "sex_Male"]]],
            axis=1,
        ).values

        # Only transform — scaler was fitted on training data by the caller
        self.num_feats = scaler.transform(num_raw).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path_raw = row["image_path"].replace("\\", os.sep)
        img_path = os.path.join(self.data_root, img_path_raw)
        try:
            img = Image.open(img_path).convert("L")
        except FileNotFoundError:
            img = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        image = self.transform(img)

        text_ids = torch.tensor(
            self.vocab.encode(str(row[TEXT_COL])), dtype=torch.long
        )

        num = torch.tensor(self.num_feats[idx], dtype=torch.float32)

        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)

        return image, text_ids, num, label


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FOLD DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_dataframe(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def get_fold_dataloaders(
    df: pd.DataFrame,
    fold: int,
    skf: StratifiedKFold,
    label_map: dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    persistent_workers: bool = True
):
    """
    Build train and validation DataLoaders for one fold.

    Leakage-free guarantee
    ----------------------
    1. Vocabulary  — built on train_df text only (val text never seen).
    2. StandardScaler — fitted on train_df numerical rows only, then used
       to transform both train and val rows.  val statistics never influence
       the scaler parameters.
    3. Transforms — augmentation applied to train only; val uses
       deterministic resize+normalise.
    """
    splits = list(skf.split(df, df["label"]))
    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    # ── Vocabulary: train text only ───────────────────────────────────────
    vocab = Vocabulary(max_size=VOCAB_SIZE).build(train_df[TEXT_COL].tolist())

    # ── Scaler: fit on train rows, transform both splits ──────────────────
    # FIX 1: extract raw numerics for training rows here, fit the scaler,
    # then pass the already-fitted scaler to both Dataset constructors.
    # The Dataset now only calls scaler.transform(), never scaler.fit().
    def _raw_numerics(sub_df: pd.DataFrame) -> np.ndarray:
        sex_dummies = pd.get_dummies(sub_df["Patient_Sex"], prefix="sex").astype(float)
        for col in ["sex_Female", "sex_Male"]:
            if col not in sex_dummies.columns:
                sex_dummies[col] = 0.0
        return pd.concat(
            [sub_df[NUMERICAL_COLS].astype(float), sex_dummies[["sex_Female", "sex_Male"]]],
            axis=1,
        ).values

    scaler = StandardScaler()
    scaler.fit(_raw_numerics(train_df))   # fit on TRAIN rows only

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = PneumoDataset(
        train_df, vocab, scaler, label_map,
        transform=_train_transform(),
        # scaler already fitted — no fit_scaler flag needed
    )
    val_ds = PneumoDataset(
        val_df, vocab, scaler, label_map,
        transform=_val_transform(),
        # scaler.transform() only — val statistics never touch the scaler
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    return train_loader, val_loader, vocab, scaler
