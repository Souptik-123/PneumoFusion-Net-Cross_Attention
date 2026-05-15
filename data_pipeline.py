"""
data_pipeline.py  –  Everything data-related for PneumoFusion-Net

OVERFITTING FIX: augmentation pipeline strengthened
====================================================
Changes vs previous version:
  • RandomErasing probability: 0.3 → 0.5  (larger, more aggressive)
  • RandomErasing scale: (0.02, 0.12) → (0.02, 0.20)  (bigger erased patches)
  • RandomAffine shear: 5 → 10 degrees
  • Added GridDistortion-style RandomPerspective(p=0.2) for structural diversity
  • Increased RandomRotation: 15 → 20 degrees
  • ColorJitter brightness/contrast: 0.3 → 0.4
  All other logic unchanged.
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
# 3.  TRANSFORMS  (strengthened for anti-overfitting)
# ═══════════════════════════════════════════════════════════════════════════

def _train_transform(image_size: int = IMAGE_SIZE):
    """
    Strong augmentation to combat overfitting on a small dataset (~5 600 samples).

    Key anti-overfitting additions vs original:
      • RandomErasing p=0.5, scale up to 0.20 (was p=0.3, scale 0.12)
      • RandomPerspective p=0.2 (new) – structural distortion diversity
      • RandomRotation 20° (was 15°)
      • ColorJitter 0.4 (was 0.3)
      • RandomAffine shear=10 (was 5)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size + 64, image_size + 64)),
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),                              # ↑ was 15
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=10),  # ↑ shear
        transforms.ColorJitter(brightness=0.4, contrast=0.4),      # ↑ was 0.3
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.2),  # NEW
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # ↑ stronger RandomErasing: p=0.5, larger scale
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.0), value=0),
    ])


def _val_transform(image_size: int = IMAGE_SIZE):
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
    Returns
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
        scaler: StandardScaler,
        label_map: dict,
        transform,
        data_root: str = DATA_ROOT,
        fit_scaler: bool = False,
    ):
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

        if fit_scaler:
            scaler.fit(num_raw)
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
):
    splits = list(skf.split(df, df["label"]))
    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    vocab = Vocabulary(max_size=VOCAB_SIZE).build(train_df[TEXT_COL].tolist())

    scaler = StandardScaler()

    train_ds = PneumoDataset(
        train_df, vocab, scaler, label_map,
        transform=_train_transform(),
        fit_scaler=True,
    )
    val_ds = PneumoDataset(
        val_df, vocab, scaler, label_map,
        transform=_val_transform(),
        fit_scaler=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, vocab, scaler
