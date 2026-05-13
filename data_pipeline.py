"""
data_pipeline.py  –  Everything data-related for PneumoFusion-Net

Responsibilities
----------------
1.  Load the CSV and resolve image paths.
2.  Build a vocabulary from the clinical-observation text (training set only).
3.  PneumoDataset  – returns (image_tensor, text_ids, num_features, label).
4.  get_fold_dataloaders  – returns train / val DataLoaders for a given fold.
5.  Data augmentation: strong for training, deterministic for validation.
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
# 1.  TOKENISER  (simple whitespace + punctuation tokeniser, no dependencies)
# ═══════════════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def _tokenise(text: str):
    """Lower-case and split on non-alphanumeric characters."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class Vocabulary:
    """Build word→index mapping from a list of sentences (training text only)."""

    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, sentences):
        counter = collections.Counter()
        for sent in sentences:
            counter.update(_tokenise(str(sent)))
        # keep only the top (max_size - 2) words  (0, 1 are reserved)
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list:
        tokens = _tokenise(str(text))[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # pad / truncate to max_len
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def build_label_map(labels: pd.Series):
    """Return dict {label_string: integer_index} sorted alphabetically."""
    unique = sorted(labels.unique())
    return {name: i for i, name in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════

def _train_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
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

        # ── numerical features ──────────────────────────────────────────
        # one-hot encode Patient_Sex
        sex_dummies = pd.get_dummies(self.df["Patient_Sex"], prefix="sex").astype(float)
        # ensure both columns exist even if only one sex in a fold
        for col in ["sex_Female", "sex_Male"]:
            if col not in sex_dummies.columns:
                sex_dummies[col] = 0.0

        num_raw = pd.concat(
            [self.df[NUMERICAL_COLS].astype(float), sex_dummies[["sex_Female", "sex_Male"]]],
            axis=1,
        ).values  # (N, NUM_NUMERICAL_FEATURES)

        if fit_scaler:
            scaler.fit(num_raw)
        self.num_feats = scaler.transform(num_raw).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── image ────────────────────────────────────────────────────────
        img_path_raw = row["image_path"].replace("\\", os.sep)
        img_path = os.path.join(self.data_root, img_path_raw)
        try:
            img = Image.open(img_path).convert("L")   # grayscale
        except FileNotFoundError:
            print(f"Warning: image not found at {img_path}. Returning blank image.")
            # return a blank image so training can continue
            img = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        image = self.transform(img)

        # ── text ─────────────────────────────────────────────────────────
        text_ids = torch.tensor(
            self.vocab.encode(str(row[TEXT_COL])), dtype=torch.long
        )

        # ── numerical ────────────────────────────────────────────────────
        num = torch.tensor(self.num_feats[idx], dtype=torch.float32)

        # ── label ────────────────────────────────────────────────────────
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
    """
    Build train/val DataLoaders for a single fold.

    Parameters
    ----------
    df         : full dataframe
    fold       : 0-based fold index
    skf        : pre-built StratifiedKFold object
    label_map  : {label_string: int}

    Returns
    -------
    train_loader, val_loader, vocab, scaler
    """
    splits = list(skf.split(df, df["label"]))
    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    # build vocab from TRAINING text only
    vocab = Vocabulary(max_size=VOCAB_SIZE).build(train_df[TEXT_COL].tolist())

    # build scaler from TRAINING numerics only
    scaler = StandardScaler()

    train_ds = PneumoDataset(
        train_df, vocab, scaler, label_map,
        transform=_train_transform(),
        fit_scaler=True,   # fits scaler on this split
    )
    val_ds = PneumoDataset(
        val_df, vocab, scaler, label_map,
        transform=_val_transform(),
        fit_scaler=False,  # reuse fitted scaler
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
