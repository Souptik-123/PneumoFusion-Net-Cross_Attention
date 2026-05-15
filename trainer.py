"""
trainer.py  –  Training engine for PneumoFusion-Net

Key fixes vs original (which produced loss spikes at epochs 4 & 11)
---------------------------------------------------------------------
1. WARMUP + COSINE  –  replaced CosineAnnealingWarmRestarts (hard LR resets)
   with linear warmup for WARMUP_EPOCHS + single-cycle CosineAnnealingLR.

2. LOWER LR  –  base LR dropped 1e-3 -> 3e-4 (safer for pretrained ResNet50).

3. TIGHTER GRAD CLIP  –  max_norm 1.0 -> 0.5.

4. LONGER PATIENCE  –  early stopping patience 10 -> 15.

5. BEST-FOLD-ONLY FINETUNE  –  finetune_fold() called only for the best fold.

6. EXPORT  –  export_vocab_scaler() saves tokenizer.json + scaler.pkl.

Provides
--------
WarmupCosineScheduler  / train_one_epoch  / validate  / EarlyStopping
train_fold  /  finetune_fold  /  export_vocab_scaler
"""

import os, time, json, pickle, math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_DIR,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_ETA_MIN,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, EARLY_STOP_PAT,
    LABEL_SMOOTHING, MIXUP_ALPHA,
    MIXED_PRECISION,
    FINETUNE_EPOCHS, FINETUNE_LR,
)


# ═══════════════════════════════════════════════════════════════════════════
# WARMUP + COSINE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for warmup_epochs, then cosine decay to eta_min.

    Epoch 0 .. warmup_epochs-1  : LR ramps linearly  LR/10 -> base_LR
    Epoch warmup_epochs .. T_max: CosineAnnealing     base_LR -> eta_min

    Eliminates the mid-training LR resets that caused the observed spikes.
    """

    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=LR_ETA_MIN, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = 0.1 + 0.9 * (e / max(self.warmup_epochs - 1, 1))
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (e - self.warmup_epochs) / max(self.T_max - self.warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m    = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            pass
    return m


# ═══════════════════════════════════════════════════════════════════════════
# MIXUP AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def mixup_batch(images, labels, num_classes, alpha=MIXUP_ALPHA):
    """
    Mixup data augmentation (Zhang et al., 2018).

    Interpolates pairs of training samples and their one-hot labels:
        x_mix = λ·x_i + (1-λ)·x_j
        y_mix = λ·y_i + (1-λ)·y_j   (soft labels)

    This directly addresses the overfitting observed at epoch 2:
    the model can no longer memorise exact training samples because
    every batch contains convex combinations it has never seen before.

    Parameters
    ----------
    images     : (B, C, H, W) image tensor
    labels     : (B,) integer class labels
    num_classes: total number of classes
    alpha      : Beta distribution parameter (0 = no mixup)

    Returns
    -------
    mixed_images : (B, C, H, W)
    mixed_labels : (B, num_classes) soft one-hot labels for use with
                   F.cross_entropy(logits, mixed_labels) where mixed_labels
                   is passed as a float tensor (not integer indices).
    """
    # Guard: catch label index OOB on CPU before it hits the CUDA scatter kernel
    # (CUDA assertion aborts the process with no recoverable traceback on Windows)
    assert labels.max().item() < num_classes, (
        f"mixup_batch: label index {labels.max().item()} >= num_classes {num_classes}. "
        f"Check label_map alignment with the dataset."
    )
    assert labels.min().item() >= 0, (
        f"mixup_batch: negative label index {labels.min().item()} found."
    )

    if alpha <= 0:
        # no mixup: return images + hard one-hot labels
        one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return images, one_hot

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)           # always >= 0.5 so prediction is unambiguous

    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)

    mixed = lam * images + (1.0 - lam) * images[idx]

    # soft one-hot labels
    one_hot   = torch.zeros(B, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    mixed_lbl = lam * one_hot + (1.0 - lam) * one_hot[idx]

    return mixed, mixed_lbl


# ═══════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PAT, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.counter = 0; self.best_val = None; self.stop = False

    def __call__(self, val_loss):
        if self.best_val is None or val_loss < self.best_val - self.delta:
            self.best_val = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE EPOCH
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, num_classes):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []
    t0 = time.time()

    for idx, (images, text_ids, num_feats, labels) in enumerate(loader):
        images    = images.to(DEVICE, non_blocking=True)
        text_ids  = text_ids.to(DEVICE, non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)

        # ── Mixup augmentation ────────────────────────────────────────────
        images, soft_labels = mixup_batch(images, labels, num_classes)

        optimizer.zero_grad(set_to_none=True)

        if MIXED_PRECISION and DEVICE.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images, text_ids, num_feats)
                # soft-label cross-entropy: sum(soft_labels * log_softmax(logits))
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer); scaler.update()
        else:
            logits    = model(images, text_ids, num_feats)
            log_probs = F.log_softmax(logits, dim=1)
            loss      = -(soft_labels * log_probs).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())

        if (idx + 1) % max(1, len(loader) // 5) == 0:
            print(f"  [Ep {epoch}] step {idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  t={time.time()-t0:.1f}s")

    return total_loss / len(loader), compute_metrics(np.array(all_labels), np.array(all_preds))


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, text_ids, num_feats, labels in loader:
        images    = images.to(DEVICE, non_blocking=True)
        text_ids  = text_ids.to(DEVICE, non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)

        if MIXED_PRECISION and DEVICE.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images, text_ids, num_feats)
                loss   = criterion(logits, labels)
        else:
            logits = model(images, text_ids, num_feats)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (total_loss / len(loader),
            compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs)))


# ═══════════════════════════════════════════════════════════════════════════
# FULL FOLD TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, fold,
               epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
               label_smoothing=LABEL_SMOOTHING):
    """Train one fold.  Returns best_metrics, history."""
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # not used directly with mixup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, epochs, LR_ETA_MIN)
    scaler    = torch.amp.GradScaler('cuda')
    es        = EarlyStopping(EARLY_STOP_PAT)

    best_val_loss = float("inf"); best_metrics = {}; history = []
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pt")

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}  |  device={DEVICE}  lr={lr:.1e}  "
          f"warmup={WARMUP_EPOCHS}ep  clip={GRAD_CLIP_NORM}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m  = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, model.cls_head.net[-1].out_features)
        val_loss, val_m = validate(model, val_loader, criterion)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        phase  = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"

        history.append(dict(epoch=epoch, lr=cur_lr, phase=phase,
                            train_loss=tr_loss, train_acc=tr_m["accuracy"], train_f1=tr_m["f1"],
                            val_loss=val_loss,   val_acc=val_m["accuracy"], val_f1=val_m["f1"],
                            val_auc=val_m.get("roc_auc", 0.0)))

        print(f"Ep [{epoch:3d}/{epochs}][{phase}]  lr={cur_lr:.2e}  "
              f"tr={tr_loss:.4f}/{tr_m['accuracy']*100:.1f}%  "
              f"val={val_loss:.4f}/{val_m['accuracy']*100:.1f}%  "
              f"f1={val_m['f1']*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_metrics = val_m
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "val_metrics": val_m}, ckpt_path)
            print(f"  ✓ checkpoint saved (val_loss={val_loss:.4f})")

        if es(val_loss):
            print(f"  ✗ early stopping at epoch {epoch}"); break

    json.dump(history, open(os.path.join(LOG_DIR, f"fold{fold}_history.json"), "w"), indent=2)
    print(f"\nFold {fold+1} best  acc={best_metrics.get('accuracy',0)*100:.2f}%  "
          f"f1={best_metrics.get('f1',0)*100:.2f}%")
    return best_metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# FINE-TUNING  (best fold only)
# ═══════════════════════════════════════════════════════════════════════════

def finetune_fold(model, train_loader, val_loader, fold,
                  epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    """
    Fine-tune the best-fold checkpoint.
    Called ONLY for the single fold with the highest validation F1.

    - Full model unfreeze.
    - Differential LR: backbone x0.1, everything else x1.
    - Short warmup (2 ep) + cosine.
    - Tighter label smoothing (0.05).
    """
    model.to(DEVICE)
    model.unfreeze_all()

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING FOLD {fold+1} (best fold)  |  "
          f"epochs={epochs}  lr={lr:.1e}  backbone_lr={lr*0.1:.1e}")
    print(f"{'='*60}")

    criterion    = nn.CrossEntropyLoss(label_smoothing=0.05)
    param_groups = model.get_parameter_groups(lr_backbone=lr * 0.1, lr_head=lr)
    optimizer    = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    ft_warmup    = min(2, max(1, epochs // 10))
    scheduler    = WarmupCosineScheduler(optimizer, ft_warmup, epochs, LR_ETA_MIN)
    scaler       = torch.amp.GradScaler('cuda')
    es           = EarlyStopping(patience=7)

    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"fold{fold}_finetuned.pt")
    best_val_loss = float("inf"); best_metrics = {}; history = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m   = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, model.cls_head.net[-1].out_features)
        val_loss, val_m = validate(model, val_loader, criterion)
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]["lr"]

        history.append(dict(epoch=epoch, lr=cur_lr,
                            train_loss=tr_loss, train_acc=tr_m["accuracy"],
                            val_loss=val_loss,   val_acc=val_m["accuracy"],
                            val_f1=val_m["f1"]))

        print(f"FT [{epoch:3d}/{epochs}]  lr={cur_lr:.2e}  "
              f"tr={tr_loss:.4f}/{tr_m['accuracy']*100:.1f}%  "
              f"val={val_loss:.4f}/{val_m['accuracy']*100:.1f}%  "
              f"f1={val_m['f1']*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_metrics = val_m
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "val_metrics": val_m}, ckpt_path)
            print(f"  ✓ fine-tuned checkpoint saved")

        if es(val_loss):
            print(f"  ✗ early stopping at epoch {epoch}"); break

    json.dump(history,
              open(os.path.join(LOG_DIR, f"fold{fold}_finetune_history.json"), "w"), indent=2)
    return best_metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT VOCAB + SCALER
# ═══════════════════════════════════════════════════════════════════════════

def export_vocab_scaler(vocab, scaler, label_map: dict, fold: int,
                        out_dir: str = CHECKPOINT_DIR):
    """
    Save three artefacts from the best fold for reuse outside this pipeline:

    tokenizer.json  –  word->index mapping + pad/unk metadata
    scaler.pkl      –  sklearn StandardScaler (fitted on training data)
    label_map.json  –  {class_name: int_index}

    These can be loaded independently of PyTorch/this codebase, e.g.:

        import json, pickle
        tok   = json.load(open("tokenizer.json"))["word2idx"]
        sc    = pickle.load(open("scaler.pkl", "rb"))
        lmap  = json.load(open("label_map.json"))
    """
    os.makedirs(out_dir, exist_ok=True)

    # tokenizer.json
    tok_path = os.path.join(out_dir, "tokenizer.json")
    json.dump({
        "fold":       fold,
        "vocab_size": len(vocab.word2idx),
        "pad_token":  "<PAD>", "pad_index": 0,
        "unk_token":  "<UNK>", "unk_index": 1,
        "word2idx":   vocab.word2idx,
    }, open(tok_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Saved tokenizer  -> {tok_path}  ({len(vocab.word2idx):,} tokens)")

    # scaler.pkl
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved scaler     -> {scaler_path}")

    # label_map.json
    lmap_path = os.path.join(out_dir, "label_map.json")
    json.dump(label_map, open(lmap_path, "w"), indent=2)
    print(f"  Saved label map  -> {lmap_path}")

    return tok_path, scaler_path, lmap_path
