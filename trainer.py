"""
trainer.py  –  Training engine for PneumoFusion-Net

Provides
--------
• train_one_epoch   – single training pass with AMP
• validate          – validation pass, returns loss + metrics
• EarlyStopping     – patience-based stopping helper
• train_fold        – complete training loop for one fold
• finetune_fold     – fine-tuning loop on a single fold with layer unfreezing
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_DIR,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_T0, LR_ETA_MIN, EARLY_STOP_PAT,
    MIXED_PRECISION, NUM_CLASSES, CLASS_NAMES,
    FINETUNE_EPOCHS, FINETUNE_LR,
)


# ═══════════════════════════════════════════════════════════════════════════
# METRICS HELPER
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Parameters
    ----------
    y_true : 1-D array of integer ground-truth labels
    y_pred : 1-D array of integer predicted labels
    y_prob : 2-D array (N, C) of softmax probabilities (optional, for AUC)

    Returns dict with accuracy, precision, recall, f1, roc_auc (if available)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["roc_auc"] = auc
        except ValueError:
            pass

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PAT, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.counter  = 0
        self.best_val = None
        self.stop     = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if self.best_val is None or val_loss < self.best_val - self.delta:
            self.best_val = val_loss
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-EPOCH TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch: int):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []
    t0 = time.time()

    for batch_idx, (images, text_ids, num_feats, labels) in enumerate(loader):
        images    = images.to(DEVICE, non_blocking=True)
        text_ids  = text_ids.to(DEVICE, non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if MIXED_PRECISION and DEVICE.type == "cuda":
            with autocast():
                logits = model(images, text_ids, num_feats)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, text_ids, num_feats)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            elapsed = time.time() - t0
            print(
                f"  [Epoch {epoch}]  step {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}  elapsed={elapsed:.1f}s"
            )

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


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
            with autocast():
                logits = model(images, text_ids, num_feats)
                loss   = criterion(logits, labels)
        else:
            logits = model(images, text_ids, num_feats)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
# FULL FOLD TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(
    model,
    train_loader,
    val_loader,
    fold: int,
    epochs: int    = EPOCHS,
    lr: float      = LEARNING_RATE,
    weight_decay   = WEIGHT_DECAY,
    label_smoothing: float = 0.1,
):
    """
    Train `model` on one fold.

    Returns
    -------
    best_metrics : dict  – metrics at the best validation loss checkpoint
    history      : list[dict]  – per-epoch log
    """
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T0, T_mult=2, eta_min=LR_ETA_MIN,
    )
    scaler        = GradScaler(enabled=(MIXED_PRECISION and DEVICE.type == "cuda"))
    early_stop    = EarlyStopping(patience=EARLY_STOP_PAT)
    best_val_loss = float("inf")
    best_metrics  = {}
    history       = []

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pt")

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}  –  Training on {DEVICE}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────
        tr_loss, tr_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )

        # ── validate ─────────────────────────────────────────────────────
        val_loss, val_metrics = validate(model, val_loader, criterion)

        # ── scheduler step ───────────────────────────────────────────────
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── logging ──────────────────────────────────────────────────────
        log = dict(
            epoch=epoch,
            lr=current_lr,
            train_loss=tr_loss,  train_acc=tr_metrics["accuracy"],
            train_f1=tr_metrics["f1"],
            val_loss=val_loss,   val_acc=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_auc=val_metrics.get("roc_auc", 0.0),
        )
        history.append(log)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"lr={current_lr:.2e}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_metrics['accuracy']*100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']*100:.2f}%  "
            f"val_f1={val_metrics['f1']*100:.2f}%"
        )

        # ── checkpoint ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_metrics
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_metrics},
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint  (val_loss={val_loss:.4f})")

        # ── early stopping ───────────────────────────────────────────────
        if early_stop(val_loss):
            print(f"  ✗ Early stopping triggered at epoch {epoch}")
            break

    # save epoch history to JSON
    log_path = os.path.join(LOG_DIR, f"fold{fold}_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFold {fold+1} best  acc={best_metrics['accuracy']*100:.2f}%  "
          f"f1={best_metrics['f1']*100:.2f}%")

    return best_metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════

def finetune_fold(
    model,
    train_loader,
    val_loader,
    fold: int,
    epochs: int = FINETUNE_EPOCHS,
    lr: float   = FINETUNE_LR,
):
    """
    Fine-tune a pre-loaded checkpoint:
      1. Unfreeze the full CNN backbone (layer3, layer4, GCSA, fc_proj).
      2. Use differential learning rates (lower for backbone).
      3. Run a short training loop with CosineAnnealing.

    Returns
    -------
    best_metrics, history
    """
    model.to(DEVICE)
    model.unfreeze_all()    # full unfreeze

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING FOLD {fold+1}  –  {epochs} epochs")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    param_groups = model.get_parameter_groups(lr_backbone=lr * 0.1, lr_head=lr)
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR_ETA_MIN)
    scaler    = GradScaler(enabled=(MIXED_PRECISION and DEVICE.type == "cuda"))
    early_stop = EarlyStopping(patience=5)

    ckpt_path    = os.path.join(CHECKPOINT_DIR, f"fold{fold}_finetuned.pt")
    best_val_loss = float("inf")
    best_metrics  = {}
    history       = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]

        log = dict(
            epoch=epoch, lr=current_lr,
            train_loss=tr_loss,  train_acc=tr_metrics["accuracy"],
            val_loss=val_loss,   val_acc=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
        )
        history.append(log)

        print(
            f"FT Epoch [{epoch:3d}/{epochs}]  "
            f"lr={current_lr:.2e}  "
            f"tr_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']*100:.2f}%  "
            f"val_f1={val_metrics['f1']*100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_metrics
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_metrics},
                ckpt_path,
            )
            print(f"  ✓ Saved fine-tuned checkpoint")

        if early_stop(val_loss):
            print(f"  ✗ Early stopping at epoch {epoch}")
            break

    log_path = os.path.join(LOG_DIR, f"fold{fold}_finetune_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_metrics, history
