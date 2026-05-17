"""
trainer.py  –  Training engine for PneumoFusion-Net

DATA-LEAKAGE / CORRECTNESS FIXES (vs previous version)
=======================================================

FIX 1 — validate() used a label-smoothed criterion  [HIGH]
-----------------------------------------------------------
Previous bug: train_fold() built one CrossEntropyLoss(label_smoothing=0.25)
and passed THE SAME criterion to both train_one_epoch() and validate().
This meant validation loss was computed with label smoothing applied,
artificially lowering the val loss numbers and making them incomparable
to a true held-out loss.  Early stopping fired on the smoothed val loss,
not the real one, corrupting the checkpoint selection signal.

Fix: validate() now uses a SEPARATE, zero-smoothing criterion:
    val_criterion = nn.CrossEntropyLoss()   # no smoothing
Label smoothing is only applied during training via the train criterion.

FIX 2 — train accuracy reported on hard labels after mixup  [HIGH]
-------------------------------------------------------------------
Previous bug: after mixup_batch() returned mixed images and SOFT labels,
the code tracked `all_labels` (the original hard integer labels) and
`all_preds` (argmax of logits on the MIXED images).  These are
misaligned: the model saw a blended image but accuracy was graded against
one of the two original clean labels.  This made training accuracy
artificially high and uninterpretable.

Fix: training accuracy is no longer tracked per-step during mixup.
Instead we track training LOSS only, and report training accuracy from a
clean (no-mixup) pass over a random subset of training batches once per
epoch via _eval_train_acc().  This gives a true, interpretable training
accuracy without wasting a full epoch.

FIX 3 — ClassificationHead token_weights initialised with label-derived priors [MEDIUM]
----------------------------------------------------------------------------------------
Moved to classification_head_fixed.py: token_weights now initialised to
uniform (ones), not [0.45, 0.22, 0.33] which baked in the paper's results
as prior knowledge.
"""

import os, time, json, pickle, math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    Epoch 0..warmup_epochs-1 : LR ramps linearly LR/10 -> base_LR
    Epoch warmup_epochs..T_max: CosineAnnealing  base_LR -> eta_min
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
    Returns mixed images and SOFT labels (float one-hot blends).
    Training accuracy is NOT tracked against these soft labels — see
    _eval_train_acc() for a proper training accuracy estimate.
    """
    assert labels.max().item() < num_classes, (
        f"mixup_batch: label index {labels.max().item()} >= num_classes {num_classes}."
    )
    assert labels.min().item() >= 0, (
        f"mixup_batch: negative label index {labels.min().item()} found."
    )

    if alpha <= 0:
        one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return images, one_hot

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)   # always >= 0.5

    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)

    mixed = lam * images + (1.0 - lam) * images[idx]

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
        self.counter  = 0; self.best_val = None; self.stop = False

    def __call__(self, val_loss):
        if self.best_val is None or val_loss < self.best_val - self.delta:
            self.best_val = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING ACCURACY HELPER  (FIX 2)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_train_acc(model, loader, num_batches: int = 10):
    """
    Estimate training accuracy on a small subset of training batches
    WITHOUT mixup, so the accuracy is graded against clean hard labels.

    This replaces the previous approach of tracking accuracy against hard
    labels while the model was processing mixup-blended images, which
    gave misleadingly high accuracy numbers.

    Parameters
    ----------
    model      : model in train mode — we temporarily switch to eval
    loader     : training DataLoader
    num_batches: how many batches to sample (default 10 ≈ ~320 samples)

    Returns
    -------
    accuracy : float
    """
    model.eval()
    all_labels, all_preds = [], []
    for i, (images, text_ids, num_feats, labels) in enumerate(loader):
        if i >= num_batches:
            break
        images    = images.to(DEVICE,    non_blocking=True)
        text_ids  = text_ids.to(DEVICE,  non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE,    non_blocking=True)
        # No mixup — raw images → clean accuracy
        logits = model(images, text_ids, num_feats)
        preds  = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    model.train()
    return accuracy_score(np.array(all_labels), np.array(all_preds))


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE EPOCH  (FIX 2 applied here)
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, amp_scaler, epoch, num_classes):
    """
    Run one training epoch with mixup augmentation.

    Returns
    -------
    avg_loss : float   — mean loss over all batches
    NOTE: accuracy is NOT returned here because it cannot be computed
    correctly against mixed labels.  Use _eval_train_acc() separately.
    """
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for idx, (images, text_ids, num_feats, labels) in enumerate(loader):
        images    = images.to(DEVICE,    non_blocking=True)
        text_ids  = text_ids.to(DEVICE,  non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE,    non_blocking=True)

        # ── Mixup augmentation ────────────────────────────────────────────
        images, soft_labels = mixup_batch(images, labels, num_classes)

        optimizer.zero_grad(set_to_none=True)

        if MIXED_PRECISION and DEVICE.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits    = model(images, text_ids, num_feats)
                # soft-label cross-entropy
                log_probs = F.log_softmax(logits, dim=1)
                loss      = -(soft_labels * log_probs).sum(dim=1).mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits    = model(images, text_ids, num_feats)
            log_probs = F.log_softmax(logits, dim=1)
            loss      = -(soft_labels * log_probs).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()

        if (idx + 1) % max(1, len(loader) // 5) == 0:
            print(f"  [Ep {epoch}] step {idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  t={time.time()-t0:.1f}s")

    return total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION  (FIX 1 applied here — no label smoothing on val loss)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader):
    """
    Evaluate on the validation set.

    FIX 1: Uses its own CrossEntropyLoss with NO label smoothing.
    Val loss is a true cross-entropy against hard labels, making it
    directly comparable across epochs and suitable for early stopping.

    The train criterion (with label smoothing) is intentionally NOT
    passed here — validation never uses label smoothing.
    """
    # FIX 1: local criterion, zero smoothing — val loss is always "real"
    val_criterion = nn.CrossEntropyLoss()

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
                loss   = val_criterion(logits, labels)
        else:
            logits = model(images, text_ids, num_feats)
            loss   = val_criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / len(loader),
        compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# FULL FOLD TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, fold,
               epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
               label_smoothing=LABEL_SMOOTHING):
    """
    Train one cross-validation fold.

    Changes vs previous version
    ---------------------------
    - validate() no longer receives the training criterion (FIX 1).
    - Training accuracy estimated via _eval_train_acc() on 10 clean
      batches, not from misaligned mixup predictions (FIX 2).
    - History dict uses 'train_acc' from _eval_train_acc().

    Returns
    -------
    best_metrics : dict   — metrics at the best checkpoint epoch
    history      : list   — one dict per epoch for plotting
    """
    model.to(DEVICE)

    # Training criterion — label smoothing applies ONLY to training loss
    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, epochs, LR_ETA_MIN)
    amp_scaler = torch.amp.GradScaler('cuda')
    es = EarlyStopping(EARLY_STOP_PAT)

    num_classes   = model.cls_head.net[-1].out_features
    best_val_loss = float("inf")
    best_metrics  = {}
    history       = []
    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pt")

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}  |  device={DEVICE}  lr={lr:.1e}  "
          f"warmup={WARMUP_EPOCHS}ep  clip={GRAD_CLIP_NORM}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # FIX 2: train_one_epoch returns only loss (no acc)
        tr_loss = train_one_epoch(
            model, train_loader, train_criterion,
            optimizer, amp_scaler, epoch, num_classes,
        )

        # FIX 2: accurate training acc estimated on clean (no-mixup) samples
        tr_acc = _eval_train_acc(model, train_loader, num_batches=10)

        # FIX 1: validate() uses its own zero-smoothing criterion internally
        val_loss, val_m = validate(model, val_loader)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        phase  = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"

        history.append(dict(
            epoch=epoch, lr=cur_lr, phase=phase,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=val_loss,
            val_acc=val_m["accuracy"], val_f1=val_m["f1"],
            val_auc=val_m.get("roc_auc", 0.0),
        ))

        print(f"Ep [{epoch:3d}/{epochs}][{phase}]  lr={cur_lr:.2e}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc*100:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_m['accuracy']*100:.1f}%  "
              f"f1={val_m['f1']*100:.2f}%")

        # Checkpoint on TRUE val loss (no smoothing, FIX 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_m
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_m},
                ckpt_path,
            )
            print(f"  ✓ checkpoint saved (val_loss={val_loss:.4f})")

        if es(val_loss):
            print(f"  ✗ early stopping at epoch {epoch}")
            break

    json.dump(
        history,
        open(os.path.join(LOG_DIR, f"fold{fold}_history.json"), "w"),
        indent=2,
    )
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
    Same FIX 1 + FIX 2 applied as in train_fold().
    Fine-tuning uses very light label smoothing (0.05) to avoid
    over-regularising an already partially-trained model.
    """
    model.to(DEVICE)
    model.unfreeze_all()

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING FOLD {fold+1} (best fold)  |  "
          f"epochs={epochs}  lr={lr:.1e}  backbone_lr={lr*0.1:.1e}")
    print(f"{'='*60}")

    # Very light label smoothing for fine-tuning — only on training loss
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    param_groups    = model.get_parameter_groups(lr_backbone=lr * 0.1, lr_head=lr)
    optimizer       = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    ft_warmup       = min(2, max(1, epochs // 10))
    scheduler       = WarmupCosineScheduler(optimizer, ft_warmup, epochs, LR_ETA_MIN)
    amp_scaler      = torch.amp.GradScaler('cuda')
    es              = EarlyStopping(patience=7)

    num_classes   = model.cls_head.net[-1].out_features
    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"fold{fold}_finetuned.pt")
    best_val_loss = float("inf")
    best_metrics  = {}
    history       = []

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, train_criterion,
            optimizer, amp_scaler, epoch, num_classes,
        )
        tr_acc = _eval_train_acc(model, train_loader, num_batches=10)

        # FIX 1: validate() builds its own zero-smoothing criterion
        val_loss, val_m = validate(model, val_loader)
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]["lr"]

        history.append(dict(
            epoch=epoch, lr=cur_lr,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=val_loss,
            val_acc=val_m["accuracy"], val_f1=val_m["f1"],
        ))

        print(f"FT [{epoch:3d}/{epochs}]  lr={cur_lr:.2e}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc*100:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_m['accuracy']*100:.1f}%  "
              f"f1={val_m['f1']*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_m
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_m},
                ckpt_path,
            )
            print(f"  ✓ fine-tuned checkpoint saved")

        if es(val_loss):
            print(f"  ✗ early stopping at epoch {epoch}")
            break

    json.dump(
        history,
        open(os.path.join(LOG_DIR, f"fold{fold}_finetune_history.json"), "w"),
        indent=2,
    )
    return best_metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT VOCAB + SCALER
# ═══════════════════════════════════════════════════════════════════════════

def export_vocab_scaler(vocab, scaler, label_map: dict, fold: int,
                        out_dir: str = CHECKPOINT_DIR):
    """
    Save tokenizer.json, scaler.pkl, and label_map.json from the best fold.
    The scaler saved here was fitted on training data only (guaranteed by
    the fixed get_fold_dataloaders() in data_pipeline.py).
    """
    os.makedirs(out_dir, exist_ok=True)

    tok_path = os.path.join(out_dir, "tokenizer.json")
    json.dump({
        "fold":       fold,
        "vocab_size": len(vocab.word2idx),
        "pad_token":  "<PAD>", "pad_index": 0,
        "unk_token":  "<UNK>", "unk_index": 1,
        "word2idx":   vocab.word2idx,
    }, open(tok_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Saved tokenizer  -> {tok_path}  ({len(vocab.word2idx):,} tokens)")

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved scaler     -> {scaler_path}")

    lmap_path = os.path.join(out_dir, "label_map.json")
    json.dump(label_map, open(lmap_path, "w"), indent=2)
    print(f"  Saved label map  -> {lmap_path}")

    return tok_path, scaler_path, lmap_path
