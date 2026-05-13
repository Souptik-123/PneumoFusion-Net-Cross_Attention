"""
evaluate.py  –  Evaluation, visualisation, and result aggregation

Functions
---------
• evaluate_model         – full evaluation on a DataLoader (returns metrics + arrays)
• plot_confusion_matrix  – saves a confusion-matrix heatmap to RESULTS_DIR
• plot_training_curves   – loss and accuracy curves for a fold
• aggregate_cv_results   – prints mean ± std across all folds
• save_fold_results      – writes per-fold JSON summary
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from config import DEVICE, RESULTS_DIR, CLASS_NAMES, MIXED_PRECISION


# ═══════════════════════════════════════════════════════════════════════════
# FULL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, loader, label_names=CLASS_NAMES):
    """
    Parameters
    ----------
    model  : trained PneumoFusionNet
    loader : DataLoader

    Returns
    -------
    metrics   : dict
    y_true    : np.ndarray (N,)
    y_pred    : np.ndarray (N,)
    y_prob    : np.ndarray (N, C)
    """
    model.eval()
    model.to(DEVICE)
    all_labels, all_preds, all_probs = [], [], []

    for images, text_ids, num_feats, labels in loader:
        images    = images.to(DEVICE, non_blocking=True)
        text_ids  = text_ids.to(DEVICE, non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)

        logits = model(images, text_ids, num_feats)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── metrics ───────────────────────────────────────────────────────────
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        pass

    # per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        metrics[f"f1_{name.replace(' ', '_')}"] = float(per_class_f1[i])

    print("\n" + "─" * 55)
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1 Macro  : {metrics['f1_macro']*100:.2f}%")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("─" * 55)
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    return metrics, y_true, y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, fold: int, label_names=CLASS_NAMES, suffix: str = ""):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix – {title}\nFold {fold+1}{suffix}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"fold{fold}_confmat{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: list, fold: int, suffix: str = ""):
    epochs      = [h["epoch"]    for h in history]
    tr_loss     = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    tr_acc      = [h["train_acc"]  for h in history]
    val_acc     = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # loss
    ax1.plot(epochs, tr_loss,  label="Train Loss",      linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss",        linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves  –  Fold {fold+1}{suffix}")
    ax1.legend(); ax1.grid(alpha=0.3)

    # accuracy
    ax2.plot(epochs, [a * 100 for a in tr_acc],  label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_acc], label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Accuracy Curves  –  Fold {fold+1}{suffix}")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"fold{fold}_curves{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training curves → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CV AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_cv_results(fold_metrics: list, label: str = "Cross-Validation"):
    """
    Parameters
    ----------
    fold_metrics : list of dicts, one per fold

    Prints mean ± std for key metrics and saves a summary JSON.
    """
    keys = ["accuracy", "f1_macro", "precision", "recall"]
    print(f"\n{'═'*55}")
    print(f"  {label} Summary  ({len(fold_metrics)} folds)")
    print(f"{'═'*55}")
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        mean = np.mean(vals)
        std  = np.std(vals)
        summary[k] = {"mean": float(mean), "std": float(std)}
        print(f"  {k:15s}: {mean*100:.2f}% ± {std*100:.2f}%")
    if "roc_auc" in fold_metrics[0]:
        aucs = [m["roc_auc"] for m in fold_metrics]
        summary["roc_auc"] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))}
        print(f"  {'roc_auc':15s}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'═'*55}")

    save_path = os.path.join(RESULTS_DIR, f"{label.lower().replace(' ', '_')}_summary.json")
    with open(save_path, "w") as f:
        json.dump({"per_fold": fold_metrics, "aggregate": summary}, f, indent=2)
    print(f"  Saved summary → {save_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# PER-FOLD RESULT SAVE
# ═══════════════════════════════════════════════════════════════════════════

def save_fold_results(metrics: dict, fold: int, suffix: str = ""):
    path = os.path.join(RESULTS_DIR, f"fold{fold}_metrics{suffix}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODALITY WEIGHT LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def log_modality_weights(model, fold: int, suffix: str = ""):
    """
    Read the learned token_weights from ClassificationHead, pretty-print
    them, save to JSON, and return the weight dict.

    Call this after loading the best/fine-tuned checkpoint for a fold.

    Example output
    --------------
      Learned modality weights - Fold 1
        CT image (CNN)   :  44.82 %   ████████████████████████
        Clinical text    :  12.03 %   ██████
        Lab numerics     :  43.15 %   ███████████████████████

    Parameters
    ----------
    model  : PneumoFusionNet instance (weights already loaded)
    fold   : fold index (0-based), used for the save filename
    suffix : optional string appended to the filename (e.g. "_finetuned")
    """
    weights = model.cls_head.get_modality_weights()   # dict {name: float}

    bar_max = 30
    max_pct = max(weights.values())

    print(f"\n  Learned modality weights - Fold {fold + 1}{suffix}")
    print(f"  {'─'*52}")
    for name, w in weights.items():
        pct     = w * 100
        print(f"    {name:20s} :  {pct:5.2f} % ")
    print(f"  {'─'*52}")
    print(f"  (weights sum to {sum(weights.values())*100:.1f} %)")

    save_path = os.path.join(
        RESULTS_DIR, f"fold{fold}_modality_weights{suffix}.json"
    )
    with open(save_path, "w") as f:
        json.dump({"fold": fold, "suffix": suffix, "weights": weights}, f, indent=2)
    print(f"  Saved modality weights -> {save_path}")

    return weights
