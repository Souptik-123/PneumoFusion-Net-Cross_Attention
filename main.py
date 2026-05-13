"""
main.py  –  Orchestration script for PneumoFusion-Net
                                                          
Usage
-----
    python main.py                     # full 5-fold CV + fine-tune best fold
    python main.py --fold 0            # train & fine-tune a single fold
    python main.py --no-finetune       # skip fine-tuning phase
    python main.py --eval-only --fold 0 --ckpt outputs/checkpoints/fold0_best.pt

Pipeline
--------
    1. Load dataset CSV
    2. Build StratifiedKFold splits
    3. For each fold:
        a. build DataLoaders (vocab + scaler built from training data only)
        b. build PneumoFusionNet
        c. train_fold  → save best checkpoint
        d. evaluate on validation set
        e. plot confusion matrix + training curves
    4. Aggregate CV metrics
    5. Fine-tune the best fold's model
    6. Re-evaluate the fine-tuned checkpoint
    7. Print final summary
"""

import argparse
import os
import random
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# ── local imports ─────────────────────────────────────────────────────────
from config import (
    CSV_PATH, DATA_ROOT, K_FOLDS, SEED, BATCH_SIZE,
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FINETUNE_EPOCHS, FINETUNE_LR,
    VOCAB_SIZE, NUM_NUMERICAL_FEATURES,
)
from data_pipeline import load_dataframe, get_fold_dataloaders, build_label_map
from models import PneumoFusionNet
from trainer     import train_fold, finetune_fold
from evaluate    import (
    evaluate_model, plot_confusion_matrix, plot_training_curves,
    aggregate_cv_results, save_fold_results, log_modality_weights,
)


# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="PneumoFusion-Net training")
    parser.add_argument("--fold",        type=int,   default=None,
                        help="Run only this fold index (0-based).  Default: all folds")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip the fine-tuning phase")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Load an existing checkpoint and evaluate (no training)")
    parser.add_argument("--ckpt",        type=str,   default=None,
                        help="Path to checkpoint to load for --eval-only")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Do NOT use ImageNet pre-trained weights for CNN")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(SEED)

    # ── 1.  Load data ──────────────────────────────────────────────────────
    print_banner("Loading dataset")
    df = load_dataframe(os.path.join(DATA_ROOT, CSV_PATH))
    label_map = build_label_map(df["label"])
    num_classes = len(label_map)
    class_names = sorted(label_map, key=lambda k: label_map[k])

    print(f"  Dataset size : {len(df)}")
    print(f"  Classes      : {class_names}")
    print(f"  Label map    : {label_map}")
    print(f"  Device       : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")

    # ── 2.  Stratified K-Fold ──────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # ── 3.  Eval-only mode ─────────────────────────────────────────────────
    if args.eval_only:
        assert args.ckpt and os.path.exists(args.ckpt), \
            f"Checkpoint not found: {args.ckpt}"
        fold = args.fold or 0
        train_loader, val_loader, vocab, _ = get_fold_dataloaders(
            df, fold, skf, label_map, batch_size=args.batch_size
        )
        model = PneumoFusionNet(num_classes=num_classes, vocab_size=len(vocab),
                                pretrained_cnn=False)
        ckpt = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print_banner(f"Evaluating checkpoint: {args.ckpt}")
        metrics, y_true, y_pred, y_prob = evaluate_model(model, val_loader, class_names)
        plot_confusion_matrix(y_true, y_pred, fold, class_names)
        return

    # ── 4.  Cross-validation ───────────────────────────────────────────────
    folds_to_run = [args.fold] if args.fold is not None else list(range(K_FOLDS))

    cv_metrics         = []
    cv_finetune_metrics= []
    best_cv_f1         = -1.0
    best_fold          = 0

    for fold in folds_to_run:
        set_seed(SEED + fold)    # different seed per fold for reproducibility

        print_banner(f"FOLD {fold+1} / {K_FOLDS}")

        # ── DataLoaders ───────────────────────────────────────────────────
        train_loader, val_loader, vocab, scaler = get_fold_dataloaders(
            df, fold, skf, label_map, batch_size=args.batch_size
        )
        print(f"  Vocab size: {len(vocab)}  |  "
              f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

        # ── Model ─────────────────────────────────────────────────────────
        model = PneumoFusionNet(
            num_classes=num_classes,
            vocab_size=len(vocab),
            pretrained_cnn=(not args.no_pretrain),
        )
        total_p, trainable_p = count_parameters(model)
        print(f"  Parameters total={total_p:,}  trainable={trainable_p:,}")

        # ── Training ──────────────────────────────────────────────────────
        best_metrics, history = train_fold(
            model, train_loader, val_loader, fold,
            epochs=args.epochs, lr=args.lr, weight_decay=WEIGHT_DECAY,
        )

        # ── Evaluate best checkpoint ──────────────────────────────────────
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

        print_banner(f"Evaluation – Fold {fold+1}")
        metrics, y_true, y_pred, y_prob = evaluate_model(model, val_loader, class_names)
        cv_metrics.append(metrics)

        # ── Plots ─────────────────────────────────────────────────────────
        plot_confusion_matrix(y_true, y_pred, fold, class_names)
        plot_training_curves(history, fold)
        save_fold_results(metrics, fold)
        log_modality_weights(model, fold)          # ← learned contribution %

        # track best fold for fine-tuning
        if metrics["f1_macro"] > best_cv_f1:
            best_cv_f1  = metrics["f1_macro"]
            best_fold   = fold
            # save vocab/scaler for best fold (for inference later)
            torch.save(
                {"vocab": vocab.word2idx, "scaler_mean": scaler.mean_,
                 "scaler_std": np.sqrt(scaler.var_), "label_map": label_map},
                os.path.join(CHECKPOINT_DIR, "best_fold_meta.pt"),
            )

        # ── Fine-tuning ───────────────────────────────────────────────────
        if not args.no_finetune:
            print_banner(f"Fine-tuning – Fold {fold+1}")
            ft_metrics, ft_history = finetune_fold(
                model, train_loader, val_loader, fold,
                epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR,
            )

            ft_ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold}_finetuned.pt")
            ft_ckpt = torch.load(ft_ckpt_path, map_location=DEVICE)
            model.load_state_dict(ft_ckpt["model_state"])

            print_banner(f"Post-Fine-Tune Evaluation – Fold {fold+1}")
            ft_eval_metrics, ft_y_true, ft_y_pred, _ = evaluate_model(
                model, val_loader, class_names
            )
            cv_finetune_metrics.append(ft_eval_metrics)

            plot_confusion_matrix(ft_y_true, ft_y_pred, fold, class_names, suffix="_finetuned")
            plot_training_curves(ft_history, fold, suffix="_finetuned")
            save_fold_results(ft_eval_metrics, fold, suffix="_finetuned")
            log_modality_weights(model, fold, suffix="_finetuned")  # ← post fine-tune weights

    # ── 5.  Aggregate results ──────────────────────────────────────────────
    if len(cv_metrics) > 1:
        print_banner("Cross-Validation Aggregate (before fine-tuning)")
        aggregate_cv_results(cv_metrics, label="CV_pretrain")

        if cv_finetune_metrics:
            print_banner("Cross-Validation Aggregate (after fine-tuning)")
            aggregate_cv_results(cv_finetune_metrics, label="CV_finetuned")

    print_banner("DONE")
    print(f"  Best fold     : {best_fold+1}  (F1={best_cv_f1*100:.2f}%)")
    print(f"  Checkpoints   : {CHECKPOINT_DIR}")
    print(f"  Results/plots : {RESULTS_DIR}")
    print(f"  Logs          : {os.path.join(os.path.dirname(CHECKPOINT_DIR), 'logs')}")


if __name__ == "__main__":
    main()
