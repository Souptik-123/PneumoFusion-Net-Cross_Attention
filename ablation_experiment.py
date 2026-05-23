"""
ablation_experiment.py
======================
Reproduces the "Validation Accuracy Comparison" chart from the paper.

7 modality combinations trained for 80 epochs each, single train/val split
(80/20 stratified), curves saved to outputs/results/ablation_curves.png.

Modality combinations
---------------------
  1. Image Only
  2. Text Only
  3. Numerical Only
  4. Image + Text
  5. Image + Numerical
  6. Text + Numerical
  7. Image + Text + Numerical  (full model)

Usage
-----
    python ablation_experiment.py

All hyper-parameters are taken from config.py.  No changes to existing
model modules are needed — each combination is handled by zeroing /
replacing the missing modality inputs with learned zero-embeddings so
the transformer always receives the same (B, 3, D) sequence.
"""

import os, random, math, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────
from config import (
    DEVICE, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    LR_ETA_MIN, WARMUP_EPOCHS, GRAD_CLIP_NORM, LABEL_SMOOTHING,
    MIXUP_ALPHA, DROPOUT_RATE, RESULTS_DIR,
    CNN_OUT_DIM, TEXT_OUT_DIM, NUM_OUT_DIM, FUSION_DIM,
    XATTN_LAYERS, XATTN_HEADS, XATTN_FF_DIM, XATTN_DROPOUT, MAX_DROP_PATH,
    CLS_HIDDEN_DIM, NUM_CLASSES, VOCAB_SIZE, NUM_NUMERICAL_FEATURES,
    IMAGE_SIZE, MAX_SEQ_LEN, NUMERICAL_COLS,
)
print("Using device:", DEVICE)
from data_pipeline import (
    Vocabulary, PneumoDataset, build_label_map,
    _train_transform, _val_transform,
)
from models.cnn_encoder        import CNNImageEncoder
from models.text_encoder       import BiLSTMTextEncoder
from models.numerical_encoder  import MLPNumericalEncoder
from models.fusion             import FeatureFusion
from models.transformer        import CrossAttentionTransformer
from models.classification_head import ClassificationHead

CSV_PATH  = r"E:\4thYearProjectCoding\dataset\unified_dataset_new3.csv"
ABLATION_EPOCHS = 25         # 80, matching the paper chart x-axis

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── colours matching the paper chart ─────────────────────────────────────
COMBO_STYLES = {
    "Image + Text + Numerical": dict(color="#2ca02c", lw=2.0, ls="-"),
    "Image + Numerical":        dict(color="#1a1a1a", lw=1.8, ls="-"),
    "Image + Text":             dict(color="#d62728", lw=1.6, ls="-"),
    "Image Only":               dict(color="#1f77b4", lw=1.5, ls="-"),
    "Numerical Only":           dict(color="#9467bd", lw=1.4, ls="-"),
    "Text + Numerical":         dict(color="#ff7f0e", lw=1.4, ls="-"),
    "Text Only":                dict(color="#00bcd4", lw=1.4, ls="-"),
}

COMBOS = list(COMBO_STYLES.keys())


# ═══════════════════════════════════════════════════════════════════════════
# SEED
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════
# MODALITY-MASKED MODEL
# ═══════════════════════════════════════════════════════════════════════════

class AblationModel(nn.Module):
    """
    Full PneumoFusionNet with per-modality on/off switches.

    When a modality is disabled its encoder output is replaced by a
    learned zero-token (nn.Parameter of zeros) so the transformer always
    receives a (B, 3, FUSION_DIM) sequence regardless of which modalities
    are active.  This avoids architecture changes per combination and lets
    all combinations share the same transformer/head.

    Parameters
    ----------
    use_image   : include CNN image encoder
    use_text    : include BiLSTM text encoder
    use_num     : include MLP numerical encoder
    vocab_size  : vocabulary size for the text encoder
    pretrained  : load ImageNet weights for CNN
    dropout     : dropout rate throughout
    """

    def __init__(
        self,
        use_image:  bool = True,
        use_text:   bool = True,
        use_num:    bool = True,
        vocab_size: int  = VOCAB_SIZE,
        pretrained: bool = True,
        dropout:    float = DROPOUT_RATE,
    ):
        super().__init__()
        self.use_image = use_image
        self.use_text  = use_text
        self.use_num   = use_num

        # ── encoders (only build what's needed to save memory) ────────────
        self.cnn_encoder  = CNNImageEncoder(out_dim=CNN_OUT_DIM, pretrained=pretrained) \
                            if use_image else None
        self.text_encoder = BiLSTMTextEncoder(vocab_size=vocab_size,
                                              out_dim=TEXT_OUT_DIM, dropout=dropout) \
                            if use_text else None
        self.num_encoder  = MLPNumericalEncoder(in_dim=NUM_NUMERICAL_FEATURES,
                                                out_dim=NUM_OUT_DIM, dropout=dropout) \
                            if use_num else None

        # ── per-modality projections to FUSION_DIM ────────────────────────
        self.cnn_proj  = nn.Linear(CNN_OUT_DIM,  FUSION_DIM) if use_image else None
        self.text_proj = nn.Linear(TEXT_OUT_DIM, FUSION_DIM) if use_text  else None
        self.num_proj  = nn.Linear(NUM_OUT_DIM,  FUSION_DIM) if use_num   else None

        # ── learned zero-token for each disabled modality ─────────────────
        # Shape (1, FUSION_DIM) — broadcast over batch in forward()
        if not use_image:
            self.zero_cnn  = nn.Parameter(torch.zeros(1, FUSION_DIM))
        if not use_text:
            self.zero_text = nn.Parameter(torch.zeros(1, FUSION_DIM))
        if not use_num:
            self.zero_num  = nn.Parameter(torch.zeros(1, FUSION_DIM))

        # ── transformer + head ────────────────────────────────────────────
        self.transformer = CrossAttentionTransformer(
            n_layers=XATTN_LAYERS, d_model=FUSION_DIM,
            n_heads=XATTN_HEADS,   ff_dim=XATTN_FF_DIM,
            dropout=XATTN_DROPOUT, max_drop_path=MAX_DROP_PATH,
        )
        self.cls_head = ClassificationHead(
            d_model=FUSION_DIM, hidden_dim=CLS_HIDDEN_DIM,
            num_classes=NUM_CLASSES, dropout=dropout,
        )

    def forward(self, image, text_ids, num_feats):
        B = image.size(0)

        # image token
        if self.use_image:
            c = self.cnn_proj(self.cnn_encoder(image))          # (B, D)
        else:
            c = self.zero_cnn.expand(B, -1)                     # (B, D)

        # text token
        if self.use_text:
            t = self.text_proj(self.text_encoder(text_ids))     # (B, D)
        else:
            t = self.zero_text.expand(B, -1)                    # (B, D)

        # numerical token
        if self.use_num:
            n = self.num_proj(self.num_encoder(num_feats))      # (B, D)
        else:
            n = self.zero_num.expand(B, -1)                     # (B, D)

        seq    = self.transformer(c, t, n)                      # (B, 3, D)
        logits = self.cls_head(seq)                             # (B, C)
        return logits

    def get_parameter_groups(self, lr_backbone=1e-5, lr_head=1e-3):
        backbone, head = [], []
        for name, p in self.named_parameters():
            if self.cnn_encoder and (
                "cnn_encoder.stem"   in name or
                "cnn_encoder.layer1" in name or
                "cnn_encoder.layer2" in name
            ):
                backbone.append(p)
            else:
                head.append(p)
        return [{"params": backbone, "lr": lr_backbone},
                {"params": head,     "lr": lr_head}]


# ═══════════════════════════════════════════════════════════════════════════
# COMBO → FLAGS
# ═══════════════════════════════════════════════════════════════════════════

def combo_flags(name: str):
    return dict(
        use_image = "Image" in name,
        use_text  = "Text"  in name,
        use_num   = "Numerical" in name,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

class WarmupCosine(optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, T_max, eta_min=LR_ETA_MIN, last_epoch=-1):
        self.warmup = warmup; self.T_max = T_max; self.eta_min = eta_min
        super().__init__(opt, last_epoch)
    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            scale = 0.1 + 0.9 * e / max(self.warmup - 1, 1)
            return [b * scale for b in self.base_lrs]
        p = (e - self.warmup) / max(self.T_max - self.warmup, 1)
        c = 0.5 * (1 + math.cos(math.pi * min(p, 1)))
        return [self.eta_min + (b - self.eta_min) * c for b in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
# MIXUP
# ═══════════════════════════════════════════════════════════════════════════

def mixup(images, labels, num_classes, alpha=MIXUP_ALPHA):
    lam  = max(float(np.random.beta(alpha, alpha)), 1 - float(np.random.beta(alpha, alpha)))
    idx  = torch.randperm(images.size(0), device=images.device)
    mix  = lam * images + (1 - lam) * images[idx]
    oh   = torch.zeros(images.size(0), num_classes, device=labels.device)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    soft = lam * oh + (1 - lam) * oh[idx]
    return mix, soft


# ═══════════════════════════════════════════════════════════════════════════
# ONE EPOCH
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, num_classes):
    model.train()
    total_loss = 0.0
    for images, text_ids, num_feats, labels in loader:
        images    = images.to(DEVICE,    non_blocking=True)
        text_ids  = text_ids.to(DEVICE,  non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE,    non_blocking=True)
        images, soft = mixup(images, labels, num_classes)
        optimizer.zero_grad(set_to_none=True)
        logits   = model(images, text_ids, num_feats)
        loss     = -(soft * F.log_softmax(logits, 1)).sum(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader):
    val_criterion = nn.CrossEntropyLoss()   # no smoothing on val
    model.eval()
    correct = total = loss_sum = 0
    for images, text_ids, num_feats, labels in loader:
        images    = images.to(DEVICE, non_blocking=True)
        text_ids  = text_ids.to(DEVICE, non_blocking=True)
        num_feats = num_feats.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)
        logits = model(images, text_ids, num_feats)
        loss_sum += val_criterion(logits, labels).item()
        preds    = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return loss_sum / len(loader), correct / total


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN ONE COMBINATION
# ═══════════════════════════════════════════════════════════════════════════

def train_combo(combo_name, train_loader, val_loader, vocab_size, num_classes):
    flags = combo_flags(combo_name)
    print(f"\n{'─'*60}")
    print(f"  Combo: {combo_name}")
    print(f"  Flags: {flags}")
    print(f"{'─'*60}")

    model = AblationModel(
        vocab_size=vocab_size,
        pretrained=flags["use_image"],   # only load ImageNet if we use image
        **flags,
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = WarmupCosine(optimizer, WARMUP_EPOCHS, ABLATION_EPOCHS)

    val_accs = []
    t0 = time.time()

    for epoch in range(1, ABLATION_EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, num_classes)
        vl_loss, vl_acc = val_epoch(model, val_loader)
        scheduler.step()
        val_accs.append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{ABLATION_EPOCHS}  "
                  f"tr_loss={tr_loss:.4f}  val_acc={vl_acc*100:.2f}%  "
                  f"t={elapsed:.0f}s")

    print(f"  Final val acc: {val_accs[-1]*100:.2f}%  "
          f"Best val acc: {max(val_accs)*100:.2f}%")
    return val_accs


# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation(results: dict, save_path: str):
    """
    Reproduce the paper's "Validation Accuracy Comparison" chart.
    results: {combo_name: [val_acc per epoch]}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, accs in results.items():
        style = COMBO_STYLES[name]
        epochs = list(range(1, len(accs) + 1))
        ax.plot(epochs, accs, label=name, **style)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Validation Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_xlim(0, ABLATION_EPOCHS)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved ablation chart → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)

    # ── load data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    print(f"  Rows: {len(df)}  |  Classes: {df['label'].value_counts().to_dict()}")

    label_map   = build_label_map(df["label"])
    num_classes = len(label_map)

    # ── single 80/20 stratified split ────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

    # ── vocabulary — train text only ─────────────────────────────────────
    vocab = Vocabulary(max_size=VOCAB_SIZE).build(train_df["Clinical_Observation"].tolist())
    print(f"  Vocab size: {len(vocab)}")

    # ── scaler — train numerics only ─────────────────────────────────────
    def raw_num(sub_df):
        import pandas as _pd
        sex = _pd.get_dummies(sub_df["Patient_Sex"], prefix="sex").astype(float)
        for c in ["sex_Female", "sex_Male"]:
            if c not in sex.columns: sex[c] = 0.0
        return _pd.concat([sub_df[NUMERICAL_COLS].astype(float),
                            sex[["sex_Female", "sex_Male"]]], axis=1).values

    scaler = StandardScaler().fit(raw_num(train_df))

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = PneumoDataset(train_df, vocab, scaler, label_map, _train_transform())
    val_ds   = PneumoDataset(val_df,   vocab, scaler, label_map, _val_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── run all combinations ──────────────────────────────────────────────
    results   = {}
    json_path = os.path.join(RESULTS_DIR, "ablation_results.json")

    # Resume support: load previously completed combos
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        print(f"\nResuming — already done: {list(results.keys())}")

    for combo in COMBOS:
        if combo in results:
            print(f"  Skipping (already done): {combo}")
            continue

        val_accs = train_combo(combo, train_loader, val_loader,
                               vocab_size=len(vocab), num_classes=num_classes)
        results[combo] = val_accs

        # Save after each combo so a crash doesn't lose everything
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Progress saved → {json_path}")

        # Re-plot after each combo so you can inspect intermediate results
        plot_ablation(
            results,
            os.path.join(RESULTS_DIR, "ablation_curves.png"),
        )

    # ── final plot ────────────────────────────────────────────────────────
    plot_ablation(results, os.path.join(RESULTS_DIR, "ablation_curves.png"))

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  Final Val Accuracy Summary")
    print(f"{'═'*50}")
    for name, accs in sorted(results.items(), key=lambda x: -x[1][-1]):
        print(f"  {name:35s}: {max(accs)*100:.2f}% (best)  {accs[-1]*100:.2f}% (ep80)")
    print(f"{'═'*50}")
    print(f"\nDone. Chart saved to: {os.path.join(RESULTS_DIR, 'ablation_curves.png')}")


if __name__ == "__main__":
    main()
