"""
debug_startup.py  –  Run this INSTEAD of main.py to find the silent crash.

    python debug_startup.py
"""

import traceback, sys, os

print("Python:", sys.version)
print("CWD:", os.getcwd())

# ── Step 1: imports ───────────────────────────────────────────────────────
print("\n[1] Importing config...", end=" ", flush=True)
try:
    from config import CSV_PATH, DATA_ROOT, K_FOLDS, SEED, DEVICE
    print("OK")
    print(f"    CSV_PATH={CSV_PATH}  DATA_ROOT={DATA_ROOT}  DEVICE={DEVICE}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 2: CSV exists? ───────────────────────────────────────────────────
print("\n[2] Checking CSV file...", end=" ", flush=True)
csv_full = os.path.join(DATA_ROOT, CSV_PATH)
if not os.path.exists(csv_full):
    print(f"FAILED\n    File not found: {csv_full}")
    print("    Files in CWD:", os.listdir("."))
    sys.exit(1)
print(f"OK  ({os.path.getsize(csv_full)//1024} KB)")

# ── Step 3: read CSV ──────────────────────────────────────────────────────
print("\n[3] Reading CSV...", end=" ", flush=True)
try:
    import pandas as pd
    print("OK  pandas", pd.__version__)
    df = pd.read_csv(csv_full)
    print(f"OK  shape={df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Label counts:\n{df['label'].value_counts().to_string()}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 4: data_pipeline import ─────────────────────────────────────────
print("\n[4] Importing data_pipeline...", end=" ", flush=True)
try:
    from data_pipeline import load_dataframe, get_fold_dataloaders, build_label_map
    print("OK")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 5: load_dataframe ────────────────────────────────────────────────
print("\n[5] load_dataframe()...", end=" ", flush=True)
try:
    df2 = load_dataframe(csv_full)
    print(f"OK  rows={len(df2)}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 6: build_label_map ───────────────────────────────────────────────
print("\n[6] build_label_map()...", end=" ", flush=True)
try:
    label_map = build_label_map(df2["label"])
    print(f"OK  {label_map}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 7: model imports ─────────────────────────────────────────────────
print("\n[7] Importing models...", end=" ", flush=True)
try:
    from models import PneumoFusionNet
    print("OK")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 8: trainer imports ───────────────────────────────────────────────
print("\n[8] Importing trainer...", end=" ", flush=True)
try:
    from trainer import train_fold, finetune_fold, export_vocab_scaler
    print("OK")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 9: one fold dataloader ───────────────────────────────────────────
print("\n[9] Building fold 0 DataLoaders...", end=" ", flush=True)
try:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    train_loader, val_loader, vocab, scaler = get_fold_dataloaders(
        df2, fold=0, skf=skf, label_map=label_map, batch_size=4, num_workers=4, persistent_workers=True
    )
    print(f"OK  train={len(train_loader)} batches  val={len(val_loader)} batches")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 10: one batch forward pass ──────────────────────────────────────
print("\n[10] One forward pass...", end=" ", flush=True)
try:
    import torch
    model = PneumoFusionNet(
        num_classes=len(label_map), vocab_size=len(vocab), pretrained_cnn=False
    )
    model.eval()
    images, text_ids, num_feats, labels = next(iter(train_loader))
    with torch.no_grad():
        out = model(images, text_ids, num_feats)
    print(f"OK  output shape={out.shape}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

print("\n" + "="*50)
print("All checks passed — main.py should work.")
print("If main.py still exits silently, the crash is")
print("inside multiprocessing (num_workers > 0).")
print("Fix: add  num_workers=0  in config.py temporarily.")
print("="*50)
