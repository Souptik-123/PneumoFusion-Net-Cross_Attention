"""
debug_csv.py  –  Find exactly what crashes during pd.read_csv()

    python debug_csv.py
"""
import sys, os, traceback
import pandas as pd

CSV_PATH = "unified_dataset_new2.csv"

print(f"pandas version : {pd.__version__}")
print(f"CSV file size  : {os.path.getsize(CSV_PATH)//1024} KB")

# ── Step 1: read with python engine (tolerates encoding issues) ───────────
print("\n[1] Reading with python engine + latin-1 encoding...", end=" ", flush=True)
try:
    df = pd.read_csv(CSV_PATH, engine="python", encoding="latin-1", on_bad_lines="skip")
    print(f"OK  shape={df.shape}")
    print(f"    Columns : {list(df.columns)}")
    print(f"    Dtypes  :\n{df.dtypes.to_string()}")
    print(f"    Null counts:\n{df.isnull().sum().to_string()}")
    print(f"    Label counts:\n{df['label'].value_counts().to_string()}")
except Exception:
    print("FAILED"); traceback.print_exc(); sys.exit(1)

# ── Step 2: check for any column whose string repr triggers a print ───────
print("\n[2] Checking for embedded terminal escape / box-drawing chars...", end=" ", flush=True)
try:
    for col in df.columns:
        sample = df[col].astype(str).str.contains(r'[═╔╗╚╝║]', regex=True)
        if sample.any():
            print(f"\n    WARNING: column '{col}' contains box-drawing characters!")
            print(f"    Example: {df.loc[sample, col].iloc[0][:200]}")
    print("OK")
except Exception:
    print("FAILED"); traceback.print_exc()

# ── Step 3: try default read (c engine, utf-8) ────────────────────────────
print("\n[3] Reading with default engine (utf-8)...", end=" ", flush=True)
try:
    df2 = pd.read_csv(CSV_PATH)
    print(f"OK  shape={df2.shape}")
except UnicodeDecodeError as e:
    print(f"\n    ENCODING ERROR: {e}")
    print("    Fix: open config.py and your data_pipeline.py and add encoding='latin-1'")
    print("    OR re-save the CSV as UTF-8 in Excel: File > Save As > CSV UTF-8")
except Exception:
    print("FAILED"); traceback.print_exc()

# ── Step 4: show first 3 rows safely ─────────────────────────────────────
print("\n[4] First 3 rows (repr, no print rendering):")
try:
    for i, row in df.head(3).iterrows():
        print(f"  Row {i}: { {k: str(v)[:60] for k, v in row.items()} }")
except Exception:
    traceback.print_exc()

print("\n[5] Check image_path column samples:")
try:
    for cls in df['label'].unique():
        sample_path = df.loc[df['label']==cls, 'image_path'].iloc[0]
        exists = os.path.exists(sample_path)
        print(f"  [{cls}] path='{sample_path[:80]}'  exists={exists}")
except Exception as e:
    print(f"  Could not check paths: {e}")

print("\nDone. Share this full output to diagnose further.")
