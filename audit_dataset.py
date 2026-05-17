"""
audit_dataset.py  –  Detect label leakage inside the CSV

Run:
    python audit_dataset.py

What it checks
--------------
1.  image_path   – does the file path string contain the class name?
2.  Clinical_Observation text – does it contain the class name verbatim?
3.  Single-modality baseline – train a tiny LogisticRegression on EACH
    modality alone (text bag-of-words, numerics) and report accuracy.
    If any single modality alone gives >95%, that modality is leaking.
4.  Class distribution and duplicate rows.
5.  Text vocabulary overlap between train and val (within one fold).
"""

import os, re, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

CSV_PATH  = r"E:\4thYearProjectCoding\dataset\unified_dataset_new3.csv"
TEXT_COL  = "Clinical_Observation"
PATH_COL  = "image_path"
LABEL_COL = "label"
SEED      = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def banner(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def warn(msg):
    print(f"  ⚠️  LEAK DETECTED: {msg}")


def ok(msg):
    print(f"  ✓  {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════

banner("Loading CSV")
if not os.path.exists(CSV_PATH):
    sys.exit(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df)}  |  Cols: {list(df.columns)}")
print(f"  Class counts:\n{df[LABEL_COL].value_counts().to_string()}")

le = LabelEncoder()
y  = le.fit_transform(df[LABEL_COL])
classes = le.classes_


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1 — image_path contains class name?
# ═══════════════════════════════════════════════════════════════════════════

banner("CHECK 1 — image_path label leakage")
path_leak_count = 0
for cls in classes:
    slug = cls.lower().replace(" ", "").replace("_", "")
    leaked = df[PATH_COL].str.lower().str.replace(r"[\s_]", "", regex=True).str.contains(slug)
    n = leaked.sum()
    if n > 0:
        pct = n / len(df) * 100
        warn(f"class '{cls}' appears in image_path for {n}/{len(df)} rows ({pct:.1f}%)")
        print(f"     Example path: {df.loc[leaked, PATH_COL].iloc[0]}")
        path_leak_count += n

if path_leak_count == 0:
    ok("No class name found verbatim in image_path strings.")
else:
    print(f"\n  → image_path encodes the class label directly.")
    print(f"    The CNN encoder sees the FILE PATH only indirectly (it loads pixels),")
    print(f"    but the text/numerical encoders do NOT see paths — so this alone")
    print(f"    does not explain 99% val accuracy unless paths were accidentally fed")
    print(f"    as text features. Check your data_pipeline image loading logic.")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2 — Clinical_Observation text contains class name verbatim?
# ═══════════════════════════════════════════════════════════════════════════

banner("CHECK 2 — Clinical_Observation text label leakage")
text_leak_count = 0
for cls in classes:
    slug = cls.lower()
    leaked = df[TEXT_COL].str.lower().str.contains(re.escape(slug), na=False)
    n = leaked.sum()
    if n > 0:
        pct = n / len(df) * 100
        warn(f"class '{cls}' appears verbatim in text for {n}/{len(df)} rows ({pct:.1f}%)")
        print(f"     Example: \"{df.loc[leaked, TEXT_COL].iloc[0][:120]}\"")
        text_leak_count += n

# also check common abbreviations
abbrevs = {
    "covid": "Corona Virus Disease",
    "covid-19": "Corona Virus Disease",
    "sars-cov": "Corona Virus Disease",
    "tb ": "Tuberculosis",
    "tuberculosis": "Tuberculosis",
    "bacterial": "Bacterial Pneumonia",
    "viral": "Viral Pneumonia",
}
for term, cls in abbrevs.items():
    leaked = df[TEXT_COL].str.lower().str.contains(re.escape(term), na=False)
    n = leaked.sum()
    if n > 0:
        rows_of_cls    = (df[LABEL_COL] == cls).sum()
        rows_other_cls = leaked.sum() - (leaked & (df[LABEL_COL] == cls)).sum()
        pct_cls   = (leaked & (df[LABEL_COL] == cls)).sum() / max(rows_of_cls, 1) * 100
        if pct_cls > 50:
            warn(
                f"term '{term}' appears in {pct_cls:.0f}% of '{cls}' rows "
                f"and in {rows_other_cls} rows of other classes — "
                f"likely a discriminative keyword leaked into text."
            )

if text_leak_count == 0:
    ok("No class name found verbatim in Clinical_Observation text.")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3 — Single-modality logistic regression baselines (one fold)
# ═══════════════════════════════════════════════════════════════════════════

banner("CHECK 3 — Single-modality baseline accuracy (1 fold, logistic regression)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(skf.split(df, y))

train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
y_tr, y_val      = y[train_idx], y[val_idx]

# 3a — text only
print("\n  3a. Text-only (TF-IDF + LogReg):")
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_tr_txt  = tfidf.fit_transform(train_df[TEXT_COL].fillna(""))
X_val_txt = tfidf.transform(val_df[TEXT_COL].fillna(""))
lr_txt = LogisticRegression(max_iter=1000, random_state=SEED)
lr_txt.fit(X_tr_txt, y_tr)
acc_txt = accuracy_score(y_val, lr_txt.predict(X_val_txt))
print(f"     Val accuracy (text only): {acc_txt*100:.2f}%")
if acc_txt > 0.95:
    warn(f"Text alone achieves {acc_txt*100:.1f}% — the text column is leaking the label!")
    # show top discriminative terms per class
    feat_names = np.array(tfidf.get_feature_names_out())
    print("\n     Top discriminative terms per class:")
    for i, cls in enumerate(classes):
        coef = lr_txt.coef_[i]
        top  = feat_names[np.argsort(coef)[-10:]][::-1]
        print(f"       {cls:30s}: {', '.join(top)}")
else:
    ok(f"Text alone: {acc_txt*100:.1f}% — acceptable.")

# 3b — numerics only
NUMERICAL_COLS = ["Patient_Age", "WBC (x10^9/L)", "NEUT%", "LYMP%", "NLR", "CRP (mg/L)", "PCT (ng/mL)"]
print("\n  3b. Numerics-only (StandardScaler + LogReg):")
try:
    sex_tr  = pd.get_dummies(train_df["Patient_Sex"], prefix="sex").reindex(columns=["sex_Female","sex_Male"], fill_value=0)
    sex_val = pd.get_dummies(val_df["Patient_Sex"],   prefix="sex").reindex(columns=["sex_Female","sex_Male"], fill_value=0)
    X_tr_num  = np.hstack([train_df[NUMERICAL_COLS].astype(float).values, sex_tr.values])
    X_val_num = np.hstack([val_df[NUMERICAL_COLS].astype(float).values,   sex_val.values])
    sc = StandardScaler().fit(X_tr_num)
    lr_num = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_num.fit(sc.transform(X_tr_num), y_tr)
    acc_num = accuracy_score(y_val, lr_num.predict(sc.transform(X_val_num)))
    print(f"     Val accuracy (numerics only): {acc_num*100:.2f}%")
    if acc_num > 0.95:
        warn(f"Numerics alone achieve {acc_num*100:.1f}% — numerical features are leaking the label!")
    else:
        ok(f"Numerics alone: {acc_num*100:.1f}% — acceptable.")
except Exception as e:
    print(f"     Could not run numerics baseline: {e}")

# 3c — text + numerics (no image)
print("\n  3c. Text + Numerics (no image):")
from scipy.sparse import hstack as sp_hstack
X_tr_both  = sp_hstack([X_tr_txt,  sc.transform(X_tr_num)])
X_val_both = sp_hstack([X_val_txt, sc.transform(X_val_num)])
lr_both = LogisticRegression(max_iter=1000, random_state=SEED)
lr_both.fit(X_tr_both, y_tr)
acc_both = accuracy_score(y_val, lr_both.predict(X_val_both))
print(f"     Val accuracy (text + numerics, no image): {acc_both*100:.2f}%")
if acc_both > 0.95:
    warn(
        f"Text+numerics alone achieve {acc_both*100:.1f}% WITHOUT ANY IMAGE.\n"
        f"     The model does not need the CT scan to classify — the non-image\n"
        f"     modalities already contain the answer. This is the root cause of\n"
        f"     near-perfect val accuracy."
    )
else:
    ok(f"Text+numerics (no image): {acc_both*100:.1f}%.")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4 — duplicate rows / near-duplicate images
# ═══════════════════════════════════════════════════════════════════════════

banner("CHECK 4 — Duplicate rows")
dup_text  = df.duplicated(subset=[TEXT_COL]).sum()
dup_path  = df.duplicated(subset=[PATH_COL]).sum()
dup_all   = df.duplicated().sum()

print(f"  Exact duplicate rows (all cols)     : {dup_all}")
print(f"  Duplicate Clinical_Observation text : {dup_text}")
print(f"  Duplicate image_path                : {dup_path}")

if dup_path > 0:
    warn(
        f"{dup_path} rows share the same image_path. "
        "If the same image appears in both train and val (across folds), "
        "that is image-level data leakage."
    )
    # show how many unique paths per class
    print("\n  Unique paths per class:")
    print(df.groupby(LABEL_COL)[PATH_COL].nunique().to_string())

if dup_text > 0:
    warn(
        f"{dup_text} rows share the same Clinical_Observation text. "
        "If a patient's identical note appears in both train and val, "
        "the model memorises the note-to-label mapping."
    )

if dup_path == 0 and dup_text == 0 and dup_all == 0:
    ok("No duplicate rows found.")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5 — text length / vocabulary size
# ═══════════════════════════════════════════════════════════════════════════

banner("CHECK 5 — Text column statistics")
df["_text_len"] = df[TEXT_COL].str.split().str.len()
print(f"  Mean text length (words) : {df['_text_len'].mean():.1f}")
print(f"  Median text length       : {df['_text_len'].median():.0f}")
print(f"  Max text length          : {df['_text_len'].max()}")
print(f"  % rows with <5 words     : {(df['_text_len'] < 5).mean()*100:.1f}%")
print(f"\n  Unique texts             : {df[TEXT_COL].nunique()} / {len(df)}")

if df[TEXT_COL].nunique() < len(df) * 0.5:
    warn(
        "Fewer than 50% of texts are unique. Many patients share identical "
        "clinical notes — the model is likely memorising text→label mappings."
    )

print(f"\n  Sample texts per class:")
for cls in classes:
    sample = df.loc[df[LABEL_COL] == cls, TEXT_COL].iloc[0]
    print(f"  [{cls}] \"{str(sample)[:120]}\"")

df.drop(columns=["_text_len"], inplace=True)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

banner("SUMMARY — action items")
print("""
  If CHECK 3a/3b/3c showed >95% accuracy from text or numerics alone:
  ─────────────────────────────────────────────────────────────────────
  The Clinical_Observation text or lab values are directly encoding the
  diagnosis (e.g. a note that says 'COVID-19 confirmed', or lab values
  that are perfectly class-discriminative because they were recorded
  AFTER the diagnosis).  Options:

  A) Inspect and sanitise the text column — remove any mention of the
     diagnosis, treatment, or final conclusion; keep only presenting
     symptoms observed on admission.

  B) Inspect lab value distributions per class — if WBC / CRP / PCT
     perfectly separate classes it means they were sampled post-diagnosis
     rather than at admission.

  C) If the dataset is synthetic / augmented, check that the generator
     did not hard-code class-specific numerical ranges or text templates
     that trivially separate the classes.

  If CHECK 4 showed duplicate image paths:
  ─────────────────────────────────────────
  Group images by patient ID (if available) and ensure all images from
  the same patient fall in the SAME fold (GroupKFold instead of
  StratifiedKFold).
""")
