# PneumoFusion-Net  
### Multi-Modal Deep Learning for Precise Pneumonia Diagnosis  
*Cross-Attention Transformer architecture (Wang et al., 2025)*

---

## Architecture Overview

```
 ┌─────────────────────────────────────────────────────────┐
 │                  Input Modalities                        │
 └─────┬──────────────────┬──────────────────┬─────────────┘
       │                  │                  │
  CT Image          Clinical Text      Lab Numerics
  Grayscale         Patient Obs.       WBC, CRP, PCT…
       │                  │                  │
 ResNet50+GCSA     BiLSTM+Attn        MLP+Residual
 Channel+Spatial   Reads text in      Nonlinear patterns
 Attention on CT   both directions    in lab values
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
               Feature Fusion Layer
          Concatenation + Linear Projection
                          │
        ┌─────────────────────────────────────┐
        │     Cross-Attention Transformer      │
        │                                     │
        │  ① Cross-modal query-key-value attn │
        │    (each modality attends to others) │
        │                                     │
        │  ② Multi-head self-attention         │
        │    (intra-modal feature refinement)  │
        │                                     │
        │  ③ FFN + Layer Normalisation         │
        │    (unchanged from standard Tx)      │
        └─────────────────┬───────────────────┘
                          │
               Classification Head
              Pooling → FC → Softmax
                          │
 ┌──────┬──────────┬────────┬─────────────┬───────┐
 Bact.  Covid-19  Normal  Tuberculosis   Viral
```

---

## Key Design Decisions

| Component | Design Choice | Justification |
|---|---|---|
| CNN | ResNet50 + GCSA + DSC | GCSA jointly models channel & spatial importance for subtle CT lesions (GGOs, consolidations). DSC reduces FLOPs by ~2× |
| Text | BiLSTM + Additive Attention | Bidirectional context captures both present illness history and findings. Additive attention highlights symptom phrases |
| Numerics | MLP + Residual | Residual connections prevent gradient vanishing; standardised inputs ensure fair weighting of WBC, CRP, PCT etc. |
| Fusion | Cross-Attention Transformer | Each modality queries the other two as key-value context, then all three undergo self-attention refinement |
| Loss | CrossEntropy + Label Smoothing 0.1 | Prevents over-confident predictions, reduces misclassification at class boundaries (viral ↔ bacterial) |
| Optimiser | AdamW + CosineAnnealingWarmRestarts | Weight-decay prevents overfitting; warm restarts help escape local minima |
| Fine-tuning | Differential LR (backbone×0.1) | Preserves ImageNet representations while adapting to medical domain |

---

## File Structure

```
pneumofusion_net/
├── config.py          ← All hyper-parameters (single source of truth)
├── data_pipeline.py   ← CSV loading, vocabulary, augmentation, Dataset, DataLoaders
├── models.py          ← GCSA · CNNEncoder · BiLSTMEncoder · MLPEncoder
│                         FeatureFusion · CrossAttentionTransformer · ClassificationHead
│                         PneumoFusionNet (full model)
├── trainer.py         ← train_one_epoch · validate · EarlyStopping
│                         train_fold · finetune_fold
├── evaluate.py        ← evaluate_model · confusion matrix · training curves
│                         aggregate_cv_results
├── main.py            ← 5-fold CV + fine-tuning orchestration (entry point)
├── inference.py       ← Single-sample and batch CSV inference
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
```
project_root/
├── unified_dataset_new.csv
└── images/
    ├── Bacterial Pneumonia/
    │   └── *.jpeg
    ├── Normal/
    ├── Tuberculosis/
    └── Viral Pneumonia/
```

Edit `config.py` → `DATA_ROOT` and `CSV_PATH` if paths differ.

### 3. Train (full 5-fold CV + fine-tuning)
```bash
python main.py
```

### 4. Train a single fold only
```bash
python main.py --fold 0
```

### 5. Skip fine-tuning
```bash
python main.py --no-finetune
```

### 6. Evaluate an existing checkpoint
```bash
python main.py --eval-only --fold 0 \
    --ckpt outputs/checkpoints/fold0_best.pt
```

### 7. Inference on a new patient
```bash
python inference.py \
    --ckpt outputs/checkpoints/fold0_finetuned.pt \
    --image /data/ct_scan.jpg \
    --text "Bilateral GGOs with peripheral distribution" \
    --age 55 --sex Male \
    --wbc 9.2 --neut 68 --lymp 22 --nlr 3.1 --crp 45 --pct 0.3
```

### 8. Batch inference from CSV
```bash
python inference.py \
    --ckpt outputs/checkpoints/fold0_finetuned.pt \
    --csv new_patients.csv
```

---

## Training Configuration (config.py)

| Parameter | Default | Notes |
|---|---|---|
| `IMAGE_SIZE` | 224 | CT resize dimension |
| `BATCH_SIZE` | 32 | Tune down if OOM |
| `EPOCHS` | 80 | + early stopping (patience=10) |
| `LEARNING_RATE` | 1e-3 | AdamW base LR |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `K_FOLDS` | 5 | Stratified CV |
| `FINETUNE_EPOCHS` | 20 | Post-CV fine-tuning |
| `FINETUNE_LR` | 1e-4 | Backbone LR = 1e-5 |
| `MIXED_PRECISION` | True | AMP fp16/fp32 (CUDA only) |
| `FUSION_DIM` | 256 | Transformer embedding dim |
| `XATTN_HEADS` | 8 | Multi-head attention heads |
| `XATTN_LAYERS` | 2 | Number of Cross-Attn layers |

---

## Outputs

After training, the following are written:

```
outputs/
├── checkpoints/
│   ├── fold0_best.pt          ← best validation checkpoint (per fold)
│   ├── fold0_finetuned.pt     ← fine-tuned checkpoint
│   └── best_fold_meta.pt      ← vocab + scaler (for inference)
├── logs/
│   ├── fold0_history.json     ← per-epoch train/val metrics
│   └── fold0_finetune_history.json
└── results/
    ├── fold0_confmat.png
    ├── fold0_confmat_finetuned.png
    ├── fold0_curves.png
    ├── fold0_metrics.json
    └── CV_pretrain_summary.json   ← mean±std across all folds
```

---

## Model Complexity

| Module | Parameters (approx.) |
|---|---|
| ResNet50 + GCSA + DSC | ~24 M |
| BiLSTM Text Encoder | ~3 M |
| MLP Numerical Encoder | ~0.05 M |
| Feature Fusion + Projection | ~0.5 M |
| Cross-Attention Transformer (×2) | ~2 M |
| Classification Head | ~0.07 M |
| **Total** | **~30 M** |

---

## Reference

Wang Y, Liu C, Fan Y, Niu C, Huang W, Pan Y, Li J, Wang Y and Li J (2025)  
*A multi-modal deep learning solution for precise pneumonia diagnosis: the PneumoFusion-Net model.*  
Front. Physiol. 16:1512835.  doi: 10.3389/fphys.2025.1512835
