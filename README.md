<div align="center">

# 🫁 PneumoFusion-Net

### Multimodal Deep Learning Framework for Enhanced Pneumonia Detection

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Manuscript-blue?style=flat&logo=arxiv)](YOUR-PAPER-LINK)

**99.18% mean accuracy · 99.79% specificity · 99.99% ROC-AUC**  
*5-fold cross-validation · 8,900 CT scans · 5 disease classes*

[🚀 Live Demo](https://xxsolitude-pneumofusion-net.hf.space/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training](#-training)
- [Web Application](#-web-application)
- [Explainability](#-explainability-grad-cam--shap)
- [Dataset](#-dataset)
- [Citation](#-citation)

---

## 🔬 Overview

PneumoFusion-Net is a **multimodal deep learning framework** that diagnoses five respiratory diseases by fusing three data streams simultaneously — exactly the way a clinician does.

| Modality | Module | Contribution |
|---|---|---|
| 🫁 CT Scan | ResNet50 + GCSA + DSC | ~34% |
| 🧬 Lab Values | MLP + Residual Connections | ~33.5% |
| 📝 Clinical Text | BiLSTM + Additive Attention | ~32.5% |

> **Key advantage over PneumoFusion-Net (Wang et al., 2025):** our Cross-Attention Transformer enables *explicit directional interaction* between modalities (each queries the other two), producing a far more **balanced modality utilisation** (34%/33.5%/32.5%) vs the 45%/33%/12%/10% seen in Swin Transformer fusion. This improves robustness when any single modality is noisy or missing.

**Detectable conditions:**
`Bacterial Pneumonia` · `Viral Pneumonia` · `COVID-19` · `Tuberculosis` · `Normal`

---

## 🏗️ Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     INPUT MODALITIES                          │
 └──────────┬──────────────────┬─────────────────┬─────────────┘
            │                  │                  │
       CT Scan             Clinical Text      Lab Values
     (1×224×224)          (token sequence)   (WBC,CRP,PCT…)
            │                  │                  │
   ┌────────▼────────┐ ┌───────▼───────┐ ┌───────▼──────────┐
   │  Modified       │ │  BiLSTM +     │ │  MLP + Residual  │
   │  ResNet50       │ │  Additive     │ │  Connections     │
   │  + DSC + GCSA   │ │  Attention    │ │  (9 features)    │
   └────────┬────────┘ └───────┬───────┘ └───────┬──────────┘
            │  (512-dim)       │ (512-dim)        │ (64-dim)
            └──────────────────┴──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Feature Fusion    │
                    │  Concat → Linear    │
                    │  Projection (256)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────────┐
              │     Cross-Attention Transformer      │
              │                                      │
              │  ① Cross-modal QKV attention         │
              │     each modality queries others     │
              │                                      │
              │  ② Multi-head self-attention          │
              │     intra-modal refinement           │
              │                                      │
              │  ③ FFN + LayerNorm + StochasticDepth │
              └────────────────┬────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Classification     │
                    │  Head (learnable    │
                    │  modality weights)  │
                    └──────────┬──────────┘
                               │
           ┌───────┬───────┬───┴───┬────────────┬──────┐
        Bacterial COVID-19 Normal Tuberculosis  Viral
```

### Sub-modules

| Module | File | Description |
|---|---|---|
| GCSA + DSC | `models/attention.py` | Channel + spatial attention, depthwise separable conv |
| CNN Encoder | `models/cnn_encoder.py` | ResNet50 adapted for single-channel CT |
| Text Encoder | `models/text_encoder.py` | Positional encoding → BiLSTM → additive attention |
| Numerical Encoder | `models/numerical_encoder.py` | Residual MLP for lab values |
| Feature Fusion | `models/fusion.py` | Concat + 2-layer linear projection |
| Transformer | `models/transformer.py` | Cross-modal attention + stochastic depth |
| Classification Head | `models/classification_head.py` | Learnable softmax-weighted modality pooling |
| Full Model | `models/pneumofusion_net.py` | All modules assembled |

---

## 📊 Results

### 5-Fold Cross-Validation (Average)

| Metric | Value |
|---|---|
| **Accuracy** | **99.18%** |
| **Precision** | **99.18%** |
| **Recall** | **99.18%** |
| **F1-Score** | **99.18%** |
| **Specificity** | **99.79%** |

### Best Fold — Fold 3

| Metric | Value |
|---|---|
| Accuracy | 99.55% |
| Precision | 99.55% |
| Recall | 99.55% |
| Macro F1-Score | 99.55% |
| Weighted F1-Score | 99.55% |
| **ROC-AUC** | **99.99%** |
| Macro Specificity | 99.89% |

### Per-Class Sensitivity (Fold 3)

| Class | Sensitivity |
|---|---|
| Bacterial Pneumonia | **100.00%** |
| Tuberculosis | **100.00%** |
| COVID-19 | 99.72% |
| Viral Pneumonia | 99.16% |
| Normal | 98.88% |

### Ablation Study

| Configuration | Accuracy |
|---|---|
| **Image + Text + Numerical (ours)** | **99.33%** |
| Image + Text | 98.43% |
| Image + Numerical | 97.81% |
| Text + Numerical | 96.01% |
| Image Only | 90.90% |
| Text Only | 89.66% |
| Numerical Only | 78.60% |

### Comparison with State of the Art

| Model | Accuracy | Fusion Strategy | Modality Balance |
|---|---|---|---|
| **Ours** | **99.18%** | Cross-Attention | 34% / 33.5% / 32.5% ✅ |
| PneumoFusion-Net (Wang et al., 2025) | 98.96% | Swin Transformer | 45% / 33% / 12% / 10% |

---

## 📁 Project Structure

```
📁 PneumoFusion-Net/
│
├── config.py                   ← All hyperparameters (single source of truth)
├── data_pipeline.py            ← Dataset, vocab, augmentation, DataLoaders
├── trainer.py                  ← Training loop, Mixup, WarmupCosine scheduler
├── evaluate.py                 ← Metrics, confusion matrix, modality weight logging
├── main.py                     ← 5-fold CV + best-fold fine-tuning orchestration
├── inference.py                ← Single-sample & batch inference CLI
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── attention.py            ← ChannelAttention, SpatialAttention, GCSA, DSC
│   ├── cnn_encoder.py          ← CNNImageEncoder (ResNet50 + GCSA + DSC)
│   ├── text_encoder.py         ← PositionalEncoding, BiLSTM, AdditiveAttention
│   ├── numerical_encoder.py    ← ResidualBlock, MLPNumericalEncoder
│   ├── fusion.py               ← FeatureFusion
│   ├── transformer.py          ← CrossModalAttention, StochasticDepth, Transformer
│   ├── classification_head.py  ← Learnable modality-weighted classification head
│   └── pneumofusion_net.py     ← Full assembled model
│
├── outputs/                    ← Auto-created at runtime
│   ├── checkpoints/            ← fold{n}_best.pt, fold{n}_finetuned.pt
│   │   ├── tokenizer.json      ← Vocabulary (word→index) for inference
│   │   ├── scaler.pkl          ← Fitted StandardScaler for inference
│   │   └── label_map.json      ← {class_name: int_index}
│   ├── logs/                   ← Per-epoch JSON training history
│   └── results/                ← Confusion matrices, curves, metrics JSON
│
└── webapp/
    ├── docker-compose.yml
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    ├── requirements_webapp.txt
    ├── run_webapp.sh            ← One-command local start
    ├── backend/
    │   ├── inference_engine.py ← Model loading, preprocessing, Grad-CAM
    │   ├── report_generator.py ← OpenAI GPT-4.1 report + rule-based fallback
    │   └── api.py              ← FastAPI: /predict /gradcam /report /health
    └── frontend/
        └── app.py              ← Streamlit UI (Home · Diagnose · About)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- CUDA 12.x (recommended) or CPU
- 8 GB+ VRAM for training, 4 GB for inference

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Souptik-123/pneumofusion-net.git
cd pneumofusion-net

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install webapp dependencies
pip install -r webapp/requirements_webapp.txt
```

---

## 🏋️ Training

### Prepare data

Place your CSV and images at the project root:

```
pneumofusion-net/
├── unified_dataset_new1.csv
└── images/
    ├── Bacterial Pneumonia/
    ├── Corona Virus Disease/
    ├── Normal/
    ├── Tuberculosis/
    └── Viral Pneumonia/
```

Edit `config.py` if your CSV path or column names differ.

### Run full 5-fold CV + fine-tune best fold

```bash
python main.py
```

### Other training options

```bash
# Single fold only
python main.py --fold 0

# Skip fine-tuning
python main.py --no-finetune

# Without ImageNet pretrained weights
python main.py --no-pretrain

# Evaluate existing checkpoint
python main.py --eval-only --fold 0 --ckpt outputs/checkpoints/fold0_best.pt
```

### Key hyperparameters (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `LEARNING_RATE` | `1e-4` | AdamW base LR |
| `WEIGHT_DECAY` | `1e-3` | L2 regularisation |
| `WARMUP_EPOCHS` | `3` | Linear warmup before cosine decay |
| `DROPOUT_RATE` | `0.5` | Applied throughout all encoders |
| `LABEL_SMOOTHING` | `0.25` | Prevents over-confidence |
| `MIXUP_ALPHA` | `0.4` | Mixup augmentation strength |
| `EARLY_STOP_PAT` | `8` | Patience epochs |
| `BATCH_SIZE` | `32` | |
| `EPOCHS` | `80` | Maximum epochs |

---

## 🌐 Web Application

### Local (one command)

```bash
chmod +x webapp/run_webapp.sh
./webapp/run_webapp.sh
```

- **UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

### Docker (production)

```bash
# Set your OpenAI API key (optional — fallback report works without it)
export OPENAI_API_KEY=sk-...

docker-compose -f webapp/docker-compose.yml up --build
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `GET` | `/classes` | List of class names |
| `POST` | `/predict` | Prediction only (JSON) |
| `POST` | `/gradcam` | Prediction + Grad-CAM images (base64) |
| `POST` | `/report` | Full pipeline: predict + Grad-CAM + AI report |

### Inference CLI

```bash
python inference.py \
    --ckpt outputs/checkpoints/fold0_finetuned.pt \
    --image /path/to/ct.jpg \
    --text "Bilateral GGOs with peripheral distribution, fever 38.5°C" \
    --age 55 --sex Male \
    --wbc 9.2 --neut 68 --lymp 22 --nlr 3.1 --crp 45 --pct 0.3
```

---

## 🔍 Explainability: Grad-CAM + SHAP

### Grad-CAM

Gradient-weighted Class Activation Maps highlight which CT regions drove the prediction:

| Disease | Model focus |
|---|---|
| COVID-19 | Peripheral ground-glass opacities (bilateral) |
| Tuberculosis | Upper-lobe lesions and infiltrative patterns |
| Bacterial Pneumonia | Central consolidation regions |
| Viral Pneumonia | Diffuse opacity, lower lobes |
| Normal | Even, unfocused activation (no abnormality) |

### SHAP Feature Importance

SHAP values (DeepExplainer, zero-vector baseline) quantify per-feature contributions:

| Feature | Direction for TB |
|---|---|
| WBC | ↓ pushes toward TB (−0.0065) |
| CRP | ↑ supports TB (+0.0049) |
| PCT | ↓ pushes toward TB (−0.0045) |
| LYMP% | ↑ supports TB (+0.0021) |
| NEUT% | ↓ pushes toward TB (−0.0016) |

### Learnable Modality Weights

After training, inspect the learned importance ratio:

```python
weights = model.cls_head.get_modality_weights()
# {'CT image (CNN)': 0.34, 'Clinical text': 0.325, 'Lab numerics': 0.335}
```

---

## 📦 Dataset

The dataset follows the construction approach of Wang et al. (2025):

| Class | Images |
|---|---|
| Normal | 1,780 |
| Bacterial Pneumonia | 1,780 |
| Viral Pneumonia | 1,780 |
| COVID-19 | 1,780 |
| Tuberculosis | 1,780 |
| **Total** | **8,900** |

- **Clinical text:** avg 43 words, covering chief complaints, present illness, physical exam, imaging findings
- **Lab values:** WBC, NEUT%, LYMP%, NLR, CRP, PCT (6 features + age + sex = 9 total)
- **Split:** 5-fold stratified CV (80/20 per fold), vocab and scaler fitted on training split only

---

## 🛠️ Regularisation Summary

Several techniques were combined to address overfitting on the ~8,900-sample dataset:

| Technique | Setting | Effect |
|---|---|---|
| Dropout | p = 0.5 | Prevents neuron co-adaptation |
| Weight Decay | 1e-3 | Strong L2 penalty |
| Label Smoothing | 0.25 | Prevents over-confidence |
| Mixup | α = 0.4 | Soft-label interpolation |
| Stochastic Depth | max 10% | Random residual branch dropping |
| RandomErasing | p = 0.5 | Simulates occluded CT regions |
| RandomPerspective | p = 0.2 | Structural distortion diversity |
| Early Stopping | patience = 8 | Stops before overfitting deepens |

---

## 💻 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| CPU | Intel Core i5-13450HX (2.40 GHz) |
| OS | Windows 11 |
| Framework | PyTorch 2.9.1 + CUDA 12.8 |
| Language | Python 3.11.4 |

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{dey2025multimodal,
  title     = {Multimodal Fusion of Chest CT and Clinical Indicators
               for Enhanced Pneumonia Detection},
  author    = {Dey, Souptik and Barik, Sidhanta and Kumari, Divya and
               Kumari, Astha and Gourav, Adarsh and Kumar, Rohit},
  journal   = {YOUR-JOURNAL/CONFERENCE},
  year      = {2025},
  url       = {YOUR-PAPER-LINK}
}
```

This work builds upon:

```bibtex
@article{wang2025pneumofusion,
  title   = {A multi-modal deep learning solution for precise pneumonia
             diagnosis: the PneumoFusion-Net model},
  author  = {Wang, Yujie and Liu, Can and Fan, Yinghan and others},
  journal = {Frontiers in Physiology},
  volume  = {16},
  pages   = {1512835},
  year    = {2025},
  doi     = {10.3389/fphys.2025.1512835}
}
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Medical Disclaimer:** This system is a research prototype and is **NOT** approved as a medical device. All predictions are AI-assisted and must be reviewed by a qualified healthcare professional before any clinical action is taken.

---


⭐ Star this repo if you found it useful!

</div>
