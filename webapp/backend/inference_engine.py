"""
webapp/backend/inference_engine.py
───────────────────────────────────
Handles ALL model inference for the web application:
  • ModelLoader        – singleton that loads PneumoFusionNet + artefacts once
  • preprocess_image   – PIL → tensor
  • encode_text        – raw string → token ids
  • encode_numerics    – dict → standardised numpy array → tensor
  • run_inference      – full multimodal forward pass → predictions
  • GradCAMPlusPlus    – Grad-CAM++ on cnn_encoder.gcsa → heatmap overlay
  • SHAPExplainer      – DeepExplainer SHAP values for numerical + modality weights
  • FusedExplainer     – Grad-CAM++ map re-weighted by SHAP-derived image modality
                         contribution; also returns per-feature SHAP bar data

Why GradCAM++ instead of vanilla GradCAM
-----------------------------------------
Vanilla Grad-CAM uses global-average-pooled gradients (one scalar weight per
channel).  When the model's cross-attention fusion already summarises spatial
information, those scalars can all be close to zero → the heatmap collapses to
a uniform blob or sticks at the image border.

Grad-CAM++ weights each *spatial position* separately before pooling, using the
second-order gradient terms.  This produces a sharper, more localised activation
map that is robust even when the raw gradient magnitudes are small.

Why combine with SHAP
---------------------
SHAP (SHapley Additive exPlanations) operates on the *numerical* feature space
via DeepExplainer, giving us two things:
  1. Per-feature importance values (WBC, CRP, NLR …) in the direction of the
     predicted class – these are displayed as a horizontal bar chart.
  2. Aggregate SHAP magnitude across the numerical modality, which we compare
     with the gradient norm from the image and text branches to compute a soft
     *image-modality weight* ∈ [0, 1].  The Grad-CAM++ map is then scaled by
     this weight: if the model was mostly driven by labs, the heatmap is dimmed
     accordingly, preventing misleading high-activation images.

Thread-safety note
------------------
Both the SHAP DeepExplainer and GradCAMExtractor install temporary PyTorch hooks.
They are created fresh per request and all hooks are removed in a `finally` block.
Do NOT share a single extractor instance across concurrent requests.
"""

import io
import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# ── locate project root (two levels up from this file) ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from models import PneumoFusionNet
from config import (
    DEVICE, IMAGE_SIZE, MAX_SEQ_LEN, NUM_NUMERICAL_FEATURES,
    NUMERICAL_COLS, CNN_OUT_DIM, TEXT_OUT_DIM, NUM_OUT_DIM,
    FUSION_DIM, XATTN_LAYERS, XATTN_HEADS, XATTN_FF_DIM, XATTN_DROPOUT,
    CLS_HIDDEN_DIM,
)

import os

CKPT_PATH      = os.getenv("CKPT_PATH",
                            str(PROJECT_ROOT / "outputs/checkpoints/fold2_best.pt"))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH",
                            str(PROJECT_ROOT / "outputs/checkpoints/tokenizer.json"))
SCALER_PATH    = os.getenv("SCALER_PATH",
                            str(PROJECT_ROOT / "outputs/checkpoints/scaler.pkl"))
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH",
                            str(PROJECT_ROOT / "outputs/checkpoints/label_map.json"))


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def encode_text(text: str, word2idx: dict, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    tokens = _tokenise(str(text))[:max_len]
    ids    = [word2idx.get(t, 1) for t in tokens]          # 1 = UNK
    ids    = ids + [0] * max(0, max_len - len(ids))         # pad
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0) # (1, T)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

from torchvision import transforms as T

_val_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL Image → (1, 1, H, W) float tensor on DEVICE."""
    return _val_transform(image).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def encode_numerics(row: dict, scaler) -> torch.Tensor:
    """
    Returns (1, NUM_NUMERICAL_FEATURES) float tensor.
    """
    sex_female = 1.0 if str(row.get("Patient_Sex", "Female")).lower() == "female" else 0.0
    sex_male   = 1.0 - sex_female
    raw = np.array(
        [float(row[c]) for c in NUMERICAL_COLS] + [sex_female, sex_male],
        dtype=np.float32,
    ).reshape(1, -1)
    normed = scaler.transform(raw).astype(np.float32)
    return torch.tensor(normed, dtype=torch.float32).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoader:
    """Load and cache the model + artefacts exactly once per process."""
    _instance: Optional["ModelLoader"] = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def load(
        self,
        ckpt_path: str      = CKPT_PATH,
        tokenizer_path: str = TOKENIZER_PATH,
        scaler_path: str    = SCALER_PATH,
        label_map_path: str = LABEL_MAP_PATH,
    ):
        if self._loaded:
            return

        tok_data        = json.load(open(tokenizer_path, encoding="utf-8"))
        self.word2idx   = tok_data["word2idx"]
        self.vocab_size = tok_data["vocab_size"]

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.label_map  = json.load(open(label_map_path))
        self.idx2label  = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.class_names = [self.idx2label[i] for i in range(self.num_classes)]

        self.model = PneumoFusionNet(
            num_classes=self.num_classes,
            vocab_size=self.vocab_size,
            pretrained_cnn=False,
        )
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE)
        self.model.eval()

        self._loaded = True
        print(f"[ModelLoader] Loaded on {DEVICE}  classes={self.class_names}")

    @property
    def loaded(self):
        return self._loaded


_loader = ModelLoader()


def get_loader() -> ModelLoader:
    if not _loader.loaded:
        _loader.load()
    return _loader


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM++  EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ on the GCSA output layer (model.cnn_encoder.gcsa).

    Improvement over vanilla Grad-CAM
    ----------------------------------
    Vanilla Grad-CAM: weight_c = mean_{i,j}( ∂score / ∂A^c_{ij} )
    Grad-CAM++:       weight_c = Σ_{i,j}  α^c_{ij} · relu( ∂score / ∂A^c_{ij} )

    where α^c_{ij} is a pixel-wise importance coefficient derived from the
    second- and third-order partial derivatives of the score with respect to the
    activation map.  In practice (see Chattopadhay 2018), α simplifies to:

        α^c_{ij} = (∂²score / ∂(A^c_{ij})²) /
                   (2·∂²score/∂(A^c_{ij})² + A^c · ∂³score/∂(A^c_{ij})³ + ε)

    The second- and third-order terms are computed from the first-order gradient
    g = ∂score/∂A^c:

        ∂²score/∂A² ≈ g²          (element-wise square)
        ∂³score/∂A³ ≈ g³          (element-wise cube)

    This avoids a second backward pass and keeps the implementation efficient.

    cuDNN / LSTM fix (same as before)
    ----------------------------------
    PyTorch requires the BiLSTM to be in train() mode during backward().
    We switch only that submodule, then restore eval() afterwards.
    """

    def __init__(self, model: PneumoFusionNet):
        self.model      = model
        self._features  = None
        self._gradients = None

        target_layer = model.cnn_encoder.gcsa

        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        # Store live tensor — do NOT detach before backward()
        self._features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().clone()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @staticmethod
    def _set_bilstm_train(model):
        model.text_encoder.bilstm.train()

    @staticmethod
    def _set_bilstm_eval(model):
        model.text_encoder.bilstm.eval()

    def generate(
        self,
        image_tensor: torch.Tensor,
        text_tensor:  torch.Tensor,
        num_tensor:   torch.Tensor,
        target_class: int,
        original_pil: Image.Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        cam_norm   : (H, W) float32 [0,1]  – raw Grad-CAM++ heatmap
        heatmap_rgb: (H, W, 3) uint8       – INFERNO colourmap (avoids JET's
                                             misleading blue-dominance when activation
                                             is moderate; warm palette reads more
                                             naturally for pathology highlighting)
        overlay    : (H, W, 3) uint8       – blended onto original CT
        """
        self.model.eval()
        self._set_bilstm_train(self.model)

        img = image_tensor.detach().to(DEVICE).requires_grad_(True)
        txt = text_tensor.detach().to(DEVICE)
        num = num_tensor.detach().to(DEVICE)

        self.model.zero_grad()
        logits = self.model(img, txt, num)
        score  = logits[0, target_class]

        score.backward()

        self._set_bilstm_eval(self.model)

        if self._features is None or self._gradients is None:
            raise RuntimeError("Grad-CAM++ hooks did not fire — check layer attachment.")

        features = self._features.detach()[0]   # (C, H', W')
        grads    = self._gradients[0]           # (C, H', W')

        # ── Grad-CAM++ alpha computation ─────────────────────────────────────
        # g  = first-order gradient (already captured in grads)
        # g² and g³ approximate 2nd/3rd partial derivatives of the score
        g2  = grads ** 2                                # (C, H', W')
        g3  = grads ** 3                                # (C, H', W')

        # denominator: 2*g² + A·g³  summed over spatial dims per channel
        denom = 2.0 * g2 + (features * g3).sum(dim=[1, 2], keepdim=True)  # (C,1,1)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))    # avoid /0

        # pixel-wise alpha
        alpha = g2 / denom                             # (C, H', W')

        # weight per channel = Σ_{i,j} alpha * relu(g)
        weights = (alpha * F.relu(grads)).sum(dim=[1, 2])   # (C,)

        cam = (weights.view(-1, 1, 1) * features).sum(0)   # (H', W')
        cam = F.relu(cam)

        # ── Normalise ─────────────────────────────────────────────────────────
        cam_np = cam.cpu().float().numpy()
        mn, mx = cam_np.min(), cam_np.max()
        cam_norm = (cam_np - mn) / (mx - mn + 1e-8)

        # ── Resize + smooth ───────────────────────────────────────────────────
        orig_w, orig_h = original_pil.size
        cam_resized = cv2.resize(
            cam_norm.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        cam_resized = cv2.GaussianBlur(cam_resized, (9, 9), sigmaX=3)
        mn2, mx2 = cam_resized.min(), cam_resized.max()
        cam_resized = (cam_resized - mn2) / (mx2 - mn2 + 1e-8)

        # ── Colourmap ─────────────────────────────────────────────────────────
        # COLORMAP_INFERNO: dark-purple→orange→yellow. More perceptually uniform
        # than JET and doesn't introduce blue artefacts in low-activation regions.
        heatmap_bgr = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        orig_rgb = np.array(
            original_pil.convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
        )
        overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap_rgb, 0.45, 0)

        return cam_resized.astype(np.float32), heatmap_rgb, overlay

    def gradient_norm(self) -> float:
        """
        L2 norm of the raw gradients at the target layer — used as a proxy
        for how much the image branch contributed to the prediction.
        A near-zero norm means the decision was driven by other modalities.
        """
        if self._gradients is None:
            return 0.0
        return float(self._gradients.norm(p=2).cpu())


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER  (numerical features + modality weight)
# ─────────────────────────────────────────────────────────────────────────────

class _NumericalWrapper(nn.Module):
    """
    Thin wrapper that freezes image + text inputs and only exposes the
    numerical tensor as a variable input.  Required by shap.DeepExplainer
    which expects a single-input (or tuple-input) nn.Module.
    """
    def __init__(self, model: PneumoFusionNet,
                 img_t: torch.Tensor, txt_t: torch.Tensor):
        super().__init__()
        self.model = model
        # register as buffers so they move with .to(device)
        self.register_buffer("img_t", img_t)
        self.register_buffer("txt_t", txt_t)

    def forward(self, num_t: torch.Tensor) -> torch.Tensor:
        batch_size = num_t.size(0)

    # Expand frozen modalities to SHAP batch size
        img = self.img_t.expand(batch_size, -1, -1, -1)
        txt = self.txt_t.expand(batch_size, -1)

        return self.model(img, txt, num_t)


class SHAPExplainer:
    """
    Uses shap.DeepExplainer to attribute the predicted-class logit to each
    normalised numerical feature.

    Background
    ----------
    DeepExplainer integrates gradients with respect to a set of background
    (reference) samples.  We use zero-vector baselines (equivalent to
    mean-normalised values from the StandardScaler perspective).

    Output
    ------
    shap_values : np.ndarray  shape (N_features,)
                  SHAP value for each numerical feature for the predicted class.
    feature_names : list[str]  matching labels (includes sex columns)
    image_weight  : float ∈ [0,1]
                  Fraction of total gradient energy attributable to the image
                  branch (estimated from the cam gradient norm vs SHAP L1 norm).
                  Used to scale the Grad-CAM++ heatmap.
    """

    # human-readable labels for the scaler output columns
    FEATURE_LABELS = [
        "Patient Age",
        "WBC (×10⁹/L)", "NEUT%", "LYMP%", "NLR",
        "CRP (mg/L)", "PCT (ng/mL)"
    ]

    def __init__(self, model: PneumoFusionNet):
        self.model = model

    def explain(
    self,
    img_t:         torch.Tensor,
    txt_t:         torch.Tensor,
    num_t:         torch.Tensor,
    target_class:  int,
    image_score: float = 1.0,
    n_background:  int   = 20,   # kept only for compatibility
    ) -> Tuple[np.ndarray, List[str], float]:
        """
        Stable explainability implementation using Gradient × Input.

        Why not SHAP DeepExplainer?
        ---------------------------
        DeepExplainer is unstable with:
        • ResNet residual connections
        • backward hooks (Grad-CAM++)
        • BiLSTM/cuDNN
        • attention transformers
        • inplace ops inside torchvision

        Gradient × Input is significantly more stable and still provides
        meaningful per-feature attribution scores for numerical features.

        Returns
        -------
        vals          : (N_features,) attribution values
        feat_labels   : feature names
        image_weight  : relative contribution of image modality
        """

        self.model.eval()

        wrapper = _NumericalWrapper(
            self.model,
            img_t.detach().to(DEVICE),
            txt_t.detach().to(DEVICE),
        ).to(DEVICE)

        wrapper.eval()

        # Numerical tensor requiring gradients
        foreground_req = (
            num_t.detach()
            .clone()
            .to(DEVICE)
            .requires_grad_(True)
        )

        # ---------------------------------------------------------
        # cuDNN RNN backward fix
        # ---------------------------------------------------------
        # PyTorch requires RNN/LSTM modules to be in train mode
        # during backward() when using cuDNN.
        self.model.text_encoder.bilstm.train()

        wrapper.zero_grad()
        self.model.zero_grad()

        # Forward pass
        out = wrapper(foreground_req)

        # Target class score
        score = out[0, target_class]

        # Backward pass
        score.backward()

        # ---------------------------------------------------------
        # Gradient × Input attribution
        # ---------------------------------------------------------
        vals = (
            foreground_req.grad[0] * foreground_req[0]
        ).detach().cpu().numpy()

        # Restore eval mode
        self.model.text_encoder.bilstm.eval()

        # ---------------------------------------------------------
        # Compute image modality contribution
        # ---------------------------------------------------------
        lab_score = float(np.mean(np.abs(vals))) + 1e-8

        image_weight = image_score / (
        image_score + lab_score)
        # Prevent extreme values
        image_weight = float(
            np.clip(image_weight, 0.15, 0.95)
        )

        # Feature labels
        n_features = num_t.shape[1]
        labels = self.FEATURE_LABELS[:n_features]

        return (
            vals.astype(np.float32),
            labels,
            image_weight,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUSED EXPLAINER  (Grad-CAM++ + SHAP → combined output)
# ─────────────────────────────────────────────────────────────────────────────

class FusedExplainer:
    """
    Orchestrates the complete explainability pipeline:

    Step 1  Grad-CAM++ on the GCSA layer.
    Step 2  SHAP on the numerical branch (DeepExplainer, zero background).
    Step 3  Re-scale the Grad-CAM++ heatmap by image_weight so the visual
            intensity honestly reflects how much the image actually influenced
            the prediction (vs. labs / clinical text).
    Step 4  Return everything the API and UI need.
    """

    def explain(
        self,
        image:         Image.Image,
        clinical_text: str,
        numerical_row: dict,
        target_class:  Optional[int] = None,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            predicted_class, confidence, probabilities, class_index
            cam_norm          (H,W) float32 [0,1]   – re-weighted Grad-CAM++ map
            heatmap_rgb       (H,W,3) uint8          – coloured heatmap
            overlay           (H,W,3) uint8          – blended overlay
            shap_values       list[float]            – per feature, for predicted class
            feature_labels    list[str]
            image_weight      float                  – fraction of prediction from image
            shap_text         str                    – human-readable narrative
        """
        loader = get_loader()

        img_t = preprocess_image(image)
        txt_t = encode_text(clinical_text, loader.word2idx).to(DEVICE)
        num_t = encode_numerics(numerical_row, loader.scaler)

        # ── Pass 1: clean prediction ─────────────────────────────────────────
        with torch.no_grad():
            loader.model.eval()
            logits = loader.model(img_t, txt_t, num_t)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
            idx    = int(np.argmax(probs))

        result = {
            "predicted_class": loader.idx2label[idx],
            "confidence":      float(probs[idx]),
            "probabilities":   {loader.idx2label[i]: float(p) for i, p in enumerate(probs)},
            "class_index":     idx,
        }

        target = target_class if target_class is not None else idx

        # ── Pass 2: Grad-CAM++ ───────────────────────────────────────────────
        cam_extractor = GradCAMPlusPlus(loader.model)
        try:
            cam_norm, heatmap_rgb, overlay = cam_extractor.generate(
                img_t, txt_t, num_t, target, image
            )
            cam_grad_norm = cam_extractor.gradient_norm()
            image_score = float(np.mean(cam_norm))
        finally:
            cam_extractor.remove_hooks()
            loader.model.eval()

        # ── Pass 3: SHAP ─────────────────────────────────────────────────────
        shap_explainer = SHAPExplainer(loader.model)
        shap_vals, feat_labels, image_weight = shap_explainer.explain(img_t,txt_t,num_t,target,image_score=image_score,)

        # ── Pass 4: Re-weight heatmap by image contribution ──────────────────
        # Scale the cam_norm map so its peak equals image_weight.
        # This means: if labs dominated (image_weight=0.2), the heatmap stays
        # faint (max 0.2) preventing false-confidence visualisations.
        cam_scaled = cam_norm * image_weight
        # Re-colourise the scaled map
        heatmap_bgr_scaled = cv2.applyColorMap(
            (cam_scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        heatmap_rgb_scaled = cv2.cvtColor(heatmap_bgr_scaled, cv2.COLOR_BGR2RGB)
        orig_rgb = np.array(image.convert("RGB"))
        orig_w, orig_h = image.size
        orig_rgb_resized = cv2.resize(orig_rgb, (orig_w, orig_h))
        overlay_scaled = cv2.addWeighted(orig_rgb_resized, 0.55, heatmap_rgb_scaled, 0.45, 0)

        # ── Build SHAP narrative ─────────────────────────────────────────────
        shap_text = _build_shap_narrative(
            shap_vals, feat_labels, result["predicted_class"], image_weight
        )

        return {
            **result,
            # images
            "cam_norm":          cam_norm.tolist(),
            "heatmap_rgb":       heatmap_rgb,        # unscaled — for raw CAM tab
            "overlay":           overlay,             # unscaled overlay
            "heatmap_rgb_fused": heatmap_rgb_scaled,  # re-weighted — for fused tab
            "overlay_fused":     overlay_scaled,
            # SHAP
            "shap_values":    shap_vals.tolist(),
            "feature_labels": feat_labels,
            "image_weight":   image_weight,
            "shap_text":      shap_text,
        }

def _build_shap_narrative(
    shap_vals: np.ndarray,
    feat_labels: List[str],
    pred_class: str,
    image_weight: float,
) -> str:

    shap_vals = np.array(shap_vals).flatten()

    n = min(len(shap_vals), len(feat_labels))

    shap_vals = shap_vals[:7]
    feat_labels = feat_labels[:7]

    order = np.argsort(shap_vals)[::-1]

    pos_feats = [
        (feat_labels[i], float(shap_vals[i]))
        for i in order
        if shap_vals[i] > 0
    ][:3]

    neg_feats = [
        (feat_labels[i], float(shap_vals[i]))
        for i in order[::-1]
        if shap_vals[i] < 0
    ][:2]

    img_pct = round(image_weight * 100)
    lab_pct = round((1.0 - image_weight) * 100)

    parts = [
        f"For the prediction of {pred_class}, "
        f"the model relied approximately "
        f"{img_pct}% on CT imaging and "
        f"{lab_pct}% on laboratory/text features."
    ]

    if pos_feats:
        drivers = ", ".join(
            f"{name} ({v:+.3f})"
            for name, v in pos_feats
        )

        parts.append(
            f"Strongest positive contributors: {drivers}."
        )

    if neg_feats:
        contra = ", ".join(
            f"{name} ({v:+.3f})"
            for name, v in neg_feats
        )

        parts.append(
            f"Features reducing confidence: {contra}."
        )

    if image_weight < 0.35:
        parts.append(
            "The prediction relied more heavily on "
            "laboratory/text information than CT imaging."
        )

    elif image_weight > 0.70:
        parts.append(
            "CT imaging was the dominant contributor "
            "to this prediction."
        )

    return " ".join(parts)
# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (unchanged signatures + new run_fused_explain)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    image: Image.Image,
    clinical_text: str,
    numerical_row: dict,
) -> dict:
    loader = get_loader()
    img_t  = preprocess_image(image)
    txt_t  = encode_text(clinical_text, loader.word2idx).to(DEVICE)
    num_t  = encode_numerics(numerical_row, loader.scaler)
    logits = loader.model(img_t, txt_t, num_t)
    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx    = int(np.argmax(probs))
    return {
        "predicted_class": loader.idx2label[idx],
        "confidence":      float(probs[idx]),
        "probabilities":   {loader.idx2label[i]: float(p) for i, p in enumerate(probs)},
        "class_index":     idx,
    }


def run_gradcam(
    image: Image.Image,
    clinical_text: str,
    numerical_row: dict,
    target_class: Optional[int] = None,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper — now uses Grad-CAM++ internally.
    Signature and return types unchanged so api.py /gradcam and /report
    endpoints require no modification.
    """
    loader = get_loader()
    img_t  = preprocess_image(image)
    txt_t  = encode_text(clinical_text, loader.word2idx).to(DEVICE)
    num_t  = encode_numerics(numerical_row, loader.scaler)

    with torch.no_grad():
        loader.model.eval()
        logits = loader.model(img_t, txt_t, num_t)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx    = int(np.argmax(probs))

    result = {
        "predicted_class": loader.idx2label[idx],
        "confidence":      float(probs[idx]),
        "probabilities":   {loader.idx2label[i]: float(p) for i, p in enumerate(probs)},
        "class_index":     idx,
    }

    target    = target_class if target_class is not None else idx
    extractor = GradCAMPlusPlus(loader.model)
    try:
        cam_norm, heatmap_rgb, overlay = extractor.generate(
            img_t, txt_t, num_t, target, image
        )
    finally:
        extractor.remove_hooks()
        loader.model.eval()

    return result, cam_norm, heatmap_rgb, overlay


def run_fused_explain(
    image: Image.Image,
    clinical_text: str,
    numerical_row: dict,
    target_class: Optional[int] = None,
) -> dict:
    """
    Full Grad-CAM++ + SHAP fused explainability.
    Called by the new /explain API endpoint.
    """
    return FusedExplainer().explain(image, clinical_text, numerical_row, target_class)
