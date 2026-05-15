"""
webapp/backend/inference_engine.py
───────────────────────────────────
Handles ALL model inference for the web application:
  • ModelLoader     – singleton that loads PneumoFusionNet + artefacts once
  • preprocess_image – PIL → tensor
  • encode_text      – raw string → token ids
  • encode_numerics  – dict → standardised numpy array → tensor
  • run_inference    – full multimodal forward pass → predictions
  • GradCAMExtractor – Grad-CAM on cnn_encoder.layer4 → heatmap overlay

The engine is completely decoupled from FastAPI; it can also be imported
by unit tests or notebooks.
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

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT ARTEFACT PATHS  (override via env vars in docker/prod)
# ─────────────────────────────────────────────────────────────────────────────
import os

CKPT_PATH      = os.getenv("CKPT_PATH",
                            str(PROJECT_ROOT / "outputs/checkpoints/fold2_finetuned.pt"))
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
    row keys expected:
        Patient_Age, Patient_Sex, WBC (x10^9/L), NEUT%, LYMP%,
        NLR, CRP (mg/L), PCT (ng/mL)

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
    """
    Load and cache the model + artefacts exactly once per process.
    Thread-safe (GIL-protected for pure Python; use a lock if needed for async).
    """
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

        # ── tokenizer ────────────────────────────────────────────────────
        tok_data        = json.load(open(tokenizer_path, encoding="utf-8"))
        self.word2idx   = tok_data["word2idx"]
        self.vocab_size = tok_data["vocab_size"]

        # ── scaler ───────────────────────────────────────────────────────
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # ── label map ────────────────────────────────────────────────────
        self.label_map  = json.load(open(label_map_path))
        self.idx2label  = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.class_names = [self.idx2label[i] for i in range(self.num_classes)]

        # ── model ─────────────────────────────────────────────────────────
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


# Global singleton
_loader = ModelLoader()


def get_loader() -> ModelLoader:
    if not _loader.loaded:
        _loader.load()
    return _loader


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMExtractor:
    """
    Gradient-weighted Class Activation Maps on cnn_encoder.layer4 (last conv block).

    Bug fixes vs original
    ----------------------
    1. cuDNN RNN backward error: PyTorch requires LSTM to be in TRAIN mode for
       backward pass (cuDNN limitation). We temporarily set only the BiLSTM
       submodule to train() during the backward step, then restore eval().
       The rest of the model stays in eval() throughout.

    2. Heatmap stuck at image border: was caused by detaching features in the
       forward hook BEFORE gradients were computed. Fixed by storing the live
       (non-detached) output tensor and only detaching AFTER backward().

    3. Wrong layer: hooks now attach to the output of the GCSA module (after
       layer4 + DSC + GCSA) rather than layer4 alone — this gives a richer,
       spatially-aware activation map focused on diagnostically relevant regions.
    """

    def __init__(self, model: PneumoFusionNet):
        self.model      = model
        self._features  = None   # stores live tensor (not detached yet)
        self._gradients = None

        # ── Hook on gcsa output — richer spatial map than raw layer4 ────────
        # layer4 → dsc → gcsa  (gcsa already applies channel + spatial attention)
        target_layer = model.cnn_encoder.gcsa

        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        # Store the LIVE output tensor (still in the computation graph).
        # We detach only AFTER backward() so gradients can flow correctly.
        self._features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().clone()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @staticmethod
    def _set_bilstm_train(model: PneumoFusionNet):
        """Set only the BiLSTM to train mode (required by cuDNN for backward)."""
        model.text_encoder.bilstm.train()

    @staticmethod
    def _set_bilstm_eval(model: PneumoFusionNet):
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
        cam_norm   : (H, W) float32 [0,1]  – raw Grad-CAM heatmap
        heatmap_rgb: (H, W, 3) uint8       – coloured heatmap (COLORMAP_JET)
        overlay    : (H, W, 3) uint8       – heatmap blended onto original image
        """
        # ── Set model to eval; only BiLSTM needs train for cuDNN backward ────
        self.model.eval()
        self._set_bilstm_train(self.model)

        # Fresh tensors — no leftover gradients from prior calls
        img = image_tensor.detach().to(DEVICE).requires_grad_(True)
        txt = text_tensor.detach().to(DEVICE)
        num = num_tensor.detach().to(DEVICE)

        # ── Forward pass (with autograd active — NO torch.no_grad() here) ───
        self.model.zero_grad()
        logits = self.model(img, txt, num)
        score  = logits[0, target_class]

        # ── Backward pass ────────────────────────────────────────────────────
        score.backward()

        # ── Restore full eval mode ────────────────────────────────────────────
        self._set_bilstm_eval(self.model)

        # ── Now safe to detach features ───────────────────────────────────────
        if self._features is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check layer attachment.")

        features = self._features.detach()[0]     # (C, H', W')
        grads    = self._gradients[0]             # (C, H', W')

        # ── Compute Grad-CAM ─────────────────────────────────────────────────
        # global average-pool gradients over spatial dims → importance weight per channel
        weights = grads.mean(dim=[1, 2])                          # (C,)
        cam     = (weights.view(-1, 1, 1) * features).sum(0)     # (H', W')
        cam     = F.relu(cam)                                     # keep only positive

        # ── Normalise ────────────────────────────────────────────────────────
        cam_np          = cam.cpu().float().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # ── Resize to original image dimensions ──────────────────────────────
        orig_w, orig_h = original_pil.size
        cam_resized    = cv2.resize(
            cam_norm.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        # smooth with slight Gaussian blur for cleaner heatmap edges
        cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), sigmaX=4)
        # re-normalise after blur
        mn, mx = cam_resized.min(), cam_resized.max()
        if mx - mn > 1e-8:
            cam_resized = (cam_resized - mn) / (mx - mn)

        # ── Colourmap → RGB heatmap ───────────────────────────────────────────
        heatmap_bgr = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # ── Overlay on original image ─────────────────────────────────────────
        # Resize original to match, convert to RGB, blend
        orig_rgb = np.array(
            original_pil.convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
        )
        overlay = cv2.addWeighted(orig_rgb, 0.5, heatmap_rgb, 0.5, 0)

        return cam_resized.astype(np.float32), heatmap_rgb, overlay


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    image: Image.Image,
    clinical_text: str,
    numerical_row: dict,
) -> dict:
    """
    Full multimodal inference pipeline.

    Parameters
    ----------
    image          : PIL Image (CT scan)
    clinical_text  : raw clinical observation string
    numerical_row  : dict with keys matching NUMERICAL_COLS + Patient_Sex

    Returns
    -------
    dict with keys:
        predicted_class  : str
        confidence       : float (0-1)
        probabilities    : {class_name: float}
        class_index      : int
    """
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
    Run inference + Grad-CAM.

    IMPORTANT: this function must NOT be wrapped in @torch.no_grad() because
    Grad-CAM requires a live computation graph for score.backward().
    Prediction is obtained via a separate no_grad forward pass first.

    Returns
    -------
    result     : prediction dict
    cam_norm   : (H, W) raw normalised heatmap [0,1]
    heatmap_rgb: (H, W, 3) uint8 coloured heatmap
    overlay    : (H, W, 3) uint8 blended overlay
    """
    loader = get_loader()

    img_t  = preprocess_image(image)
    txt_t  = encode_text(clinical_text, loader.word2idx).to(DEVICE)
    num_t  = encode_numerics(numerical_row, loader.scaler)

    # ── Pass 1: clean prediction with no_grad ────────────────────────────────
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

    # ── Pass 2: Grad-CAM (requires autograd — no torch.no_grad wrapper) ──────
    extractor = GradCAMExtractor(loader.model)
    try:
        cam_norm, heatmap_rgb, overlay = extractor.generate(
            img_t, txt_t, num_t, target, image
        )
    finally:
        extractor.remove_hooks()
        loader.model.eval()   # ensure clean eval state after backward

    return result, cam_norm, heatmap_rgb, overlay
