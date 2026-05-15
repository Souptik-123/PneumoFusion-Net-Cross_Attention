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
                            str(PROJECT_ROOT / "outputs/checkpoints/fold0_finetuned.pt"))
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
# GRAD-CAM EXTRACTOR  — FIXED
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMExtractor:
    """
    Gradient-weighted Class Activation Maps on cnn_encoder.layer4.

    Fixes applied vs original:
    1. _save_gradients stored grad_output[0] (shape: B,C,H,W) but generate()
       then indexed [0] again, discarding all but the first channel-slice.
       Fix: store the full grad_output[0] tensor and index batch dim in generate().
    2. image_tensor.requires_grad_(True) was set but never used — gradients
       flow through the registered hooks on layer4, not through the input.
       Removed to avoid confusing autograd.
    3. The extractor was instantiated AFTER a no_grad() forward pass, so
       _features was always None on the very first run. Fixed by moving
       the no_grad() prediction block into run_inference() only, and
       letting run_gradcam() do a single grad-enabled forward here.
    4. Added explicit check that hooks captured data before computing CAM.
    """

    def __init__(self, model: PneumoFusionNet):
        self.model      = model
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # hook on the last ResNet stage
        target_layer = model.cnn_encoder.layer4

        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        # output shape: (B, C, H, W) — detach so it won't hold the graph
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # FIX 1: store grad_output[0] which is (B, C, H, W).
        # Do NOT index the batch dim here — do it in generate().
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, C, H, W)
        text_tensor:  torch.Tensor,   # (1, T)
        num_tensor:   torch.Tensor,   # (1, F)
        target_class: int,
        original_pil: Image.Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        cam_norm    : (H, W) float32 in [0, 1]  — resized to original image size
        heatmap_rgb : (H, W, 3) uint8  — JET colormap
        overlay     : (H, W, 3) uint8  — blended with original image
        """
        self.model.eval()

        # FIX 2: no requires_grad_ on the image tensor — hooks capture
        # gradients at layer4, independent of whether the input leaf
        # has requires_grad set.
        image_tensor = image_tensor.to(DEVICE)
        text_tensor  = text_tensor.to(DEVICE)
        num_tensor   = num_tensor.to(DEVICE)

        self.model.zero_grad(set_to_none=True)

        # Enable grad for this scope
        with torch.set_grad_enabled(True):
            # Disable CuDNN for RNN backward compatibility
            with torch.backends.cudnn.flags(enabled=False):
                logits = self.model(image_tensor, text_tensor, num_tensor)
                score  = logits[0, target_class]
                score.backward()

        # ── sanity-check that hooks fired ─────────────────────────────────
        if self._features is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture data. "
                "Ensure the model performs a forward pass through cnn_encoder.layer4."
            )

        # FIX 1 continued: index the batch dimension HERE, not inside the hook
        # Both tensors are (B, C, H, W); we want sample 0 → (C, H, W)
        features = self._features[0]    # (C, H, W)
        grads    = self._gradients[0]   # (C, H, W)

        # Global-average-pool gradients over spatial dims → channel weights
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted sum of feature maps
        cam = torch.sum(weights[:, None, None] * features, dim=0)  # (H, W)

        # ReLU — keep only positive activations
        cam = F.relu(cam)

        # ── convert to numpy and normalise ────────────────────────────────
        cam_np  = cam.cpu().numpy().astype(np.float32)
        cam_min = cam_np.min()
        cam_max = cam_np.max()

        if (cam_max - cam_min) > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            # Flat / dead CAM — return a neutral mid-value map
            cam_np = np.full_like(cam_np, 0.5)

        # ── resize to original image dimensions ───────────────────────────
        # PIL .size → (width, height); cv2.resize dsize → (width, height) ✓
        orig_w, orig_h = original_pil.size
        cam_resized = cv2.resize(
            cam_np,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # ── build heatmap and overlay ─────────────────────────────────────
        heatmap_bgr = cv2.applyColorMap(
            np.uint8(cam_resized * 255),
            cv2.COLORMAP_JET,
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        orig_rgb = np.array(original_pil.convert("RGB"))
        overlay  = cv2.addWeighted(orig_rgb, 0.6, heatmap_rgb, 0.4, 0)

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
    Run inference then Grad-CAM.

    FIX 3: The prediction-only forward pass now uses run_inference() which
    is decorated with @torch.no_grad().  The GradCAMExtractor is created
    AFTER that, so hooks are only active during the grad-enabled backward
    pass and _features/_gradients are guaranteed to be populated.
    """
    loader = get_loader()

    img_t = preprocess_image(image)
    txt_t = encode_text(clinical_text, loader.word2idx).to(DEVICE)
    num_t = encode_numerics(numerical_row, loader.scaler)

    # ── Step 1: prediction only (no_grad, no hooks yet) ──────────────────
    result = run_inference(image, clinical_text, numerical_row)
    target = target_class if target_class is not None else result["class_index"]

    # ── Step 2: Grad-CAM (hooks registered here, after prediction) ───────
    extractor = GradCAMExtractor(loader.model)
    try:
        cam_norm, heatmap_rgb, overlay = extractor.generate(
            img_t, txt_t, num_t, target, image,
        )
    finally:
        extractor.remove_hooks()

    return result, cam_norm, heatmap_rgb, overlay
