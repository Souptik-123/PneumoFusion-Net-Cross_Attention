"""
webapp/backend/api.py
─────────────────────
FastAPI REST backend for PneumoFusion-Net web application.

Endpoints
---------
GET  /health          – liveness check
POST /predict         – multimodal inference (JSON response)
POST /gradcam         – inference + Grad-CAM++ (returns prediction + image URLs)
POST /report          – full pipeline: predict + Grad-CAM++ + AI report (JSON)
POST /explain         – Grad-CAM++ + SHAP fused explainability (NEW)
GET  /classes         – list of class names
GET  /reference_ranges – lab reference ranges for frontend display

Run
---
    uvicorn webapp.backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import time
import traceback
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2

from webapp.backend.inference_engine import (
    get_loader, run_inference, run_gradcam, run_fused_explain,
    encode_text, encode_numerics, preprocess_image,
)
from webapp.backend.report_generator import generate_report, LAB_REFERENCE


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PneumoFusion-Net API",
    description="Multimodal AI pneumonia diagnosis backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        get_loader()
        print("[API] Model loaded successfully.")
    except Exception as e:
        print(f"[API] WARNING: Could not load model at startup: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pil_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 numpy array to base64-encoded PNG string."""
    success, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _numerical_from_form(
    age: float, sex: str,
    wbc: float, neut: float, lymp: float,
    nlr: float, crp: float, pct: float,
) -> dict:
    return {
        "Patient_Age":    age,
        "Patient_Sex":    sex,
        "WBC (x10^9/L)":  wbc,
        "NEUT%":          neut,
        "LYMP%":          lymp,
        "NLR":            nlr,
        "CRP (mg/L)":     crp,
        "PCT (ng/mL)":    pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES – system
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    loader = get_loader()
    return {
        "status":   "ok",
        "model":    "PneumoFusion-Net",
        "loaded":   loader.loaded,
        "classes":  loader.class_names if loader.loaded else [],
    }


@app.get("/classes", tags=["System"])
def get_classes():
    loader = get_loader()
    return {"classes": loader.class_names}


@app.get("/reference_ranges", tags=["System"])
def reference_ranges():
    return {"reference_ranges": LAB_REFERENCE}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES – inference
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["Inference"])
async def predict(
    ct_image:         UploadFile = File(..., description="CT scan image"),
    clinical_text:    str        = Form("", description="Clinical observation text"),
    age:              float      = Form(...),
    sex:              str        = Form(...),
    wbc:              float      = Form(...),
    neut:             float      = Form(...),
    lymp:             float      = Form(...),
    nlr:              float      = Form(...),
    crp:              float      = Form(...),
    pct:              float      = Form(...),
):
    """Run multimodal inference. Returns predicted class + probabilities."""
    t0 = time.perf_counter()
    try:
        image   = _pil_from_upload(ct_image)
        num_row = _numerical_from_form(age, sex, wbc, neut, lymp, nlr, crp, pct)
        result  = run_inference(image, clinical_text, num_row)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/gradcam", tags=["Inference"])
async def gradcam(
    ct_image:         UploadFile = File(...),
    clinical_text:    str        = Form(""),
    age:              float      = Form(...),
    sex:              str        = Form(...),
    wbc:              float      = Form(...),
    neut:             float      = Form(...),
    lymp:             float      = Form(...),
    nlr:              float      = Form(...),
    crp:              float      = Form(...),
    pct:              float      = Form(...),
    target_class:     Optional[int] = Form(None),
):
    """
    Run inference + Grad-CAM++.
    Returns prediction dict + base64-encoded PNG images:
        original_b64, heatmap_b64, overlay_b64
    """
    t0 = time.perf_counter()
    try:
        pil_img = _pil_from_upload(ct_image)
        num_row = _numerical_from_form(age, sex, wbc, neut, lymp, nlr, crp, pct)

        result, cam_norm, heatmap_rgb, overlay = run_gradcam(
            pil_img, clinical_text, num_row, target_class
        )

        orig_rgb = np.array(pil_img.convert("RGB"))

        response = {
            **result,
            "inference_ms":  round((time.perf_counter() - t0) * 1000, 1),
            "original_b64":  _ndarray_to_b64(orig_rgb),
            "heatmap_b64":   _ndarray_to_b64(heatmap_rgb),
            "overlay_b64":   _ndarray_to_b64(overlay),
        }
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE – fused Grad-CAM++ + SHAP explainability  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/explain", tags=["Explainability"])
async def explain(
    ct_image:         UploadFile      = File(...),
    clinical_text:    str             = Form(""),
    age:              float           = Form(...),
    sex:              str             = Form(...),
    wbc:              float           = Form(...),
    neut:             float           = Form(...),
    lymp:             float           = Form(...),
    nlr:              float           = Form(...),
    crp:              float           = Form(...),
    pct:              float           = Form(...),
    target_class:     Optional[int]   = Form(None),
):
    """
    Full explainability pipeline:

    1. Grad-CAM++ on the GCSA layer  →  heatmap / overlay
    2. SHAP DeepExplainer on numerics →  per-feature importance values
    3. Re-weight Grad-CAM++ heatmap   →  fused overlay

    Response fields
    ---------------
    predicted_class, confidence, probabilities, class_index
    original_b64          base64 PNG  – original CT
    heatmap_b64           base64 PNG  – raw Grad-CAM++ heatmap (INFERNO)
    overlay_b64           base64 PNG  – unweighted Grad-CAM++ overlay
    heatmap_fused_b64     base64 PNG  – SHAP-weighted Grad-CAM++ heatmap
    overlay_fused_b64     base64 PNG  – SHAP-weighted overlay (main output)
    shap_values           list[float] – SHAP attribution per numerical feature
    feature_labels        list[str]
    image_weight          float       – fraction of prediction from CT image
    shap_text             str         – human-readable narrative
    inference_ms          float
    """
    t0 = time.perf_counter()
    try:
        pil_img = _pil_from_upload(ct_image)
        num_row = _numerical_from_form(age, sex, wbc, neut, lymp, nlr, crp, pct)

        expl = run_fused_explain(pil_img, clinical_text, num_row, target_class)

        orig_rgb = np.array(pil_img.convert("RGB"))

        return {
            # prediction
            "predicted_class":  expl["predicted_class"],
            "confidence":       expl["confidence"],
            "probabilities":    expl["probabilities"],
            "class_index":      expl["class_index"],
            # images (base64 PNG)
            "original_b64":         _ndarray_to_b64(orig_rgb),
            "heatmap_b64":          _ndarray_to_b64(expl["heatmap_rgb"]),
            "overlay_b64":          _ndarray_to_b64(expl["overlay"]),
            "heatmap_fused_b64":    _ndarray_to_b64(expl["heatmap_rgb_fused"]),
            "overlay_fused_b64":    _ndarray_to_b64(expl["overlay_fused"]),
            # SHAP
            "shap_values":    expl["shap_values"],
            "feature_labels": expl["feature_labels"],
            "image_weight":   expl["image_weight"],
            "shap_text":      expl["shap_text"],
            # timing
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explainability failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE – full report  (unchanged, now uses Grad-CAM++ internally)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/report", tags=["Report"])
async def full_report(
    ct_image:         UploadFile = File(...),
    clinical_text:    str        = Form(""),
    age:              float      = Form(...),
    sex:              str        = Form(...),
    wbc:              float      = Form(...),
    neut:             float      = Form(...),
    lymp:             float      = Form(...),
    nlr:              float      = Form(...),
    crp:              float      = Form(...),
    pct:              float      = Form(...),
    openai_key:       str        = Form("", description="Optional OpenAI API key override"),
):
    """
    Full pipeline: multimodal inference + Grad-CAM++ + AI report.

    Returns everything the frontend needs:
        prediction, probabilities, all 3 images (b64), AI report sections.
    """
    t0 = time.perf_counter()
    try:
        if openai_key:
            import os
            os.environ["OPENAI_API_KEY"] = openai_key

        pil_img = _pil_from_upload(ct_image)
        num_row = _numerical_from_form(age, sex, wbc, neut, lymp, nlr, crp, pct)

        result, cam_norm, heatmap_rgb, overlay = run_gradcam(
            pil_img, clinical_text, num_row
        )

        orig_rgb    = np.array(pil_img.convert("RGB"))
        orig_b64    = _ndarray_to_b64(orig_rgb)
        heatmap_b64 = _ndarray_to_b64(heatmap_rgb)
        overlay_b64 = _ndarray_to_b64(overlay)

        cam_thresh = np.percentile(cam_norm, 90)
        intensity  = "high" if cam_thresh > 0.7 else ("moderate" if cam_thresh > 0.4 else "low")
        gradcam_txt = (
            f"Grad-CAM++ heatmap (INFERNO colormap) shows {intensity}-intensity activation "
            f"in lung regions, highlighting areas most influential for the prediction of "
            f"{result['predicted_class']}."
        )

        report = generate_report(
            predicted_class  = result["predicted_class"],
            confidence       = result["confidence"],
            probabilities    = result["probabilities"],
            numerical_row    = num_row,
            clinical_text    = clinical_text,
            gradcam_findings = gradcam_txt,
        )

        return {
            "predicted_class":  result["predicted_class"],
            "confidence":       result["confidence"],
            "probabilities":    result["probabilities"],
            "class_index":      result["class_index"],
            "original_b64":     orig_b64,
            "heatmap_b64":      heatmap_b64,
            "overlay_b64":      overlay_b64,
            "report":           report,
            "total_ms":         round((time.perf_counter() - t0) * 1000, 1),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
