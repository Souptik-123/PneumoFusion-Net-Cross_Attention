"""
webapp/frontend/app.py
──────────────────────
Streamlit frontend for PneumoFusion-Net pneumonia diagnosis system.

Run
---
    streamlit run webapp/frontend/app.py

Architecture
------------
• Calls the FastAPI backend at BACKEND_URL (default: http://localhost:8000)
• All heavy computation stays in the backend; this file is purely UI.
• Pages: Home, Diagnose, About
"""

import io
import os
import base64
import time
import requests
import numpy as np
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

PAGE_ICON   = "🫁"
APP_TITLE   = "PneumoFusion-Net"
SUBTITLE    = "Multimodal AI-Assisted Pneumonia Diagnosis"

CLASS_COLORS = {
    "Bacterial Pneumonia": "#E84545",
    "Viral Pneumonia":     "#F4A237",
    "Tuberculosis":        "#9B2335",
    "Normal":              "#27AE60",
    "Covid-19":            "#2980B9",
}

LAB_FIELDS = [
    ("WBC (x10^9/L)",  "WBC",   0.0,  30.0,  7.5,   0.1),
    ("NEUT%",          "NEUT%", 0.0,  100.0, 60.0,  0.5),
    ("LYMP%",          "LYMP%", 0.0,  100.0, 25.0,  0.5),
    ("NLR",            "NLR",   0.0,  50.0,  2.5,   0.1),
    ("CRP (mg/L)",     "CRP",   0.0,  500.0, 5.0,   0.5),
    ("PCT (ng/mL)",    "PCT",   0.0,  100.0, 0.1,   0.01),
]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0F1117; color: #E8EAF0; }
[data-testid="stSidebar"] { background: #161B27; border-right: 1px solid #2D3748; }
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1A2035 0%, #1E2942 100%);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.result-card {
    background: linear-gradient(135deg, #1A2035 0%, #1E2942 100%);
    border-left: 4px solid #4A90D9;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 12px 0;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7EB8F7;
    border-bottom: 1px solid #2D3748;
    padding-bottom: 6px;
    margin: 18px 0 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Confidence bar */
.conf-bar-bg {
    background: #2D3748;
    border-radius: 8px;
    height: 14px;
    margin: 4px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Prediction badge */
.pred-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}

/* Report sections */
.report-section {
    background: #1A2035;
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}
.report-section h4 { color: #7EB8F7; margin-top: 0; }

/* Warning box */
.warning-box {
    background: rgba(220, 53, 69, 0.12);
    border: 1px solid #DC3545;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.success-box {
    background: rgba(39, 174, 96, 0.12);
    border: 1px solid #27AE60;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}

/* Disclaimer */
.disclaimer-box {
    background: rgba(74, 144, 217, 0.1);
    border: 1px solid #4A90D9;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.85rem;
    color: #A0AEC0;
}

/* Image caption */
.img-caption {
    text-align: center;
    font-size: 0.78rem;
    color: #718096;
    margin-top: 4px;
}

/* Spinner override */
.stSpinner > div { border-color: #4A90D9 transparent transparent transparent; }

/* Input labels */
label { color: #CBD5E0 !important; font-size: 0.88rem !important; }

/* Hide default streamlit footer */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def check_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def confidence_bar(label: str, prob: float, color: str = "#4A90D9"):
    pct = prob * 100
    st.markdown(f"""
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-size:0.85rem;color:#CBD5E0">{label}</span>
        <span style="font-size:0.85rem;font-weight:600;color:{color}">{pct:.1f}%</span>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def prediction_badge(label: str, confidence: float):
    color = CLASS_COLORS.get(label, "#4A90D9")
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 8px">
      <span class="pred-badge" style="background:{color}22;color:{color};border:2px solid {color}">
        {label}
      </span>
      <div style="font-size:2rem;font-weight:800;color:{color};margin-top:4px">
        {confidence*100:.1f}%
      </div>
      <div style="font-size:0.8rem;color:#718096;margin-top:2px">confidence</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {PAGE_ICON} {APP_TITLE}")
        st.markdown(f"*{SUBTITLE}*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔬 Diagnose", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()

        # Backend status
        backend_ok = check_backend()
        status_icon  = "🟢" if backend_ok else "🔴"
        status_label = "Connected" if backend_ok else "Offline"
        st.markdown(f"**API Status:** {status_icon} {status_label}")
        st.markdown(f"`{BACKEND_URL}`")

        st.divider()

        # OpenAI key input
        st.markdown("**🔑 OpenAI API Key**")
        st.markdown("<small>Required for AI report generation</small>", unsafe_allow_html=True)
        openai_api_key = st.text_input(
            "OpenAI Key", type="password",
            placeholder="sk-...", label_visibility="collapsed",
            key="openai_api_key",
        )

        st.divider()
        st.markdown("<small>⚠️ For research use only. Not a medical device.</small>",
                    unsafe_allow_html=True)

    return page.split(" ", 1)[1]


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_home():
    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 20px">
      <div style="font-size:3.5rem">🫁</div>
      <h1 style="font-size:2.4rem;font-weight:800;color:#7EB8F7;margin:8px 0">
        {APP_TITLE}
      </h1>
      <p style="font-size:1.1rem;color:#A0AEC0;max-width:640px;margin:0 auto">
        {SUBTITLE} — combining CT imaging, laboratory values, and clinical text
        through a Cross-Attention Transformer architecture.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
      ⚠️ <strong>Disclaimer:</strong> PneumoFusion-Net is an AI-assisted decision-support tool
      for research purposes. It is <strong>NOT</strong> a medical device and should not be used
      as a sole basis for clinical decisions. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)
    features = [
        ("🖼️", "CT Scan Analysis", "ResNet50 + GCSA + Depthwise-Separable Conv for fine-grained pulmonary lesion detection"),
        ("🔬", "Lab Integration",  "MLP + Residual connections processing WBC, CRP, PCT, NLR and more"),
        ("📝", "Clinical Text",    "BiLSTM + Additive Attention encoding patient observations and symptoms"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.8rem">{icon}</div>
              <div style="font-size:1rem;font-weight:600;color:#7EB8F7;margin:6px 0">{title}</div>
              <div style="font-size:0.82rem;color:#A0AEC0">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    cols2 = st.columns(3)
    features2 = [
        ("🧠", "Cross-Attention Fusion", "Each modality queries the others as key-value context for dynamic importance weighting"),
        ("🌡️", "Grad-CAM Heatmaps",    "Gradient-weighted Class Activation Maps highlight diagnostically relevant CT regions"),
        ("📋", "AI Reports",            "GPT-4.1-powered clinical summary with recommendations and emergency warning signs"),
    ]
    for col, (icon, title, desc) in zip(cols2, features2):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.8rem">{icon}</div>
              <div style="font-size:1rem;font-weight:600;color:#7EB8F7;margin:6px 0">{title}</div>
              <div style="font-size:0.82rem;color:#A0AEC0">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Classes
    st.divider()
    section_header("Detectable Conditions")
    cls_cols = st.columns(len(CLASS_COLORS))
    for col, (cls, color) in zip(cls_cols, CLASS_COLORS.items()):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:12px 8px;background:{color}15;
                        border:1px solid {color};border-radius:10px">
              <div style="font-size:0.82rem;font-weight:600;color:{color}">{cls}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_diagnose():
    st.markdown("## 🔬 Multimodal Diagnosis")
    st.markdown("<small style='color:#718096'>Fill in all three modalities for best accuracy</small>",
                unsafe_allow_html=True)

    # ── INPUT PANEL ─────────────────────────────────────────────────────────
    with st.container():
        left, right = st.columns([1, 1], gap="large")

        # ── LEFT: IMAGE + TEXT ─────────────────────────────────────────────
        with left:
            section_header("CT Scan Upload")
            uploaded_file = st.file_uploader(
                "Upload CT scan image (JPG/PNG/DICOM preview)",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                key="ct_upload",
                label_visibility="collapsed",
            )
            if uploaded_file:
                pil_preview = Image.open(uploaded_file).convert("RGB")
                st.image(pil_preview, caption="Uploaded CT scan", use_column_width=True)

            section_header("Clinical Observation")
            clinical_text = st.text_area(
                "Clinical text",
                placeholder="e.g. Patient presents with fever (38.5°C), productive cough, "
                            "bilateral lower lobe crackles on auscultation. No recent travel history.",
                height=120,
                label_visibility="collapsed",
                key="clinical_text",
            )

        # ── RIGHT: LAB PARAMETERS ──────────────────────────────────────────
        with right:
            section_header("Patient Demographics")
            d1, d2 = st.columns(2)
            with d1:
                age = st.number_input("Age (years)", min_value=0, max_value=120,
                                      value=50, step=1)
            with d2:
                sex = st.selectbox("Sex", ["Male", "Female"])

            section_header("Laboratory Parameters")
            lab_values = {}
            row1 = st.columns(2)
            row2 = st.columns(2)
            row3 = st.columns(2)
            rows = [row1, row2, row3]

            for i, (col_name, label, mn, mx, default, step) in enumerate(LAB_FIELDS):
                col_obj = rows[i // 2][i % 2]
                with col_obj:
                    lab_values[col_name] = st.number_input(
                        label, min_value=mn, max_value=mx,
                        value=default, step=step, format="%.2f",
                    )

    # ── REFERENCE RANGES TABLE ────────────────────────────────────────────
    with st.expander("📊 Reference Ranges", expanded=False):
        ref_data = {
            "Parameter": ["WBC (×10⁹/L)", "NEUT%", "LYMP%", "NLR", "CRP (mg/L)", "PCT (ng/mL)"],
            "Normal Low":  [4.0, 50.0, 20.0, 1.0, 0.0, 0.0],
            "Normal High": [11.0, 70.0, 40.0, 3.0, 10.0, 0.5],
            "Your Value":  [
                lab_values.get("WBC (x10^9/L)", 0),
                lab_values.get("NEUT%", 0),
                lab_values.get("LYMP%", 0),
                lab_values.get("NLR", 0),
                lab_values.get("CRP (mg/L)", 0),
                lab_values.get("PCT (ng/mL)", 0),
            ],
        }
        import pandas as pd
        df = pd.DataFrame(ref_data)
        def _flag(row):
            v, lo, hi = row["Your Value"], row["Normal Low"], row["Normal High"]
            if v < lo:   return [""] * 3 + ["background-color: #3730a322"]
            if v > hi:   return [""] * 3 + ["background-color: #dc354522"]
            return       [""] * 4
        st.dataframe(df.style.apply(_flag, axis=1), use_container_width=True, hide_index=True)

    # ── SUBMIT BUTTON ─────────────────────────────────────────────────────
    st.divider()
    run_btn = st.button(
        "🚀  Run Multimodal Analysis",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    if uploaded_file is None:
        st.info("👆 Please upload a CT scan image to enable analysis.")

    # ── INFERENCE + RESULTS ───────────────────────────────────────────────
    if run_btn and uploaded_file is not None:
        numerical_row = {
            "Patient_Age":   age,
            "Patient_Sex":   sex,
            **lab_values,
        }

        with st.spinner("🧠  Running multimodal inference + Grad-CAM..."):
            uploaded_file.seek(0)
            files   = {"ct_image": (uploaded_file.name, uploaded_file, "image/jpeg")}
            data    = {
                "clinical_text": clinical_text or "",
                "age":   age,   "sex":  sex,
                "wbc":   lab_values["WBC (x10^9/L)"],
                "neut":  lab_values["NEUT%"],
                "lymp":  lab_values["LYMP%"],
                "nlr":   lab_values["NLR"],
                "crp":   lab_values["CRP (mg/L)"],
                "pct":   lab_values["PCT (ng/mL)"],
                "openai_key": st.session_state.get("openai_api_key", ""),
            }
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/report",
                    files=files, data=data,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                st.session_state["last_result"] = result
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is the FastAPI server running?")
                st.code(f"uvicorn webapp.backend.api:app --host 0.0.0.0 --port 8000")
                return
            except Exception as e:
                st.error(f"❌ API error: {e}")
                if "resp" in dir():
                    st.code(resp.text[:500])
                return

    # ── RENDER RESULTS ────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        _render_results(result)


def _render_results(result: dict):
    """Render the full results dashboard."""
    st.divider()
    st.markdown("## 📊 Diagnosis Results")
    st.markdown(
        f"<small style='color:#718096'>Analysis completed in "
        f"{result.get('total_ms', 0):.0f} ms</small>",
        unsafe_allow_html=True,
    )

    # ── ROW 1: Prediction + Probabilities ─────────────────────────────────
    pred_col, prob_col = st.columns([1, 2], gap="large")

    with pred_col:
        section_header("Prediction")
        prediction_badge(result["predicted_class"], result["confidence"])

        color = CLASS_COLORS.get(result["predicted_class"], "#4A90D9")
        severity_map = {
            "Normal":              ("✅ No Pneumonia Detected", "success-box"),
            "Bacterial Pneumonia": ("⚠️ Bacterial Infection", "warning-box"),
            "Viral Pneumonia":     ("⚠️ Viral Infection",     "warning-box"),
            "Tuberculosis":        ("🚨 TB Suspected",         "warning-box"),
            "Covid-19":            ("⚠️ COVID-19 Suspected",   "warning-box"),
        }
        label, box_class = severity_map.get(
            result["predicted_class"], ("⚠️ Abnormality Detected", "warning-box")
        )
        st.markdown(f'<div class="{box_class}">{label}</div>', unsafe_allow_html=True)

    with prob_col:
        section_header("Class Probabilities")
        probs_sorted = sorted(result["probabilities"].items(), key=lambda x: -x[1])
        for cls_name, prob in probs_sorted:
            col  = CLASS_COLORS.get(cls_name, "#4A90D9")
            confidence_bar(cls_name, prob, col)

    # ── ROW 2: CT Images ──────────────────────────────────────────────────
    st.divider()
    section_header("CT Scan Visualisations")
    img_cols = st.columns(3)
    img_data = [
        ("original_b64",  "Original CT Scan"),
        ("heatmap_b64",   "Grad-CAM Heatmap"),
        ("overlay_b64",   "Heatmap Overlay"),
    ]
    for col, (key, caption) in zip(img_cols, img_data):
        with col:
            if key in result:
                img = b64_to_pil(result[key])
                st.image(img, use_column_width=True)
                st.markdown(f'<div class="img-caption">{caption}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1A2035;border:1px solid #2D3748;border-radius:8px;
                padding:12px 16px;margin:8px 0;font-size:0.82rem;color:#A0AEC0">
      🌡️ <strong>Grad-CAM Guide:</strong>
      <span style="color:#FF4444">Red/Yellow</span> = high activation (most influential regions) &nbsp;|&nbsp;
      <span style="color:#4444FF">Blue/Green</span> = low activation
    </div>
    """, unsafe_allow_html=True)

    # ── ROW 3: AI Report ──────────────────────────────────────────────────
    if "report" in result:
        report = result["report"]
        st.divider()
        source_badge = "🤖 GPT-4.1" if report.get("_source") == "openai" else "📋 Rule-based"
        section_header(f"AI Medical Report  [{source_badge}]")

        # patient summary
        if "patient_summary" in report:
            st.markdown(f"""
            <div class="report-section">
              <h4>🧑‍⚕️ Patient Summary</h4>
              <p style="color:#E2E8F0;margin:0">{report['patient_summary']}</p>
            </div>
            """, unsafe_allow_html=True)

        # clinical + confidence in 2 cols
        cli_col, conf_col = st.columns(2)
        with cli_col:
            if "clinical_interpretation" in report:
                st.markdown(f"""
                <div class="report-section">
                  <h4>🔬 Clinical Interpretation</h4>
                  <p style="color:#CBD5E0;font-size:0.88rem;margin:0">{report['clinical_interpretation']}</p>
                </div>
                """, unsafe_allow_html=True)
        with conf_col:
            if "confidence_explanation" in report:
                st.markdown(f"""
                <div class="report-section">
                  <h4>📈 Confidence Explanation</h4>
                  <p style="color:#CBD5E0;font-size:0.88rem;margin:0">{report['confidence_explanation']}</p>
                </div>
                """, unsafe_allow_html=True)

        # abnormalities + actions
        abn_col, act_col = st.columns(2)
        with abn_col:
            if "key_abnormalities" in report:
                items = "".join(
                    f"<li style='margin:4px 0;font-size:0.86rem;color:#CBD5E0'>{a}</li>"
                    for a in report["key_abnormalities"]
                )
                st.markdown(f"""
                <div class="report-section">
                  <h4>🔴 Key Abnormalities</h4>
                  <ul style="padding-left:18px;margin:0">{items}</ul>
                </div>
                """, unsafe_allow_html=True)
        with act_col:
            if "recommended_actions" in report:
                items = "".join(
                    f"<li style='margin:4px 0;font-size:0.86rem;color:#CBD5E0'>{a}</li>"
                    for a in report["recommended_actions"]
                )
                st.markdown(f"""
                <div class="report-section">
                  <h4>✅ Recommended Actions</h4>
                  <ul style="padding-left:18px;margin:0">{items}</ul>
                </div>
                """, unsafe_allow_html=True)

        # emergency signs
        if "emergency_warning_signs" in report:
            items = "".join(
                f"<li style='margin:3px 0;font-size:0.85rem'>{s}</li>"
                for s in report["emergency_warning_signs"]
            )
            st.markdown(f"""
            <div class="warning-box">
              <strong>🚨 Emergency Warning Signs — Seek Immediate Care If:</strong>
              <ul style="padding-left:18px;margin:8px 0 0">{items}</ul>
            </div>
            """, unsafe_allow_html=True)

        # disclaimer
        if "disclaimer" in report:
            st.markdown(f"""
            <div class="disclaimer-box">{report['disclaimer']}</div>
            """, unsafe_allow_html=True)

    # ── DOWNLOAD JSON REPORT ──────────────────────────────────────────────
    st.divider()
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        import json
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="pneumofusion_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        # download overlay image
        if "overlay_b64" in result:
            overlay_bytes = base64.b64decode(result["overlay_b64"])
            st.download_button(
                "⬇️ Download Grad-CAM Image",
                data=overlay_bytes,
                file_name="gradcam_overlay.png",
                mime="image/png",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_about():
    st.markdown("## ℹ️ About PneumoFusion-Net")

    st.markdown("""
    <div class="report-section">
      <h4>📖 Model Architecture</h4>
      <p style="color:#CBD5E0;font-size:0.9rem">
        PneumoFusion-Net is a multimodal deep learning framework for pneumonia classification.
        It integrates CT images, clinical text, and numerical laboratory data through a
        <strong>Cross-Attention Transformer</strong> fusion mechanism.
      </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    arch_info = [
        ("🖼️ CNN Encoder", [
            "Modified ResNet50",
            "Single-channel (grayscale) input",
            "Depthwise Separable Convolutions",
            "Global Channel-Spatial Attention (GCSA)",
            "Output: 512-dim feature vector",
        ]),
        ("📝 Text Encoder", [
            "Sinusoidal Positional Encoding",
            "2-layer Bidirectional LSTM",
            "Bahdanau Additive Attention",
            "Vocabulary: 5,000 tokens",
            "Output: 512-dim feature vector",
        ]),
        ("🔢 Numerical Encoder", [
            "StandardScaler normalisation",
            "MLP with Residual Connections",
            "7 lab values + sex encoding",
            "Dropout (p=0.3) regularisation",
            "Output: 64-dim feature vector",
        ]),
    ]
    for col, (title, items) in zip(cols, arch_info):
        with col:
            items_html = "".join(f"<li style='font-size:0.83rem;color:#CBD5E0;margin:3px 0'>{i}</li>" for i in items)
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-weight:600;color:#7EB8F7;margin-bottom:8px">{title}</div>
              <ul style="padding-left:16px;margin:0">{items_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="report-section" style="margin-top:16px">
      <h4>⚡ Cross-Attention Transformer Fusion</h4>
      <p style="color:#CBD5E0;font-size:0.88rem">
        Each modality token attends to the other two as query-key-value context
        (cross-modal attention), followed by multi-head self-attention across all three
        tokens and a position-wise FFN + LayerNorm.
        Learnable scalar weights (softmax-normalised) determine the final contribution
        of each modality to the classification decision.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
      <strong>⚠️ Important Medical Disclaimer</strong><br>
      PneumoFusion-Net is a <strong>research prototype</strong> and is NOT approved as a medical
      device. All predictions are AI-assisted estimates and must be reviewed by a qualified
      healthcare professional before any clinical action is taken. The system may produce errors,
      particularly in atypical or rare presentations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="report-section">
      <h4>📚 Reference</h4>
      <p style="color:#CBD5E0;font-size:0.85rem">
        Wang Y, Liu C, Fan Y et al. (2025). <em>A multi-modal deep learning solution for precise
        pneumonia diagnosis: the PneumoFusion-Net model.</em>
        Front. Physiol. 16:1512835. doi: 10.3389/fphys.2025.1512835
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Diagnose":
        render_diagnose()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
