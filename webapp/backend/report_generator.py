"""
webapp/backend/report_generator.py
───────────────────────────────────
Generates structured AI medical reports using the OpenAI API (GPT-4.1).

Design principles
-----------------
• Clear anti-hallucination instructions in the system prompt.
• Model is instructed it is NOT a doctor and predictions are AI-assisted only.
• Structured JSON output for reliable parsing.
• Graceful fallback if the API is unavailable.
• Prompt references all available clinical evidence (image, labs, text).
"""

import os
import json
import textwrap
from typing import Dict, Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1")  # fallback: gpt-4o


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE RANGES  (for lab interpretation)
# ─────────────────────────────────────────────────────────────────────────────

LAB_REFERENCE = {
    "WBC (x10^9/L)":  {"low": 4.0,  "high": 11.0,  "unit": "×10⁹/L"},
    "NEUT%":          {"low": 50.0,  "high": 70.0,  "unit": "%"},
    "LYMP%":          {"low": 20.0,  "high": 40.0,  "unit": "%"},
    "NLR":            {"low": 1.0,   "high": 3.0,   "unit": "ratio"},
    "CRP (mg/L)":     {"low": 0.0,   "high": 10.0,  "unit": "mg/L"},
    "PCT (ng/mL)":    {"low": 0.0,   "high": 0.5,   "unit": "ng/mL"},
}


def _flag_lab_abnormalities(numerical_row: dict) -> list:
    """Return list of abnormality strings for the prompt."""
    flags = []
    for col, ref in LAB_REFERENCE.items():
        try:
            val = float(numerical_row.get(col, ref["low"]))
        except (ValueError, TypeError):
            continue
        if val < ref["low"]:
            flags.append(f"{col}: {val} {ref['unit']}  ← LOW  (ref {ref['low']}–{ref['high']})")
        elif val > ref["high"]:
            flags.append(f"{col}: {val} {ref['unit']}  ← HIGH  (ref {ref['low']}–{ref['high']})")
        else:
            flags.append(f"{col}: {val} {ref['unit']}  (normal)")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a clinical AI assistant integrated into the PneumoFusion-Net diagnostic system.
Your role is to generate clear, accurate, evidence-based medical summaries.

CRITICAL RULES — you MUST follow these without exception:
1. You are NOT a doctor. You do NOT provide a medical diagnosis.
2. You MUST NOT invent, fabricate, or hallucinate any clinical facts.
3. Only state what is directly supported by the data provided to you.
4. Always include a prominent disclaimer that this is AI-assisted analysis only.
5. Recommend the patient consult a qualified healthcare professional.
6. Flag any emergency warning signs that require immediate attention.
7. Use plain language for the patient-facing section; use clinical language
   for the clinician-facing section.
8. Be concise, structured, and compassionate in tone.

You must return ONLY valid JSON with the exact schema specified in the user prompt.
Do not include any text outside the JSON object.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_prompt(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    numerical_row: dict,
    clinical_text: str,
    gradcam_findings: str,
) -> str:
    lab_flags  = _flag_lab_abnormalities(numerical_row)
    lab_block  = "\n".join(f"  • {f}" for f in lab_flags)
    prob_block = "\n".join(
        f"  • {cls}: {p*100:.1f}%" for cls, p in
        sorted(probabilities.items(), key=lambda x: -x[1])
    )

    return textwrap.dedent(f"""
You have received the following AI-generated multimodal pneumonia analysis.
Generate a structured medical report as a JSON object with the exact schema below.

─── INPUT DATA ─────────────────────────────────────────────────────────────
Patient age   : {numerical_row.get('Patient_Age', 'N/A')}
Patient sex   : {numerical_row.get('Patient_Sex', 'N/A')}
Clinical notes: {clinical_text or 'Not provided'}

─── AI MODEL OUTPUT ────────────────────────────────────────────────────────
Primary prediction : {predicted_class}
Confidence         : {confidence*100:.1f}%

All class probabilities:
{prob_block}

─── LABORATORY RESULTS ─────────────────────────────────────────────────────
{lab_block}

─── GRAD-CAM IMAGING FINDINGS ──────────────────────────────────────────────
{gradcam_findings}

─── REQUIRED JSON SCHEMA ───────────────────────────────────────────────────
Return ONLY this JSON object (no markdown, no extra text):

{{
  "patient_summary": "2-3 sentences in plain language explaining the finding to a patient",
  "clinical_interpretation": "3-5 sentences of clinical analysis referencing labs, imaging and text",
  "confidence_explanation": "1-2 sentences explaining what the confidence score means",
  "key_abnormalities": ["list", "of", "notable", "lab", "or", "imaging", "findings"],
  "recommended_actions": ["list", "of", "specific", "next", "steps"],
  "emergency_warning_signs": ["list", "of", "symptoms", "requiring", "immediate", "care"],
  "disclaimer": "Standard AI disclaimer (1-2 sentences, NOT a diagnosis)"
}}
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    predicted_class:  str,
    confidence:       float,
    probabilities:    Dict[str, float],
    numerical_row:    dict,
    clinical_text:    str,
    gradcam_findings: str = "Grad-CAM heatmap highlights regions of lung abnormality.",
) -> dict:
    """
    Generate a structured medical report via OpenAI.

    Returns a dict with keys:
        patient_summary, clinical_interpretation, confidence_explanation,
        key_abnormalities, recommended_actions, emergency_warning_signs,
        disclaimer, _source ("openai" | "fallback")

    Never raises — returns a fallback report if the API is unavailable.
    """
    if not _OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return _fallback_report(predicted_class, confidence, probabilities, numerical_row, clinical_text)

    try:
        client     = OpenAI(api_key=OPENAI_API_KEY)
        user_prompt = _build_user_prompt(
            predicted_class, confidence, probabilities,
            numerical_row, clinical_text, gradcam_findings,
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,           # low temp for factual consistency
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        raw    = response.choices[0].message.content
        report = json.loads(raw)
        report["_source"] = "openai"
        return report

    except Exception as exc:
        print(f"[ReportGenerator] OpenAI error: {exc}")
        fb = _fallback_report(predicted_class, confidence, probabilities, numerical_row, clinical_text)
        fb["_error"] = str(exc)
        return fb


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK REPORT  (no API key / network error)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_report(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    numerical_row: dict,
    clinical_text: str,
) -> dict:
    """Rule-based fallback report when OpenAI is unavailable."""
    flags   = _flag_lab_abnormalities(numerical_row)
    abnormal = [f for f in flags if "HIGH" in f or "LOW" in f]

    is_bacterial = "Bacterial" in predicted_class
    is_viral     = "Viral"     in predicted_class
    is_tb        = "Tuberc"    in predicted_class
    is_normal    = "Normal"    in predicted_class

    if is_normal:
        summary = ("The AI system did not detect significant signs of pneumonia in the "
                   "CT scan or laboratory values. This is a preliminary AI-assisted result only.")
        actions = [
            "Schedule a follow-up with your doctor to review the results.",
            "Monitor for any new or worsening symptoms (fever, cough, breathlessness).",
            "Maintain adequate hydration and rest.",
        ]
    elif is_bacterial:
        summary = ("The AI system identified patterns consistent with bacterial pneumonia. "
                   "This includes typical consolidation patterns on CT and elevated inflammatory markers.")
        actions = [
            "Consult a physician immediately for antibiotic therapy evaluation.",
            "Obtain blood cultures before starting antibiotics if clinically feasible.",
            "Monitor oxygen saturation and respiratory rate.",
            "Consider hospitalisation if SpO2 < 94% or respiratory rate > 24/min.",
        ]
    elif is_viral:
        summary = ("The AI system identified patterns consistent with viral pneumonia. "
                   "Ground-glass opacities and lymphocyte patterns support this finding.")
        actions = [
            "Consult a physician for antiviral therapy assessment.",
            "Monitor for secondary bacterial infection (rising WBC/CRP).",
            "Adequate rest, hydration and symptomatic management.",
            "Seek emergency care if oxygen saturation drops or breathing worsens.",
        ]
    elif is_tb:
        summary = ("The AI system identified patterns that may be consistent with tuberculosis. "
                   "Upper-lobe involvement on CT is a noted feature.")
        actions = [
            "Urgent referral to a pulmonologist or infectious disease specialist.",
            "Sputum AFB smear and culture for TB confirmation.",
            "Isolate until infectiousness is assessed.",
            "Contact tracing may be required.",
        ]
    else:
        summary = f"The AI system predicted: {predicted_class}. Please consult a physician."
        actions = ["Consult a qualified healthcare professional for further evaluation."]

    return {
        "patient_summary":          summary,
        "clinical_interpretation":  (
            f"Multimodal AI analysis (PneumoFusion-Net) predicted {predicted_class} "
            f"with {confidence*100:.1f}% confidence. "
            f"{'Notable lab abnormalities: ' + '; '.join(abnormal[:3]) + '.' if abnormal else 'Lab values within reference ranges.'} "
            f"Clinical notes: {clinical_text[:200] if clinical_text else 'None provided'}."
        ),
        "confidence_explanation":   (
            f"A confidence of {confidence*100:.1f}% indicates the model's estimated "
            f"certainty for the top prediction. Scores above 85% are generally reliable; "
            f"lower scores suggest the case may be atypical and warrants careful review."
        ),
        "key_abnormalities":        abnormal[:6] if abnormal else ["No significant laboratory abnormalities detected."],
        "recommended_actions":      actions,
        "emergency_warning_signs":  [
            "Oxygen saturation below 90%",
            "Severe shortness of breath or inability to speak full sentences",
            "High fever (>39.5°C / 103.1°F) unresponsive to medication",
            "Coughing up blood",
            "Chest pain with breathing",
            "Altered consciousness or confusion",
        ],
        "disclaimer":               (
            "⚠️ This report is generated by an AI system (PneumoFusion-Net) and is NOT a "
            "medical diagnosis. It is intended solely as a decision-support tool. "
            "Always consult a qualified healthcare professional for medical advice and treatment."
        ),
        "_source": "fallback",
    }
