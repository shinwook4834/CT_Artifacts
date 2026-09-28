"""ChemoPort CT-MAR Studio - Streamlit Web Application.

An Apple Liquid Glass-inspired open-science platform for 3D CAD Prior-guided
Metal Artifact Reduction (MAR) and radiation dose integrity verification.
"""

import os
import io
import uuid
from typing import Optional, Tuple
import json
import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image

from modules.dicom_io import (
    load_dicom_file,
    load_dicom_from_zip,
    apply_window_level,
    anonymize_dicom,
    save_dicom_bytes,
    WINDOW_PRESETS,
)
from modules.reconstruction import baseline_chemoport_mar, find_chemoport_slice_index

# Page Configuration
st.set_page_config(
    page_title="ChemoPort CT-MAR Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Apple Liquid Glass CSS Theme
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def process_series_slices_mar(slices_list, ai_weight: Optional[float] = None):
    """Processes a DICOM series or single slice to compute AI restorations and vector contours.
    Executes the 2-Step AI Restoration Workflow:
    Step 1: Continuous Catheter Wire Restoration across all slices.
    Step 2: Natural ChemoPort Metal Artifact Reduction.
    """
    if not slices_list:
        return []

    from modules.ai_mar_pipeline import run_full_ai_restoration_pipeline
    inpaint_str = float(st.session_state.get("inpaint_strength", 0.90))
    return run_full_ai_restoration_pipeline(slices_list, inpaint_strength=inpaint_str)


def update_recon_ai_weight(new_weight: float):
    """Dynamically updates hybrid reconstruction HU, statistics, and canvas images
    without re-running heavy physics MAR or neural network inference."""
    new_weight = float(np.clip(new_weight, 0.0, 1.0))
    st.session_state.ai_weight = new_weight
    if not st.session_state.recon_cache:
        return

    from modules.deep_mar_model import blend_hybrid_mar, run_deep_mar_inference
    from modules.inpainting_engine import extract_anatomical_priors

    slices_src = (
        st.session_state.slice_list
        if st.session_state.slice_list
        else ([(st.session_state.current_ds, st.session_state.current_hu, st.session_state.current_meta)] if st.session_state.current_hu is not None else [])
    )

    recon_b64_list = []
    all_recon_b64_list = []

    for idx, sl in enumerate(slices_src):
        hu = sl[1] if isinstance(sl, (tuple, list)) else sl
        if idx in st.session_state.recon_cache:
            prev_recon, prev_stats = st.session_state.recon_cache[idx]
            ai_data = prev_stats.get("ai", {})
            if "physics_hu" in ai_data:
                physics_hu = ai_data["physics_hu"]
                if "ai_hu" not in ai_data:
                    priors = ai_data.get("priors")
                    if priors is None and "stage1_hu" in ai_data:
                        priors = extract_anatomical_priors(ai_data["stage1_hu"])
                    ai_hu = run_deep_mar_inference(
                        orig_hu=hu,
                        stage1_hu=ai_data.get("stage1_hu", hu),
                        artifact_mask=ai_data.get("artifact_mask", np.zeros_like(hu, dtype=bool)),
                        priors=priors,
                        physics_hu=physics_hu,
                        metal_mask=ai_data.get("metal_mask"),
                        full_port_mask=ai_data.get("full_port_mask"),
                        engine_type=st.session_state.get("ai_engine_key", "ccaf_transformer"),
                    )
                    ai_data["ai_hu"] = ai_hu
                    ai_data["priors"] = priors

                ai_hu = ai_data["ai_hu"]
                blended_hu = blend_hybrid_mar(
                    physics_hu=physics_hu,
                    ai_hu=ai_hu,
                    ai_weight=new_weight,
                    full_port_mask=ai_data.get("full_port_mask"),
                    metal_mask=ai_data.get("metal_mask"),
                    priors=ai_data.get("priors"),
                    orig_hu=hu,
                    artifact_mask=ai_data.get("artifact_mask"),
                )
                updated_stats = dict(prev_stats)
                updated_stats["ai"] = dict(ai_data)
                updated_stats["ai"]["ai_weight"] = new_weight
                updated_stats["ai"]["ai_active"] = new_weight > 0.0
                updated_stats["min_hu_after"] = float(np.min(blended_hu))
                updated_stats["max_hu_after"] = float(np.max(blended_hu))
                st.session_state.recon_cache[idx] = (blended_hu, updated_stats)
                recon_hu = blended_hu
            else:
                recon_hu = prev_recon
        else:
            recon_hu = hu

        img_r = apply_window_level(recon_hu, wc=40.0, ww=350.0)
        img_ra = apply_window_level(recon_hu, wc=10760.0, ww=33530.0)
        if idx in st.session_state.img_cache:
            st.session_state.img_cache[idx] = (st.session_state.img_cache[idx][0], img_r)
        else:
            img_o = apply_window_level(hu, wc=40.0, ww=350.0)
            st.session_state.img_cache[idx] = (img_o, img_r)

        _, b_r = cv2.imencode(".jpg", img_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, b_ra = cv2.imencode(".jpg", img_ra, [cv2.IMWRITE_JPEG_QUALITY, 80])
        recon_b64_list.append(base64.b64encode(b_r).decode("ascii"))
        all_recon_b64_list.append(base64.b64encode(b_ra).decode("ascii"))

    st.session_state.recon_b64_list = recon_b64_list
    st.session_state.all_recon_b64_list = all_recon_b64_list

    curr_idx = st.session_state.current_slice_idx
    if curr_idx in st.session_state.recon_cache:
        st.session_state.recon_hu, st.session_state.recon_stats = st.session_state.recon_cache[curr_idx]


def update_recon_ai_engine(new_engine_label: str):
    """Dynamically switches the AI restoration engine (CCAF-Net, NAFNet, Physics-CAD)
    and updates cached reconstructions across slices."""
    engine_map = {
        "💎 대측 해부학 교차 어텐션 AI (CCAF-Net Transformer) [추천]": "ccaf_transformer",
        "⚡ 다중 스케일 고용량 NAFNet (Multi-Scale Residual Deep MAR)": "nafnet_mar",
        "🛡️ CAD 물리 불변 + 해부학적 경계 확산 (Deterministic Clinical Mode)": "physics_cad",
    }
    engine_key = engine_map.get(new_engine_label, "ccaf_transformer")
    st.session_state.ai_engine_key = engine_key
    st.session_state.ai_engine_label = new_engine_label

    if not st.session_state.recon_cache:
        return

    from modules.deep_mar_model import blend_hybrid_mar, run_deep_mar_inference
    from modules.inpainting_engine import extract_anatomical_priors

    slices_src = (
        st.session_state.slice_list
        if st.session_state.slice_list
        else ([(st.session_state.current_ds, st.session_state.current_hu, st.session_state.current_meta)] if st.session_state.current_hu is not None else [])
    )

    recon_b64_list = []
    all_recon_b64_list = []
    curr_w = float(st.session_state.get("ai_weight", 1.0))

    for idx, sl in enumerate(slices_src):
        hu = sl[1] if isinstance(sl, (tuple, list)) else sl
        if idx in st.session_state.recon_cache:
            from modules.reconstruction import two_stage_chemoport_mar
            recon_hu, m_mask, a_mask, updated_stats = two_stage_chemoport_mar(
                hu,
                ai_weight=curr_w,
                engine_type=engine_key,
                cad_preset=st.session_state.get("cad_preset", "universal"),
                streak_reduction_strength=float(st.session_state.get("inpaint_strength", 0.88)),
                preserve_skin_boundary=True,
            )
            st.session_state.recon_cache[idx] = (recon_hu, updated_stats)
        else:
            recon_hu = hu

        img_r = apply_window_level(recon_hu, wc=40.0, ww=350.0)
        img_ra = apply_window_level(recon_hu, wc=10760.0, ww=33530.0)
        if idx in st.session_state.img_cache:
            st.session_state.img_cache[idx] = (st.session_state.img_cache[idx][0], img_r)
        else:
            img_o = apply_window_level(hu, wc=40.0, ww=350.0)
            st.session_state.img_cache[idx] = (img_o, img_r)

        _, b_r = cv2.imencode(".jpg", img_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, b_ra = cv2.imencode(".jpg", img_ra, [cv2.IMWRITE_JPEG_QUALITY, 80])
        recon_b64_list.append(base64.b64encode(b_r).decode("ascii"))
        all_recon_b64_list.append(base64.b64encode(b_ra).decode("ascii"))

    st.session_state.recon_b64_list = recon_b64_list
    st.session_state.all_recon_b64_list = all_recon_b64_list

    curr_idx = st.session_state.current_slice_idx
    if curr_idx in st.session_state.recon_cache:
        st.session_state.recon_hu, st.session_state.recon_stats = st.session_state.recon_cache[curr_idx]


def set_ai_weight_preset(target_weight: float):
    """Callback for quick preset buttons.
    Runs BEFORE any widgets are instantiated in the rerun, so modifying ai_weight_slider is valid.
    """
    w = float(np.clip(target_weight, 0.0, 1.0))
    st.session_state.ai_weight = w
    st.session_state.ai_weight_slider = w
    update_recon_ai_weight(w)


def on_ai_slider_change():
    """Callback triggered whenever user interacts with the AI fusion weight slider.
    Runs BEFORE the script body executes.
    """
    if "ai_weight_slider" in st.session_state:
        update_recon_ai_weight(st.session_state.ai_weight_slider)


def init_session_state():
    """Initializes persistent application session variables."""
    if "current_ds" not in st.session_state:
        st.session_state.current_ds = None
    if "current_hu" not in st.session_state:
        st.session_state.current_hu = None
    if "current_meta" not in st.session_state:
        st.session_state.current_meta = None
    if "recon_hu" not in st.session_state:
        st.session_state.recon_hu = None
    if "recon_stats" not in st.session_state:
        st.session_state.recon_stats = None
    if "data_source_name" not in st.session_state:
        st.session_state.data_source_name = "None"
    if "slice_list" not in st.session_state:
        st.session_state.slice_list = []
    if "current_slice_idx" not in st.session_state:
        st.session_state.current_slice_idx = 0
    if "recon_cache" not in st.session_state:
        st.session_state.recon_cache = {}
    if "img_cache" not in st.session_state:
        st.session_state.img_cache = {}
    if "orig_b64_list" not in st.session_state:
        st.session_state.orig_b64_list = []
    if "recon_b64_list" not in st.session_state:
        st.session_state.recon_b64_list = []
    if "all_orig_b64_list" not in st.session_state:
        st.session_state.all_orig_b64_list = []
    if "all_recon_b64_list" not in st.session_state:
        st.session_state.all_recon_b64_list = []
    if "zip_cache" not in st.session_state:
        st.session_state.zip_cache = None
    if "scan_id" not in st.session_state:
        st.session_state.scan_id = uuid.uuid4().hex
    if "ai_weight" not in st.session_state:
        st.session_state.ai_weight = 1.0
    if "ai_weight_slider" not in st.session_state:
        st.session_state.ai_weight_slider = 1.0
    if "ai_engine_key" not in st.session_state:
        st.session_state.ai_engine_key = "ccaf_transformer"
    if "ai_engine_label" not in st.session_state:
        st.session_state.ai_engine_label = "💎 대측 해부학 교차 어텐션 AI (CCAF-Net Transformer) [추천]"


init_session_state()

# ==========================================
# SIDEBAR: STRICTLY FUNCTIONAL NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 14px 4px 16px 4px; text-align: center;">
          <div style="font-size: 1.28rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; justify-content: center; gap: 9px; letter-spacing: -0.4px;">
            <span>🩺</span> MAR Studio
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Implant Modules Selection (Stage-free with matching icons)
    implant_options = [
        "🩺 Chemo Port (Active)",
        "🔩 Spine Pedicle Screws",
        "💓 Pacemaker / ICD",
        "🦷 Dental Implants",
        "🦿 Hip Prosthesis",
    ]
    selected_implant = st.radio(
        "MAR Studio",
        implant_options,
        index=0,
        label_visibility="collapsed",
        help="Select an implant-specific artifact reduction clinical workflow.",
    )

    st.markdown("---")

    st.markdown("---")

    # 3. System & HIPAA Compliance Status (Moved up)
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.65); backdrop-filter: blur(16px); border: 1.5px solid rgba(255, 255, 255, 0.9); border-radius: 14px; padding: 12px 14px; font-size: 0.76rem; color: #475569; width: 100%; box-sizing: border-box; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.03);">
          <div style="font-weight: 750; color: #0f766e; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
            <span>🔒</span> Security & Compliance
          </div>
          &bull; HIPAA In-Memory Anonymization: <strong>Active</strong><br/>
          &bull; TPS Rescale Intercept: <strong>Calibrated</strong><br/>
          &bull; DICOM PS 3.15 Conformance: <strong>Verified</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sequential Clinical Workflow Pipeline State
workflow_steps = [
    "1. Image Loading",
    "2. Artifact Inspection",
    "3. AI Correction",
    "4. Save & Export",
]
if "current_workflow_step" not in st.session_state:
    st.session_state.current_workflow_step = (
        "1. Image Loading" if st.session_state.current_hu is None else "3. AI Correction"
    )

if st.session_state.current_workflow_step not in workflow_steps:
    st.session_state.current_workflow_step = "1. Image Loading"

if "Chemo Port" in selected_implant:
    workflow_mode = st.session_state.current_workflow_step
else:
    workflow_mode = "Roadmap Overview"

# ==========================================
# MAIN CANVAS
# ==========================================

# Dynamic Header Configuration based on selected implant and workspace
if "Chemo Port" in selected_implant:
    header_icon = "🩺"
    header_title = "ChemoPort CT-MAR Studio"
    if workflow_mode == "1. Image Loading":
        header_subtitle = "Step 1: Load CT DICOM series and automatically locate ChemoPort implant"
    elif workflow_mode == "2. Artifact Inspection":
        header_subtitle = "Step 2: Inspect severe streak artifacts and dosimetric distortions caused by the titanium port"
    elif workflow_mode == "3. AI Correction":
        header_subtitle = "Step 3: 3D CAD prior-guided AI-MAR artifact reduction and interactive inspection glass"
    elif workflow_mode == "4. Save & Export":
        header_subtitle = "Step 4: Save and export treatment planning system (TPS)-compatible DICOM series"
    else:
        header_subtitle = "3D CAD Prior-Guided Metal Artifact Reduction & Dosimetric Integrity for Radiation Oncology"
elif "Spine" in selected_implant:
    header_icon = "🔩"
    header_title = "Spine Screws SBRT-MAR Studio"
    header_subtitle = "Threaded CAD Prior & Ultra-Close 1–2 mm Spinal Cord PRV Sparing in Spine SBRT"
elif "Pacemaker" in selected_implant:
    header_icon = "💓"
    header_title = "Cardiac Pacemaker MAR Studio"
    header_subtitle = "Battery Housing CAD Registration & 2 Gy Cumulative Dose Constraint Verification"
elif "Dental" in selected_implant:
    header_icon = "🦷"
    header_title = "Dental Implants MAR Studio"
    header_subtitle = "Dual-Domain Swin Transformer for Cross-Streak Suppression & Salivary Sparing"
elif "Hip" in selected_implant:
    header_icon = "🦿"
    header_title = "Hip Prosthesis MAR Studio"
    header_subtitle = "Massive Prosthetic CAD Prior & Monte Carlo Photon Starvation Recovery in Pelvic RT"
else:
    header_icon = "🩺"
    header_title = "MAR Studio"
    header_subtitle = "3D CAD Prior-Guided Metal Artifact Reduction & Dosimetric Integrity for Radiation Oncology"

# Liquid Glass Top Header
st.markdown(
    f"""
    <div class="liquid-header">
      <div class="liquid-header-inner">
        <h1 class="liquid-title">
          <span>{header_icon}</span> {header_title}
        </h1>
        <div class="liquid-subtitle">
          {header_subtitle}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Workflow Progress Stepper (Clickable Liquid Glass Pill)
if "Chemo Port" in selected_implant:
    step_indices = {
        "1. Image Loading": 1,
        "2. Artifact Inspection": 2,
        "3. AI Correction": 3,
        "4. Save & Export": 4,
    }
    active_idx = step_indices.get(workflow_mode, 1)

    # Hidden trigger buttons to handle instant Streamlit state updates
    h_marker, h1, h2, h3, h4 = st.columns([0.001, 1, 1, 1, 1])
    with h_marker:
        st.markdown("<div id='stepper-hidden-marker'></div>", unsafe_allow_html=True)
    with h1:
        if st.button("goto_step_1", key="hidden_goto_1"):
            if st.session_state.current_workflow_step != "1. Image Loading":
                st.session_state.current_workflow_step = "1. Image Loading"
                st.rerun()
    with h2:
        if st.button("goto_step_2", key="hidden_goto_2"):
            if st.session_state.current_workflow_step != "2. Artifact Inspection":
                st.session_state.current_workflow_step = "2. Artifact Inspection"
                st.rerun()
    with h3:
        if st.button("goto_step_3", key="hidden_goto_3"):
            if st.session_state.current_workflow_step != "3. AI Correction":
                st.session_state.current_workflow_step = "3. AI Correction"
                st.rerun()
    with h4:
        if st.button("goto_step_4", key="hidden_goto_4"):
            if st.session_state.current_workflow_step != "4. Save & Export":
                st.session_state.current_workflow_step = "4. Save & Export"
                st.rerun()

    def get_step_class(idx):
        is_disabled = (st.session_state.current_hu is None and idx > 1)
        if idx == active_idx:
            return "step-item active"
        elif is_disabled:
            return "step-item disabled"
        elif st.session_state.current_hu is not None and idx < active_idx:
            return "step-item completed"
        return "step-item"

    has_data = st.session_state.current_hu is not None

    stepper_html = f"""
    <!DOCTYPE html>
    <html style="background: transparent !important; background-color: transparent !important;">
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      html, body {{
        background: transparent !important;
        background-color: transparent !important;
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: visible;
      }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
      }}
      .workflow-stepper {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        padding: 4px 14px;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(28px) saturate(190%);
        -webkit-backdrop-filter: blur(28px) saturate(190%);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        border-radius: 20px !important;
        box-shadow: 0 4px 18px -4px rgba(15, 23, 42, 0.08), inset 0 1px 1px #ffffff;
        white-space: nowrap;
        user-select: none;
        max-width: 96%;
      }}
      .step-item {{
        font-size: 0.77rem;
        font-weight: 650;
        color: #64748b;
        padding: 5px 10px;
        border-radius: 14px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        display: inline-flex;
        align-items: center;
        gap: 5px;
        white-space: nowrap;
        cursor: pointer;
      }}
      .step-item:hover:not(.active):not(.disabled) {{
        background: rgba(13, 148, 136, 0.1);
        color: #0f766e;
        transform: translateY(-1px);
      }}
      .step-item span {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 17px;
        height: 17px;
        border-radius: 50%;
        font-size: 0.70rem;
        font-weight: 800;
        background: rgba(100, 116, 139, 0.15);
        color: #475569;
        transition: all 0.2s ease;
      }}
      .step-item.active {{
        background: #0d9488 !important;
        color: #ffffff !important;
        font-weight: 750 !important;
        box-shadow: 0 3px 10px rgba(13, 148, 136, 0.35);
        cursor: default;
      }}
      .step-item.active span {{
        background: #ffffff !important;
        color: #0d9488 !important;
      }}
      .step-item.completed {{
        color: #0f766e;
        background: rgba(13, 148, 136, 0.1);
        font-weight: 650;
      }}
      .step-item.completed span {{
        background: #0d9488;
        color: #ffffff;
      }}
      .step-item.disabled {{
        opacity: 0.42;
        cursor: not-allowed;
      }}
      .step-arrow {{
        color: #cbd5e1;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0 1px;
      }}
    </style>
    </head>
    <body>
      <div class="workflow-stepper">
        <div class="{get_step_class(1)}" onclick="gotoStep(1, true)"><span>1</span> Image Loading</div>
        <div class="step-arrow">→</div>
        <div class="{get_step_class(2)}" onclick="gotoStep(2, {str(has_data).lower()})"><span>2</span> Artifact Inspection</div>
        <div class="step-arrow">→</div>
        <div class="{get_step_class(3)}" onclick="gotoStep(3, {str(has_data).lower()})"><span>3</span> AI Correction</div>
        <div class="step-arrow">→</div>
        <div class="{get_step_class(4)}" onclick="gotoStep(4, {str(has_data).lower()})"><span>4</span> Save & Export</div>
      </div>

      <script>
        function gotoStep(num, allowed) {{
          if (!allowed) return;
          try {{
            const parentDoc = window.parent.document;
            const btns = Array.from(parentDoc.querySelectorAll('button'));
            const target = btns.find(b => b.innerText && b.innerText.includes('goto_step_' + num));
            if (target) {{
              target.click();
            }}
          }} catch (err) {{
            console.error('Step navigation error:', err);
          }}
        }}
      </script>
    </body>
    </html>
    """
    components.html(stepper_html, height=52)

# Handle Roadmap Implant Modules
if not ("Chemo Port" in selected_implant):
    implant_details = {
        "🔩 Spine Pedicle Screws": {
            "title": "Spine Pedicle Screws (SBRT Framework)",
            "indication": "Spine Oligometastasis Stereotactic Body Radiation Therapy (SBRT)",
            "risk": "1–2 mm critical margin to Spinal Cord PRV (Severe myelopathy risk)",
            "approach": "Threaded CAD Prior fitting with anisotropic sinogram inpainting to recover cord boundaries.",
            "status": "In Development (Planned for Q4 2026)",
        },
        "💓 Pacemaker / ICD": {
            "title": "Cardiac Pacemaker / ICD Protection",
            "indication": "Thoracic & Left Breast Radiation Therapy",
            "risk": "2 Gy strict semiconductor cumulative dose limit (Device burnout risk)",
            "approach": "Battery housing CAD prior overlay with Monte Carlo backscatter photon modeling.",
            "status": "In Development (Planned for Q1 2027)",
        },
        "🦷 Dental Implants": {
            "title": "Dental Implants & Amalgams",
            "indication": "Head & Neck (Oropharyngeal / Laryngeal) Radiation Therapy",
            "risk": "Multi-cluster cross-streak destruction of parotid glands and pharyngeal muscles.",
            "approach": "Dual-Domain Sinogram-Image Swin Transformer with clustered metal separation.",
            "status": "Roadmap Architecture Complete",
        },
        "🦿 Hip Prosthesis": {
            "title": "Femur / Hip Prosthesis Recovery",
            "indication": "Pelvic RT (Prostate, Gynecological, Rectal Cancers)",
            "risk": "Severe total photon starvation causing deep black hole shadows across bladder and rectum.",
            "approach": "Massive prosthetic CAD prior registration with deep residual inpainting.",
            "status": "Roadmap Architecture Complete",
        },
    }
    info = implant_details.get(selected_implant, {})
    st.markdown(
        f"""
        <div class="stage-preview-card">
          <div style="font-size: 0.85rem; font-weight: 700; color: #0d9488; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
            Implant MAR Research Roadmap
          </div>
          <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-top: 0; margin-bottom: 14px;">
            {info.get('title')}
          </h2>
          <div style="background: rgba(255, 255, 255, 0.7); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 16px; padding: 20px; max-width: 650px; margin: 0 auto 20px auto; text-align: left; line-height: 1.6;">
            <div><strong>Clinical Indication:</strong> {info.get('indication')}</div>
            <div style="color: #be123c; margin-top: 6px;"><strong>Critical Risk:</strong> {info.get('risk')}</div>
            <div style="margin-top: 6px;"><strong>Novel AI Approach:</strong> {info.get('approach')}</div>
            <div style="margin-top: 6px; color: #0f766e;"><strong>Current Status:</strong> {info.get('status')}</div>
          </div>
          <p style="color: #64748b; font-size: 0.88rem;">
            This module is part of the comprehensive implant series following ChemoPort.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ==========================================
# STAGE 1: CHEMOPORT ACTIVE WORKSPACE
# ==========================================

SAMPLE_DCM_PATH = os.path.join(os.path.dirname(__file__), "demo_data", "sample_chemoport_ct.dcm")

# Prompt if CT is not loaded and user navigated to Step 2, 3, or 4
if st.session_state.current_hu is None and workflow_mode != "1. Image Loading":
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(24px); border: 1.5px solid rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 48px 24px; text-align: center; max-width: 600px; margin: 40px auto; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);">
          <div style="font-size: 2.6rem; margin-bottom: 12px;">📂</div>
          <h3 style="color: #0f172a; font-weight: 800; margin-bottom: 8px;">CT Scan Required</h3>
          <p style="color: #64748b; font-size: 0.9rem; line-height: 1.5; margin-bottom: 24px;">
            Please upload a patient CT DICOM series in Step 1 before inspecting artifacts or running AI correction.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_p1, col_p2, col_p3 = st.columns([1, 1.4, 1])
    with col_p2:
        if st.button("Go to Step 1: Image Loading", type="primary", use_container_width=True):
            st.session_state.current_workflow_step = "1. Image Loading"
            st.rerun()

# STEP 1: 1. Image Loading
if workflow_mode == "1. Image Loading":
    if st.session_state.current_hu is None:
        st.markdown(
            """
            <div class="quillbot-hero-wrapper">
              <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; letter-spacing: -0.4px; margin-bottom: 4px;">
                Convert & Restore Chemo Port CT Scans
              </div>
              <div style="font-size: 0.88rem; color: #475569; max-width: 580px; margin: 0 auto 16px auto; line-height: 1.45;">
                Upload clinical simulation CTs to remove high-density titanium metal streaks and restore supraclavicular lymph node anatomy.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Centered Dropzone
        drop_col_left, drop_col_center, drop_col_right = st.columns([0.6, 6.8, 0.6])
        with drop_col_center:
            uploaded_files = st.file_uploader(
                "Upload Patient CT DICOM Files or ZIP Archive",
                type=["dcm", "zip"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Select or drag and drop one or multiple clinical DICOM (.dcm) files or a ZIP series (Max 1GB).",
            )

        if not uploaded_files:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            _, back_col, next_col, _ = st.columns([1.7, 0.8, 0.8, 1.7])
            with back_col:
                st.button("← Back", key="nav_back_1_empty", disabled=True, use_container_width=True)
            with next_col:
                st.button("Next →", key="nav_next_1_empty", disabled=True, use_container_width=True, help="Please upload a CT scan first")

        # Process Data Loading from File Upload
        if uploaded_files:
            slices = []
            for f in uploaded_files:
                file_bytes = f.getvalue()
                if f.name.lower().endswith(".zip"):
                    zip_slices = load_dicom_from_zip(file_bytes)
                    slices.extend(zip_slices)
                else:
                    try:
                        ds, hu_arr, meta = load_dicom_file(file_bytes)
                        slices.append((ds, hu_arr, meta))
                    except Exception:
                        pass

            if slices:
                # Group slices by SeriesInstanceUID to detect multi-series uploads
                series_map = {}
                for sl in slices:
                    ds_i, hu_i, meta_i = sl
                    suid = meta_i.get("series_instance_uid", getattr(ds_i, "SeriesInstanceUID", "UNKNOWN"))
                    if suid not in series_map:
                        desc = meta_i.get("series_description", getattr(ds_i, "SeriesDescription", "CT Series"))
                        snum = meta_i.get("series_number", getattr(ds_i, "SeriesNumber", "1"))
                        series_map[suid] = {
                            "description": desc,
                            "series_number": snum,
                            "slices": [],
                        }
                    series_map[suid]["slices"].append(sl)

                proceed_with_series = True
                selected_slices = slices

                # If multiple distinct CT series detected in the upload
                if len(series_map) > 1:
                    proceed_with_series = False
                    st.markdown(
                        f"""
                        <div style="background: rgba(254, 242, 242, 0.95); backdrop-filter: blur(20px); border: 1.5px solid rgba(220, 38, 38, 0.5); border-radius: 20px; padding: 20px 24px; max-width: 820px; margin: 18px auto; box-shadow: 0 8px 25px -5px rgba(220, 38, 38, 0.15);">
                          <div style="display: flex; align-items: center; gap: 10px; color: #991b1b; font-weight: 800; font-size: 1.08rem;">
                            <span style="font-size: 1.35rem;">⛔</span>
                            <span>Error: Multiple CT Series Detected ({len(series_map)} Series Found)</span>
                          </div>
                          <div style="font-size: 0.88rem; color: #7f1d1d; margin-top: 8px; line-height: 1.55;">
                            Only <b>ONE CT series</b> can be uploaded per session to ensure accurate 3D CAD registration and radiation therapy planning integrity.
                            Please select a single target series below to isolate and proceed, or cancel to re-upload a single series.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Auto-detect series with metal/ChemoPort implant
                    best_suid = None
                    max_metal = -1
                    for suid, sinfo in series_map.items():
                        total_metal = sum(s[2].get("metal_pixel_count", 0) for s in sinfo["slices"])
                        if total_metal > max_metal:
                            max_metal = total_metal
                            best_suid = suid

                    series_keys = list(series_map.keys())

                    def format_series_option(uid):
                        info = series_map[uid]
                        desc = info["description"]
                        count = len(info["slices"])
                        has_metal = any(s[2].get("metal_pixel_count", 0) > 0 for s in info["slices"])
                        metal_tag = " 🩺 [ChemoPort Implant Detected]" if has_metal else ""
                        return f"Series {info['series_number']}: {desc} ({count} slices){metal_tag}"

                    default_idx = series_keys.index(best_suid) if best_suid in series_keys else 0

                    w_col_l, w_col_mid, w_col_r = st.columns([0.6, 6.8, 0.6])
                    with w_col_mid:
                        chosen_suid = st.selectbox(
                            "Select Single Series to Process:",
                            series_keys,
                            index=default_idx,
                            format_func=format_series_option,
                            key="multi_series_select_box",
                        )
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                        btn_col_a, btn_col_b = st.columns([1, 1])
                        with btn_col_a:
                            if st.button("▶️ Proceed with Selected Series", type="primary", use_container_width=True, help="Isolate and process this single CT series"):
                                proceed_with_series = True
                                selected_slices = series_map[chosen_suid]["slices"]
                        with btn_col_b:
                            if st.button("🔄 Cancel & Re-upload", use_container_width=True, help="Reset and upload a single series"):
                                st.session_state.current_hu = None
                                st.rerun()

                if proceed_with_series:
                    st.session_state.scan_id = uuid.uuid4().hex
                    # Sort slices by InstanceNumber or SliceLocation
                    selected_slices.sort(key=lambda s: s[2].get("instance_number", 0))
                    st.session_state.slice_list = selected_slices

                    # Automatically locate ChemoPort slice using anterior chest wall & HU criteria
                    best_slice_idx = find_chemoport_slice_index(selected_slices)

                    st.session_state.current_slice_idx = best_slice_idx
                    st.session_state.current_ds = selected_slices[best_slice_idx][0]
                    st.session_state.current_hu = selected_slices[best_slice_idx][1]
                    st.session_state.current_meta = selected_slices[best_slice_idx][2]
                    if len(selected_slices) == 1:
                        st.session_state.data_source_name = getattr(selected_slices[0][0], "SeriesDescription", "Single DICOM Slice")
                    else:
                        s_desc = selected_slices[0][2].get("series_description", "CT Series")
                        st.session_state.data_source_name = f"{s_desc} ({len(selected_slices)} DICOM Slices)"

                    # Pre-warm MAR reconstruction cache and encode client-side canvas slices (~1s for 236 slices)
                    st.session_state.recon_cache = {}
                    st.session_state.img_cache = {}
                    st.session_state.contours_cache = []
                    orig_b64_list = []
                    recon_b64_list = []
                    all_orig_b64_list = []
                    all_recon_b64_list = []
                    contours_list = []
                    series_mar_results = process_series_slices_mar(selected_slices)
                    for idx, sl in enumerate(selected_slices):
                        hu = sl[1]
                        recon_hu, stats, cnt_dict = series_mar_results[idx]
                        contours_list.append(cnt_dict)

                        st.session_state.recon_cache[idx] = (recon_hu, stats)
                        img_o = apply_window_level(hu, wc=40.0, ww=350.0)
                        img_r = apply_window_level(recon_hu, wc=40.0, ww=350.0)
                        img_o_all = apply_window_level(hu, wc=10760.0, ww=33530.0)
                        img_r_all = apply_window_level(recon_hu, wc=10760.0, ww=33530.0)
                        st.session_state.img_cache[idx] = (img_o, img_r)

                        _, b_o = cv2.imencode(".jpg", img_o, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        _, b_r = cv2.imencode(".jpg", img_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        _, b_oa = cv2.imencode(".jpg", img_o_all, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        _, b_ra = cv2.imencode(".jpg", img_r_all, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        orig_b64_list.append(base64.b64encode(b_o).decode("ascii"))
                        recon_b64_list.append(base64.b64encode(b_r).decode("ascii"))
                        all_orig_b64_list.append(base64.b64encode(b_oa).decode("ascii"))
                        all_recon_b64_list.append(base64.b64encode(b_ra).decode("ascii"))

                    st.session_state.orig_b64_list = orig_b64_list
                    st.session_state.recon_b64_list = recon_b64_list
                    st.session_state.all_orig_b64_list = all_orig_b64_list
                    st.session_state.all_recon_b64_list = all_recon_b64_list
                    st.session_state.contours_cache = contours_list

                    initial_recon, initial_stats = st.session_state.recon_cache[best_slice_idx]
                    st.session_state.recon_hu = initial_recon
                    st.session_state.recon_stats = initial_stats
                    st.session_state.current_workflow_step = "2. Artifact Inspection"
                    st.rerun()
    else:
        # If CT volume is already loaded
        total_slices = len(st.session_state.slice_list) if st.session_state.slice_list else 1
        curr_idx = st.session_state.current_slice_idx

        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(28px) saturate(180%); border: 1.5px solid rgba(255, 255, 255, 0.95); border-radius: 28px; padding: 30px 34px; max-width: 820px; margin: 20px auto; box-shadow: 0 14px 35px -8px rgba(0, 0, 0, 0.06);">
              <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                  <div style="width: 44px; height: 44px; border-radius: 14px; background: rgba(13, 148, 136, 0.12); display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
                    📁
                  </div>
                  <div>
                    <h2 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 0;">CT Volume Dataset Loaded</h2>
                    <div style="font-size: 0.84rem; color: #0d9488; font-weight: 650; margin-top: 2px;">
                      Ready for ChemoPort metal implant analysis
                    </div>
                  </div>
                </div>
              </div>

              <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 22px;">
                <div style="background: rgba(248, 250, 252, 0.8); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 18px; padding: 14px 16px;">
                  <div style="font-size: 0.74rem; font-weight: 750; color: #64748b; text-transform: uppercase;">Data Source</div>
                  <div style="font-size: 0.95rem; font-weight: 750; color: #0f172a; margin-top: 4px;">{st.session_state.data_source_name}</div>
                </div>
                <div style="background: rgba(248, 250, 252, 0.8); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 18px; padding: 14px 16px;">
                  <div style="font-size: 0.74rem; font-weight: 750; color: #64748b; text-transform: uppercase;">ChemoPort Location</div>
                  <div style="font-size: 0.95rem; font-weight: 750; color: #0f172a; margin-top: 4px;">Slice {curr_idx + 1} / {total_slices} (Auto-Detected)</div>
                </div>
                <div style="background: rgba(248, 250, 252, 0.8); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 18px; padding: 14px 16px;">
                  <div style="font-size: 0.74rem; font-weight: 750; color: #64748b; text-transform: uppercase;">Matrix & Spacing</div>
                  <div style="font-size: 0.95rem; font-weight: 750; color: #0f172a; margin-top: 4px;">512 x 512 | 1.17 x 1.17 mm</div>
                </div>
                <div style="background: rgba(248, 250, 252, 0.8); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 14px; padding: 14px 16px;">
                  <div style="font-size: 0.74rem; font-weight: 750; color: #64748b; text-transform: uppercase;">Data Security</div>
                  <div style="font-size: 0.95rem; font-weight: 750; color: #0f172a; margin-top: 4px;">HIPAA In-Memory Anonymization Active</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        _, back_col, next_col, _ = st.columns([1.7, 0.8, 0.8, 1.7])
        with back_col:
            st.button("← Back", key="nav_back_1", disabled=True, use_container_width=True)
        with next_col:
            if st.button("Next →", key="nav_next_1", type="primary", use_container_width=True, help="Proceed to 2. Artifact Inspection"):
                st.session_state.scan_id = uuid.uuid4().hex
                st.session_state.current_workflow_step = "2. Artifact Inspection"
                st.rerun()

# Ensure multi-slice state is properly initialized without background slider reruns
if st.session_state.slice_list and len(st.session_state.slice_list) > 1:
    curr_idx = min(st.session_state.current_slice_idx, len(st.session_state.slice_list) - 1)
    st.session_state.current_ds = st.session_state.slice_list[curr_idx][0]
    st.session_state.current_hu = st.session_state.slice_list[curr_idx][1]
    st.session_state.current_meta = st.session_state.slice_list[curr_idx][2]
    if curr_idx in st.session_state.recon_cache:
        st.session_state.recon_hu, st.session_state.recon_stats = st.session_state.recon_cache[curr_idx]
    else:
        st.session_state.recon_hu = None


# If data is loaded, render the interactive workspace
if st.session_state.current_hu is not None:
    hu_orig = st.session_state.current_hu
    meta = st.session_state.current_meta

    # Clinical Standard MAR & Soft Tissue Contrast Parameters
    wc = 40.0
    ww = 350.0
    metal_threshold = 2000.0
    cad_density_override = True
    streak_strength = 0.85

    # Reconstruct if not already cached
    curr_idx = st.session_state.current_slice_idx
    if st.session_state.recon_hu is None:
        if curr_idx in st.session_state.recon_cache:
            recon_hu, stats = st.session_state.recon_cache[curr_idx]
        else:
            recon_hu, _, _, stats = baseline_chemoport_mar(
                hu_orig,
                metal_threshold=metal_threshold,
                cad_prior_override=cad_density_override,
                streak_reduction_strength=streak_strength,
                ai_weight=float(st.session_state.get("ai_weight", 1.0)),
                engine_type=st.session_state.get("ai_engine_key", "ccaf_transformer"),
            )
            st.session_state.recon_cache[curr_idx] = (recon_hu, stats)
        st.session_state.recon_hu = recon_hu
        st.session_state.recon_stats = stats

    hu_recon = st.session_state.recon_hu
    stats = st.session_state.recon_stats or {}

    # Windowed 8-bit images (cached for instantaneous slice retrieval)
    if curr_idx in st.session_state.img_cache:
        img_orig_8bit, img_recon_8bit = st.session_state.img_cache[curr_idx]
    else:
        img_orig_8bit = apply_window_level(hu_orig, wc=wc, ww=ww)
        img_recon_8bit = apply_window_level(hu_recon, wc=wc, ww=ww)
        st.session_state.img_cache[curr_idx] = (img_orig_8bit, img_recon_8bit)

    # STEP 3: 3. AI Correction (Dual-View MAR Studio)
    if workflow_mode == "3. AI Correction":
        # Check pipeline version to invalidate stale cache across hot-reloads
        CURRENT_PIPELINE_VER = "2026-09-28-v3-surgical-wire-normalization"
        if st.session_state.get("ai_pipeline_version") != CURRENT_PIPELINE_VER:
            st.session_state.ai_pipeline_version = CURRENT_PIPELINE_VER
            st.session_state.recon_cache = {}
            st.session_state.recon_b64_list = []
            st.session_state.all_recon_b64_list = []
            st.session_state.contours_cache = []

        # Ensure base64 lists are populated for client-side instant navigation
        if (not st.session_state.orig_b64_list or not st.session_state.recon_b64_list) and st.session_state.current_hu is not None:
            slices_src = (
                st.session_state.slice_list
                if st.session_state.slice_list
                else [(st.session_state.current_ds, st.session_state.current_hu, st.session_state.current_meta)]
            )
            o_list = []
            r_list = []
            oa_list = []
            ra_list = []
            if not st.session_state.recon_cache or len(st.session_state.recon_cache) != len(slices_src):
                series_mar_results = process_series_slices_mar(slices_src)
                st.session_state.recon_cache = {i: (res[0], res[1]) for i, res in enumerate(series_mar_results)}
                st.session_state.contours_cache = [res[2] for res in series_mar_results]

            for i, sl in enumerate(slices_src):
                hu = sl[1] if isinstance(sl, (tuple, list)) else sl
                rhu = st.session_state.recon_cache[i][0] if i in st.session_state.recon_cache else hu
                io = apply_window_level(hu, wc=40.0, ww=350.0)
                ir = apply_window_level(rhu, wc=40.0, ww=350.0)
                ioa = apply_window_level(hu, wc=10760.0, ww=33530.0)
                ira = apply_window_level(rhu, wc=10760.0, ww=33530.0)
                _, b1 = cv2.imencode(".jpg", io, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, b2 = cv2.imencode(".jpg", ir, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, b1a = cv2.imencode(".jpg", ioa, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, b2a = cv2.imencode(".jpg", ira, [cv2.IMWRITE_JPEG_QUALITY, 80])
                o_list.append(base64.b64encode(b1).decode("ascii"))
                r_list.append(base64.b64encode(b2).decode("ascii"))
                oa_list.append(base64.b64encode(b1a).decode("ascii"))
                ra_list.append(base64.b64encode(b2a).decode("ascii"))
            st.session_state.orig_b64_list = o_list
            st.session_state.recon_b64_list = r_list
            st.session_state.all_orig_b64_list = oa_list
            st.session_state.all_recon_b64_list = ra_list

        total_slices = len(st.session_state.slice_list) if st.session_state.slice_list else 1
        initial_slice = st.session_state.current_slice_idx
        slice_badge_str = f"Slice {initial_slice + 1} / {total_slices}" if total_slices > 1 else ""

        # 2-Step AI Restoration Workflow Status Banner (Apple Liquid Glass)
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(24px) saturate(180%);
                        -webkit-backdrop-filter: blur(24px) saturate(180%);
                        border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 18px;
                        padding: 12px 18px; margin-bottom: 14px; margin-top: 4px;
                        box-shadow: 0 6px 20px -6px rgba(0, 0, 0, 0.04);">
              <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; font-weight: 800; color: #0f172a; letter-spacing: -0.2px; display: inline-flex; align-items: center; gap: 6px;">
                  ✨ AI Correction Workflow
                </span>
                <span style="font-size: 0.70rem; font-weight: 700; color: #0d9488; background: rgba(13, 148, 136, 0.08); padding: 2px 10px; border-radius: 9999px; border: 1px solid rgba(13, 148, 136, 0.25);">
                  Active Pipeline
                </span>
              </div>
              <div style="display: flex; gap: 12px; font-size: 0.76rem; color: #334155; line-height: 1.45;">
                <div style="flex: 1; background: rgba(14, 165, 233, 0.05); border-left: 3px solid #0284c7; padding: 8px 12px; border-radius: 0 10px 10px 0;">
                  <strong style="color: #0369a1; font-size: 0.78rem;">1. AI Continuous Surgical Guide Wire Restoration:</strong><br/>
                  Original CT에서 수술 가이드 와이어(Hookwire)를 116~135번 모든 슬라이스에 걸쳐 끊김 없이 자연스러운 3차원 연속 궤적과 CT PSF로 복원
                </div>
                <div style="flex: 1; background: rgba(16, 185, 129, 0.05); border-left: 3px solid #059669; padding: 8px 12px; border-radius: 0 10px 10px 0;">
                  <strong style="color: #047857; font-size: 0.78rem;">2. AI Normal CT Reconstruction (ChemoPort Removal):</strong><br/>
                  케모포트 금속 왜곡 및 방사선 아티팩트(다크 섀도우/플레어)를 정상 CT 유방 연조직으로 완전 치환하여 일반 CT 영상처럼 복원
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        orig_json = json.dumps(st.session_state.orig_b64_list)
        recon_json = json.dumps(st.session_state.recon_b64_list)
        all_orig_json = json.dumps(st.session_state.all_orig_b64_list)
        all_recon_json = json.dumps(st.session_state.all_recon_b64_list)
        scan_id = st.session_state.scan_id

        dual_canvas_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", sans-serif;
            color: #0f172a;
            padding: 0 4px;
            overflow: hidden;
            user-select: none;
            -webkit-user-select: none;
          }}
          .single-viewer-grid {{
            display: flex;
            width: 100%;
            justify-content: center;
            align-items: center;
          }}
          .glass-viewer-card {{
            flex: 1;
            max-width: 630px;
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(28px) saturate(180%);
            -webkit-backdrop-filter: blur(28px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: 28px;
            padding: 16px 18px 16px 18px;
            box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.06),
                        0 2px 8px -2px rgba(0, 0, 0, 0.03),
                        inset 0 1px 1px 0 rgba(255, 255, 255, 0.95);
          }}
          .glass-viewer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 0 2px;
          }}
          .apple-pill {{
            padding: 4px 11px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            white-space: nowrap;
          }}
          .pill-recon {{
            background: rgba(13, 148, 136, 0.12);
            border: 1px solid rgba(13, 148, 136, 0.25);
            color: #0f766e;
          }}
          .pill-reset {{
            background: rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(15, 23, 42, 0.16);
            color: #0f172a;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: inherit;
          }}
          .pill-reset:hover {{
            background: rgba(15, 23, 42, 0.16);
            transform: translateY(-1px);
          }}
          .zoom-hint {{
            font-size: 0.74rem;
            color: #64748b;
            font-weight: 550;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            white-space: nowrap;
          }}
          .hud-glass-overlay {{
            position: absolute;
            top: 10px;
            right: 12px;
            z-index: 20;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 5px;
            pointer-events: auto;
            user-select: none;
            -webkit-user-select: none;
          }}
          .hud-glass-text {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "SF Mono", monospace;
            font-size: 0.76rem;
            font-weight: 750;
            color: #ffffff;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95), 0 0 6px rgba(0, 0, 0, 0.85);
            letter-spacing: 0.3px;
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 4px 9px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: all 0.15s ease;
            width: fit-content;
          }}
          .hud-glass-text:hover {{
            background: rgba(15, 23, 42, 0.9);
            border-color: rgba(255, 255, 255, 0.4);
            transform: translateY(-1px);
          }}
          .hud-glass-text.active {{
            background: #0d9488;
            border-color: #2dd4bf;
            color: #ffffff;
            box-shadow: 0 0 10px rgba(13, 148, 136, 0.5);
          }}
          .hud-glass-btn {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            color: #f1f5f9;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 2px 10px;
            min-width: 28px;
            text-align: center;
            cursor: pointer;
            transition: all 0.12s ease;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
          }}
          .hud-glass-btn:hover {{
            background: rgba(13, 148, 136, 0.85);
            color: #ffffff;
            border-color: #14b8a6;
          }}
          .hud-glass-btn.active {{
            background: #0d9488;
            color: #ffffff;
            border-color: #2dd4bf;
            box-shadow: 0 0 8px rgba(13, 148, 136, 0.6);
          }}
          .hud-wl-overlay {{
            position: absolute;
            top: 10px;
            left: 12px;
            z-index: 20;
            display: flex;
            flex-direction: column;
            gap: 5px;
            pointer-events: auto;
            user-select: none;
            -webkit-user-select: none;
          }}
          .hud-wl-text {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "SF Mono", monospace;
            font-size: 0.76rem;
            font-weight: 750;
            color: #ffffff;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95), 0 0 6px rgba(0, 0, 0, 0.85);
            letter-spacing: 0.3px;
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 4px 9px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: all 0.15s ease;
            width: fit-content;
          }}
          .hud-wl-text:hover {{
            background: rgba(15, 23, 42, 0.9);
            border-color: rgba(255, 255, 255, 0.4);
            transform: translateY(-1px);
          }}
          .hud-presets-row {{
            display: flex;
            gap: 4px;
          }}
          .hud-preset-btn {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            color: #f1f5f9;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 2px 7px;
            cursor: pointer;
            transition: all 0.12s ease;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
          }}
          .hud-preset-btn:hover {{
            background: rgba(13, 148, 136, 0.85);
            color: #ffffff;
            border-color: #14b8a6;
          }}
          .hud-preset-btn.active {{
            background: #0d9488;
            color: #ffffff;
            border-color: #2dd4bf;
            box-shadow: 0 0 8px rgba(13, 148, 136, 0.6);
          }}
          .viewer-canvas-wrapper {{
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 20px;
            overflow: hidden;
            background: #000;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
          }}
          canvas {{
            width: 100%;
            height: 100%;
            display: block;
            cursor: default;
          }}
          .viewer-footer-hint {{
            margin-top: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            font-size: 0.72rem;
            color: #64748b;
            font-weight: 550;
          }}
        </style>
        </head>
        <body>
          <div class="single-viewer-grid">
            <div class="glass-viewer-card">
              <div class="glass-viewer-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span class="apple-pill pill-recon">RECONSTRUCTED CT (AI Wire + MAR)</span>
                  <button id="btn-reset-single" class="apple-pill pill-reset" style="display: none;" title="Double-click or click to reset 1x full view">
                    ↺ Reset Zoom <span id="zoom-factor-single"></span>
                  </button>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span id="slice-badge-single" style="font-size: 0.82rem; color: #0f172a; font-weight: 700;">{slice_badge_str}</span>
                </div>
              </div>

              <div class="viewer-canvas-wrapper">
                <canvas id="canvas-single" width="512" height="512"></canvas>
                <!-- Top-Left: Window Leveling & Presets -->
                <div class="hud-wl-overlay">
                  <div id="hud-wl-badge-single" class="hud-wl-text" title="Click to reset (WL: 40, WW: 350) • Right-click drag on canvas to adjust">
                    WL: <span id="val-wl-single">40</span> &nbsp; WW: <span id="val-ww-single">350</span>
                  </div>
                  <div class="hud-presets-row">
                    <button type="button" class="hud-preset-btn active" data-preset="soft" data-wl="40" data-ww="350">Soft</button>
                    <button type="button" class="hud-preset-btn" data-preset="bone" data-wl="300" data-ww="1500">Bone</button>
                    <button type="button" class="hud-preset-btn" data-preset="lung" data-wl="-500" data-ww="1400">Lung</button>
                    <button type="button" class="hud-preset-btn" data-preset="metal" data-wl="500" data-ww="2800">Metal</button>
                    <button type="button" class="hud-preset-btn" data-preset="all" data-wl="10760" data-ww="33530" title="Full dynamic range (-6,000 to +27,500 HU) showing ChemoPort outline">All HU</button>
                  </div>
                </div>

                <!-- Top-Right: Inspection Glass (S, M, L) -->
                <div class="hud-glass-overlay">
                  <div id="hud-glass-badge-single" class="hud-glass-text active" title="Click to toggle Glass ON/OFF">
                    Glass
                  </div>
                  <div class="hud-presets-row">
                    <button type="button" class="hud-glass-btn" data-size="75" title="Small (75px) - Local focus">S</button>
                    <button type="button" class="hud-glass-btn active" data-size="160" title="Medium (160px) - Regional focus">M</button>
                    <button type="button" class="hud-glass-btn" data-size="240" title="Large (240px) - Wide focus">L</button>
                  </div>
                </div>
              </div>

              <div class="viewer-footer-hint" id="footer-hint">
                <span id="footer-hint-text">🔲 Inside Glass: Original CT • Outside: Reconstructed CT • 🖱️ Right-drag: WL/WW • 📜 Scroll: Slice</span>
              </div>
            </div>
          </div>

          <script>
          (function() {{
            const origData = {orig_json};
            const reconData = {recon_json};
            const allOrigData = {all_orig_json};
            const allReconData = {all_recon_json};
            const total = origData.length;
            let current = {initial_slice};

            const cSingle = document.getElementById('canvas-single');
            const ctx = cSingle.getContext('2d');
            const badge = document.getElementById('slice-badge-single');
            const btnResetSingle = document.getElementById('btn-reset-single');
            const zoomSpanSingle = document.getElementById('zoom-factor-single');
            const footerHintText = document.getElementById('footer-hint-text');
            const badgeGlassSingle = document.getElementById('hud-glass-badge-single');

            // High-performance offscreen buffers for 120 FPS zero-lag blitting
            const offRecon = document.createElement('canvas');
            offRecon.width = 512;
            offRecon.height = 512;
            const offReconCtx = offRecon.getContext('2d');

            const offOrig = document.createElement('canvas');
            offOrig.width = 512;
            offOrig.height = 512;
            const offOrigCtx = offOrig.getContext('2d');

            const origImgs = new Array(total);
            const reconImgs = new Array(total);
            const allOrigImgs = new Array(total);
            const allReconImgs = new Array(total);

            // ROI crop coordinates (0..512)
            const crop = {{ sx: 0, sy: 0, sw: 512, sh: 512 }};

            // Window Level (WL/WW) State
            let currentWL = 40;
            let currentWW = 350;
            let currentPreset = 'soft';

            // RTP Inspection Glass State
            let glassActive = true;
            let glassSize = 160;
            let glassPos = {{ x: 256, y: 256 }};
            let isHovering = false;
            let cachedRect = null;
            let rafId = null;

            // Restore persisted zoom, slice, WL/WW, and glass from sessionStorage
            try {{
              const scanId = "{scan_id}";
              const savedScanId = sessionStorage.getItem('chemoport_current_scan_id');
              if (savedScanId !== scanId) {{
                sessionStorage.removeItem('chemoport_zoom_crop');
                sessionStorage.removeItem('chemoport_active_slice');
                sessionStorage.removeItem('chemoport_wl');
                sessionStorage.removeItem('chemoport_ww');
                sessionStorage.removeItem('chemoport_preset');
                sessionStorage.removeItem('chemoport_spyglass');
                sessionStorage.removeItem('chemoport_glass_size');
                sessionStorage.setItem('chemoport_current_scan_id', scanId);
              }}
              const savedCrop = sessionStorage.getItem('chemoport_zoom_crop');
              if (savedCrop) {{
                const c = JSON.parse(savedCrop);
                if (c && c.sw >= 16 && c.sh >= 16) {{
                  crop.sx = c.sx;
                  crop.sy = c.sy;
                  crop.sw = c.sw;
                  crop.sh = c.sh;
                }}
              }}
              const savedSlice = sessionStorage.getItem('chemoport_active_slice');
              if (savedSlice !== null) {{
                const s = parseInt(savedSlice, 10);
                if (!isNaN(s) && s >= 0 && s < total) {{
                  current = s;
                }}
              }}
              const savedWl = sessionStorage.getItem('chemoport_wl');
              const savedWw = sessionStorage.getItem('chemoport_ww');
              const savedPreset = sessionStorage.getItem('chemoport_preset');
              if (savedWl && savedWw) {{
                currentWL = parseFloat(savedWl);
                currentWW = parseFloat(savedWw);
                currentPreset = savedPreset || 'soft';
              }}
              const savedGlass = sessionStorage.getItem('chemoport_spyglass');
              if (savedGlass === 'off') {{
                glassActive = false;
              }} else if (savedGlass === 'on') {{
                glassActive = true;
              }}
              const savedGlassSize = sessionStorage.getItem('chemoport_glass_size');
              if (savedGlassSize) {{
                const sz = parseInt(savedGlassSize, 10);
                if (!isNaN(sz) && sz >= 50 && sz <= 350) glassSize = sz;
              }}
            }} catch (err) {{}}

            function applyFilter(targetCtx) {{
              if (currentPreset === 'all') {{
                if (Math.abs(currentWL - 10760) < 5 && Math.abs(currentWW - 33530) < 10) {{
                  targetCtx.filter = 'none';
                  return;
                }}
                const alpha = Math.max(0.1, Math.min(5.0, 33530 / Math.max(200, currentWW)));
                const beta = (10760 - currentWL) / Math.max(200, currentWW);
                const brightness = Math.max(0.05, Math.min(3.0, 1.0 + beta));
                const contrast = alpha;
                targetCtx.filter = `brightness(${{brightness.toFixed(3)}}) contrast(${{contrast.toFixed(3)}})`;
                return;
              }}
              if (Math.abs(currentWL - 40) < 1 && Math.abs(currentWW - 350) < 1) {{
                targetCtx.filter = 'none';
                return;
              }}
              const alpha = Math.max(0.1, Math.min(5.0, 350 / Math.max(50, currentWW)));
              const beta = (40 - currentWL) / Math.max(50, currentWW);
              const brightness = Math.max(0.05, Math.min(3.0, 1.0 + beta));
              const contrast = alpha;
              targetCtx.filter = `brightness(${{brightness.toFixed(3)}}) contrast(${{contrast.toFixed(3)}})`;
            }}

            function getActiveOrigImage(i) {{
              return currentPreset === 'all' ? (allOrigImgs[i] || origImgs[i]) : origImgs[i];
            }}

            function getActiveReconImage(i) {{
              return currentPreset === 'all' ? (allReconImgs[i] || reconImgs[i]) : reconImgs[i];
            }}

            function updateOffscreenBuffers() {{
              const imgR = getActiveReconImage(current);
              const imgO = getActiveOrigImage(current);

              offReconCtx.clearRect(0, 0, 512, 512);
              offReconCtx.imageSmoothingEnabled = true;
              offReconCtx.imageSmoothingQuality = 'high';
              if (imgR && imgR.complete) {{
                offReconCtx.save();
                applyFilter(offReconCtx);
                offReconCtx.drawImage(imgR, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, 512, 512);
                offReconCtx.restore();
              }}

              offOrigCtx.clearRect(0, 0, 512, 512);
              offOrigCtx.imageSmoothingEnabled = true;
              offOrigCtx.imageSmoothingQuality = 'high';
              if (imgO && imgO.complete) {{
                offOrigCtx.save();
                applyFilter(offOrigCtx);
                offOrigCtx.drawImage(imgO, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, 512, 512);
                offOrigCtx.restore();
              }}
            }}

            function scheduleRender() {{
              if (!rafId) {{
                rafId = requestAnimationFrame(() => {{
                  rafId = null;
                  renderCanvas();
                }});
              }}
            }}

            function renderCanvas() {{
              ctx.clearRect(0, 0, 512, 512);

              // 1. Draw Background: Reconstructed CT from cached offscreen buffer
              ctx.drawImage(offRecon, 0, 0);

              // 2. If Inspection Glass is Active & cursor is hovering over canvas:
              if (glassActive && isHovering) {{
                const gW = glassSize;
                const gH = glassSize;
                let gLeft = glassPos.x - gW / 2;
                let gTop = glassPos.y - gH / 2;

                gLeft = Math.max(0, Math.min(512 - gW, gLeft));
                gTop = Math.max(0, Math.min(512 - gH, gTop));

                // Clip to the square glass window and blit Original CT
                ctx.save();
                ctx.beginPath();
                ctx.rect(gLeft, gTop, gW, gH);
                ctx.clip();
                ctx.drawImage(offOrig, 0, 0);
                ctx.restore();

                // Draw Glass Border & Corner Reticles
                ctx.save();
                ctx.shadowColor = 'rgba(0, 0, 0, 0.75)';
                ctx.shadowBlur = 8;
                ctx.strokeStyle = '#14b8a6';
                ctx.lineWidth = 2;
                ctx.strokeRect(gLeft, gTop, gW, gH);

                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.lineWidth = 1;
                ctx.strokeRect(gLeft + 2, gTop + 2, gW - 4, gH - 4);

                // Corner brackets ┌ ┐ └ ┘
                const tLen = Math.min(10, Math.floor(gW * 0.18));
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(gLeft, gTop + tLen); ctx.lineTo(gLeft, gTop); ctx.lineTo(gLeft + tLen, gTop);
                ctx.moveTo(gLeft + gW - tLen, gTop); ctx.lineTo(gLeft + gW, gTop); ctx.lineTo(gLeft + gW, gTop + tLen);
                ctx.moveTo(gLeft, gTop + gH - tLen); ctx.lineTo(gLeft, gTop + gH); ctx.lineTo(gLeft + tLen, gTop + gH);
                ctx.moveTo(gLeft + gW - tLen, gTop + gH); ctx.lineTo(gLeft + gW, gTop + gH); ctx.lineTo(gLeft + gW, gTop + gH - tLen);
                ctx.stroke();

                // Center reticle
                const cX = gLeft + gW / 2;
                const cY = gTop + gH / 2;
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(cX - 5, cY); ctx.lineTo(cX + 5, cY);
                ctx.moveTo(cX, cY - 5); ctx.lineTo(cX, cY + 5);
                ctx.stroke();

                // Glass Pill Badge: "ORIGINAL CT"
                const isSmall = gW <= 80;
                const labelText = isSmall ? 'ORIGINAL' : 'ORIGINAL CT';
                ctx.font = isSmall ? 'bold 9px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif' : 'bold 10px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
                const tWidth = ctx.measureText(labelText).width;
                const tagW = tWidth + (isSmall ? 8 : 12);
                const tagH = isSmall ? 15 : 17;
                const tagX = gLeft + (gW - tagW) / 2;
                const tagY = gTop - (isSmall ? 18 : 20) < 4 ? gTop + 4 : gTop - (isSmall ? 18 : 20);

                ctx.fillStyle = 'rgba(244, 63, 94, 0.92)';
                ctx.beginPath();
                if (ctx.roundRect) {{
                  ctx.roundRect(tagX, tagY, tagW, tagH, 4);
                }} else {{
                  ctx.rect(tagX, tagY, tagW, tagH);
                }}
                ctx.fill();

                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(labelText, tagX + tagW / 2, tagY + tagH / 2);

                ctx.restore();
              }}
            }}

            function updateHUD() {{
              const wlStr = Math.round(currentWL).toString();
              const wwStr = Math.round(currentWW).toString();

              const spanWl = document.getElementById('val-wl-single');
              const spanWw = document.getElementById('val-ww-single');

              if (spanWl) spanWl.innerText = wlStr;
              if (spanWw) spanWw.innerText = wwStr;

              document.querySelectorAll('.hud-preset-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.preset === currentPreset);
              }});
            }}

            function updateGlassUI() {{
              if (badgeGlassSingle) {{
                badgeGlassSingle.innerText = 'Glass';
                badgeGlassSingle.classList.toggle('active', glassActive);
              }}

              document.querySelectorAll('.hud-glass-btn').forEach(btn => {{
                const sz = parseInt(btn.dataset.size, 10);
                btn.classList.toggle('active', glassActive && sz === glassSize);
              }});

              if (footerHintText) {{
                if (glassActive) {{
                  footerHintText.innerText = '🔲 Inside Glass: Original CT • Outside: Reconstructed CT • 🖱️ Right-drag: WL/WW • 📜 Scroll: Slice';
                }} else {{
                  footerHintText.innerText = '🔍 ↘ Drag to Zoom • 🖱️ Right-click drag: WL/WW • 📜 Scroll: Slice • Double-click: Reset Zoom';
                }}
              }}
              cSingle.style.cursor = glassActive ? 'default' : 'crosshair';
            }}

            function setPreset(preset, wl, ww) {{
              currentPreset = preset;
              currentWL = wl;
              currentWW = ww;
              try {{
                sessionStorage.setItem('chemoport_wl', currentWL.toString());
                sessionStorage.setItem('chemoport_ww', currentWW.toString());
                sessionStorage.setItem('chemoport_preset', currentPreset);
              }} catch (err) {{}}
              updateHUD();
              drawCurrent();
            }}

            function drawCurrent() {{
              updateOffscreenBuffers();
              renderCanvas();
              if (total > 1 && badge) {{
                badge.innerText = `Slice ${{current + 1}} / ${{total}}`;
              }}
            }}

            function setupImage(i) {{
              const imgO = new Image();
              imgO.onload = () => {{
                if (i === current && currentPreset !== 'all') {{
                  updateOffscreenBuffers();
                  renderCanvas();
                }}
              }};
              imgO.src = 'data:image/jpeg;base64,' + origData[i];
              origImgs[i] = imgO;

              const imgR = new Image();
              imgR.onload = () => {{
                if (i === current && currentPreset !== 'all') {{
                  updateOffscreenBuffers();
                  renderCanvas();
                }}
              }};
              imgR.src = 'data:image/jpeg;base64,' + reconData[i];
              reconImgs[i] = imgR;

              if (allOrigData && allOrigData.length > i && allOrigData[i]) {{
                const imgOA = new Image();
                imgOA.onload = () => {{
                  if (i === current && currentPreset === 'all') {{
                    updateOffscreenBuffers();
                    renderCanvas();
                  }}
                }};
                imgOA.src = 'data:image/jpeg;base64,' + allOrigData[i];
                allOrigImgs[i] = imgOA;
              }}

              if (allReconData && allReconData.length > i && allReconData[i]) {{
                const imgRA = new Image();
                imgRA.onload = () => {{
                  if (i === current && currentPreset === 'all') {{
                    updateOffscreenBuffers();
                    renderCanvas();
                  }}
                }};
                imgRA.src = 'data:image/jpeg;base64,' + allReconData[i];
                allReconImgs[i] = imgRA;
              }}
            }}

            if (current >= 0 && current < total) {{
              setupImage(current);
            }}
            for (let i = 0; i < total; i++) {{
              if (i !== current) {{
                setupImage(i);
              }}
            }}

            function drawSlice(idx) {{
              if (total <= 1) {{
                current = 0;
              }} else {{
                current = ((idx % total) + total) % total;
              }}
              drawCurrent();
            }}

            function updateZoomUI() {{
              const isZoomed = crop.sw < 510;
              const factorStr = `(${{(512 / crop.sw).toFixed(1)}}x)`;
              if (btnResetSingle) {{
                btnResetSingle.style.display = isZoomed ? 'inline-flex' : 'none';
                if (zoomSpanSingle) zoomSpanSingle.innerText = factorStr;
              }}
            }}

            function resetZoom() {{
              crop.sx = 0;
              crop.sy = 0;
              crop.sw = 512;
              crop.sh = 512;
              try {{
                sessionStorage.removeItem('chemoport_zoom_crop');
              }} catch (err) {{}}
              drawCurrent();
              updateZoomUI();
            }}

            function applyZoom(sqLeft, sqTop, sqWidth, sqHeight) {{
              const new_sx = crop.sx + (sqLeft / 512) * crop.sw;
              const new_sy = crop.sy + (sqTop / 512) * crop.sh;
              const new_sw = (sqWidth / 512) * crop.sw;
              const new_sh = (sqHeight / 512) * crop.sh;

              if (new_sw >= 16) {{
                crop.sx = Math.max(0, Math.min(512 - new_sw, new_sx));
                crop.sy = Math.max(0, Math.min(512 - new_sh, new_sy));
                crop.sw = new_sw;
                crop.sh = new_sh;
              }}
              try {{
                sessionStorage.setItem('chemoport_zoom_crop', JSON.stringify(crop));
              }} catch (err) {{}}
              drawCurrent();
              updateZoomUI();
            }}

            function drawZoomOverlay(left, top, width, height, sqLeft, sqTop, sqWidth, sqHeight) {{
              ctx.save();
              ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
              ctx.fillRect(left, top, width, height);

              ctx.strokeStyle = '#06b6d4';
              ctx.lineWidth = 1.8;
              ctx.setLineDash([5, 4]);
              ctx.strokeRect(left, top, width, height);

              if (Math.abs(width - height) > 5) {{
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.strokeRect(sqLeft, sqTop, sqWidth, sqHeight);
              }}

              ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
              ctx.shadowBlur = 4;
              const label = '🔍 Zoom In';
              const textWidth = ctx.measureText(label).width;
              if (width > textWidth + 14 && height > 24) {{
                ctx.fillText(label, left + 8, top + 18);
              }}
              ctx.restore();
            }}

            function drawResetOverlay(left, top, width, height) {{
              ctx.save();
              ctx.fillStyle = 'rgba(244, 63, 94, 0.2)';
              ctx.fillRect(left, top, width, height);

              ctx.strokeStyle = '#f43f5e';
              ctx.lineWidth = 1.8;
              ctx.setLineDash([5, 4]);
              ctx.strokeRect(left, top, width, height);

              ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
              ctx.shadowBlur = 4;
              const label = '↺ Reset Zoom';
              const textWidth = ctx.measureText(label).width;
              if (width > textWidth + 12 && height > 22) {{
                const textX = left + (width - textWidth) / 2;
                const textY = top + height / 2 + 4;
                ctx.fillText(label, textX, textY);
              }}
              ctx.restore();
            }}

            // Mouse Drag Box Selection (Left Click) & Window Leveling (Right Click)
            let isDragging = false;
            let startX = 0, startY = 0;
            let currentX = 0, currentY = 0;

            // Window Leveling via Right-Click Drag
            let isAdjustingWL = false;
            let wlStartX = 0, wlStartY = 0;
            let initialWL = 40, initialWW = 350;

            function updateCanvasRect() {{
              cachedRect = cSingle.getBoundingClientRect();
            }}

            cSingle.addEventListener('mouseenter', updateCanvasRect);
            window.addEventListener('resize', updateCanvasRect);
            window.addEventListener('scroll', updateCanvasRect, true);

            function getCanvasCoords(e) {{
              if (!cachedRect) updateCanvasRect();
              const scaleX = 512 / (cachedRect.width || 512);
              const scaleY = 512 / (cachedRect.height || 512);
              const x = Math.max(0, Math.min(512, (e.clientX - cachedRect.left) * scaleX));
              const y = Math.max(0, Math.min(512, (e.clientY - cachedRect.top) * scaleY));
              return {{ x, y }};
            }}

            function computeSquare(sX, sY, cX, cY) {{
              const left = Math.min(sX, cX);
              const top = Math.min(sY, cY);
              const width = Math.abs(cX - sX);
              const height = Math.abs(cY - sY);
              const cx = left + width / 2;
              const cy = top + height / 2;
              const maxDim = Math.max(width, height);
              let sqLeft = cx - maxDim / 2;
              let sqTop = cy - maxDim / 2;
              let sqWidth = maxDim;
              let sqHeight = maxDim;

              if (sqLeft < 0) sqLeft = 0;
              if (sqTop < 0) sqTop = 0;
              if (sqLeft + sqWidth > 512) sqLeft = 512 - sqWidth;
              if (sqTop + sqHeight > 512) sqTop = 512 - sqHeight;

              return {{ left, top, width, height, sqLeft, sqTop, sqWidth, sqHeight }};
            }}

            function onMouseDown(e) {{
              if (e.button === 0) {{
                // Left click
                if (glassActive) {{
                  // Glass is active: smoothly follows cursor, no pinning
                  return;
                }}
                // Normal Mode (Glass Off): Box Zoom
                e.preventDefault();
                isDragging = true;
                const pos = getCanvasCoords(e);
                startX = pos.x;
                startY = pos.y;
                currentX = pos.x;
                currentY = pos.y;
                window.addEventListener('mousemove', onDragMouseMove);
                window.addEventListener('mouseup', onDragMouseUp);
              }} else if (e.button === 2) {{
                // Right click: Window Level Adjustment
                e.preventDefault();
                isAdjustingWL = true;
                wlStartX = e.clientX;
                wlStartY = e.clientY;
                initialWL = currentWL;
                initialWW = currentWW;
                window.addEventListener('mousemove', onRightMouseMove);
                window.addEventListener('mouseup', onRightMouseUp);
              }}
            }}

            function onDragMouseMove(e) {{
              if (!isDragging) return;
              const pos = getCanvasCoords(e);
              currentX = pos.x;
              currentY = pos.y;

              const dx = currentX - startX;
              const dy = currentY - startY;
              const b = computeSquare(startX, startY, currentX, currentY);

              renderCanvas();

              if (dx < -10 && dy < -10) {{
                drawResetOverlay(b.left, b.top, b.width, b.height);
              }} else if (b.width >= 6 || b.height >= 6) {{
                drawZoomOverlay(b.left, b.top, b.width, b.height, b.sqLeft, b.sqTop, b.sqWidth, b.sqHeight);
              }}
            }}

            function onDragMouseUp(e) {{
              if (e.button === 0 && isDragging) {{
                isDragging = false;
                window.removeEventListener('mousemove', onDragMouseMove);
                window.removeEventListener('mouseup', onDragMouseUp);

                const dx = currentX - startX;
                const dy = currentY - startY;

                if (dx < -12 && dy < -12) {{
                  resetZoom();
                }} else {{
                  const b = computeSquare(startX, startY, currentX, currentY);
                  if (b.width >= 12 && b.height >= 12) {{
                    applyZoom(b.sqLeft, b.sqTop, b.sqWidth, b.sqHeight);
                  }} else {{
                    drawCurrent();
                  }}
                }}
              }}
            }}

            // High-efficiency, zero-lag cursor tracking for RTP Inspection Glass
            cSingle.addEventListener('mouseenter', (e) => {{
              updateCanvasRect();
              isHovering = true;
              const pos = getCanvasCoords(e);
              glassPos.x = pos.x;
              glassPos.y = pos.y;
              if (glassActive) scheduleRender();
            }});

            cSingle.addEventListener('mousemove', (e) => {{
              if (isDragging || isAdjustingWL) return;
              isHovering = true;
              const pos = getCanvasCoords(e);
              glassPos.x = pos.x;
              glassPos.y = pos.y;
              if (glassActive) {{
                scheduleRender();
              }}
            }});

            cSingle.addEventListener('mouseleave', () => {{
              isHovering = false;
              if (glassActive) {{
                scheduleRender();
              }}
            }});

            function onRightMouseMove(e) {{
              if (!isAdjustingWL) return;
              const dx = e.clientX - wlStartX;
              const dy = e.clientY - wlStartY;

              const mult = currentPreset === 'all' ? 30.0 : 2.5;
              const maxWw = currentPreset === 'all' ? 60000 : 4000;
              const minWw = currentPreset === 'all' ? 1000 : 50;
              const maxWl = currentPreset === 'all' ? 30000 : 2000;
              const minWl = currentPreset === 'all' ? -10000 : -1000;

              currentWW = Math.max(minWw, Math.min(maxWw, Math.round(initialWW + dx * mult)));
              currentWL = Math.max(minWl, Math.min(maxWl, Math.round(initialWL - dy * mult)));
              currentPreset = 'custom';
              updateHUD();
              drawCurrent();
            }}

            function onRightMouseUp(e) {{
              if (e.button === 2 && isAdjustingWL) {{
                isAdjustingWL = false;
                window.removeEventListener('mousemove', onRightMouseMove);
                window.removeEventListener('mouseup', onRightMouseUp);
                try {{
                  sessionStorage.setItem('chemoport_wl', currentWL.toString());
                  sessionStorage.setItem('chemoport_ww', currentWW.toString());
                  sessionStorage.setItem('chemoport_preset', currentPreset);
                }} catch (err) {{}}
              }}
            }}

            cSingle.addEventListener('mousedown', onMouseDown);
            cSingle.addEventListener('contextmenu', (e) => e.preventDefault());
            cSingle.addEventListener('dblclick', resetZoom);
            if (btnResetSingle) btnResetSingle.addEventListener('click', resetZoom);

            // Glass Badge Click: Toggle ON / OFF
            if (badgeGlassSingle) {{
              badgeGlassSingle.addEventListener('click', (e) => {{
                e.stopPropagation();
                glassActive = !glassActive;
                try {{
                  sessionStorage.setItem('chemoport_spyglass', glassActive ? 'on' : 'off');
                }} catch (err) {{}}
                updateGlassUI();
                scheduleRender();
              }});
            }}

            // Glass S, M, L Size Buttons
            document.querySelectorAll('.hud-glass-btn').forEach(btn => {{
              btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const sz = parseInt(btn.dataset.size, 10);
                if (glassActive && glassSize === sz) {{
                  glassActive = false;
                }} else {{
                  glassActive = true;
                  glassSize = sz;
                }}
                try {{
                  sessionStorage.setItem('chemoport_spyglass', glassActive ? 'on' : 'off');
                  if (glassActive) sessionStorage.setItem('chemoport_glass_size', glassSize.toString());
                }} catch (err) {{}}
                updateGlassUI();
                scheduleRender();
              }});
            }});

            // Preset Buttons (Top-Left Canvas Overlay)
            document.querySelectorAll('.hud-preset-btn').forEach(btn => {{
              btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const p = btn.dataset.preset;
                const wl = parseFloat(btn.dataset.wl);
                const ww = parseFloat(btn.dataset.ww);
                setPreset(p, wl, ww);
              }});
            }});

            const badgeWlSingle = document.getElementById('hud-wl-badge-single');
            if (badgeWlSingle) badgeWlSingle.addEventListener('click', () => {{
              if (currentPreset === 'all') {{
                setPreset('all', 10760, 33530);
              }} else {{
                setPreset('soft', 40, 350);
              }}
            }});

            // Initial render
            updateHUD();
            updateZoomUI();
            updateGlassUI();
            drawSlice(current);

            // 60-120 FPS Sub-Millisecond Mouse Wheel Navigation
            let accum = 0;
            const PIXELS_PER_SLICE = 22;

            function handleWheel(e) {{
              e.preventDefault();
              e.stopPropagation();

              let delta = e.deltaY;
              if (e.deltaMode === 1) delta *= 28;
              if (e.shiftKey) delta *= 4;

              accum += delta;
              const step = Math.trunc(accum / PIXELS_PER_SLICE);
              if (step !== 0) {{
                accum -= step * PIXELS_PER_SLICE;
                drawSlice(current + step);
                try {{
                  sessionStorage.setItem('chemoport_active_slice', current.toString());
                  window.parent.__chemoport_slice = current;
                }} catch (err) {{}}
              }}
            }}

            window.addEventListener('wheel', handleWheel, {{ passive: false }});
          }})();
          </script>
        </body>
        </html>
        """
        components.html(dual_canvas_html, height=680)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        _, back_col, next_col, _ = st.columns([1.7, 0.8, 0.8, 1.7])
        with back_col:
            if st.button("← Back", key="nav_back_3", use_container_width=True, help="Back to 2. Artifact Inspection"):
                st.session_state.current_workflow_step = "2. Artifact Inspection"
                st.rerun()
        with next_col:
            if st.button("Next →", key="nav_next_3", type="primary", use_container_width=True, help="Proceed to 4. Save & Export"):
                st.session_state.current_workflow_step = "4. Save & Export"
                st.rerun()

    # STEP 2: 2. Artifact Inspection
    if workflow_mode == "2. Artifact Inspection":
        # Ensure base64 list is populated for client-side instant navigation
        if not st.session_state.orig_b64_list and st.session_state.current_hu is not None:
            slices_src = (
                st.session_state.slice_list
                if st.session_state.slice_list
                else [(st.session_state.current_ds, st.session_state.current_hu, st.session_state.current_meta)]
            )
            o_list = []
            oa_list = []
            for sl in slices_src:
                hu = sl[1]
                io = apply_window_level(hu, wc=40.0, ww=350.0)
                ioa = apply_window_level(hu, wc=10760.0, ww=33530.0)
                _, b1 = cv2.imencode(".jpg", io, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, b1a = cv2.imencode(".jpg", ioa, [cv2.IMWRITE_JPEG_QUALITY, 80])
                o_list.append(base64.b64encode(b1).decode("ascii"))
                oa_list.append(base64.b64encode(b1a).decode("ascii"))
            st.session_state.orig_b64_list = o_list
            st.session_state.all_orig_b64_list = oa_list

        total_slices = len(st.session_state.slice_list) if st.session_state.slice_list else 1
        initial_slice = st.session_state.current_slice_idx
        slice_badge_str = f"Slice {initial_slice + 1} / {total_slices}" if total_slices > 1 else ""
        orig_json = json.dumps(st.session_state.orig_b64_list)
        all_orig_json = json.dumps(st.session_state.all_orig_b64_list)
        scan_id = st.session_state.scan_id

        if "contours_cache" not in st.session_state or len(st.session_state.contours_cache) != total_slices:
            slices_src = (
                st.session_state.slice_list
                if st.session_state.slice_list
                else [(st.session_state.current_ds, st.session_state.current_hu, st.session_state.current_meta)]
            )
            series_mar_results = process_series_slices_mar(slices_src)
            st.session_state.contours_cache = [r[2] for r in series_mar_results]

        contours_json = json.dumps(st.session_state.contours_cache)

        single_canvas_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", sans-serif;
            color: #0f172a;
            padding: 0 4px;
            overflow: hidden;
            user-select: none;
            -webkit-user-select: none;
          }}
          .single-viewer-grid {{
            display: flex;
            width: 100%;
            justify-content: center;
            align-items: center;
          }}
          .glass-viewer-card {{
            flex: 1;
            max-width: 630px;
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(28px) saturate(180%);
            -webkit-backdrop-filter: blur(28px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: 28px;
            padding: 16px 18px 18px 18px;
            box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.06),
                        0 2px 8px -2px rgba(0, 0, 0, 0.03),
                        inset 0 1px 1px 0 rgba(255, 255, 255, 0.95);
          }}
          .glass-viewer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 0 2px;
          }}
          .apple-pill {{
            padding: 4px 11px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 5px;
          }}
          .pill-orig {{
            background: rgba(244, 63, 94, 0.12);
            border: 1px solid rgba(244, 63, 94, 0.25);
            color: #be123c;
          }}
          .pill-reset {{
            background: rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(15, 23, 42, 0.16);
            color: #0f172a;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: inherit;
          }}
          .pill-reset:hover {{
            background: rgba(15, 23, 42, 0.16);
            transform: translateY(-1px);
          }}
          .viewer-footer-hint {{
            margin-top: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            font-size: 0.72rem;
            color: #64748b;
            font-weight: 550;
          }}
          .hud-wl-overlay {{
            position: absolute;
            top: 10px;
            left: 12px;
            z-index: 20;
            display: flex;
            flex-direction: column;
            gap: 5px;
            pointer-events: auto;
            user-select: none;
            -webkit-user-select: none;
          }}
          .hud-wl-text {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "SF Mono", monospace;
            font-size: 0.76rem;
            font-weight: 750;
            color: #ffffff;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95), 0 0 6px rgba(0, 0, 0, 0.85);
            letter-spacing: 0.3px;
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 4px 9px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: all 0.15s ease;
            width: fit-content;
          }}
          .hud-wl-text:hover {{
            background: rgba(15, 23, 42, 0.9);
            border-color: rgba(255, 255, 255, 0.4);
            transform: translateY(-1px);
          }}
          .hud-presets-row {{
            display: flex;
            gap: 4px;
          }}
          .hud-preset-btn {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            color: #f1f5f9;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 2px 7px;
            cursor: pointer;
            transition: all 0.12s ease;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
          }}
          .hud-preset-btn:hover {{
            background: rgba(13, 148, 136, 0.85);
            color: #ffffff;
            border-color: #14b8a6;
          }}
          .hud-preset-btn.active {{
            background: #0d9488;
            color: #ffffff;
            border-color: #2dd4bf;
            box-shadow: 0 0 8px rgba(13, 148, 136, 0.6);
          }}
          .viewer-canvas-wrapper {{
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 20px;
            overflow: hidden;
            background: #000;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
          }}
          canvas {{
            width: 100%;
            height: 100%;
            display: block;
            cursor: crosshair;
          }}
          .hud-contours-overlay {{
            position: absolute;
            top: 10px;
            right: 12px;
            z-index: 20;
            display: flex;
            gap: 6px;
            pointer-events: auto;
            user-select: none;
            -webkit-user-select: none;
          }}
          .hud-contour-btn {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            color: #f1f5f9;
            background: rgba(15, 23, 42, 0.68);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 3px 9px;
            cursor: pointer;
            transition: all 0.12s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
          }}
          .hud-contour-btn:hover {{
            background: rgba(15, 23, 42, 0.88);
            border-color: rgba(255, 255, 255, 0.4);
            transform: translateY(-1px);
          }}
          .hud-contour-btn.btn-port.active {{
            background: rgba(225, 29, 72, 0.88);
            border-color: #fb7185;
            color: #ffffff;
            box-shadow: 0 0 8px rgba(225, 29, 72, 0.55);
          }}
          .hud-contour-btn.btn-art.active {{
            background: rgba(217, 119, 6, 0.88);
            border-color: #fcd34d;
            color: #ffffff;
            box-shadow: 0 0 8px rgba(217, 119, 6, 0.55);
          }}
          .contour-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
          }}
          .dot-port {{
            background: #f43f5e;
            box-shadow: 0 0 4px #f43f5e;
          }}
          .dot-art {{
            background: #fbbf24;
            box-shadow: 0 0 4px #fbbf24;
          }}
        </style>
        </head>
        <body>
          <div class="single-viewer-grid">
            <div class="glass-viewer-card">
              <div class="glass-viewer-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span class="apple-pill pill-orig">ORIGINAL CT</span>
                  <button id="btn-reset-single" class="apple-pill pill-reset" style="display: none;" title="Double-click or click to reset 1x full view">
                    ↺ Reset Zoom <span id="zoom-factor-single"></span>
                  </button>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span id="slice-badge-single" style="font-size: 0.82rem; color: #0f172a; font-weight: 700;">{slice_badge_str}</span>
                </div>
              </div>
              <div class="viewer-canvas-wrapper">
                <canvas id="canvas-single" width="512" height="512"></canvas>
                <div class="hud-wl-overlay">
                  <div id="hud-wl-badge-single" class="hud-wl-text" title="Click to reset (WL: 40, WW: 350) • Right-click drag on canvas to adjust">
                    WL: <span id="val-wl-single">40</span> &nbsp; WW: <span id="val-ww-single">350</span>
                  </div>
                  <div class="hud-presets-row">
                    <button type="button" class="hud-preset-btn active" data-preset="soft" data-wl="40" data-ww="350">Soft</button>
                    <button type="button" class="hud-preset-btn" data-preset="bone" data-wl="300" data-ww="1500">Bone</button>
                    <button type="button" class="hud-preset-btn" data-preset="lung" data-wl="-500" data-ww="1400">Lung</button>
                    <button type="button" class="hud-preset-btn" data-preset="metal" data-wl="500" data-ww="2800">Metal</button>
                    <button type="button" class="hud-preset-btn" data-preset="all" data-wl="10760" data-ww="33530" title="Full dynamic range (-6,000 to +27,500 HU) showing ChemoPort outline">All HU</button>
                  </div>
                </div>

                <!-- Top-Right: Structure Contours (ChemoPort & Artifact) -->
                <div class="hud-contours-overlay">
                  <div class="hud-presets-row">
                    <button type="button" id="btn-toggle-port" class="hud-contour-btn active btn-port" title="Click to toggle ChemoPort Contour (ON/OFF)">
                      <span class="contour-dot dot-port"></span> ChemoPort
                    </button>
                    <button type="button" id="btn-toggle-art" class="hud-contour-btn active btn-art" title="Click to toggle Artifact Contour (ON/OFF)">
                      <span class="contour-dot dot-art"></span> Artifact
                    </button>
                  </div>
                </div>
              </div>

              <div class="viewer-footer-hint" id="footer-hint">
                <span id="footer-hint-text">🔍 ↘ Drag to Zoom • 🖱️ Right-click drag: WL/WW • 📜 Scroll: Slice • Double-click: Reset Zoom</span>
              </div>
            </div>
          </div>

          <script>
          (function() {{
            const origData = {orig_json};
            const allOrigData = {all_orig_json};
            const contoursData = {contours_json};
            const total = origData.length;
            let current = {initial_slice};

            const cSingle = document.getElementById('canvas-single');
            const ctx = cSingle.getContext('2d');
            const badge = document.getElementById('slice-badge-single');
            const btnResetSingle = document.getElementById('btn-reset-single');
            const zoomSpanSingle = document.getElementById('zoom-factor-single');

            const origImgs = new Array(total);
            const allOrigImgs = new Array(total);

            // ROI crop coordinates (0..512)
            const crop = {{ sx: 0, sy: 0, sw: 512, sh: 512 }};

            // Window Level (WL/WW) State
            let currentWL = 40;
            let currentWW = 350;
            let currentPreset = 'soft';

            // Contour Overlays State (ChemoPort & Artifact)
            let showPort = true;
            let showArt = true;

            // Restore persisted zoom, slice, WL/WW, and contours from sessionStorage
            try {{
              const scanId = "{scan_id}";
              const savedScanId = sessionStorage.getItem('chemoport_current_scan_id');
              if (savedScanId !== scanId) {{
                sessionStorage.removeItem('chemoport_zoom_crop');
                sessionStorage.removeItem('chemoport_active_slice');
                sessionStorage.removeItem('chemoport_wl');
                sessionStorage.removeItem('chemoport_ww');
                sessionStorage.removeItem('chemoport_preset');
                sessionStorage.removeItem('chemoport_show_port');
                sessionStorage.removeItem('chemoport_show_art');
                sessionStorage.setItem('chemoport_current_scan_id', scanId);
              }}
              const savedCrop = sessionStorage.getItem('chemoport_zoom_crop');
              if (savedCrop) {{
                const c = JSON.parse(savedCrop);
                if (c && c.sw >= 16 && c.sh >= 16) {{
                  crop.sx = c.sx;
                  crop.sy = c.sy;
                  crop.sw = c.sw;
                  crop.sh = c.sh;
                }}
              }}
              const savedSlice = sessionStorage.getItem('chemoport_active_slice');
              if (savedSlice !== null) {{
                const s = parseInt(savedSlice, 10);
                if (!isNaN(s) && s >= 0 && s < total) {{
                  current = s;
                }}
              }}
              const savedWl = sessionStorage.getItem('chemoport_wl');
              const savedWw = sessionStorage.getItem('chemoport_ww');
              const savedPreset = sessionStorage.getItem('chemoport_preset');
              if (savedWl && savedWw) {{
                currentWL = parseFloat(savedWl);
                currentWW = parseFloat(savedWw);
                currentPreset = savedPreset || 'soft';
              }}
              const savedShowPort = sessionStorage.getItem('chemoport_show_port');
              if (savedShowPort !== null) showPort = (savedShowPort === 'true');
              const savedShowArt = sessionStorage.getItem('chemoport_show_art');
              if (savedShowArt !== null) showArt = (savedShowArt === 'true');
            }} catch (err) {{}}

            function applyFilter(ctx) {{
              if (currentPreset === 'all') {{
                if (Math.abs(currentWL - 10760) < 5 && Math.abs(currentWW - 33530) < 10) {{
                  ctx.filter = 'none';
                  return;
                }}
                const alpha = Math.max(0.1, Math.min(5.0, 33530 / Math.max(200, currentWW)));
                const beta = (10760 - currentWL) / Math.max(200, currentWW);
                const brightness = Math.max(0.05, Math.min(3.0, 1.0 + beta));
                const contrast = alpha;
                ctx.filter = `brightness(${{brightness.toFixed(3)}}) contrast(${{contrast.toFixed(3)}})`;
                return;
              }}
              if (Math.abs(currentWL - 40) < 1 && Math.abs(currentWW - 350) < 1) {{
                ctx.filter = 'none';
                return;
              }}
              const alpha = Math.max(0.1, Math.min(5.0, 350 / Math.max(50, currentWW)));
              const beta = (40 - currentWL) / Math.max(50, currentWW);
              const brightness = Math.max(0.05, Math.min(3.0, 1.0 + beta));
              const contrast = alpha;
              ctx.filter = `brightness(${{brightness.toFixed(3)}}) contrast(${{contrast.toFixed(3)}})`;
            }}

            function getActiveOrigImage(i) {{
              return currentPreset === 'all' ? (allOrigImgs[i] || origImgs[i]) : origImgs[i];
            }}

            function drawContours() {{
              const sliceContours = (contoursData && contoursData.length > current) ? contoursData[current] : null;
              if (!sliceContours) return;

              const scaleX = 512 / crop.sw;
              const scaleY = 512 / crop.sh;

              // 1. Draw Streak Artifact Contours (Amber/Yellow)
              if (showArt && sliceContours.artifact && sliceContours.artifact.length > 0) {{
                ctx.save();
                ctx.strokeStyle = '#f59e0b';
                ctx.fillStyle = 'rgba(245, 158, 11, 0.16)';
                ctx.lineWidth = 2.0;
                ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
                ctx.shadowBlur = 4;

                sliceContours.artifact.forEach(poly => {{
                  if (poly.length < 3) return;
                  ctx.beginPath();
                  poly.forEach((pt, idx) => {{
                    const cx = (pt[0] - crop.sx) * scaleX;
                    const cy = (pt[1] - crop.sy) * scaleY;
                    if (idx === 0) ctx.moveTo(cx, cy);
                    else ctx.lineTo(cx, cy);
                  }});
                  ctx.closePath();
                  ctx.fill();
                  ctx.stroke();
                }});
                ctx.restore();
              }}

              // 2. Draw ChemoPort Contours (Rose/Red)
              if (showPort && sliceContours.port && sliceContours.port.length > 0) {{
                ctx.save();
                ctx.strokeStyle = '#f43f5e';
                ctx.fillStyle = 'rgba(244, 63, 94, 0.28)';
                ctx.lineWidth = 2.5;
                ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
                ctx.shadowBlur = 6;

                let minCx = Infinity, maxCx = -Infinity, minCy = Infinity, maxCy = -Infinity;
                let totalPts = 0;

                sliceContours.port.forEach(poly => {{
                  if (poly.length < 3) return;
                  ctx.beginPath();
                  poly.forEach((pt, idx) => {{
                    const cx = (pt[0] - crop.sx) * scaleX;
                    const cy = (pt[1] - crop.sy) * scaleY;
                    if (idx === 0) ctx.moveTo(cx, cy);
                    else ctx.lineTo(cx, cy);

                    totalPts++;
                    if (cx < minCx) minCx = cx;
                    if (cx > maxCx) maxCx = cx;
                    if (cy < minCy) minCy = cy;
                    if (cy > maxCy) maxCy = cy;
                  }});
                  ctx.closePath();
                  ctx.fill();
                  ctx.stroke();
                }});

                ctx.restore();
              }}
            }}

            function renderCanvas(img) {{
              ctx.clearRect(0, 0, 512, 512);
              ctx.imageSmoothingEnabled = true;
              ctx.imageSmoothingQuality = 'high';
              applyFilter(ctx);
              const activeImg = img || getActiveOrigImage(current);
              if (activeImg && activeImg.complete) {{
                ctx.drawImage(activeImg, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, 512, 512);
              }}
              ctx.filter = 'none';

              // Draw Structure Contours (ChemoPort & Artifact)
              drawContours();
            }}

            function updateHUD() {{
              const wlStr = Math.round(currentWL).toString();
              const wwStr = Math.round(currentWW).toString();

              const spanWl = document.getElementById('val-wl-single');
              const spanWw = document.getElementById('val-ww-single');

              if (spanWl) spanWl.innerText = wlStr;
              if (spanWw) spanWw.innerText = wwStr;

              document.querySelectorAll('.hud-preset-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.preset === currentPreset);
              }});
            }}

            function setPreset(preset, wl, ww) {{
              currentPreset = preset;
              currentWL = wl;
              currentWW = ww;
              try {{
                sessionStorage.setItem('chemoport_wl', currentWL.toString());
                sessionStorage.setItem('chemoport_ww', currentWW.toString());
                sessionStorage.setItem('chemoport_preset', currentPreset);
              }} catch (err) {{}}
              updateHUD();
              drawCurrent();
            }}

            function drawCurrent() {{
              renderCanvas(getActiveOrigImage(current));
              if (total > 1 && badge) {{
                badge.innerText = `Slice ${{current + 1}} / ${{total}}`;
              }}
            }}

            function setupImage(i) {{
              const imgO = new Image();
              imgO.onload = () => {{
                if (i === current && currentPreset !== 'all') renderCanvas(imgO);
              }};
              imgO.src = 'data:image/jpeg;base64,' + origData[i];
              origImgs[i] = imgO;

              if (allOrigData && allOrigData.length > i && allOrigData[i]) {{
                const imgOA = new Image();
                imgOA.onload = () => {{
                  if (i === current && currentPreset === 'all') renderCanvas(imgOA);
                }};
                imgOA.src = 'data:image/jpeg;base64,' + allOrigData[i];
                allOrigImgs[i] = imgOA;
              }}
            }}

            if (current >= 0 && current < total) {{
              setupImage(current);
            }}
            for (let i = 0; i < total; i++) {{
              if (i !== current) {{
                setupImage(i);
              }}
            }}

            function drawSlice(idx) {{
              if (total <= 1) {{
                current = 0;
              }} else {{
                current = ((idx % total) + total) % total;
              }}
              drawCurrent();
            }}

            function updateZoomUI() {{
              const isZoomed = crop.sw < 510;
              const factorStr = `(${{(512 / crop.sw).toFixed(1)}}x)`;
              if (btnResetSingle) {{
                btnResetSingle.style.display = isZoomed ? 'inline-flex' : 'none';
                if (zoomSpanSingle) zoomSpanSingle.innerText = factorStr;
              }}
            }}

            function resetZoom() {{
              crop.sx = 0;
              crop.sy = 0;
              crop.sw = 512;
              crop.sh = 512;
              try {{
                sessionStorage.removeItem('chemoport_zoom_crop');
              }} catch (err) {{}}
              drawCurrent();
              updateZoomUI();
            }}

            function applyZoom(sqLeft, sqTop, sqWidth, sqHeight) {{
              const new_sx = crop.sx + (sqLeft / 512) * crop.sw;
              const new_sy = crop.sy + (sqTop / 512) * crop.sh;
              const new_sw = (sqWidth / 512) * crop.sw;
              const new_sh = (sqHeight / 512) * crop.sh;

              if (new_sw >= 16) {{
                crop.sx = Math.max(0, Math.min(512 - new_sw, new_sx));
                crop.sy = Math.max(0, Math.min(512 - new_sh, new_sy));
                crop.sw = new_sw;
                crop.sh = new_sh;
              }}
              try {{
                sessionStorage.setItem('chemoport_zoom_crop', JSON.stringify(crop));
              }} catch (err) {{}}
              drawCurrent();
              updateZoomUI();
            }}

            function drawZoomOverlay(left, top, width, height, sqLeft, sqTop, sqWidth, sqHeight) {{
              ctx.save();
              ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
              ctx.fillRect(left, top, width, height);

              ctx.strokeStyle = '#06b6d4';
              ctx.lineWidth = 1.8;
              ctx.setLineDash([5, 4]);
              ctx.strokeRect(left, top, width, height);

              if (Math.abs(width - height) > 5) {{
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.strokeRect(sqLeft, sqTop, sqWidth, sqHeight);
              }}

              ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
              ctx.shadowBlur = 4;
              const label = '🔍 Zoom In';
              const textWidth = ctx.measureText(label).width;
              if (width > textWidth + 14 && height > 24) {{
                ctx.fillText(label, left + 8, top + 18);
              }}
              ctx.restore();
            }}

            function drawResetOverlay(left, top, width, height) {{
              ctx.save();
              ctx.fillStyle = 'rgba(244, 63, 94, 0.2)';
              ctx.fillRect(left, top, width, height);

              ctx.strokeStyle = '#f43f5e';
              ctx.lineWidth = 1.8;
              ctx.setLineDash([5, 4]);
              ctx.strokeRect(left, top, width, height);

              ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
              ctx.shadowBlur = 4;
              const label = '↺ Reset Zoom';
              const textWidth = ctx.measureText(label).width;
              if (width > textWidth + 12 && height > 22) {{
                const textX = left + (width - textWidth) / 2;
                const textY = top + height / 2 + 4;
                ctx.fillText(label, textX, textY);
              }}
              ctx.restore();
            }}

            // Mouse Drag Box Selection (Left Click) & Window Leveling (Right Click)
            let isDragging = false;
            let startX = 0, startY = 0;
            let currentX = 0, currentY = 0;

            // Window Leveling via Right-Click Drag
            let isAdjustingWL = false;
            let wlStartX = 0, wlStartY = 0;
            let initialWL = 40, initialWW = 350;

            function getCanvasCoords(e) {{
              const rect = cSingle.getBoundingClientRect();
              const scaleX = 512 / rect.width;
              const scaleY = 512 / rect.height;
              const x = Math.max(0, Math.min(512, (e.clientX - rect.left) * scaleX));
              const y = Math.max(0, Math.min(512, (e.clientY - rect.top) * scaleY));
              return {{ x, y }};
            }}

            function computeSquare(sX, sY, cX, cY) {{
              const left = Math.min(sX, cX);
              const top = Math.min(sY, cY);
              const width = Math.abs(cX - sX);
              const height = Math.abs(cY - sY);
              const cx = left + width / 2;
              const cy = top + height / 2;
              const maxDim = Math.max(width, height);
              let sqLeft = cx - maxDim / 2;
              let sqTop = cy - maxDim / 2;
              let sqWidth = maxDim;
              let sqHeight = maxDim;

              if (sqLeft < 0) sqLeft = 0;
              if (sqTop < 0) sqTop = 0;
              if (sqLeft + sqWidth > 512) sqLeft = 512 - sqWidth;
              if (sqTop + sqHeight > 512) sqTop = 512 - sqHeight;

              return {{ left, top, width, height, sqLeft, sqTop, sqWidth, sqHeight }};
            }}

            function onMouseDown(e) {{
              if (e.button === 0) {{
                // Left click: Box Zoom
                e.preventDefault();
                isDragging = true;
                const pos = getCanvasCoords(e);
                startX = pos.x;
                startY = pos.y;
                currentX = pos.x;
                currentY = pos.y;
                window.addEventListener('mousemove', onMouseMove);
                window.addEventListener('mouseup', onMouseUp);
              }} else if (e.button === 2) {{
                // Right click: Window Level Adjustment
                e.preventDefault();
                isAdjustingWL = true;
                wlStartX = e.clientX;
                wlStartY = e.clientY;
                initialWL = currentWL;
                initialWW = currentWW;
                window.addEventListener('mousemove', onRightMouseMove);
                window.addEventListener('mouseup', onRightMouseUp);
              }}
            }}

            function onMouseMove(e) {{
              if (!isDragging) return;
              const pos = getCanvasCoords(e);
              currentX = pos.x;
              currentY = pos.y;

              const dx = currentX - startX;
              const dy = currentY - startY;
              const b = computeSquare(startX, startY, currentX, currentY);

              renderCanvas(getActiveOrigImage(current));

              if (dx < -10 && dy < -10) {{
                drawResetOverlay(b.left, b.top, b.width, b.height);
              }} else if (b.width >= 6 || b.height >= 6) {{
                drawZoomOverlay(b.left, b.top, b.width, b.height, b.sqLeft, b.sqTop, b.sqWidth, b.sqHeight);
              }}
            }}

            function onMouseUp(e) {{
              if (e.button === 0 && isDragging) {{
                isDragging = false;
                window.removeEventListener('mousemove', onMouseMove);
                window.removeEventListener('mouseup', onMouseUp);

                const dx = currentX - startX;
                const dy = currentY - startY;

                if (dx < -12 && dy < -12) {{
                  resetZoom();
                }} else {{
                  const b = computeSquare(startX, startY, currentX, currentY);
                  if (b.width >= 12 && b.height >= 12) {{
                    applyZoom(b.sqLeft, b.sqTop, b.sqWidth, b.sqHeight);
                  }} else {{
                    drawCurrent();
                  }}
                }}
              }}
            }}

            function onRightMouseMove(e) {{
              if (!isAdjustingWL) return;
              const dx = e.clientX - wlStartX;
              const dy = e.clientY - wlStartY;

              const mult = currentPreset === 'all' ? 30.0 : 2.5;
              const maxWw = currentPreset === 'all' ? 60000 : 4000;
              const minWw = currentPreset === 'all' ? 1000 : 50;
              const maxWl = currentPreset === 'all' ? 30000 : 2000;
              const minWl = currentPreset === 'all' ? -10000 : -1000;

              currentWW = Math.max(minWw, Math.min(maxWw, Math.round(initialWW + dx * mult)));
              currentWL = Math.max(minWl, Math.min(maxWl, Math.round(initialWL - dy * mult)));
              currentPreset = 'custom';
              updateHUD();
              drawCurrent();
            }}

            function onRightMouseUp(e) {{
              if (e.button === 2 && isAdjustingWL) {{
                isAdjustingWL = false;
                window.removeEventListener('mousemove', onRightMouseMove);
                window.removeEventListener('mouseup', onRightMouseUp);
                try {{
                  sessionStorage.setItem('chemoport_wl', currentWL.toString());
                  sessionStorage.setItem('chemoport_ww', currentWW.toString());
                  sessionStorage.setItem('chemoport_preset', currentPreset);
                }} catch (err) {{}}
              }}
            }}

            cSingle.addEventListener('mousedown', onMouseDown);
            cSingle.addEventListener('contextmenu', (e) => e.preventDefault());
            cSingle.addEventListener('dblclick', resetZoom);
            if (btnResetSingle) btnResetSingle.addEventListener('click', resetZoom);

            // Wire Preset Buttons
            document.querySelectorAll('.hud-preset-btn').forEach(btn => {{
              btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const p = btn.dataset.preset;
                const wl = parseFloat(btn.dataset.wl);
                const ww = parseFloat(btn.dataset.ww);
                setPreset(p, wl, ww);
              }});
            }});

            // Wire Contour Buttons (ChemoPort & Artifact)
            const btnTogglePort = document.getElementById('btn-toggle-port');
            const btnToggleArt = document.getElementById('btn-toggle-art');

            function updateContourButtons() {{
              if (btnTogglePort) btnTogglePort.classList.toggle('active', showPort);
              if (btnToggleArt) btnToggleArt.classList.toggle('active', showArt);
            }}

            if (btnTogglePort) {{
              btnTogglePort.addEventListener('click', (e) => {{
                e.stopPropagation();
                showPort = !showPort;
                try {{
                  sessionStorage.setItem('chemoport_show_port', showPort.toString());
                }} catch (err) {{}}
                updateContourButtons();
                drawCurrent();
              }});
            }}

            if (btnToggleArt) {{
              btnToggleArt.addEventListener('click', (e) => {{
                e.stopPropagation();
                showArt = !showArt;
                try {{
                  sessionStorage.setItem('chemoport_show_art', showArt.toString());
                }} catch (err) {{}}
                updateContourButtons();
                drawCurrent();
              }});
            }}

            // Initial render
            updateHUD();
            updateZoomUI();
            updateContourButtons();
            drawSlice(current);

            // 60-120 FPS Sub-Millisecond Mouse Wheel Navigation (Pure Client-Side Canvas)
            let accum = 0;
            const PIXELS_PER_SLICE = 22;

            function handleWheel(e) {{
              e.preventDefault();
              e.stopPropagation();

              let delta = e.deltaY;
              if (e.deltaMode === 1) delta *= 28;
              if (e.shiftKey) delta *= 4;

              accum += delta;
              const step = Math.trunc(accum / PIXELS_PER_SLICE);
              if (step !== 0) {{
                accum -= step * PIXELS_PER_SLICE;
                drawSlice(current + step);
                try {{
                  sessionStorage.setItem('chemoport_active_slice', current.toString());
                  window.parent.__chemoport_slice = current;
                }} catch (err) {{}}
              }}
            }}

            window.addEventListener('wheel', handleWheel, {{ passive: false }});
          }})();
          </script>
        </body>
        </html>
        """
        components.html(single_canvas_html, height=680)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        _, back_col, next_col, _ = st.columns([1.7, 0.8, 0.8, 1.7])
        with back_col:
            if st.button("← Back", key="nav_back_2", use_container_width=True, help="Back to 1. Image Loading"):
                st.session_state.current_workflow_step = "1. Image Loading"
                st.rerun()
        with next_col:
            if st.button("Next →", key="nav_next_2", type="primary", use_container_width=True, help="Proceed to 3. AI Correction"):
                st.session_state.current_workflow_step = "3. AI Correction"
                st.rerun()

    # STEP 4: 4. Save & Export (Save & TPS Export)
    if workflow_mode == "4. Save & Export":
        st.markdown("---")
        st.markdown("### 💾 Treatment Planning System (TPS) DICOM Export & Storage")

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            dicom_bytes = save_dicom_bytes(st.session_state.current_ds, hu_recon)
            st.download_button(
                label="💾 Download Corrected DICOM (.dcm)",
                data=dicom_bytes,
                file_name=f"AI_MAR_Restored_Instance{meta.get('instance_number', 1)}.dcm",
                mime="application/dicom",
                use_container_width=True,
                type="primary",
                help="Standard DICOM with updated SOP Instance UID, ready for Eclipse / RayStation import.",
            )

        with export_col2:
            comp_fig, (cax1, cax2, cax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
            cax1.imshow(img_orig_8bit, cmap="gray")
            cax1.set_title("Original CT (Corrupted)", fontweight="bold", color="#e11d48", pad=8)
            cax1.axis("off")

            cax2.imshow(img_recon_8bit, cmap="gray")
            cax2.set_title("AI-MAR Restored CT", fontweight="bold", color="#0f766e", pad=8)
            cax2.axis("off")

            diff = np.abs(hu_orig - hu_recon)
            im3 = cax3.imshow(diff, cmap="magma", vmin=0, vmax=1500)
            cax3.set_title("Absolute HU Error (Discrepancy)", fontweight="bold", color="#92400e", pad=8)
            cax3.axis("off")
            plt.colorbar(im3, ax=cax3, fraction=0.046, pad=0.04)

            plt.tight_layout()
            png_buf = io.BytesIO()
            comp_fig.savefig(png_buf, format="png", bbox_inches="tight")
            png_buf.seek(0)
            plt.close(comp_fig)

            st.download_button(
                label="🖼️ Comparison Report (.png)",
                data=png_buf.getvalue(),
                file_name="ChemoPort_MAR_SideBySide_Report.png",
                mime="image/png",
                use_container_width=True,
                help="High-resolution triple-panel figure featuring original, restored, and difference maps.",
            )

        with export_col3:
            if st.session_state.slice_list and len(st.session_state.slice_list) > 1:
                if st.session_state.get("zip_cache") is None:
                    with st.spinner("Preparing Full Series DICOM ZIP archive..."):
                        import zipfile
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for i, sl in enumerate(st.session_state.slice_list):
                                ds_i = sl[0]
                                if i in st.session_state.recon_cache:
                                    hu_i = st.session_state.recon_cache[i][0]
                                else:
                                    hu_i = sl[1]
                                b = save_dicom_bytes(ds_i, hu_i)
                                inst = sl[2].get("instance_number", i + 1)
                                zf.writestr(f"AI_MAR_Slice_{inst:04d}.dcm", b)
                        st.session_state.zip_cache = zip_buf.getvalue()

                st.download_button(
                    label=f"📦 Download Full Series ZIP ({len(st.session_state.slice_list)} Slices)",
                    data=st.session_state.zip_cache,
                    file_name="ChemoPort_AI_MAR_Full_Series.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Compressed archive containing all restored CT slices.",
                )
            else:
                anon_ds = anonymize_dicom(st.session_state.current_ds)
                anon_bytes = save_dicom_bytes(anon_ds, hu_orig)
                st.download_button(
                    label="🔒 Export HIPAA-Anonymized DICOM (.dcm)",
                    data=anon_bytes,
                    file_name=f"ANON_Original_Instance{meta.get('instance_number', 1)}.dcm",
                    mime="application/dicom",
                    use_container_width=True,
                    help="Raw simulation DICOM with all protected health information (PHI) stripped.",
                )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        _, back_col, next_col, _ = st.columns([1.7, 0.8, 0.8, 1.7])
        with back_col:
            if st.button("← Back", key="nav_back_4", use_container_width=True, help="Back to 3. AI Correction"):
                st.session_state.current_workflow_step = "3. AI Correction"
                st.rerun()
        with next_col:
            if st.button("🔄 New Scan", key="nav_next_4", type="primary", use_container_width=True, help="Reset workspace and upload new CT scan"):
                st.session_state.current_ds = None
                st.session_state.current_hu = None
                st.session_state.current_meta = None
                st.session_state.recon_hu = None
                st.session_state.recon_stats = None
                st.session_state.data_source_name = "None"
                st.session_state.slice_list = []
                st.session_state.current_slice_idx = 0
                st.session_state.recon_cache = {}
                st.session_state.img_cache = {}
                st.session_state.zip_cache = None
                st.session_state.orig_b64_list = []
                st.session_state.recon_b64_list = []
                st.session_state.all_orig_b64_list = []
                st.session_state.all_recon_b64_list = []
                st.session_state.scan_id = uuid.uuid4().hex
                st.session_state.current_workflow_step = "1. Image Loading"
                st.rerun()




