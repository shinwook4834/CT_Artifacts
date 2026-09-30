"""ChemoPort CT-MAR Studio - Gradio Web Application.

3D CAD Prior-guided Metal Artifact Reduction (MAR) and Radiation Oncology Dosimetric Verification.
"""

from __future__ import annotations

import os
import sys

# Auto-dispatch to app_streamlit.py if running in a Streamlit environment (e.g. Streamlit Cloud)
_is_streamlit = False
if "streamlit" in sys.modules or any("streamlit" in str(arg).lower() for arg in sys.argv[:2]):
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _is_streamlit = get_script_run_ctx() is not None or any("streamlit" in str(arg).lower() for arg in sys.argv[:2])
    except Exception:
        _is_streamlit = True

if _is_streamlit:
    import runpy
    _st_target = os.path.join(os.path.dirname(__file__), "app_streamlit.py")
    if os.path.exists(_st_target):
        runpy.run_path(_st_target, run_name="__main__")
        sys.exit(0)

import io
import glob
import uuid
import tempfile
import zipfile
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import gradio as gr
try:
    from gradio_imageslider import ImageSlider
    HAS_IMAGE_SLIDER = True
except ImportError:
    HAS_IMAGE_SLIDER = False

# ZeroGPU support (Hugging Face Spaces). Falls back to a no-op decorator locally.
try:
    import spaces
    HAS_ZERO_GPU = True
except ImportError:
    HAS_ZERO_GPU = False
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from modules.dicom_io import (
    load_dicom_file,
    load_dicom_from_zip,
    apply_window_level,
    save_dicom_bytes,
    WINDOW_PRESETS,
)
from modules.reconstruction import find_chemoport_slice_index
from modules.ai_mar_pipeline import run_full_ai_restoration_pipeline

TEMP_EXPORT_DIR = tempfile.mkdtemp(prefix="chemoport_mar_")

# ==========================================
# BUILT-IN CLINICAL DEMO GENERATOR
# ==========================================

def generate_default_demo_series() -> List[Tuple[Dataset, np.ndarray, Dict[str, Any]]]:
    """Generates realistic clinical demo DICOM slices (slices 116..135) from repository artifacts."""
    series = []
    png_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "full_slice_z*.png")))
    if not png_files:
        png_files = sorted(glob.glob("full_slice_z*.png"))

    if not png_files:
        # Fallback synthetic phantom
        h, w = 512, 512
        for z in range(116, 136):
            hu = np.full((h, w), -1000.0, dtype=np.float32)
            Y, X = np.ogrid[:h, :w]
            body = ((Y - 256) ** 2 / 180**2 + (X - 256) ** 2 / 210**2) <= 1.0
            hu[body] = -40.0
            lung_l = ((Y - 260) ** 2 / 100**2 + (X - 180) ** 2 / 60**2) <= 1.0
            lung_r = ((Y - 260) ** 2 / 100**2 + (X - 332) ** 2 / 60**2) <= 1.0
            hu[lung_l | lung_r] = -750.0
            if 120 <= z <= 130:
                port = np.sqrt((Y - 188) ** 2 + (X - 328) ** 2) <= 12.0
                hu[port] = 2850.0
            ds = Dataset()
            meta = {
                "patient_id": "DEMO_PATIENT",
                "patient_name": "ANON^CHEMOPORT",
                "series_description": "Synthetic CT Simulation",
                "instance_number": z,
                "slice_thickness": "2.5 mm",
                "pixel_spacing": "0.977 x 0.977 mm",
                "dimensions": "512 x 512",
                "metal_pixel_count": int(np.sum(hu >= 2000.0)),
                "min_hu": float(np.min(hu)),
                "max_hu": float(np.max(hu)),
            }
            series.append((ds, hu, meta))
        return series

    for i, pf in enumerate(png_files):
        img_gray = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        hu = (img_gray.astype(np.float32) / 255.0) * 350.0 - 135.0
        Y, X = np.ogrid[:512, :512]
        port_core = np.sqrt((Y - 188)**2 + (X - 328)**2) <= 12.0
        hu[port_core] = 2850.0

        ds = Dataset()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = file_meta
        ds.PatientName = "DEMO^CHEMOPORT"
        ds.PatientID = "CT_DEMO_01"
        ds.Modality = "CT"
        ds.SeriesDescription = "Thorax Simulation CT"
        ds.InstanceNumber = 116 + i * 4
        ds.Rows, ds.Columns = 512, 512
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = ["0.9765625", "0.9765625"]
        ds.SliceThickness = "2.5"
        ds.RescaleSlope = "1.0"
        ds.RescaleIntercept = "-1024.0"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        raw_int = np.clip(np.round(hu + 1024.0), 0, 65535).astype(np.uint16)
        ds.PixelData = raw_int.tobytes()

        meta = {
            "patient_id": "CT_DEMO_01",
            "patient_name": "DEMO^CHEMOPORT",
            "series_description": "Thorax Simulation CT",
            "instance_number": 116 + i * 4,
            "slice_thickness": "2.5 mm",
            "pixel_spacing": "0.977 x 0.977 mm",
            "dimensions": "512 x 512",
            "metal_pixel_count": int(np.sum(hu >= 2000.0)),
            "min_hu": float(np.min(hu)),
            "max_hu": float(np.max(hu)),
        }
        series.append((ds, hu, meta))

    return series


# ==========================================
# CORE WORKFLOW CONTROLLERS
# ==========================================

def get_window_levels(preset: str, custom_wc: float, custom_ww: float) -> Tuple[float, float]:
    preset_map = {
        "Soft Tissue (40 / 350)": (40.0, 350.0),
        "Bone Window (400 / 1800)": (400.0, 1800.0),
        "Lung Window (-600 / 1500)": (-600.0, 1500.0),
        "Metal Detail (10760 / 33530)": (10760.0, 33530.0),
    }
    if preset in preset_map:
        return preset_map[preset]
    return float(custom_wc), float(max(1.0, custom_ww))


@spaces.GPU
def handle_upload(files: List[Any], state: Dict[str, Any]):
    if not files:
        return (
            "⚠️ No files uploaded.",
            state,
            gr.update(minimum=1, maximum=1, value=1),
            None,
            None,
            ""
        )

    slices = []
    for f in files:
        f_path = f.name if hasattr(f, "name") else str(f)
        try:
            with open(f_path, "rb") as fp:
                content = fp.read()
            if f_path.lower().endswith(".zip"):
                zip_slices = load_dicom_from_zip(content)
                slices.extend(zip_slices)
            else:
                ds, hu, meta = load_dicom_file(content)
                slices.append((ds, hu, meta))
        except Exception as e:
            continue

    if not slices:
        return (
            "❌ Failed to parse valid DICOM files from upload.",
            state,
            gr.update(minimum=1, maximum=1, value=1),
            None,
            None,
            ""
        )

    slices.sort(key=lambda s: s[2].get("instance_number", 0))
    best_idx = find_chemoport_slice_index(slices)

    # Run AI pipeline
    mar_results = run_full_ai_restoration_pipeline(slices, inpaint_strength=0.95)

    new_state = {
        "slices": slices,
        "mar_results": mar_results,
        "current_idx": best_idx,
        "inpaint_strength": 0.95,
        "total_slices": len(slices),
    }

    first_meta = slices[best_idx][2]
    info_md = f"""
### 🩺 Patient CT Series Loaded
- **Series Description**: `{first_meta.get('series_description', 'CT Series')}`
- **Patient ID (De-identified)**: `{first_meta.get('patient_id', 'ANONYMIZED')}`
- **Total Slices**: `{len(slices)} Slices`
- **Slice Dimensions**: `{first_meta.get('dimensions', '512x512')}` (Spacing: `{first_meta.get('pixel_spacing', '1.0 mm')}`)
- **ChemoPort Target Slice**: `Slice {best_idx + 1} of {len(slices)}` (Detected automatically)
"""
    slider_up = gr.update(minimum=1, maximum=len(slices), value=best_idx + 1)
    
    # Render inspection & AI view
    hu_orig = slices[best_idx][1]
    hu_recon = mar_results[best_idx][0]
    img_orig = apply_window_level(hu_orig, wc=40.0, ww=350.0)
    img_recon = apply_window_level(hu_recon, wc=40.0, ww=350.0)
    
    diff = np.abs(hu_orig - hu_recon)
    norm_diff = np.clip(diff / 500.0, 0.0, 1.0)
    diff_colored = cv2.applyColorMap((norm_diff * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    status_str = f"✅ Successfully loaded {len(slices)} slices! ChemoPort located on Slice {best_idx + 1}."
    return (
        info_md,
        new_state,
        slider_up,
        img_orig,
        (img_orig, img_recon) if HAS_IMAGE_SLIDER else img_recon,
        diff_colored,
        status_str
    )


@spaces.GPU
def handle_load_demo(state: Dict[str, Any]):
    slices = generate_default_demo_series()
    best_idx = find_chemoport_slice_index(slices)
    mar_results = run_full_ai_restoration_pipeline(slices, inpaint_strength=0.95)

    new_state = {
        "slices": slices,
        "mar_results": mar_results,
        "current_idx": best_idx,
        "inpaint_strength": 0.95,
        "total_slices": len(slices),
    }

    first_meta = slices[best_idx][2]
    info_md = f"""
### 🩺 Clinical Demo CT Series Loaded
- **Series Description**: `{first_meta.get('series_description', 'Thorax Simulation CT')}`
- **Patient ID (De-identified)**: `{first_meta.get('patient_id', 'CT_DEMO_01')}`
- **Total Slices**: `{len(slices)} Slices` (Z=116..135)
- **Slice Dimensions**: `512 x 512` (Spacing: `0.977 x 0.977 mm`, Thickness: `2.5 mm`)
- **ChemoPort Target Slice**: `Slice {best_idx + 1} of {len(slices)}` (Titanium port located)
"""
    slider_up = gr.update(minimum=1, maximum=len(slices), value=best_idx + 1)
    
    hu_orig = slices[best_idx][1]
    hu_recon = mar_results[best_idx][0]
    img_orig = apply_window_level(hu_orig, wc=40.0, ww=350.0)
    img_recon = apply_window_level(hu_recon, wc=40.0, ww=350.0)
    
    diff = np.abs(hu_orig - hu_recon)
    norm_diff = np.clip(diff / 500.0, 0.0, 1.0)
    diff_colored = cv2.applyColorMap((norm_diff * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    status_str = f"✅ Clinical demo CT loaded ({len(slices)} slices). ChemoPort active on Slice {best_idx + 1}."
    return (
        info_md,
        new_state,
        slider_up,
        img_orig,
        (img_orig, img_recon) if HAS_IMAGE_SLIDER else img_recon,
        diff_colored,
        status_str
    )


def handle_slice_change(slice_num: int, preset: str, custom_wc: float, custom_ww: float, state: Dict[str, Any]):
    if not state or "slices" not in state or not state["slices"]:
        return None, None, None, "No scan loaded."

    idx = max(0, min(int(slice_num) - 1, state["total_slices"] - 1))
    state["current_idx"] = idx

    wc, ww = get_window_levels(preset, custom_wc, custom_ww)
    hu_orig = state["slices"][idx][1]
    hu_recon = state["mar_results"][idx][0]

    img_orig = apply_window_level(hu_orig, wc=wc, ww=ww)
    img_recon = apply_window_level(hu_recon, wc=wc, ww=ww)

    diff = np.abs(hu_orig - hu_recon)
    norm_diff = np.clip(diff / 500.0, 0.0, 1.0)
    diff_colored = cv2.applyColorMap((norm_diff * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    diag_info = f"**Current Slice**: {idx + 1} / {state['total_slices']} | **HU Range**: [{np.min(hu_orig):.0f}, {np.max(hu_orig):.0f}] → Restored: [{np.min(hu_recon):.0f}, {np.max(hu_recon):.0f}]"

    slider_val = (img_orig, img_recon) if HAS_IMAGE_SLIDER else img_recon
    return img_orig, slider_val, diff_colored, diag_info


@spaces.GPU
def handle_recompute_pipeline(inpaint_strength: float, preset: str, custom_wc: float, custom_ww: float, state: Dict[str, Any]):
    if not state or "slices" not in state or not state["slices"]:
        return None, None, None, "No scan loaded to recompute."

    state["inpaint_strength"] = float(inpaint_strength)
    state["mar_results"] = run_full_ai_restoration_pipeline(state["slices"], inpaint_strength=float(inpaint_strength))

    idx = state.get("current_idx", 0)
    wc, ww = get_window_levels(preset, custom_wc, custom_ww)
    hu_orig = state["slices"][idx][1]
    hu_recon = state["mar_results"][idx][0]

    img_orig = apply_window_level(hu_orig, wc=wc, ww=ww)
    img_recon = apply_window_level(hu_recon, wc=wc, ww=ww)

    diff = np.abs(hu_orig - hu_recon)
    norm_diff = np.clip(diff / 500.0, 0.0, 1.0)
    diff_colored = cv2.applyColorMap((norm_diff * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    slider_val = (img_orig, img_recon) if HAS_IMAGE_SLIDER else img_recon
    status_str = f"✨ Recomputed AI restoration across all {len(state['slices'])} slices with inpaint strength {inpaint_strength:.2f}."
    return img_orig, slider_val, diff_colored, status_str


def export_current_slice_dcm(state: Dict[str, Any]):
    if not state or "slices" not in state or not state["slices"]:
        return None
    idx = state.get("current_idx", 0)
    ds_orig = state["slices"][idx][0]
    hu_recon = state["mar_results"][idx][0]
    dcm_bytes = save_dicom_bytes(ds_orig, hu_recon)
    
    out_path = os.path.join(TEMP_EXPORT_DIR, f"AI_MAR_Restored_Slice_{idx + 1}.dcm")
    with open(out_path, "wb") as f:
        f.write(dcm_bytes)
    return out_path


def export_report_image(preset: str, custom_wc: float, custom_ww: float, state: Dict[str, Any]):
    if not state or "slices" not in state or not state["slices"]:
        return None
    idx = state.get("current_idx", 0)
    wc, ww = get_window_levels(preset, custom_wc, custom_ww)
    hu_orig = state["slices"][idx][1]
    hu_recon = state["mar_results"][idx][0]

    img_o = apply_window_level(hu_orig, wc=wc, ww=ww)
    img_r = apply_window_level(hu_recon, wc=wc, ww=ww)
    diff = np.abs(hu_orig - hu_recon)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    ax1.imshow(img_o, cmap="gray")
    ax1.set_title(f"Original Corrupted CT (Slice {idx + 1})", color="#e11d48", fontweight="bold", pad=8)
    ax1.axis("off")

    ax2.imshow(img_r, cmap="gray")
    ax2.set_title("AI-MAR Restored Normal CT", color="#0f766e", fontweight="bold", pad=8)
    ax2.axis("off")

    im3 = ax3.imshow(diff, cmap="magma", vmin=0, vmax=1000)
    ax3.set_title("Absolute HU Error Map", color="#92400e", fontweight="bold", pad=8)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_path = os.path.join(TEMP_EXPORT_DIR, f"ChemoPort_MAR_Report_Slice_{idx + 1}.png")
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_series_zip(state: Dict[str, Any]):
    if not state or "slices" not in state or not state["slices"]:
        return None
    zip_path = os.path.join(TEMP_EXPORT_DIR, "ChemoPort_AI_MAR_Full_Series.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (sl, res) in enumerate(zip(state["slices"], state["mar_results"])):
            ds_orig = sl[0]
            hu_recon = res[0]
            dcm_bytes = save_dicom_bytes(ds_orig, hu_recon)
            zf.writestr(f"AI_MAR_Slice_{idx + 1:03d}.dcm", dcm_bytes)
    return zip_path


# ==========================================
# GRADIO UI DEFINITION
# ==========================================

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
)


custom_css = """
.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
}
.hero-box {
    background: linear-gradient(135deg, rgba(240, 253, 250, 0.85) 0%, rgba(236, 254, 255, 0.85) 100%);
    border: 1px solid rgba(13, 148, 136, 0.25);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px -4px rgba(13, 148, 136, 0.1);
}
.hero-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.hero-sub {
    font-size: 0.92rem;
    color: #475569;
    margin: 0;
    line-height: 1.5;
}
.pipeline-pill {
    display: inline-flex;
    align-items: center;
    background: #0d9488;
    color: white;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 9999px;
    margin-top: 8px;
}
.card-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    margin-top: 8px;
}
"""

with gr.Blocks(title="ChemoPort CT-MAR Studio") as demo:
    state_store = gr.State(value={})

    gr.HTML(
        """
        <div class="hero-box">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
              <div class="hero-title">🩺 ChemoPort CT-MAR Studio</div>
              <div class="hero-sub">
                3D CAD Prior-Guided Deep Learning Metal Artifact Reduction & Dosimetric Accuracy in Breast Cancer Radiation Therapy
              </div>
            </div>
            <span class="pipeline-pill">✨ 2-Step AI Pipeline Active</span>
          </div>
        </div>
        """
    )

    with gr.Tabs() as tabs:
        # -------------------------------------------------------------
        # TAB 1: IMAGE LOADING
        # -------------------------------------------------------------
        with gr.TabItem("📂 1. Image Loading", id="tab_load"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### 📥 Upload Patient CT Scan")
                    upload_files = gr.File(
                        label="Drag and Drop DICOM Files (.dcm) or ZIP Series",
                        file_count="multiple",
                        file_types=[".dcm", ".zip"],
                        type="filepath",
                    )
                    with gr.Row():
                        btn_load_demo = gr.Button("🩺 Load Sample CT Series (Z=116..135)", variant="secondary")
                        btn_process_upload = gr.Button("⚡ Process Uploaded Scan", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("#### 📋 CT Series & Implant Metadata")
                    metadata_box = gr.Markdown(
                        """
*No CT scan loaded yet.*

👉 Click **Load Sample CT Series** to explore with pre-loaded simulation CT data, or upload patient DICOM files.
"""
                    )
                    status_banner = gr.Markdown("")

        # -------------------------------------------------------------
        # TAB 2: ARTIFACT INSPECTION
        # -------------------------------------------------------------
        with gr.TabItem("🔍 2. Artifact Inspection", id="tab_inspect"):
            with gr.Row():
                with gr.Column(scale=7):
                    img_inspect = gr.Image(
                        label="Corrupted CT Slice (Titanium Metal Blooming & Streaks)",
                        type="numpy",
                        interactive=False,
                    )
                with gr.Column(scale=3):
                    gr.Markdown("#### 🎛️ Navigation & Window Controls")
                    slice_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=1,
                        step=1,
                        label="CT Slice Number",
                    )

                    window_preset = gr.Dropdown(
                        label="Window / Level Preset",
                        choices=[
                            "Soft Tissue (40 / 350)",
                            "Bone Window (400 / 1800)",
                            "Lung Window (-600 / 1500)",
                            "Metal Detail (10760 / 33530)",
                            "Custom",
                        ],
                        value="Soft Tissue (40 / 350)",
                    )
                    with gr.Row():
                        wc_slider = gr.Slider(-1000, 15000, value=40, step=10, label="Window Center (WL)")
                        ww_slider = gr.Slider(10, 35000, value=350, step=10, label="Window Width (WW)")

                    diag_text = gr.Markdown("**Slice status**: Ready")
                    gr.Markdown(
                        """
> [!WARNING]
> **Dosimetric Risk Alert**
> High-density titanium ChemoPorts trigger severe photon starvation (dark streaks below -200 HU) and bright flares, distorting dose calculations to the supraclavicular lymph node bed by up to 12.8%.
"""
                    )

        # -------------------------------------------------------------
        # TAB 3: AI CORRECTION & STUDIO
        # -------------------------------------------------------------
        with gr.TabItem("✨ 3. AI Correction & Dual-View", id="tab_correction"):
            with gr.Row():
                with gr.Column(scale=7):
                    gr.Markdown("#### ↔️ Interactive Comparison (Slide to compare Original vs AI Restored)")
                    if HAS_IMAGE_SLIDER:
                        img_compare = ImageSlider(
                            label="Original (Corrupted) ↔ AI-MAR Restored (Normal CT)",
                            type="numpy",
                            position=0.5,
                        )
                    else:
                        img_compare = gr.Image(label="AI-MAR Restored Normal CT", type="numpy")

                    img_diff = gr.Image(label="Absolute HU Discrepancy Error Map (|Original - Restored|)", type="numpy")

                with gr.Column(scale=3):
                    gr.Markdown("#### ⚙️ AI Restoration Parameters")
                    inpaint_str = gr.Slider(
                        minimum=0.50,
                        maximum=1.00,
                        value=0.95,
                        step=0.01,
                        label="Inpainting & Normalization Strength",
                    )
                    btn_recompute = gr.Button("⚡ Recompute AI MAR Restoration", variant="primary")
                    
                    gr.Markdown(
                        """
<div class="card-box">
  <strong style="color: #0284c7;">1. AI Surgical Guide Wire Continuity:</strong><br/>
  <span style="font-size: 0.85rem; color: #475569;">Restores authentic hookwire inserted for surgical guidance across slices 116..135 with physical CT PSF roll-off.</span><br/><br/>
  <strong style="color: #059669;">2. Normal CT Tissue Reconstruction:</strong><br/>
  <span style="font-size: 0.85rem; color: #475569;">Eliminates titanium port blooming and dark starvation shadows, restoring normal breast parenchymal anatomy for TPS.</span>
</div>
"""
                    )

        # -------------------------------------------------------------
        # TAB 4: SAVE & TPS EXPORT
        # -------------------------------------------------------------
        with gr.TabItem("💾 4. Save & TPS Export", id="tab_export"):
            gr.Markdown("### 💾 Treatment Planning System (TPS) DICOM Export")
            gr.Markdown(
                "Export standard DICOM files compatible with **Varian Eclipse**, **Elekta Monaco**, and **RaySearch RayStation**."
            )
            with gr.Row():
                with gr.Column():
                    btn_dl_dcm = gr.Button("💾 Generate Single Slice DICOM (.dcm)", variant="primary")
                    file_out_dcm = gr.File(label="Download Corrected DICOM")

                with gr.Column():
                    btn_dl_png = gr.Button("🖼️ Generate 3-Panel Comparison Report (.png)", variant="secondary")
                    file_out_png = gr.File(label="Download Comparison PNG")

                with gr.Column():
                    btn_dl_zip = gr.Button("📦 Generate Full Series Archive (.zip)", variant="secondary")
                    file_out_zip = gr.File(label="Download Full Series ZIP")

    # -------------------------------------------------------------
    # EVENT BINDINGS
    # -------------------------------------------------------------
    btn_process_upload.click(
        fn=handle_upload,
        inputs=[upload_files, state_store],
        outputs=[metadata_box, state_store, slice_slider, img_inspect, img_compare, img_diff, status_banner],
    )

    btn_load_demo.click(
        fn=handle_load_demo,
        inputs=[state_store],
        outputs=[metadata_box, state_store, slice_slider, img_inspect, img_compare, img_diff, status_banner],
    )

    slice_slider.change(
        fn=handle_slice_change,
        inputs=[slice_slider, window_preset, wc_slider, ww_slider, state_store],
        outputs=[img_inspect, img_compare, img_diff, diag_text],
    )

    window_preset.change(
        fn=handle_slice_change,
        inputs=[slice_slider, window_preset, wc_slider, ww_slider, state_store],
        outputs=[img_inspect, img_compare, img_diff, diag_text],
    )

    wc_slider.change(
        fn=handle_slice_change,
        inputs=[slice_slider, window_preset, wc_slider, ww_slider, state_store],
        outputs=[img_inspect, img_compare, img_diff, diag_text],
    )

    ww_slider.change(
        fn=handle_slice_change,
        inputs=[slice_slider, window_preset, wc_slider, ww_slider, state_store],
        outputs=[img_inspect, img_compare, img_diff, diag_text],
    )

    btn_recompute.click(
        fn=handle_recompute_pipeline,
        inputs=[inpaint_str, window_preset, wc_slider, ww_slider, state_store],
        outputs=[img_inspect, img_compare, img_diff, diag_text],
    )

    btn_dl_dcm.click(
        fn=export_current_slice_dcm,
        inputs=[state_store],
        outputs=[file_out_dcm],
    )

    btn_dl_png.click(
        fn=export_report_image,
        inputs=[window_preset, wc_slider, ww_slider, state_store],
        outputs=[file_out_png],
    )

    btn_dl_zip.click(
        fn=export_series_zip,
        inputs=[state_store],
        outputs=[file_out_zip],
    )

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.queue().launch(server_name=server_name, server_port=server_port, theme=custom_theme, css=custom_css)

