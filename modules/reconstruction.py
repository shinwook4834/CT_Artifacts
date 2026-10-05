"""Reconstruction and baseline Metal Artifact Reduction (MAR) algorithms for ChemoPort CT-MAR Studio."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter


def detect_metal_mask(hu_array: np.ndarray, threshold: float = 2000.0) -> np.ndarray:
    """Detects metal port components based on Hounsfield Unit thresholding and morphological cleanup."""
    raw_mask = (hu_array >= threshold).astype(np.uint8)

    # Filter out tiny spurious noise pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
    clean_mask = np.zeros_like(raw_mask)

    # Retain valid metal components including peripheral port / catheter tracks
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 2:  # Valid metal implant cluster or catheter track
            clean_mask[labels == i] = 1

    return clean_mask.astype(bool)


def detect_streak_artifacts(hu_array: np.ndarray, metal_mask: np.ndarray) -> np.ndarray:
    """Identifies bright flare streaks and deep photon-starvation dark shadows around the metal port."""
    # Dilate metal mask to define the peri-implant search zone (up to 40mm)
    dilated_zone = binary_dilation(metal_mask, iterations=45)
    peri_zone = dilated_zone & (~metal_mask)

    # Dark streak: severe photon starvation where tissue HU drops unnaturally below -200 (in soft tissue)
    # Bright streak: high flare HU > 300 in non-bone subcutaneous tissue
    dark_shadows = peri_zone & (hu_array < -300)
    bright_streaks = peri_zone & (hu_array > 350)

    artifact_mask = dark_shadows | bright_streaks
    # Morphological closing to create contiguous correction regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    artifact_mask = cv2.morphologyEx(artifact_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return artifact_mask


from modules.cad_matching import apply_stage1_cad_prior, PortGeometry
from modules.inpainting_engine import apply_stage2_conditional_inpainting


def two_stage_chemoport_mar(
    hu_array: np.ndarray,
    metal_threshold: float = 2000.0,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    cad_preset: str = "universal",
    streak_reduction_strength: float = 0.88,
    preserve_skin_boundary: bool = True,
    ai_weight: float = 0.0,
    engine_type: str = "ccaf_transformer"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Executes the full 2-Stage CAD Prior AI-MAR pipeline:
    - Stage 1: Universal CAD Prior geometry matching & tri-compartment density override.
    - Stage 2: Anatomical boundary-preserving conditional inpainting.
    - Optional: Deep learning neural synthesis blending with user-controlled ai_weight.

    Returns:
        recon_hu: Reconstructed clean CT HU array.
        metal_mask: Detected metal mask.
        artifact_mask: Detected artifact mask.
        stats: Diagnostic statistics dictionary containing Stage 1, Stage 2, and AI metrics.
    """
    # Fast path: check if any metal pixel exists
    if not np.any(hu_array >= metal_threshold):
        empty_mask = np.zeros_like(hu_array, dtype=bool)
        return (
            hu_array.copy(),
            empty_mask.copy(),
            empty_mask.copy(),
            {
                "metal_detected": False,
                "metal_pixel_count": 0,
                "artifact_pixel_count": 0,
                "streak_suppression_pct": 0.0,
                "stage1_success": False,
                "stage2_success": False,
                "ai_weight": ai_weight,
            }
        )

    # =========================================================================
    # STEP 2: Extract ChemoPort Contour & 3D Pose from all HU
    # =========================================================================
    stage1_hu, geom, masks, s1_stats = apply_stage1_cad_prior(
        hu_array,
        metal_threshold=metal_threshold,
        pixel_spacing=pixel_spacing,
        cad_preset=cad_preset
    )

    if geom is None:
        full_port_mask = detect_metal_mask(hu_array, threshold=metal_threshold)
        metal_mask = full_port_mask
        stage1_hu = hu_array.copy()
    else:
        full_port_mask = masks.get("full_port", (hu_array >= metal_threshold))
        metal_mask = masks.get("true_metal", full_port_mask)

    catheter_mask = masks.get("catheter", None)

    # =========================================================================
    # STEP 1: AI Anatomical Background Normalization (ChemoPort Ignored)
    # Generate the most natural, streak-free normal CT image of thorax & breast
    # =========================================================================
    healed_hu, artifact_mask, s2_stats = apply_stage2_conditional_inpainting(
        stage1_hu,
        full_port_mask=full_port_mask,
        inpaint_strength=streak_reduction_strength,
        preserve_skin_boundary=preserve_skin_boundary,
        catheter_mask=catheter_mask
    )

    from modules.inpainting_engine import extract_anatomical_priors, synthesize_contralateral_breast_prior
    priors = extract_anatomical_priors(hu_array)

    # Heal the implant cavity into normal breast soft tissue (contralateral breast guided)
    clean_anatomy_hu = healed_hu.copy()
    if np.any(full_port_mask):
        contra_breast = synthesize_contralateral_breast_prior(healed_hu, priors, full_port_mask=full_port_mask)
        clean_anatomy_hu[full_port_mask] = contra_breast[full_port_mask]

    # AI Texture Refinement on the clean anatomical background
    ai_stats = {
        "ai_weight": float(ai_weight),
        "ai_active": ai_weight > 0.0,
        "clean_anatomy_hu": clean_anatomy_hu,
        "full_port_mask": full_port_mask,
        "metal_mask": metal_mask,
        "catheter_mask": catheter_mask,
        "artifact_mask": artifact_mask,
        "stage1_hu": stage1_hu,
        "priors": priors,
    }

    if ai_weight > 0.0 and engine_type != "physics_cad":
        try:
            import time
            from modules.deep_mar_model import run_deep_mar_inference, blend_hybrid_mar

            t_ai_0 = time.perf_counter()
            ai_bg_hu = run_deep_mar_inference(
                orig_hu=hu_array,
                stage1_hu=stage1_hu,
                artifact_mask=artifact_mask,
                priors=priors,
                physics_hu=clean_anatomy_hu,
                metal_mask=metal_mask,
                full_port_mask=full_port_mask,
                engine_type=engine_type,
            )
            t_ai_1 = time.perf_counter()
            ai_stats["ai_inference_time_ms"] = round((t_ai_1 - t_ai_0) * 1000.0, 2)
            ai_stats["ai_hu"] = ai_bg_hu
            ai_stats["engine_type"] = engine_type

            clean_anatomy_hu = blend_hybrid_mar(
                physics_hu=clean_anatomy_hu,
                ai_hu=ai_bg_hu,
                ai_weight=ai_weight,
                full_port_mask=None,
                metal_mask=None,
                priors=priors,
                orig_hu=hu_array,
                artifact_mask=artifact_mask,
            )
            ai_stats["ai_active"] = True
        except Exception as e:
            ai_stats["ai_error"] = str(e)

    # =========================================================================
    # STEP 3: 3D CAD Shape Rotation, Pose Matching & Catheter Wire Insertion
    # Rotate multi-density CAD model according to extracted pose & insert with PSF
    # =========================================================================
    from scipy.ndimage import distance_transform_edt
    final_hu = clean_anatomy_hu.copy()

    if np.any(full_port_mask) and geom is not None and "shell" in masks:
        # Multi-density CAD compartments:
        # A. Saline fluid reservoir (25 HU)
        if "chamber" in masks and np.any(masks["chamber"]):
            final_hu[masks["chamber"]] = 25.0
        # B. Silicone septum puncture dome (120 HU)
        if "septum" in masks and np.any(masks["septum"]):
            final_hu[masks["septum"]] = 120.0
        # C. Solid titanium outer casing shell (>= 4800 HU)
        if "shell" in masks and np.any(masks["shell"]):
            final_hu[masks["shell"]] = np.maximum(hu_array[masks["shell"]], 4800.0)

        # D. Point Spread Function (PSF) edge feathering (1.8 px)
        dist_port = distance_transform_edt(~full_port_mask)
        psf_ring = (dist_port > 0) & (dist_port <= 1.8) & priors.body_mask & (~priors.lung_mask)
        if np.any(psf_ring):
            smooth_edge = cv2.GaussianBlur(final_hu.astype(np.float32), (5, 5), 1.0)
            w_psf = np.clip(dist_port[psf_ring] / 1.8, 0.0, 1.0)
            final_hu[psf_ring] = (1.0 - w_psf) * smooth_edge[psf_ring] + w_psf * final_hu[psf_ring]
    elif np.any(full_port_mask):
        final_hu[full_port_mask] = np.maximum(hu_array[full_port_mask], 4500.0)

    # E. Continuous Catheter Wire Insertion & Smooth Edge Connection
    if catheter_mask is not None and np.any(catheter_mask):
        final_hu[catheter_mask] = np.maximum(hu_array[catheter_mask], 1750.0)
        c_dil = binary_dilation(catheter_mask, iterations=1) & (~catheter_mask) & (~full_port_mask) & priors.body_mask & (~priors.lung_mask)
        if np.any(c_dil):
            final_hu[c_dil] = np.clip(final_hu[c_dil] * 0.45 + 450.0, -100.0, 950.0)

    # Combine statistics
    stats = {
        "metal_detected": True,
        "metal_pixel_count": int(np.sum(metal_mask)),
        "catheter_pixel_count": int(np.sum(catheter_mask)) if catheter_mask is not None else 0,
        "artifact_pixel_count": s2_stats.get("artifact_pixel_count", int(np.sum(artifact_mask))),
        "streak_suppression_pct": s2_stats.get("streak_suppression_pct", 0.0),
        "min_hu_before": float(np.min(hu_array)),
        "min_hu_after": float(np.min(final_hu)),
        "max_hu_before": float(np.max(hu_array)),
        "max_hu_after": float(np.max(final_hu)),
        "stage1": s1_stats,
        "stage2": s2_stats,
        "ai": ai_stats,
    }

    return final_hu, metal_mask, artifact_mask, stats


def baseline_chemoport_mar(
    hu_array: np.ndarray,
    metal_threshold: float = 2000.0,
    cad_prior_override: bool = True,
    streak_reduction_strength: float = 0.88,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    cad_preset: str = "universal",
    ai_weight: float = 0.0,
    engine_type: str = "ccaf_transformer"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Backward-compatible entry point delegating to the 2-Stage CAD Prior AI pipeline."""
    return two_stage_chemoport_mar(
        hu_array=hu_array,
        metal_threshold=metal_threshold,
        pixel_spacing=pixel_spacing,
        cad_preset=cad_preset,
        streak_reduction_strength=streak_reduction_strength,
        preserve_skin_boundary=True,
        ai_weight=ai_weight,
        engine_type=engine_type
    )


def find_chemoport_slice_index(slices) -> int:
    """Identifies the slice index containing the ChemoPort reservoir body.

    A ChemoPort is a dense titanium/plastic chamber implanted subcutaneously
    in the anterior-lateral thoracic wall (infraclavicular/pectoral pocket).
    In clinical CT scans containing both the neck/head and thorax/pelvis,
    dental amalgams and pelvic clips can confound naive HU thresholding.

    Criteria for ChemoPort localization:
    1. HU intensity: Titanium port bodies exhibit extreme peak HU (> 20,000 HU).
    2. Anterior thoracic location: Subcutaneous anterior chest wall (Y < 0.52 * H in standard supine axial CT).
    3. Lateral offset: Usually implanted in the right or left infraclavicular region (|X - W/2| > 0.05 * W).
    4. Torso vertical range: Located in the thoracic cavity (excluding cranial vertex / oral cavity slices
       and lower pelvic slices when a full multi-slice series is present).

    Args:
        slices: List of tuples (ds, hu_array, meta) or list of hu_arrays.

    Returns:
        Index of the slice containing the ChemoPort reservoir center.
    """
    if not slices:
        return 0
    if len(slices) == 1:
        return 0

    total_slices = len(slices)
    scores = []

    for idx, item in enumerate(slices):
        hu = item[1] if isinstance(item, (tuple, list)) else item
        cand = hu >= 1800.0
        if not np.any(cand):
            scores.append(-1.0)
            continue

        h, w = hu.shape
        num_c, lbls, stats, centroids = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
        slice_best_score = -1.0
        for i in range(1, num_c):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if cy >= 0.58 * h or cy < 0.08 * h:
                continue
            if cx < 0.1 * w or cx > 0.9 * w:
                continue
            if abs(cx - w / 2.0) < 0.03 * w:
                continue
            if area < 4 or area > 1500:
                continue
            peak_hu = float(np.max(hu[lbls == i]))
            comp_score = peak_hu * 1.5 + min(area, 250) * 10.0
            if peak_hu >= 2400.0:
                comp_score += 5000.0
            if abs(cx - w / 2.0) > 0.05 * w:
                comp_score += 3000.0
            rel_pos = idx / total_slices
            if total_slices > 30 and (0.15 <= rel_pos <= 0.85):
                comp_score += 3000.0
            if comp_score > slice_best_score:
                slice_best_score = comp_score

        scores.append(slice_best_score)

    best_idx = int(np.argmax(scores))
    if scores[best_idx] < 0:
        return total_slices // 2
    return best_idx

