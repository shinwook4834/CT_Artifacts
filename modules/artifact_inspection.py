"""Physics-Informed Anatomical Residual AI for Step 2 Artifact Inspection.

Approach A (Physics-Informed Anatomical Residual AI):
1. ChemoPort Titanium Core Segmentation:
   - High attenuation HU threshold (>= 2000.0 HU)
   - Anatomical anterior-lateral chest wall localization (Y < 0.62 * H)
   - 3D spatial continuity tracking across the scan to exclude dental fillings and pelvic clips
   - Rose Red vector contour extraction for the port body and catheter
2. Anatomical Prior Protection (Anti-False-Positive):
   - Strict exclusion of lung cavity (HU < -500)
   - Strict exclusion of cortical bone (ribs, sternum, clavicle)
   - Strict exclusion of deep mediastinum / heart
   - Strict exclusion of patient exterior air
3. Physics-Informed Anomaly Residuals:
   - Photon starvation dark streaks (HU < -140 HU in subcutaneous fat, HU < -50 HU in muscle/glandular)
   - Beam hardening & high-Z scatter bright flares (HU > 135 HU in soft tissue away from bone)
   - Morphological closing to establish contiguous physical streak rays
   - Amber Yellow vector contour extraction
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from modules.inpainting_engine import extract_anatomical_priors, AnatomicalPriors


def detect_approach_a_slice_contours(
    hu_slice: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    port_anchor: Optional[Tuple[float, float]] = None,
    max_anchor_dist_px: float = 65.0,
    peri_zone_radius_px: float = 95.0,
) -> Tuple[Dict[str, List], Dict[str, Any], Dict[str, np.ndarray]]:
    """Segments ChemoPort metal and radiating streak artifacts for a single 2D CT slice.

    Returns:
        contours: {"port": [...], "artifact": [...], "wire": []}
        stats: Dictionary of inspection metrics
        masks: {"port_mask": bool_array, "artifact_mask": bool_array}
    """
    h, w = hu_slice.shape
    empty_contours = {"port": [], "artifact": [], "wire": [], "port_px": 0, "art_px": 0}
    empty_masks = {
        "port_mask": np.zeros((h, w), dtype=bool),
        "artifact_mask": np.zeros((h, w), dtype=bool),
    }

    # Fast screening: if max HU is well below titanium metal, skip heavy extraction
    if np.max(hu_slice) < 1800.0:
        stats = {
            "metal_detected": False,
            "metal_pixel_count": 0,
            "artifact_pixel_count": 0,
            "dark_streak_pixels": 0,
            "bright_flare_pixels": 0,
            "port_center": None,
            "status": "No metal detected (Normal CT anatomy)",
        }
        return empty_contours, stats, empty_masks

    # 1. Extract Anatomical Landmark Priors
    priors = extract_anatomical_priors(hu_slice)

    # 2. ChemoPort Titanium Core Detection
    cand_metal = (hu_slice >= 2000.0) & priors.body_mask & (~priors.lung_mask)
    if not np.any(cand_metal):
        stats = {
            "metal_detected": False,
            "metal_pixel_count": 0,
            "artifact_pixel_count": 0,
            "dark_streak_pixels": 0,
            "bright_flare_pixels": 0,
            "port_center": None,
            "status": "No metal detected (Normal CT anatomy)",
        }
        return empty_contours, stats, empty_masks

    num_l, labels, stats_l, centroids = cv2.connectedComponentsWithStats(cand_metal.astype(np.uint8))
    port_mask = np.zeros((h, w), dtype=bool)
    has_port = False

    for i in range(1, num_l):
        cx, cy = centroids[i]
        area = stats_l[i, cv2.CC_STAT_AREA]

        # Basic anatomical criteria: anterior-lateral chest wall
        if cy >= 0.62 * h:
            continue

        # If 3D anchor is provided, enforce spatial continuity within pectoral pocket
        if port_anchor is not None:
            anchor_y, anchor_x = port_anchor
            d_anchor = np.sqrt((cy - anchor_y) ** 2 + (cx - anchor_x) ** 2)
            if d_anchor > max_anchor_dist_px:
                continue

        # Size criterion: ChemoPort chamber or catheter segment
        if area >= 10 or (area >= 4 and np.max(hu_slice[labels == i]) >= 3500.0):
            port_mask |= (labels == i)
            has_port = True

    if not has_port:
        stats = {
            "metal_detected": False,
            "metal_pixel_count": 0,
            "artifact_pixel_count": 0,
            "dark_streak_pixels": 0,
            "bright_flare_pixels": 0,
            "port_center": None,
            "status": "No ChemoPort implant on this slice",
        }
        return empty_contours, stats, empty_masks

    # Compute port centroid
    port_ys, port_xs = np.where(port_mask)
    port_center = (float(np.mean(port_ys)), float(np.mean(port_xs)))
    port_pixel_count = int(np.sum(port_mask))

    # 3. Mediastinum Protection Zone (Heart and great vessels between lungs)
    med_zone = np.zeros((h, w), dtype=bool)
    if np.sum(priors.lung_mask) >= 500:
        lung_pts = np.argwhere(priors.lung_mask)
        min_r, max_r = np.min(lung_pts[:, 0]), np.max(lung_pts[:, 0])
        for r in range(min_r, max_r + 1):
            c_in_lung = np.where(priors.lung_mask[r, :])[0]
            if len(c_in_lung) >= 2:
                min_c, max_c = np.min(c_in_lung), np.max(c_in_lung)
                if max_c - min_c > 30:
                    med_zone[r, min_c + 15 : max_c - 15] = True
        med_zone &= priors.body_mask & (~priors.lung_mask) & (~priors.bone_mask)

    # 4. Peri-Port Influence Zone (Radial Corridor)
    dist_port = distance_transform_edt(~port_mask)
    peri_zone = (dist_port <= peri_zone_radius_px) & priors.body_mask & (~priors.lung_mask) & (~med_zone)
    dist_skin = cv2.distanceTransform(priors.body_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # 5. Physics-Informed Residual Anomaly Detection
    # 5A: Photon Starvation Dark Streaks
    dark_shadows = peri_zone & (~priors.lung_mask) & (~port_mask) & (
        (hu_slice < -140.0) | ((dist_skin >= 16.0) & (hu_slice < -50.0))
    )

    # 5B: Beam Hardening & Metal Scatter Bright Flares
    bright_flares = peri_zone & (~priors.bone_mask) & (~port_mask) & (
        (hu_slice > 135.0) & (hu_slice < 2000.0)
    )

    raw_artifact = (dark_shadows | bright_flares) & peri_zone & (~port_mask)

    # 6. Morphological Cleanup & Connectivity
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    art_closed = cv2.morphologyEx(raw_artifact.astype(np.uint8), cv2.MORPH_CLOSE, k_close).astype(bool)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    art_opened = cv2.morphologyEx(art_closed.astype(np.uint8), cv2.MORPH_OPEN, k_open).astype(bool)

    # Strictly protect anatomical organs
    art_opened &= peri_zone & (~port_mask) & (~priors.lung_mask) & (~priors.bone_mask)

    # Eliminate tiny isolated noise fragments (< 25 px)
    num_a, labels_a, stats_a, _ = cv2.connectedComponentsWithStats(art_opened.astype(np.uint8))
    final_art = np.zeros_like(art_opened)
    for i in range(1, num_a):
        if stats_a[i, cv2.CC_STAT_AREA] >= 25:
            final_art |= (labels_a == i)

    artifact_pixel_count = int(np.sum(final_art))
    dark_count = int(np.sum(final_art & dark_shadows))
    bright_count = int(np.sum(final_art & bright_flares))

    # 7. Extract Smooth Vector Polygon Contours
    # Port Contours (Rose Red)
    port_polys = []
    cnts_p, _ = cv2.findContours(port_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_p:
        if cv2.contourArea(c) >= 4 or len(c) >= 3:
            poly = cv2.approxPolyDP(c, epsilon=1.0, closed=True).reshape(-1, 2).tolist()
            if len(poly) >= 3:
                port_polys.append(poly)

    # Artifact Contours (Amber Yellow)
    art_polys = []
    cnts_a, _ = cv2.findContours(final_art.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_a:
        if cv2.contourArea(c) >= 12:
            poly = cv2.approxPolyDP(c, epsilon=1.2, closed=True).reshape(-1, 2).tolist()
            if len(poly) >= 3:
                art_polys.append(poly)

    contours = {
        "port": port_polys,
        "artifact": art_polys,
        "wire": [],
        "port_px": port_pixel_count,
        "art_px": artifact_pixel_count,
    }

    stats = {
        "metal_detected": True,
        "metal_pixel_count": port_pixel_count,
        "artifact_pixel_count": artifact_pixel_count,
        "dark_streak_pixels": dark_count,
        "bright_flare_pixels": bright_count,
        "port_center": port_center,
        "status": f"ChemoPort Detected ({port_pixel_count} px metal, {artifact_pixel_count} px artifact)",
    }

    masks = {
        "port_mask": port_mask,
        "artifact_mask": final_art,
    }

    return contours, stats, masks


def detect_series_inspection_contours(
    slices_list: List[Any],
) -> List[Tuple[Dict[str, List], Dict[str, Any], Dict[str, np.ndarray]]]:
    """Processes a full CT scan series to identify the 3D ChemoPort spatial anchor
    and generate consistent Approach A inspection contours for all slices."""
    if not slices_list:
        return []

    total_slices = len(slices_list)

    # Determine 3D ChemoPort spatial anchor if multi-slice
    port_anchor = None
    best_idx = None
    max_z_span = 25
    if total_slices > 1:
        from modules.reconstruction import find_chemoport_slice_index
        best_idx = find_chemoport_slice_index(slices_list)
        best_hu = slices_list[best_idx][1] if isinstance(slices_list[best_idx], (tuple, list)) else slices_list[best_idx]
        m_best = best_hu >= 2000.0
        if np.any(m_best):
            ys, xs = np.where(m_best)
            port_anchor = (float(np.mean(ys)), float(np.mean(xs)))

    results = []
    for idx, sl in enumerate(slices_list):
        # Slices far along the Z-axis from the ChemoPort anchor cannot contain the port
        if best_idx is not None and abs(idx - best_idx) > max_z_span:
            h, w = (sl[1] if isinstance(sl, (tuple, list)) else sl).shape
            empty_cnts = {"port": [], "artifact": [], "wire": [], "port_px": 0, "art_px": 0}
            empty_stats = {
                "metal_detected": False,
                "metal_pixel_count": 0,
                "artifact_pixel_count": 0,
                "dark_streak_pixels": 0,
                "bright_flare_pixels": 0,
                "port_center": None,
                "status": "Normal CT anatomy (Outside ChemoPort Z-range)",
            }
            empty_masks = {
                "port_mask": np.zeros((h, w), dtype=bool),
                "artifact_mask": np.zeros((h, w), dtype=bool),
            }
            results.append((empty_cnts, empty_stats, empty_masks))
            continue

        hu = sl[1] if isinstance(sl, (tuple, list)) else sl
        cnts, stats, masks = detect_approach_a_slice_contours(
            hu,
            port_anchor=port_anchor,
        )
        results.append((cnts, stats, masks))

    return results

