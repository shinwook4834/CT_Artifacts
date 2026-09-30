"""Physics-Informed Anatomical Residual AI for Step 2 Artifact Inspection.

Approach A (Physics-Informed Anatomical Residual AI):
1. ChemoPort Titanium Core & Catheter Assembly Segmentation:
   - Titanium core threshold (HU >= 1800.0 HU) and subcutaneous port casing/connector (HU >= 220.0 HU)
   - Anatomical anterior-lateral chest wall localization (Y < 0.62 * H)
   - 3D spatial continuity tracking across the scan to exclude dental fillings and pelvic clips
   - Accurate 3D extent capturing the full cranio-caudal span (including top/bottom casing slices such as slices 115 and 135)
   - Rose Red vector contour extraction for the port body, casing, and catheter
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
    max_anchor_dist_px: float = 60.0,
    peri_zone_radius_px: float = 90.0,
    is_transition_slice: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """Segments ChemoPort metal and radiating streak artifacts for a single 2D CT slice.

    Returns:
        contours: {"port": [...], "artifact": [...], "wire": [], "port_px": int, "art_px": int}
        stats: Dictionary of inspection metrics
        masks: {"port_mask": bool_array, "artifact_mask": bool_array}
    """
    h, w = hu_slice.shape
    empty_contours = {"port": [], "artifact": [], "wire": [], "port_px": 0, "art_px": 0}
    empty_masks = {
        "port_mask": np.zeros((h, w), dtype=bool),
        "artifact_mask": np.zeros((h, w), dtype=bool),
    }

    # Fast screening
    if port_anchor is not None:
        anchor_y, anchor_x = port_anchor
        y_min, y_max = max(0, int(anchor_y - 75)), min(h, int(anchor_y + 75))
        x_min, x_max = max(0, int(anchor_x - 75)), min(w, int(anchor_x + 75))
        roi = hu_slice[y_min:y_max, x_min:x_max]
        if not (np.any(roi >= 220.0) or np.any(roi < -140.0)):
            stats = {
                "metal_detected": False,
                "metal_pixel_count": 0,
                "artifact_pixel_count": 0,
                "dark_streak_pixels": 0,
                "bright_flare_pixels": 0,
                "port_center": None,
                "status": "No metal or artifact detected (Normal CT anatomy)",
            }
            return empty_contours, stats, empty_masks
    else:
        if np.max(hu_slice) < 1400.0:
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

    # 2. ChemoPort Titanium Core & Catheter Assembly Detection
    cand_metal = (hu_slice >= 1800.0) & priors.body_mask & (~priors.lung_mask)
    num_l, labels, stats_l, centroids = cv2.connectedComponentsWithStats(cand_metal.astype(np.uint8))
    port_mask = np.zeros((h, w), dtype=bool)
    has_port = False

    for i in range(1, num_l):
        cx, cy = centroids[i]
        area = stats_l[i, cv2.CC_STAT_AREA]

        # Anterior chest wall constraint
        if cy >= 0.62 * h:
            continue

        if port_anchor is not None:
            anchor_y, anchor_x = port_anchor
            d_anchor = np.sqrt((cy - anchor_y) ** 2 + (cx - anchor_x) ** 2)
            if d_anchor > max_anchor_dist_px:
                continue

        if area >= 4:
            port_mask |= (labels == i)
            has_port = True

    # If titanium core (>=1800 HU) is absent on this boundary slice,
    # detect port casing / catheter assembly (HU >= 220 in the non-bone subcutaneous pectoral pocket)
    # ONLY if this slice is an immediate transition slice adjacent to confirmed titanium metal!
    if not has_port and port_anchor is not None and is_transition_slice:
        anchor_y, anchor_x = port_anchor
        Y, X = np.ogrid[:h, :w]
        dist_anc = np.sqrt((Y - anchor_y) ** 2 + (X - anchor_x) ** 2)
        pocket_casing = (dist_anc <= 30.0) & priors.body_mask & (~priors.lung_mask) & (~priors.bone_mask) & (hu_slice >= 220.0)
        num_c, lbls_c, stats_c, _ = cv2.connectedComponentsWithStats(pocket_casing.astype(np.uint8))
        for i in range(1, num_c):
            if stats_c[i, cv2.CC_STAT_AREA] >= 8:
                port_mask |= (lbls_c == i)
                has_port = True

    # If this slice has no ChemoPort metal or casing, it cannot produce metal artifacts.
    # Return empty contours immediately to prevent false positives on normal anatomy.
    if not has_port:
        stats = {
            "metal_detected": False,
            "metal_pixel_count": 0,
            "artifact_pixel_count": 0,
            "dark_streak_pixels": 0,
            "bright_flare_pixels": 0,
            "port_center": None,
            "status": "Normal CT anatomy",
        }
        return empty_contours, stats, empty_masks

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

    # 4. Peri-Port Influence Zone (Radial Corridor & Thoracic Reach)
    Y, X = np.ogrid[:h, :w]
    if has_port and np.any(port_mask):
        dist_zone = distance_transform_edt(~port_mask)
        num_cp, lbls_cp, stats_cp, centroids_cp = cv2.connectedComponentsWithStats(port_mask.astype(np.uint8))
        if num_cp > 1:
            main_c_idx = 1 + np.argmax(stats_cp[1:, cv2.CC_STAT_AREA])
            port_center = (float(centroids_cp[main_c_idx][1]), float(centroids_cp[main_c_idx][0]))
        else:
            port_ys, port_xs = np.where(port_mask)
            port_center = (float(np.mean(port_ys)), float(np.mean(port_xs)))
        port_pixel_count = int(np.sum(port_mask))
    elif port_anchor is not None:
        anchor_y, anchor_x = port_anchor
        dist_zone = np.sqrt((Y - anchor_y) ** 2 + (X - anchor_x) ** 2)
        port_center = port_anchor
        port_pixel_count = 0
    else:
        dist_zone = np.zeros((h, w), dtype=np.float32)
        port_center = None
        port_pixel_count = 0

    dist_skin = cv2.distanceTransform(priors.body_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # 5. Physics-Informed Streak and Flare Anomaly Detection
    # 5A: Peri-port Subcutaneous & Pectoral Zone
    peri_zone = (dist_zone <= peri_zone_radius_px) & priors.body_mask & (~priors.lung_mask) & (~med_zone) & (dist_skin >= 3)
    non_med_dark = peri_zone & (~port_mask) & (
        (hu_slice < -140.0) | ((dist_skin >= 16.0) & (hu_slice < -50.0))
    )
    non_med_bright = peri_zone & (~priors.bone_mask) & (~port_mask) & (
        (hu_slice > 135.0) & (hu_slice < 1800.0)
    )

    # 5B: Mediastinum / Heart Radial Streak Detection (Physics-Informed Polar Ray Decomposition)
    # Metal projection streaks travel along straight lines radiating from the port center.
    # In polar coordinates (r, theta) centered at port_center, radial streak lines become horizontal lines.
    med_streak_dark = np.zeros((h, w), dtype=bool)
    med_streak_bright = np.zeros((h, w), dtype=bool)

    if (has_port or port_anchor is not None) and port_center is not None:
        py, px = port_center
        max_polar_radius = int(min(240, max(h, w)))
        clean_hu = np.nan_to_num(hu_slice, nan=-1000.0, posinf=3000.0, neginf=-1000.0)

        # Forward polar transform (360 angles, max_polar_radius radius)
        polar = cv2.warpPolar(clean_hu, (max_polar_radius, 360), (px, py), max_polar_radius, cv2.WARP_POLAR_LINEAR)
        polar = np.nan_to_num(polar, nan=-1000.0)
        pad = 20
        polar_padded = np.pad(polar, ((pad, pad), (0, 0)), mode="wrap")
        smooth = cv2.blur(polar_padded, (1, 21))[pad:-pad, :]
        residual = polar - smooth

        # Radial streaks are horizontal lines in polar space with slight angular width (height 2 rows)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))

        polar_dark = ((residual < -12.0) & (polar < 35.0)).astype(np.uint8)
        polar_dark_h = cv2.morphologyEx(polar_dark, cv2.MORPH_OPEN, k_h).astype(bool)

        polar_bright = ((residual > 12.0) & (polar > 50.0)).astype(np.uint8)
        polar_bright_h = cv2.morphologyEx(polar_bright, cv2.MORPH_OPEN, k_h).astype(bool)

        # Bridge small radial gaps along ray projection
        k_rad = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        polar_dark_c = cv2.morphologyEx(polar_dark_h.astype(np.uint8), cv2.MORPH_CLOSE, k_rad).astype(bool)
        polar_bright_c = cv2.morphologyEx(polar_bright_h.astype(np.uint8), cv2.MORPH_CLOSE, k_rad).astype(bool)

        # Inverse polar transform back to CT Cartesian grid
        inv_dark = cv2.warpPolar(
            polar_dark_c.astype(np.float32), (w, h), (px, py), max_polar_radius,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
        ) > 0.2
        inv_bright = cv2.warpPolar(
            polar_bright_c.astype(np.float32), (w, h), (px, py), max_polar_radius,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
        ) > 0.2

        med_corridor = med_zone & (dist_zone <= 220.0) & (dist_skin >= 8)
        med_streak_dark = med_corridor & inv_dark & (~port_mask) & (~priors.lung_mask) & (~priors.bone_mask)
        med_streak_bright = med_corridor & inv_bright & (~port_mask) & (~priors.lung_mask) & (~priors.bone_mask)

    dark_shadows = non_med_dark | med_streak_dark
    bright_flares = non_med_bright | med_streak_bright
    raw_artifact = dark_shadows | bright_flares

    # 6. Morphological Cleanup & Connectivity
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    art_closed = cv2.morphologyEx(raw_artifact.astype(np.uint8), cv2.MORPH_CLOSE, k_close).astype(bool)
    art_closed &= priors.body_mask & (~priors.lung_mask) & (~priors.bone_mask) & (~port_mask) & (dist_skin >= 2)

    # Eliminate tiny isolated noise fragments (< 25 px)
    num_a, labels_a, stats_a, _ = cv2.connectedComponentsWithStats(art_closed.astype(np.uint8))
    final_art = np.zeros_like(art_closed)
    for i in range(1, num_a):
        if stats_a[i, cv2.CC_STAT_AREA] >= 25:
            final_art |= (labels_a == i)

    artifact_pixel_count = int(np.sum(final_art))
    dark_count = int(np.sum(final_art & dark_shadows))
    bright_count = int(np.sum(final_art & bright_flares))

    # If neither port nor artifacts exist on this slice, return empty
    if not has_port and artifact_pixel_count == 0:
        stats = {
            "metal_detected": False,
            "metal_pixel_count": 0,
            "artifact_pixel_count": 0,
            "dark_streak_pixels": 0,
            "bright_flare_pixels": 0,
            "port_center": None,
            "status": "Normal CT anatomy",
        }
        return empty_contours, stats, empty_masks

    # 7. Extract Smooth Vector Polygon Contours
    # Port Contours (Rose Red)
    port_polys = []
    if has_port and np.any(port_mask):
        cnts_p, _ = cv2.findContours(port_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_p:
            if cv2.contourArea(c) >= 4 or len(c) >= 3:
                poly = cv2.approxPolyDP(c, epsilon=1.0, closed=True).reshape(-1, 2).tolist()
                if len(poly) >= 3:
                    port_polys.append(poly)

    # Artifact Contours (Amber Yellow)
    art_polys = []
    if artifact_pixel_count > 0:
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

    if has_port and artifact_pixel_count > 0:
        status_msg = f"ChemoPort & Artifacts ({port_pixel_count} px port, {artifact_pixel_count} px artifact)"
    elif has_port:
        status_msg = f"ChemoPort Assembly ({port_pixel_count} px)"
    else:
        status_msg = f"Streak Artifacts ({artifact_pixel_count} px)"

    stats = {
        "metal_detected": has_port,
        "metal_pixel_count": port_pixel_count,
        "artifact_pixel_count": artifact_pixel_count,
        "dark_streak_pixels": dark_count,
        "bright_flare_pixels": bright_count,
        "port_center": port_center,
        "status": status_msg,
    }

    masks = {
        "port_mask": port_mask,
        "artifact_mask": final_art,
    }

    return contours, stats, masks


def detect_series_inspection_contours(
    slices_list: List[Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]]:
    """Processes a full CT scan series to identify the 3D ChemoPort spatial anchor
    and generate consistent Approach A inspection contours for all slices."""
    if not slices_list:
        return []

    total_slices = len(slices_list)

    # Determine 3D ChemoPort spatial anchor and contiguous physical Z-range
    port_anchor = None
    best_idx = None
    z_min_active = 0
    z_max_active = total_slices - 1

    if total_slices > 1:
        from modules.reconstruction import find_chemoport_slice_index
        best_idx = find_chemoport_slice_index(slices_list)
        best_hu = slices_list[best_idx][1] if isinstance(slices_list[best_idx], (tuple, list)) else slices_list[best_idx]
        m_best = best_hu >= 2000.0
        if np.any(m_best):
            ys, xs = np.where(m_best)
            port_anchor = (float(np.mean(ys)), float(np.mean(xs)))

            # Contiguously trace the exact 3D physical body of the ChemoPort along the Z-axis
            z_min_metal = best_idx
            while z_min_metal > 0:
                hu_prev = slices_list[z_min_metal - 1][1] if isinstance(slices_list[z_min_metal - 1], (tuple, list)) else slices_list[z_min_metal - 1]
                m_prev = hu_prev >= 1800.0
                if np.any(m_prev):
                    ys_p, xs_p = np.where(m_prev)
                    d = np.sqrt((ys_p - port_anchor[0]) ** 2 + (xs_p - port_anchor[1]) ** 2)
                    if np.sum(d <= 45.0) >= 4:
                        z_min_metal -= 1
                        continue
                break

            z_max_metal = best_idx
            while z_max_metal < total_slices - 1:
                hu_next = slices_list[z_max_metal + 1][1] if isinstance(slices_list[z_max_metal + 1], (tuple, list)) else slices_list[z_max_metal + 1]
                m_next = hu_next >= 1800.0
                if np.any(m_next):
                    ys_n, xs_n = np.where(m_next)
                    d = np.sqrt((ys_n - port_anchor[0]) ** 2 + (xs_n - port_anchor[1]) ** 2)
                    if np.sum(d <= 45.0) >= 4:
                        z_max_metal += 1
                        continue
                break

            # Allow at most 1 transition slice above and below for casing / catheter loop
            z_min_active = max(0, z_min_metal - 1)
            z_max_active = min(total_slices - 1, z_max_metal + 1)

    results = []
    for idx, sl in enumerate(slices_list):
        # Slices outside the 3D ChemoPort physical Z-span cannot contain the port or metal artifacts
        if idx < z_min_active or idx > z_max_active:
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
        is_trans = (idx == z_min_active or idx == z_max_active)
        cnts, stats, masks = detect_approach_a_slice_contours(
            hu,
            port_anchor=port_anchor,
            is_transition_slice=is_trans,
        )
        results.append((cnts, stats, masks))

    return results
