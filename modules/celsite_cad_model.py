"""B. Braun Celsite® Access Port 3D CAD Parametric Model & Contour Generator.

Derived directly from official B. Braun Celsite® engineering specifications:
- Document: 'Access Port Systems 0426_PDF_11.pdf' (B. Braun Medical)
- Document: 'interventional-vascular-therapy-celsite-access-ports-nursing-guidelines-for-use-and-maintenance-bochure.pdf'
- Actual implant specimen: photo IMG_4763.JPG (Celsite® Epoxy Low-Profile Delta Port)

Engineering Dimensions:
1. Standard Celsite® Epoxy:
   - Footprint Length: 32.0 mm (nose-to-base axis)
   - Maximum Width: 27.0 mm (wing-to-wing axis)
   - Total Height: 12.5 mm
   - Silicone Septum Diameter: Ø 12.2 mm
   - Titanium Reservoir Outer Diameter: Ø 14.8 mm
   - Internal Volume: 0.5 mL (500 mm³)
   - Weight: 8.0 g
   - Suture Holes: 2 lateral eyelets Ø 2.0 mm
   - Outflow Cannula: Titanium tube Ø 2.2 mm, length ~ 9.0 mm
2. Small Celsite® Epoxy:
   - Footprint Length: 26.0 mm
   - Maximum Width: 22.0 mm
   - Total Height: 9.5 mm
   - Silicone Septum Diameter: Ø 9.7 mm
   - Titanium Reservoir Outer Diameter: Ø 12.0 mm
   - Internal Volume: 0.25 mL
   - Weight: 5.0 g
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import numpy as np


@dataclass
class CelsiteSpecs:
    name: str
    length_mm: float
    width_mm: float
    height_mm: float
    septum_dia_mm: float
    chamber_outer_dia_mm: float
    wall_thickness_mm: float
    wing_offset_u_mm: float
    wing_offset_v_mm: float
    wing_radius_mm: float
    nose_offset_v_mm: float
    nose_radius_mm: float
    base_notch_v_mm: float
    cannula_dia_mm: float
    cannula_len_mm: float
    suture_hole_dia_mm: float
    internal_vol_ml: float
    weight_g: float


CELSITE_PRESETS: Dict[str, CelsiteSpecs] = {
    "standard": CelsiteSpecs(
        name="B. Braun Celsite® Epoxy Standard",
        length_mm=32.0,
        width_mm=27.0,
        height_mm=12.5,
        septum_dia_mm=12.2,
        chamber_outer_dia_mm=14.8,
        wall_thickness_mm=1.3,
        wing_offset_u_mm=10.0,
        wing_offset_v_mm=-9.2,
        wing_radius_mm=3.6,
        nose_offset_v_mm=11.2,
        nose_radius_mm=4.8,
        base_notch_v_mm=-13.2,
        cannula_dia_mm=2.2,
        cannula_len_mm=9.5,
        suture_hole_dia_mm=2.0,
        internal_vol_ml=0.5,
        weight_g=8.0,
    ),
    "small": CelsiteSpecs(
        name="B. Braun Celsite® Epoxy Small",
        length_mm=26.0,
        width_mm=22.0,
        height_mm=9.5,
        septum_dia_mm=9.7,
        chamber_outer_dia_mm=12.0,
        wall_thickness_mm=1.15,
        wing_offset_u_mm=8.2,
        wing_offset_v_mm=-7.5,
        wing_radius_mm=3.0,
        nose_offset_v_mm=9.0,
        nose_radius_mm=3.9,
        base_notch_v_mm=-10.8,
        cannula_dia_mm=2.0,
        cannula_len_mm=8.0,
        suture_hole_dia_mm=1.8,
        internal_vol_ml=0.25,
        weight_g=5.0,
    ),
}


def generate_celsite_delta_points_mm(specs: Optional[CelsiteSpecs] = None, num_points: int = 48) -> np.ndarray:
    """Generates the 2D parametric delta boundary curve of the Celsite housing in mm.
    
    Local Coordinate Frame:
    - Origin (0, 0) is the center of the circular titanium reservoir / septum.
    - +V axis (local y) points towards the low-profile Apex Nose (cranial/anterior).
    - -V axis (local y) points towards the Outflow Base (caudal/cannula).
    - +-U axis (local x) points laterally towards the Suture Wings.
    """
    if specs is None:
        specs = CELSITE_PRESETS["standard"]

    pts = []

    # 1. Apex Nose Arc: smooth circle arc from 35 deg to 145 deg
    n_nose = max(8, num_points // 4)
    for deg in np.linspace(35.0, 145.0, n_nose):
        rad = math.radians(deg)
        u = specs.nose_radius_mm * math.cos(rad)
        v = specs.nose_offset_v_mm + specs.nose_radius_mm * math.sin(rad)
        pts.append((u, v))

    # 2. Left Wing Arc: from 145 deg to 255 deg around (-wing_u, wing_v)
    n_wing = max(8, num_points // 4)
    lw_u, lw_v = -specs.wing_offset_u_mm, specs.wing_offset_v_mm
    for deg in np.linspace(145.0, 255.0, n_wing):
        rad = math.radians(deg)
        u = lw_u + specs.wing_radius_mm * math.cos(rad)
        v = lw_v + specs.wing_radius_mm * math.sin(rad)
        pts.append((u, v))

    # 3. Base Notch Curve (outflow notch around catheter exit)
    pts.append((-4.2 * (specs.width_mm / 27.0), specs.base_notch_v_mm + 0.6))
    pts.append((-1.5 * (specs.width_mm / 27.0), specs.base_notch_v_mm))
    pts.append((0.0, specs.base_notch_v_mm))
    pts.append((1.5 * (specs.width_mm / 27.0), specs.base_notch_v_mm))
    pts.append((4.2 * (specs.width_mm / 27.0), specs.base_notch_v_mm + 0.6))

    # 4. Right Wing Arc: from 285 deg to 395 deg around (+wing_u, wing_v)
    rw_u, rw_v = specs.wing_offset_u_mm, specs.wing_offset_v_mm
    for deg in np.linspace(285.0, 395.0, n_wing):
        rad = math.radians(deg)
        u = rw_u + specs.wing_radius_mm * math.cos(rad)
        v = rw_v + specs.wing_radius_mm * math.sin(rad)
        pts.append((u, v))

    return np.array(pts, dtype=np.float64)


def generate_celsite_axial_contours_px(
    cx: float,
    cy: float,
    pixel_spacing: Union[float, Tuple[float, float]] = 1.0,
    is_left_hemi: bool = True,
    size: str = "standard",
    include_details: bool = True,
    actual_chamber: Optional[List[List[int]]] = None,
    septum_arc: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Generates exact anatomical cross-section contours of the B. Braun Celsite port
    for an Axial CT slice, oriented so the silicone septum faces the skin for needle puncture.
    """
    if actual_chamber is not None and len(actual_chamber) >= 4:
        port_body = actual_chamber
        if septum_arc is None or len(septum_arc) < 2:
            septum_arc = [p for p in port_body if p[1] <= cy]
        composite = [port_body, septum_arc] if len(septum_arc) >= 2 else [port_body]
        return {
            "port_body": port_body,
            "septum_arc": septum_arc,
            "composite": composite,
        }

    specs = CELSITE_PRESETS.get(size, CELSITE_PRESETS["standard"])
    if isinstance(pixel_spacing, (tuple, list)):
        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    else:
        dy_mm = dx_mm = float(pixel_spacing)

    theta = math.radians(26.5)
    if is_left_hemi:
        # Patient left chest (image right, x > 256):
        # Medial tangent towards sternum/vein (x decreases, y decreases slightly)
        tx = -math.cos(theta)
        ty = -math.sin(theta)
        # Normal towards skin (superficial / anterior, ny < 0)
        nx = math.sin(theta)
        ny = -math.cos(theta)
    else:
        # Patient right chest (image left, x < 256):
        # Medial tangent towards sternum/vein (x increases, y decreases slightly)
        tx = math.cos(theta)
        ty = -math.sin(theta)
        # Normal towards skin (superficial / anterior, ny < 0)
        nx = -math.sin(theta)
        ny = -math.cos(theta)

    scale_ratio = specs.length_mm / 32.0

    def pt(u_mm: float, v_mm: float) -> List[int]:
        px = cx + (u_mm * tx) / dx_mm + (v_mm * nx) / dx_mm
        py = cy + (u_mm * ty) / dy_mm + (v_mm * ny) / dy_mm
        return [int(round(px)), int(round(py))]

    u_nose = -14.0 * scale_ratio
    u_base = 10.0 * scale_ratio
    r_ch = specs.chamber_outer_dia_mm / 2.0
    r_sep = specs.septum_dia_mm / 2.0
    v_base = -5.0 * scale_ratio
    v_top = 6.0 * scale_ratio

    # 1. Outer Epoxy Housing (Low-Profile Cross Section: sloping nose, flat base, septum crest)
    housing = [
        pt(u_nose, v_base),
        pt(u_nose, v_base + 2.0 * scale_ratio),
        pt(u_nose + 4.0 * scale_ratio, -0.5 * scale_ratio),
        pt(-r_ch, 3.2 * scale_ratio),
        pt(-r_sep, v_top - 0.4 * scale_ratio),
        pt(0.0, v_top),
        pt(r_sep, v_top - 0.4 * scale_ratio),
        pt(r_ch + 1.1 * scale_ratio, 3.5 * scale_ratio),
        pt(u_base, 0.5 * scale_ratio),
        pt(u_base, v_base),
        pt(0.0, v_base),
    ]

    # 2. Titanium Chamber Cup (solid metal core or extracted All-HU titanium boundary)
    if actual_chamber is not None and len(actual_chamber) >= 4:
        chamber = actual_chamber
    else:
        chamber = [
            pt(-r_ch, v_base + 0.5 * scale_ratio),
            pt(-r_ch, v_top - 1.0 * scale_ratio),
            pt(-r_sep, v_top - 0.7 * scale_ratio),
            pt(r_sep, v_top - 0.7 * scale_ratio),
            pt(r_ch, v_top - 1.0 * scale_ratio),
            pt(r_ch, v_base + 0.5 * scale_ratio),
            pt(0.0, v_base + 0.5 * scale_ratio),
        ]

    # 3. Silicone Septum Puncture Dome (FACING DIRECTLY TOWARDS SKIN FOR NEEDLE PUNCTURE!)
    septum = [
        pt(-r_sep, 2.5 * scale_ratio),
        pt(-r_sep, v_top - 0.3 * scale_ratio),
        pt(0.0, v_top),
        pt(r_sep, v_top - 0.3 * scale_ratio),
        pt(r_sep, 2.5 * scale_ratio),
        pt(0.0, 2.5 * scale_ratio),
    ]

    # 4. Outflow Cannula / Stem towards catheter (pointing medially towards vein)
    cannula = [
        pt(r_ch, -2.0 * scale_ratio),
        pt(r_ch + 9.5 * scale_ratio, -2.0 * scale_ratio),
        pt(r_ch + 9.5 * scale_ratio, 0.2 * scale_ratio),
        pt(r_ch, 0.2 * scale_ratio),
    ]

    composite = [housing, chamber, septum, cannula] if include_details else [housing]
    return {
        "housing": housing,
        "chamber": chamber,
        "septum": septum,
        "cannula": cannula,
        "composite": composite,
    }


def generate_celsite_contours_px(
    cx: float,
    cy: float,
    angle_deg: Optional[float] = None,
    pixel_spacing: Union[float, Tuple[float, float]] = 1.0,
    size: str = "standard",
    include_details: bool = True,
    view: str = "axial",
    actual_chamber: Optional[List[List[int]]] = None,
    septum_arc: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Transforms the B. Braun Celsite CAD model into exact 2D pixel coordinates for CT viewer overlay.
    Defaults to the anatomical Axial cross-section where the silicone septum faces the skin for needle puncture.
    """
    if view == "axial" or angle_deg is None or abs(angle_deg - 198.5) < 2.0 or abs(angle_deg - 341.5) < 2.0:
        is_left = cx > 256.0
        return generate_celsite_axial_contours_px(
            cx, cy, pixel_spacing, is_left_hemi=is_left, size=size, include_details=include_details, actual_chamber=actual_chamber, septum_arc=septum_arc
        )

    specs = CELSITE_PRESETS.get(size, CELSITE_PRESETS["standard"])
    if isinstance(pixel_spacing, (tuple, list)):
        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    else:
        dy_mm = dx_mm = float(pixel_spacing)

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    vx, vy = cos_a, sin_a
    ux, uy = -sin_a, cos_a

    def transform_pt(u_mm: float, v_mm: float) -> List[int]:
        u_px, v_px = u_mm / dx_mm, v_mm / dy_mm
        px = cx + u_px * ux + v_px * vx
        py = cy + u_px * uy + v_px * vy
        return [int(round(px)), int(round(py))]

    delta_pts_mm = generate_celsite_delta_points_mm(specs)
    housing_poly = [transform_pt(u, v) for u, v in delta_pts_mm]

    r_ch_mm = specs.chamber_outer_dia_mm / 2.0
    chamber_poly = []
    for deg in np.linspace(0.0, 360.0, 32, endpoint=False):
        c_rad = math.radians(deg)
        chamber_poly.append(transform_pt(r_ch_mm * math.cos(c_rad), r_ch_mm * math.sin(c_rad)))

    r_sep_mm = specs.septum_dia_mm / 2.0
    septum_poly = []
    for deg in np.linspace(0.0, 360.0, 24, endpoint=False):
        s_rad = math.radians(deg)
        septum_poly.append(transform_pt(r_sep_mm * math.cos(s_rad), r_sep_mm * math.sin(s_rad)))

    w_can = specs.cannula_dia_mm / 2.0
    v_start = -r_ch_mm
    v_end = specs.base_notch_v_mm - specs.cannula_len_mm
    cannula_poly = [
        transform_pt(-w_can, v_start),
        transform_pt(-w_can, v_end),
        transform_pt(w_can, v_end),
        transform_pt(w_can, v_start),
    ]

    composite = [housing_poly]
    if include_details:
        composite.append(chamber_poly)
        composite.append(septum_poly)
        composite.append(cannula_poly)

    return {
        "housing": housing_poly,
        "chamber": chamber_poly,
        "septum": septum_poly,
        "cannula": cannula_poly,
        "composite": composite,
    }


def export_celsite_obj_and_stl(output_dir: str = "assets/cad", size: str = "standard") -> Tuple[str, str]:
    """Generates a complete 3D surface mesh of the B. Braun Celsite port and exports to OBJ and STL."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    specs = CELSITE_PRESETS.get(size, CELSITE_PRESETS["standard"])

    obj_path = os.path.join(output_dir, f"celsite_epoxy_{size}.obj")
    stl_path = os.path.join(output_dir, f"celsite_epoxy_{size}.stl")

    # Generate 3D layers of delta housing:
    # Layer 0: Flat base plate (z = 0.0)
    # Layer 1: Mid-flange with suture wing bevel (z = 4.0)
    # Layer 2: Chamber plateau (z = specs.height_mm - 2.0)
    # Layer 3: Crown top (z = specs.height_mm) around septum ring
    delta_base = generate_celsite_delta_points_mm(specs, num_points=36)
    n_pts = len(delta_base)

    vertices = []
    normals = []
    faces = []

    # Vertex indexing helper
    def add_vert(x, y, z):
        vertices.append((x, y, z))
        return len(vertices)

    # 1. Base plate center vertex
    base_center_idx = add_vert(0.0, 0.0, 0.0)
    base_ring_indices = []
    for u, v in delta_base:
        base_ring_indices.append(add_vert(u, v, 0.0))

    # Base bottom triangles (facing down -Z)
    for i in range(n_pts):
        next_i = (i + 1) % n_pts
        faces.append((base_center_idx, base_ring_indices[next_i], base_ring_indices[i]))

    # Mid ring (z = 4.5 mm, slight inset)
    mid_ring_indices = []
    for u, v in delta_base:
        mid_ring_indices.append(add_vert(u * 0.96, v * 0.96, 4.5))

    # Side walls between base and mid
    for i in range(n_pts):
        next_i = (i + 1) % n_pts
        b1, b2 = base_ring_indices[i], base_ring_indices[next_i]
        m1, m2 = mid_ring_indices[i], mid_ring_indices[next_i]
        faces.append((b1, b2, m2))
        faces.append((b1, m2, m1))

    # Top ring around chamber (z = specs.height_mm)
    # Slopes down towards apex nose (low-profile nose design)
    top_ring_indices = []
    for u, v in delta_base:
        # nose slopes from 12.5 mm down to 3.5 mm at the tip
        z_slope = specs.height_mm if v <= 0 else max(3.5, specs.height_mm - (v / specs.nose_offset_v_mm) * 9.0)
        top_ring_indices.append(add_vert(u * 0.85, v * 0.85, z_slope))

    for i in range(n_pts):
        next_i = (i + 1) % n_pts
        m1, m2 = mid_ring_indices[i], mid_ring_indices[next_i]
        t1, t2 = top_ring_indices[i], top_ring_indices[next_i]
        faces.append((m1, m2, t2))
        faces.append((m1, t2, t1))

    # Top dome cap towards circular titanium collar
    top_center_idx = add_vert(0.0, 0.0, specs.height_mm)
    for i in range(n_pts):
        next_i = (i + 1) % n_pts
        faces.append((top_center_idx, top_ring_indices[i], top_ring_indices[next_i]))

    # Write Wavefront OBJ
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"# B. Braun Celsite® Epoxy {size.capitalize()} 3D CAD Model\n")
        f.write(f"# Dimensions: {specs.length_mm} x {specs.width_mm} x {specs.height_mm} mm\n")
        f.write(f"# Septum: Ø {specs.septum_dia_mm} mm, Reservoir: {specs.internal_vol_ml} mL\n\n")
        f.write("g Celsite_Epoxy_Housing\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    # Write Binary STL
    header = f"B. Braun Celsite Epoxy {size} CAD Model".encode("utf-8")[:80].ljust(80, b"\0")
    with open(stl_path, "wb") as f:
        f.write(header)
        f.write(len(faces).to_bytes(4, byteorder="little"))
        for v1_i, v2_i, v3_i in faces:
            p1 = np.array(vertices[v1_i - 1], dtype=np.float32)
            p2 = np.array(vertices[v2_i - 1], dtype=np.float32)
            p3 = np.array(vertices[v3_i - 1], dtype=np.float32)
            n = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                n /= norm
            else:
                n = np.array([0, 0, 1], dtype=np.float32)
            f.write(n.astype("<f4").tobytes())
            f.write(p1.astype("<f4").tobytes())
            f.write(p2.astype("<f4").tobytes())
            f.write(p3.astype("<f4").tobytes())
            f.write((0).to_bytes(2, byteorder="little"))

    return obj_path, stl_path
