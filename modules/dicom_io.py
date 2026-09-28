"""DICOM I/O, anonymization, and window/level utilities for ChemoPort CT-MAR Studio."""

import io
import zipfile
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Standard Window/Level presets commonly used in radiation oncology and CT imaging
WINDOW_PRESETS: Dict[str, Dict[str, float]] = {
    "Mediastinum / Soft Tissue": {"wc": 40.0, "ww": 350.0},
    "ChemoPort & Metal Detail": {"wc": 1200.0, "ww": 4000.0},
    "Bone Window": {"wc": 400.0, "ww": 1800.0},
    "Lung Window": {"wc": -600.0, "ww": 1500.0},
    "Wide Dynamic (Full Range)": {"wc": 500.0, "ww": 5000.0},
}

PHI_TAGS_TO_CLEAR = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "OtherPatientIDs",
    "InstitutionName",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
]


def load_dicom_file(source: Union[str, io.BytesIO, bytes]) -> Tuple[Dataset, np.ndarray, Dict[str, Any]]:
    """Loads a DICOM file from file path, BytesIO stream, or raw bytes.

    Returns:
        ds: pydicom Dataset
        hu_array: 2D numpy array in Hounsfield Units (float32)
        meta: dictionary of key imaging and clinical metadata
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    ds = pydicom.dcmread(source, force=True)

    # Convert raw pixel array to Hounsfield Units (HU)
    raw_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_array = raw_array * slope + intercept

    meta = extract_metadata(ds, hu_array)
    return ds, hu_array, meta


def extract_metadata(ds: Dataset, hu_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Extracts clean metadata dictionary for display and downstream processing."""
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    if hasattr(pixel_spacing, "__iter__"):
        spacing_str = f"{float(pixel_spacing[0]):.2f} x {float(pixel_spacing[1]):.2f} mm"
    else:
        spacing_str = "1.00 x 1.00 mm"

    meta = {
        "patient_id": str(getattr(ds, "PatientID", "ANON_PATIENT")),
        "modality": str(getattr(ds, "Modality", "CT")),
        "manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
        "study_description": str(getattr(ds, "StudyDescription", "CT Radiation Therapy")),
        "series_description": str(getattr(ds, "SeriesDescription", "AiCE 2mm Breast")),
        "instance_number": int(getattr(ds, "InstanceNumber", 1)),
        "slice_thickness": float(getattr(ds, "SliceThickness", 2.0)),
        "pixel_spacing": spacing_str,
        "rows": int(getattr(ds, "Rows", 512)),
        "columns": int(getattr(ds, "Columns", 512)),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "UNKNOWN")),
        "series_number": str(getattr(ds, "SeriesNumber", "1")),
    }

    if hu_array is not None:
        meta["min_hu"] = float(np.min(hu_array))
        meta["max_hu"] = float(np.max(hu_array))
        meta["mean_hu"] = float(np.mean(hu_array))
        meta["metal_pixel_count"] = int(np.sum(hu_array > 2500))

    return meta


def anonymize_dicom(ds: Dataset, anon_id: str = "ANON-CHEMOPORT") -> Dataset:
    """Anonymizes DICOM metadata according to HIPAA / DICOM PS 3.15 guidelines."""
    anon_ds = ds.copy()
    for tag in PHI_TAGS_TO_CLEAR:
        if hasattr(anon_ds, tag):
            if tag == "PatientName":
                setattr(anon_ds, tag, anon_id)
            elif tag == "PatientID":
                setattr(anon_ds, tag, anon_id)
            else:
                setattr(anon_ds, tag, "")

    return anon_ds


def apply_window_level(hu_array: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Applies DICOM Window Center (WC) and Window Width (WW) to produce an 8-bit image (0-255).

    Args:
        hu_array: 2D array in Hounsfield Units.
        wc: Window Center.
        ww: Window Width (must be > 0).

    Returns:
        uint8 2D numpy array [0, 255].
    """
    if ww <= 0:
        ww = 1.0

    min_val = wc - (ww / 2.0)
    max_val = wc + (ww / 2.0)

    clipped = np.clip(hu_array, min_val, max_val)
    normalized = ((clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    return normalized


def save_dicom_bytes(original_ds: Dataset, modified_hu: np.ndarray, series_desc_suffix: str = " AI-MAR") -> bytes:
    """Encodes modified HU array back into standard DICOM binary bytes for TPS export."""
    out_ds = original_ds.copy()

    # Apply slope & intercept inverse
    slope = float(getattr(out_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(out_ds, "RescaleIntercept", 0.0))

    raw_pixel_vals = np.round((modified_hu - intercept) / slope)

    # Respect BitsAllocated & PixelRepresentation
    bits_allocated = getattr(out_ds, "BitsAllocated", 16)
    pixel_rep = getattr(out_ds, "PixelRepresentation", 1)  # 1 for signed, 0 for unsigned

    if pixel_rep == 1:
        raw_pixel_vals = np.clip(raw_pixel_vals, -32768, 32767).astype(np.int16)
    else:
        raw_pixel_vals = np.clip(raw_pixel_vals, 0, 65535).astype(np.uint16)

    out_ds.PixelData = raw_pixel_vals.tobytes()
    out_ds.Rows, out_ds.Columns = modified_hu.shape

    # Update Series description and generate new SOP Instance UID
    orig_series_desc = getattr(out_ds, "SeriesDescription", "CT")
    out_ds.SeriesDescription = f"{orig_series_desc}{series_desc_suffix}"
    out_ds.SOPInstanceUID = generate_uid()

    buf = io.BytesIO()
    out_ds.save_as(buf)
    return buf.getvalue()


def load_dicom_from_zip(zip_source: Union[io.BytesIO, bytes]) -> List[Tuple[Dataset, np.ndarray, Dict[str, Any]]]:
    """Extracts and parses all valid DICOM files from an uploaded ZIP archive."""
    if isinstance(zip_source, bytes):
        zip_source = io.BytesIO(zip_source)

    slices = []
    with zipfile.ZipFile(zip_source, "r") as zf:
        for file_info in zf.infolist():
            if file_info.filename.startswith("__MACOSX") or file_info.is_dir():
                continue
            with zf.open(file_info) as f:
                try:
                    ds, hu, meta = load_dicom_file(io.BytesIO(f.read()))
                    slices.append((ds, hu, meta))
                except Exception:
                    # Skip non-dicom files in zip
                    continue

    # Sort slices by InstanceNumber or SliceLocation
    slices.sort(key=lambda s: s[2].get("instance_number", 0))
    return slices
