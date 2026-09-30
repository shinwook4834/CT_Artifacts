"""DICOM Upload, Volume Parsing, and Raw Slice Streaming Routes."""

from __future__ import annotations

import io
import zipfile
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse

import numpy as np

from modules.dicom_io import load_dicom_file, load_dicom_from_zip
from modules.reconstruction import find_chemoport_slice_index
from modules.artifact_inspection import detect_series_inspection_contours
from backend.session_manager import session_manager

logger = logging.getLogger("chemoport.routes.scan")
router = APIRouter(prefix="/api/scan", tags=["scan"])


@router.post("/upload")
async def upload_dicom_scan(files: List[UploadFile] = File(...)):
    """Uploads single/multiple DICOM files or ZIP archive, parses metadata,
    localizes ChemoPort, and caches the 3D volume in 16GB server RAM."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_slices = []

    for f in files:
        content = await f.read()
        f_name = f.filename.lower() if f.filename else ""
        
        if f_name.endswith(".zip") or (len(content) > 4 and content[:4] == b"PK\x03\x04"):
            try:
                zip_slices = load_dicom_from_zip(content)
                all_slices.extend(zip_slices)
            except Exception as e:
                logger.error("Error reading zip file %s: %s", f.filename, e)
        else:
            try:
                ds, hu_arr, meta = load_dicom_file(content)
                all_slices.append((ds, hu_arr, meta))
            except Exception as e:
                # Skip non-dicom files gracefully
                pass

    if not all_slices:
        raise HTTPException(
            status_code=400,
            detail="No valid CT DICOM slices could be parsed from the uploaded files."
        )

    # Detect multi-series if present
    series_map = {}
    for sl in all_slices:
        ds_i, _, meta_i = sl
        suid = meta_i.get("series_instance_uid", getattr(ds_i, "SeriesInstanceUID", "UNKNOWN"))
        if suid not in series_map:
            series_map[suid] = []
        series_map[suid].append(sl)

    # Select the primary or most relevant series (highest metal pixel count)
    if len(series_map) > 1:
        best_suid = max(
            series_map.keys(),
            key=lambda uid: sum(s[2].get("metal_pixel_count", 0) for s in series_map[uid])
        )
        selected_slices = series_map[best_suid]
    else:
        selected_slices = all_slices

    # Create session
    session = session_manager.create_session(selected_slices)

    # Localize ChemoPort slice
    best_slice_idx = find_chemoport_slice_index(session.slices)
    session.chemoport_slice_idx = best_slice_idx

    # Extract artifact inspection contours
    try:
        inspection_results = detect_series_inspection_contours(session.slices)
        contours_list = []
        for idx, (cnt_dict, stats, _) in enumerate(inspection_results):
            contours_list.append(cnt_dict)
            # Default zero-correction recon
            session.recon_cache[idx] = (session.slices[idx][1].copy(), stats)
        session.contours_cache = contours_list
    except Exception as e:
        logger.warning("Contour extraction fallback: %s", e)
        session.contours_cache = [{"port": [], "art": [], "port_px": 0, "art_px": 0} for _ in range(session.total_slices)]

    return JSONResponse({
        "status": "success",
        "scan_id": session.scan_id,
        "total_slices": session.total_slices,
        "chemoport_slice_idx": session.chemoport_slice_idx,
        "series_description": session.series_description,
        "pixel_spacing": session.pixel_spacing,
        "slice_thickness": session.slice_thickness,
        "window_defaults": {
            "center": float(session.window_center),
            "width": float(session.window_width),
        },
        "contours": session.contours_cache,
    })


@router.get("/{scan_id}/info")
async def get_scan_info(scan_id: str):
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")
    
    return {
        "scan_id": session.scan_id,
        "total_slices": session.total_slices,
        "chemoport_slice_idx": session.chemoport_slice_idx,
        "series_description": session.series_description,
        "pixel_spacing": session.pixel_spacing,
        "slice_thickness": session.slice_thickness,
        "window_defaults": {
            "center": float(session.window_center),
            "width": float(session.window_width),
        }
    }


@router.get("/{scan_id}/slice/{slice_idx}/raw")
async def get_raw_slice_buffer(scan_id: str, slice_idx: int):
    """Streams 512x512 16-bit signed integer HU buffer as binary stream.
    Total size = 512 * 512 * 2 = 524,288 bytes (~512KB)."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")
    
    hu_arr = session.get_slice_hu(slice_idx)
    if hu_arr is None:
        raise HTTPException(status_code=400, detail="Invalid slice index")

    # Clip to standard int16 range [-32768, 32767] and convert
    int16_buf = np.clip(hu_arr, -32768, 32767).astype(np.int16).tobytes()
    
    return Response(
        content=int16_buf,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Length": str(len(int16_buf)),
            "X-Slice-Index": str(slice_idx),
            "X-Slice-Width": str(hu_arr.shape[1]),
            "X-Slice-Height": str(hu_arr.shape[0]),
        }
    )


@router.get("/{scan_id}/slice/{slice_idx}/recon_raw")
async def get_recon_slice_buffer(scan_id: str, slice_idx: int):
    """Streams 512x512 16-bit signed integer reconstructed HU buffer as binary stream."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")
    
    recon_hu = session.get_recon_hu(slice_idx)
    if recon_hu is None:
        raise HTTPException(status_code=400, detail="Invalid slice index")

    int16_buf = np.clip(recon_hu, -32768, 32767).astype(np.int16).tobytes()
    
    return Response(
        content=int16_buf,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Length": str(len(int16_buf)),
            "X-Slice-Index": str(slice_idx),
        }
    )


@router.get("/{scan_id}/slice/{slice_idx}/contours")
async def get_slice_contours(scan_id: str, slice_idx: int):
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")
    
    if 0 <= slice_idx < len(session.contours_cache):
        return session.contours_cache[slice_idx]
    
    return {"port": [], "art": [], "port_px": 0, "art_px": 0}
