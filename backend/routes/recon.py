"""Reconstruction and HU Line Profile Extraction Routes."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import numpy as np

from modules.reconstruction import two_stage_chemoport_mar
from modules.deep_mar_model import blend_hybrid_mar
from backend.session_manager import session_manager

logger = logging.getLogger("chemoport.routes.recon")
router = APIRouter(prefix="/api/recon", tags=["recon"])


class ReconRequest(BaseModel):
    slice_idx: Optional[int] = None
    ai_weight: float = 0.5
    inpaint_strength: float = 0.7
    cad_preset: str = "universal"


@router.post("/{scan_id}/process")
async def run_reconstruction(scan_id: str, req: ReconRequest):
    """Executes physics-guided 3D CAD Prior MAR and DL inpainting on requested slice(s)."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")

    target_idx = req.slice_idx if req.slice_idx is not None else session.chemoport_slice_idx
    if not (0 <= target_idx < session.total_slices):
        raise HTTPException(status_code=400, detail="Invalid slice index")

    orig_hu = session.slices[target_idx][1]

    try:
        # Run 2-stage CAD prior MAR with optional AI blend
        recon_hu, stats = two_stage_chemoport_mar(
            orig_hu,
            pixel_spacing=session.pixel_spacing,
            cad_preset=req.cad_preset,
            inpaint_strength=req.inpaint_strength,
            ai_weight=req.ai_weight,
        )
        session.recon_cache[target_idx] = (recon_hu, stats)
        
        return {
            "status": "success",
            "slice_idx": target_idx,
            "stats": stats,
            "ai_weight": req.ai_weight,
        }
    except Exception as e:
        logger.error("Reconstruction error on slice %d: %s", target_idx, e)
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")


@router.get("/{scan_id}/slice/{slice_idx}/profile")
async def get_line_profile(scan_id: str, slice_idx: int, y_row: Optional[int] = None):
    """Extracts horizontal line profile across ChemoPort region for dosimetric verification."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")

    orig_hu = session.get_slice_hu(slice_idx)
    recon_hu = session.get_recon_hu(slice_idx)

    if orig_hu is None or recon_hu is None:
        raise HTTPException(status_code=400, detail="Slice not found")

    # If y_row not specified, find row with max metal density in ChemoPort zone
    if y_row is None:
        y_row = int(np.argmax(np.sum(orig_hu > 1500, axis=1)))
        if y_row == 0 or np.sum(orig_hu[y_row] > 1500) == 0:
            y_row = 180  # Default anterior chest wall row

    orig_line = orig_hu[y_row, :].tolist()
    recon_line = recon_hu[y_row, :].tolist()

    return {
        "slice_idx": slice_idx,
        "y_row": y_row,
        "orig_profile": orig_line,
        "recon_profile": recon_line,
        "diff_profile": (recon_hu[y_row, :] - orig_hu[y_row, :]).tolist(),
    }
