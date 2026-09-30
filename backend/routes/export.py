"""DICOM and ZIP Export Routes."""

from __future__ import annotations

import io
import zipfile
import logging
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from modules.dicom_io import save_dicom_bytes, anonymize_dicom
from backend.session_manager import session_manager

logger = logging.getLogger("chemoport.routes.export")
router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/{scan_id}/slice/{slice_idx}/dicom")
async def export_single_slice_dicom(scan_id: str, slice_idx: int, anonymize: bool = False):
    """Exports single reconstructed DICOM slice with new SOPInstanceUID."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")

    if not (0 <= slice_idx < session.total_slices):
        raise HTTPException(status_code=400, detail="Invalid slice index")

    ds_orig, _, _ = session.slices[slice_idx]
    recon_hu = session.get_recon_hu(slice_idx)

    # Clone dataset for export
    import copy
    out_ds = copy.deepcopy(ds_orig)

    if anonymize:
        anonymize_dicom(out_ds)

    dcm_bytes = save_dicom_bytes(out_ds, recon_hu, suffix=" AI-MAR")

    filename = f"CT_ChemoPort_MAR_Slice_{slice_idx + 1:03d}.dcm"
    return Response(
        content=dcm_bytes,
        media_type="application/dicom",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(dcm_bytes)),
        }
    )


@router.get("/{scan_id}/series/zip")
async def export_full_series_zip(scan_id: str, anonymize: bool = False):
    """Packages all reconstructed slices into a single ZIP archive."""
    session = session_manager.get(scan_id)
    if not session:
        raise HTTPException(status_code=404, detail="Scan session not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        import copy
        for idx in range(session.total_slices):
            ds_orig, _, meta = session.slices[idx]
            recon_hu = session.get_recon_hu(idx)
            
            out_ds = copy.deepcopy(ds_orig)
            if anonymize:
                anonymize_dicom(out_ds)
                
            dcm_bytes = save_dicom_bytes(out_ds, recon_hu, suffix=" AI-MAR")
            inumber = meta.get("instance_number", idx + 1)
            zf.writestr(f"CT_MAR_{inumber:04d}.dcm", dcm_bytes)

    zip_bytes = zip_buffer.getvalue()
    filename = f"ChemoPort_MAR_Series_{session.scan_id[:8]}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(zip_bytes)),
        }
    )
