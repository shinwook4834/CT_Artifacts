"""Session and Volume Memory Manager for ChemoPort CT-MAR Studio Backend.

Manages parsed CT volumes, slice arrays, contour caches, and reconstruction states
in Hugging Face's 16GB server RAM with automatic expiration and LRU cleanup.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("chemoport.session")

class ScanSession:
    def __init__(self, scan_id: str, slices: List[Tuple[Any, np.ndarray, Dict[str, Any]]]):
        self.scan_id = scan_id
        self.slices = slices  # [(ds, hu_2d_float32, meta_dict), ...]
        self.total_slices = len(slices)
        self.created_at = time.time()
        self.last_accessed = time.time()
        
        # Meta info
        first_meta = slices[0][2] if slices else {}
        first_ds = slices[0][0] if slices else None
        self.series_description = first_meta.get("series_description", getattr(first_ds, "SeriesDescription", "CT Series"))
        self.pixel_spacing = first_meta.get("pixel_spacing", [0.9765625, 0.9765625])
        self.slice_thickness = first_meta.get("slice_thickness", 2.5)
        self.window_center = first_meta.get("window_center", 40.0)
        self.window_width = first_meta.get("window_width", 350.0)
        
        # Slices sorted by InstanceNumber
        self.slices.sort(key=lambda s: s[2].get("instance_number", 0))
        
        # Caches
        self.chemoport_slice_idx: int = 0
        self.contours_cache: List[Dict[str, Any]] = []
        self.recon_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}

    def touch(self):
        self.last_accessed = time.time()

    def get_slice_hu(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < self.total_slices:
            self.touch()
            return self.slices[idx][1]
        return None

    def get_recon_hu(self, idx: int) -> Optional[np.ndarray]:
        if idx in self.recon_cache:
            self.touch()
            return self.recon_cache[idx][0]
        # Fallback to original HU
        return self.get_slice_hu(idx)


class SessionManager:
    def __init__(self, max_scans: int = 15, ttl_seconds: int = 7200):
        self.sessions: Dict[str, ScanSession] = {}
        self.max_scans = max_scans
        self.ttl = ttl_seconds

    def cleanup(self):
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s.last_accessed > self.ttl]
        for sid in expired:
            del self.sessions[sid]
            logger.info("Evicted expired session: %s", sid)
            
        if len(self.sessions) > self.max_scans:
            # Sort by last_accessed and remove oldest
            sorted_sids = sorted(self.sessions.keys(), key=lambda k: self.sessions[k].last_accessed)
            to_remove = sorted_sids[:len(self.sessions) - self.max_scans]
            for sid in to_remove:
                del self.sessions[sid]
                logger.info("Evicted LRU session: %s", sid)

    def create_session(self, slices: List[Tuple[Any, np.ndarray, Dict[str, Any]]]) -> ScanSession:
        self.cleanup()
        scan_id = uuid.uuid4().hex
        session = ScanSession(scan_id, slices)
        self.sessions[scan_id] = session
        logger.info("Created session %s with %d slices", scan_id, session.total_slices)
        return session

    def get(self, scan_id: str) -> Optional[ScanSession]:
        self.cleanup()
        session = self.sessions.get(scan_id)
        if session:
            session.touch()
        return session

    def delete(self, scan_id: str) -> bool:
        if scan_id in self.sessions:
            del self.sessions[scan_id]
            return True
        return False

# Global singleton
session_manager = SessionManager()
