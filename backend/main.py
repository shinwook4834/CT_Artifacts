"""ChemoPort CT-MAR Studio - High-Performance FastAPI Backend.

Combines Hugging Face Server 2 vCPU + 16GB RAM with Client-side WebGL/WebGPU acceleration.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import scan, recon, export

app = FastAPI(
    title="ChemoPort CT-MAR Studio API",
    description="3D CAD Prior-Guided Deep Learning Metal Artifact Reduction & Dosimetric Verification",
    version="2.0.0",
)

# CORS configuration for seamless local and remote browser interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(scan.router)
app.include_router(recon.router)
app.include_router(export.router)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ChemoPort CT-MAR Studio",
        "backend": "FastAPI + Python 3.11",
        "hardware": "2 vCPU + 16GB RAM",
    }


# Mount Frontend Single Page Application (HTML, CSS, JS, ONNX Model)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
