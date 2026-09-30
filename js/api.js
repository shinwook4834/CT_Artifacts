/**
 * ChemoPort CT-MAR Studio - API Client
 * High-speed binary streaming and REST communication with FastAPI backend.
 */

export class ApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
    // In-memory binary cache: scanId_sliceIdx -> Int16Array
    this.rawSliceCache = new Map();
    this.reconSliceCache = new Map();
  }

  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`);
      return await res.json();
    } catch (e) {
      console.warn("Backend health check failed:", e);
      return null;
    }
  }

  /**
   * Uploads files with native XHR progress tracking.
   */
  uploadScan(files, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();

      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable && onProgress) {
          const pct = Math.round((e.loaded / e.total) * 100);
          onProgress(pct);
        }
      });

      xhr.onreadystatechange = () => {
        if (xhr.readyState === XMLHttpRequest.DONE) {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText);
              resolve(data);
            } catch (err) {
              reject(new Error("Invalid JSON response from server"));
            }
          } else {
            let msg = "Upload failed";
            try {
              const errObj = JSON.parse(xhr.responseText);
              if (errObj.detail) msg = errObj.detail;
            } catch (_) {}
            reject(new Error(`${msg} (HTTP ${xhr.status})`));
          }
        }
      };

      xhr.onerror = () => reject(new Error("Network error during file upload"));
      xhr.open("POST", `${this.baseUrl}/api/scan/upload`, true);
      xhr.send(formData);
    });
  }

  /**
   * Fetches raw 512x512 Int16Array binary buffer (~512KB).
   */
  async getRawSlice(scanId, sliceIdx) {
    const cacheKey = `${scanId}_${sliceIdx}`;
    if (this.rawSliceCache.has(cacheKey)) {
      return this.rawSliceCache.get(cacheKey);
    }

    const res = await fetch(`${this.baseUrl}/api/scan/${scanId}/slice/${sliceIdx}/raw`);
    if (!res.ok) throw new Error(`Failed to load slice ${sliceIdx} (HTTP ${res.status})`);

    const arrayBuffer = await res.arrayBuffer();
    const int16Array = new Int16Array(arrayBuffer);
    this.rawSliceCache.set(cacheKey, int16Array);
    return int16Array;
  }

  /**
   * Fetches reconstructed 512x512 Int16Array binary buffer.
   */
  async getReconSlice(scanId, sliceIdx) {
    const cacheKey = `${scanId}_${sliceIdx}`;
    if (this.reconSliceCache.has(cacheKey)) {
      return this.reconSliceCache.get(cacheKey);
    }

    const res = await fetch(`${this.baseUrl}/api/scan/${scanId}/slice/${sliceIdx}/recon_raw`);
    if (!res.ok) throw new Error(`Failed to load recon slice ${sliceIdx} (HTTP ${res.status})`);

    const arrayBuffer = await res.arrayBuffer();
    const int16Array = new Int16Array(arrayBuffer);
    this.reconSliceCache.set(cacheKey, int16Array);
    return int16Array;
  }

  async getSliceContours(scanId, sliceIdx) {
    const res = await fetch(`${this.baseUrl}/api/scan/${scanId}/slice/${sliceIdx}/contours`);
    if (!res.ok) return { port: [], art: [], port_px: 0, art_px: 0 };
    return await res.json();
  }

  async runReconstruction(scanId, sliceIdx, aiWeight = 0.5) {
    const res = await fetch(`${this.baseUrl}/api/recon/${scanId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slice_idx: sliceIdx, ai_weight: aiWeight }),
    });
    if (!res.ok) throw new Error(`Reconstruction failed (HTTP ${res.status})`);
    
    // Invalidate recon cache for this slice
    this.reconSliceCache.delete(`${scanId}_${sliceIdx}`);
    return await res.json();
  }

  async getLineProfile(scanId, sliceIdx, yRow = null) {
    let url = `${this.baseUrl}/api/recon/${scanId}/slice/${sliceIdx}/profile`;
    if (yRow !== null) url += `?y_row=${yRow}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to extract line profile");
    return await res.json();
  }

  getExportDicomUrl(scanId, sliceIdx, anonymize = false) {
    return `${this.baseUrl}/api/export/${scanId}/slice/${sliceIdx}/dicom?anonymize=${anonymize}`;
  }

  getExportZipUrl(scanId, anonymize = false) {
    return `${this.baseUrl}/api/export/${scanId}/series/zip?anonymize=${anonymize}`;
  }
}

export const api = new ApiClient();
