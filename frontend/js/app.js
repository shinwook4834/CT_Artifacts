/**
 * ChemoPort CT-MAR Studio - SPA Master Controller
 * Dual-Mode: Uses FastAPI Backend (16GB RAM) if available, or 100% Client-Side Static if hosted on HF Static Space.
 */

import { api } from "./api.js?v=2.1.4";
import { LocalDicomLoader } from "./dicom-local.js?v=2.1.4";
import { MedicalViewport } from "./viewport.js?v=2.1.4";
import { ContourOverlay } from "./inspection.js?v=2.1.4";
import { InspectionGlassComparator } from "./comparator.js?v=2.1.4";
import { aiEngine } from "./onnx-mar.js?v=2.1.4";
import { ProfileChart } from "./export-view.js?v=2.1.4";

class AppState {
  constructor() {
    this.currentStep = 1;
    this.scanId = null;
    this.totalSlices = 0;
    this.currentSliceIdx = 0;
    this.chemoportSliceIdx = 0;
    this.scanMeta = null;
    this.contours = [];
    this.localSlices = null;

    // Viewport presets
    this.wc = 40.0;
    this.ww = 350.0;
    this.aiWeight = 0.5;
  }
}

const state = new AppState();

// Viewport singletons
let viewport = null;
let overlay = null;
let comparator = null;
let profileChart = null;

// Initialize when DOM is ready (supports both early and deferred loading)
function initApp() {
  initUI();
  initDropzone();
  initViewports();
  initStepNavigation();
  initPresets();
  aiEngine.init().catch(() => {});
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

function initUI() {
  updateStepUI(1);
}

function updateStepUI(step) {
  state.currentStep = step;

  // Update Stepper pills
  document.querySelectorAll(".step-item").forEach((item) => {
    const s = parseInt(item.dataset.step, 10);
    item.classList.remove("active", "completed");
    if (s === step) item.classList.add("active");
    else if (s < step) item.classList.add("completed");
  });

  // Switch active step containers
  document.querySelectorAll(".step-container").forEach((el) => {
    el.style.display = "none";
  });

  const activeContainer = document.getElementById(`step-${step}-container`);
  if (activeContainer) activeContainer.style.display = "block";

  // Trigger step-specific load
  if (step === 2) loadStep2();
  else if (step === 3) loadStep3();
  else if (step === 4) loadStep4();
}

function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const progressBox = document.getElementById("upload-progress-box");
  const progressPct = document.getElementById("upload-pct-text");
  const progressBar = document.getElementById("upload-progress-bar");
  const dropzoneContent = document.getElementById("dropzone-content");
  const summaryCard = document.getElementById("dataset-summary-card");

  if (!dropzone || !fileInput) return;

  // Native transparent overlay handles direct user clicks (100% reliable)
  fileInput.addEventListener("dragover", () => dropzone.classList.add("dragover"));
  fileInput.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  fileInput.addEventListener("drop", () => dropzone.classList.remove("dragover"));

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = Array.from(e.target.files);
      // Temporarily hide file-input overlay so user cannot re-click while progress bar is active
      fileInput.style.display = "none";
      handleFiles(selected);
    }
  });

  async function handleFiles(files) {
    dropzoneContent.style.display = "none";
    progressBox.style.display = "flex";
    progressPct.innerText = "2%";
    progressBar.style.width = "2%";

    try {
      let data;
      const isBackendAlive = await api.checkHealth();

      if (isBackendAlive) {
        console.log("Using FastAPI Backend (2 vCPU + 16GB RAM)...");
        data = await api.uploadScan(files, (pct) => {
          const clamped = Math.max(2, Math.min(99, pct));
          progressPct.innerText = `${clamped}%`;
          progressBar.style.width = `${clamped}%`;
        });
      } else {
        console.log("Using Client-Side Pure JS Loader (Static Mode)...");
        data = await LocalDicomLoader.loadFiles(files, (pct) => {
          const clamped = Math.max(2, Math.min(99, pct));
          progressPct.innerText = `${clamped}%`;
          progressBar.style.width = `${clamped}%`;
        });
        state.localSlices = data.slices;
      }

      // Upload and volume parsing complete
      progressPct.innerText = "100%";
      progressBar.style.width = "100%";

      state.scanId = data.scan_id;
      state.totalSlices = data.total_slices;
      state.chemoportSliceIdx = data.chemoport_slice_idx;
      state.currentSliceIdx = data.chemoport_slice_idx;
      state.scanMeta = data;
      state.contours = data.contours || [];

      // Update Summary Card
      document.getElementById("sum-source-name").innerText = data.series_description || "CT Series";
      document.getElementById("sum-port-slice").innerText = `Slice ${data.chemoport_slice_idx + 1} / ${data.total_slices}`;
      document.getElementById("sum-dimensions").innerText = `512 × 512 (${data.pixel_spacing[0].toFixed(2)} mm)`;
      document.getElementById("sum-slice-count").innerText = `${data.total_slices} DICOM Slices`;

      setTimeout(() => {
        progressBox.style.display = "none";
        dropzone.style.display = "none";
        summaryCard.style.display = "block";
        document.getElementById("nav-next-1").disabled = false;
        // Auto transition to Step 2
        updateStepUI(2);
      }, 350);
    } catch (err) {
      alert(`Error loading scan: ${err.message}`);
      dropzoneContent.style.display = "flex";
      progressBox.style.display = "none";
    }
  }

  // Upload another scan button
  document.getElementById("btn-reset-scan")?.addEventListener("click", () => {
    state.scanId = null;
    state.localSlices = null;
    summaryCard.style.display = "none";
    dropzone.style.display = "flex";
    dropzoneContent.style.display = "flex";
    fileInput.style.display = "block";
    fileInput.value = "";
    document.getElementById("nav-next-1").disabled = true;
    updateStepUI(1);
  });
}

function initViewports() {
  const canvasVp = document.getElementById("canvas-viewport");
  const canvasOverlay = document.getElementById("canvas-overlay");
  const canvasComp = document.getElementById("canvas-comparator");
  const canvasProf = document.getElementById("canvas-profile");

  if (canvasVp) {
    viewport = new MedicalViewport(canvasVp, {
      onWindowLevelChange: (wc, ww) => {
        state.wc = wc;
        state.ww = ww;
        updateWlDisplay();
      },
      onSliceChange: (delta) => {
        changeSlice(state.currentSliceIdx + delta);
      },
      onTransformChange: (zoom, panX, panY) => {
        if (overlay) overlay.setTransform(zoom, panX, panY);
      },
    });
  }

  if (canvasOverlay) {
    overlay = new ContourOverlay(canvasOverlay);
  }

  if (canvasComp) {
    comparator = new InspectionGlassComparator(canvasComp, {
      onSliceChange: (delta) => {
        changeSlice(state.currentSliceIdx + delta);
      },
      onWindowLevelChange: (wc, ww) => {
        state.wc = wc;
        state.ww = ww;
        if (viewport) viewport.setWindowLevel(wc, ww);
        updateWlDisplay();
      },
    });
  }

  if (canvasProf) {
    profileChart = new ProfileChart(canvasProf);
  }

  // Slice sliders in Step 2 and Step 3
  const sliceSlider = document.getElementById("slice-slider");
  if (sliceSlider) {
    sliceSlider.addEventListener("input", (e) => {
      changeSlice(parseInt(e.target.value, 10));
    });
  }
  const sliceSlider3 = document.getElementById("slice-slider-step3");
  if (sliceSlider3) {
    sliceSlider3.addEventListener("input", (e) => {
      changeSlice(parseInt(e.target.value, 10));
    });
  }

  // Card-level wheel navigation for Step 2 and Step 3 (full card coverage)
  const bindCardWheel = (cardSelector) => {
    const card = document.querySelector(cardSelector);
    if (!card) return;
    let accum = 0;
    card.addEventListener("wheel", (e) => {
      // Ignore if user is zooming with modifiers
      if (e.ctrlKey || e.altKey || e.metaKey) return;
      // Ignore if scrolling on interactive form inputs
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

      e.preventDefault();
      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 28;
      else if (e.deltaMode === 2) delta *= 500;

      if (Math.abs(delta) >= 40) {
        changeSlice(state.currentSliceIdx + Math.sign(delta));
        accum = 0;
        return;
      }

      accum += delta;
      const step = Math.trunc(accum / 18);
      if (step !== 0) {
        accum -= step * 18;
        changeSlice(state.currentSliceIdx + step);
      }
    }, { passive: false });
  };

  bindCardWheel("#step-2-container .viewer-card");
  bindCardWheel("#step-3-container .viewer-card");
}

async function changeSlice(newIdx) {
  if (!state.scanId || state.totalSlices === 0) return;
  const clamped = Math.max(0, Math.min(state.totalSlices - 1, newIdx));
  state.currentSliceIdx = clamped;

  // Update sliders & badges
  const slider = document.getElementById("slice-slider");
  if (slider) slider.value = clamped;
  const slider3 = document.getElementById("slice-slider-step3");
  if (slider3) slider3.value = clamped;

  document.querySelectorAll(".slice-indicator-text").forEach((el) => {
    el.innerText = `Slice ${clamped + 1} / ${state.totalSlices}`;
  });

  // Fetch raw slice Int16 buffer
  let rawBuffer = null;
  try {
    if (state.localSlices && state.localSlices[clamped]) {
      rawBuffer = state.localSlices[clamped].hu;
    } else {
      rawBuffer = await api.getRawSlice(state.scanId, clamped);
    }
  } catch (err) {
    console.error("Slice load error:", err);
    return;
  }

  // Step 2: WebGL Viewport & Contours
  try {
    if (viewport) viewport.loadInt16Slice(rawBuffer);
    if (overlay && state.contours[clamped]) {
      overlay.setContours(state.contours[clamped]);
      updateContourBadge(state.contours[clamped]);
    }
  } catch (err) {
    console.warn("Step 2 render warning:", err);
  }

  // Step 3: Inspection Glass Comparator (always kept synchronized)
  try {
    if (comparator) {
      let origHu = rawBuffer;
      let reconHu = null;
      if (state.localSlices && state.localSlices[clamped]) {
        origHu = state.localSlices[clamped].hu;
        reconHu = state.localSlices[clamped].reconHu || origHu.slice();
      } else {
        try {
          reconHu = await api.getReconSlice(state.scanId, clamped);
        } catch {
          reconHu = origHu.slice();
        }
      }
      comparator.setWindowLevel(state.wc, state.ww);
      comparator.setSlices(origHu, reconHu);
    }
  } catch (err) {
    console.error("Step 3 slice comparator update error:", err);
  }
}

function updateContourBadge(cnt) {
  const badge = document.getElementById("badge-contour-status");
  if (!badge) return;
  if (cnt && cnt.port && cnt.port.length > 0) {
    badge.innerText = `🩺 ChemoPort: ${cnt.port_px || 0} px • Artifacts: ${cnt.art_px || 0} px`;
    badge.style.color = "#0d9488";
    badge.style.display = "inline-flex";
  } else {
    badge.innerText = "";
    badge.style.display = "none";
  }
}

function updateWlDisplay() {
  const wlText = `WL: ${Math.round(state.wc)} / WW: ${Math.round(state.ww)}`;
  document.querySelectorAll(".badge-wl-display").forEach((el) => {
    el.innerText = wlText;
  });
}

function initPresets() {
  const presets = {
    "preset-soft": [40, 350],
    "preset-bone": [400, 1800],
    "preset-lung": [-600, 1500],
    "preset-metal": [1200, 4000],
    "preset-all": [10760, 33530],
  };

  Object.entries(presets).forEach(([id, [wc, ww]]) => {
    document.querySelectorAll(`.${id}`).forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".preset-soft, .preset-bone, .preset-lung, .preset-metal, .preset-all").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(`.${id}`).forEach((b) => b.classList.add("active"));
        state.wc = wc;
        state.ww = ww;
        if (viewport) viewport.setWindowLevel(wc, ww);
        if (comparator) comparator.setWindowLevel(wc, ww);
        updateWlDisplay();
      });
    });
  });

  // Contour toggle buttons
  document.getElementById("btn-toggle-port")?.addEventListener("click", function () {
    this.classList.toggle("active");
    if (overlay) overlay.togglePort(this.classList.contains("active"));
  });

  document.getElementById("btn-toggle-art")?.addEventListener("click", function () {
    this.classList.toggle("active");
    if (overlay) overlay.toggleArtifacts(this.classList.contains("active"));
  });

  // Zoom Out (-) / Reset (0) / Zoom In (+) controls for Step 2 and Step 3
  document.querySelectorAll(".btn-zoom-out").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.closest("#step-3-container")) {
        if (comparator) comparator.zoomOut();
      } else {
        if (viewport) viewport.zoomOut();
      }
    });
  });

  document.querySelectorAll(".btn-reset-zoom").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.closest("#step-3-container")) {
        if (comparator) comparator.resetZoom();
      } else {
        if (viewport) viewport.resetZoom();
      }
    });
  });

  document.querySelectorAll(".btn-zoom-in").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.closest("#step-3-container")) {
        if (comparator) comparator.zoomIn();
      } else {
        if (viewport) viewport.zoomIn();
      }
    });
  });

  // Inspection glass controls (Step 3)
  const btnToggleGlass = document.getElementById("btn-toggle-glass");
  const glassSizeBtns = document.querySelectorAll(".btn-glass-size");
  let lastGlassSize = 160;

  btnToggleGlass?.addEventListener("click", function () {
    const isCurrentlyActive = this.classList.contains("active");
    if (isCurrentlyActive) {
      // Turn OFF Glass: deselect Glass button AND deselect all S/M/L buttons
      this.classList.remove("active");
      glassSizeBtns.forEach((b) => b.classList.remove("active"));
      if (comparator) comparator.toggleGlass(false);
    } else {
      // Turn ON Glass: activate Glass button and activate previous size (default M)
      this.classList.add("active");
      let matched = false;
      glassSizeBtns.forEach((b) => {
        if (parseInt(b.dataset.size, 10) === lastGlassSize) {
          b.classList.add("active");
          matched = true;
        } else {
          b.classList.remove("active");
        }
      });
      if (!matched && glassSizeBtns.length > 0) {
        glassSizeBtns[1].classList.add("active");
        lastGlassSize = parseInt(glassSizeBtns[1].dataset.size, 10);
      }
      if (comparator) {
        comparator.setGlassSize(lastGlassSize);
        comparator.toggleGlass(true);
      }
    }
  });

  glassSizeBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      const size = parseInt(this.dataset.size, 10);
      lastGlassSize = size;
      // Selecting a size activates glass if it was off
      btnToggleGlass?.classList.add("active");
      glassSizeBtns.forEach((b) => b.classList.remove("active"));
      this.classList.add("active");
      if (comparator) {
        comparator.setGlassSize(size);
        comparator.toggleGlass(true);
      }
    });
  });

  // Run AI restoration button (initialized once)
  document.getElementById("btn-run-ai")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-ai");
    btn.disabled = true;
    btn.innerText = "⚡ Running AI MAR...";

    try {
      let origHu;
      if (state.localSlices && state.localSlices[state.currentSliceIdx]) {
        origHu = state.localSlices[state.currentSliceIdx].hu;
      } else {
        origHu = await api.getRawSlice(state.scanId, state.currentSliceIdx);
      }

      let reconHu = null;

      // Try running client WebGPU first!
      if (aiEngine.isReady) {
        console.log("Running on Client WebGPU...");
        reconHu = await aiEngine.runInference(origHu);
      } else {
        console.log("Running on Server 2 vCPU + 16GB RAM...");
        await api.runReconstruction(state.scanId, state.currentSliceIdx, state.aiWeight);
        reconHu = await api.getReconSlice(state.scanId, state.currentSliceIdx);
      }

      if (state.localSlices && state.localSlices[state.currentSliceIdx]) {
        state.localSlices[state.currentSliceIdx].reconHu = reconHu;
      }

      comparator.setSlices(origHu, reconHu);
      alert("✨ AI-MAR Restoration complete!");
    } catch (e) {
      alert(`AI Restoration error: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.innerText = "▶️ Run AI Restoration";
    }
  });
}

async function loadStep2() {
  if (!state.scanId) return;
  const slider = document.getElementById("slice-slider");
  if (slider) {
    slider.max = state.totalSlices - 1;
    slider.value = state.currentSliceIdx;
  }
  await changeSlice(state.currentSliceIdx);
}

async function loadStep3() {
  if (!state.scanId || !comparator) return;
  const slider3 = document.getElementById("slice-slider-step3");
  if (slider3) {
    slider3.max = state.totalSlices - 1;
    slider3.value = state.currentSliceIdx;
  }
  await changeSlice(state.currentSliceIdx);
}

async function loadStep4() {
  if (!state.scanId || !profileChart) return;
  try {
    let origProfile, reconProfile;
    if (state.localSlices && state.localSlices[state.currentSliceIdx]) {
      const origHu = state.localSlices[state.currentSliceIdx].hu;
      const reconHu = state.localSlices[state.currentSliceIdx].reconHu || origHu;
      const yRow = 180;
      origProfile = Array.from(origHu.subarray(yRow * 512, (yRow + 1) * 512));
      reconProfile = Array.from(reconHu.subarray(yRow * 512, (yRow + 1) * 512));
      profileChart.drawProfile(origProfile, reconProfile, yRow);
    } else {
      const profileData = await api.getLineProfile(state.scanId, state.currentSliceIdx);
      profileChart.drawProfile(profileData.orig_profile, profileData.recon_profile, profileData.y_row);
    }
  } catch (err) {
    console.error("Profile chart error:", err);
  }

  // Setup download links
  const anonCheck = document.getElementById("check-anonymize");
  const isAnon = () => (anonCheck ? anonCheck.checked : false);

  const btnDcm = document.getElementById("btn-download-dcm");
  if (btnDcm) {
    btnDcm.onclick = () => {
      window.location.href = api.getExportDicomUrl(state.scanId, state.currentSliceIdx, isAnon());
    };
  }

  const btnZip = document.getElementById("btn-download-zip");
  if (btnZip) {
    btnZip.onclick = () => {
      window.location.href = api.getExportZipUrl(state.scanId, isAnon());
    };
  }
}

function initStepNavigation() {
  // Step 1 -> 2
  document.getElementById("nav-next-1")?.addEventListener("click", () => updateStepUI(2));
  // Step 2 -> 1
  document.getElementById("nav-back-2")?.addEventListener("click", () => updateStepUI(1));
  // Step 2 -> 3
  document.getElementById("nav-next-2")?.addEventListener("click", () => updateStepUI(3));
  // Step 3 -> 2
  document.getElementById("nav-back-3")?.addEventListener("click", () => updateStepUI(2));
  // Step 3 -> 4
  document.getElementById("nav-next-3")?.addEventListener("click", () => updateStepUI(4));
  // Step 4 -> 3
  document.getElementById("nav-back-4")?.addEventListener("click", () => updateStepUI(3));

  // Stepper pill clicks
  document.querySelectorAll(".step-item").forEach((pill) => {
    pill.addEventListener("click", () => {
      const s = parseInt(pill.dataset.step, 10);
      if (s === 1 || state.scanId) {
        updateStepUI(s);
      }
    });
  });
}
