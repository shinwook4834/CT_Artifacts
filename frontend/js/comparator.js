/**
 * RTP Inspection Glass and Side-by-Side Comparator for Step 3 AI Correction.
 * Displays Reconstructed CT on background with an interactive square inspection glass
 * revealing the Original corrupted CT underneath the user's cursor.
 */

export class InspectionGlassComparator {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.width = options.width || 512;
    this.height = options.height || 512;
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    this.onSliceChange = options.onSliceChange || null;
    this.onWindowLevelChange = options.onWindowLevelChange || null;

    // Offscreen rendering buffers
    this.offOrig = document.createElement("canvas");
    this.offOrig.width = this.width;
    this.offOrig.height = this.height;
    this.ctxOrig = this.offOrig.getContext("2d");
    this.imgDataOrig = this.ctxOrig.createImageData(this.width, this.height);

    this.offRecon = document.createElement("canvas");
    this.offRecon.width = this.width;
    this.offRecon.height = this.height;
    this.ctxRecon = this.offRecon.getContext("2d");
    this.imgDataRecon = this.ctxRecon.createImageData(this.width, this.height);

    // Glass settings
    this.glassEnabled = false;
    this.glassSize = 160; // px
    this.mouseX = this.width / 2;
    this.mouseY = this.height / 2;
    this.isHovering = false;

    // Wheel navigation accumulator
    this.accumWheel = 0;

    // Zoom & Pan settings
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isLeftDragging = false;
    this.isRightDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.startWC = options.windowCenter || 40.0;
    this.startWW = options.windowWidth || 350.0;

    // Window level
    this.wc = options.windowCenter || 40.0;
    this.ww = options.windowWidth || 350.0;

    // Raw HU buffers
    this.origHu = null;
    this.reconHu = null;

    this.bindEvents();
  }

  zoomIn() {
    this.setZoom(this.zoom * 1.25);
  }

  zoomOut() {
    this.setZoom(this.zoom / 1.25);
  }

  setZoom(newZoom) {
    const clamped = Math.max(1.0, Math.min(5.0, newZoom));
    this.zoom = clamped;
    if (this.zoom <= 1.001) {
      this.zoom = 1.0;
      this.panX = 0;
      this.panY = 0;
    } else {
      const maxPan = 0.5 * (1.0 - 1.0 / this.zoom) + 0.15;
      this.panX = Math.max(-maxPan, Math.min(maxPan, this.panX));
      this.panY = Math.max(-maxPan, Math.min(maxPan, this.panY));
    }
    this.updateCursor();
    this.draw();
  }

  resetZoom() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.updateCursor();
    this.draw();
  }

  updateCursor() {
    if (!this.canvas) return;
    if (this.zoom > 1.001) {
      this.canvas.style.cursor = this.isLeftDragging ? "grabbing" : "grab";
    } else {
      this.canvas.style.cursor = "crosshair";
    }
  }

  setWindowLevel(wc, ww) {
    this.wc = wc;
    this.ww = ww;
    this.renderBuffers();
    this.draw();
  }

  setSlices(origInt16, reconInt16) {
    this.origHu = origInt16;
    this.reconHu = reconInt16;
    this.renderBuffers();
    this.draw();
  }

  setGlassSize(size) {
    this.glassSize = size;
    this.draw();
  }

  toggleGlass(enabled) {
    this.glassEnabled = enabled;
    this.draw();
  }

  renderBuffers() {
    if (!this.origHu || !this.reconHu) return;

    // Render orig to offscreen
    this.renderHuToCtx(this.origHu, this.ctxOrig, this.imgDataOrig);
    // Render recon to offscreen
    this.renderHuToCtx(this.reconHu, this.ctxRecon, this.imgDataRecon);
  }

  renderHuToCtx(int16Array, ctx, imgData) {
    const data = imgData.data;
    const lower = this.wc - this.ww * 0.5;
    const invWw = 255.0 / Math.max(1.0, this.ww);

    for (let i = 0; i < int16Array.length; i++) {
      const hu = int16Array[i];
      let val = Math.round((hu - lower) * invWw);
      if (val < 0) val = 0;
      else if (val > 255) val = 255;

      const idx = i * 4;
      data[idx] = val;
      data[idx + 1] = val;
      data[idx + 2] = val;
      data[idx + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const applyTransform = (targetCtx) => {
      targetCtx.translate(this.width / 2, this.height / 2);
      targetCtx.translate(this.panX * this.width, this.panY * this.height);
      targetCtx.scale(this.zoom, this.zoom);
      targetCtx.translate(-this.width / 2, -this.height / 2);
    };

    // 1. Draw base: Reconstructed CT
    ctx.save();
    applyTransform(ctx);
    ctx.drawImage(this.offRecon, 0, 0);
    ctx.restore();

    // 2. Draw Inspection Glass if active
    if (this.glassEnabled && this.isHovering) {
      const half = this.glassSize / 2;
      const x = Math.max(0, Math.min(this.width - this.glassSize, this.mouseX - half));
      const y = Math.max(0, Math.min(this.height - this.glassSize, this.mouseY - half));

      ctx.save();
      // Clip rounded rectangle for inspection glass
      ctx.beginPath();
      ctx.roundRect(x, y, this.glassSize, this.glassSize, 14);
      ctx.clip();

      // Draw original CT inside glass with identical zoom & pan
      applyTransform(ctx);
      ctx.drawImage(this.offOrig, 0, 0);
      ctx.restore();

      // Glass inner shadow & border
      ctx.save();
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(x, y, this.glassSize, this.glassSize, 14);
      ctx.stroke();

      // Corner brackets & center reticle
      ctx.strokeStyle = "rgba(244, 63, 94, 0.75)";
      ctx.lineWidth = 1.5;
      const cx = x + half;
      const cy = y + half;
      ctx.beginPath();
      ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
      ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
      ctx.stroke();

      // Glass HUD Badge
      ctx.fillStyle = "rgba(244, 63, 94, 0.9)";
      ctx.font = "bold 11px -apple-system, sans-serif";
      const badgeText = "ORIGINAL CT";
      const textW = ctx.measureText(badgeText).width;
      ctx.beginPath();
      ctx.roundRect(x + half - textW / 2 - 8, y - 24, textW + 16, 20, 10);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(badgeText, x + half - textW / 2, y - 10);
      ctx.restore();
    }
  }

  bindEvents() {
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.canvas.addEventListener("mouseenter", () => {
      this.isHovering = true;
      this.draw();
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.isHovering = false;
      this.isLeftDragging = false;
      this.isRightDragging = false;
      this.updateCursor();
      this.draw();
    });

    const handleWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();

      if (e.ctrlKey || e.altKey || e.metaKey) {
        const factor = e.deltaY < 0 ? 1.15 : 0.87;
        this.setZoom(this.zoom * factor);
        return;
      }

      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 28;
      else if (e.deltaMode === 2) delta *= 500;

      // Direct step for notched mouse wheels
      if (Math.abs(delta) >= 40) {
        const step = Math.sign(delta);
        this.accumWheel = 0;
        if (this.onSliceChange) this.onSliceChange(step);
        return;
      }

      // Smooth accumulation for Mac trackpads and precision wheels
      this.accumWheel += delta;
      const PIXELS_PER_SLICE = 18;
      const step = Math.trunc(this.accumWheel / PIXELS_PER_SLICE);
      if (step !== 0) {
        this.accumWheel -= step * PIXELS_PER_SLICE;
        if (this.onSliceChange) this.onSliceChange(step);
      }
    };

    this.canvas.addEventListener("wheel", handleWheel, { passive: false });
    if (this.canvas.parentElement) {
      this.canvas.parentElement.addEventListener("wheel", handleWheel, { passive: false });
    }

    this.canvas.addEventListener("mousedown", (e) => {
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;

      if (e.button === 2) {
        this.isRightDragging = true;
        this.startWC = this.wc;
        this.startWW = this.ww;
      } else if (e.button === 0 && (this.zoom > 1.001 || e.shiftKey)) {
        this.isLeftDragging = true;
        this.updateCursor();
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isRightDragging) {
        const dx = e.clientX - this.dragStartX;
        const dy = e.clientY - this.dragStartY;
        this.ww = Math.max(1, this.startWW + dx * 2.5);
        this.wc = this.startWC - dy * 2.0;
        this.renderBuffers();
        this.draw();
        if (this.onWindowLevelChange) {
          this.onWindowLevelChange(this.wc, this.ww);
        }
        return;
      }

      if (this.isLeftDragging) {
        const dx = (e.clientX - this.dragStartX) / this.canvas.width;
        const dy = (e.clientY - this.dragStartY) / this.canvas.height;
        this.panX += dx;
        this.panY += dy;
        const maxPan = 0.5 * (1.0 - 1.0 / this.zoom) + 0.15;
        this.panX = Math.max(-maxPan, Math.min(maxPan, this.panX));
        this.panY = Math.max(-maxPan, Math.min(maxPan, this.panY));

        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;
        this.draw();
        return;
      }

      const rect = this.canvas.getBoundingClientRect();
      if (
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom
      ) {
        const scaleX = this.width / rect.width;
        const scaleY = this.height / rect.height;
        this.mouseX = (e.clientX - rect.left) * scaleX;
        this.mouseY = (e.clientY - rect.top) * scaleY;
        this.isHovering = true;
        this.draw();
      } else if (this.isHovering) {
        this.isHovering = false;
        this.draw();
      }
    });

    window.addEventListener("mouseup", () => {
      this.isRightDragging = false;
      if (this.isLeftDragging) {
        this.isLeftDragging = false;
        this.updateCursor();
      }
    });
  }
}
