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

    // Offscreen rendering buffers
    this.offOrig = document.createElement("canvas");
    this.offOrig.width = this.width;
    this.offOrig.height = this.height;
    this.ctxOrig = this.offOrig.getContext("2d");

    this.offRecon = document.createElement("canvas");
    this.offRecon.width = this.width;
    this.offRecon.height = this.height;
    this.ctxRecon = this.offRecon.getContext("2d");

    // Glass settings
    this.glassEnabled = true;
    this.glassSize = 160; // px
    this.mouseX = this.width / 2;
    this.mouseY = this.height / 2;
    this.isHovering = false;

    // Window level
    this.wc = options.windowCenter || 40.0;
    this.ww = options.windowWidth || 350.0;

    // Raw HU buffers
    this.origHu = null;
    this.reconHu = null;

    this.bindEvents();
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
    this.renderHuToCtx(this.origHu, this.ctxOrig);
    // Render recon to offscreen
    this.renderHuToCtx(this.reconHu, this.ctxRecon);
  }

  renderHuToCtx(int16Array, ctx) {
    const imgData = ctx.createImageData(this.width, this.height);
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

    // 1. Draw base: Reconstructed CT
    ctx.drawImage(this.offRecon, 0, 0);

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

      // Draw original CT inside glass
      ctx.drawImage(this.offOrig, 0, 0);

      // Glass inner shadow & border
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2.5;
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

      ctx.restore();

      // Glass HUD Badge
      ctx.save();
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
    this.canvas.addEventListener("mouseenter", () => {
      this.isHovering = true;
      this.draw();
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.isHovering = false;
      this.draw();
    });

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.width / rect.width;
      const scaleY = this.height / rect.height;
      this.mouseX = (e.clientX - rect.left) * scaleX;
      this.mouseY = (e.clientY - rect.top) * scaleY;
      this.isHovering = true;
      this.draw();
    });
  }
}
