/**
 * Vector Contour Overlay for Step 2 Artifact Inspection.
 * Draws ChemoPort implant core and streak artifact rays on a transparent 2D canvas.
 */

export class ContourOverlay {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.showPort = true;
    this.showArtifacts = true;
    this.contours = null;
  }

  setContours(contours) {
    this.contours = contours;
    this.draw();
  }

  togglePort(show) {
    this.showPort = show;
    this.draw();
  }

  toggleArtifacts(show) {
    this.showArtifacts = show;
    this.draw();
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw() {
    this.clear();
    if (!this.contours) return;
    const ctx = this.ctx;

    // 1. Draw Streak Artifacts (Amber/Coral)
    if (this.showArtifacts && this.contours.art && this.contours.art.length > 0) {
      ctx.save();
      ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
      ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
      ctx.lineWidth = 1.8;
      ctx.shadowColor = "rgba(245, 158, 11, 0.6)";
      ctx.shadowBlur = 6;

      for (const poly of this.contours.art) {
        if (!poly || poly.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    // 2. Draw ChemoPort Implant Contour (Neon Teal / Mint)
    if (this.showPort && this.contours.port && this.contours.port.length > 0) {
      ctx.save();
      ctx.strokeStyle = "#0d9488";
      ctx.fillStyle = "rgba(13, 148, 136, 0.22)";
      ctx.lineWidth = 2.2;
      ctx.shadowColor = "#2dd4bf";
      ctx.shadowBlur = 8;

      for (const poly of this.contours.port) {
        if (!poly || poly.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }
  }
}
