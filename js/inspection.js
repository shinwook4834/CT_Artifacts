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
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
  }

  setTransform(zoom, panX, panY) {
    this.zoom = zoom || 1.0;
    this.panX = panX || 0;
    this.panY = panY || 0;
    this.draw();
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
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w * 0.5;
    const cy = h * 0.5;

    ctx.save();
    // Synchronize canvas coordinate system with WebGL shader zoom & pan
    ctx.translate(cx + this.panX * w, cy + this.panY * h);
    ctx.scale(this.zoom, this.zoom);
    ctx.translate(-cx, -cy);

    // 1. Draw Streak Artifacts (Amber/Coral)
    if (this.showArtifacts && this.contours.art && this.contours.art.length > 0) {
      ctx.save();
      ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
      ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
      ctx.lineWidth = 1.8 / this.zoom;
      ctx.shadowColor = "rgba(245, 158, 11, 0.6)";
      ctx.shadowBlur = 6 / this.zoom;

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

    // 2. Draw ChemoPort Implant Contour (Physical Titanium Reservoir + Anterior Septum Puncture Crest)
    if (this.showPort && this.contours.port && this.contours.port.length > 0) {
      for (let pIdx = 0; pIdx < this.contours.port.length; pIdx++) {
        const poly = this.contours.port[pIdx];
        if (!poly || poly.length < 2) continue;

        ctx.save();
        if (pIdx === 1) {
          // Septum Crest (Anterior skin-facing Huber needle puncture target)
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 2.8 / this.zoom;
          ctx.shadowColor = "#38bdf8";
          ctx.shadowBlur = 8 / this.zoom;

          ctx.beginPath();
          ctx.moveTo(poly[0][0], poly[0][1]);
          for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
          }
          ctx.stroke();
        } else {
          // Port Body (Physical titanium reservoir contour matching All-HU outline)
          ctx.strokeStyle = "#0d9488";
          ctx.fillStyle = "rgba(13, 148, 136, 0.20)";
          ctx.lineWidth = 2.0 / this.zoom;
          ctx.shadowColor = "#2dd4bf";
          ctx.shadowBlur = 6 / this.zoom;

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

    ctx.restore();
  }
}
