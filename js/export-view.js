/**
 * Step 4 Dosimetric Verification & HU Line Profile Chart.
 * Renders high-contrast medical physical density profile curves directly on Canvas.
 */

export class ProfileChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
  }

  drawProfile(origProfile, reconProfile, yRow = 180) {
    if (!origProfile || !reconProfile) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Padding
    const pLeft = 55;
    const pRight = 20;
    const pTop = 30;
    const pBottom = 40;
    const plotW = w - pLeft - pRight;
    const plotH = h - pTop - pBottom;

    // Y Range: -1000 HU to 2500 HU
    const minY = -1000;
    const maxY = 2500;
    const rangeY = maxY - minY;

    const toY = (hu) => pTop + plotH - ((hu - minY) / rangeY) * plotH;
    const toX = (xIdx) => pLeft + (xIdx / (origProfile.length - 1)) * plotW;

    // Background Grid
    ctx.strokeStyle = "rgba(226, 232, 240, 0.8)";
    ctx.lineWidth = 1;

    // Zero HU grid line (Water reference)
    const zeroY = toY(0);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(pLeft, zeroY);
    ctx.lineTo(w - pRight, zeroY);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.8)";
    ctx.stroke();
    ctx.setLineDash([]);

    // Y Axis labels
    ctx.fillStyle = "#64748b";
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "right";
    for (let hu = -1000; hu <= 2500; hu += 500) {
      const y = toY(hu);
      ctx.fillText(`${hu} HU`, pLeft - 8, y + 4);
      ctx.beginPath();
      ctx.moveTo(pLeft - 4, y);
      ctx.lineTo(pLeft, y);
      ctx.stroke();
    }

    // X Axis labels
    ctx.textAlign = "center";
    for (let x = 0; x <= 512; x += 128) {
      const px = toX(Math.min(x, 511));
      ctx.fillText(`${x} px`, px, h - pBottom + 18);
    }

    // 1. Draw Original HU Profile (Amber / Rose)
    ctx.beginPath();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2.0;
    for (let i = 0; i < origProfile.length; i++) {
      const px = toX(i);
      const py = toY(origProfile[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // 2. Draw Reconstructed AI-MAR Profile (Emerald Mint)
    ctx.beginPath();
    ctx.strokeStyle = "#0d9488";
    ctx.lineWidth = 2.2;
    for (let i = 0; i < reconProfile.length; i++) {
      const px = toX(i);
      const py = toY(reconProfile[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Title / Legend
    ctx.textAlign = "left";
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 13px -apple-system, sans-serif";
    ctx.fillText(`Horizontal HU Profile (Row Y = ${yRow})`, pLeft, 18);

    // Legend pills
    ctx.textAlign = "right";
    ctx.fillStyle = "#f59e0b";
    ctx.fillText("■ Original Corrupted", w - pRight - 140, 18);
    ctx.fillStyle = "#0d9488";
    ctx.fillText("■ AI-MAR Restored", w - pRight, 18);
  }
}
