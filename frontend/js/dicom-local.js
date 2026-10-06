/**
 * Client-side Pure JavaScript DICOM & ZIP parser for 100% Static Web Deployments.
 * Enables zero-server local parsing directly inside the browser memory.
 */

export class LocalDicomLoader {
  static async loadFiles(files, onProgress) {
    const slices = [];
    const totalFiles = files.length;

    for (let i = 0; i < totalFiles; i++) {
      const file = files[i];
      const name = file.name.toLowerCase();

      // Check if ZIP by extension or magic bytes (PK\x03\x04)
      let isZip = name.endsWith(".zip");
      if (!isZip && file.size > 4) {
        try {
          const header = new Uint8Array(await file.slice(0, 4).arrayBuffer());
          if (header[0] === 0x50 && header[1] === 0x4B && header[2] === 0x03 && header[3] === 0x04) {
            isZip = true;
          }
        } catch (_) {}
      }

      if (isZip) {
        // Decompress ZIP using JSZip
        const zipSlices = await LocalDicomLoader.parseZipFile(file, (p) => {
          if (onProgress) onProgress(Math.round((i / totalFiles) * 100 + (p * 0.8) / totalFiles));
        });
        slices.push(...zipSlices);
      } else {
        // Parse individual DICOM file
        try {
          const slice = await LocalDicomLoader.parseDicomFile(file);
          if (slice) slices.push(slice);
        } catch (e) {
          console.warn("Failed parsing file:", file.name, e);
        }
        if (onProgress) {
          onProgress(Math.round(((i + 1) / totalFiles) * 100));
        }
      }
    }

    if (slices.length === 0) {
      throw new Error("No valid CT DICOM slices could be parsed from the selected files.");
    }

    // Sort slices by SliceLocation (Z-axis) or InstanceNumber
    slices.sort((a, b) => {
      if (a.sliceLocation !== undefined && b.sliceLocation !== undefined && a.sliceLocation !== b.sliceLocation) {
        return a.sliceLocation - b.sliceLocation;
      }
      return a.instanceNumber - b.instanceNumber;
    });

    // Auto-detect ChemoPort slice and 3D spatial anchor across both Right & Left hemispheres
    const { bestIdx, anchor } = LocalDicomLoader.findSeriesChemoPortAnchor(slices);

    // Extract robust B. Braun Celsite CAD vector contours for all slices
    const contoursMap = slices.map((sl) => {
      const ps = sl.pixelSpacing ? sl.pixelSpacing[0] : (slices[0].pixelSpacing ? slices[0].pixelSpacing[0] : 1.0);
      return LocalDicomLoader.extractContours(sl.hu, anchor, ps);
    });

    return {
      scan_id: "local_" + Date.now(),
      total_slices: slices.length,
      chemoport_slice_idx: bestIdx,
      series_description: slices[0].seriesDescription || "Patient CT Series",
      pixel_spacing: slices[0].pixelSpacing || [0.9765625, 0.9765625],
      slice_thickness: slices[0].sliceThickness || 2.5,
      window_defaults: {
        center: slices[0].windowCenter || 40.0,
        width: slices[0].windowWidth || 350.0,
      },
      slices: slices,
      contours: contoursMap,
    };
  }

  static async parseDicomFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    return LocalDicomLoader.parseDicomBuffer(arrayBuffer, file.name);
  }

  static async parseZipFile(file, onProgress) {
    const jszip = typeof window !== "undefined" && window.JSZip ? window.JSZip : (typeof JSZip !== "undefined" ? JSZip : null);
    if (!jszip) {
      throw new Error("JSZip library not loaded");
    }
    const arrayBuffer = await file.arrayBuffer();
    const zip = await jszip.loadAsync(arrayBuffer);
    const slices = [];
    const entries = Object.values(zip.files).filter((f) => !f.dir && !f.name.includes("__MACOSX") && !f.name.startsWith("."));
    const total = entries.length;

    for (let i = 0; i < total; i++) {
      const entry = entries[i];
      const buffer = await entry.async("arraybuffer");
      try {
        const slice = LocalDicomLoader.parseDicomBuffer(buffer, entry.name);
        if (slice) slices.push(slice);
      } catch (err) {
        console.warn("Zip entry failed:", entry.name, err);
      }
      if (onProgress && i % 5 === 0) onProgress(Math.round((i / total) * 100));
    }
    return slices;
  }

  static parseDicomBuffer(buffer, filename = "") {
    const parser = typeof window !== "undefined" && window.dicomParser ? window.dicomParser : (typeof dicomParser !== "undefined" ? dicomParser : null);
    if (!parser) {
      throw new Error("dicomParser library not loaded");
    }
    let dataSet;
    const byteArray = new Uint8Array(buffer);
    try {
      dataSet = parser.parseDicom(byteArray);
    } catch (e) {
      console.warn("dicomParser failed for", filename, e);
      return null;
    }

    const rows = dataSet.uint16("x00280010") || 512;
    const cols = dataSet.uint16("x00280011") || 512;
    const slope = dataSet.floatString("x00281053") !== undefined ? dataSet.floatString("x00281053") : 1.0;
    const intercept = dataSet.floatString("x00281052") !== undefined ? dataSet.floatString("x00281052") : 0.0;
    const wc = dataSet.floatString("x00281050") !== undefined ? dataSet.floatString("x00281050") : 40.0;
    const ww = dataSet.floatString("x00281051") !== undefined ? dataSet.floatString("x00281051") : 350.0;
    const instanceNum = dataSet.intString("x00200013") || 1;
    const seriesDesc = dataSet.string("x0008103e") || "CT Series";
    const pixelSpacingStr = dataSet.string("x00280030");
    const spacing = pixelSpacingStr ? pixelSpacingStr.split("\\").map(parseFloat) : [0.976, 0.976];
    const thickness = dataSet.floatString("x00180050") || 2.5;

    // Slice Location or Image Position Patient Z
    const imagePositionStr = dataSet.string("x00200032");
    let sliceLoc = dataSet.floatString("x00201041");
    if (sliceLoc === undefined && imagePositionStr) {
      const parts = imagePositionStr.split("\\").map(parseFloat);
      if (parts.length >= 3 && !isNaN(parts[2])) {
        sliceLoc = parts[2];
      }
    }

    // Find PixelData element
    const pixelDataElement = dataSet.elements.x7fe00010;
    if (!pixelDataElement) return null;

    const pixelOffset = pixelDataElement.dataOffset;
    const numPixels = rows * cols;
    const pixelRep = dataSet.uint16("x00280103") || 0; // 0 = unsigned, 1 = signed
    
    // Slice safe copy to ensure byte alignment
    const sliceBuffer = buffer.slice(pixelOffset, pixelOffset + numPixels * 2);
    const rawArray = pixelRep === 1 ? new Int16Array(sliceBuffer) : new Uint16Array(sliceBuffer);

    // Compute HU array (Int16)
    const huArray = new Int16Array(numPixels);
    for (let i = 0; i < numPixels; i++) {
      huArray[i] = Math.round(rawArray[i] * slope + intercept);
    }

    return {
      filename,
      rows,
      cols,
      instanceNumber: instanceNum,
      sliceLocation: sliceLoc,
      seriesDescription: seriesDesc,
      pixelSpacing: spacing,
      sliceThickness: thickness,
      windowCenter: wc,
      windowWidth: ww,
      raw: rawArray,
      hu: huArray,
      buffer: buffer,
      dataSet: dataSet,
    };
  }

  static findSeriesChemoPortAnchor(slices) {
    let bestIdx = Math.floor(slices.length / 2);
    let bestScore = -Infinity;
    let anchor = null;

    for (let idx = 0; idx < slices.length; idx++) {
      const hu = slices[idx].hu;
      const comp = LocalDicomLoader.findChemoPortComponent(hu, null);
      if (comp && comp.score > bestScore) {
        bestScore = comp.score;
        bestIdx = idx;
        anchor = { x: comp.cx, y: comp.cy };
      }
    }

    return { bestIdx, anchor };
  }

  static findChemoPortComponent(huArray, anchor = null) {
    const w = 512;
    const metalCoords = [];

    const yStart = anchor ? Math.max(30, Math.floor(anchor.y - 50)) : 40;
    const yEnd = anchor ? Math.min(320, Math.ceil(anchor.y + 50)) : 310;
    const xStart = anchor ? Math.max(50, Math.floor(anchor.x - 50)) : 60;
    const xEnd = anchor ? Math.min(460, Math.ceil(anchor.x + 50)) : 452;

    const threshold = anchor ? 1600 : 1800;

    for (let y = yStart; y < yEnd; y++) {
      const rowOffset = y * w;
      for (let x = xStart; x < xEnd; x++) {
        const val = huArray[rowOffset + x];
        if (val >= threshold) {
          metalCoords.push({ x, y, val });
        }
      }
    }

    if (metalCoords.length === 0) return null;

    // Connected Component Analysis (8-connectivity)
    const coordMap = new Map();
    for (let i = 0; i < metalCoords.length; i++) {
      const p = metalCoords[i];
      coordMap.set(p.y * w + p.x, i);
    }

    const visited = new Uint8Array(metalCoords.length);
    const clusters = [];

    for (let i = 0; i < metalCoords.length; i++) {
      if (visited[i]) continue;
      const cluster = [];
      const queue = [i];
      visited[i] = 1;
      let sumX = 0;
      let sumY = 0;
      let maxHu = -Infinity;

      let head = 0;
      while (head < queue.length) {
        const currIdx = queue[head++];
        const pt = metalCoords[currIdx];
        cluster.push(pt);
        sumX += pt.x;
        sumY += pt.y;
        if (pt.val > maxHu) maxHu = pt.val;

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = pt.x + dx;
            const ny = pt.y + dy;
            const nKey = ny * w + nx;
            if (coordMap.has(nKey)) {
              const nIdx = coordMap.get(nKey);
              if (!visited[nIdx]) {
                visited[nIdx] = 1;
                queue.push(nIdx);
              }
            }
          }
        }
      }

      const count = cluster.length;
      const cx = sumX / count;
      const cy = sumY / count;
      clusters.push({ cluster, count, cx, cy, maxHu });
    }

    let bestCluster = null;
    let bestScore = -Infinity;

    for (const c of clusters) {
      if (c.count < 4 || c.count > 1500) continue;

      const distMidline = Math.abs(c.cx - 256);

      let score = c.maxHu * 1.5 + Math.min(c.count, 250) * 12.0;
      if (c.maxHu >= 2400) score += 6000;
      if (distMidline > 20) score += 3000;
      if (c.cy < 265) score += 2000;
      if (distMidline < 12) score -= 4000;

      if (anchor) {
        const dAnchor = Math.hypot(c.cx - anchor.x, c.cy - anchor.y);
        if (dAnchor > 55) continue;
        score -= dAnchor * 50.0;
      }

      c.score = score;
      if (score > bestScore) {
        bestScore = score;
        bestCluster = c;
      }
    }

    if (bestCluster && bestCluster.cluster) {
      let sumCoreX = 0, sumCoreY = 0, coreCount = 0;
      for (const pt of bestCluster.cluster) {
        if (pt.val >= 2200) {
          sumCoreX += pt.x;
          sumCoreY += pt.y;
          coreCount++;
        }
      }
      if (coreCount >= 4) {
        bestCluster.cx = sumCoreX / coreCount;
        bestCluster.cy = sumCoreY / coreCount;
      }
    }

    return bestCluster;
  }

  /**
   * Generates exact 2D vector contours based on the official B. Braun Celsite® CAD blueprint
   * (PDF 6050179 & Brochure & specimen photo IMG_4763.JPG).
  /**
   * Generates exact anatomical cross-section contours of the B. Braun Celsite® port
   * on the Axial CT plane (matching user specification and clinical implantation anatomy).
   * 
   * Anatomical orientation:
   * - Flat base plate seated flush against deep pectoral muscle wall.
   * - Silicone septum dome elevated on anterior superficial face, DIRECTLY FACING SKIN
   *   to accept Huber needle puncture from the anterior skin surface.
   * - Outflow cannula exiting medially towards the subclavian vein catheter.
   * - Low-profile nose tapering laterally into subcutaneous fat tissue.
   */
  static generateCelsiteCADContours(cx, cy, pixelSpacing = 1.0, isLeftHemisphere = true) {
    const pxScale = 1.0 / (pixelSpacing || 1.0);
    const theta = (26.5 * Math.PI) / 180.0;

    // Basis vectors for axial cross-section:
    // Tangent (tx, ty): along chest wall towards medial (subclavian vein entry)
    // Normal (nx, ny): perpendicular towards anterior chest skin (ny < 0, superficial needle entry)
    let tx, ty, nx, ny;
    if (isLeftHemisphere) {
      tx = -Math.cos(theta);
      ty = -Math.sin(theta);
      nx = Math.sin(theta);
      ny = -Math.cos(theta);
    } else {
      tx = Math.cos(theta);
      ty = -Math.sin(theta);
      nx = -Math.sin(theta);
      ny = -Math.cos(theta);
    }

    const transformPt = (uMm, vMm) => {
      const uPx = uMm * pxScale;
      const vPx = vMm * pxScale;
      return [
        Math.round(cx + uPx * tx + vPx * nx),
        Math.round(cy + uPx * ty + vPx * ny),
      ];
    };

    const uNose = -14.0;
    const uBase = 10.0;
    const rCh = 7.4;  // 14.8mm outer dia
    const rSep = 6.1; // 12.2mm septum dia
    const vBase = -5.0;
    const vTop = 6.0;

    // 1. Outer Epoxy Housing (Low-Profile Cross Section: sloping nose, flat base, septum crest)
    const housingPoly = [
      transformPt(uNose, vBase),
      transformPt(uNose, vBase + 2.0),
      transformPt(uNose + 4.0, -0.5),
      transformPt(-rCh, 3.2),
      transformPt(-rSep, vTop - 0.4),
      transformPt(0.0, vTop),
      transformPt(rSep, vTop - 0.4),
      transformPt(rCh + 1.1, 3.5),
      transformPt(uBase, 0.5),
      transformPt(uBase, vBase),
      transformPt(0.0, vBase),
    ];

    // 2. Titanium Chamber Cup (solid metal core)
    const chamberPoly = [
      transformPt(-rCh, vBase + 0.5),
      transformPt(-rCh, vTop - 1.0),
      transformPt(-rSep, vTop - 0.7),
      transformPt(rSep, vTop - 0.7),
      transformPt(rCh, vTop - 1.0),
      transformPt(rCh, vBase + 0.5),
      transformPt(0.0, vBase + 0.5),
    ];

    // 3. Silicone Septum Puncture Dome (FACING DIRECTLY TOWARDS SKIN FOR NEEDLE PUNCTURE!)
    const septumPoly = [
      transformPt(-rSep, 2.5),
      transformPt(-rSep, vTop - 0.3),
      transformPt(0.0, vTop),
      transformPt(rSep, vTop - 0.3),
      transformPt(rSep, 2.5),
      transformPt(0.0, 2.5),
    ];

    // 4. Outflow Cannula / Stem towards catheter (pointing medially towards vein)
    const cannulaPoly = [
      transformPt(rCh, -2.0),
      transformPt(rCh + 9.5, -2.0),
      transformPt(rCh + 9.5, 0.2),
      transformPt(rCh, 0.2),
    ];

    return {
      housingPoly,
      chamberPoly,
      septumPoly,
      cannulaPoly,
      composite: [housingPoly, chamberPoly, septumPoly, cannulaPoly],
    };
  }

  static tracePortContour(cluster, cx, cy, pixelSpacing = 1.0, isLeft = true) {
    const cad = LocalDicomLoader.generateCelsiteCADContours(cx, cy, pixelSpacing, isLeft);
    return cad.composite;
  }

  static detectStreaks(huArray, cx, cy, portRadius = 15) {
    const numRays = 72;
    const rayScores = new Float32Array(numRays);
    const rayMaxR = new Float32Array(numRays);

    for (let a = 0; a < numRays; a++) {
      const deg = a * (360 / numRays);
      const rad = (deg * Math.PI) / 180;
      const cosA = Math.cos(rad);
      const sinA = Math.sin(rad);

      let darkCount = 0;
      let brightCount = 0;
      let maxStreakR = portRadius + 20;

      for (let r = portRadius + 5; r <= 150; r += 3) {
        const qx = Math.round(cx + r * cosA);
        const qy = Math.round(cy + r * sinA);
        if (qx < 10 || qx >= 502 || qy < 10 || qy >= 502) break;

        const val = huArray[qy * 512 + qx];
        if (val < -160 && val > -850) {
          darkCount++;
          maxStreakR = Math.max(maxStreakR, r);
        } else if (val > 250 && val < 1700) {
          brightCount++;
          maxStreakR = Math.max(maxStreakR, r);
        }
      }

      rayScores[a] = darkCount * 1.5 + brightCount;
      rayMaxR[a] = maxStreakR;
    }

    const streaks = [];
    for (let a = 0; a < numRays; a++) {
      const prev = rayScores[(a - 1 + numRays) % numRays];
      const curr = rayScores[a];
      const next = rayScores[(a + 1) % numRays];

      if (curr >= 4 && curr >= prev && curr >= next) {
        const deg = a * (360 / numRays);
        const r = Math.min(150, Math.max(50, rayMaxR[a] + 10));
        const rad1 = ((deg - 2.5) * Math.PI) / 180;
        const rad2 = ((deg + 2.5) * Math.PI) / 180;
        const rStart = portRadius + 4;

        const poly = [
          [Math.round(cx + rStart * Math.cos(rad1)), Math.round(cy + rStart * Math.sin(rad1))],
          [Math.round(cx + r * Math.cos(rad1)), Math.round(cy + r * Math.sin(rad1))],
          [Math.round(cx + r * Math.cos(rad2)), Math.round(cy + r * Math.sin(rad2))],
          [Math.round(cx + rStart * Math.cos(rad2)), Math.round(cy + rStart * Math.sin(rad2))],
        ];
        streaks.push(poly);
      }
    }

    return streaks;
  }

  static extractContours(huArray, anchor = null, pixelSpacing = 1.0) {
    const comp = LocalDicomLoader.findChemoPortComponent(huArray, anchor);
    if (!comp || comp.count < 4 || comp.maxHu < 1800) {
      return { port: [], art: [], port_px: 0, art_px: 0 };
    }

    const { cx, cy, count } = comp;
    const isLeft = cx > 256;
    const cad = LocalDicomLoader.generateCelsiteCADContours(cx, cy, pixelSpacing, isLeft);
    const streaks = LocalDicomLoader.detectStreaks(huArray, cx, cy, 14);

    let artPx = 0;
    for (const s of streaks) {
      artPx += 45;
    }

    return {
      port: cad.composite,
      art: streaks,
      port_px: count,
      art_px: artPx,
    };
  }
}
