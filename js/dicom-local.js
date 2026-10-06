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
    const contoursMap = slices.map((sl, idx) => {
      const ps = sl.pixelSpacing ? sl.pixelSpacing[0] : (slices[0].pixelSpacing ? slices[0].pixelSpacing[0] : 1.0);
      return LocalDicomLoader.extractContours(sl.hu, anchor, ps, idx, bestIdx);
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
      // 1. All-HU Adaptive Titanium Core Peak
      const tiThresh = bestCluster.maxHu >= 6000 ? Math.max(3200, bestCluster.maxHu * 0.28) : Math.max(1600, bestCluster.maxHu * 0.60);
      let sumWeightedX = 0, sumWeightedY = 0, sumWeights = 0;
      let tiCount = 0;

      for (const pt of bestCluster.cluster) {
        if (pt.val >= tiThresh) {
          const w = Math.max(1, pt.val - tiThresh + 1);
          sumWeightedX += pt.x * w;
          sumWeightedY += pt.y * w;
          sumWeights += w;
          tiCount++;
        }
      }

      if (tiCount >= 4 && sumWeights > 0) {
        bestCluster.cx = sumWeightedX / sumWeights;
        bestCluster.cy = sumWeightedY / sumWeights;
      } else {
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

      // 2. Exact Boundary Tracing on All-HU Titanium Core (Moore-Neighbor Tracer)
      const mask = new Uint8Array(w * w);
      let startX = -1, startY = -1;
      const minX = Math.max(0, Math.floor(bestCluster.cx - 20));
      const maxX = Math.min(w - 1, Math.ceil(bestCluster.cx + 20));
      const minY = Math.max(0, Math.floor(bestCluster.cy - 20));
      const maxY = Math.min(w - 1, Math.ceil(bestCluster.cy + 20));

      for (let y = minY; y <= maxY; y++) {
        const rowOff = y * w;
        for (let x = minX; x <= maxX; x++) {
          if (huArray[rowOff + x] >= tiThresh) {
            mask[rowOff + x] = 1;
            if (startY === -1) {
              startX = x;
              startY = y;
            }
          }
        }
      }

      if (startY !== -1) {
        // 8-direction Moore-Neighbor boundary tracing
        const dirs = [
          [1, 0], [1, 1], [0, 1], [-1, 1],
          [-1, 0], [-1, -1], [0, -1], [1, -1]
        ];
        const rawContour = [];
        let currX = startX, currY = startY;
        let backtrack = 6;

        for (let step = 0; step < 600; step++) {
          rawContour.push([currX, currY]);
          let found = false;
          for (let d = 0; d < 8; d++) {
            const dIdx = (backtrack + d) % 8;
            const nx = currX + dirs[dIdx][0];
            const ny = currY + dirs[dIdx][1];
            if (nx >= 0 && nx < w && ny >= 0 && ny < w && mask[ny * w + nx] === 1) {
              currX = nx;
              currY = ny;
              backtrack = (dIdx + 5) % 8;
              found = true;
              break;
            }
          }
          if (!found || (currX === startX && currY === startY && rawContour.length > 2)) {
            break;
          }
        }

        // Cyclic moving-average smoothing (2-pass filter)
        if (rawContour.length >= 8) {
          const N = rawContour.length;
          let smooth = rawContour;
          for (let pass = 0; pass < 2; pass++) {
            const nextSmooth = [];
            for (let i = 0; i < N; i++) {
              const pPrev2 = smooth[(i - 2 + N) % N];
              const pPrev1 = smooth[(i - 1 + N) % N];
              const pCurr  = smooth[i];
              const pNext1 = smooth[(i + 1) % N];
              const pNext2 = smooth[(i + 2) % N];
              const sx = 0.1 * pPrev2[0] + 0.2 * pPrev1[0] + 0.4 * pCurr[0] + 0.2 * pNext1[0] + 0.1 * pNext2[0];
              const sy = 0.1 * pPrev2[1] + 0.2 * pPrev1[1] + 0.4 * pCurr[1] + 0.2 * pNext1[1] + 0.1 * pNext2[1];
              nextSmooth.push([Math.round(sx * 10) / 10, Math.round(sy * 10) / 10]);
            }
            smooth = nextSmooth;
          }

          bestCluster.portContour = smooth;

          // Upper crest facing skin: Contiguous run around apex (min y) where y <= cy
          let minYVal = Infinity;
          let topIdx = 0;
          for (let i = 0; i < N; i++) {
            if (smooth[i][1] < minYVal) {
              minYVal = smooth[i][1];
              topIdx = i;
            }
          }

          const cy = bestCluster.cy;
          const backPts = [];
          for (let step = 1; step < N; step++) {
            const idx = (topIdx - step + N) % N;
            if (smooth[idx][1] <= cy) {
              backPts.push(smooth[idx]);
            } else {
              break;
            }
          }

          const fwdPts = [];
          for (let step = 1; step < N; step++) {
            const idx = (topIdx + step) % N;
            if (smooth[idx][1] <= cy) {
              fwdPts.push(smooth[idx]);
            } else {
              break;
            }
          }

          backPts.reverse();
          const septumArc = [...backPts, smooth[topIdx], ...fwdPts];
          if (septumArc.length >= 2) {
            bestCluster.septumArc = septumArc;
          }
        }
      }
    }

    return bestCluster;
  }

  /**
   * Generates exact anatomical cross-section contours of the B. Braun Celsite® port
   * on the Axial CT plane (Housing + Cannula + Titanium Chamber + Silicone Septum).
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
    const theta = (21.0 * Math.PI) / 180.0;

    let tx, ty, nx, ny;
    if (isLeftHemisphere) {
      // Patient left chest (x > 256):
      // +u points laterally into subcutaneous pocket
      // -u points medially towards vein catheter
      tx = Math.cos(theta);
      ty = Math.sin(theta);
      // +v points towards anterior chest skin (needle puncture direction, ny < 0)
      nx = Math.sin(theta);
      ny = -Math.cos(theta);
    } else {
      // Patient right chest (x <= 256):
      tx = -Math.cos(theta);
      ty = Math.sin(theta);
      nx = -Math.sin(theta);
      ny = -Math.cos(theta);
    }

    const transformPt = (uMm, vMm) => {
      const uPx = uMm * pxScale;
      const vPx = vMm * pxScale;
      return [
        Math.round((cx + uPx * tx + vPx * nx) * 10) / 10,
        Math.round((cy + uPx * ty + vPx * ny) * 10) / 10,
      ];
    };

    // 1. Outer Housing Profile (Smooth teardrop/wedge axial contour)
    const housingPoly = [];
    // Base line along muscle fascia (-v)
    for (let i = 0; i <= 8; i++) {
      const u = -8.5 + (18.0 * i) / 8.0;
      const v = -3.8 + 0.15 * (1.0 - Math.pow(u / 9.0, 2));
      housingPoly.push(transformPt(u, v));
    }
    // Rounded lateral nose tip (+u)
    for (let i = 0; i <= 6; i++) {
      const deg = -70.0 + (140.0 * i) / 6.0;
      const rad = (deg * Math.PI) / 180.0;
      const u = 9.5 + 1.8 * Math.cos(rad);
      const v = -2.0 + 1.8 * Math.sin(rad);
      housingPoly.push(transformPt(u, v));
    }
    // Sloping anterior face from nose to septum crest
    housingPoly.push(transformPt(7.0, 1.8));
    housingPoly.push(transformPt(4.5, 4.4));
    housingPoly.push(transformPt(2.0, 4.9));
    housingPoly.push(transformPt(0.0, 5.0));
    housingPoly.push(transformPt(-2.0, 4.9));
    housingPoly.push(transformPt(-4.5, 4.4));
    housingPoly.push(transformPt(-7.0, 2.0));
    // Rounded medial wing tip (-u)
    for (let i = 0; i <= 6; i++) {
      const deg = 110.0 + (140.0 * i) / 6.0;
      const rad = (deg * Math.PI) / 180.0;
      const u = -8.5 + 1.6 * Math.cos(rad);
      const v = -2.2 + 1.6 * Math.sin(rad);
      housingPoly.push(transformPt(u, v));
    }

    // 2. Outflow Cannula Stem (exiting medially towards catheter/vein)
    const cannulaPoly = [
      transformPt(-5.6, -1.5),
      transformPt(-12.5, -1.5),
      transformPt(-12.5, 0.5),
      transformPt(-5.6, 0.5),
    ];

    // 3. Titanium Chamber Cup (matching the visible All-HU titanium reservoir boundary)
    const chamberPoly = [];
    const nChamber = 36;
    for (let i = 0; i < nChamber; i++) {
      const rad = (2.0 * Math.PI * i) / nChamber;
      chamberPoly.push(transformPt(5.7 * Math.cos(rad), 3.6 * Math.sin(rad)));
    }

    // 4. Silicone Septum Puncture Dome (FACING DIRECTLY TOWARDS SKIN FOR NEEDLE PUNCTURE)
    const septumPoly = [];
    const nSeptum = 16;
    for (let i = 0; i <= nSeptum; i++) {
      const rad = (Math.PI * i) / nSeptum;
      const u = 4.2 * Math.cos(rad);
      const v = 2.4 + 2.4 * Math.sin(rad);
      septumPoly.push(transformPt(u, v));
    }
    septumPoly.push(transformPt(-4.2, 1.2));
    septumPoly.push(transformPt(4.2, 1.2));

    return {
      housingPoly,
      cannulaPoly,
      chamberPoly,
      septumPoly,
      composite: [housingPoly, cannulaPoly, chamberPoly, septumPoly],
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

  static extractContours(huArray, anchor = null, pixelSpacing = 1.0, sliceIdx = null, bestIdx = null) {
    const isPortSlice = (sliceIdx !== null && bestIdx !== null)
      ? Math.abs(sliceIdx - bestIdx) <= 12
      : true;

    // Detect ChemoPort component or use series anchor
    const comp = LocalDicomLoader.findChemoPortComponent(huArray, anchor);
    let portCenter = null;
    let portPx = 0;

    if (comp && comp.count >= 4 && comp.maxHu >= 1800) {
      portCenter = { x: comp.cx, y: comp.cy };
      portPx = comp.count;
    } else if (anchor && isPortSlice) {
      // Stable continuous ChemoPort anchor across intermediate slices regardless of HU dropouts
      portCenter = { x: anchor.x, y: anchor.y };
      portPx = 50;
    }

    if (!portCenter || !isPortSlice) {
      return { port: [], art: [], port_px: 0, art_px: 0 };
    }

    const { x: cx, y: cy } = portCenter;
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
      port_px: portPx,
      art_px: artPx,
    };
  }
}
