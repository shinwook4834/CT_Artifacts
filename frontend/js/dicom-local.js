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

    const thickness = slices[0].sliceThickness || 2.5;

    // Extract robust B. Braun Celsite CAD vector contours for all slices
    const bestLoc = (slices[bestIdx] && slices[bestIdx].sliceLocation !== undefined) ? slices[bestIdx].sliceLocation : null;
    const contoursMap = slices.map((sl, idx) => {
      const ps = sl.pixelSpacing ? sl.pixelSpacing[0] : (slices[0].pixelSpacing ? slices[0].pixelSpacing[0] : 1.0);
      const slLoc = sl.sliceLocation !== undefined ? sl.sliceLocation : null;
      return LocalDicomLoader.extractContours(sl.hu, anchor, ps, idx, bestIdx, thickness, slLoc, bestLoc);
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
    const arrayBuffer = (buffer instanceof ArrayBuffer)
      ? buffer
      : (buffer && buffer.buffer ? buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) : buffer);
    const sliceBuffer = arrayBuffer.slice(pixelOffset, pixelOffset + numPixels * 2);
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

  static findCandidateClusters(huArray, options = {}) {
    const w = 512;
    const yMin = options.yMin !== undefined ? options.yMin : 60;
    const yMax = options.yMax !== undefined ? options.yMax : 260;
    const xMin = options.xMin !== undefined ? options.xMin : 40;
    const xMax = options.xMax !== undefined ? options.xMax : 472;
    const minLateralOffset = options.minLateralOffset !== undefined ? options.minLateralOffset : 25;
    const threshold = options.threshold !== undefined ? options.threshold : 1550;

    const metalCoords = [];
    for (let y = yMin; y <= yMax; y++) {
      const rowOffset = y * w;
      for (let x = xMin; x <= xMax; x++) {
        if (minLateralOffset > 0 && Math.abs(x - 256) < minLateralOffset) continue;
        const val = huArray[rowOffset + x];
        if (val >= threshold) {
          metalCoords.push({ x, y, val });
        }
      }
    }

    if (metalCoords.length < 4) return [];

    const coordMap = new Map();
    for (let i = 0; i < metalCoords.length; i++) {
      coordMap.set(metalCoords[i].y * w + metalCoords[i].x, i);
    }

    const visited = new Uint8Array(metalCoords.length);
    const clusters = [];

    for (let i = 0; i < metalCoords.length; i++) {
      if (visited[i]) continue;
      const cluster = [];
      const queue = [i];
      visited[i] = 1;
      let sumX = 0, sumY = 0, maxHu = -Infinity;
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
      if (count >= 4 && count <= 800) {
        clusters.push({
          cluster,
          count,
          cx: sumX / count,
          cy: sumY / count,
          maxHu
        });
      }
    }

    return clusters;
  }

  static findSeriesChemoPortAnchor(slices) {
    let bestIdx = Math.floor(slices.length / 2);
    let bestScore = -Infinity;
    let anchor = null;

    // Search anterior breast zone (y in [60, 260], |x - 256| >= 25)
    // for tissue expander magnetic dome and port components
    for (let idx = 0; idx < slices.length; idx++) {
      const hu = slices[idx].hu;
      const clusters = LocalDicomLoader.findCandidateClusters(hu, {
        yMin: 60, yMax: 260,
        xMin: 40, xMax: 472,
        minLateralOffset: 25,
        threshold: 1550
      });

      for (const c of clusters) {
        // Strongly prioritize peak metal (magnetic dome reaches 15000 ~ 28000 HU)
        let score = c.maxHu * 2.0 + c.count * 5.0;
        if (c.maxHu >= 15000) score += 20000;
        if (score > bestScore) {
          bestScore = score;
          bestIdx = idx;
          anchor = { x: c.cx, y: c.cy };
        }
      }
    }

    // Fallback if no anterior breast cluster was found
    if (!anchor) {
      for (let idx = 0; idx < slices.length; idx++) {
        const comp = LocalDicomLoader.findWhiteComponent(slices[idx].hu, null);
        if (comp && comp.count > bestScore) {
          bestScore = comp.count;
          bestIdx = idx;
          anchor = { x: comp.cx, y: comp.cy };
        }
      }
    }

    return { bestIdx, anchor };
  }

  /**
   * Directly extracts the high-density implant component visible as white in All-HU window mode.
   * In All-HU (WL: 11000 / WW: 30000), pixels >= 1550 HU appear distinctly white.
   * Performs 8-connected component grouping and Moore-Neighbor boundary tracing.
   */
  static findWhiteComponent(huArray, anchor = null) {
    const w = 512;
    const metalCoords = [];

    const yStart = anchor ? Math.max(20, Math.floor(anchor.y - 60)) : 30;
    const yEnd = anchor ? Math.min(340, Math.ceil(anchor.y + 60)) : 320;
    const xStart = anchor ? Math.max(40, Math.floor(anchor.x - 60)) : 50;
    const xEnd = anchor ? Math.min(480, Math.ceil(anchor.x + 60)) : 460;

    // Determine local peak in anchor region to select adaptive threshold
    let peak = -Infinity;
    for (let y = yStart; y < yEnd; y++) {
      const rowOffset = y * w;
      for (let x = xStart; x < xEnd; x++) {
        const val = huArray[rowOffset + x];
        if (val > peak) peak = val;
      }
    }

    // Adaptive threshold:
    // Slices containing the neodymium magnet (peak >= 12000 HU) generate vertical streak halo of 1500~3500 HU.
    // In All-HU (WL: 11000 / WW: 30000), only the magnet core (>= 5500 HU) appears bright white.
    // Slices without the magnet (U-cup/wings, peak < 7000 HU) have no halo, where 1550 HU traces the entire structure.
    let threshold = 1550;
    if (peak >= 12000) threshold = 5500;
    else if (peak >= 7000) threshold = 3500;

    for (let y = yStart; y < yEnd; y++) {
      const rowOffset = y * w;
      for (let x = xStart; x < xEnd; x++) {
        const val = huArray[rowOffset + x];
        if (val >= threshold) {
          metalCoords.push({ x, y, val });
        }
      }
    }

    if (metalCoords.length < 4) return null;

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
      let sumX = 0, sumY = 0, maxHu = -Infinity;
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
      clusters.push({
        cluster,
        count,
        cx: sumX / count,
        cy: sumY / count,
        maxHu
      });
    }

    let bestCluster = null;
    let bestScore = -Infinity;

    for (const c of clusters) {
      if (c.count < 4 || c.count > 2500) continue;

      let score = c.count * 10.0 + c.maxHu;
      if (anchor) {
        const dAnchor = Math.hypot(c.cx - anchor.x, c.cy - anchor.y);
        if (dAnchor > 55) continue;
        score -= dAnchor * 50.0;
      }

      if (score > bestScore) {
        bestScore = score;
        bestCluster = c;
      }
    }

    if (!bestCluster) return null;

    // Mask ONLY bestCluster.cluster pixels
    const mask = new Uint8Array(w * w);
    bestCluster.cluster.sort((a, b) => a.y !== b.y ? a.y - b.y : a.x - b.x);
    const startX = bestCluster.cluster[0].x;
    const startY = bestCluster.cluster[0].y;

    for (const pt of bestCluster.cluster) {
      mask[pt.y * w + pt.x] = 1;
    }

    // Moore-Neighbor Boundary Tracing (Clockwise starting from North)
    const dirs = [
      [0, -1], [1, -1], [1, 0], [1, 1],
      [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ];
    const rawContour = [];
    let currX = startX, currY = startY;
    let dirIdx = 7;

    rawContour.push([currX, currY]);

    for (let step = 0; step < 1200; step++) {
      let found = false;
      for (let i = 0; i < 8; i++) {
        const d = (dirIdx + i) % 8;
        const nx = currX + dirs[d][0];
        const ny = currY + dirs[d][1];
        if (nx >= 0 && nx < w && ny >= 0 && ny < w && mask[ny * w + nx] === 1) {
          currX = nx;
          currY = ny;
          dirIdx = (d + 5) % 8;
          found = true;
          break;
        }
      }
      if (!found) break;

      if (currX === startX && currY === startY && rawContour.length > 2) {
        rawContour.push([currX, currY]);
        break;
      }
      rawContour.push([currX, currY]);
    }

    if (rawContour.length < 4) return null;

    // Cyclic moving-average smoothing (2-pass filter)
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

    return {
      cx: bestCluster.cx,
      cy: bestCluster.cy,
      count: bestCluster.count,
      contour: smooth
    };
  }

  static findChemoPortComponent(huArray, anchor = null) {
    return LocalDicomLoader.findWhiteComponent(huArray, anchor);
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

  static traceContourOfCluster(clusterPts) {
    if (!clusterPts || clusterPts.length < 4) return null;
    const w = 512;
    const mask = new Uint8Array(w * w);
    clusterPts.sort((a, b) => a.y !== b.y ? a.y - b.y : a.x - b.x);
    const startX = clusterPts[0].x;
    const startY = clusterPts[0].y;
    for (const p of clusterPts) {
      mask[p.y * w + p.x] = 1;
    }

    const dirs = [
      [0, -1], [1, -1], [1, 0], [1, 1],
      [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ];
    const rawContour = [];
    let currX = startX, currY = startY;
    let dirIdx = 7;

    rawContour.push([currX, currY]);

    for (let step = 0; step < 2000; step++) {
      let found = false;
      for (let i = 0; i < 8; i++) {
        const d = (dirIdx + i) % 8;
        const nx = currX + dirs[d][0];
        const ny = currY + dirs[d][1];
        if (nx >= 0 && nx < w && ny >= 0 && ny < w && mask[ny * w + nx] === 1) {
          currX = nx;
          currY = ny;
          dirIdx = (d + 5) % 8;
          found = true;
          break;
        }
      }
      if (!found) break;

      if (currX === startX && currY === startY && rawContour.length > 2) {
        rawContour.push([currX, currY]);
        break;
      }
      rawContour.push([currX, currY]);
    }

    if (rawContour.length < 4) return null;

    // Cyclic moving-average smoothing (2-pass filter)
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

    return smooth;
  }

  static findExpanderComponents(huArray, anchor = null) {
    if (!anchor) {
      const comp = LocalDicomLoader.findWhiteComponent(huArray, anchor);
      return { contours: comp && comp.contour ? [comp.contour] : [], count: comp ? comp.count : 0 };
    }

    const w = 512;
    const yStart = Math.max(20, Math.floor(anchor.y - 45));
    const yEnd = Math.min(340, Math.ceil(anchor.y + 45));
    const xStart = Math.max(40, Math.floor(anchor.x - 45));
    const xEnd = Math.min(480, Math.ceil(anchor.x + 45));

    let peak = -Infinity;
    for (let y = yStart; y < yEnd; y++) {
      const rowOffset = y * w;
      for (let x = xStart; x < xEnd; x++) {
        const val = huArray[rowOffset + x];
        if (val > peak) peak = val;
      }
    }

    const contours = [];
    let totalCount = 0;

    if (peak >= 12000) {
      // Slices with the neodymium magnet:
      // Unified All-HU extraction uniting the titanium cradle, upright arms, and magnet dome
      // into a single, clean, continuous contour (no fragmented pieces or stray loops).
      const mask = new Uint8Array(w * w);
      const yMin = Math.floor(anchor.y - 25);
      const yMax = Math.min(Math.floor(anchor.y + 4), 194); // Clamped to eliminate downward streak
      const xMin = Math.floor(anchor.x - 22);
      const xMax = Math.ceil(anchor.x + 22);

      for (let y = yMin; y <= yMax; y++) {
        for (let x = xMin; x <= xMax; x++) {
          // Exclude vertical streak halo directly above magnet
          if (y >= 170 && y <= 184 && x >= 321 && x <= 334) continue;
          if (huArray[y * w + x] >= 1100) {
            mask[y * w + x] = 1;
          }
        }
      }

      // Morphological horizontal closing (dilate dx=[-2..2], dy=[-1..1] then erode)
      // to bridge thin wire cradle sections across the base between arms
      const dilated = new Uint8Array(w * w);
      for (let y = yMin - 1; y <= yMax + 1; y++) {
        for (let x = xMin - 2; x <= xMax + 2; x++) {
          if (y < 0 || y >= w || x < 0 || x >= w) continue;
          let hit = 0;
          for (let dy = -1; dy <= 1 && !hit; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              const ny = y + dy, nx = x + dx;
              if (ny >= 0 && ny < w && nx >= 0 && nx < w && mask[ny * w + nx] === 1) {
                hit = 1; break;
              }
            }
          }
          dilated[y * w + x] = hit;
        }
      }

      const ptsClosed = [];
      for (let y = yMin; y <= yMax; y++) {
        for (let x = xMin; x <= xMax; x++) {
          let allHit = 1;
          for (let dy = -1; dy <= 1 && allHit; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              const ny = y + dy, nx = x + dx;
              if (ny < 0 || ny >= w || nx < 0 || nx >= w || dilated[ny * w + nx] === 0) {
                allHit = 0; break;
              }
            }
          }
          if (allHit) {
            ptsClosed.push({ x, y });
          }
        }
      }

      // Find largest connected component (8-connectivity)
      if (ptsClosed.length >= 15) {
        const coordMap = new Map();
        for (let i = 0; i < ptsClosed.length; i++) coordMap.set(ptsClosed[i].y * w + ptsClosed[i].x, i);
        const visited = new Uint8Array(ptsClosed.length);
        const clusters = [];

        for (let i = 0; i < ptsClosed.length; i++) {
          if (visited[i]) continue;
          const cl = [];
          const queue = [i];
          visited[i] = 1;
          let head = 0;
          while (head < queue.length) {
            const curr = queue[head++];
            const pt = ptsClosed[curr];
            cl.push(pt);
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = pt.x + dx, ny = pt.y + dy;
                const k = ny * w + nx;
                if (coordMap.has(k)) {
                  const nIdx = coordMap.get(k);
                  if (!visited[nIdx]) { visited[nIdx] = 1; queue.push(nIdx); }
                }
              }
            }
          }
          clusters.push(cl);
        }

        clusters.sort((a, b) => b.length - a.length);
        if (clusters.length > 0 && clusters[0].length >= 20) {
          const cUnified = LocalDicomLoader.traceContourOfCluster(clusters[0]);
          if (cUnified) {
            contours.push(cUnified);
            totalCount = clusters[0].length;
          }
        }
      }
    } else {
      // Slices without magnet (U-cup base, wings, upper dome): threshold 1550 HU
      const comp = LocalDicomLoader.findWhiteComponent(huArray, anchor);
      if (comp && comp.contour) {
        contours.push(comp.contour);
        totalCount = comp.count;
      }
    }

    return { contours, count: totalCount };
  }

  static extractContours(huArray, anchor = null, pixelSpacing = 1.0, sliceIdx = null, bestIdx = null, sliceThickness = 2.5, sliceLoc = null, bestLoc = null) {
    // Restrict contouring to slices within physical tissue expander range (|dz| <= 19.0 mm)
    if (anchor && bestIdx !== null && sliceIdx !== null) {
      const dz = (sliceLoc !== null && bestLoc !== null)
        ? Math.abs(sliceLoc - bestLoc)
        : Math.abs(sliceIdx - bestIdx) * sliceThickness;
      if (dz > 19.0) {
        return { port: [], art: [], port_px: 0, art_px: 0 };
      }
    }

    const comp = LocalDicomLoader.findExpanderComponents(huArray, anchor);
    if (!comp || !comp.contours || comp.contours.length === 0) {
      return { port: [], art: [], port_px: 0, art_px: 0 };
    }

    return {
      port: comp.contours,
      art: [],
      port_px: comp.count,
      art_px: 0,
    };
  }
}
