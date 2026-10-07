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
      // 1. All-HU Adaptive Titanium Core Peak (accurately captures full U-shaped titanium cup border)
      const tiThresh = bestCluster.maxHu >= 6000 ? Math.max(3200, Math.min(4500, bestCluster.maxHu * 0.16)) : Math.max(1600, bestCluster.maxHu * 0.60);
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
  /**
   * Generates exact anatomical contours of the Breast Tissue Expander Injection Port Assembly
   * on the Axial CT plane:
   * 1. U-shaped Frame / Bracket (needle stop & suture tabs, tilted along chest wall)
   * 2. Central High-Density Magnetic/Metallic Core ("아래 밝은 물질")
   */
  static generateTissueExpanderContours(cx, cy, pixelSpacing = 1.0, isLeftHemisphere = true, dz = 0.0, actualCore = null) {
    const absDz = Math.abs(dz);
    if (absDz > 16.0) {
      return {
        uBracketPoly: [],
        corePoly: [],
        composite: []
      };
    }

    const pxScale = 1.0 / (pixelSpacing || 1.0);
    // Chest wall inclination: +11 deg on left breast (cx > 256), -11 deg on right breast (cx <= 256)
    const angleDeg = isLeftHemisphere ? 11.0 : -11.0;
    const theta = (angleDeg * Math.PI) / 180.0;

    const tx = Math.cos(theta);
    const ty = Math.sin(theta);
    // Anterior unit vector (pointing up towards skin)
    const nx = isLeftHemisphere ? Math.sin(theta) : -Math.sin(theta);
    const ny = -Math.cos(theta);

    const transformPt = (uMm, vMm) => {
      return [
        Math.round((cx + uMm * pxScale * tx + vMm * pxScale * nx) * 10) / 10,
        Math.round((cy + uMm * pxScale * ty + vMm * pxScale * ny) * 10) / 10,
      ];
    };

    // 1. U-shaped Frame / Bracket (Left arm, bottom shelf, right arm)
    // Physical dimensions: width = 34.0 mm, arm height = 26.0 mm, arm thickness = 2.8 mm, base thickness = 3.8 mm
    const wMm = 17.0;       // half-width
    const hMm = 26.0;       // arm height
    const twMm = 2.8;       // arm thickness
    const tbMm = 3.8;       // base thickness

    const uBracketPoly = [
      transformPt(-wMm, hMm),
      transformPt(-wMm, -tbMm),
      transformPt(wMm, -tbMm),
      transformPt(wMm, hMm),
      transformPt(wMm - twMm, hMm),
      transformPt(wMm - twMm, 0.5),
      transformPt(-wMm + twMm, 0.5),
      transformPt(-wMm + twMm, hMm)
    ];

    // 2. Central High-Density Magnetic/Metallic Core ("아래 밝은 물질")
    // Located at the bottom center of the U-channel.
    // Cylinder diameter = 13.0 mm (radius = 6.5 mm), vertical height = 10.0 mm
    let corePoly = [];
    if (absDz <= 13.0) {
      if (actualCore && actualCore.length >= 8) {
        corePoly = actualCore;
      } else {
        const rCoreU = 6.4;
        const rCoreV = 5.2;
        const nCore = 24;
        for (let i = 0; i < nCore; i++) {
          const rad = (2 * Math.PI * i) / nCore;
          const u = rCoreU * Math.cos(rad);
          const v = -1.5 + rCoreV * Math.sin(rad);
          corePoly.push(transformPt(u, v));
        }
      }
    }

    const composite = [uBracketPoly, corePoly];

    return {
      uBracketPoly,
      corePoly,
      composite
    };
  }

  static tracePortContour(cluster, cx, cy, pixelSpacing = 1.0, isLeft = true) {
    const exp = LocalDicomLoader.generateTissueExpanderContours(cx, cy, pixelSpacing, isLeft, 0.0);
    return exp.composite;
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

  static extractContours(huArray, anchor = null, pixelSpacing = 1.0, sliceIdx = null, bestIdx = null, sliceThickness = 2.5, sliceLoc = null, bestLoc = null) {
    let dz = 0.0;
    if (sliceLoc !== null && bestLoc !== null && !isNaN(sliceLoc) && !isNaN(bestLoc)) {
      dz = sliceLoc - bestLoc;
    } else if (sliceIdx !== null && bestIdx !== null) {
      dz = (sliceIdx - bestIdx) * (sliceThickness || 2.5);
    }
    const absDz = Math.abs(dz);

    // Physical Tissue Expander port total thickness along Z is ~30 mm (+/- 15 mm from center).
    // Beyond +/- 16.0 mm, no injection port cross-section exists on CT.
    if (absDz > 16.0) {
      return { port: [], art: [], port_px: 0, art_px: 0 };
    }

    // Detect high-density metal / magnetic core component on this slice
    const comp = LocalDicomLoader.findChemoPortComponent(huArray, anchor);
    let portCenter = null;
    let portPx = 0;

    if (comp && comp.count >= 4) {
      // Local metal component detected on this slice
      if (anchor) {
        const d = Math.hypot(comp.cx - anchor.x, comp.cy - anchor.y);
        if (d <= 50) {
          // Follow actual metal centroid directly on this slice
          portCenter = { x: comp.cx, y: comp.cy };
          portPx = comp.count;
        } else {
          portCenter = { x: anchor.x, y: anchor.y };
          portPx = comp.count;
        }
      } else {
        portCenter = { x: comp.cx, y: comp.cy };
        portPx = comp.count;
      }
    } else if (anchor) {
      portCenter = { x: anchor.x, y: anchor.y };
      portPx = 80;
    }

    if (!portCenter) {
      return { port: [], art: [], port_px: 0, art_px: 0 };
    }

    const { x: cx, y: cy } = portCenter;
    const isLeft = cx > 256;
    
    // Pass actual metal contour traced on this slice if available
    const actualCore = (comp && comp.portContour && comp.portContour.length >= 8) ? comp.portContour : null;

    const expContours = LocalDicomLoader.generateTissueExpanderContours(cx, cy, pixelSpacing, isLeft, dz, actualCore);
    const streaks = LocalDicomLoader.detectStreaks(huArray, cx, cy, 20);

    let artPx = 0;
    for (const s of streaks) {
      artPx += 45;
    }

    return {
      port: expContours.composite,
      art: streaks,
      port_px: portPx,
      art_px: artPx,
    };
  }
}
