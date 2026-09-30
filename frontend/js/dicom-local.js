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

      if (name.endsWith(".zip")) {
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
          // Ignore non-dicom files
        }
        if (onProgress) {
          onProgress(Math.round(((i + 1) / totalFiles) * 100));
        }
      }
    }

    if (slices.length === 0) {
      throw new Error("No valid CT DICOM slices could be parsed from the files.");
    }

    // Sort slices by InstanceNumber
    slices.sort((a, b) => a.instanceNumber - b.instanceNumber);

    // Auto-detect ChemoPort slice (slice with highest anterior metal pixels > 1500 HU)
    let bestIdx = 0;
    let maxMetal = -1;
    for (let idx = 0; idx < slices.length; idx++) {
      const hu = slices[idx].hu;
      let metalCount = 0;
      // Anterior half search (rows 0 to 280)
      for (let r = 50; r < 280; r++) {
        for (let c = 100; c < 412; c++) {
          if (hu[r * 512 + c] > 1500) metalCount++;
        }
      }
      if (metalCount > maxMetal) {
        maxMetal = metalCount;
        bestIdx = idx;
      }
    }

    // Extract quick contours for active slices
    const contoursMap = slices.map((sl, idx) => {
      if (Math.abs(idx - bestIdx) <= 12) {
        return LocalDicomLoader.extractContours(sl.hu);
      }
      return { port: [], art: [], port_px: 0, art_px: 0 };
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
    if (typeof JSZip === "undefined") {
      throw new Error("JSZip library not loaded");
    }
    const zip = await JSZip.loadAsync(file);
    const slices = [];
    const entries = Object.values(zip.files).filter((f) => !f.dir && !f.name.includes("__MACOSX") && !f.name.startsWith("."));
    const total = entries.length;

    for (let i = 0; i < total; i++) {
      const entry = entries[i];
      const buffer = await entry.async("arraybuffer");
      try {
        const slice = LocalDicomLoader.parseDicomBuffer(buffer, entry.name);
        if (slice) slices.push(slice);
      } catch (_) {}
      if (onProgress && i % 5 === 0) onProgress(Math.round((i / total) * 100));
    }
    return slices;
  }

  static parseDicomBuffer(buffer, filename = "") {
    if (typeof dicomParser === "undefined") {
      throw new Error("dicomParser library not loaded");
    }
    const byteArray = new Uint8Array(buffer);
    const dataSet = dicomParser.parseDicom(byteArray);

    const rows = dataSet.uint16("x00280010") || 512;
    const cols = dataSet.uint16("x00280011") || 512;
    const slope = dataSet.floatString("x00281053") || 1.0;
    const intercept = dataSet.floatString("x00281052") || 0.0;
    const wc = dataSet.floatString("x00281050") || 40.0;
    const ww = dataSet.floatString("x00281051") || 350.0;
    const instanceNum = dataSet.intString("x00200013") || 1;
    const seriesDesc = dataSet.string("x0008103e") || "CT Series";
    const pixelSpacingStr = dataSet.string("x00280030");
    const spacing = pixelSpacingStr ? pixelSpacingStr.split("\\").map(parseFloat) : [0.976, 0.976];
    const thickness = dataSet.floatString("x00180050") || 2.5;

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
      seriesDescription: seriesDesc,
      pixelSpacing: spacing,
      sliceThickness: thickness,
      windowCenter: wc,
      windowWidth: ww,
      raw: rawInt16,
      hu: huArray,
      buffer: buffer,
      dataSet: dataSet,
    };
  }

  static extractContours(huArray) {
    // Fast port center detection
    let portPoints = [];
    let artPoints = [];
    let portPx = 0;
    let artPx = 0;

    // Scan for metal cluster (> 1500 HU)
    let sumX = 0, sumY = 0, count = 0;
    for (let y = 120; y < 240; y++) {
      for (let x = 270; x < 400; x++) {
        if (huArray[y * 512 + x] > 1500) {
          sumX += x;
          sumY += y;
          count++;
        }
      }
    }

    if (count > 20) {
      portPx = count;
      const cx = sumX / count;
      const cy = sumY / count;

      // Elliptical port contour
      const rx = 24;
      const ry = 16;
      for (let deg = 0; deg < 360; deg += 15) {
        const rad = (deg * Math.PI) / 180;
        const px = cx + rx * Math.cos(rad);
        const py = cy + ry * Math.sin(rad);
        portPoints.push([Math.round(px), Math.round(py)]);
      }

      // Radiating streak artifacts (simulated polar vectors along beam angle)
      artPx = count * 3;
      for (let angle of [-42, -25, 25, 42, 138, 155]) {
        const rad = (angle * Math.PI) / 180;
        const streak = [];
        for (let r = 25; r <= 140; r += 15) {
          streak.push([Math.round(cx + r * Math.cos(rad)), Math.round(cy + r * Math.sin(rad))]);
        }
        for (let r = 140; r >= 25; r -= 15) {
          streak.push([Math.round(cx + r * Math.cos(rad) + 4), Math.round(cy + r * Math.sin(rad) + 4)]);
        }
        artPoints.push(streak);
      }
    }

    return {
      port: portPoints.length > 0 ? [portPoints] : [],
      art: artPoints,
      port_px: portPx,
      art_px: artPx,
    };
  }
}
