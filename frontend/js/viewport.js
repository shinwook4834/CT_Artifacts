/**
 * WebGL 16-bit CT Viewport with Real-Time GPU Window/Level Shader (60+ FPS)
 * Pure client-side zero-latency medical rendering.
 */

export class MedicalViewport {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.width = options.width || 512;
    this.height = options.height || 512;
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    // View state
    this.windowCenter = options.windowCenter || 40.0;
    this.windowWidth = options.windowWidth || 350.0;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Callbacks
    this.onWindowLevelChange = options.onWindowLevelChange || null;
    this.onSliceChange = options.onSliceChange || null;
    this.onTransformChange = options.onTransformChange || null;

    // Interaction state
    this.isRightDragging = false;
    this.isLeftDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.startWC = this.windowCenter;
    this.startWW = this.windowWidth;
    this.accumWheel = 0;

    this.initWebGL();
    this.bindEvents();
    this.updateCursor();
  }

  initWebGL() {
    this.gl = this.canvas.getContext("webgl2") || this.canvas.getContext("webgl");
    if (!this.gl) {
      console.error("WebGL not supported on this browser!");
      return;
    }
    const gl = this.gl;

    // Shaders
    const vsSource = [
      "attribute vec2 a_position;",
      "attribute vec2 a_texCoord;",
      "uniform vec2 u_zoom;",
      "uniform vec2 u_pan;",
      "varying vec2 v_texCoord;",
      "void main() {",
      "  gl_Position = vec4(a_position, 0.0, 1.0);",
      "  v_texCoord = (a_texCoord - 0.5 - u_pan) / u_zoom + 0.5;",
      "}"
    ].join("\n");

    const fsSource = [
      "precision highp float;",
      "uniform sampler2D u_image;",
      "uniform float u_wc;",
      "uniform float u_ww;",
      "varying vec2 v_texCoord;",
      "void main() {",
      "  if (v_texCoord.x < 0.0 || v_texCoord.x > 1.0 || v_texCoord.y < 0.0 || v_texCoord.y > 1.0) {",
      "    gl_FragColor = vec4(0.04, 0.06, 0.08, 1.0);",
      "    return;",
      "  }",
      "  vec4 tex = texture2D(u_image, v_texCoord);",
      "  // Unpack unsigned 16-bit value from R (low byte) and G (high byte)",
      "  float low = floor(tex.r * 255.0 + 0.5);",
      "  float high = floor(tex.g * 255.0 + 0.5);",
      "  float hu = (low + high * 256.0) - 32768.0;",
      "  float lower = u_wc - (u_ww * 0.5);",
      "  float gray = clamp((hu - lower) / u_ww, 0.0, 1.0);",
      "  gl_FragColor = vec4(gray, gray, gray, 1.0);",
      "}"
    ].join("\n");

    this.program = this.createProgram(gl, vsSource, fsSource);
    gl.useProgram(this.program);

    // Quad geometry
    const positions = new Float32Array([
      -1, -1,  0, 1,
       1, -1,  1, 1,
      -1,  1,  0, 0,
      -1,  1,  0, 0,
       1, -1,  1, 1,
       1,  1,  1, 0,
    ]);

    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    this.aPosition = gl.getAttribLocation(this.program, "a_position");
    this.aTexCoord = gl.getAttribLocation(this.program, "a_texCoord");

    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 16, 0);

    gl.enableVertexAttribArray(this.aTexCoord);
    gl.vertexAttribPointer(this.aTexCoord, 2, gl.FLOAT, false, 16, 8);

    // Uniforms
    this.uWc = gl.getUniformLocation(this.program, "u_wc");
    this.uWw = gl.getUniformLocation(this.program, "u_ww");
    this.uZoom = gl.getUniformLocation(this.program, "u_zoom");
    this.uPan = gl.getUniformLocation(this.program, "u_pan");

    // Texture (using NEAREST filter to avoid byte interpolation between low and high bytes)
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  createProgram(gl, vs, fs) {
    const vShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vShader, vs);
    gl.compileShader(vShader);

    const fShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fShader, fs);
    gl.compileShader(fShader);

    const prog = gl.createProgram();
    gl.attachShader(prog, vShader);
    gl.attachShader(prog, fShader);
    gl.linkProgram(prog);
    return prog;
  }

  /**
   * Loads 16-bit Int16Array into GPU texture and renders.
   */
  loadInt16Slice(int16Array) {
    if (!this.gl || !int16Array) return;
    const gl = this.gl;
    const N = this.width * this.height;
    const rgba = new Uint8Array(N * 4);
    
    for (let i = 0; i < N; i++) {
      const hu = int16Array[i];
      // Map signed HU [-32768, 32767] to unsigned [0, 65535]
      const u16 = (hu + 32768) & 0xffff;
      const idx = i * 4;
      rgba[idx] = u16 & 0xff;            // R = Low byte
      rgba[idx + 1] = (u16 >> 8) & 0xff;    // G = High byte
      rgba[idx + 2] = 0;                 // B
      rgba[idx + 3] = 255;               // A
    }

    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, rgba
    );

    this.render();
  }

  render() {
    if (!this.gl) return;
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.04, 0.06, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.uniform1f(this.uWc, this.windowCenter);
    gl.uniform1f(this.uWw, Math.max(1.0, this.windowWidth));
    gl.uniform2f(this.uZoom, this.zoom, this.zoom);
    gl.uniform2f(this.uPan, this.panX, this.panY);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  setWindowLevel(wc, ww) {
    this.windowCenter = wc;
    this.windowWidth = ww;
    this.render();
    if (this.onWindowLevelChange) {
      this.onWindowLevelChange(this.windowCenter, this.windowWidth);
    }
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
    this.render();
    if (this.onTransformChange) {
      this.onTransformChange(this.zoom, this.panX, this.panY);
    }
  }

  resetZoom() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.updateCursor();
    this.render();
    if (this.onTransformChange) {
      this.onTransformChange(this.zoom, this.panX, this.panY);
    }
  }

  updateCursor() {
    if (!this.canvas) return;
    if (this.zoom > 1.001) {
      this.canvas.style.cursor = this.isLeftDragging ? "grabbing" : "grab";
    } else {
      this.canvas.style.cursor = "crosshair";
    }
  }

  bindEvents() {
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    const handleWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();

      // Trackpad pinch (ctrlKey) or Alt/Meta key triggers smooth zoom
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
        this.startWC = this.windowCenter;
        this.startWW = this.windowWidth;
      } else if (e.button === 0) {
        this.isLeftDragging = true;
        this.updateCursor();
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isRightDragging) {
        const dx = e.clientX - this.dragStartX;
        const dy = e.clientY - this.dragStartY;
        this.windowWidth = Math.max(1, this.startWW + dx * 2.5);
        this.windowCenter = this.startWC - dy * 2.0;
        this.render();
        if (this.onWindowLevelChange) {
          this.onWindowLevelChange(this.windowCenter, this.windowWidth);
        }
      } else if (this.isLeftDragging && (this.zoom > 1.001 || e.shiftKey)) {
        const dx = (e.clientX - this.dragStartX) / this.canvas.width;
        const dy = (e.clientY - this.dragStartY) / this.canvas.height;
        this.panX += dx;
        this.panY += dy;
        const maxPan = 0.5 * (1.0 - 1.0 / this.zoom) + 0.15;
        this.panX = Math.max(-maxPan, Math.min(maxPan, this.panX));
        this.panY = Math.max(-maxPan, Math.min(maxPan, this.panY));

        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;
        this.render();
        if (this.onTransformChange) {
          this.onTransformChange(this.zoom, this.panX, this.panY);
        }
      }
    });

    window.addEventListener("mouseup", () => {
      this.isRightDragging = false;
      this.isLeftDragging = false;
      this.updateCursor();
    });

    this.canvas.addEventListener("dblclick", () => this.resetZoom());
  }
}
