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

    // Interaction state
    this.isRightDragging = false;
    this.isLeftDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.startWC = this.windowCenter;
    this.startWW = this.windowWidth;

    this.initWebGL();
    this.bindEvents();
  }

  initWebGL() {
    this.gl = this.canvas.getContext("webgl2") || this.canvas.getContext("webgl");
    if (!this.gl) {
      console.error("WebGL not supported on this browser!");
      return;
    }
    const gl = this.gl;
    this.isWebGL2 = !!this.canvas.getContext("webgl2");

    // Shaders
    const vsSource = `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      uniform vec2 u_zoom;
      uniform vec2 u_pan;
      varying vec2 v_texCoord;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = (a_texCoord - 0.5 - u_pan) / u_zoom + 0.5;
      }
    `;

    const fsSource = `
      precision highp float;
      uniform sampler2D u_image;
      uniform float u_wc;
      uniform float u_ww;
      varying vec2 v_texCoord;

      void main() {
        if (v_texCoord.x < 0.0 || v_texCoord.x > 1.0 || v_texCoord.y < 0.0 || v_texCoord.y > 1.0) {
          gl_FragColor = vec4(0.05, 0.07, 0.1, 1.0);
          return;
        }
        vec4 tex = texture2D(u_image, v_texCoord);
        // Unpack 16-bit signed integer HU: red is low byte, green is high byte
        float raw = (tex.r * 255.0 + tex.g * 255.0 * 256.0);
        if (raw >= 32768.0) raw -= 65536.0;
        
        float hu = raw;
        float lower = u_wc - (u_ww * 0.5);
        float gray = clamp((hu - lower) / u_ww, 0.0, 1.0);
        gl_FragColor = vec4(gray, gray, gray, 1.0);
      }
    `;

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

    // Texture
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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
    
    // Convert Int16Array to Uint8Array [low, high, low, high] (Luminance Alpha)
    const uint8 = new Uint8Array(this.width * this.height * 2);
    const view = new DataView(int16Array.buffer, int16Array.byteOffset, int16Array.byteLength);
    
    for (let i = 0; i < int16Array.length; i++) {
      const val = view.getUint16(i * 2, true); // little endian
      uint8[i * 2] = val & 0xff;
      uint8[i * 2 + 1] = (val >> 8) & 0xff;
    }

    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, this.width, this.height, 0,
      gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, uint8
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

  resetZoom() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.render();
  }

  bindEvents() {
    // Prevent default context menu on canvas for right-click drag W/L
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    // Mouse wheel slice scrolling
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const delta = Math.sign(e.deltaY);
      if (this.onSliceChange) {
        this.onSliceChange(delta);
      }
    }, { passive: false });

    // Drag interactions (Left = Zoom/Pan, Right = Window/Level)
    this.canvas.addEventListener("mousedown", (e) => {
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;

      if (e.button === 2) {
        // Right click: Window / Level
        this.isRightDragging = true;
        this.startWC = this.windowCenter;
        this.startWW = this.windowWidth;
      } else if (e.button === 0) {
        // Left click: Pan or Zoom
        this.isLeftDragging = true;
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isRightDragging) {
        const dx = e.clientX - this.dragStartX;
        const dy = e.clientY - this.dragStartY;
        // Horizontal: Window Width, Vertical: Window Center
        this.windowWidth = Math.max(1, this.startWW + dx * 2.5);
        this.windowCenter = this.startWC - dy * 2.0;
        this.render();
        if (this.onWindowLevelChange) {
          this.onWindowLevelChange(this.windowCenter, this.windowWidth);
        }
      } else if (this.isLeftDragging && e.shiftKey) {
        // Shift + Left Drag = Pan
        const dx = (e.clientX - this.dragStartX) / this.canvas.width;
        const dy = (e.clientY - this.dragStartY) / this.canvas.height;
        this.panX += dx / this.zoom;
        this.panY -= dy / this.zoom;
        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;
        this.render();
      }
    });

    window.addEventListener("mouseup", (e) => {
      this.isRightDragging = false;
      this.isLeftDragging = false;
    });

    // Double click reset
    this.canvas.addEventListener("dblclick", () => this.resetZoom());
  }
}
