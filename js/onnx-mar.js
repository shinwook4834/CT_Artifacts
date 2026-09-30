/**
 * Client-Side AI MAR Engine using ONNX Runtime Web (WebGPU / WASM).
 * Runs deep residual artifact reduction directly on visitor's PC GPU.
 */

export class ClientAiEngine {
  constructor(modelPath = "./models/mar-net.onnx") {
    this.modelPath = modelPath;
    this.session = null;
    this.isReady = false;
    this.hasWebGPU = false;
  }

  async init() {
    if (this.isReady) return true;

    // Check if onnxruntime-web is loaded globally
    if (typeof ort === "undefined") {
      console.warn("ONNX Runtime Web script not loaded; falling back to server inference.");
      return false;
    }

    try {
      // Check WebGPU availability
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.hasWebGPU = true;
          console.log("⚡ WebGPU Accelerator detected on client PC!");
        }
      }

      const ep = this.hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ep,
        graphOptimizationLevel: "all",
      });

      this.isReady = true;
      console.log(`✅ Loaded mar-net.onnx using ${this.hasWebGPU ? "WebGPU" : "WASM"} engine!`);
      return true;
    } catch (err) {
      console.warn("Failed to initialize ONNX Runtime Web session:", err);
      this.isReady = false;
      return false;
    }
  }

  /**
   * Preconditions 4-channel tensor and executes forward pass on client GPU.
   */
  async runInference(origHu, cadPriorHu = null, artifactMask = null) {
    if (!this.isReady && !(await this.init())) {
      throw new Error("Client AI engine is unavailable");
    }

    const N = 512 * 512;
    const inputData = new Float32Array(4 * N);

    // Default cadPrior to origHu if null
    const prior = cadPriorHu || origHu;

    // Channel 0: Normalized Original HU [-1, 1]
    for (let i = 0; i < N; i++) {
      inputData[i] = Math.max(-1.0, Math.min(1.0, origHu[i] / 1000.0));
    }

    // Channel 1: Normalized CAD Prior HU [-1, 1]
    const ch1Offset = N;
    for (let i = 0; i < N; i++) {
      inputData[ch1Offset + i] = Math.max(-1.0, Math.min(1.0, prior[i] / 1000.0));
    }

    // Channel 2: Binary artifact streak mask
    const ch2Offset = 2 * N;
    if (artifactMask) {
      for (let i = 0; i < N; i++) inputData[ch2Offset + i] = artifactMask[i];
    }

    // Channel 3: Anatomical boundary (dermis / ribs)
    const ch3Offset = 3 * N;
    for (let i = 0; i < N; i++) {
      inputData[ch3Offset + i] = origHu[i] > 200 ? 1.0 : 0.0;
    }

    const tensor = new ort.Tensor("float32", inputData, [1, 4, 512, 512]);
    const feeds = { input: tensor };
    const results = await this.session.run(feeds);

    const outputData = results.output.data; // Float32Array (1, 1, 512, 512)
    const reconInt16 = new Int16Array(N);

    // Rescale back to authentic Hounsfield Units
    for (let i = 0; i < N; i++) {
      reconInt16[i] = Math.round(outputData[i] * 1000.0);
    }

    return reconInt16;
  }
}

export const aiEngine = new ClientAiEngine();
