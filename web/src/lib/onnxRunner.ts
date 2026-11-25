import * as ort from "onnxruntime-web";
import type { HealthStatus, OnnxInferenceResult, OnnxSample } from "../types";

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let configuredOrt = false;

const PLAIN_CLASS_COPY: Record<string, { status: HealthStatus; headline: string; actionText: string }> = {
  healthy: {
    status: "HEALTHY",
    headline: "Motor bearing looks healthy",
    actionText:
      "Even as speed changes, the vibration pattern stays smooth with no repeating bearing shock spikes.",
  },
  inner: {
    status: "WATCH",
    headline: "Inner ring wear detected",
    actionText:
      "Repeating high-frequency impact pulses match a worn inner race. Schedule maintenance at the next planned stop.",
  },
  outer: {
    status: "WATCH",
    headline: "Outer ring wear detected",
    actionText:
      "Steady periodic hits align with outer-race pitting across changing motor speeds. Plan a bearing swap soon.",
  },
  ball: {
    status: "WATCH",
    headline: "Rolling ball wear detected",
    actionText:
      "Repeating double-impact modulation matches surface wear on a rolling ball. Monitor closely and plan inspection.",
  },
  combined: {
    status: "STOP_AND_INSPECT",
    headline: "Severe multi-part bearing wear — inspect now",
    actionText:
      "Strong shock harmonics appear across multiple bearing components. Stop or derate the motor before seizure.",
  },
};

export function softmax(logits: Float32Array | number[]): number[] {
  const arr = Array.from(logits);
  if (arr.length === 0) return [];
  let maxVal = Number.NEGATIVE_INFINITY;
  for (const v of arr) {
    if (v > maxVal) maxVal = v;
  }
  const exps = arr.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => (sum > 0 ? v / sum : 0));
}

function ensureOrtConfigured(): void {
  if (configuredOrt) return;
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = 1;
  configuredOrt = true;
}

export async function getEdgeSession(modelUrl: string): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    ensureOrtConfigured();
    sessionPromise = ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }).catch((err: unknown) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

export async function runOnnxSample(
  modelUrl: string,
  classes: string[],
  sample: OnnxSample,
): Promise<OnnxInferenceResult> {
  const session = await getEdgeSession(modelUrl);
  const inputName = session.inputNames[0] ?? "order_ses";
  const outputName = session.outputNames[0] ?? "logits";

  const tensorData = Float32Array.from(sample.tensor_4x400);
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 4, 400]);

  const t0 = performance.now();
  const outputs = await session.run({ [inputName]: inputTensor });
  const latencyMs = performance.now() - t0;

  const logitsOut = outputs[outputName];
  if (!logitsOut) {
    throw new Error(`Missing output tensor '${outputName}' from ONNX Runtime`);
  }
  const probs = softmax(logitsOut.data as Float32Array);

  let bestIdx = 0;
  let bestProb = -1;
  for (let i = 0; i < probs.length; i += 1) {
    const p = probs[i] ?? 0;
    if (p > bestProb) {
      bestProb = p;
      bestIdx = i;
    }
  }

  const predictedClass = classes[bestIdx] ?? "healthy";
  const copy = PLAIN_CLASS_COPY[predictedClass] ?? {
    status: "WATCH" as HealthStatus,
    headline: `Detected class: ${predictedClass}`,
    actionText: "Inspect machine telemetry before next production run.",
  };

  let maxParityDiff = 0;
  for (let i = 0; i < probs.length; i += 1) {
    const diff = Math.abs((probs[i] ?? 0) - (sample.python_probs[i] ?? 0));
    if (diff > maxParityDiff) maxParityDiff = diff;
  }

  return {
    sampleId: sample.id,
    probs,
    predictedClass,
    confidence: bestProb,
    status: copy.status,
    headline: copy.headline,
    actionText: copy.actionText,
    latencyMs,
    maxParityDiff,
  };
}
