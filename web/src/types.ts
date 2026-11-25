export interface ModelMetadata {
  file: string;
  size_kb: number;
  classes: string[];
  in_channels: number;
  n_bins: number;
}

export interface OnnxSample {
  id: string;
  title: string;
  stem: string;
  true_class: string;
  rpm: number;
  python_probs: number[];
  tensor_4x400: number[];
}

export interface BaselineProfile {
  machine_name: string;
  fs_hz: number;
  units: string;
  n_windows: number;
  healthy_rms: number;
  healthy_kurtosis: number;
  feature_median: number[];
  feature_scale: number[];
}

export interface ReferenceExampleMetrics {
  file: string;
  url: string;
  fs_hz: number;
  duration_s: number;
  quality_ok: boolean;
  iso_velocity_mm_s: number | null;
  iso_zone: "A" | "B" | "C" | "D" | null;
  drift_score: number;
  raw_distance: number;
  rms_ratio: number;
  kurtosis_delta: number;
  spectral_shift: number;
}

export interface MotorSamplesPayload {
  model: ModelMetadata;
  onnx_samples: OnnxSample[];
  healthy_baseline: BaselineProfile;
  reference_examples: {
    healthy: ReferenceExampleMetrics;
    inner_fault: ReferenceExampleMetrics;
  };
}

export type HealthStatus = "HEALTHY" | "WATCH" | "STOP_AND_INSPECT";

export interface OnnxInferenceResult {
  sampleId: string;
  probs: number[];
  predictedClass: string;
  confidence: number;
  status: HealthStatus;
  headline: string;
  actionText: string;
  latencyMs: number;
  maxParityDiff: number;
}

export interface ParsedSignal {
  values: Float64Array;
  fsHz: number;
  units: "g" | "m/s2" | "audio";
  sourceName: string;
  channel: string;
  durationS: number;
}

export interface SignalQuality {
  ok: boolean;
  clippedRatio: number;
  dropoutRatio: number;
  dcOffset: number;
  issues: string[];
}

export interface IsoSeverity {
  velocityRmsMmS: number;
  zone: "A" | "B" | "C" | "D";
  limitsMmS: [number, number, number];
}

export interface BaselineDriftResult {
  driftScore: number;
  rawDistance: number;
  rmsRatio: number;
  kurtosisDelta: number;
  spectralShift: number;
}

export interface SignalAnalysisResult {
  signal: ParsedSignal;
  quality: SignalQuality;
  iso: IsoSeverity | null;
  drift: BaselineDriftResult;
  rms: number;
  kurtosis: number;
  crestFactor: number;
  status: HealthStatus;
  headline: string;
  actionText: string;
  isoTrapTriggered: boolean;
  waveformPreview: { x: number; y: number }[];
  spectrumPreview: { freqHz: number; magnitude: number }[];
}
