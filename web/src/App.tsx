import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Cpu,
  FileAudio,
  ShieldAlert,
  Upload,
} from "lucide-react";
import { type ChangeEvent, type DragEvent, type ReactElement, useEffect, useState } from "react";
import { OrderSpectrumChart, SignalCharts } from "./components/SpectrumChart";
import { analyzeUploadedSignal, parseCsvSignal, parseWavSignal } from "./lib/dsp";
import { runOnnxSample } from "./lib/onnxRunner";
import type {
  HealthStatus,
  MotorSamplesPayload,
  OnnxInferenceResult,
  OnnxSample,
  SignalAnalysisResult,
} from "./types";

type Tab = "onnx" | "upload";

const CLASS_DISPLAY: Record<string, string> = {
  healthy: "Healthy",
  inner: "Inner Ring",
  outer: "Outer Ring",
  ball: "Rolling Ball",
  combined: "Combined",
};

function statusStyles(status: HealthStatus): {
  banner: string;
  badge: string;
  label: string;
} {
  if (status === "HEALTHY") {
    return {
      banner: "border-emerald-500/40 bg-emerald-950/40 text-emerald-100",
      badge: "border-emerald-400/50 bg-emerald-500/20 text-emerald-200",
      label: "HEALTHY",
    };
  }
  if (status === "STOP_AND_INSPECT") {
    return {
      banner: "border-rose-500/45 bg-rose-950/45 text-rose-100",
      badge: "border-rose-400/50 bg-rose-500/20 text-rose-200",
      label: "STOP & INSPECT",
    };
  }
  return {
    banner: "border-amber-500/45 bg-amber-950/40 text-amber-100",
    badge: "border-amber-400/50 bg-amber-500/20 text-amber-200",
    label: "WATCH",
  };
}

function StatusIcon({ status }: { status: HealthStatus }): ReactElement {
  if (status === "HEALTHY") return <CheckCircle2 className="h-6 w-6 text-emerald-400" />;
  if (status === "STOP_AND_INSPECT") return <ShieldAlert className="h-6 w-6 text-rose-400" />;
  return <AlertTriangle className="h-6 w-6 text-amber-400" />;
}

export default function App(): ReactElement {
  const [tab, setTab] = useState<Tab>("onnx");
  const [payload, setPayload] = useState<MotorSamplesPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedSampleId, setSelectedSampleId] = useState<string>("healthy_c3");
  const [selectedBand, setSelectedBand] = useState<number>(0);
  const [onnxResult, setOnnxResult] = useState<OnnxInferenceResult | null>(null);
  const [onnxRunning, setOnnxRunning] = useState<boolean>(false);
  const [onnxError, setOnnxError] = useState<string | null>(null);

  const [fallbackFsInput, setFallbackFsInput] = useState<string>("");
  const [uploadAnalysis, setUploadAnalysis] = useState<SignalAnalysisResult | null>(null);
  const [uploadLoading, setUploadLoading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/data/motor_samples.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load motor sample bundle (${res.status})`);
        return res.json() as Promise<MotorSamplesPayload>;
      })
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        setLoadError(null);
        if (data.onnx_samples[0]) {
          setSelectedSampleId(data.onnx_samples[0].id);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : "Failed to load diagnostic assets");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSample: OnnxSample | undefined =
    payload?.onnx_samples.find((s) => s.id === selectedSampleId) ?? payload?.onnx_samples[0];

  useEffect(() => {
    if (!payload || !activeSample) return;
    let cancelled = false;
    setOnnxRunning(true);
    setOnnxError(null);

    runOnnxSample(payload.model.file, payload.model.classes, activeSample)
      .then((res) => {
        if (!cancelled) setOnnxResult(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setOnnxError(err instanceof Error ? err.message : "ONNX browser inference failed");
          setOnnxResult(null);
        }
      })
      .finally(() => {
        if (!cancelled) setOnnxRunning(false);
      });

    return () => {
      cancelled = true;
    };
  }, [payload, activeSample]);

  async function processCsvText(text: string, sourceName: string): Promise<void> {
    if (!payload) return;
    const parsedFs = fallbackFsInput.trim() ? Number(fallbackFsInput.trim()) : undefined;
    const sig = parseCsvSignal(text, parsedFs);
    sig.sourceName = sourceName;
    const analysis = analyzeUploadedSignal(sig, payload.healthy_baseline);
    setUploadAnalysis(analysis);
    setUploadError(null);
  }

  async function loadExampleCsv(url: string, labelName: string): Promise<void> {
    if (!payload) return;
    setUploadLoading(true);
    setUploadError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to fetch sample ${labelName} (${res.status})`);
      const text = await res.text();
      await processCsvText(text, labelName);
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : "Failed to analyze sample file");
    } finally {
      setUploadLoading(false);
    }
  }

  async function handleFile(file: File): Promise<void> {
    if (!payload) return;
    setUploadLoading(true);
    setUploadError(null);
    try {
      const lower = file.name.toLowerCase();
      if (lower.endsWith(".wav")) {
        const buf = await file.arrayBuffer();
        const sig = parseWavSignal(buf, file.name);
        setUploadAnalysis(analyzeUploadedSignal(sig, payload.healthy_baseline));
      } else if (lower.endsWith(".csv") || lower.endsWith(".txt")) {
        const text = await file.text();
        await processCsvText(text, file.name);
      } else {
        throw new Error("Unsupported file type. Drop a .csv, .txt, or .wav motor recording.");
      }
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : "Could not read file");
    } finally {
      setUploadLoading(false);
    }
  }

  function handleFileInputChange(e: ChangeEvent<HTMLInputElement>): void {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>): void {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-xl border border-sky-500/30 bg-sky-500/10 p-2.5 text-sky-400">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-slate-50 sm:text-xl">
                Motor Health Check
              </h1>
              <p className="text-sm text-slate-400">
                Check if a spinning motor bearing is healthy or wearing down — running live inside your
                browser.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900 px-3 py-1">
              <Cpu className="h-3.5 w-3.5 text-sky-400" />
              {payload ? `${payload.model.size_kb} KB INT8 ONNX` : "35.77 KB INT8 ONNX"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <div
          className="mb-6 inline-flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-900/70 p-1.5"
          role="tablist"
          aria-label="Diagnostic mode"
        >
          <button
            type="button"
            role="tab"
            aria-selected={tab === "onnx"}
            onClick={() => setTab("onnx")}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === "onnx"
                ? "bg-sky-500 text-slate-950 shadow-sm"
                : "text-slate-300 hover:bg-slate-800/70 hover:text-white"
            }`}
          >
            Live Variable-Speed Check (ONNX)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "upload"}
            onClick={() => setTab("upload")}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === "upload"
                ? "bg-sky-500 text-slate-950 shadow-sm"
                : "text-slate-300 hover:bg-slate-800/70 hover:text-white"
            }`}
          >
            Check Your Own File (.csv / .wav)
          </button>
        </div>

        {loadError && (
          <div className="mb-6 rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
            {loadError}
          </div>
        )}

        {tab === "onnx" && payload && (
          <section className="space-y-6" aria-label="Live variable-speed ONNX diagnostics">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="mb-3 text-sm font-medium text-slate-200">
                1. Pick a real motor recording (recorded while motor speed ramps up and down):
              </p>
              <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-5">
                {payload.onnx_samples.map((sample) => {
                  const selected = activeSample?.id === sample.id;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => setSelectedSampleId(sample.id)}
                      className={`rounded-xl border p-3 text-left transition ${
                        selected
                          ? "border-sky-400 bg-sky-500/15 text-white"
                          : "border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-700 hover:text-white"
                      }`}
                    >
                      <div className="text-sm font-semibold leading-snug">{sample.title}</div>
                      <div className="mt-1.5 flex items-center justify-between font-mono text-[11px] text-slate-400">
                        <span>{sample.stem}</span>
                        <span>{Math.round(sample.rpm)} RPM</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {onnxError && (
              <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
                {onnxError}
              </div>
            )}

            {onnxRunning && !onnxResult && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6 text-sm text-slate-300">
                Loading {payload.model.size_kb} KB INT8 ONNX model into browser WebAssembly...
              </div>
            )}

            {onnxResult &&
              activeSample &&
              (() => {
                const style = statusStyles(onnxResult.status);
                return (
                  <div className="space-y-5">
                    <div className={`rounded-xl border p-5 ${style.banner}`}>
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5">
                            <StatusIcon status={onnxResult.status} />
                          </div>
                          <div>
                            <div className="mb-1.5 flex flex-wrap items-center gap-2">
                              <span
                                className={`rounded-full border px-2.5 py-0.5 font-mono text-xs font-semibold ${style.badge}`}
                              >
                                {style.label}
                              </span>
                              <span className="font-mono text-xs text-slate-300">
                                Confidence: {(onnxResult.confidence * 100).toFixed(1)}%
                              </span>
                              {onnxRunning && (
                                <span className="font-mono text-xs text-sky-300">Updating...</span>
                              )}
                            </div>
                            <h2 className="text-xl font-semibold text-white">{onnxResult.headline}</h2>
                            <p className="mt-1 text-sm text-slate-200/90">{onnxResult.actionText}</p>
                          </div>
                        </div>

                        <div className="rounded-lg border border-slate-700/60 bg-slate-950/60 px-3.5 py-2 text-right font-mono text-xs">
                          <div className="text-slate-400">Browser WASM Time</div>
                          <div className="text-base font-semibold text-sky-300">
                            {onnxResult.latencyMs.toFixed(2)} ms
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                      <h3 className="mb-3 text-sm font-semibold text-slate-200">
                        Model Confidence Across Bearing States
                      </h3>
                      <div className="space-y-2.5">
                        {payload.model.classes.map((clsName, idx) => {
                          const p = onnxResult.probs[idx] ?? 0;
                          const pct = Math.max(0, Math.min(100, p * 100));
                          const isTop = clsName === onnxResult.predictedClass;
                          return (
                            <div key={clsName} className="grid grid-cols-[110px_1fr_64px] items-center gap-3">
                              <span
                                className={`text-xs font-medium ${isTop ? "text-white" : "text-slate-400"}`}
                              >
                                {CLASS_DISPLAY[clsName] ?? clsName}
                              </span>
                              <div className="h-2.5 overflow-hidden rounded-full bg-slate-950">
                                <div
                                  className={`h-full rounded-full transition-all ${
                                    isTop ? "bg-sky-400" : "bg-slate-700"
                                  }`}
                                  style={{ width: `${Math.max(pct, 0.5)}%` }}
                                />
                              </div>
                              <span className="text-right font-mono text-xs text-slate-300">
                                {pct.toFixed(1)}%
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <OrderSpectrumChart
                      tensor4x400={activeSample.tensor_4x400}
                      selectedBand={selectedBand}
                      onSelectBand={setSelectedBand}
                    />

                    <details className="group rounded-xl border border-slate-800 bg-slate-900/50 p-4">
                      <summary className="flex cursor-pointer list-none items-center justify-between font-mono text-xs text-slate-300 hover:text-white">
                        <span>Under the hood — model, parity &amp; technical scope</span>
                        <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
                      </summary>
                      <div className="mt-4 grid gap-3 border-t border-slate-800 pt-4 text-xs text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">ONNX Artifact</div>
                          <div className="mt-1 font-mono text-slate-100">
                            order_mrl_int8.onnx ({payload.model.size_kb} KB)
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Recording / Speed Ramp</div>
                          <div className="mt-1 font-mono text-slate-100">
                            {activeSample.stem} · {activeSample.rpm} RPM (Trial-3 C)
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Max Diff vs. Python CPU</div>
                          <div className="mt-1 font-mono text-emerald-300">
                            {onnxResult.maxParityDiff.toExponential(2)}
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Ground Truth Class</div>
                          <div className="mt-1 font-mono text-slate-100">{activeSample.true_class}</div>
                        </div>
                      </div>
                      <p className="mt-3 text-xs leading-relaxed text-slate-400">
                        Running live client-side via <code>onnxruntime-web</code> (single-threaded CPU WASM)
                        with zero server calls. The INT8 model takes 4 structural resonance carrier bands
                        resampled into 400 shaft-angle order bins (<code>200 kHz</code> + tachometer input).
                        For general <code>.csv</code> or <code>.wav</code> files without a tachometer, switch
                        to <strong>Check Your Own File</strong>.
                      </p>
                    </details>
                  </div>
                );
              })()}
          </section>
        )}

        {tab === "upload" && payload && (
          <section className="space-y-6" aria-label="Upload your own motor vibration CSV or WAV">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-white">
                    Test a real vibration CSV or motor audio WAV file
                  </h2>
                  <p className="text-xs text-slate-400">
                    Runs locally in your browser — compares standard plant ISO speed rules against a
                    high-frequency shock &amp; spectrum check.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      void loadExampleCsv(payload.reference_examples.healthy.url, "motor_healthy.csv")
                    }
                    className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-200 transition hover:bg-emerald-500/20"
                  >
                    Load Real Healthy Motor (.csv)
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      void loadExampleCsv(payload.reference_examples.inner_fault.url, "motor_inner_fault.csv")
                    }
                    className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-200 transition hover:bg-amber-500/20"
                  >
                    Load Real Faulty Motor (.csv)
                  </button>
                </div>
              </div>

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition ${
                  dragOver
                    ? "border-sky-400 bg-sky-500/10"
                    : "border-slate-700 bg-slate-950/60 hover:border-slate-600"
                }`}
              >
                <div className="mb-2 flex items-center gap-2 text-slate-300">
                  <Upload className="h-5 w-5 text-sky-400" />
                  <FileAudio className="h-5 w-5 text-slate-400" />
                </div>
                <p className="text-sm font-medium text-slate-200">
                  Drop a <code>.csv</code>, <code>.txt</code>, or <code>.wav</code> motor recording here
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Your file stays on your machine — nothing is uploaded anywhere.
                </p>

                <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
                  <label className="cursor-pointer rounded-lg bg-slate-100 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-white">
                    Choose File
                    <input
                      type="file"
                      accept=".csv,.txt,.wav"
                      onChange={handleFileInputChange}
                      className="sr-only"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-slate-400">
                    <span>Sample Rate (Hz, optional):</span>
                    <input
                      type="number"
                      min="40"
                      step="any"
                      placeholder="auto"
                      value={fallbackFsInput}
                      onChange={(e) => setFallbackFsInput(e.target.value)}
                      className="w-24 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
                    />
                  </label>
                </div>
              </div>

              {uploadLoading && <p className="mt-3 text-xs text-sky-300">Parsing and analyzing signal...</p>}
              {uploadError && (
                <div className="mt-3 rounded-lg border border-rose-500/40 bg-rose-950/40 p-3 text-xs text-rose-200">
                  {uploadError}
                </div>
              )}
            </div>

            {uploadAnalysis &&
              (() => {
                const style = statusStyles(uploadAnalysis.status);
                return (
                  <div className="space-y-5">
                    <div className={`rounded-xl border p-5 ${style.banner}`}>
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5">
                            <StatusIcon status={uploadAnalysis.status} />
                          </div>
                          <div>
                            <div className="mb-1.5 flex flex-wrap items-center gap-2">
                              <span
                                className={`rounded-full border px-2.5 py-0.5 font-mono text-xs font-semibold ${style.badge}`}
                              >
                                {style.label}
                              </span>
                              <span className="font-mono text-xs text-slate-300">
                                {uploadAnalysis.signal.sourceName}
                              </span>
                            </div>
                            <h3 className="text-xl font-semibold text-white">{uploadAnalysis.headline}</h3>
                            <p className="mt-1 text-sm text-slate-200/90">{uploadAnalysis.actionText}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {uploadAnalysis.isoTrapTriggered && (
                      <div className="rounded-xl border border-amber-400/50 bg-amber-950/35 p-4 text-sm text-amber-100">
                        <strong>Why standard checks miss this fault:</strong> Standard ISO 20816-3 meters only
                        watch slow vibration (10–1000 Hz) and call this motor{" "}
                        <strong>Zone A (Healthy)</strong> at{" "}
                        <code>{uploadAnalysis.iso?.velocityRmsMmS} mm/s</code>, while high-frequency shock
                        tracking scores <strong>{uploadAnalysis.drift.driftScore} / 100</strong> drift.
                      </div>
                    )}

                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                        <div className="text-xs font-medium text-slate-400">
                          Standard ISO 20816-3 Check (10–1000 Hz)
                        </div>
                        {uploadAnalysis.iso ? (
                          <div className="mt-2 flex items-baseline justify-between">
                            <span className="font-mono text-2xl font-semibold text-white">
                              Zone {uploadAnalysis.iso.zone}
                            </span>
                            <span className="font-mono text-sm text-slate-300">
                              {uploadAnalysis.iso.velocityRmsMmS} mm/s RMS
                            </span>
                          </div>
                        ) : (
                          <div className="mt-2 text-sm text-slate-400">
                            Unavailable for uncalibrated audio files (needs acceleration in g or m/s²).
                          </div>
                        )}
                      </div>

                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                        <div className="text-xs font-medium text-slate-400">
                          High-Frequency Drift Check (vs. Healthy Motor)
                        </div>
                        <div className="mt-2 flex items-baseline justify-between">
                          <span className="font-mono text-2xl font-semibold text-sky-300">
                            {uploadAnalysis.drift.driftScore} / 100
                          </span>
                          <span className="font-mono text-sm text-slate-300">
                            {uploadAnalysis.drift.rmsRatio}× baseline energy
                          </span>
                        </div>
                      </div>
                    </div>

                    <SignalCharts
                      waveform={uploadAnalysis.waveformPreview}
                      spectrum={uploadAnalysis.spectrumPreview}
                    />

                    <details className="group rounded-xl border border-slate-800 bg-slate-900/50 p-4">
                      <summary className="flex cursor-pointer list-none items-center justify-between font-mono text-xs text-slate-300 hover:text-white">
                        <span>Under the hood — signal metrics &amp; scope notes</span>
                        <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
                      </summary>
                      <div className="mt-4 grid gap-3 border-t border-slate-800 pt-4 text-xs text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Sample Rate &amp; Length</div>
                          <div className="mt-1 font-mono text-slate-100">
                            {Math.round(uploadAnalysis.signal.fsHz)} Hz ·{" "}
                            {uploadAnalysis.signal.durationS.toFixed(2)} s
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Kurtosis / Crest</div>
                          <div className="mt-1 font-mono text-slate-100">
                            {uploadAnalysis.kurtosis} / {uploadAnalysis.crestFactor}
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Median/MAD Distance</div>
                          <div className="mt-1 font-mono text-slate-100">
                            {uploadAnalysis.drift.rawDistance}
                          </div>
                        </div>
                        <div className="rounded-lg bg-slate-950/70 p-3">
                          <div className="text-slate-400">Signal Quality</div>
                          <div className="mt-1 font-mono text-slate-100">
                            {uploadAnalysis.quality.ok ? "Clean" : uploadAnalysis.quality.issues[0]}
                          </div>
                        </div>
                      </div>
                      <p className="mt-3 text-xs leading-relaxed text-slate-400">
                        Uploaded <code>.csv</code> / <code>.wav</code> files without an encoder pulse channel
                        cannot be resampled into 200 kHz shaft orders, so this tab computes zero-label
                        Median/MAD spectral &amp; statistical drift plus ISO 20816-3 velocity directly in
                        TypeScript.
                      </p>
                    </details>
                  </div>
                );
              })()}
          </section>
        )}
      </main>
    </div>
  );
}
