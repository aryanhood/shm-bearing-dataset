import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { EnvelopeChart, type Marker, Waveform } from "./components/Charts";
import { CLASS_TITLES, decide } from "./lib/decision";
import { type Demo, loadDemo, parseSignal } from "./lib/demo";
import { SAMPLE_RATE, WINDOW, analyze } from "./lib/features";
import { predictProba } from "./lib/model";

const CLASS_COLORS = ["#10b981", "#f43f5e", "#f59e0b", "#a855f7"];
const STEP_SAMPLES = 256;

interface CustomSignal {
  name: string;
  signal: Float64Array;
  rpm: number;
}

export function App() {
  const [demo, setDemo] = useState<Demo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recIndex, setRecIndex] = useState<number>(1); // default to 0.007" inner-race fault
  const [custom, setCustom] = useState<CustomSignal | null>(null);
  const [offset, setOffset] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(true);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDemo(import.meta.env.BASE_URL)
      .then(setDemo)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const activeSignal = custom ? custom.signal : (demo?.signals[recIndex] ?? null);
  const activeRpm = custom ? custom.rpm : (demo?.meta.recordings[recIndex]?.rpm ?? 1730);
  const activeRec = custom ? null : (demo?.meta.recordings[recIndex] ?? null);
  const maxOffset = activeSignal ? Math.max(0, activeSignal.length - WINDOW) : 0;

  useEffect(() => {
    if (!playing || !activeSignal || maxOffset === 0) return;
    const id = window.setInterval(() => {
      setOffset((prev) => (prev + STEP_SAMPLES > maxOffset ? 0 : prev + STEP_SAMPLES));
    }, 110);
    return () => window.clearInterval(id);
  }, [playing, activeSignal, maxOffset]);

  const result = useMemo(() => {
    if (!demo || !activeSignal || activeSignal.length < WINDOW) return null;
    const safeOffset = Math.min(offset, activeSignal.length - WINDOW);
    const slice = activeSignal.subarray(safeOffset, safeOffset + WINDOW);
    const analysis = analyze(slice, activeRpm);
    const proba = predictProba(demo.model, analysis.features);
    const decision = decide(proba);
    return { analysis, proba, decision, safeOffset };
  }, [demo, activeSignal, activeRpm, offset]);

  const handleSelectRecording = (idx: number) => {
    setCustom(null);
    setRecIndex(idx);
    setOffset(0);
    setError(null);
  };

  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const signal = parseSignal(text);
    if (signal.length < WINDOW) {
      setError(`CSV needs at least ${WINDOW} numeric samples (12 kHz), found ${signal.length}.`);
      return;
    }
    setError(null);
    setCustom({ name: file.name, signal, rpm: 1750 });
    setOffset(0);
    e.target.value = "";
  };

  if (!demo) {
    return (
      <div className="loading">
        {error ? `Failed to load demo data: ${error}` : "Loading unseen 3 HP CWRU recordings..."}
      </div>
    );
  }

  const markers: Marker[] = result
    ? [
        {
          hz: result.analysis.faults.bpfi,
          label: "BPFI",
          color: "#f43f5e",
          active: result.decision.classIndex === 1,
        },
        {
          hz: result.analysis.faults.bpfo,
          label: "BPFO",
          color: "#f59e0b",
          active: result.decision.classIndex === 2,
        },
        {
          hz: result.analysis.faults.bsf,
          label: "BSF",
          color: "#a855f7",
          active: result.decision.classIndex === 3,
        },
      ]
    : [];

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <title>BearingScope</title>
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.2" />
              <circle cx="12" cy="12" r="3" fill="currentColor" />
            </svg>
          </div>
          <div>
            <h1>BearingScope</h1>
            <p>
              Explainable bearing-fault diagnosis · 100% in-browser · streaming unseen{" "}
              {demo.meta.demo_load_hp} HP CWRU recordings
            </p>
          </div>
        </div>
        <div className="controls">
          <button
            type="button"
            className={`btn ${playing ? "btn-primary" : ""}`}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? "Pause stream" : "Resume stream"}
          </button>
          <button type="button" className="btn" onClick={() => fileRef.current?.click()}>
            Upload 12 kHz CSV
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.txt"
            className="file-input"
            onChange={handleUpload}
          />
          <a
            className="btn"
            href="https://github.com/aryanhood/shm-bearing-dataset"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="selector-bar" role="tablist" aria-label="Unseen 3 HP recordings">
        {demo.meta.recordings.map((rec, idx) => (
          <button
            key={rec.id}
            type="button"
            role="tab"
            aria-selected={!custom && recIndex === idx}
            className={`pill ${!custom && recIndex === idx ? "active" : ""}`}
            onClick={() => handleSelectRecording(idx)}
          >
            <span className={`dot dot-${rec.label}`} />
            {rec.title}
            {rec.diameter ? ` · ${rec.diameter}` : ""}
          </button>
        ))}
        {custom && (
          <button type="button" className="pill active">
            Custom: {custom.name}
          </button>
        )}
      </div>

      {result && activeSignal && (
        <main className="grid">
          <section className="card">
            <h2 className="card-title">
              <span>Live Verdict</span>
              <span>
                {activeRec ? `CWRU #${activeRec.id} · ${activeRec.load_hp} HP` : "Custom"}
              </span>
            </h2>
            <div className="verdict-header">
              <span className={`badge badge-${result.decision.status}`}>
                {result.decision.status}
              </span>
              <span className="conf-num">{(result.decision.confidence * 100).toFixed(1)}%</span>
            </div>
            <p className="verdict-title">{result.decision.title}</p>
            <p className="verdict-action">{result.decision.action}</p>

            <div className="probs">
              {CLASS_TITLES.map((title, i) => {
                const p = result.proba[i] ?? 0;
                return (
                  <div className="prob-row" key={title}>
                    <span>{title}</span>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{
                          width: `${(p * 100).toFixed(1)}%`,
                          background: CLASS_COLORS[i],
                        }}
                      />
                    </div>
                    <span className="prob-val">{(p * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>

            <div className="stats-row">
              <div className="stat">
                <div className="stat-label">RMS (g)</div>
                <div className="stat-value">{result.analysis.rms.toFixed(3)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Kurtosis</div>
                <div className="stat-value">{result.analysis.kurtosis.toFixed(2)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Shaft RPM</div>
                <div className="stat-value">{activeRpm}</div>
              </div>
            </div>

            {custom && (
              <div className="scrubber-row">
                <span>RPM</span>
                <input
                  type="range"
                  min={600}
                  max={3600}
                  step={10}
                  value={custom.rpm}
                  onChange={(e) => setCustom({ ...custom, rpm: Number(e.target.value) })}
                />
                <span>{custom.rpm}</span>
              </div>
            )}
          </section>

          <section className="charts-col">
            <div className="card">
              <h2 className="card-title">
                <span>Raw Vibration · 12 kHz Drive-End Accelerometer</span>
                <span>
                  Window {(result.safeOffset / SAMPLE_RATE).toFixed(2)}s –{" "}
                  {((result.safeOffset + WINDOW) / SAMPLE_RATE).toFixed(2)}s
                </span>
              </h2>
              <Waveform signal={activeSignal} start={result.safeOffset} length={WINDOW} />
              <div className="scrubber-row">
                <span>Window position</span>
                <input
                  type="range"
                  min={0}
                  max={maxOffset}
                  step={64}
                  value={result.safeOffset}
                  onChange={(e) => {
                    setPlaying(false);
                    setOffset(Number(e.target.value));
                  }}
                />
                <span>
                  {result.safeOffset} / {activeSignal.length}
                </span>
              </div>
            </div>

            <div className="card">
              <h2 className="card-title">
                <span>Hilbert Envelope Spectrum (1.5–5.5 kHz Demodulated)</span>
                <span>1× / 2× / 3× Fault Harmonics</span>
              </h2>
              <EnvelopeChart
                spectrum={result.analysis.envelope}
                sampleRate={SAMPLE_RATE}
                window={WINDOW}
                markers={markers}
              />
              <div className="freq-legend">
                <span className="freq-item">
                  <span className="dot dot-inner" />
                  BPFI {result.analysis.faults.bpfi.toFixed(1)} Hz (score{" "}
                  {(result.analysis.features[13] ?? 0).toFixed(2)})
                </span>
                <span className="freq-item">
                  <span className="dot dot-outer" />
                  BPFO {result.analysis.faults.bpfo.toFixed(1)} Hz (score{" "}
                  {(result.analysis.features[14] ?? 0).toFixed(2)})
                </span>
                <span className="freq-item">
                  <span className="dot dot-ball" />
                  BSF {result.analysis.faults.bsf.toFixed(1)} Hz (score{" "}
                  {(result.analysis.features[15] ?? 0).toFixed(2)})
                </span>
              </div>
            </div>
          </section>
        </main>
      )}

      <footer className="footer-strip">
        <div className="metric-chip">
          <span>Model trained on</span>
          <strong>Loads 0, 1, 2 HP only</strong>
          <span>3 HP stream above is 100% unseen</span>
        </div>
        <div className="metric-chip">
          <span>Leave-one-load-out (40 CWRU files)</span>
          <strong>99.91% accuracy · 0.999 F1</strong>
          <span>2,331 non-overlapping windows</span>
        </div>
        <div className="metric-chip">
          <span>Unseen fault-size stress test</span>
          <strong>59.15% accuracy · 0.669 F1</strong>
          <span>Honest limit when defect diameter shifts</span>
        </div>
        <div className="metric-chip">
          <span>Inference engine</span>
          <strong>16 features · 14 KB JSON MLP</strong>
          <span>Bit-exact Python ↔ TypeScript parity</span>
        </div>
      </footer>
    </div>
  );
}
