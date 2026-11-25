import type { ReactElement } from "react";

interface OrderSpectrumProps {
  tensor4x400: number[];
  selectedBand: number;
  onSelectBand: (band: number) => void;
}

const BAND_LABELS = ["0.5–5 kHz", "5–15 kHz", "15–35 kHz", "35–65 kHz"];

export function OrderSpectrumChart({
  tensor4x400,
  selectedBand,
  onSelectBand,
}: OrderSpectrumProps): ReactElement {
  const bins = 400;
  const offset = selectedBand * bins;
  const slice = tensor4x400.slice(offset, offset + bins);

  let maxVal = 0.01;
  for (const v of slice) {
    if (v > maxVal) maxVal = v;
  }

  const width = 700;
  const height = 210;
  const padL = 36;
  const padR = 14;
  const padT = 16;
  const padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const points = slice
    .map((val, idx) => {
      const x = padL + (idx / Math.max(1, bins - 1)) * plotW;
      const y = padT + plotH - (val / maxVal) * plotH;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const xTicks = [0, 5, 10, 15, 20, 25];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Vibration Shock Pattern Across Motor Rotations
          </h3>
          <p className="text-xs text-slate-400">
            Peaks stay lined up at fixed rotation multiples (0–25× shaft speed) even while RPM changes.
          </p>
        </div>
        <fieldset className="flex flex-wrap gap-1.5 border-0 p-0" aria-label="Frequency carrier band">
          {BAND_LABELS.map((label, idx) => (
            <button
              key={label}
              type="button"
              onClick={() => onSelectBand(idx)}
              className={`rounded-md border px-2.5 py-1 font-mono text-xs transition ${
                selectedBand === idx
                  ? "border-sky-400 bg-sky-500/20 text-sky-200"
                  : "border-slate-700 bg-slate-950/60 text-slate-400 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </fieldset>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-52 w-full rounded-lg bg-slate-950/80"
        role="img"
        aria-label="Order-domain squared envelope spectrum chart"
      >
        {[0.25, 0.5, 0.75, 1].map((frac) => {
          const y = padT + plotH * (1 - frac);
          return (
            <line
              key={frac}
              x1={padL}
              x2={width - padR}
              y1={y}
              y2={y}
              stroke="#1e293b"
              strokeDasharray="3 3"
            />
          );
        })}

        {xTicks.map((tick) => {
          const x = padL + (tick / 25) * plotW;
          return (
            <g key={tick}>
              <line x1={x} x2={x} y1={padT} y2={padT + plotH} stroke="#1e293b" />
              <text x={x} y={height - 8} textAnchor="middle" className="fill-slate-400 font-mono text-[10px]">
                {tick}×
              </text>
            </g>
          );
        })}

        <polyline fill="none" stroke="#38bdf8" strokeWidth="1.8" points={points} />
      </svg>
    </div>
  );
}

interface SignalChartsProps {
  waveform: { x: number; y: number }[];
  spectrum: { freqHz: number; magnitude: number }[];
}

export function SignalCharts({ waveform, spectrum }: SignalChartsProps): ReactElement {
  const width = 540;
  const height = 165;
  const pad = 26;
  const plotW = width - pad * 2;
  const plotH = height - pad * 1.6;

  let maxAmp = 0.001;
  for (const pt of waveform) {
    if (Math.abs(pt.y) > maxAmp) maxAmp = Math.abs(pt.y);
  }
  const wavePoints = waveform
    .map((pt, idx) => {
      const x = pad + (idx / Math.max(1, waveform.length - 1)) * plotW;
      const y = 14 + plotH / 2 - (pt.y / maxAmp) * (plotH * 0.45);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  let maxSpec = 0.0001;
  for (const pt of spectrum) {
    if (pt.magnitude > maxSpec) maxSpec = pt.magnitude;
  }
  const specPoints = spectrum
    .map((pt, idx) => {
      const x = pad + (idx / Math.max(1, spectrum.length - 1)) * plotW;
      const y = 14 + plotH - (pt.magnitude / maxSpec) * (plotH - 6);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const maxFreqHz = Math.round(spectrum[spectrum.length - 1]?.freqHz ?? 0);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-300">
          Raw Vibration Waveform
        </h4>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-40 w-full rounded bg-slate-950/80"
          role="img"
          aria-label="Raw vibration waveform plot"
        >
          <line x1={pad} x2={width - pad} y1={14 + plotH / 2} y2={14 + plotH / 2} stroke="#1e293b" />
          <polyline fill="none" stroke="#38bdf8" strokeWidth="1.5" points={wavePoints} />
        </svg>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Frequency Breakdown (0–{maxFreqHz} Hz)
          </h4>
          <span className="font-mono text-[11px] text-amber-300">ISO stops at 1000 Hz</span>
        </div>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-40 w-full rounded bg-slate-950/80"
          role="img"
          aria-label="Vibration frequency spectrum plot"
        >
          <polyline fill="none" stroke="#f59e0b" strokeWidth="1.5" points={specPoints} />
        </svg>
      </div>
    </div>
  );
}
