import { useEffect, useRef } from "react";

export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[],
) {
  const ref = useRef<HTMLCanvasElement>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: caller controls redraws via deps
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    draw(ctx, width, height);
  }, deps);
  return ref;
}

export function Waveform({
  signal,
  start,
  length,
}: { signal: Float64Array; start: number; length: number }) {
  const ref = useCanvas(
    (ctx, w, h) => {
      let peak = 1e-9;
      for (const v of signal) peak = Math.max(peak, Math.abs(v));
      const span = Math.min(signal.length, length * 4);
      const from = Math.max(0, Math.min(start + length / 2 - span / 2, signal.length - span));
      const x = (i: number) => ((i - from) / span) * w;
      // analysis window highlight
      ctx.fillStyle = "rgba(34, 211, 238, 0.08)";
      ctx.fillRect(x(start), 0, x(start + length) - x(start), h);
      ctx.strokeStyle = "rgba(34, 211, 238, 0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x(start) + 0.5, 0.5, x(start + length) - x(start) - 1, h - 1);
      // min/max envelope per pixel column
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      const perPx = span / w;
      for (let px = 0; px < w; px++) {
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        const a = Math.floor(from + px * perPx);
        const b = Math.min(signal.length, Math.floor(from + (px + 1) * perPx) + 1);
        for (let i = a; i < b; i++) {
          const v = signal[i] as number;
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
        ctx.moveTo(px + 0.5, h / 2 - (hi / peak) * (h / 2 - 4));
        ctx.lineTo(px + 0.5, h / 2 - (lo / peak) * (h / 2 - 4) + 0.5);
      }
      ctx.stroke();
    },
    [signal, start, length],
  );
  return <canvas ref={ref} className="chart chart-wave" aria-label="vibration waveform" />;
}

export interface Marker {
  hz: number;
  label: string;
  color: string;
  active: boolean;
}

export function EnvelopeChart({
  spectrum,
  sampleRate,
  window,
  markers,
  maxHz = 600,
}: {
  spectrum: Float64Array;
  sampleRate: number;
  window: number;
  markers: Marker[];
  maxHz?: number;
}) {
  const ref = useCanvas(
    (ctx, w, h) => {
      const df = sampleRate / window;
      const bins = Math.min(spectrum.length, Math.floor(maxHz / df));
      let peak = 1e-12;
      for (let k = 2; k < bins; k++) peak = Math.max(peak, spectrum[k] as number);
      const pad = 18;
      const x = (hz: number) => (hz / maxHz) * w;
      const y = (v: number) => h - pad - (Math.min(v, peak) / peak) * (h - pad - 10);

      for (const m of markers) {
        for (let hn = 1; hn <= 3; hn++) {
          const xx = x(m.hz * hn);
          if (xx > w) continue;
          ctx.strokeStyle = m.color;
          ctx.globalAlpha = m.active ? (hn === 1 ? 0.9 : 0.5) : 0.18;
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(xx, 8);
          ctx.lineTo(xx, h - pad);
          ctx.stroke();
          ctx.setLineDash([]);
          if (hn === 1) {
            ctx.fillStyle = m.color;
            ctx.font = "600 11px Inter, system-ui, sans-serif";
            ctx.fillText(m.label, Math.min(xx + 4, w - 40), 18);
          }
        }
      }
      ctx.globalAlpha = 1;

      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "rgba(34, 211, 238, 0.35)");
      grad.addColorStop(1, "rgba(34, 211, 238, 0)");
      ctx.beginPath();
      ctx.moveTo(0, h - pad);
      for (let k = 1; k < bins; k++) ctx.lineTo(x(k * df), y(spectrum[k] as number));
      ctx.lineTo(x(bins * df), h - pad);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      for (let k = 1; k < bins; k++) {
        const xx = x(k * df);
        const yy = y(spectrum[k] as number);
        if (k === 1) ctx.moveTo(xx, yy);
        else ctx.lineTo(xx, yy);
      }
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#64748b";
      ctx.font = "11px Inter, system-ui, sans-serif";
      for (let hz = 0; hz <= maxHz; hz += 100)
        ctx.fillText(`${hz} Hz`, Math.min(x(hz) + 2, w - 40), h - 4);
    },
    [spectrum, markers, sampleRate, window, maxHz],
  );
  return <canvas ref={ref} className="chart chart-env" aria-label="envelope spectrum" />;
}
