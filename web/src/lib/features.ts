// TypeScript port of bearingscope/features.py. Parity is enforced by features.test.ts.
import { fft } from "./fft";

export const SAMPLE_RATE = 12_000;
export const WINDOW = 2048;
const N_BANDS = 8;
const ENVELOPE_BAND_HZ = [1500, 5500] as const;
const HARMONICS = [1, 2, 3];
const PEAK_TOLERANCE_BINS = 1;
const RING_BINS = 8;
const EPS = 1e-12;

export const ORDERS = { bpfi: 5.4152, bpfo: 3.5848, bsf: 2.357 } as const;

export interface FaultFrequencies {
  shaft: number;
  bpfi: number;
  bpfo: number;
  bsf: number;
}

export function faultFrequencies(rpm: number): FaultFrequencies {
  const shaft = rpm / 60;
  return {
    shaft,
    bpfi: ORDERS.bpfi * shaft,
    bpfo: ORDERS.bpfo * shaft,
    bsf: ORDERS.bsf * shaft,
  };
}

function centered(window: ArrayLike<number>): Float64Array {
  if (window.length !== WINDOW) {
    throw new Error(`expected a window of ${WINDOW} samples, got ${window.length}`);
  }
  let mean = 0;
  for (let i = 0; i < WINDOW; i++) {
    const v = window[i] as number;
    if (!Number.isFinite(v)) throw new Error("window contains NaN or inf");
    mean += v;
  }
  mean /= WINDOW;
  const x = new Float64Array(WINDOW);
  for (let i = 0; i < WINDOW; i++) x[i] = (window[i] as number) - mean;
  return x;
}

/** Band-pass around the structural resonance, demodulate, and return the envelope spectrum. */
export function envelopeSpectrum(window: ArrayLike<number>): Float64Array {
  const n = WINDOW;
  const re = centered(window);
  const im = new Float64Array(n);
  fft(re, im);
  const lo = Math.round((ENVELOPE_BAND_HZ[0] / SAMPLE_RATE) * n);
  const hi = Math.round((ENVELOPE_BAND_HZ[1] / SAMPLE_RATE) * n);
  const ar = new Float64Array(n);
  const ai = new Float64Array(n);
  for (let k = lo; k <= hi; k++) {
    ar[k] = 2 * (re[k] as number);
    ai[k] = 2 * (im[k] as number);
  }
  fft(ar, ai, true);
  const env = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    env[i] = Math.hypot(ar[i] as number, ai[i] as number);
    mean += env[i] as number;
  }
  mean /= n;
  for (let i = 0; i < n; i++) env[i] = (env[i] as number) - mean;
  const ei = new Float64Array(n);
  fft(env, ei);
  const mag = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) mag[k] = Math.hypot(env[k] as number, ei[k] as number);
  return mag;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
}

/** Python's round() is banker's rounding; match it for exact bin parity. */
function pyRound(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function peakScore(env: Float64Array, freqHz: number): number {
  const df = SAMPLE_RATE / WINDOW;
  let total = 0;
  for (const h of HARMONICS) {
    const k = pyRound((h * freqHz) / df);
    let core = Number.NEGATIVE_INFINITY;
    for (let i = k - PEAK_TOLERANCE_BINS; i <= k + PEAK_TOLERANCE_BINS; i++) {
      core = Math.max(core, env[i] as number);
    }
    const ring: number[] = [];
    for (let i = k - PEAK_TOLERANCE_BINS - RING_BINS; i < k - PEAK_TOLERANCE_BINS; i++) {
      ring.push(env[i] as number);
    }
    for (let i = k + PEAK_TOLERANCE_BINS + 1; i < k + PEAK_TOLERANCE_BINS + 1 + RING_BINS; i++) {
      ring.push(env[i] as number);
    }
    total += Math.log10(core / (median(ring) + EPS) + EPS);
  }
  return total;
}

export interface Analysis {
  features: number[];
  envelope: Float64Array;
  faults: FaultFrequencies;
  kurtosis: number;
  rms: number;
}

export function analyze(window: ArrayLike<number>, rpm: number): Analysis {
  const n = WINDOW;
  const x = centered(window);
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let sAbs = 0;
  let peak = 0;
  for (let i = 0; i < n; i++) {
    const v = x[i] as number;
    s2 += v * v;
    s3 += v * v * v;
    s4 += v * v * v * v;
    sAbs += Math.abs(v);
    peak = Math.max(peak, Math.abs(v));
  }
  const rms = Math.sqrt(s2 / n) + EPS;
  const kurtosis = s4 / n / rms ** 4;
  const time = [Math.log10(rms), kurtosis, peak / rms, s3 / n / rms ** 3, rms / (sAbs / n + EPS)];

  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++)
    re[i] = (x[i] as number) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  fft(re, im);
  const power = new Float64Array(n / 2);
  let total = 0;
  for (let k = 0; k < n / 2; k++) {
    power[k] = (re[k] as number) ** 2 + (im[k] as number) ** 2;
    total += power[k] as number;
  }
  total += EPS;
  const width = n / 2 / N_BANDS;
  const bands: number[] = [];
  for (let b = 0; b < N_BANDS; b++) {
    let e = 0;
    for (let k = b * width; k < (b + 1) * width; k++) e += power[k] as number;
    bands.push(Math.log10(e / total + EPS));
  }

  const envelope = envelopeSpectrum(x);
  const faults = faultFrequencies(rpm);
  const env = [faults.bpfi, faults.bpfo, faults.bsf].map((f) => peakScore(envelope, f));
  return { features: [...time, ...bands, ...env], envelope, faults, kurtosis, rms };
}
