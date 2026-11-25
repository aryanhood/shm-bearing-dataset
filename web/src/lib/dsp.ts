import type {
  BaselineDriftResult,
  BaselineProfile,
  HealthStatus,
  IsoSeverity,
  ParsedSignal,
  SignalAnalysisResult,
  SignalQuality,
} from "../types";

const G_TO_MS2 = 9.80665;
const EPS = 1e-12;
const N_BANDS = 8;
const ISO_GROUP2_RIGID_LIMITS: [number, number, number] = [1.4, 2.8, 4.5];

function isFiniteFloat(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;
  const val = Number(trimmed);
  return Number.isFinite(val);
}

export function parseCsvSignal(text: string, fallbackFs?: number): ParsedSignal {
  const lines = text
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));

  if (lines.length < 16) {
    throw new Error("CSV file needs at least 16 numeric samples.");
  }

  const firstLine = lines[0] ?? "";
  const semiCount = (firstLine.match(/;/g) ?? []).length;
  const commaCount = (firstLine.match(/,/g) ?? []).length;
  const delim = semiCount > commaCount ? ";" : ",";

  const rows = lines.map((line) => line.split(delim).map((cell) => cell.trim()));
  const firstRow = rows[0] ?? [];
  const hasHeader = !firstRow.every((cell) => cell === "" || isFiniteFloat(cell));
  const headers = hasHeader ? firstRow : firstRow.map((_, idx) => `c${idx}`);
  const dataRows = hasHeader ? rows.slice(1) : rows;

  const numericRows: number[][] = [];
  for (const row of dataRows) {
    if (row.length < headers.length) continue;
    const slice = row.slice(0, headers.length);
    if (!slice.every((cell) => isFiniteFloat(cell))) continue;
    numericRows.push(slice.map((cell) => Number(cell)));
  }

  if (numericRows.length < 16) {
    throw new Error("Not enough valid numeric rows found in CSV.");
  }

  const timeColIdx = headers.findIndex((h) => {
    const lower = h.toLowerCase();
    return lower.includes("time") || lower === "t";
  });

  let fsHz = fallbackFs && fallbackFs > 0 ? fallbackFs : 0;
  if (fsHz <= 0) {
    if (timeColIdx === -1) {
      throw new Error("No time column found — enter a sample rate (Hz) to analyze this CSV.");
    }
    const tFirst = numericRows[0]?.[timeColIdx] ?? 0;
    const tLast = numericRows[numericRows.length - 1]?.[timeColIdx] ?? 0;
    const span = tLast - tFirst;
    if (span <= 0) {
      throw new Error("Time column must increase monotonically.");
    }
    fsHz = (numericRows.length - 1) / span;
  }

  const candidateCols: number[] = [];
  for (let c = 0; c < headers.length; c += 1) {
    if (c !== timeColIdx) candidateCols.push(c);
  }
  if (candidateCols.length === 0) {
    throw new Error("CSV must include at least one signal column.");
  }

  let bestCol = candidateCols[0] ?? 0;
  let bestVar = -1;
  for (const col of candidateCols) {
    let mean = 0;
    for (const row of numericRows) {
      mean += row[col] ?? 0;
    }
    mean /= numericRows.length;
    let sumSq = 0;
    for (const row of numericRows) {
      const diff = (row[col] ?? 0) - mean;
      sumSq += diff * diff;
    }
    if (sumSq > bestVar) {
      bestVar = sumSq;
      bestCol = col;
    }
  }

  const values = new Float64Array(numericRows.length);
  for (let i = 0; i < numericRows.length; i += 1) {
    values[i] = numericRows[i]?.[bestCol] ?? 0;
  }

  return {
    values,
    fsHz,
    units: "g",
    sourceName: "uploaded.csv",
    channel: headers[bestCol] ?? "signal",
    durationS: values.length / fsHz,
  };
}

export function parseWavSignal(buffer: ArrayBuffer, sourceName = "uploaded.wav"): ParsedSignal {
  if (buffer.byteLength < 44) {
    throw new Error("File is too small to be a valid WAV recording.");
  }
  const view = new DataView(buffer);
  const readAscii = (offset: number, len: number): string => {
    let out = "";
    for (let i = 0; i < len; i += 1) {
      out += String.fromCharCode(view.getUint8(offset + i));
    }
    return out;
  };

  if (readAscii(0, 4) !== "RIFF" || readAscii(8, 4) !== "WAVE") {
    throw new Error("Unsupported audio header: expected RIFF/WAVE.");
  }

  let offset = 12;
  let audioFormat = 0;
  let numChannels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buffer.byteLength) {
    const chunkId = readAscii(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const payloadStart = offset + 8;

    if (chunkId === "fmt " && chunkSize >= 16 && payloadStart + 16 <= buffer.byteLength) {
      audioFormat = view.getUint16(payloadStart, true);
      numChannels = view.getUint16(payloadStart + 2, true);
      sampleRate = view.getUint32(payloadStart + 4, true);
      bitsPerSample = view.getUint16(payloadStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = payloadStart;
      dataSize = Math.min(chunkSize, buffer.byteLength - payloadStart);
      break;
    }

    offset = payloadStart + chunkSize + (chunkSize % 2);
  }

  if (dataOffset < 0 || sampleRate <= 0 || numChannels <= 0) {
    throw new Error("Invalid WAV structure: missing fmt or data chunk.");
  }
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(`Unsupported WAV encoding format (${audioFormat}); expected PCM or IEEE float.`);
  }

  const bytesPerSample = Math.floor(bitsPerSample / 8);
  const frameBytes = bytesPerSample * numChannels;
  if (frameBytes <= 0) {
    throw new Error("Unsupported WAV bit depth.");
  }
  const nFrames = Math.floor(dataSize / frameBytes);
  if (nFrames < 16) {
    throw new Error("WAV recording has too few samples.");
  }

  const values = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f += 1) {
    let sum = 0;
    const frameBase = dataOffset + f * frameBytes;
    for (let ch = 0; ch < numChannels; ch += 1) {
      const pos = frameBase + ch * bytesPerSample;
      let sample = 0;
      if (audioFormat === 3 && bitsPerSample === 32) {
        sample = view.getFloat32(pos, true);
      } else if (bitsPerSample === 8) {
        sample = (view.getUint8(pos) - 128) / 128.0;
      } else if (bitsPerSample === 16) {
        sample = view.getInt16(pos, true) / 32768.0;
      } else if (bitsPerSample === 24) {
        const b0 = view.getUint8(pos);
        const b1 = view.getUint8(pos + 1);
        const b2 = view.getUint8(pos + 2);
        let raw = b0 | (b1 << 8) | (b2 << 16);
        if (raw & 0x800000) raw |= ~0xffffff;
        sample = raw / 8388608.0;
      } else if (bitsPerSample === 32) {
        sample = view.getInt32(pos, true) / 2147483648.0;
      } else {
        throw new Error(`Unsupported WAV bit depth: ${bitsPerSample}-bit`);
      }
      sum += sample;
    }
    values[f] = sum / numChannels;
  }

  return {
    values,
    fsHz: sampleRate,
    units: "audio",
    sourceName,
    channel: "mono",
    durationS: nFrames / sampleRate,
  };
}

export function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Iterative radix-2 Cooley-Tukey FFT returning one-sided magnitude spectrum (0..N/2).
 */
export function radix2FftMagnitude(input: Float64Array): Float64Array {
  const n = nextPowerOfTwo(input.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(input);

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tmpRe = re[i] ?? 0;
      re[i] = re[j] ?? 0;
      re[j] = tmpRe;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2.0 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    const half = len >> 1;

    for (let i = 0; i < n; i += len) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let k = 0; k < half; k += 1) {
        const evenIdx = i + k;
        const oddIdx = evenIdx + half;
        const oRe = re[oddIdx] ?? 0;
        const oIm = im[oddIdx] ?? 0;
        const tRe = wRe * oRe - wIm * oIm;
        const tIm = wRe * oIm + wIm * oRe;
        const eRe = re[evenIdx] ?? 0;
        const eIm = im[evenIdx] ?? 0;

        re[evenIdx] = eRe + tRe;
        im[evenIdx] = eIm + tIm;
        re[oddIdx] = eRe - tRe;
        im[oddIdx] = eIm - tIm;

        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }

  const halfN = (n >> 1) + 1;
  const mag = new Float64Array(halfN);
  for (let k = 0; k < halfN; k += 1) {
    const rk = re[k] ?? 0;
    const ik = im[k] ?? 0;
    mag[k] = Math.hypot(rk, ik);
  }
  return mag;
}

export function auditSignal(values: Float64Array): SignalQuality {
  if (values.length === 0) {
    return {
      ok: false,
      clippedRatio: 1,
      dropoutRatio: 1,
      dcOffset: 0,
      issues: ["Signal is empty."],
    };
  }

  let sum = 0;
  let peak = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] ?? 0;
    if (!Number.isFinite(v)) {
      return {
        ok: false,
        clippedRatio: 1,
        dropoutRatio: 1,
        dcOffset: 0,
        issues: ["Signal contains NaN or infinite values."],
      };
    }
    sum += v;
    const absV = Math.abs(v);
    if (absV > peak) peak = absV;
  }

  const dc = sum / values.length;
  let sqSum = 0;
  let dropCount = 0;
  let clipCount = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] ?? 0;
    const centered = v - dc;
    sqSum += centered * centered;
    if (i > 0 && Math.abs(v - (values[i - 1] ?? 0)) < 1e-9) {
      dropCount += 1;
    }
    if (peak > 0 && Math.abs(v) >= 0.998 * peak) {
      clipCount += 1;
    }
  }

  const rms = Math.sqrt(sqSum / values.length);
  const dropout = values.length > 1 ? dropCount / (values.length - 1) : 1.0;
  const clipped = peak > 0 ? clipCount / values.length : 0.0;
  const issues: string[] = [];
  if (rms < 1e-7 || dropout > 0.3) {
    issues.push("Signal looks flat or disconnected.");
  }
  if (clipped > 0.01) {
    issues.push("Sensor clipping detected.");
  }

  return {
    ok: issues.length === 0,
    clippedRatio: Number(clipped.toFixed(5)),
    dropoutRatio: Number(dropout.toFixed(5)),
    dcOffset: Number(dc.toFixed(6)),
    issues,
  };
}

export function computeIsoVelocity(
  values: Float64Array,
  fsHz: number,
  units: "g" | "m/s2" | "audio",
): IsoSeverity | null {
  if ((units !== "g" && units !== "m/s2") || fsHz < 40.0 || values.length < 16) {
    return null;
  }

  let mean = 0;
  for (let i = 0; i < values.length; i += 1) {
    mean += values[i] ?? 0;
  }
  mean /= values.length;

  const scale = units === "g" ? G_TO_MS2 : 1.0;
  const acc = new Float64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    acc[i] = ((values[i] ?? 0) - mean) * scale;
  }

  const nOrig = acc.length;
  const nFft = nextPowerOfTwo(nOrig);
  const mag = radix2FftMagnitude(acc);
  const fHi = Math.min(1000.0, 0.45 * fsHz);

  // Parseval's theorem across zero-padded FFT bins inside [10 Hz, fHi]
  let powerSum = 0;
  const lastBin = mag.length - 1;
  for (let k = 1; k <= lastBin; k += 1) {
    const freq = (k * fsHz) / nFft;
    if (freq < 10.0 || freq > fHi) continue;
    const velMag = (mag[k] ?? 0) / (2.0 * Math.PI * freq);
    const weight = k === lastBin ? 1.0 : 2.0;
    powerSum += weight * velMag * velMag;
  }

  const meanSqVel = powerSum / (nOrig * nFft);
  const vRmsMmS = 1000.0 * Math.sqrt(Math.max(0, meanSqVel));
  const rounded = Number(vRmsMmS.toFixed(3));

  const [limAB, limBC, limCD] = ISO_GROUP2_RIGID_LIMITS;
  let zone: "A" | "B" | "C" | "D" = "D";
  if (rounded < limAB) zone = "A";
  else if (rounded < limBC) zone = "B";
  else if (rounded < limCD) zone = "C";

  return {
    velocityRmsMmS: rounded,
    zone,
    limitsMmS: ISO_GROUP2_RIGID_LIMITS,
  };
}

function medianOf(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid] ?? 0;
  return 0.5 * ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0));
}

function windowFeatures(values: Float64Array, fsHz: number): number[][] {
  let mean = 0;
  for (let i = 0; i < values.length; i += 1) {
    mean += values[i] ?? 0;
  }
  mean /= Math.max(1, values.length);

  const nTotal = values.length;
  const targetWin = Math.max(128, Math.min(Math.floor(0.25 * fsHz), Math.floor(nTotal / 6)));
  const nWin = Math.max(1, Math.floor(nTotal / targetWin));
  const winLen = Math.min(targetWin, nTotal);

  const hann = new Float64Array(winLen);
  if (winLen > 1) {
    for (let i = 0; i < winLen; i += 1) {
      hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (winLen - 1)));
    }
  } else {
    hann[0] = 1.0;
  }

  const rows: number[][] = [];
  for (let wIdx = 0; wIdx < nWin; wIdx += 1) {
    const start = wIdx * winLen;
    let wMean = 0;
    for (let i = 0; i < winLen; i += 1) {
      wMean += (values[start + i] ?? 0) - mean;
    }
    wMean /= winLen;

    const frame = new Float64Array(winLen);
    let sq = 0;
    let absSum = 0;
    let maxAbs = 0;
    let pow4 = 0;

    for (let i = 0; i < winLen; i += 1) {
      const val = (values[start + i] ?? 0) - mean - wMean;
      frame[i] = val * (hann[i] ?? 0);
      const av = Math.abs(val);
      const v2 = val * val;
      sq += v2;
      pow4 += v2 * v2;
      absSum += av;
      if (av > maxAbs) maxAbs = av;
    }

    const rms = Math.sqrt(sq / winLen) + EPS;
    const kurt = pow4 / winLen / rms ** 4;
    const crest = maxAbs / rms;
    const shape = rms / (absSum / winLen + EPS);

    const mag = radix2FftMagnitude(frame);
    let totalP = 0;
    for (let k = 0; k < mag.length; k += 1) {
      const m = mag[k] ?? 0;
      totalP += m * m;
    }
    totalP += EPS;

    const width = Math.max(1, Math.floor(mag.length / N_BANDS));
    const bands: number[] = [];
    for (let b = 0; b < N_BANDS; b += 1) {
      const bStart = b * width;
      const bEnd = Math.min(mag.length, (b + 1) * width);
      let bandSum = 0;
      for (let k = bStart; k < bEnd; k += 1) {
        const m = mag[k] ?? 0;
        bandSum += m * m;
      }
      bands.push(Math.log10(bandSum / totalP + EPS));
    }

    rows.push([Math.log10(rms), kurt, crest, shape, ...bands]);
  }
  return rows;
}

export function scoreAgainstBaseline(
  values: Float64Array,
  fsHz: number,
  baseline: BaselineProfile,
): BaselineDriftResult {
  const feats = windowFeatures(values, fsHz);
  const med = baseline.feature_median;
  const scale = baseline.feature_scale;

  const rowDists: number[] = [];
  for (const row of feats) {
    let zSqSum = 0;
    const dim = Math.min(row.length, med.length, scale.length);
    for (let c = 0; c < dim; c += 1) {
      const z = Math.abs((row[c] ?? 0) - (med[c] ?? 0)) / Math.max(scale[c] ?? 1e-6, 1e-6);
      zSqSum += z * z;
    }
    rowDists.push(Math.sqrt(zSqSum / Math.max(1, dim)));
  }

  const medianDist = medianOf(rowDists);
  const rawScore = 100.0 * (1.0 - Math.exp(-0.24 * Math.max(0.0, medianDist - 0.5)));
  const driftScore = Number(Math.min(100.0, Math.max(0.0, rawScore)).toFixed(2));

  let mean = 0;
  for (let i = 0; i < values.length; i += 1) mean += values[i] ?? 0;
  mean /= Math.max(1, values.length);

  let sq = 0;
  let pow4 = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = (values[i] ?? 0) - mean;
    const v2 = v * v;
    sq += v2;
    pow4 += v2 * v2;
  }
  const rms = Math.sqrt(sq / Math.max(1, values.length));
  const kurt = pow4 / Math.max(1, values.length) / (rms + EPS) ** 4;

  let specDiffSum = 0;
  const bandCount = Math.max(1, (med.length || 12) - 4);
  for (let b = 4; b < med.length; b += 1) {
    const colVals = feats.map((r) => r[b] ?? 0);
    specDiffSum += Math.abs(medianOf(colVals) - (med[b] ?? 0));
  }

  return {
    driftScore,
    rawDistance: Number(medianDist.toFixed(4)),
    rmsRatio: Number((rms / (baseline.healthy_rms + EPS)).toFixed(3)),
    kurtosisDelta: Number((kurt - baseline.healthy_kurtosis).toFixed(3)),
    spectralShift: Number((specDiffSum / bandCount).toFixed(4)),
  };
}

export function analyzeUploadedSignal(sig: ParsedSignal, baseline: BaselineProfile): SignalAnalysisResult {
  const quality = auditSignal(sig.values);
  const iso = computeIsoVelocity(sig.values, sig.fsHz, sig.units);
  const drift = scoreAgainstBaseline(sig.values, sig.fsHz, baseline);

  let mean = 0;
  for (let i = 0; i < sig.values.length; i += 1) mean += sig.values[i] ?? 0;
  mean /= Math.max(1, sig.values.length);

  let sq = 0;
  let pow4 = 0;
  let maxAbs = 0;
  for (let i = 0; i < sig.values.length; i += 1) {
    const v = (sig.values[i] ?? 0) - mean;
    const v2 = v * v;
    sq += v2;
    pow4 += v2 * v2;
    if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  }
  const rms = Math.sqrt(sq / Math.max(1, sig.values.length));
  const kurtosis = pow4 / Math.max(1, sig.values.length) / (rms + EPS) ** 4;
  const crestFactor = maxAbs / (rms + EPS);

  // Waveform preview (~240 points)
  const targetWavePts = 240;
  const step = Math.max(1, Math.floor(sig.values.length / targetWavePts));
  const waveformPreview: { x: number; y: number }[] = [];
  for (let i = 0; i < sig.values.length; i += step) {
    waveformPreview.push({ x: i / sig.fsHz, y: sig.values[i] ?? 0 });
  }

  // Spectrum preview up to 5 kHz / Nyquist using first 2048 centered Hann samples
  const fftLen = Math.min(2048, nextPowerOfTwo(Math.min(sig.values.length, 2048)));
  const fftSlice = new Float64Array(fftLen);
  for (let i = 0; i < fftLen; i += 1) {
    const hannVal = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / Math.max(1, fftLen - 1)));
    fftSlice[i] = ((sig.values[i] ?? 0) - mean) * hannVal;
  }
  const specMag = radix2FftMagnitude(fftSlice);
  const nyq = 0.5 * sig.fsHz;
  const maxShowHz = Math.min(nyq, 5500);
  const specBins: { freqHz: number; magnitude: number }[] = [];
  const stride = Math.max(1, Math.floor(specMag.length / 120));
  for (let k = 1; k < specMag.length; k += stride) {
    const freqHz = (k * sig.fsHz) / (2 * (specMag.length - 1));
    if (freqHz > maxShowHz) break;
    specBins.push({ freqHz, magnitude: (specMag[k] ?? 0) / fftLen });
  }

  const isoTrapTriggered = iso !== null && iso.zone === "A" && drift.driftScore >= 35;

  let status: HealthStatus = "HEALTHY";
  let headline = "Vibration looks normal compared to the healthy motor baseline";
  let actionText = "No abnormal shock energy or frequency shift detected.";

  if (!quality.ok) {
    status = "WATCH";
    headline = "Sensor signal issue detected — check wiring before trusting reading";
    actionText = quality.issues.join(" ");
  } else if (drift.driftScore >= 65 || (iso !== null && (iso.zone === "C" || iso.zone === "D"))) {
    status = "STOP_AND_INSPECT";
    headline = "High vibration alert — inspect motor bearing";
    actionText = "Strong shock energy and high-frequency shift detected relative to healthy baseline.";
  } else if (drift.driftScore >= 30 || (iso !== null && iso.zone === "B")) {
    status = "WATCH";
    headline = "Early wear pattern detected — high-frequency drift above baseline";
    actionText = isoTrapTriggered
      ? "Standard low-frequency ISO speed check still reads Zone A (healthy), while high-frequency tracking catches early bearing shocks."
      : "Energy and shock sharpness climbed above the healthy baseline. Plan an inspection.";
  }

  return {
    signal: sig,
    quality,
    iso,
    drift,
    rms: Number(rms.toFixed(4)),
    kurtosis: Number(kurtosis.toFixed(2)),
    crestFactor: Number(crestFactor.toFixed(2)),
    status,
    headline,
    actionText,
    isoTrapTriggered,
    waveformPreview,
    spectrumPreview: specBins,
  };
}
