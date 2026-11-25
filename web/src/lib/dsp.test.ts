import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { MotorSamplesPayload } from "../types";
import {
  analyzeUploadedSignal,
  computeIsoVelocity,
  parseCsvSignal,
  parseWavSignal,
  scoreAgainstBaseline,
} from "./dsp";

function buildPcm16Wav(samples: number[], sampleRate: number): ArrayBuffer {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i += 1) {
      view.setUint8(offset + i, s.charCodeAt(i));
    }
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i] ?? 0));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  return buf;
}

describe("Browser DSP parity", () => {
  it("computes analytical ISO 20816-3 velocity for a 50 Hz sine wave", () => {
    const fs = 12_000;
    const n = 12_000;
    const vTargetRmsMmS = 3.0;
    const aAmpG = ((vTargetRmsMmS / 1000.0) * Math.SQRT2 * (2.0 * Math.PI * 50.0)) / 9.80665;
    const values = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      values[i] = aAmpG * Math.sin((2.0 * Math.PI * 50.0 * i) / fs);
    }
    const iso = computeIsoVelocity(values, fs, "g");
    expect(iso).not.toBeNull();
    expect(iso?.zone).toBe("C");
    expect(Math.abs((iso?.velocityRmsMmS ?? 0) - vTargetRmsMmS)).toBeLessThan(0.1);
  });

  it("parses WAV headers and normalizes PCM16 audio", () => {
    const fs = 8000;
    const samples = Array.from({ length: 400 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 200 * i) / fs));
    const buf = buildPcm16Wav(samples, fs);
    const sig = parseWavSignal(buf, "test.wav");
    expect(sig.fsHz).toBe(fs);
    expect(sig.units).toBe("audio");
    expect(sig.values.length).toBe(400);
    expect(computeIsoVelocity(sig.values, sig.fsHz, sig.units)).toBeNull();
  });

  it("matches Python baseline drift & ISO Zone A trap on real motor CSV examples", () => {
    const publicDir = resolve(process.cwd(), "public");
    const samplesJson = JSON.parse(
      readFileSync(resolve(publicDir, "data/motor_samples.json"), "utf-8"),
    ) as MotorSamplesPayload;
    const healthyCsv = readFileSync(resolve(publicDir, "examples/motor_healthy.csv"), "utf-8");
    const faultCsv = readFileSync(resolve(publicDir, "examples/motor_inner_fault.csv"), "utf-8");

    const healthySig = parseCsvSignal(healthyCsv);
    const faultSig = parseCsvSignal(faultCsv);

    const healthyRes = analyzeUploadedSignal(healthySig, samplesJson.healthy_baseline);
    const faultRes = analyzeUploadedSignal(faultSig, samplesJson.healthy_baseline);

    expect(healthyRes.iso?.zone).toBe("A");
    expect(faultRes.iso?.zone).toBe("A");
    expect(healthyRes.drift.driftScore).toBeLessThan(5);
    expect(faultRes.drift.driftScore).toBeGreaterThan(40);
    expect(faultRes.isoTrapTriggered).toBe(true);

    const scoreCheck = scoreAgainstBaseline(faultSig.values, faultSig.fsHz, samplesJson.healthy_baseline);
    expect(scoreCheck.driftScore).toBeGreaterThan(healthyRes.drift.driftScore);
  });
});
