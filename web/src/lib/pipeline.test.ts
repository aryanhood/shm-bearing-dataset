import { describe, expect, it } from "vitest";
import modelJson from "../../public/data/model.json";
import golden from "./__fixtures__/golden.json";
import { decide } from "./decision";
import { analyze, envelopeSpectrum, faultFrequencies } from "./features";
import { fft } from "./fft";
import { type ModelJson, predictProba } from "./model";

interface Golden {
  id: number;
  rpm: number;
  window: number[];
  features: number[];
  proba: number[];
  envelope_head: number[];
}
const cases = golden as Golden[];
const model = modelJson as ModelJson;

const close = (a: number, b: number, tol = 1e-6) =>
  Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

describe("fft", () => {
  it("round-trips a signal", () => {
    const re = Float64Array.from({ length: 64 }, (_, i) => Math.sin(i) + i * 0.01);
    const orig = Float64Array.from(re);
    const im = new Float64Array(64);
    fft(re, im);
    fft(re, im, true);
    re.forEach((v, i) => expect(close(v, orig[i] as number, 1e-9)).toBe(true));
  });

  it("finds a pure tone in the right bin", () => {
    const re = Float64Array.from({ length: 256 }, (_, i) => Math.cos((2 * Math.PI * 10 * i) / 256));
    const im = new Float64Array(256);
    fft(re, im);
    expect(re[10]).toBeCloseTo(128, 6);
  });

  it("rejects non power-of-two input", () => {
    expect(() => fft(new Float64Array(3), new Float64Array(3))).toThrow();
  });
});

describe("parity with the Python pipeline", () => {
  it.each(cases.map((c) => [c.id, c] as const))("recording %i: features", (_, c) => {
    const { features } = analyze(c.window, c.rpm);
    features.forEach((v, i) => expect(close(v, c.features[i] as number)).toBe(true));
  });

  it.each(cases.map((c) => [c.id, c] as const))("recording %i: envelope spectrum", (_, c) => {
    const env = envelopeSpectrum(c.window);
    c.envelope_head.forEach((v, i) => expect(close(env[i] as number, v)).toBe(true));
  });

  it.each(cases.map((c) => [c.id, c] as const))("recording %i: class probabilities", (_, c) => {
    const p = predictProba(model, analyze(c.window, c.rpm).features);
    p.forEach((v, i) => expect(close(v, c.proba[i] as number)).toBe(true));
  });
});

describe("physics + decision", () => {
  it("computes CWRU defect frequencies", () => {
    const f = faultFrequencies(1797);
    expect(f.bpfi).toBeCloseTo(162.19, 1);
    expect(f.bpfo).toBeCloseTo(107.36, 1);
  });

  it("maps probabilities to a maintenance status", () => {
    expect(decide([0.95, 0.02, 0.02, 0.01]).status).toBe("SAFE");
    expect(decide([0.05, 0.9, 0.03, 0.02]).status).toBe("CRITICAL");
    expect(decide([0.4, 0.35, 0.15, 0.1]).status).toBe("WATCH");
  });

  it("rejects malformed windows", () => {
    expect(() => analyze([1, 2, 3], 1772)).toThrow();
  });
});
