import type { ModelJson } from "./model";

export interface Recording {
  id: number;
  label: string;
  title: string;
  diameter: string;
  load_hp: number;
  rpm: number;
  offset: number;
  length: number;
  scale: number;
}

export interface DemoMeta {
  sample_rate: number;
  window: number;
  classes: string[];
  class_titles: string[];
  train_loads_hp: number[];
  demo_load_hp: number;
  recordings: Recording[];
}

export interface Demo {
  meta: DemoMeta;
  model: ModelJson;
  signals: Float64Array[];
}

export async function loadDemo(base: string): Promise<Demo> {
  const [meta, model, raw] = await Promise.all([
    fetch(`${base}data/demo.json`).then((r) => r.json() as Promise<DemoMeta>),
    fetch(`${base}data/model.json`).then((r) => r.json() as Promise<ModelJson>),
    fetch(`${base}data/demo.i16`).then((r) => r.arrayBuffer()),
  ]);
  const view = new DataView(raw);
  const signals = meta.recordings.map((rec) => {
    const out = new Float64Array(rec.length);
    for (let i = 0; i < rec.length; i++) {
      out[i] = view.getInt16((rec.offset + i) * 2, true) * rec.scale;
    }
    return out;
  });
  return { meta, model, signals };
}

/** Parse a CSV / whitespace-separated list of numbers (first numeric column). */
export function parseSignal(text: string): Float64Array {
  const values: number[] = [];
  for (const line of text.split(/\r?\n/)) {
    const cell = line.split(/[,;\t ]+/).find((c) => c.trim() !== "");
    if (cell === undefined) continue;
    const v = Number(cell);
    if (Number.isFinite(v)) values.push(v);
  }
  return Float64Array.from(values);
}
