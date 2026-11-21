// Executes the JSON model exported by bearingscope/model.py: standardize -> MLP (ReLU) -> softmax.

export interface ModelJson {
  classes: string[];
  features: string[];
  mean: number[];
  scale: number[];
  layers: { weights: number[][]; bias: number[] }[];
}

export function predictProba(model: ModelJson, features: number[]): number[] {
  if (features.length !== model.mean.length) {
    throw new Error(`model expects ${model.mean.length} features, got ${features.length}`);
  }
  let h = features.map((v, i) => (v - (model.mean[i] as number)) / (model.scale[i] as number));
  model.layers.forEach((layer, li) => {
    const out = [...layer.bias];
    for (let i = 0; i < h.length; i++) {
      const row = layer.weights[i] as number[];
      const hi = h[i] as number;
      for (let j = 0; j < out.length; j++) out[j] = (out[j] as number) + hi * (row[j] as number);
    }
    h = li < model.layers.length - 1 ? out.map((v) => Math.max(v, 0)) : out;
  });
  const max = Math.max(...h);
  const exp = h.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}
