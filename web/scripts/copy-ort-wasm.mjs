import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const webRoot = resolve(here, "..");
const srcDir = join(webRoot, "node_modules", "onnxruntime-web", "dist");
const outDir = join(webRoot, "public", "ort");

if (!existsSync(srcDir)) {
  console.error(`Missing onnxruntime-web dist folder at ${srcDir}`);
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });

let copied = 0;
for (const entry of readdirSync(srcDir)) {
  if (entry.startsWith("ort-wasm") && (entry.endsWith(".wasm") || entry.endsWith(".mjs"))) {
    copyFileSync(join(srcDir, entry), join(outDir, entry));
    copied += 1;
  }
}

console.log(`Copied ${copied} ONNX Runtime WASM files to ${outDir}`);
