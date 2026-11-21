import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Served from https://<user>.github.io/shm-bearing-dataset/ on GitHub Pages.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [react()],
});
