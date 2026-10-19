import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  publicDir: "public",
  plugins: [react()],
  worker: { rollupOptions: { output: { entryFileNames: "assets/[name]-solver-r2-[hash].js" } } },
});
