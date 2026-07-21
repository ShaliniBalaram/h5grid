import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands directly in the Python package so `pip install h5grid` ships
// one artifact and the server can serve the SPA as static files.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../backend/h5grid/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
