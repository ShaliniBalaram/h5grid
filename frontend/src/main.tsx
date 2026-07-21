import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";

import "@glideapps/glide-data-grid/dist/index.css";
// uPlot's own stylesheet, not a hand-written subset: it carries the rule that
// constrains the canvas to its CSS box, without which the canvas renders at
// devicePixelRatio scale and spills out of its container on any HiDPI screen.
// styles.css is imported after so the theme overrides in it win.
import "uplot/dist/uPlot.min.css";
import "./styles.css";

import App from "./App";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // File contents cannot change under us without the mtime guard firing,
      // so cached pages stay valid and refetching on focus is pure waste.
      refetchOnWindowFocus: false,
      retry: false,
      staleTime: 5 * 60 * 1000,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
