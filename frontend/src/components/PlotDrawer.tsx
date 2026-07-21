import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import uPlot from "uplot";

import { api } from "../api/client";
import type { NodeMeta } from "../api/types";

interface Props {
  fileId: string;
  meta: NodeMeta;
  slice?: string;
  useTimeIndex: boolean;
  onClose: () => void;
}

const PALETTE = [
  "#2563cf",
  "#17784a",
  "#c0392b",
  "#8e44ad",
  "#b8860b",
  "#0e7490",
  "#c2410c",
  "#4d5e80",
];

// uPlot's `height` covers the axes as well as the data area, so a small number
// here leaves almost nothing to draw in. The plot is sized from its container
// instead, with this as the floor.
const MIN_PLOT_HEIGHT = 150;
// More lines than this stop being readable and the palette starts repeating.
const MAX_SERIES = 12;
// What the legend below the canvas occupies, kept out of the height handed to
// uPlot so the drawer does not overflow. The legend is capped in CSS, so this
// stays accurate however many series are showing.
const LEGEND_ALLOWANCE = 46;

export default function PlotDrawer({
  fileId,
  meta,
  slice,
  useTimeIndex,
  onClose,
}: Props) {
  const plottable = meta.columns.filter((c) => !c.is_datetime);
  const [selected, setSelected] = useState<string[]>(() =>
    plottable.slice(0, 3).map((c) => c.name),
  );
  const [pickerOpen, setPickerOpen] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    setSelected(meta.columns.filter((c) => !c.is_datetime).slice(0, 3).map((c) => c.name));
    setFilter("");
  }, [meta.path, meta.columns]);

  const indices = selected
    .map((name) => meta.columns.findIndex((c) => c.name === name))
    .filter((i) => i >= 0);

  // The visible row window. null means the whole dataset. Zooming narrows this
  // and re-requests, so zooming in genuinely reveals more detail rather than
  // magnifying the points already on screen.
  const [range, setRange] = useState<{ start: number; stop: number } | null>(null);
  useEffect(() => setRange(null), [meta.path, slice]);

  const { data, isFetching, error } = useQuery({
    queryKey: [
      "plot",
      fileId,
      meta.path,
      slice ?? "",
      indices.join(","),
      useTimeIndex,
      range?.start ?? 0,
      range?.stop ?? -1,
    ],
    queryFn: () =>
      api.plotData(fileId, meta.path, {
        cols: indices.join(","),
        max_points: 4000,
        slice,
        use_time_index: useTimeIndex,
        start: range?.start,
        stop: range?.stop,
      }),
    enabled: indices.length > 0,
    placeholderData: (previous) => previous,
  });

  const totalRows = data?.total_rows ?? meta.nrows;

  /** Narrow or widen the window about its centre. factor < 1 zooms in. */
  const zoomBy = useCallback(
    (factor: number) => {
      setRange((current) => {
        const start = current?.start ?? 0;
        const stop = current?.stop ?? totalRows;
        const centre = (start + stop) / 2;
        const half = ((stop - start) * factor) / 2;
        if (half >= totalRows / 2) return null;
        const next = {
          start: Math.max(0, Math.round(centre - half)),
          stop: Math.min(totalRows, Math.round(centre + half)),
        };
        // Two points is the least that still draws a line.
        return next.stop - next.start < 2 ? current : next;
      });
    },
    [totalRows],
  );

  /** Zoom about a fraction across the current window, for wheel zooming. */
  const zoomAt = useCallback(
    (factor: number, atFraction: number) => {
      setRange((current) => {
        const start = current?.start ?? 0;
        const stop = current?.stop ?? totalRows;
        const width = stop - start;
        const anchor = start + width * Math.min(1, Math.max(0, atFraction));
        const nextWidth = width * factor;
        if (nextWidth >= totalRows) return null;
        if (nextWidth < 2) return current;
        let nextStart = Math.round(anchor - (anchor - start) * factor);
        nextStart = Math.max(0, Math.min(totalRows - Math.round(nextWidth), nextStart));
        return { start: nextStart, stop: Math.round(nextStart + nextWidth) };
      });
    },
    [totalRows],
  );

  const panBy = useCallback(
    (fraction: number) => {
      setRange((current) => {
        if (!current) return current;
        const width = current.stop - current.start;
        const shift = Math.round(width * fraction);
        const start = Math.max(0, Math.min(totalRows - width, current.start + shift));
        return { start, stop: start + width };
      });
    },
    [totalRows],
  );

  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [size, setSize] = useState({ width: 600, height: MIN_PLOT_HEIGHT });

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const measure = (width: number, height: number) =>
      setSize((previous) => {
        const next = {
          width: Math.max(240, Math.round(width)),
          height: Math.max(MIN_PLOT_HEIGHT, Math.round(height) - LEGEND_ALLOWANCE),
        };
        // Same box, same object: rebuilding the plot on every notification
        // would throw away the user's zoom.
        return previous.width === next.width && previous.height === next.height
          ? previous
          : next;
      });

    // Measure once, synchronously, so the first plot is drawn at the right size
    // rather than at the fallback while waiting for the observer — which is
    // delivered with the rendering steps and may not arrive promptly.
    const box = host.getBoundingClientRect();
    if (box.width > 0 && box.height > 0) measure(box.width, box.height);

    const observer = new ResizeObserver(([entry]) =>
      measure(entry.contentRect.width, entry.contentRect.height),
    );
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    plotRef.current?.destroy();
    plotRef.current = null;
    if (!host || !data || data.series.length === 0) return;

    const css = getComputedStyle(document.documentElement);
    const axisColor = css.getPropertyValue("--text-faint").trim();
    const gridColor = css.getPropertyValue("--border").trim();

    // uPlot wants numbers: dates become epoch seconds, and the string forms of
    // Infinity that keep the JSON valid become gaps rather than fake spikes.
    const x = data.x_is_date
      ? data.x.map((value) => new Date(String(value)).getTime() / 1000)
      : (data.x as number[]);

    const series = data.series.map((s) =>
      s.y.map((value) => (typeof value === "number" ? value : null)),
    );

    /** Nearest plotted point to an x value, as a source row number. */
    const rowAt = (value: number) => {
      let lo = 0;
      let hi = x.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (x[mid] < value) lo = mid + 1;
        else hi = mid;
      }
      return data.rows[lo] ?? 0;
    };

    plotRef.current = new uPlot(
      {
        width: size.width,
        height: size.height,
        // setScale off: a drag selects a range, and the handler below turns it
        // into a new row window and refetches, so zooming in fetches finer
        // data rather than magnifying the points already drawn.
        cursor: { drag: { x: true, y: false, setScale: false } },
        hooks: {
          setSelect: [
            (u) => {
              if (u.select.width < 4) return;
              const from = rowAt(u.posToVal(u.select.left, "x"));
              const to = rowAt(u.posToVal(u.select.left + u.select.width, "x"));
              const start = Math.min(from, to);
              const stop = Math.max(from, to) + 1;
              u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
              if (stop - start >= 2) setRange({ start, stop });
            },
          ],
        },
        scales: { x: { time: data.x_is_date } },
        axes: [
          { stroke: axisColor, grid: { stroke: gridColor, width: 1 }, ticks: { stroke: gridColor } },
          { stroke: axisColor, grid: { stroke: gridColor, width: 1 }, ticks: { stroke: gridColor }, size: 58 },
        ],
        series: [
          { label: data.x_is_date ? "date" : "row" },
          ...data.series.map((s, i) => ({
            label: s.name,
            stroke: PALETTE[i % PALETTE.length],
            width: 1.25,
            spanGaps: false,
            points: { show: false },
          })),
        ],
      },
      [x, ...series] as uPlot.AlignedData,
      host,
    );

    // Wheel zoom about the cursor, which is what people reach for first.
    const over = plotRef.current.over;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const fraction = event.offsetX / Math.max(1, over.clientWidth);
      zoomAt(event.deltaY < 0 ? 0.72 : 1 / 0.72, fraction);
    };
    over.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      over.removeEventListener("wheel", onWheel);
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, [data, size, zoomAt]);

  const toggle = (name: string) =>
    setSelected((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : prev.length >= MAX_SERIES
          ? prev
          : [...prev, name],
    );

  const needle = filter.trim().toLowerCase();
  const matching = needle
    ? plottable.filter((c) => c.name.toLowerCase().includes(needle))
    : plottable;

  const colourOf = (name: string) =>
    PALETTE[selected.indexOf(name) % PALETTE.length];

  const atLimit = selected.length >= MAX_SERIES;

  // Prefer the dates at the window edges when there are any; row numbers mean
  // little on a time series.
  const windowLabel = (() => {
    if (!data || data.x.length === 0) return "";
    if (!range) return `all ${totalRows.toLocaleString()} rows`;
    if (data.x_is_date) return `${data.x[0]} → ${data.x[data.x.length - 1]}`;
    return `rows ${range.start.toLocaleString()}–${range.stop.toLocaleString()}`;
  })();

  return (
    <div className="plot-drawer">
      <div className="plot-header">
        <strong>Plot</strong>
        <span className="hint">
          {data?.decimated
            ? `min–max decimated, ${data.bucket_size} rows per point — spikes preserved`
            : "full resolution"}
        </span>
        {isFetching && <span className="spin" />}
        <div style={{ flex: 1 }} />

        <span className="control-group">
          <button
            className="ghost"
            onClick={() => panBy(-0.5)}
            disabled={!range || range.start === 0}
            title="Pan left"
          >
            ‹
          </button>
          <button
            className="ghost"
            onClick={() => zoomBy(0.5)}
            title="Zoom in (or drag across the plot, or scroll)"
          >
            +
          </button>
          <button
            className="ghost"
            onClick={() => zoomBy(2)}
            disabled={!range}
            title="Zoom out"
          >
            −
          </button>
          <button
            className="ghost"
            onClick={() => panBy(0.5)}
            disabled={!range || range.stop >= totalRows}
            title="Pan right"
          >
            ›
          </button>
          <button onClick={() => setRange(null)} disabled={!range} title="Show everything">
            Reset
          </button>
        </span>

        <span className="hint">{windowLabel}</span>

        <span className="hint">
          {selected.length} of {plottable.length} series
        </span>
        <button
          className={pickerOpen ? "toggled" : ""}
          onClick={() => setPickerOpen((open) => !open)}
          title="Choose which columns to plot"
        >
          Choose series
        </button>
        <button className="ghost" onClick={onClose} title="Close plot">
          ✕
        </button>
      </div>

      {/* Currently plotted series, always visible so you can drop one without
          opening the picker. */}
      <div className="series-picker">
        {selected.length === 0 ? (
          <span className="hint">No series selected.</span>
        ) : (
          selected.map((name) => (
            <button
              key={name}
              className="series-chip on"
              onClick={() => toggle(name)}
              style={{ borderColor: colourOf(name), color: colourOf(name) }}
              title={`Remove ${name}`}
            >
              {name} ✕
            </button>
          ))
        )}
      </div>

      {/* A searchable list rather than a wall of chips: these files routinely
          have a hundred or more columns, and scanning them by eye does not
          scale. Collapsed by default so the chart keeps the space. */}
      {pickerOpen && (
        <div className="series-list">
          <div className="series-list-bar">
            <input
              autoFocus
              placeholder={`Filter ${plottable.length} columns…`}
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            />
            <span className="hint">{matching.length} shown</span>
            <button
              onClick={() =>
                setSelected((prev) => {
                  const next = [...prev];
                  for (const c of matching) {
                    if (next.length >= MAX_SERIES) break;
                    if (!next.includes(c.name)) next.push(c.name);
                  }
                  return next;
                })
              }
              disabled={atLimit}
              title={`Add the filtered columns, up to ${MAX_SERIES}`}
            >
              Add shown
            </button>
            <button onClick={() => setSelected([])} disabled={!selected.length}>
              Clear
            </button>
          </div>

          <div className="series-options">
            {matching.map((c) => {
              const on = selected.includes(c.name);
              return (
                <label
                  key={c.name}
                  className={`series-option${on ? " on" : ""}`}
                  title={c.name}
                >
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={!on && atLimit}
                    onChange={() => toggle(c.name)}
                  />
                  <span
                    className="swatch"
                    style={{ background: on ? colourOf(c.name) : "transparent" }}
                  />
                  <span className="series-option-name">{c.name}</span>
                </label>
              );
            })}
            {matching.length === 0 && (
              <div className="hint" style={{ padding: "6px 4px" }}>
                Nothing matching “{filter}”.
              </div>
            )}
          </div>

          {atLimit && (
            <div className="hint" style={{ padding: "4px 2px 0" }}>
              {MAX_SERIES} series is the limit — remove one to add another.
            </div>
          )}
        </div>
      )}

      <div className="plot-body">
        {error && (
          <div style={{ color: "var(--danger)" }}>{(error as Error).message}</div>
        )}
        {indices.length === 0 && (
          <div className="hint">Pick one or more columns to plot.</div>
        )}
        <div className="plot-host" ref={hostRef} />
      </div>
    </div>
  );
}
