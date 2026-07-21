import {
  CompactSelection,
  DataEditor,
  GridCellKind,
  type DataEditorRef,
  type GridCell,
  type GridColumn,
  type GridSelection,
  type Item,
  type Theme,
} from "@glideapps/glide-data-grid";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { formatNumber, usePagedRows } from "../api/hooks";
import type { CellValue, NodeMeta } from "../api/types";

interface Props {
  fileId: string;
  meta: NodeMeta;
  slice?: string;
  useTimeIndex: boolean;
  fullPrecision: boolean;
  scrollTarget: { row: number; nonce: number } | null;
}

/** Reads the app's CSS variables so the canvas grid matches the DOM around it. */
function useGlideTheme(): Partial<Theme> {
  const [theme, setTheme] = useState<Partial<Theme>>({});

  useEffect(() => {
    const build = () => {
      const css = getComputedStyle(document.documentElement);
      const v = (name: string) => css.getPropertyValue(name).trim();
      setTheme({
        accentColor: v("--accent"),
        accentLight: v("--accent-soft"),
        textDark: v("--text"),
        textMedium: v("--text-dim"),
        textLight: v("--text-faint"),
        textBubble: v("--text"),
        bgCell: v("--bg"),
        bgCellMedium: v("--bg-panel"),
        bgHeader: v("--bg-panel"),
        bgHeaderHasFocus: v("--bg-active"),
        bgHeaderHovered: v("--bg-hover"),
        borderColor: v("--border"),
        horizontalBorderColor: v("--border"),
        drilldownBorder: v("--border"),
        textHeader: v("--text-dim"),
        fontFamily: v("--font-ui"),
        baseFontStyle: "12px",
        headerFontStyle: "600 12px",
        cellHorizontalPadding: 8,
        cellVerticalPadding: 3,
      });
    };

    build();
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", build);
    return () => media.removeEventListener("change", build);
  }, []);

  return theme;
}

// Pinned rather than left to the library default so the scroll arithmetic
// below is exact instead of inferred.
const ROW_HEIGHT = 34;
const HEADER_HEIGHT = 36;

export default function DataGrid({
  fileId,
  meta,
  slice,
  useTimeIndex,
  fullPrecision,
  scrollTarget,
}: Props) {
  const gridRef = useRef<DataEditorRef>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const theme = useGlideTheme();

  const showTimeColumn = useTimeIndex && meta.time_index_available;

  const columnInfos = useMemo(
    () =>
      showTimeColumn
        ? [
            { name: "date", dtype: "datetime64[ns]", is_datetime: true },
            ...meta.columns,
          ]
        : meta.columns,
    [meta.columns, showTimeColumn],
  );

  // Columns that identify a row rather than carry data: the frame's own index,
  // and the date column injected from a /time table. Kept out of the column
  // count so the grid agrees with the table's shape and with the tree badge.
  const isPandasNode =
    meta.kind === "pandas_frame" || meta.kind === "pandas_table";
  const indexColumnCount = (isPandasNode ? 1 : 0) + (showTimeColumn ? 1 : 0);

  const { getRow, ensureRange, revision, loadingPages, error } = usePagedRows({
    fileId,
    path: meta.path,
    totalRows: meta.nrows,
    columnCount: columnInfos.length,
    slice,
    useTimeIndex: showTimeColumn,
  });

  const [widths, setWidths] = useState<Record<string, number>>({});
  const [selection, setSelection] = useState<GridSelection>({
    columns: CompactSelection.empty(),
    rows: CompactSelection.empty(),
  });

  // A different node means different columns; drop stale widths and selection.
  useEffect(() => {
    setWidths({});
    setSelection({
      columns: CompactSelection.empty(),
      rows: CompactSelection.empty(),
    });
  }, [meta.path, slice, showTimeColumn]);

  const columns = useMemo<GridColumn[]>(
    () =>
      columnInfos.map((info) => ({
        id: info.name,
        title: info.name,
        width: widths[info.name] ?? (info.is_datetime ? 108 : 116),
        // The dtype sits under the name in the header menu / tooltip.
        overlayIcon: undefined,
        themeOverride: undefined,
        group: undefined,
        hasMenu: false,
      })),
    [columnInfos, widths],
  );

  useEffect(() => {
    ensureRange(0, 200);
  }, [ensureRange]);

  // Glide reports the visible rows on every scroll; tracking them tells the
  // jump logic below whether the target row has actually been reached.
  const visible = useRef({ y: 0, height: 0 });
  const pendingScroll = useRef<{ row: number; attempts: number } | null>(null);

  /**
   * Move the grid so `pendingScroll` is on screen, and keep re-asserting it
   * until the grid confirms it got there.
   *
   * Two things make a single call unreliable. The grid's own `scrollTo` moves
   * by a delta worked out from a visible region it only refreshes once it has
   * drawn, so it does nothing from a cold cache. And a data page arriving
   * mid-jump re-renders the grid, which restores the scroll offset it had
   * before. Re-applying after each render survives both; the attempt cap stops
   * it fighting the user if the target can never be reached.
   */
  const applyPendingScroll = useCallback(() => {
    const pending = pendingScroll.current;
    if (!pending) return;

    const { y, height } = visible.current;
    if (height > 0 && pending.row >= y && pending.row < y + height) {
      pendingScroll.current = null;
      return;
    }
    if (pending.attempts >= 6) {
      pendingScroll.current = null;
      return;
    }
    pending.attempts += 1;

    const scroller = hostRef.current?.querySelector<HTMLElement>(".dvn-scroller");
    if (!scroller) {
      gridRef.current?.scrollTo(0, pending.row, "vertical", 0, 0, {
        vAlign: "center",
      });
      return;
    }
    const centred =
      pending.row * ROW_HEIGHT - (scroller.clientHeight - ROW_HEIGHT) / 2;
    const maxTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    scroller.scrollTop = Math.max(0, Math.min(maxTop, centred));
  }, []);

  // Runs after every render, which is what lets a scroll undone by an arriving
  // page be re-applied.
  useEffect(applyPendingScroll);

  useEffect(() => {
    if (!scrollTarget) return;
    const { row } = scrollTarget;
    ensureRange(row, row + 1);
    // Row highlight only, deliberately without a `current` cell: setting one
    // makes the grid scroll to reveal that cell on the next render, which
    // fights the explicit scroll and lands somewhere in between.
    setSelection({
      columns: CompactSelection.empty(),
      rows: CompactSelection.fromSingleSelection(row),
    });
    pendingScroll.current = { row, attempts: 0 };
    applyPendingScroll();
  }, [scrollTarget, ensureRange, applyPendingScroll]);

  const renderCell = useCallback(
    (value: CellValue | undefined, isDatetime: boolean): GridCell => {
      if (value === undefined) {
        return { kind: GridCellKind.Loading, allowOverlay: false };
      }

      if (value === null) {
        // NaN and NaT: an empty cell tinted red, so gaps are visible at a glance
        // rather than reading as a zero.
        return {
          kind: GridCellKind.Text,
          data: "",
          displayData: "",
          allowOverlay: false,
          themeOverride: { bgCell: "rgba(200, 60, 40, 0.10)" },
        };
      }

      if (typeof value === "number") {
        return {
          kind: GridCellKind.Number,
          data: value,
          displayData: formatNumber(value, fullPrecision),
          allowOverlay: false,
          contentAlign: "right",
        };
      }

      if (typeof value === "boolean") {
        return {
          kind: GridCellKind.Text,
          data: String(value),
          displayData: value ? "true" : "false",
          allowOverlay: false,
        };
      }

      return {
        kind: GridCellKind.Text,
        data: value,
        displayData: value,
        allowOverlay: value.length > 24,
        contentAlign: isDatetime ? "left" : undefined,
      };
    },
    [fullPrecision],
  );

  const getCellContent = useCallback(
    ([col, row]: Item): GridCell => {
      const values = getRow(row);
      const info = columnInfos[col];
      return renderCell(values?.[col], info?.is_datetime ?? false);
    },
    [getRow, columnInfos, renderCell],
  );

  // Excel-style aggregate of the current selection.
  const summary = useMemo(() => {
    const range = selection.current?.range;
    if (!range || (range.width <= 1 && range.height <= 1)) return null;

    let count = 0;
    let numeric = 0;
    let sum = 0;
    let pending = false;

    for (let row = range.y; row < range.y + range.height; row++) {
      const values = getRow(row);
      if (!values) {
        pending = true;
        continue;
      }
      for (let col = range.x; col < range.x + range.width; col++) {
        const value = values[col];
        if (value === null || value === undefined) continue;
        count += 1;
        if (typeof value === "number" && Number.isFinite(value)) {
          numeric += 1;
          sum += value;
        }
      }
    }
    return { count, numeric, sum, pending, cells: range.width * range.height };
    // `revision` participates so the summary refreshes as pages arrive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection, getRow, revision]);

  const onColumnResize = useCallback(
    (column: GridColumn, newSize: number) => {
      setWidths((prev) => ({ ...prev, [String(column.id)]: newSize }));
    },
    [],
  );

  const freezeCount = Math.min(indexColumnCount, 1);

  return (
    <>
      {error && (
        <div className="banner error">
          <span>Could not load rows: {error.message}</span>
        </div>
      )}
      <div className="grid-host" ref={hostRef}>
        <DataEditor
          ref={gridRef}
          theme={theme}
          columns={columns}
          rows={meta.nrows}
          rowHeight={ROW_HEIGHT}
          headerHeight={HEADER_HEIGHT}
          getCellContent={getCellContent}
          getCellsForSelection={true}
          onVisibleRegionChanged={(range) => {
            visible.current = { y: range.y, height: range.height };
            ensureRange(range.y, range.y + range.height);
          }}
          onColumnResize={onColumnResize}
          gridSelection={selection}
          onGridSelectionChange={setSelection}
          rowMarkers="number"
          rowMarkerWidth={Math.max(52, String(meta.nrows).length * 9 + 26)}
          freezeColumns={freezeCount}
          smoothScrollX
          smoothScrollY
          rangeSelect="multi-rect"
          columnSelect="multi"
          rowSelect="multi"
          keybindings={{ copy: true, selectAll: true, search: false }}
          width="100%"
          height="100%"
          overscrollX={0}
          overscrollY={0}
        />
        {loadingPages > 0 && meta.nrows > 0 && !getRow(0) && (
          <div className="skeleton-rows">
            <span className="spin" style={{ marginRight: 8 }} /> loading rows…
          </div>
        )}
      </div>

      <div className="status-bar">
        {/* Count data columns only, so this matches the table's own shape and
            the badge in the tree. The date/index column is shown but is not
            part of the data, and counting it made a 169-column frame read as
            170 in one place and 169 in the other. */}
        <span title={`${indexColumnCount} index column shown alongside`}>
          <b>{meta.nrows.toLocaleString()}</b> rows ×{" "}
          <b>{(columnInfos.length - indexColumnCount).toLocaleString()}</b> columns
          {indexColumnCount > 0 && <span className="hint"> + index</span>}
        </span>
        {summary ? (
          <>
            <span>
              Selected <b>{summary.cells.toLocaleString()}</b>
            </span>
            <span>
              Count <b>{summary.count.toLocaleString()}</b>
            </span>
            {summary.numeric > 0 && (
              <>
                <span>
                  Sum <b>{formatNumber(summary.sum, false)}</b>
                </span>
                <span>
                  Average <b>{formatNumber(summary.sum / summary.numeric, false)}</b>
                </span>
              </>
            )}
            {summary.pending && <span className="hint">(still loading rows)</span>}
          </>
        ) : (
          <span className="hint">
            Drag to select a range, then ⌘C / Ctrl+C to copy for Excel
          </span>
        )}
        <span className="sep" />
        {loadingPages > 0 && (
          <span>
            <span className="spin" /> loading
          </span>
        )}
      </div>
    </>
  );
}
