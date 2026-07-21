import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "./client";
import type { CellValue, DataPage } from "./types";

/** The server refuses requests above this many cells, so pages are sized to fit. */
const MAX_CELLS = 200_000;
const MAX_PAGE_ROWS = 1000;
const MIN_PAGE_ROWS = 25;

export function pageRowsFor(columnCount: number): number {
  if (columnCount <= 0) return MAX_PAGE_ROWS;
  const fitted = Math.floor(MAX_CELLS / columnCount);
  return Math.max(MIN_PAGE_ROWS, Math.min(MAX_PAGE_ROWS, fitted));
}

export interface PagedRowsOptions {
  fileId: string | null;
  path: string | null;
  totalRows: number;
  columnCount: number;
  slice?: string;
  useTimeIndex: boolean;
}

export interface PagedRows {
  /** Row values, or undefined while the page is still in flight. */
  getRow: (row: number) => CellValue[] | undefined;
  ensureRange: (start: number, stop: number) => void;
  revision: number;
  loadingPages: number;
  error: Error | null;
}

/**
 * Backs the grid's virtual scrolling: pages are fetched on demand, cached by
 * TanStack Query, and kept in a ref so a redraw does not rebuild the map. One
 * page ahead of the viewport is prefetched so scrolling stays smooth.
 */
export function usePagedRows(options: PagedRowsOptions): PagedRows {
  const { fileId, path, totalRows, columnCount, slice, useTimeIndex } = options;
  const queryClient = useQueryClient();

  const pageRows = useMemo(() => pageRowsFor(columnCount), [columnCount]);
  const pages = useRef(new Map<number, CellValue[][]>());
  const inFlight = useRef(new Set<number>());
  const [revision, setRevision] = useState(0);
  const [loadingPages, setLoadingPages] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const key = `${fileId}|${path}|${slice ?? ""}|${useTimeIndex}|${pageRows}`;
  const activeKey = useRef(key);

  useEffect(() => {
    // Any change of file, node, slice or time-index toggle invalidates every
    // cached page: the same row number now means something different.
    activeKey.current = key;
    pages.current.clear();
    inFlight.current.clear();
    setLoadingPages(0);
    setError(null);
    setRevision((r) => r + 1);
  }, [key]);

  const fetchPage = useCallback(
    async (page: number) => {
      if (!fileId || !path) return;
      if (pages.current.has(page) || inFlight.current.has(page)) return;

      const requestKey = activeKey.current;
      inFlight.current.add(page);
      setLoadingPages(inFlight.current.size);

      try {
        const start = page * pageRows;
        if (start >= totalRows) return;
        const payload = await queryClient.fetchQuery({
          queryKey: ["data", requestKey, page],
          queryFn: () =>
            api.data(fileId, path, {
              start,
              stop: Math.min(start + pageRows, totalRows),
              slice,
              use_time_index: useTimeIndex,
            }),
          staleTime: 5 * 60 * 1000,
        });

        // A slow page for a node the user has already navigated away from must
        // not land in the current map.
        if (activeKey.current !== requestKey) return;
        pages.current.set(page, (payload as DataPage).rows);
        setRevision((r) => r + 1);
        setError(null);
      } catch (err) {
        if (activeKey.current === requestKey) setError(err as Error);
      } finally {
        inFlight.current.delete(page);
        setLoadingPages(inFlight.current.size);
      }
    },
    [fileId, path, pageRows, queryClient, slice, totalRows, useTimeIndex],
  );

  const ensureRange = useCallback(
    (start: number, stop: number) => {
      if (!fileId || !path || totalRows === 0) return;
      const firstPage = Math.max(0, Math.floor(start / pageRows));
      const lastPage = Math.floor(Math.max(start, stop - 1) / pageRows);
      const maxPage = Math.floor(Math.max(0, totalRows - 1) / pageRows);
      for (let page = firstPage; page <= Math.min(lastPage + 1, maxPage); page++) {
        void fetchPage(page);
      }
    },
    [fetchPage, fileId, path, pageRows, totalRows],
  );

  const getRow = useCallback(
    (row: number) => {
      const page = pages.current.get(Math.floor(row / pageRows));
      return page?.[row % pageRows];
    },
    [pageRows],
  );

  return { getRow, ensureRange, revision, loadingPages, error };
}

/** Formats a float for display: 6 significant digits unless full precision is on. */
export function formatNumber(value: number, fullPrecision: boolean): string {
  if (!Number.isFinite(value)) return String(value);
  if (fullPrecision) return String(value);
  if (Number.isInteger(value) && Math.abs(value) < 1e15) return String(value);
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e10)) return value.toExponential(5);
  return String(Number(value.toPrecision(6)));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}
