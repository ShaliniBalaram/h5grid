import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, initialFileId } from "./api/client";
import { formatBytes } from "./api/hooks";
import { ApiError, type NodeMeta, type TreeNode } from "./api/types";
import DataGrid from "./components/DataGrid";
import FileBrowser from "./components/FileBrowser";
import Inspector from "./components/Inspector";
import PlotDrawer from "./components/PlotDrawer";
import TreePanel from "./components/TreePanel";

interface OpenTab {
  fileId: string;
  path: string;
  name: string;
  sizeBytes: number;
}

function useDragWidth(initial: number, min: number, max: number, fromLeft: boolean) {
  const [width, setWidth] = useState(initial);
  const [dragging, setDragging] = useState(false);

  const onMouseDown = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      setDragging(true);
      const startX = event.clientX;
      const startWidth = width;

      const onMove = (move: MouseEvent) => {
        const delta = fromLeft ? move.clientX - startX : startX - move.clientX;
        setWidth(Math.max(min, Math.min(max, startWidth + delta)));
      };
      const onUp = () => {
        setDragging(false);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [width, min, max, fromLeft],
  );

  return { width, dragging, onMouseDown };
}

export default function App() {
  const queryClient = useQueryClient();

  const [tabs, setTabs] = useState<OpenTab[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const [browserOpen, setBrowserOpen] = useState(false);
  const [openBusy, setOpenBusy] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);

  const [treeFilter, setTreeFilter] = useState("");
  const [showRaw, setShowRaw] = useState(false);
  const [useTimeIndex, setUseTimeIndex] = useState(true);
  const [fullPrecision, setFullPrecision] = useState(false);
  const [plotOpen, setPlotOpen] = useState(false);

  const [dimIndices, setDimIndices] = useState<number[]>([]);
  const [scrollTarget, setScrollTarget] = useState<{ row: number; nonce: number } | null>(
    null,
  );

  const [searchCol, setSearchCol] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [matches, setMatches] = useState<number[] | null>(null);
  const [matchIndex, setMatchIndex] = useState(0);
  const [searchBusy, setSearchBusy] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const left = useDragWidth(290, 150, 560, true);
  const right = useDragWidth(298, 190, 620, false);

  const bootstrapped = useRef(false);

  const openPath = useCallback(
    async (path: string) => {
      setOpenBusy(true);
      setOpenError(null);
      try {
        const info = await api.openFile(path);
        setTabs((prev) => {
          const without = prev.filter((t) => t.path !== info.path);
          return [
            ...without,
            {
              fileId: info.file_id,
              path: info.path,
              name: info.name,
              sizeBytes: info.size_bytes,
            },
          ];
        });
        setActiveId(info.file_id);
        setSelectedPath(null);
        setBrowserOpen(false);
      } catch (err) {
        setOpenError(
          err instanceof ApiError ? err.message : String((err as Error).message),
        );
      } finally {
        setOpenBusy(false);
      }
    },
    [],
  );

  // The CLI pre-opens the file and passes its id on the URL.
  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    const fid = initialFileId();
    if (!fid) return;
    api
      .tree(fid)
      .then(() => {
        setTabs([{ fileId: fid, path: "", name: "opened file", sizeBytes: 0 }]);
        setActiveId(fid);
      })
      .catch(() => setBrowserOpen(true));
  }, []);

  const activeTab = tabs.find((t) => t.fileId === activeId) ?? null;

  const treeQuery = useQuery({
    queryKey: ["tree", activeId, showRaw],
    queryFn: () => api.tree(activeId!, showRaw),
    enabled: Boolean(activeId),
  });

  // Fill in a tab's real name once the tree confirms the file is readable.
  useEffect(() => {
    if (!activeTab || activeTab.path || !treeQuery.isSuccess) return;
    setTabs((prev) =>
      prev.map((t) => (t.fileId === activeTab.fileId ? { ...t, path: "(opened)" } : t)),
    );
  }, [treeQuery.isSuccess, activeTab]);

  const sliceSpec = useMemo(() => {
    if (dimIndices.length === 0) return undefined;
    return ["", "", ...dimIndices.map(String)].join(",");
  }, [dimIndices]);

  const metaQuery = useQuery({
    queryKey: ["meta", activeId, selectedPath, sliceSpec ?? ""],
    queryFn: () => api.meta(activeId!, selectedPath!, sliceSpec),
    enabled: Boolean(activeId && selectedPath),
  });

  const meta: NodeMeta | null = metaQuery.data ?? null;

  useEffect(() => {
    setMatches(null);
    setSearchError(null);
    setMatchIndex(0);
  }, [selectedPath, activeId]);

  useEffect(() => {
    if (!meta) return;
    const firstNumeric = meta.columns.find((c) => !c.is_datetime);
    setSearchCol(firstNumeric?.name ?? meta.columns[0]?.name ?? "");
  }, [meta?.path]);

  const onSelectNode = useCallback((node: TreeNode) => {
    setSelectedPath(node.path);
    // Extra dimensions beyond rows and columns default to index 0.
    setDimIndices(node.ndim && node.ndim > 2 ? Array(node.ndim - 2).fill(0) : []);
  }, []);

  const fileChanged =
    (treeQuery.error instanceof ApiError && treeQuery.error.isFileChanged) ||
    (metaQuery.error instanceof ApiError && metaQuery.error.isFileChanged);

  const reload = useCallback(async () => {
    if (!activeTab?.path || activeTab.path === "(opened)") {
      setBrowserOpen(true);
      return;
    }
    await queryClient.invalidateQueries();
    await openPath(activeTab.path);
  }, [activeTab, openPath, queryClient]);

  const closeTab = useCallback(
    (fileId: string) => {
      void api.closeFile(fileId).catch(() => undefined);
      setTabs((prev) => prev.filter((t) => t.fileId !== fileId));
      if (activeId === fileId) {
        const remaining = tabs.filter((t) => t.fileId !== fileId);
        setActiveId(remaining.at(-1)?.fileId ?? null);
        setSelectedPath(null);
      }
    },
    [activeId, tabs],
  );

  const runSearch = useCallback(async () => {
    if (!activeId || !selectedPath || !searchCol || !searchQuery.trim()) return;
    setSearchBusy(true);
    setSearchError(null);
    try {
      const result = await api.search(
        activeId,
        selectedPath,
        searchCol,
        searchQuery.trim(),
        sliceSpec,
      );
      setMatches(result.rows);
      setMatchIndex(0);
      if (result.rows.length > 0) {
        setScrollTarget({ row: result.rows[0], nonce: Date.now() });
      }
    } catch (err) {
      setSearchError((err as Error).message);
      setMatches(null);
    } finally {
      setSearchBusy(false);
    }
  }, [activeId, selectedPath, searchCol, searchQuery, sliceSpec]);

  const stepMatch = useCallback(
    (delta: number) => {
      if (!matches || matches.length === 0) return;
      const next = (matchIndex + delta + matches.length) % matches.length;
      setMatchIndex(next);
      setScrollTarget({ row: matches[next], nonce: Date.now() });
    },
    [matches, matchIndex],
  );

  const jumpToRow = useCallback(
    (value: string) => {
      const row = Number.parseInt(value, 10);
      if (!Number.isFinite(row) || !meta) return;
      const clamped = Math.max(0, Math.min(meta.nrows - 1, row - 1));
      setScrollTarget({ row: clamped, nonce: Date.now() });
    },
    [meta],
  );

  const [exportBusy, setExportBusy] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  const downloadExport = useCallback(
    async (format: "csv" | "xlsx") => {
      if (!activeId || !selectedPath) return;
      setExportBusy(format);
      setExportError(null);
      try {
        const { blob, filename } = await api.exportFile(activeId, selectedPath, {
          format,
          slice: sliceSpec,
          use_time_index: useTimeIndex && (meta?.time_index_available ?? false),
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
      } catch (err) {
        setExportError(
          err instanceof ApiError
            ? err.message
            : `Export failed: ${(err as Error).message}`,
        );
      } finally {
        setExportBusy(null);
      }
    },
    [activeId, selectedPath, sliceSpec, useTimeIndex, meta],
  );

  return (
    <div className="app">
      <div className="toolbar">
        <span className="brand">H5Grid</span>
        <button onClick={() => setBrowserOpen(true)}>Open file…</button>

        <div className="file-tabs">
          {tabs.map((tab) => (
            <div
              key={tab.fileId}
              className={`file-tab${tab.fileId === activeId ? " active" : ""}`}
              onClick={() => {
                setActiveId(tab.fileId);
                setSelectedPath(null);
              }}
              title={tab.path}
            >
              <span className="label">{tab.name}</span>
              {tab.sizeBytes > 0 && (
                <span className="hint">{formatBytes(tab.sizeBytes)}</span>
              )}
              <button
                className="close"
                onClick={(event) => {
                  event.stopPropagation();
                  closeTab(tab.fileId);
                }}
                title="Close file"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        <span className="spacer" />

        <button
          className={fullPrecision ? "toggled" : ""}
          onClick={() => setFullPrecision((prev) => !prev)}
          title="Show every digit instead of 6 significant figures"
        >
          Full precision
        </button>
        <button
          className={showRaw ? "toggled" : ""}
          onClick={() => setShowRaw((prev) => !prev)}
          title="Reveal the raw pandas/PyTables structure behind decoded tables"
        >
          Raw structure
        </button>
        <button
          className={plotOpen ? "toggled" : ""}
          disabled={!meta}
          onClick={() => setPlotOpen((prev) => !prev)}
        >
          Plot
        </button>
        {(["csv", "xlsx"] as const).map((format) => (
          <button
            key={format}
            onClick={() => void downloadExport(format)}
            disabled={
              !meta || Boolean(meta.decode_fallback) || exportBusy !== null
            }
            title={`Export this dataset as ${format.toUpperCase()}`}
          >
            {exportBusy === format ? (
              <>
                <span className="spin" /> {format.toUpperCase()}
              </>
            ) : (
              format.toUpperCase()
            )}
          </button>
        ))}
      </div>

      {exportError && (
        <div className="banner error">
          <span>{exportError}</span>
          <span className="spacer" />
          <button onClick={() => setExportError(null)}>Dismiss</button>
        </div>
      )}

      {fileChanged && (
        <div className="banner warn">
          <span>
            This file changed on disk — most likely the model run that writes it is
            still going. The view you are looking at is from before the change.
          </span>
          <span className="spacer" />
          <button onClick={reload}>Reload</button>
        </div>
      )}

      <div className="main">
        <div className="panel" style={{ width: left.width, flex: `0 0 ${left.width}px` }}>
          <div className="panel-header">
            <span>Tree</span>
            <input
              style={{ flex: 1, minWidth: 0, padding: "2px 6px" }}
              placeholder="filter…"
              value={treeFilter}
              onChange={(event) => setTreeFilter(event.target.value)}
            />
          </div>
          <div className="panel-body">
            {treeQuery.isLoading && (
              <div style={{ padding: 12 }}>
                <span className="spin" /> <span className="hint">reading tree…</span>
              </div>
            )}
            {treeQuery.error && !fileChanged && (
              <div className="banner error">{(treeQuery.error as Error).message}</div>
            )}
            <TreePanel
              root={treeQuery.data ?? null}
              selectedPath={selectedPath}
              onSelect={onSelectNode}
              filter={treeFilter}
            />
          </div>
        </div>

        <div
          className={`resizer${left.dragging ? " dragging" : ""}`}
          onMouseDown={left.onMouseDown}
        />

        <div className="center">
          {!activeId ? (
            <div className="empty-state">
              <h2>No file open</h2>
              <p>
                Open an .h5 file to browse its groups and datasets. Pandas tables and
                pywr recorder outputs are decoded into real columns and dates rather
                than raw storage blocks.
              </p>
              <button className="primary" onClick={() => setBrowserOpen(true)}>
                Open file…
              </button>
            </div>
          ) : !selectedPath ? (
            <div className="empty-state">
              <h2>Pick a dataset</h2>
              <p>Choose a dataset or table from the tree on the left.</p>
            </div>
          ) : (
            <>
              <div className="grid-bar">
                <span className="path">{selectedPath}</span>

                {meta && meta.ndim > 2 && (
                  <span className="dim-picker">
                    {dimIndices.map((value, i) => {
                      const size = meta.dim_sizes?.[i + 2] ?? 1;
                      // Named from the file's /scenarios table when it is a
                      // pywr run, so this reads "demand" rather than "dim 2".
                      const label = meta.dim_names?.[i + 2] ?? `dim ${i + 2}`;
                      return (
                        <label key={i}>
                          {label}:{" "}
                          <select
                            value={value}
                            onChange={(event) =>
                              setDimIndices((prev) =>
                                prev.map((v, j) =>
                                  j === i ? Number(event.target.value) : v,
                                ),
                              )
                            }
                          >
                            {Array.from({ length: size }, (_, k) => (
                              <option key={k} value={k}>
                                {k}
                              </option>
                            ))}
                          </select>
                        </label>
                      );
                    })}
                  </span>
                )}

                {meta?.time_index_available && (
                  <button
                    className={useTimeIndex ? "toggled" : ""}
                    onClick={() => setUseTimeIndex((prev) => !prev)}
                    title="Use the file's /time table as the row index"
                  >
                    /time index
                  </button>
                )}

                <span style={{ flex: 1 }} />

                <label className="control-group" title="Jump to a row number (1-based)">
                  <span>Row</span>
                  <input
                    style={{ width: 74 }}
                    placeholder="1"
                    onKeyDown={(event) => {
                      if (event.key === "Enter")
                        jumpToRow((event.target as HTMLInputElement).value);
                    }}
                  />
                </label>

                {meta && meta.columns.length > 0 && (
                  // Laid out to read as one sentence — "find <query> in <column>"
                  // — because the column picker sitting loose next to the row
                  // box looked like it belonged to the row box instead.
                  <span className="control-group">
                    <span>Find</span>
                    <input
                      style={{ width: 118 }}
                      placeholder="&gt;100"
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void runSearch();
                      }}
                      title="Value search: >100, <=0, =5, 3..7, or plain text"
                    />
                    <span>in</span>
                    <select
                      value={searchCol}
                      onChange={(event) => setSearchCol(event.target.value)}
                      style={{ maxWidth: 150 }}
                      title="Which column to search. Takes effect when you press Find."
                    >
                      {meta.columns.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                    <button onClick={() => void runSearch()} disabled={searchBusy}>
                      {searchBusy ? "…" : "Find"}
                    </button>
                    {matches && (
                      <>
                        <span className="hint">
                          {matches.length === 0
                            ? "no matches"
                            : `${matchIndex + 1} / ${matches.length}`}
                        </span>
                        <button
                          className="ghost"
                          onClick={() => stepMatch(-1)}
                          disabled={matches.length === 0}
                          title="Previous match"
                        >
                          ↑
                        </button>
                        <button
                          className="ghost"
                          onClick={() => stepMatch(1)}
                          disabled={matches.length === 0}
                          title="Next match"
                        >
                          ↓
                        </button>
                      </>
                    )}
                  </span>
                )}
              </div>

              {searchError && <div className="banner error">{searchError}</div>}

              {meta?.decode_fallback && (
                <div className="banner warn">{meta.decode_fallback}</div>
              )}

              {metaQuery.isLoading && (
                <div className="empty-state">
                  <span className="spin" />
                </div>
              )}

              {metaQuery.error && !fileChanged && (
                <div className="banner error">{(metaQuery.error as Error).message}</div>
              )}

              {meta && !meta.decode_fallback && activeId && (
                <DataGrid
                  key={`${activeId}|${meta.path}|${sliceSpec ?? ""}`}
                  fileId={activeId}
                  meta={meta}
                  slice={sliceSpec}
                  useTimeIndex={useTimeIndex}
                  fullPrecision={fullPrecision}
                  scrollTarget={scrollTarget}
                />
              )}

              {plotOpen && meta && activeId && !meta.decode_fallback && (
                <PlotDrawer
                  fileId={activeId}
                  meta={meta}
                  slice={sliceSpec}
                  useTimeIndex={useTimeIndex && meta.time_index_available}
                  onClose={() => setPlotOpen(false)}
                />
              )}
            </>
          )}
        </div>

        <div
          className={`resizer${right.dragging ? " dragging" : ""}`}
          onMouseDown={right.onMouseDown}
        />

        <div
          className="panel"
          style={{ width: right.width, flex: `0 0 ${right.width}px` }}
        >
          <div className="panel-header">Inspector</div>
          <div className="panel-body">
            {activeId && (
              <Inspector
                fileId={activeId}
                meta={meta}
                slice={sliceSpec}
                loading={metaQuery.isLoading}
              />
            )}
          </div>
        </div>
      </div>

      {browserOpen && (
        <FileBrowser
          onPick={openPath}
          onClose={() => setBrowserOpen(false)}
          busy={openBusy}
          error={openError}
        />
      )}
    </div>
  );
}
