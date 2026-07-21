import { useQuery } from "@tanstack/react-query";
import { Fragment, useEffect, useState } from "react";

import { api } from "../api/client";
import { formatNumber } from "../api/hooks";
import type { NodeMeta } from "../api/types";

interface Props {
  fileId: string;
  meta: NodeMeta | null;
  slice?: string;
  loading: boolean;
}

function AttrValue({ value }: { value: unknown }) {
  const [expanded, setExpanded] = useState(false);
  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, expanded ? 1 : 0);
  const long = text.length > 70;

  return (
    <span
      className={`attr-value${expanded ? " expanded" : long ? " truncated" : ""}`}
      onClick={() => long && setExpanded((prev) => !prev)}
      title={long && !expanded ? "Click to expand" : undefined}
    >
      {expanded || !long ? text : text.slice(0, 70)}
    </span>
  );
}

function StatsSection({
  fileId,
  meta,
  slice,
}: {
  fileId: string;
  meta: NodeMeta;
  slice?: string;
}) {
  const [column, setColumn] = useState<string>("");

  useEffect(() => {
    const firstNumeric = meta.columns.find((c) => !c.is_datetime);
    setColumn(firstNumeric?.name ?? meta.columns[0]?.name ?? "");
  }, [meta.path, meta.columns]);

  const [requested, setRequested] = useState<string | null>(null);

  const { data, isFetching, error } = useQuery({
    queryKey: ["stats", fileId, meta.path, slice ?? "", requested],
    queryFn: () => api.stats(fileId, meta.path, requested!, slice),
    enabled: Boolean(requested),
    staleTime: Infinity,
  });

  // A fresh node invalidates whatever was computed for the previous one.
  useEffect(() => setRequested(null), [meta.path, slice]);

  if (meta.columns.length === 0) return null;

  const rows: [string, string][] = [];
  if (data) {
    const fmt = (v: number | string | null) =>
      v === null ? "—" : typeof v === "number" ? formatNumber(v, false) : String(v);
    if (data.numeric) {
      rows.push(["Min", fmt(data.min)], ["Max", fmt(data.max)]);
      rows.push(["Mean", fmt(data.mean)], ["Std dev", fmt(data.std)]);
    } else {
      rows.push(["Distinct", String(data.distinct_count ?? "—")]);
    }
    rows.push(["Count", data.count.toLocaleString()]);
    rows.push(["Missing", data.nan_count.toLocaleString()]);
  }

  return (
    <div className="inspector-section">
      <h4>Column statistics</h4>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <select
          value={column}
          onChange={(e) => setColumn(e.target.value)}
          style={{ flex: 1, minWidth: 0 }}
        >
          {meta.columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
        <button onClick={() => setRequested(column)} disabled={!column || isFetching}>
          {isFetching ? "…" : "Compute"}
        </button>
      </div>

      {error && <div style={{ color: "var(--danger)" }}>{(error as Error).message}</div>}

      {data && (
        <div className="stat-grid">
          {rows.map(([label, value]) => (
            <Fragment key={label}>
              <span className="label">{label}</span>
              <span className="value">{value}</span>
            </Fragment>
          ))}
        </div>
      )}
      {!data && !isFetching && (
        <div className="hint">
          Scans the whole column in chunks — instant on small nodes, a moment on
          very large ones.
        </div>
      )}
    </div>
  );
}

export default function Inspector({ fileId, meta, slice, loading }: Props) {
  if (loading) {
    return (
      <div className="inspector-section">
        <span className="spin" /> <span className="hint">loading…</span>
      </div>
    );
  }

  if (!meta) {
    return (
      <div className="inspector-section">
        <div className="hint">Select a dataset to see its details.</div>
      </div>
    );
  }

  const attrs = Object.entries(meta.attrs);
  const indexColumnCount =
    meta.kind === "pandas_frame" || meta.kind === "pandas_table" ? 1 : 0;
  const dataColumnCount = meta.columns.length - indexColumnCount;

  return (
    <>
      <div className="inspector-section">
        <h4>Node</h4>
        <dl className="kv">
          <dt>Path</dt>
          <dd>{meta.path}</dd>
          <dt>Kind</dt>
          <dd>{meta.kind.replace("_", " ")}</dd>
          <dt>Shape</dt>
          <dd>{meta.shape ? meta.shape.join(" × ") : "—"}</dd>
          <dt>Dtype</dt>
          <dd>{meta.dtype ?? "—"}</dd>
          <dt>Chunks</dt>
          <dd>{meta.chunks ? meta.chunks.join(" × ") : "contiguous"}</dd>
          <dt>Compression</dt>
          <dd>{meta.compression ?? "none"}</dd>
          <dt>Rows</dt>
          <dd>{meta.nrows.toLocaleString()}</dd>
        </dl>
      </div>

      {(meta.pywr || meta.scenarios) && (
        <div className="inspector-section">
          <h4>pywr</h4>
          <dl className="kv">
            {meta.pywr?.type && (
              <>
                <dt>Node type</dt>
                <dd>{meta.pywr.type}</dd>
              </>
            )}
            {meta.pywr?.attribute && (
              <>
                <dt>Records</dt>
                <dd>{meta.pywr.attribute}</dd>
              </>
            )}
            {meta.scenarios?.scenarios.map((s) => (
              <Fragment key={s.name}>
                <dt>Scenario</dt>
                <dd>
                  {s.name} ({s.size})
                </dd>
              </Fragment>
            ))}
          </dl>
          {meta.scenarios?.collapsed && (
            <div className="hint" style={{ marginTop: 6 }}>
              This run used {meta.scenarios.combination_count} explicit scenario
              combinations, so the scenario axes are collapsed into one.
            </div>
          )}
        </div>
      )}

      <div className="inspector-section">
        {/* Count data columns only, matching the tree badge and the status bar.
            A frame's index is listed below but is not one of its columns. */}
        <h4>
          Columns ({dataColumnCount}
          {indexColumnCount > 0 ? " + index" : ""})
        </h4>
        <dl className="kv col-list">
          {meta.columns.slice(0, 60).map((c) => (
            <Fragment key={c.name}>
              <dt title={c.name}>{c.name}</dt>
              <dd>{c.is_datetime ? "datetime" : c.dtype}</dd>
            </Fragment>
          ))}
        </dl>
        {meta.columns.length > 60 && (
          <div className="hint" style={{ marginTop: 6 }}>
            + {meta.columns.length - 60} more
          </div>
        )}
      </div>

      <StatsSection fileId={fileId} meta={meta} slice={slice} />

      <div className="inspector-section">
        <h4>Attributes ({attrs.length})</h4>
        {attrs.length === 0 ? (
          <div className="hint">No attributes on this node.</div>
        ) : (
          <dl className="kv">
            {attrs.map(([key, value]) => (
              <Fragment key={key}>
                <dt>{key}</dt>
                <dd>
                  <AttrValue value={value} />
                </dd>
              </Fragment>
            ))}
          </dl>
        )}
      </div>
    </>
  );
}
