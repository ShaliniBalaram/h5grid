import { useQuery } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { api } from "../api/client";
import { formatBytes } from "../api/hooks";

interface Props {
  onPick: (path: string) => void;
  onClose: () => void;
  busy: boolean;
  error: string | null;
}

const ROOT_ICONS: Record<string, string> = {
  home: "🏠",
  volume: "💾",
  root: "🖥️",
  cwd: "📌",
  folder: "📁",
};

/**
 * A browser cannot give a web page a filesystem path — drag-and-drop yields
 * file contents, not a location the backend can open — so the picker is served
 * by the backend's /api/browse endpoint instead.
 *
 * Model data usually sits on an external or network drive, so the shortcuts
 * list mounted volumes: walking up from the home directory and scrolling the
 * filesystem root to find /Volumes is not a reasonable way to get there.
 */
export default function FileBrowser({ onPick, onClose, busy, error }: Props) {
  const [dir, setDir] = useState<string | undefined>(undefined);
  const [typed, setTyped] = useState("");
  // Where we have been, so Back returns to the previous folder rather than
  // simply stepping up a level — those are different things once you have
  // jumped to a shortcut or a breadcrumb.
  const history = useRef<string[]>([]);

  const { data, isFetching, error: browseError } = useQuery({
    queryKey: ["browse", dir ?? "~"],
    queryFn: () => api.browse(dir),
  });

  const roots = useQuery({
    queryKey: ["browse-roots"],
    queryFn: () => api.browseRoots(),
    staleTime: 60_000,
  });

  const goTo = useCallback(
    (next: string) => {
      if (data?.dir) history.current.push(data.dir);
      setDir(next);
    },
    [data?.dir],
  );

  const goBack = useCallback(() => {
    const previous = history.current.pop();
    if (previous !== undefined) {
      setDir(previous);
    }
  }, []);

  /** The bottom box takes a folder or a file; folders navigate, files open. */
  const submitTyped = useCallback(() => {
    const value = typed.trim();
    if (!value) return;
    if (/\.(h5|hdf5|hdf|he5|nc)$/i.test(value)) {
      onPick(value);
      return;
    }
    api
      .browse(value)
      .then((result) => {
        if (data?.dir) history.current.push(data.dir);
        setDir(result.dir);
        setTyped("");
      })
      .catch(() => onPick(value));
  }, [typed, onPick, data?.dir]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <header>
          <span>Open an HDF5 file</span>
          {(isFetching || busy) && <span className="spin" />}
          <div style={{ flex: 1 }} />
          <button className="ghost" onClick={onClose}>
            ✕
          </button>
        </header>

        <div className="browse-nav">
          <button
            onClick={goBack}
            disabled={history.current.length === 0}
            title="Back to where you were"
          >
            ‹ Back
          </button>
          <button
            onClick={() => data?.parent && goTo(data.parent)}
            disabled={!data?.parent}
            title="Up one folder"
          >
            ↑ Up
          </button>

          <nav className="breadcrumbs">
            {(data?.breadcrumbs ?? []).map((crumb, i, all) => (
              <span key={crumb.path}>
                <button
                  className="crumb"
                  onClick={() => goTo(crumb.path)}
                  disabled={i === all.length - 1}
                >
                  {crumb.name}
                </button>
                {i < all.length - 1 && <span className="crumb-sep">›</span>}
              </span>
            ))}
          </nav>
        </div>

        <div className="browse-split">
          <aside className="browse-roots">
            {roots.data?.roots.map((root) => (
              <button
                key={root.path}
                className={`root-item${data?.dir === root.path ? " on" : ""}`}
                onClick={() => goTo(root.path)}
                title={root.path}
              >
                <span>{ROOT_ICONS[root.kind] ?? "📁"}</span>
                <span className="root-name">{root.name}</span>
              </button>
            ))}
          </aside>

          <div className="body">
            {browseError && (
              <div className="banner error">{(browseError as Error).message}</div>
            )}

            {data?.entries.map((entry) => {
              const openable = entry.is_dir || entry.is_h5;
              return (
                <div
                  key={entry.path}
                  className={`browse-row${openable ? "" : " dim"}`}
                  onClick={() => {
                    if (entry.is_dir) goTo(entry.path);
                    else if (entry.is_h5) onPick(entry.path);
                  }}
                  title={entry.path}
                >
                  <span>{entry.is_dir ? "📁" : entry.is_h5 ? "🧊" : "📄"}</span>
                  <span className="browse-name">{entry.name}</span>
                  {entry.size_bytes !== null && (
                    <span className="size">{formatBytes(entry.size_bytes)}</span>
                  )}
                </div>
              );
            })}

            {data && data.entries.length === 0 && (
              <div style={{ padding: "18px 14px", color: "var(--text-faint)" }}>
                Nothing here.
              </div>
            )}
          </div>
        </div>

        <footer>
          <input
            style={{ flex: 1 }}
            placeholder="…or paste a full path to a file or folder"
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitTyped();
            }}
          />
          <button className="primary" disabled={!typed.trim() || busy} onClick={submitTyped}>
            {busy ? "Opening…" : "Go"}
          </button>
        </footer>

        {error && (
          <div className="banner error" style={{ borderTop: "1px solid var(--border)" }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
