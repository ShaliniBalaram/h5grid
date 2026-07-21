import {
  ApiError,
  type BrowseResult,
  type BrowseRoot,
  type ColumnStats,
  type DataPage,
  type NodeMeta,
  type OpenFileInfo,
  type PlotPayload,
  type SearchResult,
  type TreeNode,
} from "./types";

const TOKEN_STORAGE_KEY = "h5grid.token";

/**
 * The session token arrives as a query parameter on the URL the CLI opens. It
 * is stripped from the address bar so it does not end up in browser history or
 * get copied around in a shared link, and kept in sessionStorage so reloading
 * the page does not lose the session. sessionStorage is per-tab and cleared
 * when the tab closes, which matches how long the token is good for anyway.
 */
function captureToken(): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get("token");

  if (fromUrl) {
    try {
      window.sessionStorage.setItem(TOKEN_STORAGE_KEY, fromUrl);
    } catch {
      /* private browsing can refuse storage; the in-memory copy still works */
    }
    params.delete("token");
    const rest = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (rest ? `?${rest}` : ""),
    );
    return fromUrl;
  }

  try {
    return window.sessionStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

const TOKEN = captureToken();

export function initialFileId(): string | null {
  return new URLSearchParams(window.location.search).get("file");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(TOKEN ? { "X-H5Grid-Token": TOKEN } : {}),
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
    },
  });

  if (!response.ok) {
    let code = "error";
    let detail = response.statusText;
    try {
      const body = await response.json();
      code = body.error ?? code;
      detail = body.detail ?? detail;
    } catch {
      /* a non-JSON error body; the status text will have to do */
    }
    throw new ApiError(response.status, code, detail);
  }

  return (await response.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | undefined>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  return search.toString();
};

export const api = {
  openFile: (path: string) =>
    request<OpenFileInfo>("/api/files/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  closeFile: (fileId: string) =>
    request<{ closed: boolean }>(`/api/files/${fileId}/close`, {
      method: "POST",
    }),

  tree: (fileId: string, raw = false) =>
    request<TreeNode>(`/api/files/${fileId}/tree?${qs({ raw })}`),

  meta: (fileId: string, path: string, slice?: string) =>
    request<NodeMeta>(
      `/api/files/${fileId}/node/meta?${qs({ path, slice })}`,
    ),

  data: (
    fileId: string,
    path: string,
    opts: {
      start: number;
      stop: number;
      cols?: string;
      slice?: string;
      use_time_index?: boolean;
    },
  ) =>
    request<DataPage>(
      `/api/files/${fileId}/node/data?${qs({ path, ...opts })}`,
    ),

  stats: (fileId: string, path: string, col: string, slice?: string) =>
    request<ColumnStats>(
      `/api/files/${fileId}/node/stats?${qs({ path, col, slice })}`,
    ),

  search: (
    fileId: string,
    path: string,
    col: string,
    q: string,
    slice?: string,
  ) =>
    request<SearchResult>(
      `/api/files/${fileId}/node/search?${qs({ path, col, q, slice, limit: 1000 })}`,
    ),

  plotData: (
    fileId: string,
    path: string,
    opts: {
      cols: string;
      max_points?: number;
      slice?: string;
      use_time_index?: boolean;
      start?: number;
      stop?: number;
    },
  ) =>
    request<PlotPayload>(
      `/api/files/${fileId}/node/plotdata?${qs({ path, ...opts })}`,
    ),

  browse: (dir?: string) =>
    request<BrowseResult>(`/api/browse?${qs({ dir })}`),

  browseRoots: () => request<{ roots: BrowseRoot[] }>("/api/browse/roots"),

  /**
   * Fetch an export as a blob rather than pointing an <a download> at the URL.
   * A plain link cannot tell success from failure: when the server answers 400
   * the browser cheerfully saves the JSON error body as the "export", which
   * looks like a download that worked until you open it.
   */
  exportFile: async (
    fileId: string,
    path: string,
    opts: {
      format: "csv" | "xlsx";
      start?: number;
      stop?: number;
      cols?: string;
      slice?: string;
      use_time_index?: boolean;
    },
  ): Promise<{ blob: Blob; filename: string }> => {
    const url = `/api/files/${fileId}/node/export?${qs({ path, ...opts })}`;
    const response = await fetch(url, {
      headers: TOKEN ? { "X-H5Grid-Token": TOKEN } : {},
    });

    if (!response.ok) {
      let code = "error";
      let detail = response.statusText;
      try {
        const body = await response.json();
        code = body.error ?? code;
        detail = body.detail ?? detail;
      } catch {
        /* not JSON; the status text will have to do */
      }
      throw new ApiError(response.status, code, detail);
    }

    const disposition = response.headers.get("content-disposition") ?? "";
    const match = /filename="?([^"]+)"?/.exec(disposition);
    const leaf = path.replace(/\/+$/, "").split("/").pop() || "data";
    return {
      blob: await response.blob(),
      filename: match?.[1] ?? `${leaf}.${opts.format}`,
    };
  },

  /**
   * Download URLs followed by the browser itself cannot attach the token
   * header, so it goes in the query string instead. Kept for right-click and
   * "copy link"; the button path uses exportFile so failures are visible.
   */
  exportUrl: (
    fileId: string,
    path: string,
    opts: {
      format: "csv" | "xlsx";
      start?: number;
      stop?: number;
      cols?: string;
      slice?: string;
      use_time_index?: boolean;
    },
  ) =>
    `/api/files/${fileId}/node/export?${qs({
      path,
      ...opts,
      token: TOKEN || undefined,
    })}`,
};
