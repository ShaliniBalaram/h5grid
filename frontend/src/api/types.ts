export type NodeKind =
  | "group"
  | "dataset"
  | "pandas_frame"
  | "pandas_table"
  | "broken_link";

export interface TreeNode {
  name: string;
  path: string;
  kind: NodeKind;
  shape: number[] | null;
  dtype: string | null;
  nrows: number | null;
  ncols: number | null;
  ndim: number | null;
  decode_fallback: boolean;
  /** Present only on broken_link nodes: why the link could not be resolved. */
  error?: string;
  /** What pywr recorded here, when the file came from a TablesRecorder. */
  pywr?: PywrNodeInfo | null;
  children: TreeNode[];
}

export interface PywrNodeInfo {
  /** flow | volume | parameter | parameter-index */
  attribute?: string;
  /** The model node's Python class, e.g. Reservoir */
  type?: string;
}

export interface ScenarioInfo {
  scenarios: { name: string; size: number }[];
  collapsed: boolean;
  combination_count: number | null;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  is_datetime: boolean;
}

export interface NodeMeta {
  path: string;
  kind: NodeKind;
  shape: number[] | null;
  ndim: number;
  dtype: string | null;
  chunks: number[] | null;
  compression: string | null;
  nrows: number;
  columns: ColumnInfo[];
  attrs: Record<string, unknown>;
  time_index_available: boolean;
  supports_row_slicing: boolean;
  decode_fallback: string | null;
  dim_sizes: number[] | null;
  /** One entry per dimension; the scenario name where one is known. */
  dim_names: (string | null)[] | null;
  pywr: PywrNodeInfo | null;
  scenarios: ScenarioInfo | null;
}

/** Infinities arrive as strings so the payload stays valid JSON. */
export type CellValue = number | string | boolean | null;

export interface DataPage {
  path: string;
  start: number;
  stop: number;
  total_rows: number;
  columns: string[];
  column_types: ColumnInfo[];
  rows: CellValue[][];
  time_index_applied: boolean;
}

export interface OpenFileInfo {
  file_id: string;
  path: string;
  name: string;
  size_bytes: number;
  mtime: number;
}

export interface ColumnStats {
  column: string;
  numeric: boolean;
  min: number | string | null;
  max: number | string | null;
  mean: number | string | null;
  std: number | null;
  nan_count: number;
  count: number;
  total: number;
  distinct_count?: number;
}

export interface SearchResult {
  column: string;
  query: string;
  rows: number[];
  truncated: boolean;
  scanned_rows: number;
}

export interface PlotPayload {
  x: (number | string)[];
  x_is_date: boolean;
  /** Source row of each point, so a zoom selection maps back to a row range. */
  rows: number[];
  series: { name: string; y: (number | string | null)[] }[];
  decimated: boolean;
  bucket_size?: number;
  start: number;
  stop: number;
  window_rows: number;
  total_rows: number;
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size_bytes: number | null;
  is_h5: boolean;
}

export interface BrowseResult {
  dir: string;
  parent: string | null;
  breadcrumbs: { name: string; path: string }[];
  entries: BrowseEntry[];
}

export interface BrowseRoot {
  name: string;
  path: string;
  kind: "home" | "volume" | "root" | "cwd" | "folder";
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get isFileChanged() {
    return this.status === 409 || this.code === "file_changed";
  }

  get isUnknownFile() {
    return this.status === 404 && this.code === "not_found";
  }
}
