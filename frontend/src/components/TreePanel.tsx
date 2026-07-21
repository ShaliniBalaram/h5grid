import { useCallback, useMemo, useState } from "react";

import type { TreeNode } from "../api/types";

interface Props {
  root: TreeNode | null;
  selectedPath: string | null;
  onSelect: (node: TreeNode) => void;
  filter: string;
}

const ICONS: Record<string, string> = {
  group: "▸",
  dataset: "▦",
  pandas_frame: "▤",
  pandas_table: "▤",
  broken_link: "⚠",
};

function iconClass(kind: string) {
  if (kind === "group") return "group";
  if (kind === "dataset") return "dataset";
  if (kind === "broken_link") return "broken";
  return "table";
}

/** "30000 x 12  f64" — enough to judge a node without clicking it. */
function badge(node: TreeNode): string {
  if (node.kind === "broken_link") return "unresolved";
  if (node.kind === "group" || !node.shape) return "";
  const dims = node.shape.join(" × ");
  const dtype = shortDtype(node.dtype);
  return dtype ? `${dims}  ${dtype}` : dims;
}

function shortDtype(dtype: string | null): string {
  if (!dtype) return "";
  const map: Record<string, string> = {
    float64: "f64",
    float32: "f32",
    int64: "i64",
    int32: "i32",
    int16: "i16",
    int8: "i8",
    uint64: "u64",
    uint32: "u32",
    uint16: "u16",
    uint8: "u8",
    bool: "bool",
    compound: "rec",
    frame: "df",
    frame_table: "df",
  };
  return map[dtype] ?? dtype.replace(/^\|?[SU](\d+)$/, "str$1");
}

/** Abbreviated so the tag never crowds out the node name in a narrow panel. */
function shortAttribute(attribute: string): string {
  const map: Record<string, string> = {
    flow: "flow",
    volume: "vol",
    parameter: "param",
    "parameter-index": "idx",
  };
  return map[attribute] ?? attribute;
}

function matches(node: TreeNode, needle: string): boolean {
  if (!needle) return true;
  if (node.name.toLowerCase().includes(needle)) return true;
  return node.children.some((child) => matches(child, needle));
}

export default function TreePanel({ root, selectedPath, onSelect, filter }: Props) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const needle = filter.trim().toLowerCase();

  const toggle = useCallback((path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const rows = useMemo(() => {
    if (!root) return [];
    const out: { node: TreeNode; depth: number }[] = [];

    const walk = (node: TreeNode, depth: number) => {
      for (const child of node.children) {
        if (!matches(child, needle)) continue;
        out.push({ node: child, depth });
        const isOpen = needle ? true : !collapsed.has(child.path);
        if (child.children.length > 0 && isOpen) walk(child, depth + 1);
      }
    };

    walk(root, 0);
    return out;
  }, [root, collapsed, needle]);

  if (!root) return null;

  if (rows.length === 0) {
    return (
      <div className="tree" style={{ padding: "14px 12px", color: "var(--text-faint)" }}>
        {needle ? `Nothing matching “${filter}”.` : "This file has no datasets."}
      </div>
    );
  }

  return (
    <div className="tree" role="tree">
      {rows.map(({ node, depth }) => {
        const expandable = node.children.length > 0;
        const isOpen = needle ? true : !collapsed.has(node.path);
        const selected = node.path === selectedPath;
        const selectable =
          node.kind !== "group" && node.kind !== "broken_link";

        return (
          <div
            key={node.path}
            role="treeitem"
            aria-selected={selected}
            aria-expanded={expandable ? isOpen : undefined}
            className={`tree-row${selected ? " selected" : ""}`}
            style={{ paddingLeft: 6 + depth * 13 }}
            title={
              node.kind === "broken_link"
                ? `${node.path} — cannot be resolved (${node.error ?? "unknown"})`
                : `${node.path}${node.dtype ? ` — ${node.dtype}` : ""}`
            }
            onClick={() => {
              if (selectable) onSelect(node);
              else if (expandable) toggle(node.path);
            }}
          >
            <span
              className="tree-twisty"
              onClick={(event) => {
                if (!expandable) return;
                event.stopPropagation();
                toggle(node.path);
              }}
            >
              {expandable ? (isOpen ? "▼" : "▶") : ""}
            </span>
            <span className={`tree-icon ${iconClass(node.kind)}`}>
              {ICONS[node.kind] ?? "•"}
            </span>
            <span className="tree-name">{node.name}</span>
            {node.pywr?.attribute && (
              <span
                className={`pywr-tag ${node.pywr.attribute.split("-")[0]}`}
                title={
                  node.pywr.type
                    ? `${node.pywr.type} — recorded ${node.pywr.attribute}`
                    : `recorded ${node.pywr.attribute}`
                }
              >
                {shortAttribute(node.pywr.attribute)}
              </span>
            )}
            <span className="tree-badge">{badge(node)}</span>
          </div>
        );
      })}
    </div>
  );
}
