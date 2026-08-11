import type { CropSpec, GridFigureSpec, PanelRef, PanelStyle, RoiSpec } from "./types";

export const LS_KEY = "vesslab.figureSpec";

export function defaultRoi(): RoiSpec {
  return {
    x: 0.35,
    y: 0.35,
    w: 0.2,
    h: 0.2,
    mode: "marker",
    color: "#ff2d55",
    linewidth: 1.5,
    inset_corner: "lower right",
    // Fraction of the panel the inset occupies (matches RoiSpec.inset_scale
    // in schemas.py) — NOT a zoom factor.
    inset_scale: 0.42,
    inset_connectors: true,
  };
}

/** Full frame — a crop equal to this is the same as having none. */
export function defaultCrop(): CropSpec {
  return { x: 0, y: 0, w: 1, h: 1 };
}

export function isFullFrame(c: CropSpec | undefined): boolean {
  return !c || (c.x <= 0 && c.y <= 0 && c.w >= 1 && c.h >= 1);
}

export function defaultPanelStyle(): PanelStyle {
  return {
    show_label: true,
    show_metric: true,
    metric: "Dice",
    decimals: 3,
    label_loc: "bottom",
    metric_loc: "lower right",
    metric_color: "#ffffff",
    metric_fontsize: 9,
    metric_bg_color: "#000000",
    metric_bg_alpha: 0.55,
    border: true,
    border_color: "#333333",
    border_width: 1,
    invert: false,
    missing_mark: true,
    missing_color: null,
    missing_width: 0.8,
  };
}

export function emptyPanel(): PanelRef {
  return {
    kind: "empty",
    dataset: "",
    image: "",
    stage: null,
    experiment: null,
    run_name: null,
    model_type: null,
    num_samples: null,
    run: null,
    rep: null,
    score: null,
    missing: null,
  };
}

/** Empty cells that stand for a row x column combination with no inference.
 *
 *  `panel.missing` is an explicit user override and always wins. Otherwise a
 *  cell that kept its identity counts — that is what autofill emits when no run
 *  exists for a (model, num_samples) pair. Failing that, position decides: an
 *  empty cell whose row *and* column both hold predictions is a hole inside the
 *  populated block, while the blank beside an input/GT column sits in a column
 *  with no predictions and stays a plain gap. The positional rule is what
 *  catches hand-built grids and cells that lost their identity to a clear or a
 *  drag. Mirrors `_unavailable_indices` in backend/app/figuregrid.py. */
export function unavailableIndices(panels: PanelRef[], cols: number): Set<number> {
  const predRows = new Set<number>();
  const predCols = new Set<number>();
  panels.forEach((p, i) => {
    if (p.kind === "pred") {
      predRows.add(Math.floor(i / cols));
      predCols.add(i % cols);
    }
  });
  const out = new Set<number>();
  panels.forEach((p, i) => {
    if (p.kind !== "empty") return;
    if (p.missing != null) {
      if (p.missing) out.add(i);
      return;
    }
    if (p.model_type || p.num_samples != null) out.add(i);
    else if (predRows.has(Math.floor(i / cols)) && predCols.has(i % cols)) out.add(i);
  });
  return out;
}

export function defaultGridSpec(): GridFigureSpec {
  const rows = 3;
  const cols = 4;
  return {
    rows,
    cols,
    panels: Array.from({ length: rows * cols }, emptyPanel),
    row_labels: Array.from({ length: rows }, () => ""),
    col_labels: Array.from({ length: cols }, () => ""),
    roi_scope: "image",
    rois: {},
    crops: {},
    panel_style: defaultPanelStyle(),
    panel_size: 2.4,
    wspace: 0.05,
    hspace: 0.05,
    background: "#ffffff",
    label_color: "#111111",
    label_fontsize: 11,
    row_label_fontsize: 11,
    col_label_fontsize: 11,
    title: { text: "", fontsize: 16 },
    export: { format: "svg", dpi: 300, save_path: null },
  };
}

function hydratePanel(raw: unknown): PanelRef {
  return { ...emptyPanel(), ...((raw ?? {}) as Partial<PanelRef>) };
}

function hydrateRoi(raw: unknown): RoiSpec {
  return { ...defaultRoi(), ...((raw ?? {}) as Partial<RoiSpec>) };
}

/** Pad/truncate a flat panel list to exactly rows*cols entries. */
function normalizePanels(raw: unknown[], rows: number, cols: number): PanelRef[] {
  const n = rows * cols;
  const panels = raw.slice(0, n).map(hydratePanel);
  while (panels.length < n) panels.push(emptyPanel());
  return panels;
}

function normalizeLabels(raw: unknown[], n: number): string[] {
  const labels = raw.slice(0, n).map((v) => (typeof v === "string" ? v : ""));
  while (labels.length < n) labels.push("");
  return labels;
}

/** Merge a saved/older GridFigureSpec with current defaults so a schema change
 *  (or a stale localStorage blob) never white-screens the app. Also re-pads
 *  `panels`/`row_labels`/`col_labels` to match rows*cols exactly. */
export function hydrateGridSpec(raw: unknown): GridFigureSpec {
  const d = defaultGridSpec();
  const s = (raw && typeof raw === "object" ? raw : {}) as Partial<GridFigureSpec>;
  const rows = typeof s.rows === "number" && s.rows > 0 ? s.rows : d.rows;
  const cols = typeof s.cols === "number" && s.cols > 0 ? s.cols : d.cols;
  return {
    ...d,
    ...s,
    rows,
    cols,
    panels: normalizePanels(Array.isArray(s.panels) ? s.panels : [], rows, cols),
    row_labels: normalizeLabels(Array.isArray(s.row_labels) ? s.row_labels : [], rows),
    col_labels: normalizeLabels(Array.isArray(s.col_labels) ? s.col_labels : [], cols),
    rois: Object.fromEntries(
      Object.entries(s.rois ?? {}).map(([k, v]) => [k, hydrateRoi(v)]),
    ),
    crops: Object.fromEntries(
      Object.entries(s.crops ?? {}).map(([k, v]) => [
        k,
        { ...defaultCrop(), ...((v ?? {}) as Partial<CropSpec>) },
      ]),
    ),
    panel_style: { ...d.panel_style, ...(s.panel_style ?? {}) },
    title: { ...d.title, ...(s.title ?? {}) },
    export: { ...d.export, ...(s.export ?? {}) },
  };
}

/** The ROI group key for a panel at `index` — this is what makes one drawn ROI
 *  apply to every panel showing the same image (scope "image"), a whole
 *  column, a single panel, or the whole figure. */
export function roiKey(spec: GridFigureSpec, panel: PanelRef, index: number): string {
  switch (spec.roi_scope) {
    case "image":
      return `${panel.dataset}:${panel.image}`;
    case "column":
      return `col:${index % spec.cols}`;
    case "panel":
      return `panel:${index}`;
    case "figure":
    default:
      return "figure";
  }
}

/** Re-lay out `panels` row-major when the grid dimensions change, preserving
 *  every panel that still fits in its original row/column. */
export function resizeGrid(spec: GridFigureSpec, rows: number, cols: number): GridFigureSpec {
  const panels: PanelRef[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const fits = r < spec.rows && c < spec.cols;
      panels.push(fits ? spec.panels[r * spec.cols + c] : emptyPanel());
    }
  }
  const row_labels = Array.from({ length: rows }, (_, r) => spec.row_labels[r] ?? "");
  const col_labels = Array.from({ length: cols }, (_, c) => spec.col_labels[c] ?? "");
  return { ...spec, rows, cols, panels, row_labels, col_labels };
}

/** Short human label for a panel, e.g. "VSUNet18 · 1-shot", "Zero-shot", "Input". */
export function panelLabel(p: PanelRef): string {
  if (p.kind === "input") return "Input";
  if (p.kind === "gt") return "Ground truth";
  if (p.kind === "empty") return "";
  const shot =
    p.num_samples === 0 ? "Zero-shot" : p.num_samples != null ? `${p.num_samples}-shot` : null;
  if (p.model_type && shot) return `${p.model_type} · ${shot}`;
  return p.model_type ?? shot ?? "Prediction";
}
