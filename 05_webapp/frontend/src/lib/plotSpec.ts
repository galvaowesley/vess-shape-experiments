import type { PlotSpec, TableSpec, WilcoxonSpec } from "./types";

export const COUNT_VALUE = "__count__";
export const ALL_METRICS = ["Dice", "IoU", "Precision", "Recall", "Accuracy", "AUC"];
export const TABLE_DIMENSIONS = ["model_type", "dataset", "stage", "regime", "num_samples"];
export const AGGREGATIONS = [
  { value: "mean_std", label: "Mean ± std" },
  { value: "mean", label: "Mean" },
  { value: "std", label: "Std" },
  { value: "median", label: "Median" },
  { value: "min", label: "Min" },
  { value: "max", label: "Max" },
  { value: "count", label: "Count" },
] as const;

export function defaultTableSpec(): TableSpec {
  return {
    source: "per_run",
    datasets: ["vessmap"],
    model_types: [],
    stages: [],
    regimes: [],
    sample_filter: [],
    exclude_mock: true,
    rows: ["model_type", "num_samples"],
    columns: [],
    values: ["Dice", "IoU", "AUC"],
    aggregation: "mean_std",
    decimals: 3,
    percentage: false,
    sort_by: null,
    ascending: false,
    title: "",
  };
}

export function hydrateTableSpec(saved: Partial<TableSpec> | Record<string, unknown>): TableSpec {
  return { ...defaultTableSpec(), ...(saved as Partial<TableSpec>) };
}

// One-click presets mirroring the notebook tables.
export const TABLE_PRESETS: { name: string; description: string; build: () => TableSpec }[] = [
  {
    name: "Summary (mean ± std)",
    description: "Metrics per model & N ∈ {0,1,20} — like build_summary_table.",
    build: () => {
      const s = defaultTableSpec();
      s.rows = ["model_type", "num_samples"];
      s.values = ["Dice", "IoU", "Precision", "Recall", "AUC"];
      s.aggregation = "mean_std";
      s.sample_filter = [0, 1, 20];
      s.title = "Summary (mean ± std)";
      return s;
    },
  },
  {
    name: "Coverage (count per dataset)",
    description: "How many runs exist per model × dataset.",
    build: () => {
      const s = defaultTableSpec();
      s.datasets = ["vessmap", "drive", "dca1", "octa2d"];
      s.rows = ["model_type"];
      s.columns = ["dataset"];
      s.values = [COUNT_VALUE];
      s.aggregation = "count";
      s.title = "Experiment coverage";
      return s;
    },
  },
  {
    name: "Zero-shot ranking",
    description: "Zero-shot models ranked by Dice (N=0).",
    build: () => {
      const s = defaultTableSpec();
      s.stages = ["zero_shot"];
      s.rows = ["model_type"];
      s.values = ["Dice", "IoU", "Precision", "Recall", "AUC"];
      s.aggregation = "mean";
      s.sort_by = "Dice";
      s.ascending = false;
      s.title = "Zero-shot ranking";
      return s;
    },
  },
];

export function defaultPlotSpec(): PlotSpec {
  return {
    chart_type: "line",
    data: {
      source: "per_run",
      datasets: ["vessmap"],
      model_types: [],
      stages: [],
      regimes: [],
      num_samples_range: null,
    },
    encoding: { x: "num_samples", y: "Dice", series: "model_type" },
    facet: null,
    aggregate_datasets: "none",
    palette: null,
    x_axis: {
      label: null,
      label_fontsize: 14,
      tick_fontsize: 11,
      tick_rotation: 0,
      min: null,
      max: null,
      tick_step: null,
      percentage: false,
    },
    y_axis: {
      label: null,
      label_fontsize: 14,
      tick_fontsize: 11,
      tick_rotation: 0,
      min: 0,
      max: 1,
      tick_step: null,
      percentage: false,
    },
    title: { text: "", fontsize: 16 },
    fonts: { base: 12 },
    series_styles: {},
    series_order: [],
    category_order: [],
    legend: { show: true, title: "Model", loc: "best", fontsize: 11 },
    show_error_band: true,
    error_type: "std",
    error_alpha: 0.15,
    show_grid: true,
    marker_size: 36,
    figure: { size: [9, 5.5] },
    export: { format: "svg", dpi: 300, save_path: null },
  };
}

/** Merge a saved/older PlotSpec with current defaults so re-editing a pinned
 *  figure never trips over fields added after it was pinned. */
export function hydratePlotSpec(saved: Partial<PlotSpec> | Record<string, unknown>): PlotSpec {
  const d = defaultPlotSpec();
  const s = saved as Partial<PlotSpec>;
  return {
    ...d,
    ...s,
    data: { ...d.data, ...(s.data ?? {}) },
    encoding: { ...d.encoding, ...(s.encoding ?? {}) },
    x_axis: { ...d.x_axis, ...(s.x_axis ?? {}) },
    y_axis: { ...d.y_axis, ...(s.y_axis ?? {}) },
    title: { ...d.title, ...(s.title ?? {}) },
    fonts: { ...d.fonts, ...(s.fonts ?? {}) },
    legend: { ...d.legend, ...(s.legend ?? {}) },
    figure: { ...d.figure, ...(s.figure ?? {}) },
    export: { ...d.export, ...(s.export ?? {}) },
    series_styles: s.series_styles ?? {},
    series_order: s.series_order ?? [],
    category_order: s.category_order ?? [],
  };
}

export function defaultWilcoxonSpec(): WilcoxonSpec {
  return {
    reference_model: "VSUNet18",
    comparison_models: ["IN-UNet18", "LiteMedSAM-FT"],
    datasets: ["dca1", "drive", "octa2d", "vessmap"],
    metric: "Dice",
    n_values: [0, 1],
    alpha: 0.05,
    alternative: "greater",
    min_common_images: 5,
    row_axis: "dataset",
    col_axis: "comparison",
    annotate_pvalues: false,
    colors: { significant: "#1a5276", nonsignificant: "#d6e4f0", nan: "#cccccc" },
    title: { text: "", fontsize: 15 },
    fonts: { base: 12 },
    figure: { size: [12, 4.5] },
    export: { format: "svg", dpi: 300, save_path: null },
  };
}

// One-click presets that approximate the existing paper figures.
export const PRESETS: { name: string; description: string; build: () => PlotSpec }[] = [
  {
    name: "Few-shot learning curve",
    description: "Dice vs # samples, one line per model (mean ± std).",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "line";
      s.encoding = { x: "num_samples", y: "Dice", series: "model_type" };
      s.x_axis.label = "# Samples";
      s.y_axis = { ...s.y_axis, label: "Dice", min: 0, max: 1 };
      s.title.text = "Few-shot performance";
      return s;
    },
  },
  {
    name: "Mean across datasets",
    description: "One line per model: metric averaged over all datasets vs # samples.",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "line";
      s.data.datasets = ["dca1", "drive", "octa2d", "vessmap"];
      s.aggregate_datasets = "macro";
      s.encoding = { x: "num_samples", y: "Dice", series: "model_type" };
      s.x_axis.label = "# Samples";
      s.y_axis = { ...s.y_axis, label: "Dice (mean over datasets)", min: 0, max: 1 };
      s.title.text = "Mean across datasets";
      return s;
    },
  },
  {
    name: "Grouped bar @ N",
    description: "Bars per dataset grouped by model (filter N via samples range).",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "bar";
      s.data.datasets = ["dca1", "drive", "octa2d", "vessmap"];
      s.data.num_samples_range = [1, 1];
      s.encoding = { x: "dataset", y: "Dice", series: "model_type" };
      s.x_axis.label = "Dataset";
      s.y_axis = { ...s.y_axis, label: "Dice", min: 0, max: 1 };
      s.title.text = "One-shot Dice by dataset";
      return s;
    },
  },
  {
    name: "Zero-shot bar (N=0)",
    description: "Bars per dataset for the zero-shot models (num_samples = 0).",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "bar";
      s.data.datasets = ["dca1", "drive", "octa2d", "vessmap"];
      s.data.num_samples_range = [0, 0];
      s.encoding = { x: "dataset", y: "Dice", series: "model_type" };
      s.x_axis.label = "Dataset";
      s.y_axis = { ...s.y_axis, label: "Dice", min: 0, max: 1 };
      s.title.text = "Zero-shot Dice by dataset";
      return s;
    },
  },
  {
    name: "Precision vs Recall",
    description: "Scatter of per-run Precision/Recall, colored by model.",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "scatter";
      s.encoding = { x: "Recall", y: "Precision", series: "model_type" };
      s.x_axis = { ...s.x_axis, label: "Recall", min: 0, max: 1 };
      s.y_axis = { ...s.y_axis, label: "Precision", min: 0, max: 1 };
      s.title.text = "Precision vs Recall";
      return s;
    },
  },
  {
    name: "Curves faceted by regime",
    description: "Learning curves split into Zero-/One-/Few-shot panels.",
    build: () => {
      const s = defaultPlotSpec();
      s.chart_type = "line";
      s.facet = "regime";
      s.encoding = { x: "num_samples", y: "Dice", series: "model_type" };
      s.x_axis.label = "# Samples";
      s.y_axis = { ...s.y_axis, label: "Dice", min: 0, max: 1 };
      return s;
    },
  },
];

/** Merge a saved/older WilcoxonSpec with current defaults (re-edit safety). */
export function hydrateWilcoxonSpec(
  saved: Partial<WilcoxonSpec> | Record<string, unknown>,
): WilcoxonSpec {
  const d = defaultWilcoxonSpec();
  const s = saved as Partial<WilcoxonSpec>;
  return {
    ...d,
    ...s,
    colors: { ...d.colors, ...(s.colors ?? {}) },
    title: { ...d.title, ...(s.title ?? {}) },
    fonts: { ...d.fonts, ...(s.fonts ?? {}) },
    figure: { ...d.figure, ...(s.figure ?? {}) },
    export: { ...d.export, ...(s.export ?? {}) },
  };
}

export const LINESTYLES = [
  { value: "-", label: "Solid" },
  { value: "--", label: "Dashed" },
  { value: "-.", label: "Dash-dot" },
  { value: ":", label: "Dotted" },
];

export const MARKERS = [
  { value: "o", label: "Circle" },
  { value: "s", label: "Square" },
  { value: "^", label: "Triangle" },
  { value: "D", label: "Diamond" },
  { value: "v", label: "Tri-down" },
  { value: "*", label: "Star" },
  { value: "x", label: "Cross" },
  { value: "none", label: "None" },
];

export const LEGEND_LOCS = [
  "best",
  "upper right",
  "upper left",
  "lower left",
  "lower right",
  "center left",
  "center right",
  "upper center",
  "lower center",
];
