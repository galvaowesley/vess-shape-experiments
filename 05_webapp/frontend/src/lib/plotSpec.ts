import type { PlotSpec, WilcoxonSpec } from "./types";

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
    palette: null,
    x_axis: {
      label: null,
      label_fontsize: 14,
      tick_fontsize: 11,
      tick_rotation: 0,
      min: null,
      max: null,
      percentage: false,
    },
    y_axis: {
      label: null,
      label_fontsize: 14,
      tick_fontsize: 11,
      tick_rotation: 0,
      min: 0,
      max: 1,
      percentage: false,
    },
    title: { text: "", fontsize: 16 },
    fonts: { base: 12 },
    series_styles: {},
    legend: { show: true, title: "Model", loc: "best", fontsize: 11 },
    show_error_band: true,
    show_grid: true,
    marker_size: 36,
    figure: { size: [9, 5.5] },
    export: { format: "svg", dpi: 300, save_path: null },
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
