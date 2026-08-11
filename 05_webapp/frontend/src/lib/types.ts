// Mirrors backend app/schemas.py

export type ChartType = "line" | "bar" | "scatter";
export type Source = "per_run" | "per_image" | "summary";
export type ExportFormat = "svg" | "png" | "jpg" | "pdf";

export interface DataSpec {
  source: Source;
  datasets: string[];
  model_types: string[];
  stages: string[];
  regimes: string[];
  num_samples_range?: [number, number] | null;
}

export interface Encoding {
  x: string;
  y: string;
  series?: string | null;
}

export interface AxisSpec {
  label?: string | null;
  label_fontsize: number;
  tick_fontsize: number;
  tick_rotation: number;
  min?: number | null;
  max?: number | null;
  tick_step?: number | null;
  percentage: boolean;
}

export interface TitleSpec {
  text: string;
  fontsize: number;
}

export interface SeriesStyle {
  color?: string | null;
  linestyle?: string | null;
  marker?: string | null;
  width?: number | null;
}

export interface LegendSpec {
  show: boolean;
  title?: string | null;
  loc: string;
  fontsize: number;
}

export interface ExportSpec {
  format: ExportFormat;
  dpi: number;
  save_path?: string | null;
}

export type AggregateDatasets = "none" | "pool" | "macro";

export interface PlotSpec {
  chart_type: ChartType;
  data: DataSpec;
  encoding: Encoding;
  facet?: string | null;
  aggregate_datasets: AggregateDatasets;
  palette?: string | null;
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  title: TitleSpec;
  fonts: Record<string, number>;
  series_styles: Record<string, SeriesStyle>;
  series_order: string[];
  category_order: string[];
  legend: LegendSpec;
  show_error_band: boolean;
  error_type: "std" | "sem";
  error_alpha: number;
  show_grid: boolean;
  marker_size: number;
  figure: { size: [number, number] };
  export: ExportSpec;
}

/** Global colors/order applied to every chart (persisted webapp-side). */
export interface StylePrefs {
  colors: Record<string, string>;
  order: Record<string, string[]>;
}
export interface StylePrefsResponse extends StylePrefs {
  defaults: Record<string, string>;
}

export interface WilcoxonColors {
  significant: string;
  nonsignificant: string;
  nan: string;
}

export interface WilcoxonSpec {
  reference_model: string;
  comparison_models: string[];
  datasets: string[];
  metric: string;
  n_values: number[];
  alpha: number;
  alternative: "greater" | "less" | "two-sided";
  min_common_images: number;
  row_axis: "dataset" | "comparison";
  col_axis: "comparison" | "dataset";
  annotate_pvalues: boolean;
  colors: WilcoxonColors;
  title: TitleSpec;
  fonts: Record<string, number>;
  figure: { size: [number, number] };
  export: ExportSpec;
}

export interface FieldOption {
  type: "categorical" | "numeric";
  values?: (string | number)[];
  min?: number;
  max?: number;
}

export interface Metadata {
  datasets: string[];
  metrics: string[];
  source: string;
  available_columns: string[];
  x_fields: string[];
  y_fields: string[];
  series_fields: string[];
  facet_fields: string[];
  options: Record<string, FieldOption>;
  regimes: string[];
  row_count: number;
}

export interface PaletteEntry {
  name: string;
  colors: string[];
}
export type PaletteCatalog = Record<string, PaletteEntry[]>;

export type Aggregation = "mean" | "std" | "mean_std" | "median" | "min" | "max" | "count";

export interface TableSpec {
  source: Source;
  datasets: string[];
  model_types: string[];
  stages: string[];
  regimes: string[];
  sample_filter: number[];
  exclude_mock: boolean;
  rows: string[];
  columns: string[];
  values: string[]; // metric names or "__count__"
  aggregation: Aggregation;
  decimals: number;
  percentage: boolean;
  sort_by?: string | null;
  ascending: boolean;
  title: string;
}

export interface TableResult {
  columns: string[];
  rows: (string | number | null)[][];
  caption: string;
  n_rows: number;
  markdown: string;
  csv: string;
  latex: string;
}

// --------------------------------------------------------------------------- //
// Figure Studio — inference browser + paper-figure grid composer
// --------------------------------------------------------------------------- //

export type PanelKind = "pred" | "input" | "gt" | "empty";
export type RoiMode = "marker" | "crop" | "inset";
export type RoiScope = "image" | "panel" | "column" | "figure";
export type RunPolicy = "median" | "best" | "worst" | "fixed";

export interface PanelRef {
  kind: PanelKind;
  dataset: string;
  image: string;
  stage?: string | null;
  experiment?: string | null;
  run_name?: string | null;
  // display-only metadata (never used to resolve the file path)
  model_type?: string | null;
  num_samples?: number | null;
  run?: number | null;
  rep?: number | null;
  score?: number | null;
  /** Empty cells only: force the "no inference" mark on/off. null = automatic. */
  missing?: boolean | null;
}

/** ROI in normalized frame coords (0..1) so it transfers across resolutions. */
export interface RoiSpec {
  x: number;
  y: number;
  w: number;
  h: number;
  mode: RoiMode;
  color: string;
  linewidth: number;
  inset_corner: "upper right" | "upper left" | "lower right" | "lower left";
  inset_scale: number;
  inset_connectors: boolean;
}

/** Pre-ROI zoom: how much of the frame a panel shows, before the ROI is drawn
 *  inside it. Same normalized coords and same group key as RoiSpec. */
export interface CropSpec {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PanelStyle {
  show_label: boolean;
  show_metric: boolean;
  metric: string;
  decimals: number;
  label_loc: "top" | "bottom" | "none";
  metric_loc: "upper left" | "upper right" | "lower left" | "lower right";
  metric_color: string;
  metric_fontsize: number;
  /** Chip behind the metric text; alpha 0 hides it. */
  metric_bg_color: string;
  metric_bg_alpha: number;
  border: boolean;
  border_color: string;
  border_width: number;
  invert: boolean;
  /** Draw a boxed X where a (model, num_samples) combination has no inference. */
  missing_mark: boolean;
  missing_color: string | null;   // null -> follow label_color
  missing_width: number;
}

export interface GridFigureSpec {
  rows: number;
  cols: number;
  /** Row-major, length rows*cols. `kind:"empty"` leaves a gap. */
  panels: PanelRef[];
  row_labels: string[];
  col_labels: string[];
  roi_scope: RoiScope;
  /** Keyed by scope, e.g. "<dataset>:<image>" when roi_scope === "image". */
  rois: Record<string, RoiSpec>;
  /** Pre-ROI zoom, keyed exactly like `rois`. Absent = whole frame. */
  crops: Record<string, CropSpec>;
  panel_style: PanelStyle;
  panel_size: number;
  wspace: number;
  hspace: number;
  background: string;
  label_color: string;
  label_fontsize: number;
  row_label_fontsize: number;
  col_label_fontsize: number;
  title: TitleSpec;
  export: ExportSpec;
}

export interface RankImagesRequest {
  dataset: string;
  num_samples: number[];
  model_types: string[];
  stages: string[];
  metric: string;
  agg: "mean" | "median" | "min" | "max" | "spread";
  ascending: boolean;
  limit: number;
}

export interface RankedImage {
  image: string;
  score: number;
  n: number;
  per_model: Record<string, number>;
}

export interface RankRunsRequest {
  dataset: string;
  image: string;
  model_type: string;
  num_samples?: number | null;
  metric: string;
  ascending: boolean;
  limit: number;
}

/** One candidate run for a cell — already carries everything a PanelRef needs. */
export interface RankedRun {
  image: string;
  score: number;
  stage: string;
  experiment: string;
  run_name: string;
  model_type: string;
  num_samples: number;
  run?: number | null;
  rep?: number | null;
  is_median: boolean;
  is_best: boolean;
}

export interface AutoFillRequest {
  dataset: string;
  image: string;
  model_types: string[];
  num_samples: number[];
  metric: string;
  policy: RunPolicy;
  fixed_run?: number | null;
  fixed_rep?: number | null;
  orientation: "models_as_cols" | "models_as_rows";
  include_input: boolean;
  include_gt: boolean;
}

export interface FigureOptions {
  datasets: string[];
  metrics: string[];
  model_types: string[];
  stages: string[];
  /** Per dataset: available num_samples and the test-image list. */
  by_dataset: Record<
    string,
    { num_samples: number[]; images: string[]; model_types: string[] }
  >;
}

/** Where the backend reads input images and ground truth from. */
export interface FigureSettings {
  dataset_root: string;
  default_root: string;
  is_default: boolean;
  found: boolean;
  sources: Record<string, { path: string; found: boolean }>;
}

export interface GridLayout {
  id?: string;
  name: string;
  spec: GridFigureSpec | Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export type DashboardKind = "figure" | "wilcoxon" | "table" | "grid";

export interface DashboardItem {
  id?: string;
  dashboard_id?: string | null;
  title: string;
  kind: DashboardKind;
  regime?: string | null;
  spec: PlotSpec | WilcoxonSpec | TableSpec | GridFigureSpec | Record<string, unknown>;
  created_at?: string;
}

export interface DashboardMeta {
  id: string;
  name: string;
  created_at?: string;
  count?: number;
}

export interface TrainingOptions {
  datasets: string[];
  stages: {
    value: "scratch" | "finetune";
    label: string;
    models: string[];
    pretraining: string[];
  }[];
  optimizers: string[];
  loss_functions: string[];
  checkpoint_types: string[];
  channels: string[];
  tta_types: string[];
  validation_metrics: string[];
  dataset_defaults: Record<string, { resize: string; channels: string }>;
}

export interface TrainingRequest {
  stage: "scratch" | "finetune";
  dataset: string;
  model_class: "resnet18_unet" | "resnet50_unet" | "litemedsam";
  pretraining: "scratch" | "imagenet" | "vessshape";
  weights_path?: string | null;
  experiment_name?: string | null;
  config_filename?: string | null;
  dataset_path: string;
  csv_path: string;
  lr: number;
  bs_train: number;
  bs_valid: number;
  num_epochs: number;
  weight_decay: number;
  lr_decay: number;
  optimizer: string;
  momentum: number;
  validate_every: number;
  validation_metric: string;
  maximize_validation_metric: boolean;
  patience?: number | null;
  loss_function?: string | null;
  resize_size: string;
  channels?: string | null;
  augmentation_strategy?: string | null;
  dataset_params?: string | null;
  model_params?: string | null;
  num_workers: number;
  ignore_class_weights: boolean;
  // logging & wandb
  log_wandb: boolean;
  wandb_project?: string | null;
  wandb_group?: string | null;
  save_val_imgs: boolean;
  val_img_indices: string;
  disable_tqdm: boolean;
  meta?: string | null;
  // checkpointing
  checkpoint_every: number;
  copy_model_every: number;
  suppress_checkpoint: boolean;
  suppress_best_checkpoint: boolean;
  // device & efficiency
  device: string;
  use_amp: boolean;
  deterministic: boolean;
  benchmark: boolean;
  // few-shot loop
  min_samples: number;
  max_samples: number;
  step: number;
  runs: number;
  reps: number;
  with_replacement: boolean;
  output_dir: string;
  weights_id?: string | null;
  // inference
  checkpoint_type: "last" | "best";
  enable_inference: boolean;
  batch_inference: boolean;
  save_inference_images: boolean;
  delete_checkpoint: boolean;
  inference_dir_name: string;
  tta_type: "none" | "logits" | "probs";
  threshold: number;
  test_use_amp: boolean;
  imagenet_normalize: boolean;
  skip_checkpoint_loading: boolean;
  force_headless: boolean;
  skip_boxplot: boolean;
  max_inference_retries: number;
  delete_only_on_success: boolean;
  aggregate_inference_means: boolean;
}

export interface TrainingStatus {
  running: boolean;
  returncode?: number | null;
  stage?: string;
  dataset?: string;
  config_file?: string;
  experiment_name?: string;
  pid?: number;
  started_at?: string;
  cmd?: string;
}
