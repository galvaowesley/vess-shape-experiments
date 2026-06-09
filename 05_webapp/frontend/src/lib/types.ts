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

export interface PlotSpec {
  chart_type: ChartType;
  data: DataSpec;
  encoding: Encoding;
  facet?: string | null;
  palette?: string | null;
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  title: TitleSpec;
  fonts: Record<string, number>;
  series_styles: Record<string, SeriesStyle>;
  legend: LegendSpec;
  show_error_band: boolean;
  show_grid: boolean;
  marker_size: number;
  figure: { size: [number, number] };
  export: ExportSpec;
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

export interface DashboardItem {
  id?: string;
  title: string;
  kind: "figure" | "wilcoxon";
  regime?: string | null;
  spec: PlotSpec | WilcoxonSpec | Record<string, unknown>;
  created_at?: string;
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
  validate_every: number;
  validation_metric: string;
  loss_function?: string | null;
  resize_size: string;
  channels?: string | null;
  num_workers: number;
  ignore_class_weights: boolean;
  min_samples: number;
  max_samples: number;
  step: number;
  runs: number;
  reps: number;
  with_replacement: boolean;
  output_dir: string;
  weights_id?: string | null;
  checkpoint_type: "last" | "best";
  enable_inference: boolean;
  batch_inference: boolean;
  save_inference_images: boolean;
  delete_checkpoint: boolean;
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
