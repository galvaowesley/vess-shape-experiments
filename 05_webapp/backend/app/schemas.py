"""Pydantic contracts shared with the frontend (mirrored in lib/plotSpec.ts)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ChartType = Literal["line", "bar", "scatter"]
Source = Literal["per_run", "per_image", "summary"]
ExportFormat = Literal["svg", "png", "jpg", "pdf"]


# --------------------------------------------------------------------------- #
# Chart builder
# --------------------------------------------------------------------------- #
class DataSpec(BaseModel):
    source: Source = "per_run"
    datasets: list[str] = Field(default_factory=list)
    model_types: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    regimes: list[str] = Field(default_factory=list)  # Zero-shot / One-shot / Few-shot
    num_samples_range: Optional[list[float]] = None    # [min, max] inclusive


class Encoding(BaseModel):
    x: str = "num_samples"
    y: str = "Dice"
    series: Optional[str] = "model_type"


class AxisSpec(BaseModel):
    label: Optional[str] = None
    label_fontsize: float = 14
    tick_fontsize: float = 11
    tick_rotation: float = 0
    min: Optional[float] = None
    max: Optional[float] = None
    tick_step: Optional[float] = None   # spacing between numeric ticks (e.g. 2 -> 0,2,4,…)
    percentage: bool = False


class TitleSpec(BaseModel):
    text: str = ""
    fontsize: float = 16


class SeriesStyle(BaseModel):
    color: Optional[str] = None
    linestyle: Optional[str] = None     # '-', '--', '-.', ':'
    marker: Optional[str] = None        # 'o', 's', '^', 'D', 'x', '*', None
    width: Optional[float] = None


class LegendSpec(BaseModel):
    show: bool = True
    title: Optional[str] = None
    loc: str = "best"
    fontsize: float = 11


class FigureSpec(BaseModel):
    size: list[float] = Field(default_factory=lambda: [9.0, 5.5])


class ExportSpec(BaseModel):
    format: ExportFormat = "svg"
    dpi: int = 300
    save_path: Optional[str] = None


class PlotSpec(BaseModel):
    chart_type: ChartType = "line"
    data: DataSpec = Field(default_factory=DataSpec)
    encoding: Encoding = Field(default_factory=Encoding)
    facet: Optional[str] = None                       # e.g. "regime" -> subplots
    # Collapse the selected datasets into one mean curve vs the x axis:
    #   "none"  -> keep every row as-is
    #   "pool"  -> simple mean over all rows (bigger datasets weigh more)
    #   "macro" -> mean of per-dataset means (each dataset weighs equally)
    aggregate_datasets: Literal["none", "pool", "macro"] = "none"
    palette: Optional[str] = None                     # seaborn named palette
    x_axis: AxisSpec = Field(default_factory=AxisSpec)
    y_axis: AxisSpec = Field(default_factory=AxisSpec)
    title: TitleSpec = Field(default_factory=TitleSpec)
    fonts: dict[str, float] = Field(default_factory=lambda: {"base": 12.0})
    series_styles: dict[str, SeriesStyle] = Field(default_factory=dict)
    # Per-figure ordering overrides (fall back to the global style prefs).
    series_order: list[str] = Field(default_factory=list)
    category_order: list[str] = Field(default_factory=list)   # bar x categories
    legend: LegendSpec = Field(default_factory=LegendSpec)
    show_error_band: bool = True
    error_type: Literal["std", "sem"] = "std"   # ±std vs ±standard error of the mean
    error_alpha: float = 0.15                    # opacity of the error band / bar caps
    show_grid: bool = True
    marker_size: float = 36.0
    figure: FigureSpec = Field(default_factory=FigureSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)


# --------------------------------------------------------------------------- #
# Global style preferences (consistent colors + order across every chart)
# --------------------------------------------------------------------------- #
class StylePrefs(BaseModel):
    # name (model_type or dataset) -> hex color
    colors: dict[str, str] = Field(default_factory=dict)
    # categorical field -> desired display order, e.g. {"model_type": [...], "dataset": [...]}
    order: dict[str, list[str]] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pivot-table builder (BI-style: rows × columns × values, with filters)
# --------------------------------------------------------------------------- #
Aggregation = Literal["mean", "std", "mean_std", "median", "min", "max", "count"]


class TableSpec(BaseModel):
    source: Source = "per_run"
    # filters
    datasets: list[str] = Field(default_factory=list)
    model_types: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    regimes: list[str] = Field(default_factory=list)
    sample_filter: list[int] = Field(default_factory=list)   # specific num_samples; empty = all
    exclude_mock: bool = True
    # pivot layout (BI-style)
    rows: list[str] = Field(default_factory=lambda: ["model_type"])  # index dimension(s)
    columns: list[str] = Field(default_factory=list)                 # optional column dimension(s)
    values: list[str] = Field(default_factory=lambda: ["Dice"])      # metric(s) or "__count__"
    aggregation: Aggregation = "mean_std"
    # formatting / ordering
    decimals: int = 3
    percentage: bool = False
    sort_by: Optional[str] = None     # column label to sort rows by (numeric)
    ascending: bool = False
    title: str = ""


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank significance
# --------------------------------------------------------------------------- #
class WilcoxonColors(BaseModel):
    significant: str = "#1a5276"
    nonsignificant: str = "#d6e4f0"
    nan: str = "#cccccc"


class WilcoxonSpec(BaseModel):
    reference_model: str
    comparison_models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metric: str = "Dice"
    # One panel per N value; 0 -> "Zero-shot", 1 -> "One-shot", else "N=k".
    n_values: list[int] = Field(default_factory=lambda: [0, 1])
    alpha: float = 0.05
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    min_common_images: int = 5
    row_axis: Literal["dataset", "comparison"] = "dataset"
    col_axis: Literal["comparison", "dataset"] = "comparison"
    annotate_pvalues: bool = False
    colors: WilcoxonColors = Field(default_factory=WilcoxonColors)
    title: TitleSpec = Field(default_factory=TitleSpec)
    fonts: dict[str, float] = Field(default_factory=lambda: {"base": 12.0})
    figure: FigureSpec = Field(default_factory=lambda: FigureSpec(size=[12.0, 4.5]))
    export: ExportSpec = Field(default_factory=ExportSpec)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
DashboardKind = Literal["figure", "wilcoxon", "table"]


class DashboardItem(BaseModel):
    id: Optional[str] = None
    dashboard_id: Optional[str] = None  # which dashboard this figure is pinned to
    title: str = "Untitled figure"
    kind: DashboardKind = "figure"
    regime: Optional[str] = None       # badge / grouping hint
    spec: dict                         # PlotSpec / WilcoxonSpec / TableSpec as raw dict
    created_at: Optional[str] = None


class DashboardItemUpdate(BaseModel):
    """Partial update for an already-pinned item (re-edit & save)."""
    title: Optional[str] = None
    regime: Optional[str] = None
    kind: Optional[DashboardKind] = None
    dashboard_id: Optional[str] = None
    spec: Optional[dict] = None


class DashboardCreate(BaseModel):
    name: str


class DashboardRename(BaseModel):
    name: str


class ReorderRequest(BaseModel):
    order: list[str]                   # list of ids in desired order


class DashboardExportRequest(BaseModel):
    out_dir: str
    format: ExportFormat = "png"
    dpi: int = 300
    dashboard_id: Optional[str] = None  # None -> export every dashboard's figures


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class TrainingRequest(BaseModel):
    stage: Literal["scratch", "finetune"]
    dataset: str                                   # vessmap | drive | dca1 | octa2d
    model_class: Literal["resnet18_unet", "resnet50_unet", "litemedsam"]
    pretraining: Literal["scratch", "imagenet", "vessshape"] = "scratch"
    weights_path: Optional[str] = None             # vessshape checkpoint (finetune)

    experiment_name: Optional[str] = None
    config_filename: Optional[str] = None          # default derived from selection

    # paths
    dataset_path: str
    csv_path: str

    # train hyperparameters
    lr: float = 1e-2
    bs_train: int = 8
    bs_valid: int = 20
    num_epochs: int = 500
    weight_decay: float = 0.0
    lr_decay: float = 1.0
    optimizer: str = "adam"
    momentum: float = 0.9                          # SGD momentum / Adam beta1
    validate_every: int = 5
    validation_metric: str = "Dice"
    maximize_validation_metric: bool = False
    patience: Optional[int] = None                 # early stopping; None disables
    loss_function: Optional[str] = None            # auto: bce for litemedsam else cross_entropy
    resize_size: str = "256 256"
    channels: Optional[str] = None                 # auto per-dataset; all | rgb | gray
    augmentation_strategy: Optional[str] = None
    dataset_params: Optional[str] = None           # "par1=v1 par2=v2"
    model_params: Optional[str] = None             # "par1=v1 par2=v2" (litemedsam auto if blank)
    num_workers: int = 12
    ignore_class_weights: bool = False

    # logging & weights-and-biases
    log_wandb: bool = False
    wandb_project: Optional[str] = None            # blank -> experiment_name
    wandb_group: Optional[str] = None              # auto-overwritten per run by launcher
    save_val_imgs: bool = False
    val_img_indices: str = "0 1 2"                 # only emitted when save_val_imgs
    disable_tqdm: bool = False
    meta: Optional[str] = None

    # checkpointing
    checkpoint_every: int = -1                     # 0 every epoch, N every N, -1 last only
    copy_model_every: int = 0                      # 0 = never
    suppress_checkpoint: bool = False
    suppress_best_checkpoint: bool = True

    # device & efficiency (shared by train + inference)
    device: str = "cuda:0"
    use_amp: bool = False                          # training automatic mixed precision
    deterministic: bool = False
    benchmark: bool = False

    # few-shot loop
    min_samples: int = 1
    max_samples: int = 20
    step: int = 2
    runs: int = 5
    reps: int = 3
    with_replacement: bool = False
    output_dir: str = "experiments"
    weights_id: Optional[str] = None

    # inference
    checkpoint_type: Literal["last", "best"] = "last"
    enable_inference: bool = True
    batch_inference: bool = True
    save_inference_images: bool = False
    delete_checkpoint: bool = True
    inference_dir_name: str = "inference_results"
    tta_type: Literal["none", "logits", "probs"] = "none"
    threshold: float = 0.5
    test_use_amp: bool = False                     # inference automatic mixed precision
    imagenet_normalize: bool = False
    skip_checkpoint_loading: bool = False
    force_headless: bool = True
    skip_boxplot: bool = True
    max_inference_retries: int = 1
    delete_only_on_success: bool = True
    aggregate_inference_means: bool = True


class SaveConfigRequest(BaseModel):
    request: TrainingRequest
    overwrite: bool = False
