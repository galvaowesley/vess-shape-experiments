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
    palette: Optional[str] = None                     # seaborn named palette
    x_axis: AxisSpec = Field(default_factory=AxisSpec)
    y_axis: AxisSpec = Field(default_factory=AxisSpec)
    title: TitleSpec = Field(default_factory=TitleSpec)
    fonts: dict[str, float] = Field(default_factory=lambda: {"base": 12.0})
    series_styles: dict[str, SeriesStyle] = Field(default_factory=dict)
    legend: LegendSpec = Field(default_factory=LegendSpec)
    show_error_band: bool = True
    show_grid: bool = True
    marker_size: float = 36.0
    figure: FigureSpec = Field(default_factory=FigureSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)


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
class DashboardItem(BaseModel):
    id: Optional[str] = None
    title: str = "Untitled figure"
    kind: Literal["figure", "wilcoxon"] = "figure"
    regime: Optional[str] = None       # badge / grouping hint
    spec: dict                         # PlotSpec or WilcoxonSpec as raw dict
    created_at: Optional[str] = None


class ReorderRequest(BaseModel):
    order: list[str]                   # list of ids in desired order


class DashboardExportRequest(BaseModel):
    out_dir: str
    format: ExportFormat = "png"
    dpi: int = 300


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
    validate_every: int = 5
    validation_metric: str = "Dice"
    loss_function: Optional[str] = None            # auto: bce for litemedsam else cross_entropy
    resize_size: str = "256 256"
    channels: Optional[str] = None
    num_workers: int = 12
    ignore_class_weights: bool = False

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


class SaveConfigRequest(BaseModel):
    request: TrainingRequest
    overwrite: bool = False
