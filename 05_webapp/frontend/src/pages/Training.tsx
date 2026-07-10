import { useCallback, useEffect, useState } from "react";
import { Activity, Play, Save, Settings, Square } from "lucide-react";
import { api } from "../lib/api";
import type { TrainingOptions, TrainingRequest, TrainingStatus } from "../lib/types";
import { useDebounced } from "../lib/util";
import {
  Badge,
  Button,
  Checkbox,
  Field,
  NumberInput,
  Section,
  Segmented,
  Select,
  TextInput,
} from "../components/ui";
import { MonitorPanel } from "../components/MonitorPanel";
import { PageHeader, InlineNote } from "../components/common";

const DS_DIR: Record<string, string> = {
  vessmap: "VessMAP",
  drive: "DRIVE",
  dca1: "DCA1",
  octa2d: "OCTA2D",
};
const DATA_ROOT = "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels";

function datasetPaths(ds: string) {
  const dir = `${DATA_ROOT}/${DS_DIR[ds] ?? ds}`;
  return { dataset_path: dir, csv_path: `${dir}/train.csv` };
}

function defaults(): TrainingRequest {
  return {
    stage: "scratch",
    dataset: "vessmap",
    model_class: "resnet18_unet",
    pretraining: "scratch",
    weights_path: "",
    experiment_name: "",
    config_filename: "",
    ...datasetPaths("vessmap"),
    lr: 0.01,
    bs_train: 8,
    bs_valid: 20,
    num_epochs: 500,
    weight_decay: 0,
    lr_decay: 1,
    optimizer: "adam",
    momentum: 0.9,
    validate_every: 5,
    validation_metric: "Dice",
    maximize_validation_metric: false,
    patience: null,
    loss_function: null,
    resize_size: "256 256",
    channels: null,
    augmentation_strategy: null,
    dataset_params: null,
    model_params: null,
    num_workers: 12,
    ignore_class_weights: false,
    log_wandb: false,
    wandb_project: "",
    wandb_group: "",
    save_val_imgs: false,
    val_img_indices: "0 1 2",
    disable_tqdm: false,
    meta: null,
    checkpoint_every: -1,
    copy_model_every: 0,
    suppress_checkpoint: false,
    suppress_best_checkpoint: true,
    device: "cuda:0",
    use_amp: false,
    deterministic: false,
    benchmark: false,
    min_samples: 1,
    max_samples: 20,
    step: 2,
    runs: 5,
    reps: 3,
    with_replacement: false,
    output_dir: "experiments",
    weights_id: null,
    checkpoint_type: "last",
    enable_inference: true,
    batch_inference: true,
    save_inference_images: false,
    delete_checkpoint: true,
    inference_dir_name: "inference_results",
    tta_type: "none",
    threshold: 0.5,
    test_use_amp: false,
    imagenet_normalize: false,
    skip_checkpoint_loading: false,
    force_headless: true,
    skip_boxplot: true,
    max_inference_retries: 1,
    delete_only_on_success: true,
    aggregate_inference_means: true,
  };
}

export default function Training() {
  const [req, setReq] = useState<TrainingRequest>(defaults());
  const [opts, setOpts] = useState<TrainingOptions | null>(null);
  const [yaml, setYaml] = useState("");
  const [filename, setFilename] = useState("");
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<"config" | "monitor">("config");
  const debounced = useDebounced(req, 300);

  useEffect(() => {
    api.trainingOptions().then(setOpts).catch(() => undefined);
  }, []);

  useEffect(() => {
    api
      .trainingConfig(debounced)
      .then((r) => {
        setYaml(r.yaml);
        setFilename(r.filename);
      })
      .catch((e) => setYaml(`# error: ${e.message ?? e}`));
  }, [debounced]);

  // poll job status
  useEffect(() => {
    const tick = () => api.trainingStatus().then(setStatus).catch(() => undefined);
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, []);

  const patch = useCallback((p: Partial<TrainingRequest>) => setReq((s) => ({ ...s, ...p })), []);

  const stage = opts?.stages.find((s) => s.value === req.stage);
  const models = stage?.models ?? ["resnet18_unet", "resnet50_unet"];
  const isLms = req.model_class === "litemedsam";
  const showPretraining = req.stage === "finetune" && !isLms;

  function onStageChange(v: "scratch" | "finetune") {
    const st = opts?.stages.find((s) => s.value === v);
    const model = st?.models.includes(req.model_class) ? req.model_class : "resnet18_unet";
    patch({
      stage: v,
      model_class: model as TrainingRequest["model_class"],
      pretraining: v === "scratch" ? "scratch" : "imagenet",
    });
  }

  function onDatasetChange(ds: string) {
    patch({ dataset: ds, ...datasetPaths(ds) });
  }

  async function save(overwrite: boolean) {
    setMsg(null);
    try {
      const r = await api.trainingSave(req, overwrite);
      if (r.saved) setMsg(`Saved config → ${r.path}`);
      else if (r.reason === "exists")
        setMsg(`Config already exists at ${r.path}. Use “Overwrite & Save”.`);
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    }
  }

  async function run() {
    const ok = window.confirm(
      `Launch training now?\n\nStage: ${req.stage}\nDataset: ${req.dataset}\nConfig: ${filename}\n\n` +
        "This writes the YAML (overwriting an existing one of the same name) and starts the " +
        "real serial launcher. Make sure parameters are correct.",
    );
    if (!ok) return;
    setMsg(null);
    try {
      const s = await api.trainingRun(req);
      setStatus(s);
      setMsg(`Launched: ${s.experiment_name}`);
    } catch (e) {
      setMsg(`Run failed: ${e}`);
    }
  }

  async function stop() {
    await api.trainingStop();
    setMsg("Stop signal sent.");
  }

  const running = status?.running;

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Training"
        subtitle="Configure & launch stages 02 (from scratch) and 03 (fine-tuning)."
        actions={
          running ? (
            <Badge tone="accent">running · {status?.dataset}</Badge>
          ) : (
            <Badge>idle</Badge>
          )
        }
      />

      <div className="grid flex-1 grid-cols-1 gap-0 overflow-hidden lg:grid-cols-[400px_1fr]">
        {/* form */}
        <div className="overflow-y-auto border-r border-border bg-surface px-5">
          <Section title="Experiment">
            <Field label="Stage">
              <Segmented
                value={req.stage}
                onChange={(v) => onStageChange(v as any)}
                options={[
                  { value: "scratch", label: "02 · Scratch" },
                  { value: "finetune", label: "03 · Fine-tune" },
                ]}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Dataset">
                <Select
                  value={req.dataset}
                  onChange={onDatasetChange}
                  options={(opts?.datasets ?? ["vessmap", "drive", "dca1", "octa2d"]).map((d) => ({ value: d, label: d }))}
                />
              </Field>
              <Field label="Model">
                <Select
                  value={req.model_class}
                  onChange={(v) => patch({ model_class: v as any })}
                  options={models.map((m) => ({ value: m, label: m }))}
                />
              </Field>
            </div>
            {showPretraining && (
              <Field label="Pretraining">
                <Segmented
                  value={req.pretraining}
                  onChange={(v) => patch({ pretraining: v as any })}
                  options={[
                    { value: "imagenet", label: "ImageNet" },
                    { value: "vessshape", label: "VessShape" },
                  ]}
                />
              </Field>
            )}
            {showPretraining && req.pretraining === "vessshape" && (
              <Field label="VessShape checkpoint path">
                <TextInput
                  value={req.weights_path ?? ""}
                  placeholder="/abs/path/checkpoint.pt"
                  onChange={(e) => patch({ weights_path: e.target.value })}
                />
              </Field>
            )}
            <Field label="Experiment name (blank = auto)">
              <TextInput value={req.experiment_name ?? ""} onChange={(e) => patch({ experiment_name: e.target.value })} />
            </Field>
          </Section>

          <Section title="Paths">
            <Field label="Dataset path">
              <TextInput value={req.dataset_path} onChange={(e) => patch({ dataset_path: e.target.value })} />
            </Field>
            <Field label="Train CSV path">
              <TextInput value={req.csv_path} onChange={(e) => patch({ csv_path: e.target.value })} />
            </Field>
          </Section>

          <Section title="Hyperparameters">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Learning rate">
                <NumberInput value={req.lr} step={0.0001} onChange={(v) => patch({ lr: v ?? 0.01 })} />
              </Field>
              <Field label="Epochs">
                <NumberInput value={req.num_epochs} onChange={(v) => patch({ num_epochs: v ?? 500 })} />
              </Field>
              <Field label="Batch (train)">
                <NumberInput value={req.bs_train} onChange={(v) => patch({ bs_train: v ?? 8 })} />
              </Field>
              <Field label="Batch (valid)">
                <NumberInput value={req.bs_valid} onChange={(v) => patch({ bs_valid: v ?? 20 })} />
              </Field>
              <Field label="Weight decay">
                <NumberInput value={req.weight_decay} step={0.0001} onChange={(v) => patch({ weight_decay: v ?? 0 })} />
              </Field>
              <Field label="LR decay">
                <NumberInput value={req.lr_decay} step={0.01} onChange={(v) => patch({ lr_decay: v ?? 1 })} />
              </Field>
              <Field label="Optimizer">
                <Select value={req.optimizer} onChange={(v) => patch({ optimizer: v })} options={(opts?.optimizers ?? ["adam"]).map((o) => ({ value: o, label: o }))} />
              </Field>
              <Field label="Momentum / β1">
                <NumberInput value={req.momentum} step={0.01} onChange={(v) => patch({ momentum: v ?? 0.9 })} />
              </Field>
              <Field label="Validate every">
                <NumberInput value={req.validate_every} onChange={(v) => patch({ validate_every: v ?? 5 })} />
              </Field>
              <Field label="Validation metric">
                <Select
                  value={req.validation_metric}
                  onChange={(v) => patch({ validation_metric: v })}
                  options={(opts?.validation_metrics ?? ["Dice"]).map((m) => ({ value: m, label: m }))}
                />
              </Field>
              <Field label="Patience (blank = off)">
                <NumberInput value={req.patience ?? null} onChange={(v) => patch({ patience: v })} />
              </Field>
              <Field label="Loss function">
                <Select
                  value={req.loss_function ?? "auto"}
                  onChange={(v) => patch({ loss_function: v === "auto" ? null : v })}
                  options={[{ value: "auto", label: "auto" }, ...(opts?.loss_functions ?? ["cross_entropy", "bce"]).map((o) => ({ value: o, label: o }))]}
                />
              </Field>
              <Field label="Channels">
                <Select
                  value={req.channels ?? "auto"}
                  onChange={(v) => patch({ channels: v === "auto" ? null : v })}
                  options={[{ value: "auto", label: "auto (per dataset)" }, ...(opts?.channels ?? ["all", "rgb", "gray"]).map((o) => ({ value: o, label: o }))]}
                />
              </Field>
              <Field label="Resize (H W)">
                <TextInput value={req.resize_size} onChange={(e) => patch({ resize_size: e.target.value })} />
              </Field>
              <Field label="Workers">
                <NumberInput value={req.num_workers} onChange={(v) => patch({ num_workers: v ?? 12 })} />
              </Field>
            </div>
            <Field label="Augmentation strategy (optional)">
              <TextInput value={req.augmentation_strategy ?? ""} placeholder="passed to dataset fn" onChange={(e) => patch({ augmentation_strategy: e.target.value || null })} />
            </Field>
            <Checkbox checked={req.maximize_validation_metric} onChange={(v) => patch({ maximize_validation_metric: v })} label="Maximize validation metric (early stopping)" />
            <Checkbox checked={req.ignore_class_weights} onChange={(v) => patch({ ignore_class_weights: v })} label="Ignore class weights" />
          </Section>

          <Section title="Logging & W&B">
            <Checkbox checked={req.log_wandb} onChange={(v) => patch({ log_wandb: v })} label="Log to Weights & Biases" />
            {req.log_wandb && (
              <>
                <Field label="W&B project" hint="blank → experiment name">
                  <TextInput value={req.wandb_project ?? ""} placeholder={req.experiment_name || "experiment name"} onChange={(e) => patch({ wandb_project: e.target.value })} />
                </Field>
                <Field label="W&B group" hint="set automatically per run (model | lr | weights_id | n_samples)">
                  <TextInput value={req.wandb_group ?? ""} placeholder="auto per run" disabled onChange={(e) => patch({ wandb_group: e.target.value })} />
                </Field>
              </>
            )}
            <Checkbox checked={req.save_val_imgs} onChange={(v) => patch({ save_val_imgs: v })} label="Save validation images" />
            {req.save_val_imgs && (
              <Field label="Validation image indices">
                <TextInput value={req.val_img_indices} placeholder="0 1 2" onChange={(e) => patch({ val_img_indices: e.target.value })} />
              </Field>
            )}
            <Field label="Meta (optional)" hint="extra text saved to config.json">
              <TextInput value={req.meta ?? ""} onChange={(e) => patch({ meta: e.target.value || null })} />
            </Field>
            <Checkbox checked={req.disable_tqdm} onChange={(v) => patch({ disable_tqdm: v })} label="Disable tqdm progress bar" />
          </Section>

          <Section title="Checkpointing" defaultOpen={false}>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Checkpoint every" hint="0 each · N every N · -1 last">
                <NumberInput value={req.checkpoint_every} onChange={(v) => patch({ checkpoint_every: v ?? -1 })} />
              </Field>
              <Field label="Copy model every" hint="0 = never">
                <NumberInput value={req.copy_model_every} onChange={(v) => patch({ copy_model_every: v ?? 0 })} />
              </Field>
            </div>
            <Checkbox checked={req.suppress_best_checkpoint} onChange={(v) => patch({ suppress_best_checkpoint: v })} label="Suppress best checkpoint" />
            <Checkbox checked={req.suppress_checkpoint} onChange={(v) => patch({ suppress_checkpoint: v })} label="Suppress all checkpointing" />
          </Section>

          <Section title="Device & efficiency" defaultOpen={false}>
            <Field label="Device">
              <TextInput value={req.device} placeholder="cuda:0" onChange={(e) => patch({ device: e.target.value })} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Dataset params (optional)">
                <TextInput value={req.dataset_params ?? ""} placeholder="p1=v1 p2=v2" onChange={(e) => patch({ dataset_params: e.target.value || null })} />
              </Field>
              <Field label="Model params (optional)">
                <TextInput value={req.model_params ?? ""} placeholder="p1=v1 p2=v2" onChange={(e) => patch({ model_params: e.target.value || null })} />
              </Field>
            </div>
            <Checkbox checked={req.use_amp} onChange={(v) => patch({ use_amp: v })} label="Automatic mixed precision (AMP)" />
            <Checkbox checked={req.deterministic} onChange={(v) => patch({ deterministic: v })} label="Deterministic algorithms" />
            <Checkbox checked={req.benchmark} onChange={(v) => patch({ benchmark: v })} label="cuDNN benchmark" />
          </Section>

          <Section title="Few-shot loop">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Min samples">
                <NumberInput value={req.min_samples} onChange={(v) => patch({ min_samples: v ?? 1 })} />
              </Field>
              <Field label="Max samples">
                <NumberInput value={req.max_samples} onChange={(v) => patch({ max_samples: v ?? 20 })} />
              </Field>
              <Field label="Step">
                <NumberInput value={req.step} onChange={(v) => patch({ step: v ?? 2 })} />
              </Field>
              <Field label="Runs (splits)">
                <NumberInput value={req.runs} onChange={(v) => patch({ runs: v ?? 5 })} />
              </Field>
              <Field label="Reps (seeds)">
                <NumberInput value={req.reps} onChange={(v) => patch({ reps: v ?? 3 })} />
              </Field>
              <Field label="Output dir">
                <TextInput value={req.output_dir} onChange={(e) => patch({ output_dir: e.target.value })} />
              </Field>
              <Field label="Weights id (blank = auto)">
                <TextInput value={req.weights_id ?? ""} placeholder="auto" onChange={(e) => patch({ weights_id: e.target.value || null })} />
              </Field>
            </div>
            <Checkbox checked={req.with_replacement} onChange={(v) => patch({ with_replacement: v })} label="Sample with replacement" />
          </Section>

          <Section title="Inference" defaultOpen={false}>
            <Field label="Checkpoint type">
              <Segmented
                value={req.checkpoint_type}
                onChange={(v) => patch({ checkpoint_type: v as any })}
                options={[
                  { value: "last", label: "Last" },
                  { value: "best", label: "Best" },
                ]}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Inference dir name">
                <TextInput value={req.inference_dir_name} onChange={(e) => patch({ inference_dir_name: e.target.value })} />
              </Field>
              <Field label="TTA type">
                <Select
                  value={req.tta_type}
                  onChange={(v) => patch({ tta_type: v as any })}
                  options={(opts?.tta_types ?? ["none", "logits", "probs"]).map((o) => ({ value: o, label: o }))}
                />
              </Field>
              <Field label="Threshold" hint="-1 = optimize on train">
                <NumberInput value={req.threshold} step={0.05} onChange={(v) => patch({ threshold: v ?? 0.5 })} />
              </Field>
              <Field label="Max inference retries">
                <NumberInput value={req.max_inference_retries} onChange={(v) => patch({ max_inference_retries: v ?? 1 })} />
              </Field>
            </div>
            <Checkbox checked={req.enable_inference} onChange={(v) => patch({ enable_inference: v })} label="Run inference after training" />
            <Checkbox checked={req.batch_inference} onChange={(v) => patch({ batch_inference: v })} label="Batch inference (defer to end)" />
            <Checkbox checked={req.save_inference_images} onChange={(v) => patch({ save_inference_images: v })} label="Save inference images" />
            <Checkbox checked={req.delete_checkpoint} onChange={(v) => patch({ delete_checkpoint: v })} label="Delete checkpoint after inference" />
            <Checkbox checked={req.delete_only_on_success} onChange={(v) => patch({ delete_only_on_success: v })} label="Delete only if inference succeeded" />
            <Checkbox checked={req.aggregate_inference_means} onChange={(v) => patch({ aggregate_inference_means: v })} label="Aggregate mean metrics to CSV" />
            <Checkbox checked={req.test_use_amp} onChange={(v) => patch({ test_use_amp: v })} label="Inference AMP" />
            <Checkbox checked={req.imagenet_normalize} onChange={(v) => patch({ imagenet_normalize: v })} label="ImageNet normalize inputs" />
            <Checkbox checked={req.skip_checkpoint_loading} onChange={(v) => patch({ skip_checkpoint_loading: v })} label="Skip checkpoint loading (random/encoder weights)" />
            <Checkbox checked={req.force_headless} onChange={(v) => patch({ force_headless: v })} label="Headless matplotlib backend" />
            <Checkbox checked={req.skip_boxplot} onChange={(v) => patch({ skip_boxplot: v })} label="Skip boxplot generation" />
          </Section>
        </div>

        {/* right panel */}
        <div className="flex flex-col overflow-hidden">
          {/* Tab bar + action buttons */}
          <div className="flex items-center justify-between border-b border-border px-5 py-2 gap-2">
            {/* Tabs */}
            <div className="inline-flex rounded-lg border border-border bg-surface-2 p-0.5">
              <button
                onClick={() => setTab("config")}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer ${
                  tab === "config"
                    ? "bg-primary text-primary-fg"
                    : "text-muted-fg hover:text-fg"
                }`}
              >
                <Settings className="h-3.5 w-3.5" /> Config
              </button>
              <button
                onClick={() => setTab("monitor")}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer ${
                  tab === "monitor"
                    ? "bg-primary text-primary-fg"
                    : "text-muted-fg hover:text-fg"
                }`}
              >
                <Activity className="h-3.5 w-3.5" /> Monitor
              </button>
            </div>
            {/* Action buttons */}
            <div className="flex items-center gap-2">
              <Button variant="default" onClick={() => save(false)}>
                <Save className="h-4 w-4" /> Save config
              </Button>
              <Button variant="default" onClick={() => save(true)} title="Overwrite if it exists">
                Overwrite & Save
              </Button>
              {running ? (
                <Button variant="danger" onClick={stop}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button variant="primary" onClick={run}>
                  <Play className="h-4 w-4" /> Save & Run
                </Button>
              )}
            </div>
          </div>

          {msg && <div className="px-5 pt-2"><InlineNote tone="success">{msg}</InlineNote></div>}

          {/* Config tab: YAML preview */}
          {tab === "config" && (
            <div className="flex flex-col gap-2 overflow-hidden p-5 flex-1">
              <div className="flex items-center gap-2 text-sm text-muted-fg">
                Generated config:
                <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-fg">{filename}</span>
              </div>
              <pre className="flex-1 overflow-auto rounded-lg border border-border bg-surface-2 p-3 font-mono text-[12px] leading-relaxed text-fg">
                {yaml}
              </pre>
            </div>
          )}

          {/* Monitor tab */}
          {tab === "monitor" && (
            <div className="flex-1 overflow-hidden">
              <MonitorPanel active={tab === "monitor"} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
