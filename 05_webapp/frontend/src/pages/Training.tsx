import { useCallback, useEffect, useState } from "react";
import { Play, Save, Square } from "lucide-react";
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
import { LogConsole } from "../components/LogConsole";
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
    validate_every: 5,
    validation_metric: "Dice",
    loss_function: null,
    resize_size: "256 256",
    channels: null,
    num_workers: 12,
    ignore_class_weights: false,
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
  };
}

export default function Training() {
  const [req, setReq] = useState<TrainingRequest>(defaults());
  const [opts, setOpts] = useState<TrainingOptions | null>(null);
  const [yaml, setYaml] = useState("");
  const [filename, setFilename] = useState("");
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
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
              <Field label="Validate every">
                <NumberInput value={req.validate_every} onChange={(v) => patch({ validate_every: v ?? 5 })} />
              </Field>
              <Field label="Resize (H W)">
                <TextInput value={req.resize_size} onChange={(e) => patch({ resize_size: e.target.value })} />
              </Field>
              <Field label="Workers">
                <NumberInput value={req.num_workers} onChange={(v) => patch({ num_workers: v ?? 12 })} />
              </Field>
            </div>
            <Checkbox checked={req.ignore_class_weights} onChange={(v) => patch({ ignore_class_weights: v })} label="Ignore class weights" />
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
            <Checkbox checked={req.enable_inference} onChange={(v) => patch({ enable_inference: v })} label="Run inference after training" />
            <Checkbox checked={req.batch_inference} onChange={(v) => patch({ batch_inference: v })} label="Batch inference (defer to end)" />
            <Checkbox checked={req.save_inference_images} onChange={(v) => patch({ save_inference_images: v })} label="Save inference images" />
            <Checkbox checked={req.delete_checkpoint} onChange={(v) => patch({ delete_checkpoint: v })} label="Delete checkpoint after inference" />
          </Section>
        </div>

        {/* yaml preview + console */}
        <div className="flex flex-col gap-3 overflow-hidden p-5">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm text-muted-fg">
              Generated config:
              <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-fg">{filename}</span>
            </div>
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

          {msg && <InlineNote tone="success">{msg}</InlineNote>}

          <div className="grid flex-1 grid-rows-2 gap-3 overflow-hidden">
            <pre className="overflow-auto rounded-lg border border-border bg-surface-2 p-3 font-mono text-[12px] leading-relaxed text-fg">
              {yaml}
            </pre>
            <LogConsole />
          </div>
        </div>
      </div>
    </div>
  );
}
