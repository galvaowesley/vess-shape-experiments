import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Download, Pin } from "lucide-react";
import { api } from "../lib/api";
import type { Metadata, WilcoxonSpec } from "../lib/types";
import { defaultWilcoxonSpec, hydrateWilcoxonSpec } from "../lib/plotSpec";
import { useDebounced, useObjectUrl } from "../lib/util";
import {
  Button,
  Checkbox,
  ChipMultiSelect,
  Field,
  NumberInput,
  Section,
  Segmented,
  Select,
  Slider,
  TextInput,
} from "../components/ui";
import { ColorWheel } from "../components/ColorWheel";
import { PinDialog, type PinResult } from "../components/PinDialog";
import { PageHeader, PreviewPane, InlineNote } from "../components/common";

const ALL_DATASETS = ["dca1", "drive", "octa2d", "vessmap"];

export default function Significance() {
  const [spec, setSpec] = useState<WilcoxonSpec>(defaultWilcoxonSpec());
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [pinOpen, setPinOpen] = useState(false);
  const [editItemId, setEditItemId] = useState<string | null>(null);

  const location = useLocation();
  const navigate = useNavigate();

  useObjectUrl(url);
  const debounced = useDebounced(spec, 300);

  useEffect(() => {
    api.metadata("per_run", ALL_DATASETS).then(setMeta).catch(() => undefined);
  }, []);

  // Re-open a pinned Wilcoxon figure for editing (navigated from the Dashboard).
  useEffect(() => {
    const st = location.state as { editSpec?: Record<string, unknown>; editItemId?: string } | null;
    if (st?.editSpec) {
      setSpec(hydrateWilcoxonSpec(st.editSpec));
      setEditItemId(st.editItemId ?? null);
      navigate(location.pathname, { replace: true, state: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .renderWilcoxon(debounced)
      .then((u) => !cancelled && setUrl(u))
      .catch((e) => !cancelled && setError(String(e.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  const patch = useCallback((p: Partial<WilcoxonSpec>) => setSpec((s) => ({ ...s, ...p })), []);

  const allModels = (meta?.options.model_type?.values ?? [])
    .map(String)
    .filter((m) => !m.startsWith("Zero-Shot"));
  const nOptions = (meta?.options.num_samples?.values ?? [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).map(Number);

  async function exportWilcoxon() {
    setMsg(null);
    const res = await fetch("/api/stats/wilcoxon/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const j = await res.json();
      setMsg(j.saved ? `Saved to ${j.path}` : `Error: ${j.detail ?? "unknown"}`);
    } else {
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `wilcoxon.${spec.export.format}`;
      a.click();
      URL.revokeObjectURL(a.href);
    }
  }

  const pinTitle = spec.title.text || `Wilcoxon · ${spec.reference_model} · ${spec.metric}`;

  async function handlePin({ dashboardId, title, mode }: PinResult) {
    const payload = {
      title,
      kind: "wilcoxon" as const,
      regime: spec.n_values.includes(0) ? "Zero-shot" : null,
      dashboard_id: dashboardId,
      spec: spec as unknown as Record<string, unknown>,
    };
    if (mode === "update" && editItemId) {
      await api.updatePin(editItemId, payload);
      setMsg("Updated pinned figure.");
    } else {
      const created = await api.pin(payload);
      setEditItemId(created.id ?? null);
      setMsg("Pinned to dashboard.");
    }
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Wilcoxon Signed-Rank"
        subtitle="Paired per-image significance — choose reference, comparisons, datasets and N per panel."
      />
      <div className="flex flex-1 overflow-hidden">
        <div className="w-[380px] shrink-0 overflow-y-auto border-r border-border bg-surface px-5">
          <Section title="Comparison">
            <Field label="Reference model">
              <Select
                value={spec.reference_model}
                onChange={(v) => patch({ reference_model: v })}
                options={allModels.map((m) => ({ value: m, label: m }))}
              />
            </Field>
            <Field label="Comparison models" hint="Reference is tested against each of these.">
              <ChipMultiSelect
                options={allModels.filter((m) => m !== spec.reference_model)}
                selected={spec.comparison_models}
                onChange={(v) => patch({ comparison_models: v })}
                emptyMeansAll={false}
              />
            </Field>
            <Field label="Datasets">
              <ChipMultiSelect
                options={ALL_DATASETS}
                selected={spec.datasets}
                onChange={(v) => patch({ datasets: v.length ? v : ALL_DATASETS })}
                emptyMeansAll={false}
              />
            </Field>
            <Field label="Metric">
              <Select
                value={spec.metric}
                onChange={(v) => patch({ metric: v })}
                options={(meta?.metrics ?? ["Dice"]).map((m) => ({ value: m, label: m }))}
              />
            </Field>
            <Field label="Panels by N (0=Zero-shot, 1=One-shot)">
              <ChipMultiSelect
                options={nOptions.map(String)}
                selected={spec.n_values.map(String)}
                onChange={(v) => patch({ n_values: v.map(Number).sort((a, b) => a - b) })}
                emptyMeansAll={false}
              />
            </Field>
          </Section>

          <Section title="Test parameters">
            <Field label={`Alpha = ${spec.alpha}`}>
              <Slider value={spec.alpha} onChange={(v) => patch({ alpha: v })} min={0.001} max={0.1} step={0.001} />
            </Field>
            <Field label="Alternative hypothesis">
              <Select
                value={spec.alternative}
                onChange={(v) => patch({ alternative: v as any })}
                options={[
                  { value: "greater", label: "reference > comparison" },
                  { value: "less", label: "reference < comparison" },
                  { value: "two-sided", label: "two-sided" },
                ]}
              />
            </Field>
            <Field label="Min common images">
              <NumberInput value={spec.min_common_images} onChange={(v) => patch({ min_common_images: v ?? 5 })} />
            </Field>
            <Checkbox checked={spec.annotate_pvalues} onChange={(v) => patch({ annotate_pvalues: v })} label="Annotate p-values in cells" />
          </Section>

          <Section title="Layout & colors">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Rows">
                <Segmented
                  value={spec.row_axis}
                  onChange={(v) => patch({ row_axis: v, col_axis: v === "dataset" ? "comparison" : "dataset" })}
                  options={[
                    { value: "dataset", label: "Datasets" },
                    { value: "comparison", label: "Models" },
                  ]}
                />
              </Field>
              <Field label="Columns">
                <span className="block px-1 py-2 text-xs text-muted-fg">
                  {spec.col_axis === "comparison" ? "Comparison models" : "Datasets"}
                </span>
              </Field>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-xs text-muted-fg">
                <ColorWheel color={spec.colors.significant} onChange={(c) => patch({ colors: { ...spec.colors, significant: c } })} /> Significant
              </label>
              <label className="flex items-center gap-2 text-xs text-muted-fg">
                <ColorWheel color={spec.colors.nonsignificant} onChange={(c) => patch({ colors: { ...spec.colors, nonsignificant: c } })} /> n.s.
              </label>
              <label className="flex items-center gap-2 text-xs text-muted-fg">
                <ColorWheel color={spec.colors.nan} onChange={(c) => patch({ colors: { ...spec.colors, nan: c } })} /> N/A
              </label>
            </div>
          </Section>

          <Section title="Title & figure" defaultOpen={false}>
            <Field label="Title (blank = auto)">
              <TextInput value={spec.title.text} onChange={(e) => setSpec((s) => ({ ...s, title: { ...s.title, text: e.target.value } }))} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Base font">
                <NumberInput value={spec.fonts.base} onChange={(v) => setSpec((s) => ({ ...s, fonts: { ...s.fonts, base: v ?? 12 } }))} />
              </Field>
              <Field label="Title size">
                <NumberInput value={spec.title.fontsize} onChange={(v) => setSpec((s) => ({ ...s, title: { ...s.title, fontsize: v ?? 15 } }))} />
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Width (in)">
                <NumberInput value={spec.figure.size[0]} step={0.5} onChange={(v) => setSpec((s) => ({ ...s, figure: { size: [v ?? 12, s.figure.size[1]] } }))} />
              </Field>
              <Field label="Height (in)">
                <NumberInput value={spec.figure.size[1]} step={0.5} onChange={(v) => setSpec((s) => ({ ...s, figure: { size: [s.figure.size[0], v ?? 4.5] } }))} />
              </Field>
            </div>
          </Section>
        </div>

        <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-6">
          <PreviewPane url={url} loading={loading} error={error} />
          <div className="flex flex-wrap items-end justify-between gap-3 rounded-lg border border-border bg-surface p-4">
            <div className="flex flex-wrap items-end gap-3">
              <Field label="Format" className="w-28">
                <Select
                  value={spec.export.format}
                  onChange={(v) => setSpec((s) => ({ ...s, export: { ...s.export, format: v as any } }))}
                  options={["svg", "png", "jpg", "pdf"].map((f) => ({ value: f, label: f.toUpperCase() }))}
                />
              </Field>
              <Field label="DPI" className="w-24">
                <NumberInput value={spec.export.dpi} onChange={(v) => setSpec((s) => ({ ...s, export: { ...s.export, dpi: v ?? 300 } }))} />
              </Field>
              <Field label="Save path (blank = download)" className="w-72">
                <TextInput
                  value={spec.export.save_path ?? ""}
                  placeholder="/abs/path/wilcoxon.svg"
                  onChange={(e) => setSpec((s) => ({ ...s, export: { ...s.export, save_path: e.target.value || null } }))}
                />
              </Field>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="default" onClick={() => setPinOpen(true)}>
                <Pin className="h-4 w-4" /> {editItemId ? "Save / pin" : "Pin to dashboard"}
              </Button>
              <Button variant="primary" onClick={exportWilcoxon}>
                <Download className="h-4 w-4" /> Export
              </Button>
            </div>
          </div>
          {editItemId && (
            <InlineNote>Editing a pinned figure — “Save / pin” can update it in place.</InlineNote>
          )}
          {msg && <InlineNote tone="success">{msg}</InlineNote>}
        </div>
      </div>

      <PinDialog
        open={pinOpen}
        defaultTitle={pinTitle}
        editItemId={editItemId}
        onConfirm={handlePin}
        onClose={() => setPinOpen(false)}
      />
    </div>
  );
}
