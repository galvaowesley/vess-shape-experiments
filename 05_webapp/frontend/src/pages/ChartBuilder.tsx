import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Download, Pin, RefreshCw, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import type { AxisSpec, Metadata, PaletteCatalog, PlotSpec } from "../lib/types";
import { defaultPlotSpec, hydratePlotSpec, LEGEND_LOCS, LINESTYLES, MARKERS, PRESETS } from "../lib/plotSpec";
import { useDebounced, useObjectUrl } from "../lib/util";
import {
  Badge,
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
import { PalettePicker } from "../components/PalettePicker";
import { PinDialog, type PinResult } from "../components/PinDialog";
import { PathField } from "../components/DirectoryPicker";
import { PageHeader, PreviewPane, InlineNote } from "../components/common";

const ALL_DATASETS = ["vessmap", "drive", "dca1", "octa2d"];
const SOURCES = [
  { value: "per_run", label: "Per-run (mean per run)" },
  { value: "per_image", label: "Per-image" },
  { value: "summary", label: "Summary (mean ± std)" },
];

export default function ChartBuilder() {
  const [spec, setSpec] = useState<PlotSpec>(defaultPlotSpec());
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [catalog, setCatalog] = useState<PaletteCatalog | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rescanning, setRescanning] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [pinOpen, setPinOpen] = useState(false);
  const [editItemId, setEditItemId] = useState<string | null>(null);

  const location = useLocation();
  const navigate = useNavigate();

  useObjectUrl(url);
  const debounced = useDebounced(spec, 250);

  // Re-open a pinned figure for editing (navigated from the Dashboard).
  useEffect(() => {
    const st = location.state as { editSpec?: Record<string, unknown>; editItemId?: string } | null;
    if (st?.editSpec) {
      setSpec(hydratePlotSpec(st.editSpec));
      setEditItemId(st.editItemId ?? null);
      navigate(location.pathname, { replace: true, state: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // palettes once
  useEffect(() => {
    api.palettes().then(setCatalog).catch(() => undefined);
  }, []);

  // metadata when source / datasets change
  useEffect(() => {
    api
      .metadata(spec.data.source, spec.data.datasets)
      .then(setMeta)
      .catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.data.source, spec.data.datasets.join(",")]);

  // live render
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .renderFigure(debounced)
      .then((u) => !cancelled && setUrl(u))
      .catch((e) => !cancelled && setError(String(e.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  /* ---- immutable update helpers ---- */
  const patch = useCallback((p: Partial<PlotSpec>) => setSpec((s) => ({ ...s, ...p })), []);
  const patchData = (p: Partial<PlotSpec["data"]>) =>
    setSpec((s) => ({ ...s, data: { ...s.data, ...p } }));
  const patchEnc = (p: Partial<PlotSpec["encoding"]>) =>
    setSpec((s) => ({ ...s, encoding: { ...s.encoding, ...p } }));
  const patchAxis = (axis: "x_axis" | "y_axis", p: Partial<AxisSpec>) =>
    setSpec((s) => ({ ...s, [axis]: { ...s[axis], ...p } }));
  const patchSeriesStyle = (name: string, p: Partial<PlotSpec["series_styles"][string]>) =>
    setSpec((s) => ({
      ...s,
      series_styles: { ...s.series_styles, [name]: { ...s.series_styles[name], ...p } },
    }));

  const opt = (field: string): string[] =>
    (meta?.options[field]?.values ?? []).map((v) => String(v));

  const seriesNames = useMemo(() => {
    const sf = spec.encoding.series;
    if (!sf || !meta?.options[sf]) return [] as string[];
    const all = opt(sf);
    if (sf === "model_type" && spec.data.model_types.length) return spec.data.model_types;
    return all;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.encoding.series, spec.data.model_types, meta]);

  const xFields = (meta?.x_fields ?? ["num_samples"]).map((f) => ({ value: f, label: f }));
  const yFields = (meta?.y_fields ?? ["Dice"]).map((f) => ({ value: f, label: f }));
  const seriesFields = [
    { value: "", label: "None" },
    ...(meta?.series_fields ?? ["model_type"]).map((f) => ({ value: f, label: f })),
  ];
  const facetFields = [
    { value: "", label: "None" },
    ...(meta?.facet_fields ?? ["regime"]).map((f) => ({ value: f, label: f })),
  ];

  // sample-size (N) options for bar charts — N=0 surfaces the zero-shot models
  const barNOptions = [
    { value: "all", label: "All N (pooled)" },
    ...opt("num_samples").map((n) => ({
      value: n,
      label: n === "0" ? "0 — Zero-shot" : n === "1" ? "1 — One-shot" : `N = ${n}`,
    })),
  ];
  const currentBarN = (() => {
    const r = spec.data.num_samples_range;
    return r && r[0] === r[1] ? String(r[0]) : "all";
  })();
  const setBarN = (v: string) =>
    patchData({ num_samples_range: v === "all" ? null : [Number(v), Number(v)] });

  /* ---- actions ---- */
  async function doExport() {
    setMsg(null);
    const res = await api.exportFigure(spec);
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const j = await res.json();
      setMsg(j.saved ? `Saved to ${j.path}` : `Error: ${j.detail ?? "unknown"}`);
    } else {
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `figure.${spec.export.format}`;
      a.click();
      URL.revokeObjectURL(a.href);
    }
  }

  const pinTitle = spec.title.text || `${spec.chart_type} · ${spec.encoding.y}`;

  async function handlePin({ dashboardId, title, mode }: PinResult) {
    const regime =
      spec.facet === "regime"
        ? "faceted"
        : spec.data.regimes.length === 1
          ? spec.data.regimes[0]
          : null;
    const payload = {
      title,
      kind: "figure" as const,
      regime,
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

  async function doRescan() {
    setRescanning(true);
    setMsg(null);
    try {
      await api.rescan(undefined, true);
      const m = await api.metadata(spec.data.source, spec.data.datasets);
      setMeta(m);
      setSpec((s) => ({ ...s })); // re-trigger render
      setMsg("Experiments rescanned — data refreshed.");
    } catch (e) {
      setMsg(`Rescan failed: ${e}`);
    } finally {
      setRescanning(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Chart Builder"
        subtitle="Build publication-quality figures — live preview equals the exported file."
        actions={
          <Button variant="ghost" size="sm" onClick={doRescan} disabled={rescanning}>
            <RefreshCw className={rescanning ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
            Rescan experiments
          </Button>
        }
      />

      <div className="flex flex-1 overflow-hidden">
        {/* control panel */}
        <div className="w-[380px] shrink-0 overflow-y-auto border-r border-border bg-surface px-5">
          {/* presets */}
          <Section title="Presets" defaultOpen>
            <div className="grid grid-cols-1 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => setSpec(p.build())}
                  className="flex items-start gap-2 rounded-lg border border-border bg-surface-2 px-3 py-2 text-left hover:border-primary cursor-pointer"
                >
                  <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
                  <span>
                    <span className="block text-sm font-medium text-fg">{p.name}</span>
                    <span className="block text-[11px] text-muted-fg">{p.description}</span>
                  </span>
                </button>
              ))}
            </div>
          </Section>

          {/* data */}
          <Section title="Data">
            <Field label="Chart type">
              <Segmented
                value={spec.chart_type}
                onChange={(v) => patch({ chart_type: v })}
                options={[
                  { value: "line", label: "Line" },
                  { value: "bar", label: "Bar" },
                  { value: "scatter", label: "Scatter" },
                ]}
              />
            </Field>
            <Field label="Source">
              <Select value={spec.data.source} onChange={(v) => patchData({ source: v as any })} options={SOURCES} />
            </Field>
            <Field label="Datasets">
              <ChipMultiSelect
                options={ALL_DATASETS}
                selected={spec.data.datasets}
                onChange={(v) => patchData({ datasets: v.length ? v : ALL_DATASETS })}
                emptyMeansAll={false}
              />
            </Field>
            <Field label="Models / methods (empty = all)">
              <ChipMultiSelect
                options={opt("model_type")}
                selected={spec.data.model_types}
                onChange={(v) => patchData({ model_types: v })}
              />
            </Field>
            <Field label="Stage">
              <ChipMultiSelect
                options={opt("stage")}
                selected={spec.data.stages}
                onChange={(v) => patchData({ stages: v })}
              />
            </Field>
          </Section>

          {/* regime + encoding */}
          <Section title="Regime & axes">
            <Field label="Shot regime (segmentation)">
              <ChipMultiSelect
                options={meta?.regimes ?? ["Zero-shot", "One-shot", "Few-shot"]}
                selected={spec.data.regimes}
                onChange={(v) => patchData({ regimes: v })}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="X axis data">
                <Select value={spec.encoding.x} onChange={(v) => patchEnc({ x: v })} options={xFields} />
              </Field>
              <Field label="Y axis data">
                <Select value={spec.encoding.y} onChange={(v) => patchEnc({ y: v })} options={yFields} />
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Series (color)">
                <Select
                  value={spec.encoding.series ?? ""}
                  onChange={(v) => patchEnc({ series: v || null })}
                  options={seriesFields}
                />
              </Field>
              <Field label="Facet (small multiples)">
                <Select
                  value={spec.facet ?? ""}
                  onChange={(v) => patch({ facet: v || null })}
                  options={facetFields}
                />
              </Field>
            </div>
            <Field label="Aggregate across datasets">
              <Segmented
                value={spec.aggregate_datasets}
                onChange={(v) => patch({ aggregate_datasets: v })}
                options={[
                  { value: "none", label: "Off" },
                  { value: "macro", label: "Mean of datasets" },
                  { value: "pool", label: "Pooled" },
                ]}
              />
            </Field>
            {spec.chart_type === "bar" ? (
              <Field label="Sample size (N)" hint="Pick the N to compare across datasets. N=0 = zero-shot models.">
                <Select value={currentBarN} onChange={setBarN} options={barNOptions} />
              </Field>
            ) : (
              <Field label="Samples range (min / max)">
                <div className="flex items-center gap-2">
                  <NumberInput
                    value={spec.data.num_samples_range?.[0] ?? null}
                    onChange={(v) =>
                      patchData({
                        num_samples_range:
                          v == null && spec.data.num_samples_range?.[1] == null
                            ? null
                            : [v ?? 0, spec.data.num_samples_range?.[1] ?? 20],
                      })
                    }
                  />
                  <span className="text-muted-fg">–</span>
                  <NumberInput
                    value={spec.data.num_samples_range?.[1] ?? null}
                    onChange={(v) =>
                      patchData({
                        num_samples_range:
                          v == null && spec.data.num_samples_range?.[0] == null
                            ? null
                            : [spec.data.num_samples_range?.[0] ?? 0, v ?? 20],
                      })
                    }
                  />
                </div>
              </Field>
            )}
          </Section>

          {/* X axis */}
          <AxisControls
            title="X axis"
            axis={spec.x_axis}
            fallbackLabel={spec.encoding.x}
            onChange={(p) => patchAxis("x_axis", p)}
            showRotation
          />
          {/* Y axis */}
          <AxisControls
            title="Y axis"
            axis={spec.y_axis}
            fallbackLabel={spec.encoding.y}
            onChange={(p) => patchAxis("y_axis", p)}
          />

          {/* title & fonts */}
          <Section title="Title & fonts">
            <Field label="Chart title">
              <TextInput
                value={spec.title.text}
                placeholder="(none)"
                onChange={(e) => setSpec((s) => ({ ...s, title: { ...s.title, text: e.target.value } }))}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Title size">
                <NumberInput
                  value={spec.title.fontsize}
                  onChange={(v) => setSpec((s) => ({ ...s, title: { ...s.title, fontsize: v ?? 16 } }))}
                />
              </Field>
              <Field label="Global base font">
                <NumberInput
                  value={spec.fonts.base}
                  onChange={(v) => setSpec((s) => ({ ...s, fonts: { ...s.fonts, base: v ?? 12 } }))}
                />
              </Field>
            </div>
          </Section>

          {/* colors & series styles */}
          <Section title="Colors & series">
            <Field label="Palette (seaborn)">
              <PalettePicker catalog={catalog} value={spec.palette} onChange={(v) => patch({ palette: v })} />
            </Field>
            {spec.chart_type !== "scatter" && (
              <>
                <Checkbox
                  checked={spec.show_error_band}
                  onChange={(v) => patch({ show_error_band: v })}
                  label={spec.chart_type === "bar" ? "Show error bars" : "Show ± error band"}
                />
                {spec.show_error_band && (
                  <>
                    <Field label="Error measure">
                      <Segmented
                        value={spec.error_type}
                        onChange={(v) => patch({ error_type: v })}
                        options={[
                          { value: "std", label: "Std (±σ)" },
                          { value: "sem", label: "Std error (±SEM)" },
                        ]}
                      />
                    </Field>
                    {spec.chart_type === "line" && (
                      <Field label={`Band opacity (${spec.error_alpha.toFixed(2)})`}>
                        <Slider
                          value={spec.error_alpha}
                          onChange={(v) => patch({ error_alpha: v })}
                          min={0}
                          max={1}
                          step={0.05}
                        />
                      </Field>
                    )}
                  </>
                )}
              </>
            )}
            <Checkbox checked={spec.show_grid} onChange={(v) => patch({ show_grid: v })} label="Show grid" />
            <Field label="Marker size">
              <Slider value={spec.marker_size} onChange={(v) => patch({ marker_size: v })} min={0} max={120} />
            </Field>
            {seriesNames.length > 0 && (
              <div className="space-y-2">
                <span className="text-[11px] font-medium text-muted-fg">Per-series overrides</span>
                {seriesNames.map((name) => {
                  const st = spec.series_styles[name] ?? {};
                  return (
                    <div key={name} className="flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-2 py-1.5">
                      <ColorWheel color={st.color} onChange={(hex) => patchSeriesStyle(name, { color: hex })} />
                      <span className="flex-1 truncate text-xs text-fg" title={name}>
                        {name}
                      </span>
                      {spec.chart_type === "line" && (
                        <select
                          value={st.linestyle ?? "-"}
                          onChange={(e) => patchSeriesStyle(name, { linestyle: e.target.value })}
                          className="rounded border border-border bg-surface px-1 py-0.5 text-[11px] text-fg"
                        >
                          {LINESTYLES.map((l) => (
                            <option key={l.value} value={l.value}>
                              {l.label}
                            </option>
                          ))}
                        </select>
                      )}
                      {spec.chart_type !== "bar" && (
                        <select
                          value={st.marker ?? "o"}
                          onChange={(e) => patchSeriesStyle(name, { marker: e.target.value })}
                          className="rounded border border-border bg-surface px-1 py-0.5 text-[11px] text-fg"
                        >
                          {MARKERS.map((m) => (
                            <option key={m.value} value={m.value}>
                              {m.label}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </Section>

          {/* legend & figure */}
          <Section title="Legend & figure" defaultOpen={false}>
            <Checkbox checked={spec.legend.show} onChange={(v) => setSpec((s) => ({ ...s, legend: { ...s.legend, show: v } }))} label="Show legend" />
            <div className="grid grid-cols-2 gap-2">
              <Field label="Legend title">
                <TextInput
                  value={spec.legend.title ?? ""}
                  onChange={(e) => setSpec((s) => ({ ...s, legend: { ...s.legend, title: e.target.value || null } }))}
                />
              </Field>
              <Field label="Legend position">
                <Select
                  value={spec.legend.loc}
                  onChange={(v) => setSpec((s) => ({ ...s, legend: { ...s.legend, loc: v } }))}
                  options={LEGEND_LOCS.map((l) => ({ value: l, label: l }))}
                />
              </Field>
            </div>
            <Field label="Legend font size">
              <NumberInput value={spec.legend.fontsize} onChange={(v) => setSpec((s) => ({ ...s, legend: { ...s.legend, fontsize: v ?? 11 } }))} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Figure width (in)">
                <NumberInput value={spec.figure.size[0]} step={0.5} onChange={(v) => setSpec((s) => ({ ...s, figure: { size: [v ?? 9, s.figure.size[1]] } }))} />
              </Field>
              <Field label="Figure height (in)">
                <NumberInput value={spec.figure.size[1]} step={0.5} onChange={(v) => setSpec((s) => ({ ...s, figure: { size: [s.figure.size[0], v ?? 5.5] } }))} />
              </Field>
            </div>
          </Section>
        </div>

        {/* preview + export */}
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
                <PathField
                  value={spec.export.save_path ?? ""}
                  onChange={(v) => setSpec((s) => ({ ...s, export: { ...s.export, save_path: v || null } }))}
                  mode="file"
                  placeholder="/abs/path/figure.svg"
                  defaultFilename={`figure.${spec.export.format}`}
                />
              </Field>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="default" onClick={() => setPinOpen(true)}>
                <Pin className="h-4 w-4" /> {editItemId ? "Save / pin" : "Pin to dashboard"}
              </Button>
              <Button variant="primary" onClick={doExport}>
                <Download className="h-4 w-4" /> Export
              </Button>
            </div>
          </div>
          {editItemId && (
            <InlineNote>Editing a pinned figure — “Save / pin” can update it in place.</InlineNote>
          )}
          {msg && <InlineNote tone="success">{msg}</InlineNote>}
          {meta && (
            <InlineNote>
              {meta.row_count.toLocaleString()} rows loaded · <Badge>{spec.data.source}</Badge>
            </InlineNote>
          )}
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

function AxisControls({
  title,
  axis,
  fallbackLabel,
  onChange,
  showRotation,
}: {
  title: string;
  axis: AxisSpec;
  fallbackLabel: string;
  onChange: (p: Partial<AxisSpec>) => void;
  showRotation?: boolean;
}) {
  return (
    <Section title={title} defaultOpen={false}>
      <Field label="Axis label">
        <TextInput
          value={axis.label ?? ""}
          placeholder={fallbackLabel}
          onChange={(e) => onChange({ label: e.target.value || null })}
        />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Min">
          <NumberInput value={axis.min} onChange={(v) => onChange({ min: v })} />
        </Field>
        <Field label="Max">
          <NumberInput value={axis.max} onChange={(v) => onChange({ max: v })} />
        </Field>
      </div>
      <Field label="Tick step (spacing)" hint="e.g. 2 → ticks at 0, 2, 4, … (blank = auto)">
        <NumberInput value={axis.tick_step} step={0.5} onChange={(v) => onChange({ tick_step: v })} />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Label size">
          <NumberInput value={axis.label_fontsize} onChange={(v) => onChange({ label_fontsize: v ?? 14 })} />
        </Field>
        <Field label="Tick size">
          <NumberInput value={axis.tick_fontsize} onChange={(v) => onChange({ tick_fontsize: v ?? 11 })} />
        </Field>
      </div>
      {showRotation && (
        <Field label={`Tick rotation (${axis.tick_rotation}°)`}>
          <Slider value={axis.tick_rotation} onChange={(v) => onChange({ tick_rotation: v })} min={0} max={90} />
        </Field>
      )}
      <Checkbox checked={axis.percentage} onChange={(v) => onChange({ percentage: v })} label="Show as percentage (×100)" />
    </Section>
  );
}
