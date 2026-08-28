import { useEffect, useState } from "react";
import { Download, FolderOpen, RotateCcw, Save, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import type { ExportFormat, FigureOptions, FigureSettings, GridFigureSpec, GridLayout, PanelRef, PanelStyle, RoiScope } from "../lib/types";
import { defaultGridSpec, hydrateGridSpec, LS_KEY, resizeGrid } from "../lib/figureSpec";
import { useDebounced, useObjectUrl } from "../lib/util";
import { Button, Checkbox, Field, NumberInput, Section, Segmented, Select, Slider, TextInput } from "../components/ui";
import { ColorWheel } from "../components/ColorWheel";
import { PathField } from "../components/DirectoryPicker";
import { PageHeader, PreviewPane, InlineNote } from "../components/common";
import { GridCanvas } from "../components/figure/GridCanvas";
import { RoiEditor } from "../components/figure/RoiEditor";
import { AutoFillPanel } from "../components/figure/AutoFillPanel";
import { InferenceBrowser } from "../components/figure/InferenceBrowser";

function loadSavedSpec(): GridFigureSpec {
  try {
    const saved = localStorage.getItem(LS_KEY);
    if (saved) return hydrateGridSpec(JSON.parse(saved));
  } catch {
    /* ignore corrupt storage */
  }
  return defaultGridSpec();
}

/** Round + clamp a NumberInput value for the rows/cols fields (grid stays sane). */
function clampDim(v: number | null, fallback: number): number {
  if (v == null || !Number.isFinite(v)) return fallback;
  return Math.max(1, Math.min(12, Math.round(v)));
}

/** Next empty cell after `from`, wrapping around — powers "auto-advance on pick". */
function nextEmptyIndex(panels: PanelRef[], from: number): number | null {
  const n = panels.length;
  for (let step = 1; step <= n; step++) {
    const i = (from + step) % n;
    if (panels[i].kind === "empty") return i;
  }
  return null;
}

export default function FigureStudio() {
  const [spec, setSpec] = useState<GridFigureSpec>(loadSavedSpec);
  const [selected, setSelected] = useState<number | null>(0);
  const [options, setOptions] = useState<FigureOptions | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [dataset, setDataset] = useState("");
  const [image, setImage] = useState("");

  const [tab, setTab] = useState<"live" | "rendered">("live");
  const [renderedUrl, setRenderedUrl] = useState<string | null>(null);
  const [renderedLoading, setRenderedLoading] = useState(false);
  const [renderedError, setRenderedError] = useState<string | null>(null);

  const [layouts, setLayouts] = useState<GridLayout[]>([]);
  const [selectedLayoutId, setSelectedLayoutId] = useState("");
  const [layoutName, setLayoutName] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [exportPanels, setExportPanels] = useState(false);
  const [panelScope, setPanelScope] = useState<"refs" | "all">("refs");
  const [panelNaming, setPanelNaming] = useState<"full" | "label">("full");

  useObjectUrl(renderedUrl);
  const debouncedSpec = useDebounced(spec, 500);

  // keep the grid across reloads
  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(spec));
  }, [spec]);

  useEffect(() => {
    api
      .figureOptions()
      .then((o) => {
        setOptions(o);
        setDataset((d) => d || o.datasets[0] || "");
      })
      .catch((e) => setOptionsError(String((e as Error).message ?? e)));
  }, []);

  // backend /figure/render half may not be finished yet — fail soft, Live tab keeps working
  useEffect(() => {
    if (tab !== "rendered") return;
    let cancelled = false;
    setRenderedLoading(true);
    setRenderedError(null);
    api
      .renderGrid(debouncedSpec)
      .then((u) => !cancelled && setRenderedUrl(u))
      .catch((e) => !cancelled && setRenderedError(String((e as Error).message ?? e)))
      .finally(() => !cancelled && setRenderedLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tab, debouncedSpec]);

  useEffect(() => {
    api
      .listLayouts()
      .then(setLayouts)
      .catch((e) => setMsg(`Couldn't load saved layouts: ${String((e as Error).message ?? e)}`));
  }, []);

  const patch = (p: Partial<GridFigureSpec>) => setSpec((s) => ({ ...s, ...p }));
  const patchPanelStyle = (p: Partial<PanelStyle>) =>
    setSpec((s) => ({ ...s, panel_style: { ...s.panel_style, ...p } }));
  const patchExport = (p: Partial<GridFigureSpec["export"]>) =>
    setSpec((s) => ({ ...s, export: { ...s.export, ...p } }));
  const patchTitle = (p: Partial<GridFigureSpec["title"]>) =>
    setSpec((s) => ({ ...s, title: { ...s.title, ...p } }));

  function handleDatasetChange(d: string) {
    setDataset(d);
    setImage("");
  }

  function handlePick(panel: PanelRef) {
    if (selected == null) return;
    const panels = [...spec.panels];
    panels[selected] = panel;
    setSpec({ ...spec, panels });
    const ne = nextEmptyIndex(panels, selected);
    if (ne != null) setSelected(ne);
  }

  function handleAutoFillResult(next: GridFigureSpec) {
    setSpec(hydrateGridSpec(next));
    setSelected(0);
  }

  function doReset() {
    if (!confirm("Reset the figure grid? This clears all panels, ROIs and layout settings.")) return;
    setSpec(defaultGridSpec());
    setSelected(0);
    setMsg("Grid reset.");
  }

  function download(blob: Blob, filename: string) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function detailOf(res: Response): Promise<string> {
    try {
      return ((await res.json()) as { detail?: string }).detail ?? res.statusText;
    } catch {
      return res.statusText;
    }
  }

  async function doExport() {
    setMsg(null);
    const notes: string[] = [];
    try {
      const res = await api.exportGrid(spec);
      const ct = res.headers.get("content-type") ?? "";
      if (ct.includes("application/json")) {
        const j = (await res.json()) as { saved?: boolean; path?: string; detail?: string };
        notes.push(j.saved ? `Saved to ${j.path}` : `Export error: ${j.detail ?? "unknown"}`);
      } else if (res.ok) {
        download(await res.blob(), `figure.${spec.export.format}`);
      } else {
        notes.push(`Export failed: ${await detailOf(res)}`);
      }

      // Separate per-panel files are a second request: same spec, different
      // packaging (zip on download, siblings on disk).
      if (exportPanels) {
        const pres = await api.exportGridPanels(spec, panelScope, panelNaming);
        const pct = pres.headers.get("content-type") ?? "";
        if (pct.includes("application/json")) {
          const j = (await pres.json()) as { saved?: boolean; count?: number; detail?: string };
          notes.push(
            j.saved ? `+ ${j.count} panel file(s)` : `Panel export error: ${j.detail ?? "unknown"}`,
          );
        } else if (pres.ok) {
          download(await pres.blob(), "figure_panels.zip");
        } else {
          notes.push(`Panel export failed: ${await detailOf(pres)}`);
        }
      }
    } catch (e) {
      notes.push(`Export failed: ${String((e as Error).message ?? e)}`);
    }
    if (notes.length) setMsg(notes.join(" · "));
  }

  async function doSaveLayout() {
    const name = layoutName.trim();
    if (!name) {
      setMsg("Name the layout before saving.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const existing = layouts.find((l) => l.name === name && l.id);
      const saved = existing?.id
        ? await api.updateLayout(existing.id, { name, spec })
        : await api.saveLayout(name, spec);
      setLayouts((ls) => [...ls.filter((l) => l.id !== saved.id), saved]);
      setSelectedLayoutId(saved.id ?? "");
      setMsg(`Saved layout "${name}".`);
    } catch (e) {
      setMsg(`Save failed: ${String((e as Error).message ?? e)}`);
    } finally {
      setBusy(false);
    }
  }

  function doLoadLayout() {
    const l = layouts.find((x) => x.id === selectedLayoutId);
    if (!l) return;
    setSpec(hydrateGridSpec(l.spec));
    setSelected(0);
    setMsg(`Loaded layout "${l.name}".`);
  }

  async function doDeleteLayout() {
    if (!selectedLayoutId) return;
    try {
      await api.deleteLayout(selectedLayoutId);
      setLayouts((ls) => ls.filter((l) => l.id !== selectedLayoutId));
      setSelectedLayoutId("");
      setMsg("Layout deleted.");
    } catch (e) {
      setMsg(`Delete failed: ${String((e as Error).message ?? e)}`);
    }
  }

  const selectionLabel =
    selected != null ? `row ${Math.floor(selected / spec.cols) + 1} · col ${(selected % spec.cols) + 1}` : undefined;
  const selectedPanel = selected != null ? spec.panels[selected] : null;

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Figure Studio"
        subtitle="Compose model × sample-count grids for the paper — draw a ROI once, it appears everywhere."
        actions={
          <>
            <div className="flex items-center gap-1.5">
              <div className="w-36">
                <Select
                  value={selectedLayoutId}
                  onChange={setSelectedLayoutId}
                  options={[
                    { value: "", label: layouts.length ? "Choose layout…" : "No saved layouts" },
                    ...layouts.map((l) => ({ value: l.id ?? "", label: l.name })),
                  ]}
                />
              </div>
              <Button size="sm" variant="ghost" title="Load layout" onClick={doLoadLayout} disabled={!selectedLayoutId}>
                <FolderOpen className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="ghost" title="Delete layout" onClick={doDeleteLayout} disabled={!selectedLayoutId}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-32">
                <TextInput
                  value={layoutName}
                  placeholder="Layout name"
                  onChange={(e) => setLayoutName(e.target.value)}
                />
              </div>
              <Button size="sm" variant="ghost" title="Save layout" onClick={doSaveLayout} disabled={busy}>
                <Save className="h-4 w-4" />
              </Button>
            </div>
            <Button size="sm" variant="ghost" title="Reset grid" onClick={doReset}>
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button variant="primary" onClick={doExport}>
              <Download className="h-4 w-4" /> Export
            </Button>
          </>
        }
      />

      <div className="flex flex-1 overflow-hidden">
        {/* control rail */}
        <div className="w-[380px] shrink-0 overflow-y-auto border-r border-border bg-surface px-5">
          <Section title="Layout" defaultOpen>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Rows">
                <NumberInput
                  value={spec.rows}
                  min={1}
                  max={12}
                  onChange={(v) => setSpec((s) => resizeGrid(s, clampDim(v, s.rows), s.cols))}
                />
              </Field>
              <Field label="Columns">
                <NumberInput
                  value={spec.cols}
                  min={1}
                  max={12}
                  onChange={(v) => setSpec((s) => resizeGrid(s, s.rows, clampDim(v, s.cols)))}
                />
              </Field>
            </div>
            <Field label="Panel size (inches)">
              <NumberInput value={spec.panel_size} step={0.1} min={0.5} onChange={(v) => patch({ panel_size: v ?? spec.panel_size })} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Column spacing">
                <NumberInput value={spec.wspace} step={0.01} min={0} max={1} onChange={(v) => patch({ wspace: v ?? 0 })} />
              </Field>
              <Field label="Row spacing">
                <NumberInput value={spec.hspace} step={0.01} min={0} max={1} onChange={(v) => patch({ hspace: v ?? 0 })} />
              </Field>
            </div>
            <Field label="Background">
              <div className="flex items-center gap-2">
                <ColorWheel color={spec.background} onChange={(hex) => patch({ background: hex })} />
                <TextInput value={spec.background} onChange={(e) => patch({ background: e.target.value })} />
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Label color">
                <ColorWheel color={spec.label_color} onChange={(hex) => patch({ label_color: hex })} />
              </Field>
              <Field label="Label size">
                <NumberInput value={spec.label_fontsize} onChange={(v) => patch({ label_fontsize: v ?? 11 })} />
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Row label size">
                <NumberInput value={spec.row_label_fontsize} onChange={(v) => patch({ row_label_fontsize: v ?? 11 })} />
              </Field>
              <Field label="Col label size">
                <NumberInput value={spec.col_label_fontsize} onChange={(v) => patch({ col_label_fontsize: v ?? 11 })} />
              </Field>
            </div>
            <Field label="Figure title">
              <TextInput value={spec.title.text} placeholder="(none)" onChange={(e) => patchTitle({ text: e.target.value })} />
            </Field>
            <Field label="Title size">
              <NumberInput value={spec.title.fontsize} onChange={(v) => patchTitle({ fontsize: v ?? 16 })} />
            </Field>
          </Section>

          <Section title="Auto-fill" defaultOpen={false}>
            <AutoFillPanel options={options} dataset={dataset} image={image} onResult={handleAutoFillResult} />
          </Section>

          <Section title="ROI" defaultOpen={false}>
            <Field label="ROI scope" hint="Which panels share one drawn ROI.">
              <Select
                value={spec.roi_scope}
                onChange={(v) => patch({ roi_scope: v as RoiScope })}
                options={[
                  { value: "image", label: "Same image (all models/N)" },
                  { value: "column", label: "Same column" },
                  { value: "panel", label: "This panel only" },
                  { value: "figure", label: "Whole figure" },
                ]}
              />
            </Field>
            <RoiEditor spec={spec} panel={selectedPanel} index={selected} onChange={setSpec} />
          </Section>

          <Section title="Panel style" defaultOpen={false}>
            <div className="flex items-center gap-4">
              <Checkbox checked={spec.panel_style.show_label} onChange={(v) => patchPanelStyle({ show_label: v })} label="Show label" />
              <Checkbox checked={spec.panel_style.show_metric} onChange={(v) => patchPanelStyle({ show_metric: v })} label="Show metric" />
            </div>
            <Field label="Label position">
              <Segmented
                value={spec.panel_style.label_loc}
                onChange={(v) => patchPanelStyle({ label_loc: v })}
                options={[
                  { value: "top", label: "Top" },
                  { value: "bottom", label: "Bottom" },
                  { value: "none", label: "None" },
                ]}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Metric">
                <Select
                  value={spec.panel_style.metric}
                  onChange={(v) => patchPanelStyle({ metric: v })}
                  options={(options?.metrics ?? []).map((m) => ({ value: m, label: m }))}
                />
              </Field>
              <Field label="Decimals">
                <NumberInput value={spec.panel_style.decimals} min={0} max={6} onChange={(v) => patchPanelStyle({ decimals: v ?? 3 })} />
              </Field>
            </div>
            <Field label="Metric position">
              <Select
                value={spec.panel_style.metric_loc}
                onChange={(v) => patchPanelStyle({ metric_loc: v as PanelStyle["metric_loc"] })}
                options={[
                  { value: "upper left", label: "Upper left" },
                  { value: "upper right", label: "Upper right" },
                  { value: "lower left", label: "Lower left" },
                  { value: "lower right", label: "Lower right" },
                ]}
              />
            </Field>
            <div className="flex items-end gap-3">
              <Field label="Metric color" className="w-auto">
                <ColorWheel color={spec.panel_style.metric_color} onChange={(hex) => patchPanelStyle({ metric_color: hex })} />
              </Field>
              <Field label="Metric size" className="flex-1">
                <NumberInput value={spec.panel_style.metric_fontsize} onChange={(v) => patchPanelStyle({ metric_fontsize: v ?? 9 })} />
              </Field>
            </div>
            <div className="flex items-end gap-3">
              <Field label="Metric box color" className="w-auto">
                <ColorWheel
                  color={spec.panel_style.metric_bg_color}
                  onChange={(hex) => patchPanelStyle({ metric_bg_color: hex })}
                />
              </Field>
              <Field
                label={`Metric box opacity (${Math.round(spec.panel_style.metric_bg_alpha * 100)}%)`}
                className="flex-1"
              >
                <Slider
                  value={spec.panel_style.metric_bg_alpha}
                  onChange={(v) => patchPanelStyle({ metric_bg_alpha: v })}
                  min={0}
                  max={1}
                  step={0.05}
                />
              </Field>
            </div>
            <Checkbox checked={spec.panel_style.border} onChange={(v) => patchPanelStyle({ border: v })} label="Draw border" />
            {spec.panel_style.border && (
              <div className="flex items-end gap-3">
                <Field label="Border color" className="w-auto">
                  <ColorWheel color={spec.panel_style.border_color} onChange={(hex) => patchPanelStyle({ border_color: hex })} />
                </Field>
                <Field label="Border width" className="flex-1">
                  <NumberInput
                    value={spec.panel_style.border_width}
                    step={0.5}
                    onChange={(v) => patchPanelStyle({ border_width: v ?? 1 })}
                  />
                </Field>
              </div>
            )}
            <Checkbox checked={spec.panel_style.invert} onChange={(v) => patchPanelStyle({ invert: v })} label="Invert grayscale (masks)" />
            <Checkbox
              checked={spec.panel_style.missing_mark}
              onChange={(v) => patchPanelStyle({ missing_mark: v })}
              label="Mark missing combinations with a boxed X"
            />
            {spec.panel_style.missing_mark && (
              <div className="flex items-end gap-3">
                <Field label="X color" className="w-auto">
                  <ColorWheel
                    color={spec.panel_style.missing_color ?? spec.label_color}
                    onChange={(hex) => patchPanelStyle({ missing_color: hex })}
                  />
                </Field>
                <Field label="X width" className="flex-1">
                  <NumberInput
                    value={spec.panel_style.missing_width}
                    step={0.2}
                    onChange={(v) => patchPanelStyle({ missing_width: v ?? 0.8 })}
                  />
                </Field>
              </div>
            )}
          </Section>

          <Section title="Export" defaultOpen={false}>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Format">
                <Select
                  value={spec.export.format}
                  onChange={(v) => patchExport({ format: v as ExportFormat })}
                  options={["svg", "png", "jpg", "pdf"].map((f) => ({ value: f, label: f.toUpperCase() }))}
                />
              </Field>
              <Field label="DPI">
                <NumberInput value={spec.export.dpi} onChange={(v) => patchExport({ dpi: v ?? 300 })} />
              </Field>
            </div>
            <Field label="Save path (blank = download)">
              <PathField
                value={spec.export.save_path ?? ""}
                onChange={(v) => patchExport({ save_path: v || null })}
                mode="file"
                placeholder="/abs/path/figure.svg"
                defaultFilename={`figure.${spec.export.format}`}
              />
            </Field>
            <Checkbox
              checked={exportPanels}
              onChange={setExportPanels}
              label="Also export panels as separate files"
            />
            {exportPanels && (
              <>
                <Segmented
                  value={panelScope}
                  onChange={(v) => setPanelScope(v as "refs" | "all")}
                  options={[
                    { value: "refs", label: "Input + GT" },
                    { value: "all", label: "+ every panel" },
                  ]}
                />
                <Field label="File names">
                  <Segmented
                    value={panelNaming}
                    onChange={(v) => setPanelNaming(v as "full" | "label")}
                    options={[
                      { value: "full", label: "Full detail" },
                      { value: "label", label: "Panel name" },
                    ]}
                  />
                </Field>
                <InlineNote>
                  {panelNaming === "label"
                    ? "Named after the panel: VSUNet18_20-shot.svg, Input.svg, Ground-truth.svg. Repeats get a numeric suffix."
                    : "Named with dataset, image and run identity — unambiguous across figures, but long."}
                </InlineNote>
                <InlineNote>
                  Input and ground truth are always included, whether or not the grid shows them,
                  and inherit the figure's crop and ROI so a zoomed figure gets zoomed reference
                  frames. Each file is rendered on its own, without row/column labels.
                  {spec.export.save_path
                    ? panelNaming === "label"
                      ? " Written next to the figure."
                      : " Written next to the figure with its name as prefix."
                    : " Downloaded as a zip alongside the figure."}
                </InlineNote>
              </>
            )}
          </Section>

          <Section title="Dataset source" defaultOpen={false}>
            <DatasetSource />
          </Section>
        </div>

        {/* main area: canvas + inference browser */}
        <div className="flex min-w-0 flex-1 overflow-hidden">
          <div className="flex min-w-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
            <div className="flex items-center justify-between gap-3">
              <Segmented
                value={tab}
                onChange={setTab}
                options={[
                  { value: "live", label: "Live" },
                  { value: "rendered", label: "Rendered" },
                ]}
              />
              {selectionLabel && <span className="text-xs text-muted-fg">Selected: {selectionLabel}</span>}
            </div>

            {tab === "live" ? (
              <GridCanvas spec={spec} onChange={setSpec} selected={selected} onSelect={setSelected} />
            ) : (
              <PreviewPane url={renderedUrl} loading={renderedLoading} error={renderedError} />
            )}

            {optionsError && <InlineNote tone="danger">Couldn't load figure options: {optionsError}</InlineNote>}
            {msg && <InlineNote tone={/fail|error|couldn't/i.test(msg) ? "danger" : "success"}>{msg}</InlineNote>}
          </div>

          <div className="w-96 shrink-0 overflow-hidden border-l border-border p-3">
            <InferenceBrowser
              options={options}
              dataset={dataset}
              onDatasetChange={handleDatasetChange}
              image={image}
              onImageChange={setImage}
              onPick={handlePick}
              selectionLabel={selectionLabel}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/** Where the backend reads input images and ground truth from.
 *
 *  Predictions live in the repo and always resolve; only inputs/GT sit on an
 *  external drive, so this is the one path that can go wrong (a drive mounted
 *  elsewhere, or another machine) — hence the per-dataset probe rather than a
 *  bare text field. */
function DatasetSource() {
  const [settings, setSettings] = useState<FigureSettings | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function load(p: Promise<FigureSettings>) {
    setBusy(true);
    setError(null);
    p.then((s) => {
      setSettings(s);
      setDraft(s.dataset_root);
    })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  }

  useEffect(() => {
    load(api.figureSettings());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const missing = settings ? Object.entries(settings.sources).filter(([, v]) => !v.found) : [];

  return (
    <div className="space-y-2">
      <Field label="Images + ground truth folder">
        <TextInput
          value={draft}
          placeholder={settings?.default_root ?? "/path/to/blood_vessels"}
          onChange={(e) => setDraft(e.target.value)}
        />
      </Field>
      <div className="flex gap-2">
        <Button size="sm" disabled={busy} onClick={() => load(api.setFigureSettings(draft.trim() || null))}>
          Apply
        </Button>
        <Button size="sm" variant="ghost" disabled={busy} onClick={() => load(api.setFigureSettings(null))}>
          Reset to default
        </Button>
      </div>
      {error && <InlineNote tone="danger">{error}</InlineNote>}
      {settings && !settings.found && (
        <InlineNote tone="danger">Folder not found: {settings.dataset_root}</InlineNote>
      )}
      {settings && settings.found && missing.length > 0 && (
        <InlineNote tone="danger">
          Missing subfolder(s): {missing.map(([k, v]) => `${k} (${v.path})`).join(", ")}
        </InlineNote>
      )}
      {settings && settings.found && missing.length === 0 && (
        <InlineNote tone="success">
          All {Object.keys(settings.sources).length} datasets found.
        </InlineNote>
      )}
    </div>
  );
}
