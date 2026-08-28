import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Copy, Download, FolderOpen, Pin, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import type { Aggregation, Metadata, TableResult, TableSpec } from "../lib/types";
import {
  AGGREGATIONS,
  ALL_METRICS,
  COUNT_VALUE,
  defaultTableSpec,
  hydrateTableSpec,
  TABLE_DIMENSIONS,
  TABLE_PRESETS,
} from "../lib/plotSpec";
import { useDebounced } from "../lib/util";
import {
  Badge,
  Button,
  Checkbox,
  ChipMultiSelect,
  Field,
  NumberInput,
  Section,
  Select,
  TextInput,
} from "../components/ui";
import { PinDialog, type PinResult } from "../components/PinDialog";
import { DirectoryPicker } from "../components/DirectoryPicker";
import { TableView } from "../components/TableView";
import { PageHeader, InlineNote } from "../components/common";

const ALL_DATASETS = ["vessmap", "drive", "dca1", "octa2d"];
const SOURCES = [
  { value: "per_run", label: "Per-run" },
  { value: "per_image", label: "Per-image" },
  { value: "summary", label: "Summary" },
];
const COUNT_LABEL = "Count (#runs)";
const SPEC_KEY = "vesslab.tablesSpec";
const SAVE_DIR_KEY = "vesslab.tablesSaveDir";

function loadSavedSpec(): TableSpec {
  try {
    const saved = localStorage.getItem(SPEC_KEY);
    if (saved) return hydrateTableSpec(JSON.parse(saved));
  } catch {
    /* ignore corrupt storage */
  }
  return defaultTableSpec();
}

export default function Tables() {
  const [spec, setSpec] = useState<TableSpec>(loadSavedSpec);
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [result, setResult] = useState<TableResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [pinOpen, setPinOpen] = useState(false);
  const [editItemId, setEditItemId] = useState<string | null>(null);
  const [saveTarget, setSaveTarget] = useState<{ label: string; text: string; ext: string } | null>(null);
  const [saveDir, setSaveDir] = useState(() => localStorage.getItem(SAVE_DIR_KEY) ?? "");

  const location = useLocation();
  const navigate = useNavigate();
  const debounced = useDebounced(spec, 300);

  // edit a pinned table (navigated from the Dashboard)
  useEffect(() => {
    const st = location.state as { editSpec?: Record<string, unknown>; editItemId?: string } | null;
    if (st?.editSpec) {
      setSpec(hydrateTableSpec(st.editSpec));
      setEditItemId(st.editItemId ?? null);
      navigate(location.pathname, { replace: true, state: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // keep the last table configuration across reloads
  useEffect(() => {
    localStorage.setItem(SPEC_KEY, JSON.stringify(spec));
  }, [spec]);

  useEffect(() => {
    api.metadata(spec.source, spec.datasets).then(setMeta).catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.source, spec.datasets.join(",")]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .renderTable(debounced)
      .then((r) => !cancelled && setResult(r))
      .catch((e) => !cancelled && setError(String(e.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  const patch = (p: Partial<TableSpec>) => setSpec((s) => ({ ...s, ...p }));
  const opt = (field: string): string[] => (meta?.options[field]?.values ?? []).map((v) => String(v));
  const sampleOptions = (meta?.options.num_samples?.values ?? []).map(Number);

  // values <-> display ("Count" chip maps to the COUNT sentinel)
  const valueOptions = [...ALL_METRICS, COUNT_LABEL];
  const valueSelected = spec.values.map((v) => (v === COUNT_VALUE ? COUNT_LABEL : v));
  const onValues = (v: string[]) =>
    patch({ values: v.map((x) => (x === COUNT_LABEL ? COUNT_VALUE : x)) });

  const sortOptions = [
    { value: "", label: "(none)" },
    ...spec.values.map((v) => ({
      value: v === COUNT_VALUE ? "count" : v,
      label: v === COUNT_VALUE ? "count" : v,
    })),
  ];

  const pinTitle = spec.title || result?.caption || "Table";

  function copy(text: string, what: string) {
    navigator.clipboard.writeText(text).then(() => setMsg(`${what} copied to clipboard.`));
  }
  function download(text: string, ext: string, mime: string) {
    const blob = new Blob([text], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `table.${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
  }
  async function saveToDisk(path: string) {
    if (!saveTarget) return;
    try {
      await api.writeFile(path, saveTarget.text);
      const sep = path.includes("\\") ? "\\" : "/";
      const dir = path.slice(0, path.lastIndexOf(sep));
      if (dir) {
        setSaveDir(dir);
        localStorage.setItem(SAVE_DIR_KEY, dir);
      }
      setMsg(`Saved to ${path}.`);
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    }
  }

  async function handlePin({ dashboardId, title, mode }: PinResult) {
    const payload = {
      title,
      kind: "table" as const,
      regime: null,
      dashboard_id: dashboardId,
      spec: spec as unknown as Record<string, unknown>,
    };
    if (mode === "update" && editItemId) {
      await api.updatePin(editItemId, payload);
      setMsg("Updated pinned table.");
    } else {
      const created = await api.pin(payload);
      setEditItemId(created.id ?? null);
      setMsg("Pinned to dashboard.");
    }
  }

  const exporters = useMemo(
    () => [
      { label: "Markdown", text: result?.markdown ?? "", ext: "md", mime: "text/markdown" },
      { label: "CSV", text: result?.csv ?? "", ext: "csv", mime: "text/csv" },
      { label: "LaTeX", text: result?.latex ?? "", ext: "tex", mime: "text/plain" },
    ],
    [result],
  );

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Tables"
        subtitle="Pivot the results like a BI tool — rows × columns × measures, then export or pin."
      />

      <div className="flex flex-1 overflow-hidden">
        {/* control panel */}
        <div className="w-[380px] shrink-0 overflow-y-auto border-r border-border bg-surface px-5">
          <Section title="Presets" defaultOpen>
            <div className="grid grid-cols-1 gap-2">
              {TABLE_PRESETS.map((p) => (
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

          <Section title="Data & filters">
            <Field label="Source">
              <Select value={spec.source} onChange={(v) => patch({ source: v as TableSpec["source"] })} options={SOURCES} />
            </Field>
            <Field label="Datasets">
              <ChipMultiSelect
                options={ALL_DATASETS}
                selected={spec.datasets}
                onChange={(v) => patch({ datasets: v.length ? v : ALL_DATASETS })}
                emptyMeansAll={false}
              />
            </Field>
            <Field label="Models / methods (empty = all)">
              <ChipMultiSelect options={opt("model_type")} selected={spec.model_types} onChange={(v) => patch({ model_types: v })} />
            </Field>
            <Field label="Stage (empty = all)">
              <ChipMultiSelect options={opt("stage")} selected={spec.stages} onChange={(v) => patch({ stages: v })} />
            </Field>
            <Field label="Num. samples filter (empty = all)">
              <ChipMultiSelect
                options={sampleOptions.map(String)}
                selected={spec.sample_filter.map(String)}
                onChange={(v) => patch({ sample_filter: v.map(Number) })}
                emptyMeansAll={false}
              />
            </Field>
            <Checkbox checked={spec.exclude_mock} onChange={(v) => patch({ exclude_mock: v })} label="Exclude mock placeholders" />
          </Section>

          <Section title="Pivot layout">
            <Field label="Rows (dimensions)" hint="Order = nesting order.">
              <ChipMultiSelect options={TABLE_DIMENSIONS} selected={spec.rows} onChange={(v) => patch({ rows: v })} emptyMeansAll={false} />
            </Field>
            <Field label="Columns (optional pivot)" hint="When set, uses the first measure only.">
              <ChipMultiSelect options={TABLE_DIMENSIONS} selected={spec.columns} onChange={(v) => patch({ columns: v })} emptyMeansAll={false} />
            </Field>
            <Field label="Measures (values)">
              <ChipMultiSelect options={valueOptions} selected={valueSelected} onChange={onValues} emptyMeansAll={false} />
            </Field>
            <Field label="Aggregation">
              <Select
                value={spec.aggregation}
                onChange={(v) => patch({ aggregation: v as Aggregation })}
                options={AGGREGATIONS.map((a) => ({ value: a.value, label: a.label }))}
              />
            </Field>
          </Section>

          <Section title="Format & sort" defaultOpen={false}>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Decimals">
                <NumberInput value={spec.decimals} min={0} max={6} onChange={(v) => patch({ decimals: v ?? 3 })} />
              </Field>
              <Field label="Sort rows by">
                <Select value={spec.sort_by ?? ""} onChange={(v) => patch({ sort_by: v || null })} options={sortOptions} />
              </Field>
            </div>
            <div className="flex items-center gap-4">
              <Checkbox checked={spec.percentage} onChange={(v) => patch({ percentage: v })} label="As percentage (×100)" />
              <Checkbox checked={spec.ascending} onChange={(v) => patch({ ascending: v })} label="Ascending" />
            </div>
            <Field label="Table title (blank = auto)">
              <TextInput value={spec.title} placeholder={result?.caption ?? ""} onChange={(e) => patch({ title: e.target.value })} />
            </Field>
          </Section>
        </div>

        {/* preview + export */}
        <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-6">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-fg">{result?.caption ?? "Table"}</h2>
              {result && (
                <p className="text-[11px] text-muted-fg">
                  {result.n_rows} row(s) · <Badge>{spec.source}</Badge>
                </p>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button variant="default" onClick={() => setPinOpen(true)}>
                <Pin className="h-4 w-4" /> {editItemId ? "Save / pin" : "Pin to dashboard"}
              </Button>
            </div>
          </div>

          {error ? (
            <InlineNote tone="danger">{error}</InlineNote>
          ) : result ? (
            <TableView columns={result.columns} rows={result.rows} />
          ) : (
            <InlineNote>{loading ? "Computing…" : "No table yet."}</InlineNote>
          )}

          {/* export */}
          {result && (
            <div className="rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-xs font-medium text-muted-fg">Export (already formatted)</div>
              <div className="flex flex-wrap gap-4">
                {exporters.map((ex) => (
                  <div key={ex.label} className="flex items-center gap-1.5">
                    <span className="text-xs font-medium text-fg">{ex.label}</span>
                    <Button size="sm" variant="ghost" onClick={() => copy(ex.text, ex.label)} title={`Copy ${ex.label}`}>
                      <Copy className="h-3.5 w-3.5" /> Copy
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => download(ex.text, ex.ext, ex.mime)} title={`Download ${ex.label}`}>
                      <Download className="h-3.5 w-3.5" /> .{ex.ext}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setSaveTarget(ex)} title={`Save ${ex.label} to a folder`}>
                      <FolderOpen className="h-3.5 w-3.5" /> Save…
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {editItemId && <InlineNote>Editing a pinned table — “Save / pin” can update it in place.</InlineNote>}
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

      <DirectoryPicker
        open={saveTarget != null}
        mode="file"
        initialPath={saveDir || undefined}
        defaultFilename={saveTarget ? `table.${saveTarget.ext}` : undefined}
        title={saveTarget ? `Save ${saveTarget.label} to...` : undefined}
        onConfirm={saveToDisk}
        onClose={() => setSaveTarget(null)}
      />
    </div>
  );
}
