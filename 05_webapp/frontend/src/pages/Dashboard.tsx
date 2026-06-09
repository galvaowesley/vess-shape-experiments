import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Download, RefreshCw, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import type { DashboardItem, PlotSpec, WilcoxonSpec } from "../lib/types";
import { useObjectUrl } from "../lib/util";
import { Badge, Button, ChipMultiSelect, Field, NumberInput, Select, TextInput } from "../components/ui";
import { PageHeader, InlineNote } from "../components/common";

export default function Dashboard() {
  const [items, setItems] = useState<DashboardItem[]>([]);
  const [filter, setFilter] = useState<string[]>([]);
  const [outDir, setOutDir] = useState("");
  const [fmt, setFmt] = useState("png");
  const [dpi, setDpi] = useState(300);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setItems(await api.listDashboard());
  }
  useEffect(() => {
    refresh();
  }, []);

  const regimes = useMemo(
    () => Array.from(new Set(items.map((i) => i.regime || "Ungrouped"))),
    [items],
  );
  const visible = filter.length
    ? items.filter((i) => filter.includes(i.regime || "Ungrouped"))
    : items;

  async function move(id: string, dir: -1 | 1) {
    const ids = items.map((i) => i.id!);
    const idx = ids.indexOf(id);
    const swap = idx + dir;
    if (swap < 0 || swap >= ids.length) return;
    [ids[idx], ids[swap]] = [ids[swap], ids[idx]];
    setItems(await api.reorder(ids));
  }

  async function remove(id: string) {
    await api.unpin(id);
    refresh();
  }

  async function exportAll() {
    if (!outDir) {
      setMsg("Set an output directory first.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const res = await api.exportDashboard(outDir, fmt, dpi);
      const written = (res.written as unknown[])?.length ?? 0;
      setMsg(`Exported ${written} figure(s) to ${res.out_dir}.`);
    } catch (e) {
      setMsg(`Export failed: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  // group visible items by regime
  const grouped = useMemo(() => {
    const g: Record<string, DashboardItem[]> = {};
    for (const it of visible) {
      const k = it.regime || "Ungrouped";
      (g[k] ??= []).push(it);
    }
    return g;
  }, [visible]);

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Dashboard"
        subtitle="Pinned figures, grouped by shot-regime, for side-by-side comparison."
        actions={
          <Button variant="ghost" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        }
      />

      <div className="flex flex-wrap items-end gap-3 border-b border-border bg-surface px-6 py-3">
        {regimes.length > 0 && (
          <Field label="Filter by regime" className="min-w-[220px]">
            <ChipMultiSelect options={regimes} selected={filter} onChange={setFilter} />
          </Field>
        )}
        <div className="ml-auto flex flex-wrap items-end gap-2">
          <Field label="Export all → directory" className="w-72">
            <TextInput value={outDir} placeholder="/abs/path/figures" onChange={(e) => setOutDir(e.target.value)} />
          </Field>
          <Field label="Format" className="w-24">
            <Select value={fmt} onChange={setFmt} options={["png", "svg", "jpg", "pdf"].map((f) => ({ value: f, label: f.toUpperCase() }))} />
          </Field>
          <Field label="DPI" className="w-20">
            <NumberInput value={dpi} onChange={(v) => setDpi(v ?? 300)} />
          </Field>
          <Button variant="primary" onClick={exportAll} disabled={busy || items.length === 0}>
            <Download className="h-4 w-4" /> Export all
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {msg && <InlineNote tone="success">{msg}</InlineNote>}
        {items.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-fg">
            No figures pinned yet. Use “Pin to dashboard” in the Chart Builder or Significance pages.
          </div>
        ) : (
          Object.entries(grouped).map(([regime, group]) => (
            <div key={regime} className="mb-8">
              <div className="mb-3 flex items-center gap-2">
                <h2 className="text-sm font-bold text-fg">{regime}</h2>
                <Badge>{group.length}</Badge>
              </div>
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {group.map((it) => (
                  <DashboardCard
                    key={it.id}
                    item={it}
                    onRemove={() => remove(it.id!)}
                    onUp={() => move(it.id!, -1)}
                    onDown={() => move(it.id!, 1)}
                  />
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function DashboardCard({
  item,
  onRemove,
  onUp,
  onDown,
}: {
  item: DashboardItem;
  onRemove: () => void;
  onUp: () => void;
  onDown: () => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useObjectUrl(url);

  useEffect(() => {
    let cancelled = false;
    const render =
      item.kind === "wilcoxon"
        ? api.renderWilcoxon(item.spec as unknown as WilcoxonSpec)
        : api.renderFigure(item.spec as unknown as PlotSpec);
    render
      .then((u) => !cancelled && setUrl(u))
      .catch((e) => !cancelled && setError(String(e.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, [item]);

  return (
    <div className="flex flex-col rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Badge tone={item.kind === "wilcoxon" ? "accent" : "primary"}>
            {item.kind === "wilcoxon" ? "Wilcoxon" : "Figure"}
          </Badge>
          <span className="truncate text-sm font-medium text-fg" title={item.title}>
            {item.title}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="sm" onClick={onUp} title="Move up">
            <ArrowUp className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onDown} title="Move down">
            <ArrowDown className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onRemove} title="Remove">
            <Trash2 className="h-3.5 w-3.5 text-danger" />
          </Button>
        </div>
      </div>
      <div className="flex min-h-[240px] items-center justify-center p-3">
        {error ? (
          <span className="text-xs text-danger">{error}</span>
        ) : url ? (
          <img src={url} alt={item.title} className="max-h-[360px] max-w-full object-contain" />
        ) : (
          <span className="text-xs text-muted-fg">rendering…</span>
        )}
      </div>
    </div>
  );
}
