import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  Download,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { api } from "../lib/api";
import type {
  DashboardItem,
  DashboardMeta,
  PlotSpec,
  TableResult,
  TableSpec,
  WilcoxonSpec,
} from "../lib/types";
import { cn, useObjectUrl } from "../lib/util";
import { Badge, Button, Field, NumberInput, Select, TextInput } from "../components/ui";
import { TableView } from "../components/TableView";
import { PathField } from "../components/DirectoryPicker";
import { PageHeader, InlineNote } from "../components/common";

const ACTIVE_KEY = "vesslab.activeDashboard";

export default function Dashboard() {
  const navigate = useNavigate();
  const [dashboards, setDashboards] = useState<DashboardMeta[]>([]);
  const [openIds, setOpenIds] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [items, setItems] = useState<DashboardItem[]>([]);

  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [outDir, setOutDir] = useState("");
  const [fmt, setFmt] = useState("png");
  const [dpi, setDpi] = useState(300);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadDashboards = useCallback(async () => {
    const ds = await api.listDashboards();
    setDashboards(ds);
    return ds;
  }, []);

  // first load: reopen the last-used dashboard (or the first) as a tab
  useEffect(() => {
    loadDashboards().then((ds) => {
      if (!ds.length) return;
      const saved = localStorage.getItem(ACTIVE_KEY);
      const start = ds.find((d) => d.id === saved)?.id ?? ds[0].id;
      setOpenIds([start]);
      setActiveId(start);
    });
  }, [loadDashboards]);

  // remember the active dashboard across reloads
  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);

  const loadItems = useCallback(async (id: string | null) => {
    if (!id) {
      setItems([]);
      return;
    }
    setItems(await api.listDashboard(id));
  }, []);

  useEffect(() => {
    loadItems(activeId);
  }, [activeId, loadItems]);

  const activeMeta = dashboards.find((d) => d.id === activeId) ?? null;
  const openTabs = openIds
    .map((id) => dashboards.find((d) => d.id === id))
    .filter((d): d is DashboardMeta => Boolean(d));
  const closedDashboards = dashboards.filter((d) => !openIds.includes(d.id));

  function openTab(id: string) {
    setOpenIds((ids) => (ids.includes(id) ? ids : [...ids, id]));
    setActiveId(id);
  }
  function closeTab(id: string) {
    setOpenIds((ids) => {
      const next = ids.filter((x) => x !== id);
      if (activeId === id) setActiveId(next[next.length - 1] ?? null);
      return next;
    });
  }

  async function createDashboard() {
    const name = newName.trim();
    if (!name) return;
    const created = await api.createDashboard(name);
    setNewName("");
    setCreating(false);
    await loadDashboards();
    openTab(created.id);
  }

  async function deleteDashboard() {
    if (!activeMeta) return;
    if (!window.confirm(`Delete dashboard “${activeMeta.name}” and its ${activeMeta.count ?? 0} figure(s)?`))
      return;
    await api.deleteDashboard(activeMeta.id);
    closeTab(activeMeta.id);
    await loadDashboards();
    setMsg(`Deleted dashboard “${activeMeta.name}”.`);
  }

  async function move(id: string, dir: -1 | 1) {
    const ids = items.map((i) => i.id!);
    const idx = ids.indexOf(id);
    const swap = idx + dir;
    if (swap < 0 || swap >= ids.length) return;
    [ids[idx], ids[swap]] = [ids[swap], ids[idx]];
    await api.reorder(ids);
    await loadItems(activeId);
  }

  async function remove(id: string) {
    await api.unpin(id);
    await loadItems(activeId);
    loadDashboards();
  }

  function edit(item: DashboardItem) {
    const path =
      item.kind === "wilcoxon" ? "/significance" : item.kind === "table" ? "/tables" : "/charts";
    navigate(path, { state: { editSpec: item.spec, editItemId: item.id } });
  }

  async function exportActive() {
    if (!activeId) return;
    if (!outDir) {
      setMsg("Set an output directory first.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const res = await api.exportDashboard(outDir, fmt, dpi, activeId);
      const written = (res.written as unknown[])?.length ?? 0;
      setMsg(`Exported ${written} figure(s) to ${res.out_dir}.`);
    } catch (e) {
      setMsg(`Export failed: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  // group active items by regime
  const grouped = useMemo(() => {
    const g: Record<string, DashboardItem[]> = {};
    for (const it of items) {
      const k = it.regime || "Ungrouped";
      (g[k] ??= []).push(it);
    }
    return g;
  }, [items]);

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Dashboard"
        subtitle="Named dashboards of pinned figures — load any as a tab, edit, or export."
        actions={
          <Button variant="ghost" size="sm" onClick={() => { loadDashboards(); loadItems(activeId); }}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        }
      />

      {/* tab bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border bg-surface px-6 py-2">
        {openTabs.map((d) => (
          <div
            key={d.id}
            className={cn(
              "group flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm cursor-pointer transition-colors",
              d.id === activeId
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface-2 text-muted-fg hover:text-fg",
            )}
            onClick={() => setActiveId(d.id)}
          >
            <span className="truncate max-w-[180px]">{d.name}</span>
            <Badge>{d.count ?? 0}</Badge>
            <button
              onClick={(e) => { e.stopPropagation(); closeTab(d.id); }}
              title="Close tab"
              className="rounded p-0.5 opacity-60 hover:bg-muted hover:opacity-100"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}

        {/* open another dashboard */}
        {closedDashboards.length > 0 && (
          <div className="w-48">
            <Select
              value=""
              onChange={(id) => id && openTab(id)}
              options={[
                { value: "", label: "Load dashboard…" },
                ...closedDashboards.map((d) => ({ value: d.id, label: `${d.name} (${d.count ?? 0})` })),
              ]}
            />
          </div>
        )}

        {/* new dashboard */}
        {creating ? (
          <div className="flex items-center gap-1">
            <div className="w-44">
              <TextInput
                autoFocus
                value={newName}
                placeholder="Dashboard name"
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") createDashboard();
                  if (e.key === "Escape") { setCreating(false); setNewName(""); }
                }}
              />
            </div>
            <Button size="sm" variant="primary" onClick={createDashboard}>Create</Button>
            <Button size="sm" variant="ghost" onClick={() => { setCreating(false); setNewName(""); }}>Cancel</Button>
          </div>
        ) : (
          <Button size="sm" variant="ghost" onClick={() => setCreating(true)}>
            <Plus className="h-4 w-4" /> New dashboard
          </Button>
        )}

        {activeMeta && (
          <Button size="sm" variant="ghost" className="ml-auto text-danger" onClick={deleteDashboard}>
            <Trash2 className="h-4 w-4" /> Delete “{activeMeta.name}”
          </Button>
        )}
      </div>

      {/* export bar */}
      <div className="flex flex-wrap items-end gap-3 border-b border-border bg-surface px-6 py-3">
        <div className="ml-auto flex flex-wrap items-end gap-2">
          <Field label="Export this dashboard → directory" className="w-72">
            <PathField value={outDir} onChange={setOutDir} mode="directory" placeholder="/abs/path/figures" />
          </Field>
          <Field label="Format" className="w-24">
            <Select value={fmt} onChange={setFmt} options={["png", "svg", "jpg", "pdf"].map((f) => ({ value: f, label: f.toUpperCase() }))} />
          </Field>
          <Field label="DPI" className="w-20">
            <NumberInput value={dpi} onChange={(v) => setDpi(v ?? 300)} />
          </Field>
          <Button variant="primary" onClick={exportActive} disabled={busy || items.length === 0}>
            <Download className="h-4 w-4" /> Export all
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {msg && <InlineNote tone="success">{msg}</InlineNote>}
        {dashboards.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-sm text-muted-fg">
            No dashboards yet. Pin a figure from the Chart Builder or Significance pages, or create one here.
          </div>
        ) : !activeId ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-fg">
            Open a dashboard tab above to view its figures.
          </div>
        ) : items.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-fg">
            “{activeMeta?.name}” is empty — pin a figure to it from the Chart Builder.
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
                    onEdit={() => edit(it)}
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
  onEdit,
  onRemove,
  onUp,
  onDown,
}: {
  item: DashboardItem;
  onEdit: () => void;
  onRemove: () => void;
  onUp: () => void;
  onDown: () => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [table, setTable] = useState<TableResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useObjectUrl(url);

  useEffect(() => {
    let cancelled = false;
    if (item.kind === "table") {
      api
        .renderTable(item.spec as unknown as TableSpec)
        .then((t) => !cancelled && setTable(t))
        .catch((e) => !cancelled && setError(String(e.message ?? e)));
    } else {
      const render =
        item.kind === "wilcoxon"
          ? api.renderWilcoxon(item.spec as unknown as WilcoxonSpec)
          : api.renderFigure(item.spec as unknown as PlotSpec);
      render
        .then((u) => !cancelled && setUrl(u))
        .catch((e) => !cancelled && setError(String(e.message ?? e)));
    }
    return () => {
      cancelled = true;
    };
  }, [item]);

  const badge =
    item.kind === "wilcoxon"
      ? { tone: "accent" as const, label: "Wilcoxon" }
      : item.kind === "table"
        ? { tone: "muted" as const, label: "Table" }
        : { tone: "primary" as const, label: "Figure" };

  return (
    <div className="flex flex-col rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Badge tone={badge.tone}>{badge.label}</Badge>
          <span className="truncate text-sm font-medium text-fg" title={item.title}>
            {item.title}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="sm" onClick={onEdit} title="Edit in builder">
            <Pencil className="h-3.5 w-3.5" />
          </Button>
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
      <div className="flex min-h-[240px] flex-col items-center justify-center gap-2 p-3">
        {error ? (
          <span className="text-xs text-danger">{error}</span>
        ) : item.kind === "table" ? (
          table ? (
            <div className="w-full">
              <TableView columns={table.columns} rows={table.rows} />
              <p className="mt-1 text-[11px] text-muted-fg">{table.caption}</p>
            </div>
          ) : (
            <span className="text-xs text-muted-fg">computing…</span>
          )
        ) : url ? (
          <img src={url} alt={item.title} className="max-h-[360px] max-w-full object-contain" />
        ) : (
          <span className="text-xs text-muted-fg">rendering…</span>
        )}
      </div>
    </div>
  );
}
