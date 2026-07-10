import { useEffect, useMemo, useState } from "react";
import { GripVertical, RotateCcw, Save } from "lucide-react";
import { api } from "../lib/api";
import type { Metadata } from "../lib/types";
import { cn } from "../lib/util";
import { Button, Section } from "../components/ui";
import { ColorWheel } from "../components/ColorWheel";
import { ModelLegend } from "../components/ModelLegend";
import { PageHeader, InlineNote } from "../components/common";

const ALL_DATASETS = ["vessmap", "drive", "dca1", "octa2d"];

/** Put saved-order names first (in that order), append the rest in place. */
function ordered(names: string[], saved: string[] | undefined): string[] {
  const set = new Set(names);
  const out: string[] = [];
  const seen = new Set<string>();
  for (const n of saved ?? []) {
    if (set.has(n) && !seen.has(n)) {
      out.push(n);
      seen.add(n);
    }
  }
  for (const n of names) {
    if (!seen.has(n)) {
      out.push(n);
      seen.add(n);
    }
  }
  return out;
}

/** Move the item at `from` to position `to` (used by drag-and-drop). */
function reorder(list: string[], from: number, to: number): string[] {
  if (from === to || from < 0 || to < 0 || from >= list.length || to >= list.length) return list;
  const next = [...list];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}

export default function Appearance() {
  const [colors, setColors] = useState<Record<string, string>>({});
  const [defaults, setDefaults] = useState<Record<string, string>>({});
  const [modelOrder, setModelOrder] = useState<string[]>([]);
  const [datasetOrder, setDatasetOrder] = useState<string[]>([]);
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    Promise.all([api.getStyle(), api.metadata("per_run")])
      .then(([style, m]) => {
        setColors(style.colors ?? {});
        setDefaults(style.defaults ?? {});
        setMeta(m);
        const models = (m.options["model_type"]?.values ?? []).map(String);
        setModelOrder(ordered(models, style.order?.model_type));
        setDatasetOrder(ordered(ALL_DATASETS, style.order?.dataset));
      })
      .catch((e) => setMsg(`Failed to load: ${e}`));
  }, []);

  const modelNames = useMemo(
    () => (meta?.options["model_type"]?.values ?? []).map(String),
    [meta],
  );

  function setColor(name: string, hex: string) {
    setColors((c) => ({ ...c, [name]: hex }));
    setMsg(null);
  }
  function clearColor(name: string) {
    setColors((c) => {
      const next = { ...c };
      delete next[name];
      return next;
    });
    setMsg(null);
  }

  async function save() {
    setSaving(true);
    setMsg(null);
    try {
      await api.saveStyle({
        colors,
        order: { model_type: modelOrder, dataset: datasetOrder },
      });
      setMsg("Saved — these colors and order now apply to every chart.");
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Appearance"
        subtitle="One color and order per model/dataset, applied consistently to every chart."
        actions={
          <Button variant="primary" onClick={save} disabled={saving}>
            <Save className="h-4 w-4" /> Save
          </Button>
        }
      />

      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto max-w-2xl">
          <Section title="Models / methods" defaultOpen>
            <ReorderList
              names={modelOrder}
              colors={colors}
              defaults={defaults}
              onColor={setColor}
              onClear={clearColor}
              onReorder={(from, to) => setModelOrder((l) => reorder(l, from, to))}
            />
            {modelNames.length === 0 && (
              <InlineNote>No model types found — run a rescan in Chart Builder first.</InlineNote>
            )}
          </Section>

          <Section title="Datasets" defaultOpen>
            <ReorderList
              names={datasetOrder}
              colors={colors}
              defaults={defaults}
              onColor={setColor}
              onClear={clearColor}
              onReorder={(from, to) => setDatasetOrder((l) => reorder(l, from, to))}
            />
          </Section>

          <Section title="Model nomenclature (legend)" defaultOpen={false}>
            <ModelLegend />
          </Section>

          {msg && (
            <div className="pt-3">
              <InlineNote tone={msg.startsWith("Saved") ? "success" : "danger"}>{msg}</InlineNote>
            </div>
          )}
          <p className="pt-3 text-[11px] text-muted-fg">
            Drag the <GripVertical className="inline h-3 w-3 align-text-bottom" /> handle to reorder.
            A manually set color always wins over a chart's seaborn palette; models left unset fall
            back to the paper palette (<code>model_colors.yaml</code>). Order drives the legend, line
            order and the grouped-bar clusters/categories.
          </p>
        </div>
      </div>
    </div>
  );
}

function ReorderList({
  names,
  colors,
  defaults,
  onColor,
  onClear,
  onReorder,
}: {
  names: string[];
  colors: Record<string, string>;
  defaults: Record<string, string>;
  onColor: (name: string, hex: string) => void;
  onClear: (name: string) => void;
  onReorder: (from: number, to: number) => void;
}) {
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [overIdx, setOverIdx] = useState<number | null>(null);

  return (
    <div className="space-y-1.5">
      {names.map((name, i) => {
        const custom = colors[name];
        const shown = custom ?? defaults[name];
        const dragging = dragIdx === i;
        const isOver = overIdx === i && dragIdx !== null && dragIdx !== i;
        return (
          <div
            key={name}
            draggable
            onDragStart={(e) => {
              setDragIdx(i);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
              if (overIdx !== i) setOverIdx(i);
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (dragIdx !== null && dragIdx !== i) onReorder(dragIdx, i);
              setDragIdx(null);
              setOverIdx(null);
            }}
            onDragEnd={() => {
              setDragIdx(null);
              setOverIdx(null);
            }}
            className={cn(
              "flex items-center gap-2 rounded-lg border bg-surface-2 px-2 py-1.5 transition-colors",
              dragging ? "border-primary opacity-50" : "border-border",
              isOver && "border-primary ring-1 ring-primary",
            )}
          >
            <span
              title="Drag to reorder"
              className="cursor-grab text-muted-fg active:cursor-grabbing"
            >
              <GripVertical className="h-4 w-4" />
            </span>
            <span className="w-5 text-right font-mono text-[11px] text-muted-fg">{i + 1}</span>
            <ColorWheel color={shown} onChange={(hex) => onColor(name, hex)} />
            <span className="flex-1 truncate text-sm text-fg" title={name}>
              {name}
            </span>
            {custom && (
              <button
                title="Reset to default color"
                onClick={() => onClear(name)}
                className="rounded p-1 text-muted-fg hover:bg-muted hover:text-fg cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
