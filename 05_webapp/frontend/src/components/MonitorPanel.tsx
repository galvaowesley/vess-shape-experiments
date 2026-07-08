import { useEffect, useMemo, useRef, useState } from "react";
import {
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../lib/api";
import { LogConsole } from "./LogConsole";

// ─── Types ────────────────────────────────────────────────────────────────────

interface NsProgress {
  num_samples: number;
  done: number;
  total: number;
  pct: number;
}

interface Progress {
  overall_pct: number;
  done_steps: number;
  total_steps: number;
  current_ns: number | null;
  current_run: number | null;
  current_rep: number | null;
  per_ns: NsProgress[];
}

interface EpochPoint {
  run_id: string;
  num_samples: number;
  epoch: number;
  total_epochs: number;
  [metric: string]: number | string | null;
}

interface SystemCurrent {
  cpu_pct: number | null;
  cpu_temp: number | null;
  cpu_cores: number[];
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  ram_pct: number | null;
  gpu_temp: number | null;
  gpu_util: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  vram_pct: number | null;
  gpu_power_w: number | null;
  gpu_power_limit_w: number | null;
}

// ─── Palette for epoch metric series ─────────────────────────────────────────

const SERIES_COLORS = [
  "var(--primary)",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#06b6d4",
  "#f97316",
  "#84cc16",
];

// ─── Color helpers ────────────────────────────────────────────────────────────

/** Utilization-based color: green → amber → red. */
function utilColor(pct: number): string {
  if (pct >= 70) return "#ef4444";
  if (pct >= 40) return "#f59e0b";
  return "#10b981";
}

/** Temperature color. */
function tempColor(temp: number, warnAt = 60, dangerAt = 80): string {
  if (temp >= dangerAt) return "var(--danger)";
  if (temp >= warnAt) return "var(--accent)";
  return "var(--primary)";
}

// ─── Shared bar component ─────────────────────────────────────────────────────

function HorizBar({
  label,
  pct,
  text,
  color,
}: {
  label: string;
  pct: number;
  text: string;
  color?: string;
}) {
  const c = color ?? utilColor(pct);
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="flex items-center gap-2">
      <span className="w-10 shrink-0 text-[11px] text-muted-fg">{label}</span>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${clamped}%`, background: c }}
        />
      </div>
      <span className="w-28 shrink-0 text-right font-mono text-[11px] text-fg">{text}</span>
    </div>
  );
}

// ─── CPU core grid (btop-style) ───────────────────────────────────────────────

function CoreGrid({ cores }: { cores: number[] }) {
  if (cores.length === 0) return null;
  // Layout: up to 8 per row
  const cols = cores.length <= 8 ? cores.length : 8;
  return (
    <div
      className="grid gap-1 mt-1"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {cores.map((pct, i) => {
        const bg = utilColor(pct);
        return (
          <div
            key={i}
            title={`Core ${i}: ${pct}%`}
            className="flex flex-col items-center justify-center rounded py-1 select-none"
            style={{ background: bg + "22", border: `1px solid ${bg}44` }}
          >
            <span className="font-mono text-[8px] leading-none" style={{ color: bg }}>
              C{i}
            </span>
            <span
              className="font-mono text-[10px] font-bold leading-tight"
              style={{ color: bg }}
            >
              {pct.toFixed(0)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Progress bar ─────────────────────────────────────────────────────────────

function ProgressBar({ pct, done, total }: { pct: number; done: number; total: number }) {
  const color = pct >= 100 ? "var(--success)" : "var(--primary)";
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="w-16 text-right font-mono text-[11px] text-muted-fg">
        {done}/{total}
      </span>
      <span className="w-9 text-right font-mono text-[11px] text-muted-fg">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

// ─── Epoch metrics chart ──────────────────────────────────────────────────────

const KNOWN_METRICS = ["Dice", "Train loss", "Validation loss", "IoU", "Accuracy", "Precision", "Recall"];

function EpochChart({ points }: { points: EpochPoint[] }) {
  const availableMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const p of points) {
      for (const k of Object.keys(p)) {
        if (!["run_id", "num_samples", "epoch", "total_epochs"].includes(k)) {
          keys.add(k);
        }
      }
    }
    const known = KNOWN_METRICS.filter((m) => keys.has(m));
    const others = [...keys].filter((m) => !KNOWN_METRICS.includes(m));
    return [...known, ...others];
  }, [points]);

  const [activeMetrics, setActiveMetrics] = useState<string[]>([]);

  useEffect(() => {
    if (availableMetrics.length > 0 && activeMetrics.length === 0) {
      setActiveMetrics([availableMetrics[0]]);
    }
  }, [availableMetrics]);

  const chartData = useMemo(() => {
    if (activeMetrics.length === 0) return [];
    const byEpoch = new Map<number, Record<string, number>>();
    for (const p of points) {
      if (!byEpoch.has(p.epoch)) byEpoch.set(p.epoch, { epoch: p.epoch });
      const row = byEpoch.get(p.epoch)!;
      for (const m of activeMetrics) {
        if (typeof p[m] === "number") {
          const key = `ns${p.num_samples}_${m.replace(/ /g, "_")}`;
          row[key] = row[key] == null ? (p[m] as number) : (row[key] + (p[m] as number)) / 2;
        }
      }
    }
    return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  }, [points, activeMetrics]);

  const seriesKeys = useMemo(() => {
    if (chartData.length === 0) return [];
    return Object.keys(chartData[0]).filter((k) => k !== "epoch");
  }, [chartData]);

  const toggleMetric = (m: string) =>
    setActiveMetrics((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m],
    );

  if (points.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-xs text-muted-fg">
        Waiting for epoch data…
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {availableMetrics.map((m) => (
          <button
            key={m}
            onClick={() => toggleMetric(m)}
            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
              activeMetrics.includes(m)
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface-2 text-muted-fg hover:text-fg"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      {activeMetrics.length > 0 && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 10, fill: "var(--muted-fg)" }}
              axisLine={{ stroke: "var(--border)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "var(--muted-fg)" }}
              axisLine={false}
              tickLine={false}
              width={48}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 11,
              }}
              labelFormatter={(v) => `Epoch ${v}`}
            />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} formatter={(v) => v.replace(/_/g, " ")} />
            {seriesKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={key}
                dot={false}
                strokeWidth={1.5}
                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ─── System Resources Panel ───────────────────────────────────────────────────

function SystemResources({ c }: { c: SystemCurrent }) {
  const powerPct =
    c.gpu_power_w != null && c.gpu_power_limit_w
      ? (c.gpu_power_w / c.gpu_power_limit_w) * 100
      : 0;

  return (
    <div className="flex flex-col gap-3">
      {/* CPU */}
      <div className="rounded-lg border border-border bg-surface-2 p-3">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-fg">CPU</span>
          {c.cpu_temp != null && (
            <span
              className="font-mono text-xs font-medium"
              style={{ color: tempColor(c.cpu_temp) }}
            >
              {c.cpu_temp.toFixed(0)} °C
            </span>
          )}
        </div>
        <HorizBar
          label="Overall"
          pct={c.cpu_pct ?? 0}
          text={`${(c.cpu_pct ?? 0).toFixed(1)}%`}
          color={utilColor(c.cpu_pct ?? 0)}
        />
        <CoreGrid cores={c.cpu_cores ?? []} />
      </div>

      {/* GPU + RAM side by side */}
      <div className="grid grid-cols-2 gap-3">
        {/* GPU */}
        <div className="rounded-lg border border-border bg-surface-2 p-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-fg">GPU</span>
            {c.gpu_temp != null && (
              <span
                className="font-mono text-xs font-medium"
                style={{ color: tempColor(c.gpu_temp, 65, 85) }}
              >
                {c.gpu_temp.toFixed(0)} °C
              </span>
            )}
          </div>
          <div className="space-y-1.5">
            <HorizBar
              label="Util"
              pct={c.gpu_util ?? 0}
              text={`${(c.gpu_util ?? 0).toFixed(0)}%`}
              color={utilColor(c.gpu_util ?? 0)}
            />
            <HorizBar
              label="VRAM"
              pct={c.vram_pct ?? 0}
              text={`${c.vram_used_gb?.toFixed(1) ?? "–"} / ${c.vram_total_gb?.toFixed(0) ?? "–"} GB`}
              color={utilColor(c.vram_pct ?? 0)}
            />
            {c.gpu_power_w != null && (
              <HorizBar
                label="Power"
                pct={powerPct}
                text={`${c.gpu_power_w.toFixed(0)} / ${c.gpu_power_limit_w?.toFixed(0) ?? "–"} W`}
                color={utilColor(powerPct)}
              />
            )}
          </div>
        </div>

        {/* RAM */}
        <div className="rounded-lg border border-border bg-surface-2 p-3">
          <div className="mb-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-fg">RAM</span>
          </div>
          <HorizBar
            label="Used"
            pct={c.ram_pct ?? 0}
            text={`${c.ram_used_gb?.toFixed(1) ?? "–"} / ${c.ram_total_gb?.toFixed(0) ?? "–"} GB`}
            color={utilColor(c.ram_pct ?? 0)}
          />
        </div>
      </div>
    </div>
  );
}

// ─── Resizable console divider ────────────────────────────────────────────────

function ResizeHandle({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      onMouseDown={onMouseDown}
      className="group flex h-3 cursor-row-resize items-center justify-center"
      title="Drag to resize console"
    >
      <div className="h-0.5 w-10 rounded-full bg-border transition-colors group-hover:bg-muted-fg" />
    </div>
  );
}

// ─── Main MonitorPanel ────────────────────────────────────────────────────────

export function MonitorPanel({ active }: { active: boolean }) {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [epochPoints, setEpochPoints] = useState<EpochPoint[]>([]);
  const [sys, setSys] = useState<{ current: SystemCurrent } | null>(null);
  const [consoleHeight, setConsoleHeight] = useState(200);
  const consoleDragRef = useRef<{ startY: number; startH: number } | null>(null);

  useEffect(() => {
    if (!active) return;

    const fetchProgress = () =>
      api
        .get<{ progress: Progress; epoch_metrics: EpochPoint[] }>("/api/training/progress")
        .then((d) => {
          setProgress(d.progress);
          setEpochPoints(d.epoch_metrics);
        })
        .catch(() => undefined);

    const fetchSys = () =>
      api
        .get<{ current: SystemCurrent }>("/api/training/system-metrics")
        .then(setSys)
        .catch(() => undefined);

    fetchProgress();
    fetchSys();
    const id1 = setInterval(fetchProgress, 2000);
    const id2 = setInterval(fetchSys, 2000);
    return () => {
      clearInterval(id1);
      clearInterval(id2);
    };
  }, [active]);

  function startConsoleResize(e: React.MouseEvent) {
    e.preventDefault();
    consoleDragRef.current = { startY: e.clientY, startH: consoleHeight };
    const onMove = (me: MouseEvent) => {
      if (!consoleDragRef.current) return;
      const delta = consoleDragRef.current.startY - me.clientY; // drag up → taller
      setConsoleHeight(Math.max(80, Math.min(600, consoleDragRef.current.startH + delta)));
    };
    const onUp = () => {
      consoleDragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  const c = sys?.current;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Scrollable content above console */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">

        {/* ── Experiment Progress ── */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-fg">
            Experiment Progress
          </h3>
          {progress && progress.total_steps > 0 ? (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="w-16 shrink-0 text-xs font-medium text-fg">Overall</span>
                <ProgressBar
                  pct={progress.overall_pct}
                  done={progress.done_steps}
                  total={progress.total_steps}
                />
              </div>
              {progress.per_ns.map((ns) => (
                <div key={ns.num_samples} className="flex items-center gap-2">
                  <span className="w-16 shrink-0 text-xs text-muted-fg">
                    ns={ns.num_samples}
                  </span>
                  <ProgressBar pct={ns.pct} done={ns.done} total={ns.total} />
                </div>
              ))}
              {progress.current_ns != null && (
                <p className="pt-1 text-[11px] text-muted-fg">
                  Active: ns={progress.current_ns}
                  {progress.current_run != null && `, run ${progress.current_run}`}
                  {progress.current_rep != null && `, rep ${progress.current_rep}`}
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-fg">No experiment running.</p>
          )}
        </section>

        {/* ── System Resources ── */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-fg">
            System Resources
          </h3>
          {c ? (
            <SystemResources c={c} />
          ) : (
            <p className="text-xs text-muted-fg">Collecting metrics…</p>
          )}
        </section>

        {/* ── Epoch Metrics ── */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-fg">
            Epoch Metrics
          </h3>
          <EpochChart points={epochPoints} />
        </section>
      </div>

      {/* Resize handle */}
      <ResizeHandle onMouseDown={startConsoleResize} />

      {/* Console — fixed height, resizable by dragging the handle above */}
      <section
        className="flex shrink-0 flex-col border-t border-border"
        style={{ height: consoleHeight }}
      >
        <div className="px-4 py-1.5">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-fg">
            Console
          </span>
        </div>
        <div className="flex-1 overflow-hidden">
          <LogConsole />
        </div>
      </section>
    </div>
  );
}
