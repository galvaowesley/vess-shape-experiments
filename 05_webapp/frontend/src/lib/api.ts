import type {
  DashboardItem,
  Metadata,
  PaletteCatalog,
  PlotSpec,
  TrainingOptions,
  TrainingRequest,
  TrainingStatus,
  WilcoxonSpec,
} from "./types";

const BASE = "/api";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** POST a spec, return an object-URL for the rendered image (svg/png/...). */
async function renderToBlobUrl(path: string, spec: unknown): Promise<string> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export const api = {
  // --- eval ---
  metadata: (source: string, datasets?: string[]) => {
    const q = new URLSearchParams({ source });
    if (datasets?.length) q.set("datasets", datasets.join(","));
    return fetch(`${BASE}/eval/metadata?${q}`).then((r) => jsonOrThrow<Metadata>(r));
  },
  palettes: () => fetch(`${BASE}/eval/palettes`).then((r) => jsonOrThrow<PaletteCatalog>(r)),
  renderFigure: (spec: PlotSpec) => renderToBlobUrl("/eval/render", spec),
  exportFigure: (spec: PlotSpec) =>
    fetch(`${BASE}/eval/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }),
  rescan: (datasets?: string[], per_image = true) =>
    fetch(`${BASE}/eval/rescan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ datasets, per_image }),
    }).then((r) => jsonOrThrow<Record<string, unknown>>(r)),

  // --- wilcoxon ---
  renderWilcoxon: (spec: WilcoxonSpec) => renderToBlobUrl("/stats/wilcoxon/render", spec),
  wilcoxonData: (spec: WilcoxonSpec) =>
    fetch(`${BASE}/stats/wilcoxon/data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<Record<string, unknown>>(r)),

  // --- dashboard ---
  listDashboard: () => fetch(`${BASE}/dashboard`).then((r) => jsonOrThrow<DashboardItem[]>(r)),
  pin: (item: DashboardItem) =>
    fetch(`${BASE}/dashboard`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    }).then((r) => jsonOrThrow<DashboardItem>(r)),
  unpin: (id: string) =>
    fetch(`${BASE}/dashboard/${id}`, { method: "DELETE" }).then((r) =>
      jsonOrThrow<Record<string, unknown>>(r),
    ),
  reorder: (order: string[]) =>
    fetch(`${BASE}/dashboard/reorder`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order }),
    }).then((r) => jsonOrThrow<DashboardItem[]>(r)),
  exportDashboard: (out_dir: string, format: string, dpi: number) =>
    fetch(`${BASE}/dashboard/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out_dir, format, dpi }),
    }).then((r) => jsonOrThrow<Record<string, unknown>>(r)),

  // --- training ---
  trainingOptions: () =>
    fetch(`${BASE}/training/options`).then((r) => jsonOrThrow<TrainingOptions>(r)),
  trainingConfig: (req: TrainingRequest) =>
    fetch(`${BASE}/training/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => jsonOrThrow<{ yaml: string; filename: string }>(r)),
  trainingSave: (req: TrainingRequest, overwrite: boolean) =>
    fetch(`${BASE}/training/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: req, overwrite }),
    }).then((r) => jsonOrThrow<Record<string, unknown>>(r)),
  trainingRun: (req: TrainingRequest) =>
    fetch(`${BASE}/training/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => jsonOrThrow<TrainingStatus>(r)),
  trainingStop: () =>
    fetch(`${BASE}/training/stop`, { method: "POST" }).then((r) =>
      jsonOrThrow<Record<string, unknown>>(r),
    ),
  trainingStatus: () =>
    fetch(`${BASE}/training/status`).then((r) => jsonOrThrow<TrainingStatus>(r)),
};

export function trainingLogSocket(): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${location.host}/api/training/logs`);
}
