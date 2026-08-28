import type {
  AutoFillRequest,
  DashboardItem,
  DashboardMeta,
  FigureOptions,
  FigureSettings,
  FsListing,
  GridFigureSpec,
  GridLayout,
  Metadata,
  PaletteCatalog,
  PanelRef,
  PlotSpec,
  RankImagesRequest,
  RankRunsRequest,
  RankedImage,
  RankedRun,
  StylePrefs,
  StylePrefsResponse,
  TableResult,
  TableSpec,
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
  getStyle: () => fetch(`${BASE}/eval/style`).then((r) => jsonOrThrow<StylePrefsResponse>(r)),
  saveStyle: (prefs: StylePrefs) =>
    fetch(`${BASE}/eval/style`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prefs),
    }).then((r) => jsonOrThrow<StylePrefs>(r)),
  renderTable: (spec: TableSpec) =>
    fetch(`${BASE}/eval/table`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<TableResult>(r)),
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
  listDashboards: () =>
    fetch(`${BASE}/dashboard/dashboards`).then((r) => jsonOrThrow<DashboardMeta[]>(r)),
  createDashboard: (name: string) =>
    fetch(`${BASE}/dashboard/dashboards`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => jsonOrThrow<DashboardMeta>(r)),
  renameDashboard: (id: string, name: string) =>
    fetch(`${BASE}/dashboard/dashboards/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => jsonOrThrow<DashboardMeta>(r)),
  deleteDashboard: (id: string) =>
    fetch(`${BASE}/dashboard/dashboards/${id}`, { method: "DELETE" }).then((r) =>
      jsonOrThrow<Record<string, unknown>>(r),
    ),
  listDashboard: (dashboardId?: string) => {
    const q = dashboardId ? `?dashboard_id=${encodeURIComponent(dashboardId)}` : "";
    return fetch(`${BASE}/dashboard${q}`).then((r) => jsonOrThrow<DashboardItem[]>(r));
  },
  pin: (item: DashboardItem) =>
    fetch(`${BASE}/dashboard`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    }).then((r) => jsonOrThrow<DashboardItem>(r)),
  updatePin: (id: string, patch: Partial<DashboardItem>) =>
    fetch(`${BASE}/dashboard/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
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
  exportDashboard: (out_dir: string, format: string, dpi: number, dashboard_id?: string) =>
    fetch(`${BASE}/dashboard/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out_dir, format, dpi, dashboard_id }),
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

  /** Generic GET helper for new endpoints (monitor panel). */
  get: <T>(path: string) => fetch(path).then((r) => jsonOrThrow<T>(r)),

  // --- filesystem browser (save/export destination picker) ---
  browseFs: (path?: string) => {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    return fetch(`${BASE}/fs/browse${q}`).then((r) => jsonOrThrow<FsListing>(r));
  },
  makeDir: (path: string, name: string) =>
    fetch(`${BASE}/fs/mkdir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, name }),
    }).then((r) => jsonOrThrow<FsListing>(r)),
  writeFile: (path: string, content: string) =>
    fetch(`${BASE}/fs/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, content }),
    }).then((r) => jsonOrThrow<{ saved: boolean; path: string }>(r)),

  // --- figure studio ---
  figureOptions: () => fetch(`${BASE}/figure/options`).then((r) => jsonOrThrow<FigureOptions>(r)),
  figureSettings: () => fetch(`${BASE}/figure/settings`).then((r) => jsonOrThrow<FigureSettings>(r)),
  setFigureSettings: (dataset_root: string | null) =>
    fetch(`${BASE}/figure/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_root }),
    }).then((r) => jsonOrThrow<FigureSettings>(r)),
  rankImages: (req: RankImagesRequest) =>
    fetch(`${BASE}/figure/rank/images`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => jsonOrThrow<RankedImage[]>(r)),
  rankRuns: (req: RankRunsRequest) =>
    fetch(`${BASE}/figure/rank/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => jsonOrThrow<RankedRun[]>(r)),
  autoFill: (req: AutoFillRequest) =>
    fetch(`${BASE}/figure/autofill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => jsonOrThrow<GridFigureSpec>(r)),
  renderGrid: (spec: GridFigureSpec) => renderToBlobUrl("/figure/render", spec),
  exportGrid: (spec: GridFigureSpec) =>
    fetch(`${BASE}/figure/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }),
  /** Each panel as its own file — a zip when downloading, files on disk when
   *  the spec carries a save_path. */
  exportGridPanels: (spec: GridFigureSpec, which: "refs" | "all", naming: "full" | "label") =>
    fetch(`${BASE}/figure/export/panels?which=${which}&naming=${naming}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }),
  listLayouts: () => fetch(`${BASE}/figure/layouts`).then((r) => jsonOrThrow<GridLayout[]>(r)),
  saveLayout: (name: string, spec: GridFigureSpec) =>
    fetch(`${BASE}/figure/layouts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, spec }),
    }).then((r) => jsonOrThrow<GridLayout>(r)),
  updateLayout: (id: string, patch: { name?: string; spec?: GridFigureSpec }) =>
    fetch(`${BASE}/figure/layouts/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then((r) => jsonOrThrow<GridLayout>(r)),
  deleteLayout: (id: string) =>
    fetch(`${BASE}/figure/layouts/${id}`, { method: "DELETE" }).then((r) =>
      jsonOrThrow<Record<string, unknown>>(r),
    ),
};

export function trainingLogSocket(): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${location.host}/api/training/logs`);
}

/** Plain URL builder for <img src> (a GET, unlike the POST-for-blob helpers above). */
export function figureImgUrl(p: PanelRef, width?: number): string {
  if (p.kind === "empty") return "";
  const q = new URLSearchParams({ kind: p.kind, dataset: p.dataset, image: p.image });
  // Run identity only resolves a prediction. Leaving it on an input/GT request
  // would give every panel of the same image a different URL and so a separate
  // browser cache entry, for a byte-identical response.
  if (p.kind === "pred") {
    if (p.stage != null) q.set("stage", p.stage);
    if (p.experiment != null) q.set("experiment", p.experiment);
    if (p.run_name != null) q.set("run_name", p.run_name);
  }
  if (width != null) q.set("w", String(width));
  return `${BASE}/figure/img?${q}`;
}
