import { useEffect, useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { api, figureImgUrl } from "../../lib/api";
import type { FigureOptions, PanelRef, RankedImage, RankedRun } from "../../lib/types";
import { cn, useDebounced } from "../../lib/util";
import { Badge, Card, ChipMultiSelect, Field, Segmented, Select } from "../ui";
import { InlineNote } from "../common";

type Tab = "images" | "runs";
type Agg = "mean" | "median" | "min" | "max" | "spread";

const AGG_OPTIONS: { value: Agg; label: string }[] = [
  { value: "spread", label: "Spread (disagreement)" },
  { value: "mean", label: "Mean" },
  { value: "median", label: "Median" },
  { value: "min", label: "Min" },
  { value: "max", label: "Max" },
];

const MAX_ROWS = 120;

const nsLabel = (n: number) => (n === 0 ? "Zero-shot" : `${n}-shot`);

/** Thumbnail box with a dark backdrop — masks are white-on-black, so without
 *  this a black mask reads as a hole in the layout rather than a panel. */
function Thumb({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded bg-black">
      {src && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          className="h-full w-full object-contain"
        />
      )}
    </div>
  );
}

function SearchBox({
  value,
  onChange,
  onEnter,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onEnter?: () => void;
  placeholder: string;
}) {
  return (
    <div className="relative">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-fg" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onEnter?.();
          if (e.key === "Escape") onChange("");
        }}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete="off"
        className="w-full rounded-lg border border-border bg-surface-2 py-1.5 pl-8 pr-7 text-sm text-fg placeholder:text-muted-fg/60 focus:outline-none focus:ring-2 focus:ring-ring"
      />
      {value && (
        <button
          onClick={() => onChange("")}
          title="Clear (Esc)"
          className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-muted-fg hover:bg-muted hover:text-fg cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

export function InferenceBrowser({
  options,
  dataset,
  onDatasetChange,
  image,
  onImageChange,
  onPick,
  selectionLabel,
}: {
  options: FigureOptions | null;
  dataset: string;
  onDatasetChange: (ds: string) => void;
  image: string;
  onImageChange: (image: string) => void;
  onPick: (panel: PanelRef) => void;
  selectionLabel?: string;
}) {
  const [tab, setTab] = useState<Tab>("images");
  const dsOpts = options?.by_dataset?.[dataset];

  // metric is shared by both tabs — it's the same "which score to rank by" concept
  const [metric, setMetric] = useState("Dice");
  useEffect(() => {
    if (options && options.metrics.length && !options.metrics.includes(metric)) {
      setMetric(options.metrics[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options]);

  // --- Images tab ---
  const [numSamples, setNumSamples] = useState<number[]>([]);
  const [modelTypes, setModelTypes] = useState<string[]>([]);
  const [agg, setAgg] = useState<Agg>("spread");
  const [rankedImages, setRankedImages] = useState<RankedImage[]>([]);
  const [imgLoading, setImgLoading] = useState(false);
  const [imgError, setImgError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  // dataset-scoped filters don't carry over to a different dataset's option set
  useEffect(() => {
    setNumSamples([]);
    setModelTypes([]);
    setQuery("");
  }, [dataset]);

  const imagesFiltersRaw = useMemo(
    () => ({ dataset, numSamples, modelTypes, metric, agg }),
    [dataset, numSamples.join(","), modelTypes.join(","), metric, agg],
  );
  const imagesFilters = useDebounced(imagesFiltersRaw, 250);

  useEffect(() => {
    if (tab !== "images" || !imagesFilters.dataset) return;
    let cancelled = false;
    setImgLoading(true);
    setImgError(null);
    api
      .rankImages({
        dataset: imagesFilters.dataset,
        num_samples: imagesFilters.numSamples,
        model_types: imagesFilters.modelTypes,
        stages: [],
        metric: imagesFilters.metric,
        agg: imagesFilters.agg,
        ascending: false,
        limit: MAX_ROWS,
      })
      .then((r) => !cancelled && setRankedImages(r))
      .catch((e) => !cancelled && setImgError(String(e.message ?? e)))
      .finally(() => !cancelled && setImgLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tab, imagesFilters]);

  // --- Runs tab ---
  const [runModelType, setRunModelType] = useState("");
  const [runNumSamples, setRunNumSamples] = useState<number | null>(null);
  const [rankedRuns, setRankedRuns] = useState<RankedRun[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);

  // keep model/num_samples selections valid as the dataset changes
  useEffect(() => {
    if (!dsOpts) return;
    if (!runModelType || !dsOpts.model_types.includes(runModelType)) {
      setRunModelType(dsOpts.model_types[0] ?? "");
    }
    if (runNumSamples == null || !dsOpts.num_samples.includes(runNumSamples)) {
      setRunNumSamples(dsOpts.num_samples[0] ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dsOpts, dataset]);

  const runsFiltersRaw = useMemo(
    () => ({ dataset, image, runModelType, runNumSamples, metric }),
    [dataset, image, runModelType, runNumSamples, metric],
  );
  const runsFilters = useDebounced(runsFiltersRaw, 250);

  useEffect(() => {
    if (tab !== "runs" || !runsFilters.image || !runsFilters.runModelType) return;
    let cancelled = false;
    setRunsLoading(true);
    setRunsError(null);
    api
      .rankRuns({
        dataset: runsFilters.dataset,
        image: runsFilters.image,
        model_type: runsFilters.runModelType,
        num_samples: runsFilters.runNumSamples,
        metric: runsFilters.metric,
        ascending: false,
        limit: MAX_ROWS,
      })
      .then((r) => !cancelled && setRankedRuns(r))
      .catch((e) => !cancelled && setRunsError(String(e.message ?? e)))
      .finally(() => !cancelled && setRunsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tab, runsFilters]);

  function buildPanel(rr: RankedRun): PanelRef {
    return {
      kind: "pred",
      dataset,
      image,
      stage: rr.stage,
      experiment: rr.experiment,
      run_name: rr.run_name,
      model_type: rr.model_type,
      num_samples: rr.num_samples,
      run: rr.run ?? null,
      rep: rr.rep ?? null,
      score: rr.score,
    };
  }

  const q = query.trim().toLowerCase();

  const matchedImages = useMemo(
    () => (q ? rankedImages.filter((ri) => ri.image.toLowerCase().includes(q)) : rankedImages),
    [rankedImages, q],
  );

  /** Names that exist in the dataset but are absent from the current ranking —
   *  either filtered out by the model/num_samples chips or never evaluated. Without
   *  this a search for a real image name would look like the image doesn't exist. */
  const unrankedMatches = useMemo(() => {
    if (!q) return [];
    const ranked = new Set(rankedImages.map((ri) => ri.image));
    return (dsOpts?.images ?? [])
      .filter((name) => !ranked.has(name) && name.toLowerCase().includes(q))
      .slice(0, MAX_ROWS);
  }, [q, rankedImages, dsOpts]);

  const shownImages = matchedImages.slice(0, MAX_ROWS);
  const shownRuns = rankedRuns.slice(0, MAX_ROWS);

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-3 py-2">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: "images", label: "Images" },
            { value: "runs", label: "Runs" },
          ]}
        />
        <div className="w-36 shrink-0">
          <Select
            value={dataset}
            onChange={onDatasetChange}
            options={(options?.datasets ?? []).map((d) => ({ value: d, label: d }))}
          />
        </div>
      </div>

      {selectionLabel && (
        <div className="shrink-0 border-b border-border bg-surface-2 px-3 py-1.5 text-[11px] text-muted-fg">
          Picking for <span className="font-medium text-fg">{selectionLabel}</span>
        </div>
      )}

      <div className="shrink-0 space-y-3 border-b border-border p-3">
        {tab === "images" ? (
          <>
            <SearchBox
              value={query}
              onChange={setQuery}
              onEnter={() => {
                const top = shownImages[0]?.image ?? unrankedMatches[0];
                if (top) onImageChange(top);
              }}
              placeholder="Search image name…  (Enter = pick first)"
            />
            <Field label="Num. samples (empty = all)">
              <ChipMultiSelect
                options={(dsOpts?.num_samples ?? []).map(nsLabel)}
                selected={numSamples.map(nsLabel)}
                onChange={(labels) => {
                  const set = new Set(labels);
                  setNumSamples((dsOpts?.num_samples ?? []).filter((n) => set.has(nsLabel(n))));
                }}
              />
            </Field>
            <Field label="Models (empty = all)">
              <ChipMultiSelect
                options={dsOpts?.model_types ?? []}
                selected={modelTypes}
                onChange={setModelTypes}
              />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Metric">
                <Select
                  value={metric}
                  onChange={setMetric}
                  options={(options?.metrics ?? []).map((m) => ({ value: m, label: m }))}
                />
              </Field>
              <Field label="Aggregate">
                <Select value={agg} onChange={(v) => setAgg(v as Agg)} options={AGG_OPTIONS} />
              </Field>
            </div>
            {agg === "spread" && (
              <InlineNote>
                Spread surfaces the images where models disagree most — usually the most
                informative choice for a qualitative figure.
              </InlineNote>
            )}
          </>
        ) : (
          <>
            <div className="text-xs text-muted-fg">
              Featured image:{" "}
              <span className="font-medium text-fg">{image || "(none selected)"}</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Model">
                <Select
                  value={runModelType}
                  onChange={setRunModelType}
                  options={(dsOpts?.model_types ?? []).map((m) => ({ value: m, label: m }))}
                />
              </Field>
              <Field label="Num. samples">
                <Select
                  value={runNumSamples == null ? "" : String(runNumSamples)}
                  onChange={(v) => setRunNumSamples(v === "" ? null : Number(v))}
                  options={(dsOpts?.num_samples ?? []).map((n) => ({
                    value: String(n),
                    label: nsLabel(n),
                  }))}
                />
              </Field>
            </div>
          </>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-2">
        {tab === "images" ? (
          imgError ? (
            <InlineNote tone="danger">{imgError}</InlineNote>
          ) : shownImages.length === 0 && unrankedMatches.length === 0 ? (
            <InlineNote>
              {imgLoading
                ? "Ranking images…"
                : q
                  ? `No image name matches “${query.trim()}”.`
                  : "No images ranked yet."}
            </InlineNote>
          ) : (
            <div className="space-y-1.5">
              {shownImages.map((ri, i) => (
                <button
                  key={ri.image}
                  onClick={() => onImageChange(ri.image)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg border px-2.5 py-2 text-left transition-colors cursor-pointer",
                    ri.image === image
                      ? "border-primary bg-primary/10"
                      : "border-border bg-surface-2 hover:border-primary/50",
                  )}
                >
                  <span className="w-5 shrink-0 text-right font-mono text-xs text-muted-fg">
                    {i + 1}
                  </span>
                  <Thumb
                    src={figureImgUrl({ kind: "gt", dataset, image: ri.image }, 96)}
                    alt={ri.image}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium text-fg">{ri.image}</div>
                    <div className="truncate text-[11px] text-muted-fg">
                      {Object.entries(ri.per_model)
                        .map(([m, v]) => `${m}: ${v.toFixed(3)}`)
                        .join(" · ")}
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="font-mono text-xs text-fg">{ri.score.toFixed(3)}</div>
                    <div className="text-[10px] text-muted-fg">n={ri.n}</div>
                  </div>
                </button>
              ))}
              {matchedImages.length > MAX_ROWS && (
                <InlineNote>
                  Showing first {MAX_ROWS} of {matchedImages.length} images.
                </InlineNote>
              )}

              {unrankedMatches.length > 0 && (
                <>
                  <div className="px-1 pt-2 text-[11px] font-medium text-muted-fg">
                    Matches outside the current ranking
                  </div>
                  {unrankedMatches.map((name) => (
                    <button
                      key={`unranked:${name}`}
                      onClick={() => onImageChange(name)}
                      className={cn(
                        "flex w-full items-center gap-3 rounded-lg border border-dashed px-2.5 py-2 text-left transition-colors cursor-pointer",
                        name === image
                          ? "border-primary bg-primary/10"
                          : "border-border bg-surface-2 hover:border-primary/50",
                      )}
                    >
                      <span className="w-5 shrink-0" />
                      <Thumb src={figureImgUrl({ kind: "gt", dataset, image: name }, 96)} alt={name} />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium text-fg">{name}</div>
                        <div className="text-[11px] text-muted-fg">
                          not scored under the current filters
                        </div>
                      </div>
                    </button>
                  ))}
                </>
              )}
            </div>
          )
        ) : runsError ? (
          <InlineNote tone="danger">{runsError}</InlineNote>
        ) : !image ? (
          <InlineNote>Pick an image in the Images tab first.</InlineNote>
        ) : shownRuns.length === 0 ? (
          <InlineNote>{runsLoading ? "Ranking runs…" : "No runs ranked yet."}</InlineNote>
        ) : (
          <div className="space-y-1.5">
            {shownRuns.map((rr, i) => {
              const panel = buildPanel(rr);
              return (
                <button
                  key={`${rr.run_name}:${rr.run ?? ""}:${rr.rep ?? ""}`}
                  onClick={() => onPick(panel)}
                  className="flex w-full items-center gap-3 rounded-lg border border-border bg-surface-2 px-2.5 py-2 text-left transition-colors hover:border-primary/50 cursor-pointer"
                >
                  <span className="w-5 shrink-0 text-right font-mono text-xs text-muted-fg">
                    {i + 1}
                  </span>
                  <Thumb src={figureImgUrl(panel, 96)} alt={rr.run_name} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium text-fg">
                      run:{rr.run ?? "—"} rep:{rr.rep ?? "—"}
                    </div>
                    <div className="flex gap-1">
                      {rr.is_best && <Badge tone="primary">best</Badge>}
                      {rr.is_median && <Badge tone="accent">median</Badge>}
                    </div>
                  </div>
                  <div className="shrink-0 font-mono text-xs text-fg">{rr.score.toFixed(3)}</div>
                </button>
              );
            })}
            {rankedRuns.length > MAX_ROWS && (
              <InlineNote>
                Showing first {MAX_ROWS} of {rankedRuns.length} runs.
              </InlineNote>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
