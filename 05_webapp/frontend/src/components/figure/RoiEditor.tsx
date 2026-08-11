import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { figureImgUrl } from "../../lib/api";
import type { CropSpec, GridFigureSpec, PanelRef, RoiSpec } from "../../lib/types";
import { defaultRoi, isFullFrame, roiKey } from "../../lib/figureSpec";
import { Button, Checkbox, ChipMultiSelect, Field, NumberInput, Segmented, Select, Slider } from "../ui";
import { InlineNote } from "../common";
import { ColorWheel } from "../ColorWheel";

const MIN_ROI = 0.02;
const THUMB_W = 384;

const INSET_CORNERS: { value: RoiSpec["inset_corner"]; label: string }[] = [
  { value: "upper left", label: "Upper left" },
  { value: "upper right", label: "Upper right" },
  { value: "lower left", label: "Lower left" },
  { value: "lower right", label: "Lower right" },
];

const SCOPE_NOUN: Record<GridFigureSpec["roi_scope"], string> = {
  image: "panel(s) showing this image",
  column: "panel(s) in this column",
  panel: "this panel only",
  figure: "panel(s) in the whole figure",
};

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** Clamp a rectangle to a sane, always-visible minimum and keep it inside [0,1]. */
function clampRect(x: number, y: number, w: number, h: number): { x: number; y: number; w: number; h: number } {
  const cw = Math.max(w, MIN_ROI);
  const ch = Math.max(h, MIN_ROI);
  const cx = Math.min(Math.max(x, 0), 1 - cw);
  const cy = Math.min(Math.max(y, 0), 1 - ch);
  return { x: cx, y: cy, w: cw, h: ch };
}

/** Friendly label for a ROI group key, given the current scope. */
function friendlyKeyLabel(spec: GridFigureSpec, index: number, key: string): string {
  switch (spec.roi_scope) {
    case "column": {
      const c = index % spec.cols;
      return spec.col_labels[c]?.trim() || `Column ${c + 1}`;
    }
    case "panel":
      return `Row ${Math.floor(index / spec.cols) + 1} · Col ${(index % spec.cols) + 1}`;
    case "image":
    case "figure":
    default:
      return key;
  }
}

function otherKeys(spec: GridFigureSpec, currentKey: string): { key: string; label: string }[] {
  const seen = new Map<string, string>();
  spec.panels.forEach((p, i) => {
    if (p.kind === "empty") return;
    const k = roiKey(spec, p, i);
    if (k === currentKey || seen.has(k)) return;
    seen.set(k, friendlyKeyLabel(spec, i, k));
  });
  return Array.from(seen, ([key, label]) => ({ key, label }));
}

type DrawOn = "input" | "gt" | "panel";
/** Which rectangle a drag writes to: the ROI marker or the pre-ROI zoom. */
type DrawTarget = "roi" | "crop";

const DRAW_ON: { value: DrawOn; label: string }[] = [
  { value: "input", label: "Input" },
  { value: "gt", label: "GT" },
  { value: "panel", label: "Panel" },
];

export function RoiEditor({
  spec,
  panel,
  index,
  onChange,
}: {
  spec: GridFigureSpec;
  panel: PanelRef | null;
  index: number | null;
  onChange: (spec: GridFigureSpec) => void;
}) {
  const [copyTargets, setCopyTargets] = useState<string[]>([]);
  const [drawOn, setDrawOn] = useState<DrawOn>("input");
  const [target, setTarget] = useState<DrawTarget>("roi");
  const [square, setSquare] = useState(true);
  const [sourceMissing, setSourceMissing] = useState(false);
  // Width/height of the drawing image, needed to turn "square in pixels" into
  // the normalized w/h the ROI is stored in.
  const [aspect, setAspect] = useState(1);

  if (!panel || index == null || panel.kind === "empty") {
    return <InlineNote>Select a panel in the grid to draw its ROI.</InlineNote>;
  }

  const key = roiKey(spec, panel, index);
  const roi = spec.rois[key];
  const crop = isFullFrame(spec.crops[key]) ? undefined : spec.crops[key];
  // Drawing on the input is the default because the shared canvas every panel
  // is resampled to takes the input's dimensions (it is the largest of the
  // group), so a square drawn here stays square in the rendered figure.
  const drawKind = drawOn === "panel" ? panel.kind : drawOn;
  const src = figureImgUrl({ ...panel, kind: drawKind }, THUMB_W);
  const matchCount = spec.panels.filter((p, i) => p.kind !== "empty" && roiKey(spec, p, i) === key).length;
  const scopeNoun = SCOPE_NOUN[spec.roi_scope];
  const others = otherKeys(spec, key);

  function commitRoi(patch: Partial<RoiSpec>) {
    const current = roi ?? defaultRoi();
    onChange({ ...spec, rois: { ...spec.rois, [key]: { ...current, ...patch } } });
  }

  function commitRect(rect: { x: number; y: number; w: number; h: number }) {
    const current = roi ?? defaultRoi();
    const clamped = clampRect(rect.x, rect.y, rect.w, rect.h);
    onChange({ ...spec, rois: { ...spec.rois, [key]: { ...current, ...clamped } } });
  }

  function commitCrop(rect: { x: number; y: number; w: number; h: number }) {
    const clamped = clampRect(rect.x, rect.y, rect.w, rect.h);
    onChange({ ...spec, crops: { ...spec.crops, [key]: clamped } });
  }

  function clearCrop() {
    const crops = { ...spec.crops };
    delete crops[key];
    onChange({ ...spec, crops });
  }

  function clearRoi() {
    const rois = { ...spec.rois };
    delete rois[key];
    onChange({ ...spec, rois });
  }

  function resetRoi() {
    onChange({ ...spec, rois: { ...spec.rois, [key]: defaultRoi() } });
  }

  /** Make an existing rectangle square in *image pixels* around its centre —
   *  the fix-up for a ROI drawn freehand before the lock was on. With the image
   *  measured as `aspect` wide by 1 tall, a square has h == w * aspect. */
  function snapSquare() {
    if (!roi) return;
    const side = Math.max(roi.w * aspect, roi.h);
    const w = side / aspect;
    const h = side;
    commitRect({ x: roi.x + roi.w / 2 - w / 2, y: roi.y + roi.h / 2 - h / 2, w, h });
  }

  function doCopy() {
    if (!roi || copyTargets.length === 0) return;
    const labelToKey = new Map(others.map((o) => [o.label, o.key]));
    const next = { ...spec.rois };
    for (const label of copyTargets) {
      const target = labelToKey.get(label);
      if (target) next[target] = { ...roi };
    }
    onChange({ ...spec, rois: next });
    setCopyTargets([]);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <Segmented value={drawOn} onChange={(v) => setDrawOn(v as DrawOn)} options={DRAW_ON} />
        <Checkbox checked={square} onChange={setSquare} label="Square" />
      </div>

      <Field label="Dragging draws">
        <Segmented
          value={target}
          onChange={(v) => setTarget(v as DrawTarget)}
          options={[
            { value: "roi", label: "ROI (marker)" },
            { value: "crop", label: "Crop (zoom)" },
          ]}
        />
      </Field>

      <RoiCanvas
        src={src}
        roi={roi}
        crop={crop}
        target={target}
        square={square}
        onDraw={target === "crop" ? commitCrop : commitRect}
        onAspect={setAspect}
        onMissing={() => setSourceMissing(true)}
      />
      <InlineNote>
        {target === "crop"
          ? "The crop is a plain zoom applied before the ROI — every panel of this group shows only that region, with the ROI marked inside it."
          : roi
            ? `Applies to all ${matchCount} ${scopeNoun}.`
            : `Drag on the image above to draw a ROI — it will apply to all ${matchCount} ${scopeNoun}.`}
        {square ? " Square lock is on; hold Shift to force it while off." : " Hold Shift to snap to a square."}
      </InlineNote>

      {crop && (
        <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-surface-2 px-2.5 py-1.5">
          <span className="text-[11px] text-muted-fg">
            Zoom {(1 / Math.max(crop.w, 0.01)).toFixed(1)}× · {Math.round(crop.w * 100)}×
            {Math.round(crop.h * 100)}% of the frame
          </span>
          <Button size="sm" variant="ghost" onClick={clearCrop}>
            Clear crop
          </Button>
        </div>
      )}

      {sourceMissing && (
        <InlineNote tone="danger">
          Input / ground truth could not be loaded. Check the dataset folder under Settings →
          Dataset source.
        </InlineNote>
      )}

      <ReferenceStrip panel={panel} roi={roi} onMissing={() => setSourceMissing(true)} />

      {roi && (
        <>
          <div className="grid grid-cols-4 gap-2">
            <Field label="X %">
              <NumberInput
                value={Math.round(roi.x * 100)}
                min={0}
                max={100}
                onChange={(v) => commitRect({ x: (v ?? 0) / 100, y: roi.y, w: roi.w, h: roi.h })}
              />
            </Field>
            <Field label="Y %">
              <NumberInput
                value={Math.round(roi.y * 100)}
                min={0}
                max={100}
                onChange={(v) => commitRect({ x: roi.x, y: (v ?? 0) / 100, w: roi.w, h: roi.h })}
              />
            </Field>
            <Field label="W %">
              <NumberInput
                value={Math.round(roi.w * 100)}
                min={1}
                max={100}
                onChange={(v) => commitRect({ x: roi.x, y: roi.y, w: (v ?? MIN_ROI * 100) / 100, h: roi.h })}
              />
            </Field>
            <Field label="H %">
              <NumberInput
                value={Math.round(roi.h * 100)}
                min={1}
                max={100}
                onChange={(v) => commitRect({ x: roi.x, y: roi.y, w: roi.w, h: (v ?? MIN_ROI * 100) / 100 })}
              />
            </Field>
          </div>

          <Field label="Mode">
            <Segmented
              value={roi.mode}
              onChange={(v) => commitRoi({ mode: v })}
              options={[
                { value: "marker", label: "Marker" },
                { value: "crop", label: "Crop" },
                { value: "inset", label: "Inset" },
              ]}
            />
          </Field>

          <div className="flex items-end gap-3">
            <Field label="Color" className="w-auto">
              <ColorWheel color={roi.color} onChange={(hex) => commitRoi({ color: hex })} />
            </Field>
            <Field label="Line width" className="flex-1">
              <NumberInput value={roi.linewidth} step={0.5} min={0.5} max={8} onChange={(v) => commitRoi({ linewidth: v ?? 1.5 })} />
            </Field>
          </div>

          {roi.mode === "inset" && (
            <>
              <Field label="Inset corner">
                <Select
                  value={roi.inset_corner}
                  onChange={(v) => commitRoi({ inset_corner: v as RoiSpec["inset_corner"] })}
                  options={INSET_CORNERS}
                />
              </Field>
              <Field label={`Inset size (${Math.round(roi.inset_scale * 100)}% of panel)`}>
                <Slider value={roi.inset_scale} onChange={(v) => commitRoi({ inset_scale: clamp01(v) })} min={0.1} max={0.8} step={0.01} />
              </Field>
              <Checkbox
                checked={roi.inset_connectors}
                onChange={(v) => commitRoi({ inset_connectors: v })}
                label="Draw connector lines"
              />
            </>
          )}

          {others.length > 0 && (
            <Field label="Copy this ROI to…">
              <ChipMultiSelect
                options={others.map((o) => o.label)}
                selected={copyTargets}
                onChange={setCopyTargets}
                emptyMeansAll={false}
              />
              <div className="pt-1.5">
                <Button size="sm" onClick={doCopy} disabled={copyTargets.length === 0}>
                  Copy to {copyTargets.length || ""} selected
                </Button>
              </div>
            </Field>
          )}

          <div className="flex gap-2 pt-1">
            <Button size="sm" variant="ghost" onClick={snapSquare}>
              Snap to square
            </Button>
            <Button size="sm" variant="ghost" onClick={resetRoi}>
              Reset
            </Button>
            <Button size="sm" variant="danger" onClick={clearRoi}>
              Clear ROI
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

/** Drag-to-draw rectangle on top of an image, in normalized 0..1 coords.
 *
 *  The box takes the image's own aspect ratio (measured on load) instead of a
 *  fixed square. That is not cosmetic: with a fixed square and `object-contain`
 *  a non-square image such as DRIVE (565x584) is letterboxed, so pointer
 *  coordinates taken against the container do not match the image and the ROI
 *  lands off-target. Matching the aspect makes container coords == image coords.
 *
 *  `square` then constrains the drag to equal *pixel* sides — equal normalized
 *  w/h would be a rectangle on any non-square image. Holding Shift does the same
 *  without the checkbox. */
function RoiCanvas({
  src,
  roi,
  crop,
  target,
  square,
  onDraw,
  onAspect,
  onMissing,
}: {
  src: string;
  roi: RoiSpec | undefined;
  crop: CropSpec | undefined;
  target: DrawTarget;
  square: boolean;
  onDraw: (rect: { x: number; y: number; w: number; h: number }) => void;
  onAspect?: (aspect: number) => void;
  onMissing?: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  const [draft, setDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [aspect, setAspect] = useState(1);

  function box(): { width: number; height: number } | null {
    const r = ref.current?.getBoundingClientRect();
    return r && r.width > 0 && r.height > 0 ? r : null;
  }

  function toNorm(e: ReactPointerEvent<HTMLDivElement>): { x: number; y: number } {
    const r = ref.current?.getBoundingClientRect();
    if (!r || r.width === 0 || r.height === 0) return { x: 0, y: 0 };
    return {
      x: clamp01((e.clientX - r.left) / r.width),
      y: clamp01((e.clientY - r.top) / r.height),
    };
  }

  function onPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = toNorm(e);
    start.current = p;
    setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
  }

  function onPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    if (!start.current) return;
    const p = toNorm(e);
    const s = start.current;
    let w = Math.abs(p.x - s.x);
    let h = Math.abs(p.y - s.y);
    const b = box();
    if ((square || e.shiftKey) && b) {
      const side = Math.max(w * b.width, h * b.height);   // longer drag wins
      w = side / b.width;
      h = side / b.height;
    }
    setDraft({
      x: p.x < s.x ? Math.max(0, s.x - w) : s.x,
      y: p.y < s.y ? Math.max(0, s.y - h) : s.y,
      w,
      h,
    });
  }

  function onPointerUp() {
    if (draft) onDraw(draft);
    start.current = null;
    setDraft(null);
  }

  // The canvas always shows the whole frame, both rectangles drawn on it — you
  // cannot place a crop by dragging inside an already-cropped view.
  const shownRoi = target === "roi" ? draft ?? roi : roi;
  const shownCrop = target === "crop" ? draft ?? crop : crop;

  return (
    <div
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{ aspectRatio: String(aspect) }}
      className="relative w-full touch-none select-none overflow-hidden rounded-lg bg-black cursor-crosshair"
    >
      {src && (
        <img
          src={src}
          alt=""
          draggable={false}
          onLoad={(e) => {
            const el = e.currentTarget;
            if (el.naturalWidth && el.naturalHeight) {
              const a = el.naturalWidth / el.naturalHeight;
              setAspect(a);
              onAspect?.(a);
            }
          }}
          onError={onMissing}
          className="pointer-events-none h-full w-full object-contain"
        />
      )}
      {shownCrop && (
        <div
          className="pointer-events-none absolute box-border"
          style={{
            left: `${shownCrop.x * 100}%`,
            top: `${shownCrop.y * 100}%`,
            width: `${shownCrop.w * 100}%`,
            height: `${shownCrop.h * 100}%`,
            border: "2px dashed #38bdf8",
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.45)",
          }}
        />
      )}
      {shownRoi && (
        <div
          className="pointer-events-none absolute box-border"
          style={{
            left: `${shownRoi.x * 100}%`,
            top: `${shownRoi.y * 100}%`,
            width: `${shownRoi.w * 100}%`,
            height: `${shownRoi.h * 100}%`,
            border: `2px solid ${roi?.color ?? "#ff2d55"}`,
          }}
        />
      )}
    </div>
  );
}

/** Input + ground truth for the panel's image, always full-frame with the ROI
 *  drawn on top — the reference for judging a crop, which the grid itself no
 *  longer shows once ROI mode is "crop". */
function ReferenceStrip({
  panel,
  roi,
  onMissing,
}: {
  panel: PanelRef;
  roi: RoiSpec | undefined;
  onMissing: () => void;
}) {
  if (!panel.dataset || !panel.image) return null;
  const refs: { kind: "input" | "gt"; label: string }[] = [
    { kind: "input", label: "Input" },
    { kind: "gt", label: "Ground truth" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2">
      {refs.map(({ kind, label }) => (
        <div key={kind} className="space-y-1">
          <div className="text-[10px] font-medium uppercase tracking-wide text-muted-fg">{label}</div>
          <div className="relative overflow-hidden rounded-md bg-black">
            <img
              src={figureImgUrl({ ...panel, kind }, THUMB_W)}
              alt={label}
              draggable={false}
              loading="lazy"
              onError={onMissing}
              className="block h-full w-full object-contain"
            />
            {roi && (
              <div
                className="pointer-events-none absolute box-border"
                style={{
                  left: `${roi.x * 100}%`,
                  top: `${roi.y * 100}%`,
                  width: `${roi.w * 100}%`,
                  height: `${roi.h * 100}%`,
                  border: `1.5px solid ${roi.color}`,
                }}
              />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
