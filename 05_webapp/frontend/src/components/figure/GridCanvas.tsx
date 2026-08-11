import { Fragment, useMemo, useState } from "react";
import { Plus, Square, Target, X } from "lucide-react";
import { figureImgUrl } from "../../lib/api";
import type { CropSpec, GridFigureSpec, PanelRef, RoiSpec } from "../../lib/types";
import { emptyPanel, isFullFrame, panelLabel, roiKey, unavailableIndices } from "../../lib/figureSpec";
import { cn } from "../../lib/util";
import { TextInput } from "../ui";

const THUMB_W = 384;
const MIN_ROI = 0.02;

type Editing = { axis: "row" | "col"; index: number } | null;

/** Live CSS-grid preview — instant, no server round-trip. Drag a filled cell
 *  onto another to swap them (matches the pattern in pages/Appearance.tsx). */
export function GridCanvas({
  spec,
  onChange,
  selected,
  onSelect,
}: {
  spec: GridFigureSpec;
  onChange: (spec: GridFigureSpec) => void;
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [overIdx, setOverIdx] = useState<number | null>(null);
  const [editing, setEditing] = useState<Editing>(null);

  function swap(a: number, b: number) {
    if (a === b) return;
    const panels = [...spec.panels];
    [panels[a], panels[b]] = [panels[b], panels[a]];
    onChange({ ...spec, panels });
  }
  function clearCell(i: number) {
    const panels = [...spec.panels];
    panels[i] = emptyPanel();
    onChange({ ...spec, panels });
  }
  /** Pin an empty cell to the opposite of what it currently shows, so one click
   *  always flips between the "no inference" mark and a plain gap. */
  function toggleMissing(i: number, showing: boolean) {
    const panels = [...spec.panels];
    panels[i] = { ...panels[i], missing: !showing };
    onChange({ ...spec, panels });
  }
  function setRowLabel(i: number, v: string) {
    const row_labels = [...spec.row_labels];
    row_labels[i] = v;
    onChange({ ...spec, row_labels });
  }
  function setColLabel(i: number, v: string) {
    const col_labels = [...spec.col_labels];
    col_labels[i] = v;
    onChange({ ...spec, col_labels });
  }

  const unavailable = useMemo(
    () => unavailableIndices(spec.panels, spec.cols),
    [spec.panels, spec.cols],
  );

  const cellPx = Math.round(Math.min(320, Math.max(64, spec.panel_size * 90)));
  const gapX = Math.max(0, Math.round(spec.wspace * 140));
  const gapY = Math.max(0, Math.round(spec.hspace * 140));

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface p-3">
      <div
        className="inline-grid"
        style={{
          gridTemplateColumns: `72px repeat(${spec.cols}, ${cellPx}px)`,
          gridTemplateRows: `28px repeat(${spec.rows}, ${cellPx}px)`,
          columnGap: gapX,
          rowGap: gapY,
          background: spec.background,
        }}
      >
        <div />
        {spec.col_labels.map((label, c) => (
          <LabelCell
            key={`col-${c}`}
            label={label}
            editing={editing?.axis === "col" && editing.index === c}
            color={spec.label_color}
            fontSize={spec.col_label_fontsize}
            onStartEdit={() => setEditing({ axis: "col", index: c })}
            onCommit={(v) => {
              setColLabel(c, v);
              setEditing(null);
            }}
          />
        ))}

        {Array.from({ length: spec.rows }).map((_, r) => (
          <Fragment key={`row-${r}`}>
            <LabelCell
              label={spec.row_labels[r] ?? ""}
              editing={editing?.axis === "row" && editing.index === r}
              color={spec.label_color}
              fontSize={spec.row_label_fontsize}
              vertical
              onStartEdit={() => setEditing({ axis: "row", index: r })}
              onCommit={(v) => {
                setRowLabel(r, v);
                setEditing(null);
              }}
            />
            {Array.from({ length: spec.cols }).map((_, c) => {
              const i = r * spec.cols + c;
              const panel = spec.panels[i];
              return (
                <Cell
                  key={i}
                  index={i}
                  panel={panel}
                  roi={panel.kind !== "empty" ? spec.rois[roiKey(spec, panel, i)] : undefined}
                  crop={panel.kind !== "empty" ? spec.crops[roiKey(spec, panel, i)] : undefined}
                  showCross={spec.panel_style.missing_mark && unavailable.has(i)}
                  missingColor={spec.panel_style.missing_color ?? spec.label_color}
                  missingWidth={spec.panel_style.missing_width}
                  onToggleMissing={(showing) => toggleMissing(i, showing)}
                  isSelected={selected === i}
                  isDragging={dragIdx === i}
                  isOver={overIdx === i && dragIdx !== null && dragIdx !== i}
                  onSelect={() => onSelect(i)}
                  onClear={() => clearCell(i)}
                  onDragStart={() => setDragIdx(i)}
                  onDragOver={() => {
                    if (overIdx !== i) setOverIdx(i);
                  }}
                  onDrop={() => {
                    if (dragIdx !== null && dragIdx !== i) swap(dragIdx, i);
                    setDragIdx(null);
                    setOverIdx(null);
                  }}
                  onDragEnd={() => {
                    setDragIdx(null);
                    setOverIdx(null);
                  }}
                />
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function LabelCell({
  label,
  editing,
  color,
  fontSize,
  vertical,
  onStartEdit,
  onCommit,
}: {
  label: string;
  editing: boolean;
  color: string;
  fontSize: number;
  vertical?: boolean;
  onStartEdit: () => void;
  onCommit: (v: string) => void;
}) {
  if (editing) {
    return (
      <div className="flex items-center px-0.5">
        <TextInput
          autoFocus
          defaultValue={label}
          onBlur={(e) => onCommit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            if (e.key === "Escape") onCommit(label);
          }}
          className="!px-1.5 !py-0.5 text-[11px]"
        />
      </div>
    );
  }
  return (
    <button
      onClick={onStartEdit}
      title="Click to edit label"
      className={cn(
        "flex items-center justify-center truncate rounded px-1 text-center font-medium hover:bg-muted cursor-text",
        vertical && "[writing-mode:vertical-rl] rotate-180",
      )}
      style={{ color, fontSize }}
    >
      {label || <span className="text-muted-fg/50">+ label</span>}
    </button>
  );
}

function Cell({
  index,
  panel,
  roi,
  crop,
  showCross,
  missingColor,
  missingWidth,
  onToggleMissing,
  isSelected,
  isDragging,
  isOver,
  onSelect,
  onClear,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  index: number;
  panel: PanelRef;
  roi: RoiSpec | undefined;
  crop: CropSpec | undefined;
  showCross: boolean;
  missingColor: string;
  missingWidth: number;
  onToggleMissing: (showing: boolean) => void;
  isSelected: boolean;
  isDragging: boolean;
  isOver: boolean;
  onSelect: () => void;
  onClear: () => void;
  onDragStart: () => void;
  onDragOver: () => void;
  onDrop: () => void;
  onDragEnd: () => void;
}) {
  const isEmpty = panel.kind === "empty";
  const src = isEmpty ? "" : figureImgUrl(panel, THUMB_W);
  // What the cell actually shows: the pre-crop, unless the ROI is in "crop"
  // mode, which sets the view itself and so supersedes it (same precedence as
  // _render_cell in figuregrid.py).
  const view = roi?.mode === "crop" ? roi : isFullFrame(crop) ? undefined : crop;

  return (
    <div
      draggable={!isEmpty}
      onDragStart={(e) => {
        onDragStart();
        e.dataTransfer.effectAllowed = "move";
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        onDragOver();
      }}
      onDrop={(e) => {
        e.preventDefault();
        onDrop();
      }}
      onDragEnd={onDragEnd}
      onClick={onSelect}
      className={cn(
        "group relative overflow-hidden rounded-md border transition-opacity",
        isEmpty ? "bg-transparent" : "bg-black",
        isSelected ? "border-primary ring-2 ring-primary" : "border-border",
        isOver && "border-primary ring-1 ring-primary",
        isDragging && "opacity-40",
        !isEmpty && "cursor-grab active:cursor-grabbing",
      )}
      title={showCross ? `No inference for ${comboLabel(panel) || "this combination"}` : undefined}
    >
      {isEmpty ? (
        <>
          {showCross ? (
            <MissingCross color={missingColor} width={missingWidth} />
          ) : (
            <div className="flex h-full w-full flex-col items-center justify-center gap-1 border-2 border-dashed border-border/70 text-muted-fg hover:border-primary hover:text-primary">
              <Plus className="h-5 w-5" />
            </div>
          )}
          <div className="absolute right-1 top-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button
              title={showCross ? "Leave this cell blank instead" : "Mark as: no inference"}
              onClick={(e) => {
                e.stopPropagation();
                onToggleMissing(showCross);
              }}
              className="rounded bg-black/70 p-1 text-white hover:bg-black cursor-pointer"
            >
              {showCross ? <Square className="h-3 w-3" /> : <X className="h-3 w-3" />}
            </button>
          </div>
        </>
      ) : (
        <>
          <PanelImage src={src} alt={panelLabel(panel)} view={view} />
          {roi && roi.mode !== "crop" && <MarkerRect roi={roi} view={view} />}
          {roi && roi.mode === "inset" && <InsetBox src={src} alt={panelLabel(panel)} roi={roi} />}
          <div className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-black/60 px-1 py-0.5 text-[10px] text-white">
            {panelLabel(panel)}
          </div>
          <div className="absolute right-1 top-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button
              title="Clear cell"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="rounded bg-black/70 p-1 text-white hover:bg-black cursor-pointer"
            >
              <X className="h-3 w-3" />
            </button>
            <button
              title="Make this the ROI reference"
              onClick={(e) => {
                e.stopPropagation();
                onSelect();
              }}
              className="rounded bg-black/70 p-1 text-white hover:bg-black cursor-pointer"
            >
              <Target className="h-3 w-3" />
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/** `panelLabel` is empty for `empty` panels by design (it feeds figure captions),
 *  but an unavailable cell still knows which combination it stands for. */
function comboLabel(p: PanelRef): string {
  const shot = p.num_samples == null ? null : p.num_samples === 0 ? "zero-shot" : `${p.num_samples}-shot`;
  return [p.model_type, shot].filter(Boolean).join(" · ");
}

/** Boxed diagonal cross for a combination that has no inference — the DOM twin
 *  of `_draw_missing` in backend/app/figuregrid.py. `vectorEffect` keeps the
 *  stroke one pixel wide regardless of the cell size the viewBox scales to. */
function MissingCross({ color, width }: { color: string; width: number }) {
  const stroke = Math.max(width, 0.5);
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="pointer-events-none h-full w-full"
      aria-label="no inference"
    >
      <g stroke={color} strokeWidth={stroke} fill="none" vectorEffect="non-scaling-stroke">
        <rect x="0" y="0" width="100" height="100" vectorEffect="non-scaling-stroke" />
        <line x1="0" y1="0" x2="100" y2="100" vectorEffect="non-scaling-stroke" />
        <line x1="0" y1="100" x2="100" y2="0" vectorEffect="non-scaling-stroke" />
      </g>
    </svg>
  );
}

/** Full image normally, or scaled/translated so only `view` is visible — the
 *  pre-crop zoom, or the ROI itself when its mode is "crop". Coords are
 *  normalized so no per-image math is needed beyond this. */
function PanelImage({ src, alt, view }: { src: string; alt: string; view: Rect | undefined }) {
  if (view) return <CropImage src={src} alt={alt} roi={view} />;
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      decoding="async"
      className="h-full w-full object-contain"
    />
  );
}

/** Any normalized rectangle — an ROI or a crop; the geometry helpers only ever
 *  need x/y/w/h and treating them alike keeps one implementation. */
type Rect = { x: number; y: number; w: number; h: number };

function CropImage({ src, alt, roi }: { src: string; alt: string; roi: Rect }) {
  const w = Math.max(roi.w, MIN_ROI);
  const h = Math.max(roi.h, MIN_ROI);
  return (
    <div className="relative h-full w-full overflow-hidden">
      <img
        src={src}
        alt={alt}
        loading="lazy"
        decoding="async"
        className="absolute max-w-none"
        style={{
          left: `${-(roi.x / w) * 100}%`,
          top: `${-(roi.y / h) * 100}%`,
          width: `${(1 / w) * 100}%`,
          height: `${(1 / h) * 100}%`,
        }}
      />
    </div>
  );
}

/** Re-express a full-frame rectangle in the coordinates of a cropped view, so
 *  the ROI marker stays on the same pixels once the panel is zoomed. */
export function relativeTo(rect: Rect, view: Rect | undefined): Rect {
  if (!view) return rect;
  const w = Math.max(view.w, MIN_ROI);
  const h = Math.max(view.h, MIN_ROI);
  return { x: (rect.x - view.x) / w, y: (rect.y - view.y) / h, w: rect.w / w, h: rect.h / h };
}

function MarkerRect({ roi, view }: { roi: RoiSpec; view: Rect | undefined }) {
  const r = relativeTo(roi, view);
  return (
    <div
      className="pointer-events-none absolute box-border"
      style={{
        left: `${r.x * 100}%`,
        top: `${r.y * 100}%`,
        width: `${r.w * 100}%`,
        height: `${r.h * 100}%`,
        border: `2px solid ${roi.color}`,
      }}
    />
  );
}

const CORNER_STYLE: Record<RoiSpec["inset_corner"], { top?: number; bottom?: number; left?: number; right?: number }> = {
  "upper right": { top: 4, right: 4 },
  "upper left": { top: 4, left: 4 },
  "lower right": { bottom: 4, right: 4 },
  "lower left": { bottom: 4, left: 4 },
};

function InsetBox({ src, alt, roi }: { src: string; alt: string; roi: RoiSpec }) {
  const scale = Math.min(Math.max(roi.inset_scale, 0.1), 0.9);
  return (
    <div
      className="pointer-events-none absolute overflow-hidden box-border"
      style={{
        ...CORNER_STYLE[roi.inset_corner],
        width: `${scale * 100}%`,
        height: `${scale * 100}%`,
        border: `2px solid ${roi.color}`,
      }}
    >
      <CropImage src={src} alt={alt} roi={roi} />
    </div>
  );
}
