import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { HexColorPicker } from "react-colorful";
import { GripHorizontal } from "lucide-react";
import { cn } from "../lib/util";

const POPUP_W = 232; // approximate rendered width of the popup (w-56 = 224px + borders)
const POPUP_H = 340; // approximate rendered height

/** A Photoshop-style color swatch that opens an HSV wheel + hex/RGB inputs.
 *
 * Rendered via React Portal at document.body so it escapes any overflow ancestor.
 * The popup is draggable via the grip handle at the top.
 */
export function ColorWheel({
  color,
  onChange,
  size = 22,
}: {
  color?: string | null;
  onChange: (hex: string) => void;
  size?: number;
}) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const current = color || "#3b82f6";

  // Reset drag offset whenever popup opens/closes
  useEffect(() => {
    if (!open) setDragOffset({ x: 0, y: 0 });
  }, [open]);

  // Compute initial popup position anchored to the button
  useLayoutEffect(() => {
    if (!open || !btnRef.current) {
      setPos(null);
      return;
    }
    const r = btnRef.current.getBoundingClientRect();

    // Prefer opening to the right of the button; fall back to the left if it would overflow
    let left = r.right + 8;
    if (left + POPUP_W > window.innerWidth - 8) {
      left = r.left - POPUP_W - 8;
    }
    left = Math.max(8, Math.min(left, window.innerWidth - POPUP_W - 8));

    // Vertically align with the top of the button; clamp so the popup stays visible
    let top = r.top;
    top = Math.max(8, Math.min(top, window.innerHeight - POPUP_H - 8));

    setPos({ top, left });
  }, [open]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (btnRef.current?.contains(e.target as Node)) return;
      if (popupRef.current?.contains(e.target as Node)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  function startDrag(e: React.MouseEvent) {
    e.preventDefault();
    const startX = e.clientX, startY = e.clientY;
    const ox = dragOffset.x, oy = dragOffset.y;
    const onMove = (me: MouseEvent) => {
      setDragOffset({ x: ox + me.clientX - startX, y: oy + me.clientY - startY });
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  const rgb = hexToRgb(current);

  const popup =
    open && pos
      ? createPortal(
          <div
            ref={popupRef}
            style={{
              position: "fixed",
              top: pos.top + dragOffset.y,
              left: pos.left + dragOffset.x,
              zIndex: 9999,
            }}
            className="w-56 space-y-2 rounded-lg border border-border bg-surface p-3 shadow-xl"
          >
            {/* Drag handle */}
            <div
              onMouseDown={startDrag}
              className="flex cursor-move items-center justify-between pb-1 border-b border-border select-none"
            >
              <span className="text-[10px] uppercase tracking-wide text-muted-fg">Color</span>
              <GripHorizontal className="h-3.5 w-3.5 text-muted-fg" />
            </div>

            <HexColorPicker color={current} onChange={onChange} />
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-muted-fg">HEX</span>
              <input
                value={current}
                onChange={(e) => onChange(e.target.value)}
                className="w-full rounded-md border border-border bg-surface-2 px-2 py-1 font-mono text-xs text-fg focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            {rgb && (
              <div className="grid grid-cols-3 gap-2">
                {(["r", "g", "b"] as const).map((ch) => (
                  <label key={ch} className="block">
                    <span className="text-[10px] uppercase text-muted-fg">{ch}</span>
                    <input
                      type="number"
                      min={0}
                      max={255}
                      value={rgb[ch]}
                      onChange={(e) => {
                        const next = { ...rgb, [ch]: clamp(Number(e.target.value)) };
                        onChange(rgbToHex(next.r, next.g, next.b));
                      }}
                      className="w-full rounded-md border border-border bg-surface-2 px-1.5 py-1 font-mono text-xs text-fg focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                  </label>
                ))}
              </div>
            )}
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen((o) => !o)}
        title="Pick color"
        className="rounded-md border border-border shadow-sm cursor-pointer"
        style={{ background: current, width: size, height: size }}
      />
      {popup}
    </>
  );
}

function clamp(n: number) {
  return Math.max(0, Math.min(255, Math.round(n || 0)));
}
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim());
  if (!m) return null;
  return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
}
function rgbToHex(r: number, g: number, b: number) {
  return "#" + [r, g, b].map((x) => clamp(x).toString(16).padStart(2, "0")).join("");
}

export function Swatches({ colors }: { colors: string[] }) {
  return (
    <div className="flex overflow-hidden rounded">
      {colors.map((c, i) => (
        <span key={i} className={cn("h-3.5 flex-1")} style={{ background: c }} />
      ))}
    </div>
  );
}
