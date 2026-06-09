import { useEffect, useRef, useState } from "react";
import { HexColorPicker } from "react-colorful";
import { cn } from "../lib/util";

/** A Photoshop-style color swatch that opens an HSV wheel + hex/RGB inputs. */
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
  const ref = useRef<HTMLDivElement>(null);
  const current = color || "#3b82f6";

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const rgb = hexToRgb(current);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        title="Pick color"
        className="rounded-md border border-border shadow-sm cursor-pointer"
        style={{ background: current, width: size, height: size }}
      />
      {open && (
        <div className="absolute right-0 z-30 mt-2 w-56 space-y-2 rounded-lg border border-border bg-surface p-3 shadow-xl">
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
        </div>
      )}
    </div>
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
