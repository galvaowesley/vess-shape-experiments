import { useEffect, useRef, useState } from "react";
import { ChevronDown, X } from "lucide-react";
import type { PaletteCatalog } from "../lib/types";
import { Swatches } from "./ColorWheel";
import { cn } from "../lib/util";

/** Seaborn named-palette dropdown with swatch previews. */
export function PalettePicker({
  catalog,
  value,
  onChange,
}: {
  catalog: PaletteCatalog | null;
  value: string | null | undefined;
  onChange: (name: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const selected = value
    ? Object.values(catalog ?? {})
        .flat()
        .find((p) => p.name === value)
    : null;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-lg border border-border bg-surface-2 px-2.5 py-1.5 text-sm cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring"
      >
        <span className="flex flex-1 items-center gap-2 truncate">
          {selected ? (
            <>
              <span className="w-20 shrink-0">
                <Swatches colors={selected.colors} />
              </span>
              <span className="truncate text-fg">{selected.name}</span>
            </>
          ) : (
            <span className="text-muted-fg">Model palette (default)</span>
          )}
        </span>
        <span className="flex items-center gap-1">
          {value && (
            <X
              className="h-3.5 w-3.5 text-muted-fg hover:text-fg"
              onClick={(e) => {
                e.stopPropagation();
                onChange(null);
              }}
            />
          )}
          <ChevronDown className="h-4 w-4 text-muted-fg" />
        </span>
      </button>

      {open && catalog && (
        <div className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-lg border border-border bg-surface p-2 shadow-xl">
          {Object.entries(catalog).map(([group, entries]) => (
            <div key={group} className="mb-2">
              <div className="px-1 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-fg">
                {group}
              </div>
              {entries.map((p) => (
                <button
                  key={p.name}
                  onClick={() => {
                    onChange(p.name);
                    setOpen(false);
                  }}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left cursor-pointer hover:bg-muted",
                    value === p.name && "bg-muted",
                  )}
                >
                  <span className="w-24 shrink-0">
                    <Swatches colors={p.colors} />
                  </span>
                  <span className="truncate text-xs text-fg">{p.name}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
