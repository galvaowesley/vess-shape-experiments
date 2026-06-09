import { useEffect, useRef, useState } from "react";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/** Debounce a changing value (e.g. a PlotSpec) for live-but-not-jittery render. */
export function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

/** Revoke previous object URL when a new one replaces it. */
export function useObjectUrl(url: string | null) {
  const prev = useRef<string | null>(null);
  useEffect(() => {
    if (prev.current && prev.current !== url) URL.revokeObjectURL(prev.current);
    prev.current = url;
    return () => {
      if (prev.current) URL.revokeObjectURL(prev.current);
    };
  }, [url]);
}
