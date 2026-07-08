import { TABLE_DIMENSIONS } from "../lib/plotSpec";

/** Renders a computed pivot table (columns + rows). Shared by the Tables page
 *  and the dashboard cards so a pinned table looks the same everywhere. */
export function TableView({
  columns,
  rows,
  dimCount,
}: {
  columns: string[];
  rows: (string | number | null)[][];
  dimCount?: number; // leading row-dimension columns; auto-detected when omitted
}) {
  const dims =
    dimCount ??
    (() => {
      const i = columns.findIndex((c) => !TABLE_DIMENSIONS.includes(c));
      return i === -1 ? columns.length : i;
    })();
  return (
    <div className="overflow-auto rounded-lg border border-border">
      <table className="w-full border-collapse text-left text-xs">
        <thead>
          <tr className="bg-surface-2 text-[11px] uppercase tracking-wide text-muted-fg">
            {columns.map((c, i) => (
              <th
                key={c + i}
                className={`whitespace-nowrap px-3 py-2 font-semibold ${i < dims ? "" : "text-right"}`}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-t border-border odd:bg-surface even:bg-surface-2/40">
              {row.map((v, ci) => (
                <td
                  key={ci}
                  className={`whitespace-nowrap px-3 py-1.5 ${
                    ci < dims ? "font-medium text-fg" : "text-right tabular-nums text-muted-fg"
                  }`}
                >
                  {v === null || v === "" ? "—" : String(v)}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={Math.max(columns.length, 1)} className="px-3 py-6 text-center text-muted-fg">
                No rows — adjust the filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
