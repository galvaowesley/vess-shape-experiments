import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api } from "../lib/api";
import type { DashboardMeta } from "../lib/types";
import { Button, Field, Segmented, Select, TextInput } from "./ui";

const NEW = "__new__";

export type PinResult = { dashboardId: string; title: string; mode: "new" | "update" };

/**
 * Modal to choose (or create) the dashboard a figure is pinned to. When
 * `editItemId` is set it also offers to update that pinned figure in place.
 * Resolves the chosen/created dashboard id and hands it back via `onConfirm`.
 */
export function PinDialog({
  open,
  defaultTitle,
  editItemId,
  onConfirm,
  onClose,
}: {
  open: boolean;
  defaultTitle: string;
  editItemId?: string | null;
  onConfirm: (r: PinResult) => Promise<void> | void;
  onClose: () => void;
}) {
  const [dashboards, setDashboards] = useState<DashboardMeta[]>([]);
  const [target, setTarget] = useState<string>(NEW);
  const [newName, setNewName] = useState("");
  const [title, setTitle] = useState(defaultTitle);
  const [mode, setMode] = useState<"new" | "update">(editItemId ? "update" : "new");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setTitle(defaultTitle);
    setMode(editItemId ? "update" : "new");
    setErr(null);
    api
      .listDashboards()
      .then((d) => {
        setDashboards(d);
        setTarget(d.length ? d[0].id : NEW);
      })
      .catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, defaultTitle, editItemId]);

  if (!open) return null;

  async function confirm() {
    setBusy(true);
    setErr(null);
    try {
      let dashboardId = target;
      if (target === NEW) {
        const created = await api.createDashboard(newName.trim() || "New dashboard");
        dashboardId = created.id;
      }
      await onConfirm({ dashboardId, title: title.trim() || defaultTitle, mode });
      onClose();
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  const options = [
    ...dashboards.map((d) => ({ value: d.id, label: `${d.name} (${d.count ?? 0})` })),
    { value: NEW, label: "+ New dashboard…" },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-xl border border-border bg-surface shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h3 className="text-sm font-bold text-fg">Pin to dashboard</h3>
          <button onClick={onClose} className="rounded p-1 text-muted-fg hover:bg-muted hover:text-fg cursor-pointer">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-3 px-4 py-4">
          {editItemId && (
            <Field label="Action">
              <Segmented
                value={mode}
                onChange={(v) => setMode(v)}
                options={[
                  { value: "update", label: "Update this figure" },
                  { value: "new", label: "Pin as new copy" },
                ]}
              />
            </Field>
          )}

          <Field label="Figure title">
            <TextInput value={title} onChange={(e) => setTitle(e.target.value)} />
          </Field>

          <Field label="Dashboard">
            <Select value={target} onChange={setTarget} options={options} />
          </Field>

          {target === NEW && (
            <Field label="New dashboard name">
              <TextInput
                value={newName}
                placeholder="e.g. Paper figures"
                onChange={(e) => setNewName(e.target.value)}
              />
            </Field>
          )}

          {err && <p className="text-xs text-danger">{err}</p>}
        </div>

        <div className="flex justify-end gap-2 border-t border-border px-4 py-3">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={confirm} disabled={busy}>
            {editItemId && mode === "update" ? "Save changes" : "Pin"}
          </Button>
        </div>
      </div>
    </div>
  );
}
