import { useEffect, useState } from "react";
import { ChevronRight, File, Folder, FolderOpen, FolderPlus, HardDrive, X } from "lucide-react";
import { api } from "../lib/api";
import type { FsListing } from "../lib/types";
import { cn } from "../lib/util";
import { Button, TextInput } from "./ui";

/**
 * Native-feeling "Save to..." folder browser, backed by `/api/fs/*` on the
 * same machine the backend runs on (this is a local single-user tool, so the
 * server's filesystem is the user's filesystem). Works the same on
 * Linux/macOS/Windows since the split is done on whichever separator the
 * backend's paths actually use.
 */
export function DirectoryPicker({
  open,
  initialPath,
  mode = "directory",
  defaultFilename,
  title,
  onConfirm,
  onClose,
}: {
  open: boolean;
  initialPath?: string | null;
  /** "directory": pick a folder only. "file": also collect a filename. */
  mode?: "directory" | "file";
  defaultFilename?: string;
  title?: string;
  onConfirm: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<FsListing | null>(null);
  const [pathInput, setPathInput] = useState("");
  const [filename, setFilename] = useState(defaultFilename ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");

  function load(path?: string) {
    setBusy(true);
    setError(null);
    api
      .browseFs(path)
      .then((l) => {
        setListing(l);
        setPathInput(l.path);
      })
      .catch((e) => setError(String((e as Error).message ?? e)))
      .finally(() => setBusy(false));
  }

  useEffect(() => {
    if (!open) return;
    setFilename(defaultFilename ?? "");
    setCreating(false);
    setError(null);
    load(initialPath ?? undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  async function confirmNewFolder() {
    const name = newFolderName.trim();
    if (!name || !listing) return;
    setBusy(true);
    setError(null);
    try {
      const l = await api.makeDir(listing.path, name);
      setListing(l);
      setPathInput(l.path);
      setCreating(false);
      setNewFolderName("");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  function choose() {
    if (!listing) return;
    if (mode === "file") {
      const name = filename.trim();
      if (!name) {
        setError("Enter a file name.");
        return;
      }
      const dir = listing.path;
      const sep = dir.endsWith("/") || dir.endsWith("\\") ? "" : "/";
      onConfirm(`${dir}${sep}${name}`);
    } else {
      onConfirm(listing.path);
    }
    onClose();
  }

  const crumbs = breadcrumbs(listing?.path ?? "");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="flex h-[30rem] w-full max-w-2xl flex-col rounded-xl border border-border bg-surface shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h3 className="text-sm font-bold text-fg">
            {title ?? (mode === "file" ? "Save to..." : "Choose a folder")}
          </h3>
          <button onClick={onClose} className="rounded p-1 text-muted-fg hover:bg-muted hover:text-fg cursor-pointer">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-2 border-b border-border px-4 py-2">
          {listing && listing.favorites.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {listing.favorites.map((f) => (
                <button
                  key={f.path}
                  onClick={() => load(f.path)}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-2 px-2.5 py-1 text-xs text-muted-fg hover:text-fg cursor-pointer"
                >
                  <HardDrive className="h-3 w-3" /> {f.name}
                </button>
              ))}
            </div>
          )}
          <div className="flex items-center gap-0.5 overflow-x-auto whitespace-nowrap pb-0.5 text-xs text-muted-fg">
            {crumbs.map((c, i) => (
              <span key={c.path} className="flex shrink-0 items-center gap-0.5">
                {i > 0 && <ChevronRight className="h-3 w-3 shrink-0" />}
                <button
                  onClick={() => load(c.path)}
                  className="shrink-0 rounded px-1 py-0.5 hover:bg-muted hover:text-fg cursor-pointer"
                >
                  {c.label}
                </button>
              </span>
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              load(pathInput);
            }}
          >
            <TextInput
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              placeholder="/abs/path"
            />
          </form>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-2">
          {error && <p className="px-2 py-1 text-xs text-danger">{error}</p>}
          {busy && !listing ? (
            <p className="px-2 py-4 text-center text-xs text-muted-fg">Loading…</p>
          ) : (
            <div className="space-y-0.5">
              {listing?.parent && (
                <button
                  onClick={() => load(listing.parent!)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-muted-fg hover:bg-muted cursor-pointer"
                >
                  <Folder className="h-4 w-4 shrink-0" /> ..
                </button>
              )}
              {listing?.entries.map((e) => (
                <button
                  key={e.path}
                  onClick={() => e.is_dir && load(e.path)}
                  disabled={!e.is_dir}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                    e.is_dir ? "text-fg hover:bg-muted cursor-pointer" : "text-muted-fg/50 cursor-default",
                  )}
                >
                  {e.is_dir ? <Folder className="h-4 w-4 shrink-0" /> : <File className="h-4 w-4 shrink-0" />}
                  <span className="truncate">{e.name}</span>
                </button>
              ))}
              {listing && listing.entries.length === 0 && (
                <p className="px-2 py-4 text-center text-xs text-muted-fg">Empty folder.</p>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border px-4 py-2">
          {creating ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                confirmNewFolder();
              }}
              className="flex items-center gap-2"
            >
              <TextInput
                autoFocus
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="Folder name"
                className="flex-1"
              />
              <Button size="sm" variant="default" type="submit">
                Create
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setCreating(false)}>
                Cancel
              </Button>
            </form>
          ) : (
            <Button size="sm" variant="ghost" onClick={() => setCreating(true)}>
              <FolderPlus className="h-3.5 w-3.5" /> New folder
            </Button>
          )}
        </div>

        {mode === "file" && (
          <div className="border-t border-border px-4 py-2">
            <TextInput
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="filename.ext"
            />
          </div>
        )}

        <div className="flex items-center justify-between gap-2 border-t border-border px-4 py-3">
          <span className="truncate text-[11px] text-muted-fg">{listing?.path}</span>
          <div className="flex shrink-0 gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={choose} disabled={!listing}>
              {mode === "file" ? "Save here" : "Choose this folder"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function breadcrumbs(path: string): { label: string; path: string }[] {
  if (!path) return [];
  const sep = path.includes("\\") ? "\\" : "/";
  const segs = path.split(sep).filter(Boolean);
  const crumbs: { label: string; path: string }[] = [];
  if (sep === "\\") {
    let acc = (segs[0] ?? "") + sep;
    crumbs.push({ label: acc, path: acc });
    for (const s of segs.slice(1)) {
      acc = acc.endsWith(sep) ? acc + s : acc + sep + s;
      crumbs.push({ label: s, path: acc });
    }
  } else {
    crumbs.push({ label: "/", path: "/" });
    let acc = "";
    for (const s of segs) {
      acc = `${acc}/${s}`;
      crumbs.push({ label: s, path: acc });
    }
  }
  return crumbs;
}

function dirnameOf(p: string): string {
  if (!p) return "";
  const sep = p.includes("\\") ? "\\" : "/";
  const idx = p.lastIndexOf(sep);
  return idx >= 0 ? p.slice(0, idx) : "";
}

function basenameOf(p: string): string {
  if (!p) return "";
  const sep = p.includes("\\") ? "\\" : "/";
  const idx = p.lastIndexOf(sep);
  return idx >= 0 ? p.slice(idx + 1) : p;
}

/** Drop-in replacement for a `TextInput` bound to a save-path/output-directory
 *  string: still freely editable by hand, plus a "Browse..." button that opens
 *  the `DirectoryPicker` and writes the result back. */
export function PathField({
  value,
  onChange,
  mode = "file",
  placeholder,
  defaultFilename,
  className,
}: {
  value: string;
  onChange: (v: string) => void;
  mode?: "directory" | "file";
  placeholder?: string;
  defaultFilename?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className={cn("flex gap-1.5", className)}>
        <TextInput
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1"
        />
        <Button size="sm" variant="default" onClick={() => setOpen(true)} title="Browse…">
          <FolderOpen className="h-3.5 w-3.5" />
        </Button>
      </div>
      <DirectoryPicker
        open={open}
        mode={mode}
        initialPath={dirnameOf(value) || undefined}
        defaultFilename={basenameOf(value) || defaultFilename}
        onConfirm={onChange}
        onClose={() => setOpen(false)}
      />
    </>
  );
}
