import { type ReactNode } from "react";
import { AlertTriangle, ImageOff, Loader2 } from "lucide-react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border bg-surface px-6 py-4">
      <div>
        <h1 className="text-lg font-bold text-fg">{title}</h1>
        {subtitle && <p className="text-sm text-muted-fg">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Renders a backend-produced image (SVG/PNG) with loading/error/empty states. */
export function PreviewPane({
  url,
  loading,
  error,
}: {
  url: string | null;
  loading: boolean;
  error: string | null;
}) {
  return (
    <div className="relative flex min-h-[320px] flex-1 items-center justify-center rounded-lg border border-border bg-surface p-4">
      {loading && (
        <div className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs text-muted-fg">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> rendering…
        </div>
      )}
      {error ? (
        <div className="flex max-w-md flex-col items-center gap-2 text-center text-sm text-danger">
          <AlertTriangle className="h-6 w-6" />
          <span>{error}</span>
        </div>
      ) : url ? (
        <img src={url} alt="figure preview" className="max-h-full max-w-full object-contain" />
      ) : (
        !loading && (
          <div className="flex flex-col items-center gap-2 text-muted-fg">
            <ImageOff className="h-6 w-6" />
            <span className="text-sm">No data to display — adjust the controls.</span>
          </div>
        )
      )}
    </div>
  );
}

export function InlineNote({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "danger" | "success" }) {
  const tones = {
    muted: "text-muted-fg",
    danger: "text-danger",
    success: "text-success",
  };
  return <p className={`text-xs ${tones[tone]}`}>{children}</p>;
}
