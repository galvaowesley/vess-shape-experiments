import {
  type ReactNode,
  type InputHTMLAttributes,
  type SelectHTMLAttributes,
  useState,
} from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "../lib/util";

/* ----------------------------- Button ----------------------------- */
export function Button({
  children,
  onClick,
  variant = "default",
  size = "md",
  disabled,
  title,
  className,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost" | "danger" | "accent";
  size?: "sm" | "md";
  disabled?: boolean;
  title?: string;
  className?: string;
  type?: "button" | "submit";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer";
  const sizes = { sm: "text-xs px-2.5 py-1.5", md: "text-sm px-3.5 py-2" };
  const variants = {
    default: "bg-surface-2 text-fg border border-border hover:bg-muted",
    primary: "bg-primary text-primary-fg hover:opacity-90",
    accent: "bg-accent text-accent-fg hover:opacity-90",
    ghost: "text-muted-fg hover:bg-muted hover:text-fg",
    danger: "bg-danger text-white hover:opacity-90",
  };
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(base, sizes[size], variants[variant], className)}
    >
      {children}
    </button>
  );
}

/* ----------------------------- Card ----------------------------- */
export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-lg border border-border bg-surface", className)}>{children}</div>
  );
}

/* ----------------------------- Field ----------------------------- */
export function Field({
  label,
  children,
  hint,
  className,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
  className?: string;
}) {
  return (
    <label className={cn("block space-y-1", className)}>
      <span className="text-xs font-medium text-muted-fg">{label}</span>
      {children}
      {hint && <span className="block text-[11px] text-muted-fg/80">{hint}</span>}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-border bg-surface-2 px-2.5 py-1.5 text-sm text-fg placeholder:text-muted-fg/60 focus:outline-none focus:ring-2 focus:ring-ring";

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn(inputCls, props.className)} />;
}

export function NumberInput({
  value,
  onChange,
  step,
  min,
  max,
  placeholder,
}: {
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
}) {
  return (
    <input
      type="number"
      className={inputCls}
      value={value ?? ""}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder ?? "auto"}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
    />
  );
}

export function Select({
  value,
  onChange,
  options,
  ...rest
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
} & Omit<SelectHTMLAttributes<HTMLSelectElement>, "value" | "onChange">) {
  return (
    <div className="relative">
      <select
        {...rest}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn(inputCls, "appearance-none pr-8 cursor-pointer")}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-fg" />
    </div>
  );
}

/* ----------------------------- Checkbox ----------------------------- */
export function Checkbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm text-fg">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 cursor-pointer rounded border-border accent-[var(--primary)]"
      />
      {label}
    </label>
  );
}

/* ----------------------------- Slider ----------------------------- */
export function Slider({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={step ?? 1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-muted accent-[var(--primary)]"
      />
      <span className="w-10 text-right font-mono text-xs text-muted-fg">{value}</span>
    </div>
  );
}

/* ----------------------------- Segmented ----------------------------- */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer",
            value === o.value ? "bg-primary text-primary-fg" : "text-muted-fg hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ----------------------------- Chip MultiSelect ----------------------------- */
export function ChipMultiSelect({
  options,
  selected,
  onChange,
  emptyMeansAll = true,
}: {
  options: string[];
  selected: string[];
  onChange: (v: string[]) => void;
  emptyMeansAll?: boolean;
}) {
  const toggle = (opt: string) => {
    onChange(selected.includes(opt) ? selected.filter((s) => s !== opt) : [...selected, opt]);
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {emptyMeansAll && (
        <button
          onClick={() => onChange([])}
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs transition-colors cursor-pointer",
            selected.length === 0
              ? "border-primary bg-primary text-primary-fg"
              : "border-border bg-surface-2 text-muted-fg hover:text-fg",
          )}
        >
          All
        </button>
      )}
      {options.map((opt) => {
        const on = selected.includes(opt);
        return (
          <button
            key={opt}
            onClick={() => toggle(opt)}
            className={cn(
              "rounded-full border px-2.5 py-1 text-xs transition-colors cursor-pointer",
              on
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface-2 text-muted-fg hover:text-fg",
            )}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

/* ----------------------------- Accordion section ----------------------------- */
export function Section({
  title,
  children,
  defaultOpen = true,
  right,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  right?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-border last:border-b-0">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setOpen(!open)}
          className="flex flex-1 items-center gap-2 py-3 text-left text-sm font-semibold text-fg cursor-pointer"
        >
          <ChevronDown
            className={cn("h-4 w-4 text-muted-fg transition-transform", !open && "-rotate-90")}
          />
          {title}
        </button>
        {right}
      </div>
      {open && <div className="space-y-3 pb-4">{children}</div>}
    </div>
  );
}

/* ----------------------------- Badge ----------------------------- */
export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "primary" | "accent" }) {
  const tones = {
    muted: "bg-muted text-muted-fg",
    primary: "bg-primary/10 text-primary",
    accent: "bg-accent/15 text-accent",
  };
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-medium", tones[tone])}>
      {children}
    </span>
  );
}
