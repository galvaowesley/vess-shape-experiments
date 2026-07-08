import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  LayoutGrid,
  Microscope,
  Moon,
  Palette,
  Rocket,
  Sigma,
  Sun,
  Table2,
} from "lucide-react";
import { cn } from "./lib/util";
import ChartBuilder from "./pages/ChartBuilder";
import Significance from "./pages/Significance";
import Dashboard from "./pages/Dashboard";
import Training from "./pages/Training";
import Pretraining from "./pages/Pretraining";
import Appearance from "./pages/Appearance";
import Tables from "./pages/Tables";

const NAV = [
  { to: "/pretraining", label: "Pretraining", icon: Rocket },
  { to: "/training", label: "Training", icon: FlaskConical },
  { to: "/charts", label: "Chart Builder", icon: BarChart3 },
  { to: "/tables", label: "Tables", icon: Table2 },
  { to: "/significance", label: "Significance", icon: Sigma },
  { to: "/dashboard", label: "Dashboard", icon: LayoutGrid },
  { to: "/appearance", label: "Appearance", icon: Palette },
];

const KNOWN_ROUTES = NAV.map((n) => n.to);
const LAST_ROUTE_KEY = "vesslab.lastRoute";

/** Remember the current page so a reload (e.g. an embedded browser re-opening
 *  "/") returns here instead of always snapping back to the Chart Builder. */
function useRouteMemory() {
  const location = useLocation();
  useEffect(() => {
    if (KNOWN_ROUTES.includes(location.pathname)) {
      localStorage.setItem(LAST_ROUTE_KEY, location.pathname);
    }
  }, [location.pathname]);
}

function lastRoute(): string {
  const saved = localStorage.getItem(LAST_ROUTE_KEY);
  return saved && KNOWN_ROUTES.includes(saved) ? saved : "/charts";
}

function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);
  return { dark, toggle: () => setDark((d) => !d) };
}

function Sidebar({
  dark,
  toggle,
  open,
  onToggleSidebar,
}: {
  dark: boolean;
  toggle: () => void;
  open: boolean;
  onToggleSidebar: () => void;
}) {
  return (
    <aside
      className={cn(
        "relative flex shrink-0 flex-col border-r border-border bg-surface transition-all duration-200",
        open ? "w-60" : "w-14",
      )}
    >
      {/* Logo / brand */}
      <div className="flex h-[69px] items-center gap-2.5 overflow-hidden px-3 py-5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-fg">
          <Microscope className="h-5 w-5" />
        </div>
        {open && (
          <div className="min-w-0">
            <div className="truncate text-sm font-bold leading-tight text-fg">VessShape Lab</div>
            <div className="text-[11px] text-muted-fg">Training & Evaluation</div>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-2 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            title={!open ? label : undefined}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-fg hover:bg-muted hover:text-fg",
                !open && "justify-center",
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {open && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border p-2 space-y-1">
        <button
          onClick={toggle}
          title={dark ? "Light mode" : "Dark mode"}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm font-medium text-muted-fg hover:bg-muted hover:text-fg cursor-pointer",
            !open && "justify-center",
          )}
        >
          {dark ? <Sun className="h-4 w-4 shrink-0" /> : <Moon className="h-4 w-4 shrink-0" />}
          {open && (dark ? "Light mode" : "Dark mode")}
        </button>

        {/* Sidebar collapse toggle */}
        <button
          onClick={onToggleSidebar}
          title={open ? "Collapse sidebar" : "Expand sidebar"}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm font-medium text-muted-fg hover:bg-muted hover:text-fg cursor-pointer",
            !open && "justify-center",
          )}
        >
          {open
            ? <><ChevronLeft className="h-4 w-4 shrink-0" /><span>Collapse</span></>
            : <ChevronRight className="h-4 w-4 shrink-0" />}
        </button>
      </div>
    </aside>
  );
}

export default function App() {
  const { dark, toggle } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  useRouteMemory();

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        dark={dark}
        toggle={toggle}
        open={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to={lastRoute()} replace />} />
          <Route path="/pretraining" element={<Pretraining />} />
          <Route path="/training" element={<Training />} />
          <Route path="/charts" element={<ChartBuilder />} />
          <Route path="/tables" element={<Tables />} />
          <Route path="/significance" element={<Significance />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/appearance" element={<Appearance />} />
          <Route path="*" element={<Navigate to={lastRoute()} replace />} />
        </Routes>
      </main>
    </div>
  );
}
