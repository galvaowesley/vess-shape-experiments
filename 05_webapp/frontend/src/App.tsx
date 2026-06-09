import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import {
  BarChart3,
  FlaskConical,
  LayoutGrid,
  Moon,
  Sigma,
  Sun,
  Microscope,
  Rocket,
} from "lucide-react";
import { cn } from "./lib/util";
import ChartBuilder from "./pages/ChartBuilder";
import Significance from "./pages/Significance";
import Dashboard from "./pages/Dashboard";
import Training from "./pages/Training";
import Pretraining from "./pages/Pretraining";

const NAV = [
  { to: "/pretraining", label: "Pretraining", icon: Rocket },
  { to: "/training", label: "Training", icon: FlaskConical },
  { to: "/charts", label: "Chart Builder", icon: BarChart3 },
  { to: "/significance", label: "Significance", icon: Sigma },
  { to: "/dashboard", label: "Dashboard", icon: LayoutGrid },
];

function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);
  return { dark, toggle: () => setDark((d) => !d) };
}

function Sidebar({ dark, toggle }: { dark: boolean; toggle: () => void }) {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-fg">
          <Microscope className="h-5 w-5" />
        </div>
        <div>
          <div className="text-sm font-bold leading-tight text-fg">VessShape Lab</div>
          <div className="text-[11px] text-muted-fg">Training & Evaluation</div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-fg hover:bg-muted hover:text-fg",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border p-3">
        <button
          onClick={toggle}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-fg hover:bg-muted hover:text-fg cursor-pointer"
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          {dark ? "Light mode" : "Dark mode"}
        </button>
      </div>
    </aside>
  );
}

export default function App() {
  const { dark, toggle } = useTheme();
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar dark={dark} toggle={toggle} />
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/charts" replace />} />
          <Route path="/pretraining" element={<Pretraining />} />
          <Route path="/training" element={<Training />} />
          <Route path="/charts" element={<ChartBuilder />} />
          <Route path="/significance" element={<Significance />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </main>
    </div>
  );
}
