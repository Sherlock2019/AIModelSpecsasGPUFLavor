import {
  BookOpen,
  Boxes,
  Calculator,
  Cpu,
  FolderClock,
  GitCompareArrows,
  Layers3,
  LayoutGrid,
  Menu,
  ScanEye,
  Monitor,
  Moon,
  Settings,
  Sun,
  Timer,
  type LucideIcon,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { cx } from './components/ui'
import { applyTheme, loadTheme, type ThemeChoice } from './lib/theme'
import { BenchmarksPage } from './pages/BenchmarksPage'
import { CalculatorPage } from './pages/CalculatorPage'
import { ComparePage } from './pages/ComparePage'
import { FlavorAdminPage } from './pages/FlavorAdminPage'
import { FlavorPage } from './pages/FlavorPage'
import { GpuCatalogPage } from './pages/GpuCatalogPage'
import { GuidePage } from './pages/GuidePage'
import { HomePage } from './pages/HomePage'
import { MLPage } from './pages/MLPage'
import { ModelsPage } from './pages/ModelsPage'
import { SavedPage } from './pages/SavedPage'
import { SettingsPage } from './pages/SettingsPage'
import { CalcProvider } from './state'

const NAV: { to: string; label: string; icon: LucideIcon; group?: string }[] = [
  { to: '/', label: 'AI VM Catalog', icon: LayoutGrid },
  { to: '/calculator', label: 'LLM Calculator', icon: Calculator },
  { to: '/ml', label: 'Custom ML Model', icon: ScanEye },
  { to: '/compare', label: 'Compare', icon: GitCompareArrows },
  { to: '/saved', label: 'Saved & Specs', icon: FolderClock },
  { to: '/guide', label: 'How it works', icon: BookOpen },
  { to: '/models', label: 'Models', icon: Boxes, group: 'Admin' },
  { to: '/admin/flavors', label: 'AI Flavors', icon: Layers3, group: 'Admin' },
  { to: '/gpus', label: 'GPU Inventory', icon: Cpu, group: 'Admin' },
  { to: '/benchmarks', label: 'Benchmarks', icon: Timer, group: 'Admin' },
  { to: '/settings', label: 'Settings', icon: Settings, group: 'Admin' },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="8" fill="var(--brand)" />
        <path d="M9 11h14v10H9z" fill="none" stroke="white" strokeWidth="2.4" />
        <path d="M13 8v3M19 8v3M13 21v3M19 21v3" stroke="white" strokeWidth="2.4" />
      </svg>
      <div className="leading-tight">
        <div className="text-sm font-semibold text-ink">LLM GPU Sizing</div>
        <div className="text-[11px] text-ink-3">From model to infrastructure</div>
      </div>
    </div>
  )
}

function ThemeSwitch() {
  const [theme, setTheme] = useState<ThemeChoice>(loadTheme)
  useEffect(() => applyTheme(theme), [theme])
  const options: { value: ThemeChoice; icon: LucideIcon; label: string }[] = [
    { value: 'light', icon: Sun, label: 'Light theme' },
    { value: 'system', icon: Monitor, label: 'System theme' },
    { value: 'dark', icon: Moon, label: 'Dark theme' },
  ]
  return (
    <div className="flex rounded-lg border border-line bg-surface-2 p-0.5" role="radiogroup" aria-label="Theme">
      {options.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={theme === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={cx(
            'flex h-7 flex-1 items-center justify-center rounded-md transition',
            theme === value ? 'bg-surface text-ink shadow-sm' : 'text-ink-3 hover:text-ink',
          )}
        >
          <Icon className="h-3.5 w-3.5" />
        </button>
      ))}
    </div>
  )
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  let lastGroup: string | undefined
  return (
    <CalcProvider>
      <div className="min-h-screen lg:grid lg:grid-cols-[236px_minmax(0,1fr)]">
        <aside className="border-b border-line bg-surface lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col lg:border-r lg:border-b-0">
          <div className="flex items-center justify-between px-4 py-4 lg:px-5">
            <Logo />
            <button
              type="button"
              className="rounded-md p-1.5 text-ink-2 hover:bg-surface-2 lg:hidden"
              aria-label="Toggle navigation"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen(!menuOpen)}
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>
          <nav className={cx('flex-1 px-3 pb-4 lg:block', menuOpen ? 'block' : 'hidden')}>
            {NAV.map((n) => {
              const header = n.group && n.group !== lastGroup
              lastGroup = n.group
              const Icon = n.icon
              return (
                <div key={n.to}>
                  {header && <div className="mt-5 mb-1.5 px-3 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">{n.group}</div>}
                  <NavLink
                    to={n.to}
                    end={n.to === '/'}
                    onClick={() => setMenuOpen(false)}
                    className={({ isActive }) =>
                      cx(
                        'mb-0.5 flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition',
                        isActive ? 'bg-brand-soft text-brand-text' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                      )
                    }
                  >
                    <Icon className="h-4 w-4" aria-hidden />
                    {n.label}
                  </NavLink>
                </div>
              )
            })}
            <div className="mt-6 lg:hidden">
              <ThemeSwitch />
            </div>
          </nav>
          <div className="hidden border-t border-line p-4 lg:block">
            <ThemeSwitch />
          </div>
        </aside>
        <main className="min-w-0 px-4 py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/calculator" element={<CalculatorPage />} />
            <Route path="/flavors/:id" element={<FlavorPage />} />
            <Route path="/ml" element={<MLPage />} />
            <Route path="/admin/flavors" element={<FlavorAdminPage />} />
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/gpus" element={<GpuCatalogPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/benchmarks" element={<BenchmarksPage />} />
            <Route path="/saved" element={<SavedPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </CalcProvider>
  )
}
