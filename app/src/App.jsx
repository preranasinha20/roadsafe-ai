import React, { Suspense, lazy, useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { BarChart3, Brain, Database, Home as HomeIcon, Map as MapIcon, Menu, Moon, Sparkles, Sun, X, Gauge } from 'lucide-react'
import Home from './pages/Home.jsx'
import { Skeleton } from './components/ui.jsx'

const Predict = lazy(() => import('./pages/Predict.jsx'))
const BlackSpots = lazy(() => import('./pages/BlackSpots.jsx'))
const Explain = lazy(() => import('./pages/Explain.jsx'))
const ModelLab = lazy(() => import('./pages/ModelLab.jsx'))
const DataPage = lazy(() => import('./pages/DataPage.jsx'))

const LINKS = [
  { to: '/', label: 'Overview', icon: HomeIcon },
  { to: '/predict', label: 'Predictor', icon: Gauge },
  { to: '/map', label: 'Black spots', icon: MapIcon },
  { to: '/explain', label: 'Explainability', icon: Brain },
  { to: '/models', label: 'Model lab', icon: BarChart3 },
  { to: '/data', label: 'Data & method', icon: Database },
]

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('rs-theme') || 'dark'
    } catch {
      return 'dark'
    }
  })
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('rs-theme', theme)
    } catch {}
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

export default function App() {
  const [theme, toggle] = useTheme()
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => {
    setOpen(false)
    window.scrollTo({ top: 0 })
  }, [loc.pathname])

  return (
    <>
      <div className="bg-orbs" />
      <div className="bg-grid" />
      <header className="nav">
        <div className="wrap nav-in">
          <NavLink to="/" className="brand">
            <span className="brand-mark">
              <Sparkles size={18} color="#fff" />
            </span>
            <span>
              RoadSafe AI
              <small>Explainable accident intelligence</small>
            </span>
          </NavLink>
          <nav className={`nav-links ${open ? 'open' : ''}`}>
            {LINKS.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'active' : '')}>
                <Icon size={16} /> {label}
              </NavLink>
            ))}
          </nav>
          <button className="icon-btn" onClick={toggle} aria-label="Toggle theme" title="Toggle theme">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button className="icon-btn burger" onClick={() => setOpen(!open)} aria-label="Menu">
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </header>

      <main className="page">
        <div className="wrap fade-in" key={loc.pathname}>
          <Suspense fallback={<Skeleton h={420} />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/map" element={<BlackSpots />} />
              <Route path="/explain" element={<Explain />} />
              <Route path="/models" element={<ModelLab />} />
              <Route path="/data" element={<DataPage />} />
              <Route path="*" element={<Home />} />
            </Routes>
          </Suspense>
        </div>
      </main>

      <footer className="footer">
        <div className="wrap">
          <span>
            <b>RoadSafe AI</b> · AUCML case-study project · Symbiosis Institute of Technology, Pune
          </span>
          <span>
            Data: UK Department for Transport road safety statistics (Open Government Licence). Predictions support planning
            decisions; they do not establish causes.
          </span>
        </div>
      </footer>
    </>
  )
}
