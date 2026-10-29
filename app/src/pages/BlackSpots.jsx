import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Download, Flame, Layers, Search } from 'lucide-react'
import MapBase from '../components/MapBase.jsx'
import { SectionHead, Skeleton } from '../components/ui.jsx'
import { loadJSON, num, pct, useJSON } from '../lib/data.js'

const sevColor = (k) => (k >= 0.3 ? '#fb4b6e' : k >= 0.2 ? '#f97316' : k >= 0.15 ? '#fbbf24' : '#34d399')

function profile(s) {
  const bits = []
  bits.push(`${s.urban === 'Urban' ? 'Urban' : 'Rural'} ${s.road_type.toLowerCase()}`)
  bits.push(`${s.speed_limit} mph`)
  bits.push(s.junction.startsWith('Not at') ? 'no junction' : s.junction.toLowerCase())
  if (s.night_share > 0.35) bits.push(`${Math.round(s.night_share * 100)}% after dark`)
  bits.push(`peak ${String(s.peak_hour).padStart(2, '0')}:00`)
  return bits.join(' · ')
}

function csv(rows) {
  const keys = Object.keys(rows[0])
  return [keys.join(','), ...rows.map((r) => keys.map((k) => JSON.stringify(r[k] ?? '')).join(','))].join('\n')
}

export default function BlackSpots() {
  const { data: spots } = useJSON('blackspots')
  const { data: summary } = useJSON('summary')
  const [topN, setTopN] = useState(100)
  const [minKsi, setMinKsi] = useState(0)
  const [query, setQuery] = useState('')
  const [urban, setUrban] = useState('All')
  const [persistOnly, setPersistOnly] = useState(false)
  const [heatOn, setHeatOn] = useState(true)
  const [sel, setSel] = useState(null)
  const [ready, setReady] = useState(false)
  const ref = useRef({})

  const rows = useMemo(() => {
    if (!spots) return []
    return spots
      .filter((s) => s.ksi_share >= minKsi / 100 && (urban === 'All' || s.urban === urban) && (!persistOnly || s.persistent) &&
        (!query || s.la.toLowerCase().includes(query.toLowerCase())))
      .slice(0, topN)
  }, [spots, topN, minKsi, urban, persistOnly, query])

  // draw layers
  useEffect(() => {
    const { map, L, layer } = ref.current
    if (!map || !layer) return
    layer.clearLayers()
    rows.forEach((s) => {
      const r = 5 + Math.sqrt(s.score) * 0.55
      const m = L.circleMarker([s.lat, s.lon], { radius: Math.min(r, 26), color: '#fff', weight: 1, fillColor: sevColor(s.ksi_share), fillOpacity: 0.82 })
      m.bindPopup(
        `<b>#${s.rank} · ${s.la}</b><br/>${profile(s)}<hr style="border:0;border-top:1px solid var(--border);margin:8px 0"/>` +
          `<span class="mono">${num(s.accidents)}</span> accidents in 10 yrs · <b style="color:#fb4b6e">${s.fatal} fatal</b>, ${s.serious} serious<br/>` +
          `KSI share <b>${pct(s.ksi_share)}</b> · severity score <b>${num(s.score)}</b><br/>` +
          `Later years (2015-17): ${s.valid_per_year.toFixed(1)}/yr vs ${s.per_year.toFixed(1)}/yr · ${s.persistent ? '✅ persistent' : '↘ faded'}`,
      )
      m.on('click', () => setSel(s.rank))
      m.addTo(layer)
    })
  }, [rows, ready])

  useEffect(() => {
    const { map, heat } = ref.current
    if (!map || !heat) return
    if (heatOn) heat.addTo(map)
    else map.removeLayer(heat)
  }, [heatOn, ready])

  const fly = (s) => {
    setSel(s.rank)
    const { map } = ref.current
    map?.flyTo([s.lat, s.lon], 15, { duration: 1.1 })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const tot = useMemo(() => rows.reduce((a, s) => ({ acc: a.acc + s.accidents, fatal: a.fatal + s.fatal }), { acc: 0, fatal: 0 }), [rows])

  return (
    <>
      <SectionHead eyebrow="Black-spot detection" title="Where severe accidents concentrate">
        A sliding-window detector scans 1.64 million accident locations from 2005-2014 (150 m radius, Fatal = 10, Serious = 5, Slight = 1) and keeps the highest-scoring,
        non-overlapping sites. They were then checked against 2015-2017, which the detector never saw.
      </SectionHead>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0,1.6fr) minmax(0,1fr)' }} id="bs-grid">
        <style>{'@media (max-width: 1000px){#bs-grid{grid-template-columns:minmax(0,1fr)!important}}'}</style>
        <div className="card" style={{ padding: 10 }}>
          <div style={{ height: 'min(70vh, 640px)', minHeight: 420 }}>
            <MapBase
              onReady={async (map, L) => {
                const layer = L.layerGroup().addTo(map)
                const pts = await loadJSON('heat')
                await import('leaflet.heat')
                const heat = L.heatLayer(pts, { radius: 9, blur: 12, maxZoom: 11, max: 4, minOpacity: 0.05, gradient: { 0.25: '#312e81', 0.5: '#7c3aed', 0.75: '#f97316', 1: '#fb4b6e' } })
                heat.addTo(map)
                ref.current = { map, L, layer, heat }
                setReady(true)
              }}
            />
          </div>
        </div>

        <div className="grid" style={{ alignContent: 'start' }}>
          <div className="card">
            <h3>Filters</h3>
            <div className="grid" style={{ marginTop: 12, gap: 14 }}>
              <div className="field">
                <label>Top black spots shown <b>{topN}</b></label>
                <input type="range" min="10" max="500" step="10" value={topN} onChange={(e) => setTopN(+e.target.value)} />
              </div>
              <div className="field">
                <label>Minimum Serious+Fatal share <b>{minKsi}%</b></label>
                <input type="range" min="0" max="50" step="1" value={minKsi} onChange={(e) => setMinKsi(+e.target.value)} />
              </div>
              <div className="field">
                <label>Local authority contains</label>
                <div style={{ position: 'relative' }}>
                  <Search size={15} style={{ position: 'absolute', left: 12, top: 13, color: 'var(--faint)' }} />
                  <input type="text" style={{ paddingLeft: 34 }} placeholder="e.g. Leeds" value={query} onChange={(e) => setQuery(e.target.value)} />
                </div>
              </div>
              <div className="seg">
                {['All', 'Urban', 'Rural'].map((u) => (
                  <button key={u} className={urban === u ? 'on' : ''} onClick={() => setUrban(u)}>{u}</button>
                ))}
              </div>
              <label className="switch"><input type="checkbox" checked={persistOnly} onChange={(e) => setPersistOnly(e.target.checked)} /> Only spots that persisted in 2015-17</label>
              <label className="switch"><input type="checkbox" checked={heatOn} onChange={(e) => setHeatOn(e.target.checked)} /> <Flame size={15} /> Severity heatmap layer</label>
            </div>
          </div>
          <div className="card">
            <h3><Layers size={16} style={{ verticalAlign: -2, marginRight: 6, color: 'var(--accent)' }} />Validation</h3>
            {summary ? (
              <div style={{ display: 'grid', gap: 10, marginTop: 10, fontSize: 14 }}>
                <div className="note">{summary.blackspots.persistence_pct} of the top-100 spots still averaged ≥ 1 accident a year and ≥ half their earlier rate in 2015-17 (the persistence criterion is lenient: ordinary dense sites at rank 1000 score {summary.blackspots.control_pct}).</div>
                <div className="note">{pct(summary.blackspots.kde_overlap, 0)} of the top-50 coincide with an independent KDE hotspot, and {pct(summary.blackspots.dbscan_overlap, 0)} fall inside a DBSCAN cluster.</div>
                <div className="note">Top-100 spots hold {pct(summary.blackspots.top100_share, 1)} of all accidents. Their Serious+Fatal share is {pct(summary.blackspots.ksi_share_top100, 1)} (overall {pct(summary.blackspots.ksi_share_all, 1)}): the busiest sites are urban junctions with many slight accidents.</div>
              </div>
            ) : <Skeleton h={90} />}
            <div className="chip" style={{ marginTop: 14 }}>Showing {rows.length} · {num(tot.acc)} accidents · {tot.fatal} fatal</div>
          </div>
        </div>
      </div>

      <section className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'end', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
          <div>
            <span className="eyebrow">Ranked list</span>
            <h3 style={{ fontSize: 24, marginTop: 4 }}>Black spots by severity-weighted score</h3>
          </div>
          <button className="btn sm" disabled={!rows.length} onClick={() => {
            const blob = new Blob([csv(rows)], { type: 'text/csv' })
            const a = document.createElement('a')
            a.href = URL.createObjectURL(blob)
            a.download = 'black_spots.csv'
            a.click()
          }}><Download size={14} /> Download CSV</button>
        </div>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead>
              <tr>{['#', 'Local authority', 'Profile', 'Accidents', 'Fatal', 'Serious', 'KSI share', 'Score', 'Radius', '2015-17 /yr', ''].map((h) => <th key={h}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {rows.slice(0, 200).map((s) => (
                <tr key={s.rank} onClick={() => fly(s)} style={{ cursor: 'pointer', background: sel === s.rank ? 'var(--card)' : undefined }}>
                  <td className="num">{s.rank}</td>
                  <td>{s.la}</td>
                  <td style={{ whiteSpace: 'normal', minWidth: 260, color: 'var(--muted)' }}>{profile(s)}</td>
                  <td className="num">{num(s.accidents)}</td>
                  <td className="num" style={{ color: 'var(--fatal)' }}>{s.fatal}</td>
                  <td className="num">{s.serious}</td>
                  <td className="num"><i className="dot" style={{ background: sevColor(s.ksi_share), marginRight: 6 }} />{pct(s.ksi_share)}</td>
                  <td className="num">{num(s.score)}</td>
                  <td className="num">{Math.round(s.radius_m)} m</td>
                  <td className="num">{s.valid_per_year.toFixed(1)}</td>
                  <td>{s.persistent ? '✅' : '↘'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="sub" style={{ marginTop: 12 }}>
          A black spot shows <b>concentration</b>, not cause, and the data has no traffic volumes, so it cannot say which road is riskiest per vehicle.
        </p>
      </section>
    </>
  )
}
