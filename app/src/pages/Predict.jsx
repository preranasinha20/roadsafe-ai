import React, { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Info, MapPin, RotateCcw, Sparkles } from 'lucide-react'
import MapBase from '../components/MapBase.jsx'
import { SectionHead, Skeleton } from '../components/ui.jsx'
import { loadJSON, SEV, SEV_COLOR, pct } from '../lib/data.js'
import { buildEncoder, groupShap, predict, shapForClass } from '../lib/model.js'
import { DAYS, DEFAULTS, MONTHS, PRESETS, buildRow, nearestLA } from '../lib/features.js'

const NICE = {
  speed_limit: 'Speed limit', hour: 'Hour of day', lat: 'Latitude', lon: 'Longitude', h_cell_ksi_rate: 'Local KSI history',
  h_cell_n: 'Local accident count', h_cell_score: 'Local severity score', h_la_ksi_rate: 'District KSI history',
  h_la_n: 'District accident count', h_la_score: 'District severity score', h_dist_hotspot_m: 'Distance to hotspot',
  h_in_hotspot: 'Inside hotspot', n_vehicles: 'Vehicles involved', dow: 'Day of week', month: 'Month',
  Light_Conditions: 'Light conditions', Road_Type: 'Road type', Junction_Detail: 'Junction type', Junction_Control: 'Junction control',
  Weather_Conditions: 'Weather', Road_Surface_Conditions: 'Road surface', '1st_Road_Class': 'Road class', '2nd_Road_Class': '2nd road class',
  Urban_or_Rural_Area: 'Urban / rural', la_district: 'Local authority', Police_Force: 'Police force', season: 'Season',
  speed_band: 'Speed band', is_night: 'Night', is_weekend: 'Weekend', is_rush_hour: 'Rush hour', wet_surface: 'Wet surface',
  bad_weather: 'Bad weather', dark_and_unlit: 'Dark & unlit', junction_at_night: 'Junction at night', high_speed_rural: 'High-speed rural',
  is_urban: 'Urban', hour_sin: 'Hour (sin)', hour_cos: 'Hour (cos)', Carriageway_Hazards: 'Carriageway hazards',
  Special_Conditions_at_Site: 'Special conditions', 'Pedestrian_Crossing-Human_Control': 'Ped. crossing (human)',
  'Pedestrian_Crossing-Physical_Facilities': 'Ped. crossing (physical)',
}
const niceVal = (col, v) => {
  if (col === 'dow') return DAYS[v]
  if (col === 'month') return MONTHS[v - 1]
  if (col === 'hour') return `${String(v).padStart(2, '0')}:00`
  if (col === 'h_dist_hotspot_m') return v > 5000 ? '> 5 km' : `${Math.round(v)} m`
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  return String(v)
}

function Select({ label, value, options, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </div>
  )
}

function Waterfall({ contrib, base, total, cls }) {
  const rows = contrib.slice(0, 8)
  const rest = contrib.slice(8).reduce((a, b) => a + b.shap, 0)
  const all = rest !== 0 ? [...rows, { feature: `${contrib.length - 8} other features`, value: '', shap: rest }] : rows
  const max = Math.max(...all.map((r) => Math.abs(r.shap)), 0.05)
  return (
    <div>
      <div className="sub" style={{ marginBottom: 10, fontSize: 12.5 }}>
        Contribution of each factor to the model's <b style={{ color: SEV_COLOR[cls] }}>{cls}</b> score (log-odds). Base value {base.toFixed(2)} → final {total.toFixed(2)}.
      </div>
      {all.map((r) => (
        <div key={r.feature} style={{ display: 'grid', gridTemplateColumns: 'minmax(120px,1.2fr) 2fr 58px', gap: 10, alignItems: 'center', margin: '7px 0', fontSize: 13 }}>
          <div style={{ lineHeight: 1.25 }}>
            {r.feature}
            {r.value !== '' && <div className="mono" style={{ fontSize: 11.5, color: 'var(--faint)' }}>{r.value}</div>}
          </div>
          <div style={{ position: 'relative', height: 14, background: 'var(--border)', borderRadius: 8 }}>
            <div style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--border-strong)' }} />
            <motion.div
              layout
              initial={false}
              animate={{ width: `${(Math.abs(r.shap) / max) * 50}%` }}
              style={{
                position: 'absolute', top: 0, bottom: 0, borderRadius: 8,
                [r.shap >= 0 ? 'left' : 'right']: '50%',
                background: r.shap >= 0 ? 'linear-gradient(90deg,#fb4b6e,#f97316)' : 'linear-gradient(270deg,#22d3ee,#6366f1)',
              }}
            />
          </div>
          <span className="mono" style={{ textAlign: 'right', color: r.shap >= 0 ? 'var(--fatal)' : 'var(--accent)' }}>
            {r.shap >= 0 ? '+' : ''}{r.shap.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function Predict() {
  const [assets, setAssets] = useState(null)
  const [spots, setSpots] = useState([])
  const [s, setS] = useState(DEFAULTS)
  const [shapCls, setShapCls] = useState(null)
  const [shap, setShap] = useState(null)
  const mapRef = useRef({})

  useEffect(() => {
    Promise.all([loadJSON('model'), loadJSON('meta'), loadJSON('blackspots')]).then(([model, meta, bs]) => {
      setAssets({ model, meta, enc: buildEncoder(meta.encoder) })
      setSpots(bs.slice(0, 120))
    })
  }, [])

  const set = (patch) => setS((p) => ({ ...p, ...patch }))
  const row = useMemo(() => (assets ? buildRow(s, assets.meta) : null), [s, assets])
  const res = useMemo(() => (assets ? predict(assets.model, assets.enc, row) : null), [row, assets])
  const cls = shapCls ?? (res ? (res.label === 0 ? 1 : res.label) : 1)

  // exact TreeSHAP for the chosen class (debounced so sliders stay smooth)
  useEffect(() => {
    if (!assets || !res) return
    const id = setTimeout(() => {
      const { phi, base } = shapForClass(assets.model, res.x, cls)
      const g = groupShap(assets.enc, phi)
      const contrib = Object.entries(g)
        .map(([col, v]) => ({ feature: NICE[col] || col, value: niceVal(col, row[col]), shap: v }))
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      setShap({ contrib, base, total: res.raw[cls], cls })
    }, 140)
    return () => clearTimeout(id)
  }, [assets, res, cls, row])

  // map marker follows state
  useEffect(() => {
    const { map, marker } = mapRef.current
    if (map && marker) marker.setLatLng([s.lat, s.lon])
  }, [s.lat, s.lon])

  const whatIf = useMemo(() => {
    if (!assets || !res) return []
    const ksi = res.probs[1] + res.probs[2]
    const trials = []
    if (s.speed_limit > 20) trials.push(['Lower the speed limit by 10 mph', { speed_limit: s.speed_limit - 10 }])
    if (s.Light_Conditions.startsWith('Darkness')) trials.push(['Same journey in daylight', { Light_Conditions: 'Daylight' }])
    if (s.Road_Surface_Conditions !== 'Dry') trials.push(['Dry road surface', { Road_Surface_Conditions: 'Dry', Weather_Conditions: 'Fine no high winds' }])
    if (s.Junction_Detail !== 'Not at junction or within 20 metres') trials.push(['Away from the junction', { Junction_Detail: 'Not at junction or within 20 metres' }])
    if (s.n_vehicles > 1) trials.push(['Single-vehicle instead', { n_vehicles: 1 }])
    if (!trials.length) trials.push(['Raise the speed limit by 10 mph', { speed_limit: Math.min(70, s.speed_limit + 10) }])
    return trials.slice(0, 4).map(([label, patch]) => {
      const r2 = predict(assets.model, assets.enc, buildRow({ ...s, ...patch }, assets.meta))
      return { label, d: r2.probs[1] + r2.probs[2] - ksi, dF: r2.probs[2] - res.probs[2] }
    })
  }, [s, assets, res])

  if (!assets)
    return (
      <>
        <SectionHead eyebrow="Predictor" title="Loading the model…">Downloading the gradient-boosted trees and the location history (a few MB, cached afterwards).</SectionHead>
        <Skeleton h={420} />
      </>
    )

  const { meta } = assets
  const opts = meta.options
  const ksi = res.probs[1] + res.probs[2]
  const labelName = SEV[res.label]
  const avg = meta.test_ksi_rate
  const drivers = shap?.contrib?.slice(0, 3).filter((c) => c.shap > 0).map((c) => c.feature.toLowerCase())

  return (
    <>
      <SectionHead eyebrow="Predictor" title="Estimate the severity of an accident at any location">
        Set the scene. Probabilities are isotonic-calibrated on 2015 data, and the explanation is exact TreeSHAP computed live in your browser.
      </SectionHead>

      <div className="kicker-row" style={{ marginBottom: 18 }}>
        {PRESETS.map((p) => (
          <button key={p.name} className="btn sm" onClick={() => set({ ...DEFAULTS, ...p.patch })}>
            <span>{p.emoji}</span> {p.name}
          </button>
        ))}
        <button className="btn sm" onClick={() => { setS(DEFAULTS); setShapCls(null) }}>
          <RotateCcw size={14} /> Reset
        </button>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0,1.05fr) minmax(0,1fr)', alignItems: 'start' }} id="predict-grid">
        <style>{'@media (max-width: 980px){#predict-grid{grid-template-columns:minmax(0,1fr)!important}}'}</style>

        <div className="grid">
          <div className="card">
            <h3 style={{ display: 'flex', gap: 8, alignItems: 'center' }}><MapPin size={17} color="var(--accent)" /> Location</h3>
            <p className="sub" style={{ marginBottom: 12 }}>Click the map. Local history (accident counts and KSI rate within ~500 m, distance to the nearest hotspot) is looked up automatically.</p>
            <div style={{ height: 300 }}>
              <MapBase
                center={[s.lat, s.lon]}
                zoom={8}
                onReady={(map, L) => {
                  const marker = L.circleMarker([s.lat, s.lon], { radius: 9, color: '#fff', weight: 2, fillColor: '#22d3ee', fillOpacity: 1 }).addTo(map)
                  spots.forEach((sp) => L.circle([sp.lat, sp.lon], { radius: 250, color: '#fb4b6e', weight: 1, fillOpacity: 0.25 }).addTo(map))
                  map.on('click', (e) => {
                    const la = nearestLA(meta, e.latlng.lat, e.latlng.lng)
                    setS((p) => ({ ...p, lat: +e.latlng.lat.toFixed(5), lon: +e.latlng.lng.toFixed(5), la_district: la.name }))
                  })
                  mapRef.current = { map, marker }
                }}
              />
            </div>
            <div className="grid g2" style={{ marginTop: 14 }}>
              <Select label="Local authority" value={s.la_district} options={meta.la.map((l) => l.name)} onChange={(v) => set({ la_district: v })} />
              <div className="field">
                <label>Coordinates</label>
                <input type="text" readOnly value={`${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}`} />
              </div>
            </div>
            <div className="grid g3" style={{ marginTop: 12 }}>
              <div className="stat"><span className="l">Accidents within ~500 m (history)</span><b className="mono">{row.h_cell_n}</b></div>
              <div className="stat"><span className="l">Local KSI rate</span><b className="mono">{pct(row.h_cell_ksi_rate)}</b></div>
              <div className="stat"><span className="l">Nearest black spot</span><b className="mono">{row.h_dist_hotspot_m > 5000 ? '> 5 km' : `${Math.round(row.h_dist_hotspot_m)} m`}</b></div>
            </div>
          </div>

          <div className="card">
            <h3>Road</h3>
            <div className="grid g2" style={{ marginTop: 12 }}>
              <div className="field">
                <label>Speed limit <b>{s.speed_limit} mph</b></label>
                <input type="range" min="0" max="5" step="1" value={[20, 30, 40, 50, 60, 70].indexOf(s.speed_limit)} onChange={(e) => set({ speed_limit: [20, 30, 40, 50, 60, 70][e.target.value] })} />
              </div>
              <Select label="Urban or rural" value={s.Urban_or_Rural_Area} options={['Urban', 'Rural']} onChange={(v) => set({ Urban_or_Rural_Area: v })} />
              <Select label="Road type" value={s.Road_Type} options={opts.Road_Type} onChange={(v) => set({ Road_Type: v })} />
              <Select label="Road class" value={s['1st_Road_Class']} options={opts['1st_Road_Class']} onChange={(v) => set({ '1st_Road_Class': v })} />
              <Select label="Junction" value={s.Junction_Detail} options={opts.Junction_Detail} onChange={(v) => set({ Junction_Detail: v })} />
              <Select label="Junction control" value={s.Junction_Control} options={opts.Junction_Control} onChange={(v) => set({ Junction_Control: v })} />
              <div className="field">
                <label>Vehicles involved <b>{s.n_vehicles}</b></label>
                <input type="range" min="1" max="6" value={s.n_vehicles} onChange={(e) => set({ n_vehicles: +e.target.value })} />
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Conditions &amp; time</h3>
            <div className="grid g2" style={{ marginTop: 12 }}>
              <Select label="Light" value={s.Light_Conditions} options={opts.Light_Conditions} onChange={(v) => set({ Light_Conditions: v })} />
              <Select label="Weather" value={s.Weather_Conditions} options={opts.Weather_Conditions} onChange={(v) => set({ Weather_Conditions: v })} />
              <Select label="Road surface" value={s.Road_Surface_Conditions} options={opts.Road_Surface_Conditions} onChange={(v) => set({ Road_Surface_Conditions: v })} />
              <Select label="Day" value={DAYS[s.dow]} options={DAYS} onChange={(v) => set({ dow: DAYS.indexOf(v) })} />
              <div className="field">
                <label>Hour <b>{String(s.hour).padStart(2, '0')}:00</b></label>
                <input type="range" min="0" max="23" value={s.hour} onChange={(e) => set({ hour: +e.target.value })} />
              </div>
              <Select label="Month" value={MONTHS[s.month - 1]} options={MONTHS} onChange={(v) => set({ month: MONTHS.indexOf(v) + 1 })} />
            </div>
          </div>
        </div>

        <div className="grid" style={{ position: 'sticky', top: 'calc(var(--nav-h) + 16px)' }}>
          <div className="card glow">
            <span className="eyebrow">Predicted severity</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', margin: '12px 0 6px' }}>
              <motion.span key={labelName} initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="badge-big"
                style={{ color: SEV_COLOR[labelName], borderColor: SEV_COLOR[labelName], background: `color-mix(in srgb, ${SEV_COLOR[labelName]} 14%, transparent)` }}>
                {labelName}
              </motion.span>
              <div>
                <div className="sub">Risk of death or serious injury</div>
                <div className="mono" style={{ fontSize: 28, fontWeight: 700 }}>{pct(ksi)}</div>
                <div className="sub" style={{ fontSize: 12.5 }}>
                  {(ksi / avg).toFixed(1)}× the {pct(avg)} test-period average
                </div>
              </div>
            </div>
            {SEV.map((n, k) => (
              <div className="bar-row" key={n}>
                <span style={{ color: SEV_COLOR[n], fontWeight: 600 }}>{n}</span>
                <div className="bar"><i style={{ width: `${res.probs[k] * 100}%`, background: SEV_COLOR[n] }} /></div>
                <span className="mono" style={{ textAlign: 'right' }}>{pct(res.probs[k])}</span>
              </div>
            ))}
            <p className="note" style={{ marginTop: 14 }}>
              {labelName === 'Slight'
                ? 'Most likely a slight-injury accident.'
                : `Flagged as ${labelName.toLowerCase()} because the model's decision scales (tuned on 2015 to maximise macro-F1) favour catching severe cases.`}
              {drivers?.length ? ` Main drivers pushing risk up: ${drivers.join(', ')}.` : ''}
            </p>
          </div>

          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
              <h3>Why? <span className="chip accent" style={{ marginLeft: 6 }}>live TreeSHAP</span></h3>
              <div className="seg">
                {SEV.map((n, k) => (
                  <button key={n} className={cls === k ? 'on' : ''} onClick={() => setShapCls(k)}>{n}</button>
                ))}
              </div>
            </div>
            {shap ? <Waterfall contrib={shap.contrib} base={shap.base} total={shap.total} cls={SEV[shap.cls]} /> : <Skeleton h={220} />}
          </div>

          <div className="card">
            <h3><Sparkles size={16} style={{ verticalAlign: -2, marginRight: 6, color: 'var(--accent)' }} />What-if analysis</h3>
            <p className="sub" style={{ marginBottom: 8 }}>Change in the probability of death or serious injury (model association, not proven cause).</p>
            {whatIf.map((w) => (
              <div key={w.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '9px 0', borderBottom: '1px solid var(--border)', fontSize: 14 }}>
                <span>{w.label}</span>
                <b className="mono" style={{ color: w.d <= 0 ? 'var(--slight)' : 'var(--fatal)' }}>{w.d > 0 ? '+' : ''}{(w.d * 100).toFixed(1)} pts</b>
              </div>
            ))}
            <p className="note warn" style={{ marginTop: 14, display: 'flex', gap: 8 }}>
              <Info size={15} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>Trained on UK injury accidents. It scores conditional severity given that an accident happened; it does not estimate how likely an accident is.</span>
            </p>
          </div>
        </div>
      </div>
    </>
  )
}
