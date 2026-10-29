import React, { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { Loading, SectionHead, Tip } from '../components/ui.jsx'
import { SEV, SEV_COLOR, pct, useJSON } from '../lib/data.js'

export default function Explain() {
  const { data, error } = useJSON('explain')
  const [cls, setCls] = useState('Fatal')
  const [dep, setDep] = useState('speed_limit')

  return (
    <>
      <SectionHead eyebrow="Explainability" title="What does the model actually pay attention to?">
        SHAP attributes each prediction to its inputs using Shapley values. These plots explain the <b>model</b>, not the real world, so a large SHAP
        value is evidence about what the model learned, never proof of cause.
      </SectionHead>
      <Loading data={data} error={error} h={420}>
        {data && (
          <>
            <div className="grid g2">
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                  <h3>Global importance (mean |SHAP|)</h3>
                  <div className="seg">
                    {SEV.map((c) => <button key={c} className={cls === c ? 'on' : ''} onClick={() => setCls(c)}>{c}</button>)}
                  </div>
                </div>
                <div style={{ height: 420, marginTop: 10 }}>
                  <ResponsiveContainer>
                    <BarChart data={[...data.importance[cls]].reverse()} layout="vertical" margin={{ left: 30, right: 14 }}>
                      <CartesianGrid horizontal={false} />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
                      <Tooltip content={<Tip fmt={(v) => v.toFixed(3)} />} cursor={{ fill: 'var(--card)' }} />
                      <Bar dataKey="value" name="mean |SHAP|" radius={[0, 8, 8, 0]} fill={SEV_COLOR[cls]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="note" style={{ marginTop: 8 }}>
                  Spatial history features (how severe earlier accidents nearby were) and the speed limit dominate; the weather matters far less than people expect.
                </p>
              </div>

              <div className="card imgcard">
                <h3>SHAP beeswarm: Fatal class</h3>
                <p className="sub" style={{ marginBottom: 10 }}>Each dot is one accident. Right of zero means the feature raised the model's fatal score for that accident.</p>
                <img src={`${import.meta.env.BASE_URL}figures/21_shap_beeswarm_fatal.png`} alt="SHAP beeswarm plot for the Fatal class" loading="lazy" />
              </div>
            </div>

            <section className="section">
              <SectionHead eyebrow="Dependence" title="How a single feature moves the fatal score" />
              <div className="card">
                <div className="seg" style={{ marginBottom: 12 }}>
                  {Object.entries(data.dependence).map(([k, v]) => (
                    <button key={k} className={dep === k ? 'on' : ''} onClick={() => setDep(k)}>{v.label}</button>
                  ))}
                </div>
                <div style={{ height: 340 }}>
                  <ResponsiveContainer>
                    <ScatterChart margin={{ left: 10, right: 20, bottom: 18 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="x" name={data.dependence[dep].label} domain={['auto', 'auto']} label={{ value: data.dependence[dep].label, position: 'insideBottom', offset: -10, fill: 'var(--muted)' }} />
                      <YAxis type="number" dataKey="y" name="SHAP" label={{ value: 'SHAP value (Fatal)', angle: -90, position: 'insideLeft', fill: 'var(--muted)' }} />
                      <Tooltip content={<Tip fmt={(v) => v.toFixed(3)} />} />
                      <Legend />
                      <Scatter name="Daytime" fill="#22d3ee" fillOpacity={0.55}
                        data={data.dependence[dep].x.map((x, i) => ({ x, y: data.dependence[dep].y[i], n: data.dependence[dep].night[i] })).filter((p) => !p.n)} />
                      <Scatter name="Night" fill="#fb4b6e" fillOpacity={0.55}
                        data={data.dependence[dep].x.map((x, i) => ({ x, y: data.dependence[dep].y[i], n: data.dependence[dep].night[i] })).filter((p) => p.n)} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Black spots" title="Do black spots have different risk drivers?">
                {pct(data.inside_share_of_accidents, 1)} of sampled test accidents fall inside a detected black spot. Their Serious+Fatal rate is {pct(data.inside_ksi_rate, 1)} versus{' '}
                {pct(data.outside_ksi_rate, 1)} elsewhere.
              </SectionHead>
              <div className="card">
                <div style={{ height: 380 }}>
                  <ResponsiveContainer>
                    <BarChart data={[...data.inside_outside].reverse().map((d) => ({ ...d, inside: d.inside * 100, outside: d.outside * 100 }))} layout="vertical" margin={{ left: 30, right: 14 }}>
                      <CartesianGrid horizontal={false} />
                      <XAxis type="number" unit="%" />
                      <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
                      <Tooltip content={<Tip fmt={(v) => `${v.toFixed(1)}%`} />} cursor={{ fill: 'var(--card)' }} />
                      <Legend />
                      <Bar dataKey="outside" name="Outside black spots" fill="#6366f1" radius={[0, 6, 6, 0]} />
                      <Bar dataKey="inside" name="Inside black spots" fill="#fb4b6e" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="sub">Share of total |SHAP| attributed to each feature for the Fatal class.</div>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Local explanations" title="Four real test accidents, explained" />
              <div className="grid g2">
                {data.local.map((c) => (
                  <div className="card" key={c.title}>
                    <h3>{c.title}</h3>
                    <div className="sub" style={{ marginBottom: 10 }}>
                      P(Slight / Serious / Fatal) = {c.probs.map((p) => pct(p, 0)).join(' / ')} · explaining <b style={{ color: SEV_COLOR[c.class] }}>{c.class}</b>
                    </div>
                    {c.contrib.map((r) => (
                      <div key={r.feature} style={{ display: 'grid', gridTemplateColumns: '1.4fr 1.6fr 52px', gap: 8, alignItems: 'center', margin: '6px 0', fontSize: 12.5 }}>
                        <span>{r.feature}<span className="mono" style={{ color: 'var(--faint)' }}> = {r.value.length > 18 ? `${r.value.slice(0, 16)}…` : r.value}</span></span>
                        <div style={{ height: 10, borderRadius: 6, background: 'var(--border)', position: 'relative' }}>
                          <div style={{ position: 'absolute', top: 0, bottom: 0, left: r.shap >= 0 ? '50%' : `${50 - Math.min(50, Math.abs(r.shap) * 22)}%`, width: `${Math.min(50, Math.abs(r.shap) * 22)}%`, borderRadius: 6, background: r.shap >= 0 ? 'var(--fatal)' : 'var(--accent)' }} />
                        </div>
                        <span className="mono" style={{ textAlign: 'right' }}>{r.shap > 0 ? '+' : ''}{r.shap.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="SHAP vs LIME" title="Two explainers, compared honestly">
                LIME fits a small local surrogate model around each case by random sampling, so it can change between runs. SHAP is deterministic for tree models.
              </SectionHead>
              <div className="tbl-wrap">
                <table className="tbl">
                  <thead><tr><th>Case</th><th>Class</th><th>SHAP top-5</th><th>LIME top-5</th><th>Overlap</th><th>LIME stability</th></tr></thead>
                  <tbody>
                    {data.lime.map((r) => (
                      <tr key={r.case}>
                        <td className="num">{r.case}</td><td>{r.explained_class}</td>
                        <td style={{ whiteSpace: 'normal', minWidth: 220 }}>{r.shap_top5}</td>
                        <td style={{ whiteSpace: 'normal', minWidth: 220 }}>{r.lime_top5}</td>
                        <td className="num">{pct(r.shap_lime_top5_overlap, 0)}</td>
                        <td className="num">{r.lime_stability_jaccard.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note warn" style={{ marginTop: 14 }}>
                Stability is the mean Jaccard overlap of LIME's top-5 features across five random seeds (1.00 = identical every time).
              </p>
            </section>
          </>
        )}
      </Loading>
    </>
  )
}
