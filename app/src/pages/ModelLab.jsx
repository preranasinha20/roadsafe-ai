import React, { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from 'recharts'
import { Loading, SectionHead, Tip } from '../components/ui.jsx'
import { SEV, SEV_COLOR, pct, useJSON } from '../lib/data.js'

const MODEL_NAME = { dummy: 'Majority baseline', logreg: 'Logistic Regression', tree: 'Decision Tree', rf: 'Random Forest', lgbm: 'LightGBM', xgb: 'XGBoost' }
const METHOD_NAME = { none: 'No handling', class_weight: 'Class weights', undersample: 'Under-sampling', smote: 'SMOTE' }
const f3 = (v) => (v === undefined || v === null || Number.isNaN(v) ? '–' : v.toFixed(3))

function Confusion({ m, norm }) {
  const tot = m.map((r) => r.reduce((a, b) => a + b, 0))
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'auto repeat(3, 1fr)', gap: 6, alignItems: 'stretch' }}>
      <div />
      {SEV.map((c) => <div key={c} style={{ textAlign: 'center', fontSize: 12, color: 'var(--muted)', fontWeight: 600 }}>Pred. {c}</div>)}
      {m.map((row, i) => (
        <React.Fragment key={i}>
          <div style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 600, alignSelf: 'center', paddingRight: 6 }}>True {SEV[i]}</div>
          {row.map((v, j) => {
            const share = v / tot[i]
            return (
              <div key={j} style={{ padding: '18px 6px', textAlign: 'center', borderRadius: 12, background: `color-mix(in srgb, ${i === j ? 'var(--accent-2)' : 'var(--fatal)'} ${Math.round(8 + share * 70)}%, var(--card))`, border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontWeight: 700, fontSize: 16 }}>{norm ? pct(share, 1) : v.toLocaleString('en-GB')}</div>
                {!norm && <div style={{ fontSize: 11, color: 'var(--muted)' }}>{pct(share, 1)} of row</div>}
              </div>
            )
          })}
        </React.Fragment>
      ))}
    </div>
  )
}

export default function ModelLab() {
  const { data: f, error } = useJSON('final')
  const { data: grid } = useJSON('experiments')
  const [decision, setDecision] = useState('tuned')
  const [norm, setNorm] = useState(true)
  const [curve, setCurve] = useState('Fatal')
  const [sort, setSort] = useState('test_macro_f1')

  const rows = useMemo(() => (grid ? grid.filter((r) => r.decision === decision).sort((a, b) => b[sort] - a[sort]) : []), [grid, decision, sort])
  const bestF1 = rows.length ? Math.max(...rows.map((r) => r.test_macro_f1)) : 0

  const imb = useMemo(() => {
    if (!grid) return []
    const out = []
    for (const m of ['lgbm', 'xgb']) {
      for (const d of ['argmax', 'tuned']) {
        const r = { name: `${MODEL_NAME[m]} · ${d === 'tuned' ? 'tuned scales' : 'argmax'}` }
        for (const g of grid.filter((x) => x.model === m && x.decision === d)) r[g.imbalance] = g.test_macro_f1
        out.push(r)
      }
    }
    return out
  }, [grid])

  return (
    <>
      <SectionHead eyebrow="Model lab" title="Every number, on years the models never saw">
        Models are trained on 2006-2014, tuned on 2015 and scored once on 2016-2017. Comparison experiments use a 200k stratified training sample so the whole grid is
        reproducible on a laptop; final models are refit on all 1.45M training rows.
      </SectionHead>
      <Loading data={f} error={error} h={400}>
        {f && (
          <>
            <div className="grid g4">
              {[
                ['Macro-F1', f3(f.test.tuned.macro_f1), `majority baseline ${f3(f.baseline.macro_f1)}`],
                ['Recall · Fatal', pct(f.test.tuned.recall_Fatal, 1), `Serious ${pct(f.test.tuned.recall_Serious, 1)}`],
                ['ROC-AUC (OvR)', f3(f.test.tuned.roc_auc_ovr), `PR-AUC Fatal ${f3(f.test.tuned.pr_auc_Fatal)}`],
                ['Accuracy · time test', pct(f.test.argmax.accuracy, 1), `majority baseline ${pct(f.baseline.accuracy, 1)}`],
              ].map(([l, v, sub]) => (
                <div className="card glow hover" key={l}>
                  <div className="stat"><span className="v">{v}</span><span className="l">{l}</span><span className="chip" style={{ width: 'fit-content', marginTop: 6 }}>{sub}</span></div>
                </div>
              ))}
            </div>
            <div className="note warn" style={{ marginTop: 16 }}>
              <b>On accuracy.</b> {f.accuracy_note}
            </div>

            <section className="section">
              <SectionHead eyebrow="Comparison" title="Models × imbalance strategies" />
              <div className="card">
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 14 }}>
                  <div className="seg">
                    <button className={decision === 'tuned' ? 'on' : ''} onClick={() => setDecision('tuned')}>Tuned decision scales</button>
                    <button className={decision === 'argmax' ? 'on' : ''} onClick={() => setDecision('argmax')}>Plain argmax</button>
                  </div>
                  <span className="sub">Click a column header to sort · scales fitted on 2015 only</span>
                </div>
                <div className="tbl-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Model</th><th>Imbalance</th>
                        {[['test_macro_f1', 'Macro-F1'], ['test_recall_Fatal', 'Recall Fatal'], ['test_recall_Serious', 'Recall Serious'], ['test_roc_auc_ovr', 'ROC-AUC'], ['test_pr_auc_Fatal', 'PR-AUC Fatal'], ['test_balanced_accuracy', 'Bal. acc'], ['test_mcc', 'MCC'], ['test_accuracy', 'Accuracy'], ['fit_seconds', 'Fit s'], ['infer_ms_per_row', 'ms/row']].map(([k, l]) => (
                          <th key={k} style={{ cursor: 'pointer', color: sort === k ? 'var(--accent)' : undefined }} onClick={() => setSort(k)}>{l}{sort === k ? ' ↓' : ''}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={`${r.model}${r.imbalance}`}>
                          <td>{MODEL_NAME[r.model]}</td><td style={{ color: 'var(--muted)' }}>{METHOD_NAME[r.imbalance]}</td>
                          <td className={`num ${r.test_macro_f1 === bestF1 ? 'best' : ''}`}>{f3(r.test_macro_f1)}</td>
                          <td className="num">{f3(r.test_recall_Fatal)}</td><td className="num">{f3(r.test_recall_Serious)}</td>
                          <td className="num">{f3(r.test_roc_auc_ovr)}</td><td className="num">{f3(r.test_pr_auc_Fatal)}</td>
                          <td className="num">{f3(r.test_balanced_accuracy)}</td><td className="num">{f3(r.test_mcc)}</td>
                          <td className="num">{f3(r.test_accuracy)}</td><td className="num">{r.fit_seconds}</td><td className="num">{r.infer_ms_per_row}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            <section className="section">
              <div className="grid g2">
                <div className="card">
                  <h3>Effect of the imbalance strategy</h3>
                  <p className="sub">Test macro-F1 for the two boosted models. Re-weighting and tuned scales trade accuracy for recall on the rare classes.</p>
                  <div style={{ height: 330, marginTop: 10 }}>
                    <ResponsiveContainer>
                      <BarChart data={imb} margin={{ left: 0, right: 10 }}>
                        <CartesianGrid vertical={false} />
                        <XAxis dataKey="name" tick={{ fontSize: 10.5 }} interval={0} angle={-12} height={60} textAnchor="end" />
                        <YAxis domain={[0.28, 'auto']} />
                        <Tooltip content={<Tip />} cursor={{ fill: 'var(--card)' }} />
                        <Legend formatter={(v) => METHOD_NAME[v]} />
                        {['none', 'class_weight', 'undersample', 'smote'].map((k, i) => <Bar key={k} dataKey={k} name={k} fill={['#64748b', '#22d3ee', '#8b5cf6', '#fbbf24'][i]} radius={[5, 5, 0, 0]} />)}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                    <h3>Confusion matrix · 2016-17</h3>
                    <div className="seg"><button className={norm ? 'on' : ''} onClick={() => setNorm(true)}>% of true class</button><button className={!norm ? 'on' : ''} onClick={() => setNorm(false)}>Counts</button></div>
                  </div>
                  <p className="sub" style={{ margin: '4px 0 14px' }}>{f.confusion_note}</p>
                  <Confusion m={f.confusion} norm={norm} />
                </div>
              </div>
            </section>

            <section className="section">
              <div className="grid g2">
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                    <h3>ROC &amp; precision-recall</h3>
                    <div className="seg">{SEV.map((c) => <button key={c} className={curve === c ? 'on' : ''} onClick={() => setCurve(c)}>{c}</button>)}</div>
                  </div>
                  <div className="grid g2" style={{ marginTop: 12 }}>
                    {[['roc', 'ROC curve', 'False positive rate', 'True positive rate'], ['pr', 'Precision-recall', 'Recall', 'Precision']].map(([k, t, xl, yl]) => (
                      <div key={k} style={{ height: 250 }}>
                        <div className="sub" style={{ textAlign: 'center' }}>{t} · AUC {f3(f.curves[curve][k === 'roc' ? 'auc' : 'ap'])}</div>
                        <ResponsiveContainer>
                          <LineChart data={f.curves[curve][k].map(([x, y]) => ({ x, y }))} margin={{ left: -8, right: 8, bottom: 14, top: 6 }}>
                            <CartesianGrid />
                            <XAxis dataKey="x" type="number" domain={[0, 1]} tickCount={5} label={{ value: xl, position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 11 }} />
                            <YAxis domain={[0, 1]} tickCount={5} />
                            {k === 'roc' && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--faint)" strokeDasharray="4 4" />}
                            <Tooltip content={<Tip fmt={(v) => v.toFixed(3)} />} />
                            <Line dataKey="y" name={yl} dot={false} stroke={SEV_COLOR[curve]} strokeWidth={2.5} isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <h3>Calibration (reliability) · Fatal</h3>
                  <p className="sub">ECE before {f3(f.ece.raw)} → after isotonic {f3(f.ece.calibrated)} (lower is better). Class weighting inflates raw probabilities; calibration restores them.</p>
                  <div style={{ height: 290, marginTop: 8 }}>
                    <ResponsiveContainer>
                      <LineChart margin={{ left: -6, right: 8, bottom: 14 }}>
                        <CartesianGrid />
                        <XAxis dataKey="p" type="number" domain={[0, 'auto']} tickFormatter={(v) => v.toFixed(2)} label={{ value: 'mean predicted probability', position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 11 }} />
                        <YAxis type="number" domain={[0, 'auto']} tickFormatter={(v) => v.toFixed(2)} />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 0.12, y: 0.12 }]} stroke="var(--faint)" strokeDasharray="4 4" />
                        <Tooltip content={<Tip fmt={(v) => v.toFixed(4)} />} />
                        <Legend />
                        <Line data={f.reliability.raw} dataKey="o" name="Raw model" stroke="#fbbf24" strokeWidth={2.5} dot />
                        <Line data={f.reliability.calibrated} dataKey="o" name="Calibrated" stroke="#22d3ee" strokeWidth={2.5} dot />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Ablation study" title="What each design choice is worth">
                Same model, same tuned scales, one change at a time. The last row shows why leakage columns are excluded: the score looks better and means nothing.
              </SectionHead>
              <div className="card">
                <div style={{ height: 380 }}>
                  <ResponsiveContainer>
                    <BarChart data={f.ablations} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid horizontal={false} />
                      <XAxis type="number" domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
                      <YAxis type="category" dataKey="variant" width={290} tick={{ fontSize: 11.5 }} />
                      <Tooltip content={<Tip fmt={(v) => v.toFixed(4)} />} cursor={{ fill: 'var(--card)' }} />
                      <Legend />
                      <Bar dataKey="macro_f1" name="Macro-F1" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
                      <Bar dataKey="roc_auc_ovr" name="ROC-AUC" fill="#22d3ee" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>

            <section className="section">
              <div className="grid g2">
                <div className="card">
                  <h3>5-fold stratified cross-validation</h3>
                  <p className="sub" style={{ marginBottom: 12 }}>Macro-F1 on 150k training rows; paired t-test of the best model against each rival across the 5 folds.</p>
                  <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Model</th><th>Mean</th><th>Std</th><th>p vs best</th></tr></thead><tbody>
                    {f.cv.map((r) => (
                      <tr key={r.model}><td>{r.model}</td><td className={`num ${r.best ? 'best' : ''}`}>{f3(r.mean)}</td><td className="num">{f3(r.std)}</td><td className="num">{r.best ? 'best' : r.p < 0.001 ? '< 0.001' : r.p.toFixed(3)}</td></tr>
                    ))}</tbody></table></div>
                </div>
                <div className="card">
                  <h3>Other evaluation protocols</h3>
                  <p className="sub" style={{ marginBottom: 12 }}>The same model under the protocols most papers use, so accuracy figures can be compared fairly.</p>
                  <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Protocol</th><th>Accuracy</th><th>Majority</th><th>Macro-F1</th><th>AUC</th></tr></thead><tbody>
                    {f.protocols.map((r) => (
                      <tr key={r.name}><td style={{ whiteSpace: 'normal', minWidth: 210 }}>{r.name}</td><td className="num best">{pct(r.accuracy, 1)}</td><td className="num">{pct(r.majority, 1)}</td><td className="num">{f3(r.macro_f1)}</td><td className="num">{f3(r.auc)}</td></tr>
                    ))}</tbody></table></div>
                </div>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Error analysis" title="Where the model is weakest" />
              <div className="tbl-wrap">
                <table className="tbl"><thead><tr><th>Subgroup</th><th>Accidents</th><th>Fatal cases</th><th>Fatal recall</th><th>KSI recall</th><th>Accuracy</th></tr></thead><tbody>
                  {f.errors.map((r) => (
                    <tr key={r.group}><td>{r.group}</td><td className="num">{r.n.toLocaleString('en-GB')}</td><td className="num">{r.fatal_n}</td><td className="num">{pct(r.recall_Fatal, 1)}</td><td className="num">{pct(r.recall_KSI, 1)}</td><td className="num">{pct(r.accuracy, 1)}</td></tr>
                  ))}</tbody></table>
              </div>
            </section>
          </>
        )}
      </Loading>
    </>
  )
}
