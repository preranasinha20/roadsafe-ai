import React, { useMemo, useState } from 'react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Loading, SectionHead, Tip } from '../components/ui.jsx'
import { SEV, SEV_COLOR, num, pct, useJSON } from '../lib/data.js'

const REFS = [
  'Department for Transport (2018). Road Safety Data: STATS19, 2005-2017. Open Government Licence v3.0. data.gov.uk',
  'Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.',
  'Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). “Why should I trust you?”: explaining the predictions of any classifier. KDD.',
  'Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.',
  'Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. KDD.',
  'Chawla, N. V. et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR, 16.',
  'Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD.',
  'Campello, R., Moulavi, D. & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD.',
  'Akiba, T. et al. (2019). Optuna: a next-generation hyperparameter optimization framework. KDD.',
  'Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.',
  'Ministry of Road Transport & Highways, India. Road Accidents in India (annual reports).',
]

function Chart({ title, data, keys = SEV, stacked = true, h = 260, kind = 'bar', note }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      {note && <div className="sub">{note}</div>}
      <div style={{ height: h, marginTop: 10 }}>
        <ResponsiveContainer>
          {kind === 'line' ? (
            <LineChart data={data} margin={{ left: -10, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip content={<Tip fmt={(v) => v.toFixed(2)} />} />
              <Legend />
              {keys.map((k) => <Line key={k} dataKey={k} stroke={SEV_COLOR[k]} strokeWidth={2.5} dot />)}
            </LineChart>
          ) : (
            <BarChart data={data} margin={{ left: -10, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip content={<Tip fmt={(v) => v.toFixed(1)} />} cursor={{ fill: 'var(--card)' }} />
              <Legend />
              {keys.map((k) => <Bar key={k} dataKey={k} stackId={stacked ? 'a' : undefined} fill={SEV_COLOR[k]} />)}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function DataPage() {
  const { data: d, error } = useJSON('data')
  const { data: eda } = useJSON('eda')
  const [q, setQ] = useState('')
  const dict = useMemo(() => (d ? d.dictionary.filter((r) => (r.feature + r.meaning).toLowerCase().includes(q.toLowerCase())) : []), [d, q])

  return (
    <>
      <SectionHead eyebrow="Problem · data · method" title="From a vague idea to a measurable problem" />

      <div className="grid g2">
        <div className="card glow">
          <span className="eyebrow">Problem statement</span>
          <h3 style={{ fontSize: 22, margin: '8px 0' }}>Refined from Phase I</h3>
          <p className="sub">
            Road authorities have to decide where to spend limited safety budgets and which conditions deserve intervention. Raw accident counts say where crashes
            happen, but not <b>how bad</b> they are or <b>why</b>. We build one framework that (a) predicts whether an accident is Slight, Serious or Fatal from conditions
            known at the scene, (b) locates severity-weighted black spots, and (c) explains both with SHAP and LIME.
          </p>
          <div style={{ marginTop: 14, display: 'grid', gap: 8 }}>
            <div className="note"><b>Research gap.</b> Most published work reports overall accuracy on a 85/14/1 imbalanced target, studies prediction and hotspots separately, and uses black-box models.</div>
            <div className="note"><b>Stakeholders.</b> Highway authorities, traffic police, urban planners, insurers and road-safety NGOs.</div>
            <div className="note warn"><b>Scope.</b> UK STATS19 data (2005-2017) is used because it is the largest open, labelled source. India's MoRTH statistics motivate the work, and the framework transfers; a trained model would need local data.</div>
          </div>
        </div>
        <div className="card">
          <span className="eyebrow">Objectives &amp; research questions</span>
          <ol style={{ margin: '12px 0 0', paddingLeft: 20, color: 'var(--muted)', display: 'grid', gap: 8 }}>
            <li>Predict severity with metrics suited to class imbalance (macro-F1, recall on Fatal/Serious, PR-AUC), validated on later years.</li>
            <li>Compare imbalance strategies and model families fairly, on identical features and splits.</li>
            <li>Detect black spots with density clustering and prove they persist out-of-time.</li>
            <li>Explain predictions globally and locally, and compare SHAP with LIME.</li>
          </ol>
          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            {[
              'RQ1. Which conditions most increase the probability of a Serious or Fatal outcome?',
              'RQ2. Do black spots differ in their risk drivers from other locations?',
              'RQ3. Does adding leak-free spatial history improve severity prediction?',
            ].map((r) => <div key={r} className="chip" style={{ borderRadius: 12, padding: '9px 13px', whiteSpace: 'normal' }}>{r}</div>)}
          </div>
        </div>
      </div>

      <Loading data={d} error={error} h={300}>
        {d && (
          <>
            <section className="section">
              <SectionHead eyebrow="Dataset" title="UK Road Safety: accidents 2005-2017">
                Source: Department for Transport STATS19 (Open Government Licence), via the Kaggle mirror by tsiaras. One row is one police-reported injury accident.
              </SectionHead>
              <div className="grid g4">
                {d.facts.map(([v, l]) => <div className="card hover" key={l}><div className="stat"><span className="v">{v}</span><span className="l">{l}</span></div></div>)}
              </div>
            </section>

            <section className="section">
              <div className="grid g2">
                <div className="card">
                  <h3>Cleaning log</h3>
                  <p className="sub" style={{ marginBottom: 12 }}>Every step records how many rows survive; only {pct(1 - d.funnel[d.funnel.length - 1].rows / d.funnel[0].rows, 3)} of rows were lost.</p>
                  <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Step</th><th>Rows remaining</th></tr></thead><tbody>
                    {d.funnel.map((r) => <tr key={r.step}><td>{r.step}</td><td className="num">{num(r.rows)}</td></tr>)}</tbody></table></div>
                </div>
                <div className="card">
                  <h3>Validation strategy</h3>
                  <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
                    {d.splits.map((s) => (
                      <div key={s.name} style={{ display: 'grid', gridTemplateColumns: '90px 1fr', gap: 12, alignItems: 'center' }}>
                        <span className="chip accent" style={{ justifyContent: 'center' }}>{s.name}</span>
                        <div><b>{s.years}</b> <span className="sub">· {num(s.n)} accidents · {s.use}</span></div>
                      </div>
                    ))}
                  </div>
                  <p className="note" style={{ marginTop: 16 }}>Time-based splitting mimics deployment: the model only ever predicts the future. 2005 only seeds the spatial history features.</p>
                </div>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Leakage control" title="Which columns are allowed to predict severity?">
                A feature is only used if it is known at, or describes, the scene. Anything produced by the accident's outcome is excluded.
              </SectionHead>
              <div className="tbl-wrap">
                <table className="tbl"><thead><tr><th>Feature</th><th>Type</th><th>Known before outcome</th><th>Used</th><th>Reason</th></tr></thead><tbody>
                  {d.inclusion.map((r) => (
                    <tr key={r.feature}><td className="mono">{r.feature}</td><td>{r.type}</td><td>{r.pre ? '✅' : '❌'}</td><td>{r.used ? '✅' : '🚫'}</td><td style={{ whiteSpace: 'normal', minWidth: 300, color: 'var(--muted)' }}>{r.reason}</td></tr>
                  ))}</tbody></table>
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Feature engineering" title="Features built from the raw columns">
                Temporal flags, interaction flags, banded speed limits, cyclical hour encoding and expanding-window spatial history, all documented below.
              </SectionHead>
              <div className="grid g3">
                {d.engineered.map((g) => (
                  <div className="card" key={g.group}><h3>{g.group}</h3><ul style={{ margin: '10px 0 0', paddingLeft: 18, color: 'var(--muted)', fontSize: 13.5, display: 'grid', gap: 4 }}>{g.items.map((i) => <li key={i}>{i}</li>)}</ul></div>
                ))}
              </div>
            </section>

            <section className="section">
              <SectionHead eyebrow="Data dictionary" title="Every processed column" />
              <input type="text" placeholder="Search columns…" value={q} onChange={(e) => setQ(e.target.value)} style={{ maxWidth: 360, marginBottom: 12 }} />
              <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Column</th><th>Meaning</th><th>Type</th><th>Example</th><th>Missing %</th><th>Handling</th></tr></thead><tbody>
                {dict.map((r) => (
                  <tr key={r.feature}><td className="mono">{r.feature}</td><td style={{ whiteSpace: 'normal', minWidth: 230 }}>{r.meaning}</td><td>{r.type}</td><td className="mono" style={{ color: 'var(--muted)' }}>{r.example}</td><td className="num">{r.missing}</td><td style={{ whiteSpace: 'normal', minWidth: 220, color: 'var(--muted)' }}>{r.handling}</td></tr>
                ))}</tbody></table></div>
            </section>
          </>
        )}
      </Loading>

      <section className="section">
        <SectionHead eyebrow="Exploratory analysis" title="What the data looks like" />
        <Loading data={eda} h={300}>
          {eda && (
            <div className="grid g2">
              <Chart title="Accidents by hour" data={eda.hour} note="Slight accidents peak with commuter traffic; fatal ones peak late at night." kind="bar" />
              <Chart title="Serious + Fatal share by hour (%)" data={eda.hour_share} keys={['Serious', 'Fatal']} kind="line" />
              <Chart title="Severity share by speed limit (%)" data={eda.speed} stacked note="The share of severe accidents rises steeply above 40 mph." />
              <Chart title="Severity share by light conditions (%)" data={eda.light} />
              <Chart title="Severity share by road type (%)" data={eda.road} />
              <Chart title="Severity share by weekday (%)" data={eda.dow} />
              <Chart title="Serious + Fatal share per year (%)" data={eda.year} keys={['Serious', 'Fatal']} kind="line" note="Note the jump in 2016-17, linked to changes in how police forces classify injuries." />
              <Chart title="Urban vs rural (%)" data={eda.urban} />
            </div>
          )}
        </Loading>
      </section>

      <section className="section">
        <div className="grid g2">
          <div className="card">
            <span className="eyebrow">Limitations &amp; ethics</span>
            <ul style={{ margin: '12px 0 0', paddingLeft: 18, color: 'var(--muted)', display: 'grid', gap: 7, fontSize: 14 }}>
              <li>UK-only, injury accidents only: slight accidents are under-reported and uninjured collisions are missing.</li>
              <li>No traffic volumes: black spots show where accidents concentrate, not per-vehicle risk.</li>
              <li>No driver behaviour, vehicle speed or vehicle type in the scene features, which caps achievable accuracy.</li>
              <li>SHAP and LIME describe the model, not causes; associations are not interventions.</li>
              <li>Severity classification changed across police forces around 2016, creating a distribution shift that we measure and correct for.</li>
              <li>Locations are accident sites, not people. Outputs support planning and should never be used to blame individuals.</li>
            </ul>
          </div>
          <div className="card">
            <span className="eyebrow">Future work</span>
            <ul style={{ margin: '12px 0 0', paddingLeft: 18, color: 'var(--muted)', display: 'grid', gap: 7, fontSize: 14 }}>
              <li>Add Indian data (MoRTH, state police records) and test transfer of the framework.</li>
              <li>Join the vehicle and casualty tables (vehicle type, driver age, pedestrians) for scene-level predictors.</li>
              <li>Traffic-exposure normalisation and road-network (graph) clustering instead of raw coordinates.</li>
              <li>Spatio-temporal models for forecasting next-year hotspots, with live data deployment.</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="References" title="Sources and methods cited" />
        <div className="card"><ol style={{ margin: 0, paddingLeft: 20, color: 'var(--muted)', display: 'grid', gap: 7, fontSize: 14 }}>{REFS.map((r) => <li key={r}>{r}</li>)}</ol></div>
      </section>
    </>
  )
}
