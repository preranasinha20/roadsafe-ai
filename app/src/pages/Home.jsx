import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Brain, Gauge, Layers, Map as MapIcon, ShieldCheck, Sparkles, TrendingUp, Database, Scale } from 'lucide-react'
import { useJSON, pct, num } from '../lib/data.js'
import { SectionHead, Skeleton } from '../components/ui.jsx'
import HeroViz from '../components/HeroViz.jsx'

const fade = (i = 0) => ({ initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: '-40px' }, transition: { duration: 0.5, delay: i * 0.07 } })

const PIPE = [
  ['01', 'Clean & engineer', '2.05M DfT records, UK bounding-box and validity checks, 40+ scene features'],
  ['02', 'Leak-free history', 'Expanding-window spatial risk features built only from earlier years'],
  ['03', 'Time-based split', 'Train 2006-14 · validate 2015 · test 2016-17, never resampled'],
  ['04', 'Imbalance study', 'Class weights, under-sampling, SMOTE and tuned decision scales compared'],
  ['05', 'Boosted ensembles', 'LightGBM & XGBoost tuned with Optuna, isotonic-calibrated'],
  ['06', 'Explain & locate', 'SHAP + LIME, black spots detected 2005-14 and validated on 2015-17'],
]

export default function Home() {
  const { data: s } = useJSON('summary')
  const h = s?.headline
  return (
    <>
      <section className="hero">
        <div>
          <motion.span className="chip accent" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Sparkles size={13} /> Applications of Machine Learning · Case study
          </motion.span>
          <motion.h1 style={{ marginTop: 18 }} initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
            Predict how <em>severe</em> a road accident will be, and where the next one is likely to happen.
          </motion.h1>
          <motion.p className="lead" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
            An explainable machine-learning framework trained on {s ? num(s.dataset.rows) : '2 million'} UK road accidents. It links severity
            prediction, spatial black-spot detection and SHAP explanations, and every number is measured on later, unseen years.
          </motion.p>
          <motion.div className="hero-cta" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
            <Link to="/predict" className="btn primary">
              Try the predictor <ArrowRight size={17} />
            </Link>
            <Link to="/map" className="btn">
              <MapIcon size={17} /> Explore black spots
            </Link>
          </motion.div>
        </div>
        <HeroViz />
      </section>

      <section className="section">
        <div className="grid g4">
          {[
            [h ? h.macro_f1.toFixed(3) : '–', 'Macro-F1 on unseen 2016-17 data', 'baseline 0.301'],
            [h ? pct(h.recall_KSI, 0) : '–', 'Killed or seriously injured cases flagged', 'recall on the test years'],
            [h ? h.roc_auc.toFixed(3) : '–', 'ROC-AUC (one-vs-rest, macro)', 'ranking quality across classes'],
            [s ? s.blackspots.persistence_pct : '–', 'Top-100 black spots still dangerous in later years', 'temporal validation'],
          ].map(([v, l, sub], i) => (
            <motion.div key={l} className="card glow hover" {...fade(i)}>
              <div className="stat">
                <span className="v">{v}</span>
                <span className="l">{l}</span>
                <span className="chip" style={{ width: 'fit-content', marginTop: 6 }}>{sub}</span>
              </div>
            </motion.div>
          ))}
        </div>
        {!s && <Skeleton h={10} />}
      </section>

      <section className="section">
        <SectionHead eyebrow="Why it matters" title="Road safety is a decision problem, not a prediction contest">
          Around 85% of injury accidents are slight, so a model that always says “slight” looks accurate while missing every fatality. This
          project is built around that fact.
        </SectionHead>
        <div className="grid g3">
          {[
            [Gauge, 'Severity prediction', 'Slight, Serious or Fatal from road, weather, light, time and local history, with calibrated probabilities that can be trusted as probabilities.'],
            [MapIcon, 'Black-spot detection', 'A fixed-radius, severity-weighted detector finds compact sites; DBSCAN and KDE agree, and spots detected in 2005-14 persist in 2015-17.'],
            [Brain, 'Explainability', 'Exact TreeSHAP runs in your browser for every prediction, and LIME is compared against it to show where explanations agree and where they do not.'],
          ].map(([Icon, t, d], i) => (
            <motion.div key={t} className="card hover" {...fade(i)}>
              <div className="brand-mark" style={{ marginBottom: 14 }}>
                <Icon size={18} color="#fff" />
              </div>
              <h3>{t}</h3>
              <p className="sub" style={{ marginTop: 8 }}>{d}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="section">
        <SectionHead eyebrow="Methodology" title="One leak-free pipeline, end to end" />
        <div className="pipe">
          {PIPE.map(([n, t, d], i) => (
            <motion.div className="step" key={n} {...fade(i)}>
              <span className="n">{n}</span>
              <h4>{t}</h4>
              <p>{d}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="card glow" style={{ display: 'grid', gap: 18, gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
          <div>
            <span className="eyebrow">A note on accuracy</span>
            <h3 style={{ fontSize: 24, margin: '8px 0 10px' }}>Why we report macro-F1 and not just accuracy</h3>
            <p className="sub">
              On the 2016-17 test years, 82.1% of accidents are “Slight”, so a classifier that never predicts a fatality already scores{' '}
              <b>82.1% accuracy</b>. We therefore headline macro-F1, per-class recall, ROC-AUC and PR-AUC, and still report plain accuracy under
              both protocols so nothing is hidden.
            </p>
          </div>
          <div style={{ display: 'grid', gap: 10 }}>
            {[
              [ShieldCheck, 'No leakage', 'Casualty counts and police attendance are excluded; history uses only earlier years.'],
              [Scale, 'Honest evaluation', 'Time-based test, tuning on 2015 only, resampling never touches validation or test data.'],
              [TrendingUp, 'Distribution shift handled', 'Serious+Fatal share jumped from 15.5% to 18.6% in 2016-17; the model is calibrated and re-scaled on 2015.'],
              [Layers, 'Ablations', 'Every feature group is removed in turn so each design choice is justified by data.'],
            ].map(([Icon, t, d]) => (
              <div key={t} style={{ display: 'flex', gap: 12 }}>
                <Icon size={18} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 3 }} />
                <div>
                  <b>{t}</b>
                  <div className="sub">{d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section" style={{ textAlign: 'center' }}>
        <h2 style={{ fontSize: 'clamp(26px,4vw,40px)' }}>See it work on your own scenario</h2>
        <p className="sub" style={{ margin: '10px auto 22px', maxWidth: 560 }}>
          Pick a location on the map and set the conditions. The model, calibration and SHAP explanation all run locally in the browser.
        </p>
        <Link to="/predict" className="btn primary">
          Open the predictor <ArrowRight size={17} />
        </Link>
      </section>
    </>
  )
}
