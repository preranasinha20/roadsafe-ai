// In-browser inference engine for the exported gradient-boosted model.
//
// The Python pipeline (src/export_web.py) writes the trained trees, the one-hot encoder layout,
// the isotonic calibration knots and the decision scales to JSON. Everything below reproduces the
// scikit-learn pipeline exactly (verified by app/scripts/parity.mjs), so the deployed site needs
// no server. It also implements exact path-dependent TreeSHAP (Lundberg et al., 2018).

export const CLASSES = ['Slight', 'Serious', 'Fatal']

export function buildEncoder(meta) {
  // meta.columns: [{type:'cat',col,cat}|{type:'num',col}], meta.infrequent: {col:[cats...]}
  const catIndex = {}
  const infIndex = {}
  const numIndex = {}
  meta.columns.forEach((c, i) => {
    if (c.type === 'num') numIndex[c.col] = i
    else if (c.cat === '__infrequent__') infIndex[c.col] = i
    else (catIndex[c.col] ||= {})[c.cat] = i
  })
  return { n: meta.columns.length, catIndex, infIndex, numIndex, columns: meta.columns }
}

export function encode(enc, row) {
  const x = new Float64Array(enc.n)
  for (const col in enc.numIndex) {
    const v = row[col]
    x[enc.numIndex[col]] = v === undefined || v === null || Number.isNaN(v) ? 0 : v
  }
  for (const col in enc.catIndex) {
    const v = row[col] ?? 'Unknown'
    const j = enc.catIndex[col][v]
    if (j !== undefined) x[j] = 1
    else if (enc.infIndex[col] !== undefined) x[enc.infIndex[col]] = 1
  }
  return x
}

// ── tree traversal ─────────────────────────────────────────────────────────────
// XGBoost was trained on a sparse one-hot matrix, where an exact 0 means "missing" and follows the
// node's default branch (t.m). LightGBM trees carry no t.m and use a plain threshold test.
function child(t, n, x) {
  const v = x[t.f[n]]
  if (t.m && v === 0) return t.m[n]
  return v <= t.t[n] ? t.l[n] : t.r[n]
}
function leafValue(t, x) {
  let n = 0
  while (t.f[n] >= 0) n = child(t, n, x)
  return t.v[n]
}

export function rawScores(trees, x, nClasses = 3, bias = null) {
  const s = Float64Array.from(bias || new Array(nClasses).fill(0))
  for (const t of trees) s[t.k] += leafValue(t, x)
  return s
}

export function softmax(s) {
  const m = Math.max(...s)
  const e = Array.from(s, (v) => Math.exp(v - m))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

function interp(xs, ys, v) {
  if (v <= xs[0]) return ys[0]
  const n = xs.length
  if (v >= xs[n - 1]) return ys[n - 1]
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] <= v) lo = mid
    else hi = mid
  }
  const w = (v - xs[lo]) / (xs[hi] - xs[lo] || 1)
  return ys[lo] + w * (ys[hi] - ys[lo])
}

export function calibrate(cal, p) {
  const c = p.map((v, k) => interp(cal[k].x, cal[k].y, v))
  const z = c.reduce((a, b) => a + b, 0)
  return c.map((v) => v / z)
}

export function predict(model, enc, row) {
  const x = encode(enc, row)
  const raw = rawScores(model.trees, x, model.classes.length, model.bias)
  const probs = calibrate(model.calibration, softmax(raw))
  const scaled = probs.map((p, k) => p * model.scales[k])
  const label = scaled.indexOf(Math.max(...scaled))
  return { x, raw: Array.from(raw), probs, label }
}

// ── exact TreeSHAP (path dependent) ────────────────────────────────────────────
function extend(m, depth, pz, po, pi) {
  m[depth] = { d: pi, z: pz, o: po, w: depth === 0 ? 1 : 0 }
  for (let i = depth - 1; i >= 0; i--) {
    m[i + 1].w += (po * m[i].w * (i + 1)) / (depth + 1)
    m[i].w = (pz * m[i].w * (depth - i)) / (depth + 1)
  }
}
function unwind(m, depth, idx) {
  const po = m[idx].o
  const pz = m[idx].z
  let next = m[depth].w
  for (let i = depth - 1; i >= 0; i--) {
    if (po !== 0) {
      const tmp = m[i].w
      m[i].w = (next * (depth + 1)) / ((i + 1) * po)
      next = tmp - (m[i].w * pz * (depth - i)) / (depth + 1)
    } else {
      m[i].w = (m[i].w * (depth + 1)) / (pz * (depth - i))
    }
  }
  for (let i = idx; i < depth; i++) {
    m[i].d = m[i + 1].d
    m[i].z = m[i + 1].z
    m[i].o = m[i + 1].o
  }
}
function unwoundSum(m, depth, idx) {
  const po = m[idx].o
  const pz = m[idx].z
  let next = m[depth].w
  let total = 0
  for (let i = depth - 1; i >= 0; i--) {
    if (po !== 0) {
      const tmp = (next * (depth + 1)) / ((i + 1) * po)
      total += tmp
      next = m[i].w - (tmp * pz * (depth - i)) / (depth + 1)
    } else {
      total += m[i].w / pz / ((depth - i) / (depth + 1))
    }
  }
  return total
}

function recurse(t, x, phi, node, depth, parent, pz, po, pi) {
  const m = parent.slice(0, depth).map((e) => ({ ...e }))
  extend(m, depth, pz, po, pi)
  if (t.f[node] < 0) {
    for (let i = 1; i <= depth; i++) {
      const w = unwoundSum(m, depth, i)
      phi[m[i].d] += w * (m[i].o - m[i].z) * t.v[node]
    }
    return
  }
  const f = t.f[node]
  const hot = child(t, node, x)
  const cold = hot === t.l[node] ? t.r[node] : t.l[node]
  let iz = 1
  let io = 1
  let d = depth
  let k = -1
  for (let i = 1; i <= d; i++) if (m[i].d === f) { k = i; break }
  if (k >= 0) {
    iz = m[k].z
    io = m[k].o
    unwind(m, d, k)
    d -= 1
  }
  const cn = t.c[node]
  recurse(t, x, phi, hot, d + 1, m, (iz * t.c[hot]) / cn, io, f)
  recurse(t, x, phi, cold, d + 1, m, (iz * t.c[cold]) / cn, 0, f)
}

// Returns {phi: Float64Array(nFeatures) for class k, base: expected raw score of class k}
export function shapForClass(model, x, k) {
  const phi = new Float64Array(x.length)
  let base = 0
  for (const t of model.trees) {
    if (t.k !== k) continue
    recurse(t, x, phi, 0, 0, [], 1, 1, -1)
    base += t.base
  }
  return { phi, base }
}

// Collapse one-hot columns back to the original feature name (SHAP values are additive).
export function groupShap(enc, phi) {
  const out = {}
  enc.columns.forEach((c, i) => {
    out[c.col] = (out[c.col] || 0) + phi[i]
  })
  return out
}
