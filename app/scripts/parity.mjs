// Verifies that the in-browser engine reproduces scikit-learn + SHAP.
// Run after `python -m src.export_web`:  node scripts/parity.mjs
import { readFileSync } from 'node:fs'
import { buildEncoder, predict, shapForClass, groupShap } from '../src/lib/model.js'

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)))
const model = read('../public/data/model.json')
const meta = read('../public/data/meta.json')
const ref = read('./parity.json')
const enc = buildEncoder(meta.encoder)

let maxP = 0
let maxS = 0
let labelMismatch = 0
for (const r of ref) {
  const out = predict(model, enc, r.row)
  out.probs.forEach((p, k) => { maxP = Math.max(maxP, Math.abs(p - r.probs[k])) })
  for (const k of [1, 2]) {
    const { phi } = shapForClass(model, out.x, k)
    const g = groupShap(enc, phi)
    for (const [col, v] of Object.entries(g)) maxS = Math.max(maxS, Math.abs(v - r.shap[col][k]))
  }
}
console.log(`rows checked: ${ref.length}`)
console.log(`max |probability difference| : ${maxP.toExponential(2)}`)
console.log(`max |SHAP difference|        : ${maxS.toExponential(2)}`)
if (maxP > 1e-4 || maxS > 1e-3) {
  console.error('PARITY FAILED')
  process.exit(1)
}
console.log('PARITY OK')
