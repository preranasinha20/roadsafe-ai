import { useEffect, useState } from 'react'

// Every dataset is a small static JSON produced by the Python pipeline (see src/export_web.py).
const cache = new Map()

export function loadJSON(name) {
  if (!cache.has(name)) {
    const url = `${import.meta.env.BASE_URL}data/${name}.json`
    cache.set(
      name,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`Could not load ${name}`)
        return r.json()
      }),
    )
  }
  return cache.get(name)
}

export function useJSON(name) {
  const [state, set] = useState({ data: null, error: null })
  useEffect(() => {
    let live = true
    loadJSON(name)
      .then((data) => live && set({ data, error: null }))
      .catch((error) => live && set({ data: null, error }))
    return () => {
      live = false
    }
  }, [name])
  return state
}

export const SEV = ['Slight', 'Serious', 'Fatal']
export const SEV_COLOR = { Slight: 'var(--slight)', Serious: 'var(--serious)', Fatal: 'var(--fatal)' }
export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`
export const num = (v) => Number(v).toLocaleString('en-GB')
