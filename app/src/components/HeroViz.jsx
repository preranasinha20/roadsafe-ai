import React, { useEffect, useRef } from 'react'

// A canvas "road network": drifting cars on glowing lanes, with pulsing red black-spot rings.
export default function HeroViz() {
  const ref = useRef(null)

  useEffect(() => {
    const cv = ref.current
    const ctx = cv.getContext('2d')
    let raf
    let w = 0
    let h = 0
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const roads = []
    const cars = []
    const spots = []
    const rnd = (a, b) => a + Math.random() * (b - a)

    function resize() {
      const r = cv.getBoundingClientRect()
      w = r.width
      h = r.height
      cv.width = w * dpr
      cv.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      roads.length = cars.length = spots.length = 0
      for (let i = 0; i < 7; i++) {
        const y = (h / 8) * (i + 1) + rnd(-12, 12)
        roads.push({ a: { x: -20, y }, b: { x: w + 20, y: y + rnd(-60, 60) }, c: { x: w * rnd(0.3, 0.7), y: y + rnd(-90, 90) } })
      }
      for (let i = 0; i < 5; i++) {
        const x = (w / 6) * (i + 1) + rnd(-14, 14)
        roads.push({ a: { x, y: -20 }, b: { x: x + rnd(-50, 50), y: h + 20 }, c: { x: x + rnd(-80, 80), y: h * rnd(0.3, 0.7) } })
      }
      roads.forEach((r, i) => {
        for (let k = 0; k < 3; k++) cars.push({ road: i, t: Math.random(), v: rnd(0.0016, 0.004) * (Math.random() < 0.5 ? 1 : -1), hue: Math.random() < 0.5 ? 190 : 265 })
      })
      for (let i = 0; i < 6; i++) spots.push({ x: rnd(0.12, 0.88) * w, y: rnd(0.12, 0.88) * h, p: Math.random() * 6, s: rnd(0.7, 1.4), lvl: Math.random() })
    }

    const q = (r, t) => {
      const u = 1 - t
      return { x: u * u * r.a.x + 2 * u * t * r.c.x + t * t * r.b.x, y: u * u * r.a.y + 2 * u * t * r.c.y + t * t * r.b.y }
    }

    function frame(time) {
      ctx.clearRect(0, 0, w, h)
      ctx.lineCap = 'round'
      roads.forEach((r) => {
        ctx.beginPath()
        ctx.moveTo(r.a.x, r.a.y)
        ctx.quadraticCurveTo(r.c.x, r.c.y, r.b.x, r.b.y)
        ctx.strokeStyle = 'rgba(148,163,200,0.14)'
        ctx.lineWidth = 7
        ctx.stroke()
        ctx.strokeStyle = 'rgba(148,163,200,0.28)'
        ctx.lineWidth = 1
        ctx.setLineDash([8, 10])
        ctx.stroke()
        ctx.setLineDash([])
      })
      cars.forEach((c) => {
        if (!reduce) {
          c.t += c.v
          if (c.t > 1) c.t = 0
          if (c.t < 0) c.t = 1
        }
        const p = q(roads[c.road], c.t)
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 11)
        g.addColorStop(0, `hsla(${c.hue},95%,70%,0.95)`)
        g.addColorStop(1, `hsla(${c.hue},95%,60%,0)`)
        ctx.fillStyle = g
        ctx.beginPath()
        ctx.arc(p.x, p.y, 11, 0, 7)
        ctx.fill()
      })
      spots.forEach((s) => {
        const pulse = (Math.sin(time / 700 + s.p) + 1) / 2
        const col = s.lvl > 0.65 ? '251,75,110' : s.lvl > 0.3 ? '251,191,36' : '52,211,153'
        for (let k = 0; k < 3; k++) {
          const rr = (14 + k * 14 + pulse * 10) * s.s
          ctx.beginPath()
          ctx.arc(s.x, s.y, rr, 0, 7)
          ctx.strokeStyle = `rgba(${col},${0.5 - k * 0.14})`
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
        ctx.beginPath()
        ctx.arc(s.x, s.y, 4.5, 0, 7)
        ctx.fillStyle = `rgb(${col})`
        ctx.fill()
      })
      raf = requestAnimationFrame(frame)
    }

    resize()
    raf = requestAnimationFrame(frame)
    window.addEventListener('resize', resize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <div className="card glow" style={{ padding: 0, aspectRatio: '1 / 0.92', overflow: 'hidden', position: 'relative' }}>
      <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />
      <div style={{ position: 'absolute', left: 16, bottom: 14, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span className="chip"><i className="dot" style={{ background: 'var(--slight)' }} /> Slight</span>
        <span className="chip"><i className="dot" style={{ background: 'var(--serious)' }} /> Serious</span>
        <span className="chip"><i className="dot" style={{ background: 'var(--fatal)' }} /> Fatal</span>
      </div>
    </div>
  )
}
