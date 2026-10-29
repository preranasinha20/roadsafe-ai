import React from 'react'

export function SectionHead({ eyebrow, title, children }) {
  return (
    <div className="section-head">
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  )
}

export function Stat({ value, label }) {
  return (
    <div className="stat">
      <span className="v">{value}</span>
      <span className="l">{label}</span>
    </div>
  )
}

export function Skeleton({ h = 260 }) {
  return <div className="skeleton" style={{ height: h }} />
}

export function Tip({ active, payload, label, fmt }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tt">
      <b>{label ?? payload[0].payload?.name}</b>
      {payload.map((p) => (
        <div key={p.dataKey ?? p.name} style={{ color: p.color || p.fill }}>
          {p.name}: <span className="mono">{fmt ? fmt(p.value, p) : typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

export function Loading({ data, error, h, children }) {
  if (error) return <div className="note warn">Could not load data: {String(error.message)}</div>
  if (!data) return <Skeleton h={h} />
  return children
}
