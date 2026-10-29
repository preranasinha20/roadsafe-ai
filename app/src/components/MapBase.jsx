
import React, { useEffect, useRef } from 'react'
import L from 'leaflet'

// Shared map component for Predictor and Black Spots.
// Use OpenStreetMap tiles instead of the affected CARTO basemap URLs.
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

export function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light'
    ? 'light'
    : 'dark'
}

export default function MapBase({
  onReady,
  center = [54.3, -2.6],
  zoom = 6,
  className = 'map-box',
  minZoom = 5,
}) {
  const el = useRef(null)

  useEffect(() => {
    if (!el.current) return undefined

    const map = L.map(el.current, {
      zoomControl: true,
      minZoom,
      maxZoom: 19,
      preferCanvas: true,
    }).setView(center, zoom)

    const tile = L.tileLayer(TILE_URL, {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors',
    }).addTo(map)

    const cleanup = onReady?.(map, L)

    const resizeFrame = window.requestAnimationFrame(() => {
      map.invalidateSize()
    })

    return () => {
      window.cancelAnimationFrame(resizeFrame)

      if (typeof cleanup === 'function') cleanup()

      map.removeLayer(tile)
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div ref={el} className={className} />
}
