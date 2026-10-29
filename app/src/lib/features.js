// Turns what a user picks in the UI into the exact feature row the model was trained on.
// Mirrors src/data_prep.py (engineer) and the history lookup in add_history_features.

export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const SEASON = ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'autumn', 'autumn', 'autumn', 'winter']
const BAD_WEATHER = ['Raining no high winds', 'Raining + high winds', 'Snowing no high winds', 'Snowing + high winds', 'Fog or mist']

export const DEFAULTS = {
  lat: 51.5074,
  lon: -0.1278,
  la_district: 'Westminster',
  speed_limit: 30,
  Road_Type: 'Single carriageway',
  Junction_Detail: 'Not at junction or within 20 metres',
  Junction_Control: 'Give way or uncontrolled',
  Light_Conditions: 'Daylight',
  Weather_Conditions: 'Fine no high winds',
  Road_Surface_Conditions: 'Dry',
  Urban_or_Rural_Area: 'Urban',
  '1st_Road_Class': 'A',
  '2nd_Road_Class': 'None',
  Carriageway_Hazards: 'None',
  Special_Conditions_at_Site: 'None',
  'Pedestrian_Crossing-Human_Control': 'code_0',
  'Pedestrian_Crossing-Physical_Facilities': 'code_0',
  n_vehicles: 2,
  hour: 17,
  dow: 2,
  month: 6,
}

export const cellKey = (lat, lon, deg) => `${Math.floor(lat / deg)}_${Math.floor(lon / deg)}`

function haversineM(a, b, c, d) {
  const R = 6371008.8
  const r = Math.PI / 180
  const dLat = (c - a) * r
  const dLon = (d - b) * r
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

export function nearestHotspotM(meta, lat, lon) {
  let best = Infinity
  for (const [la, lo] of meta.hot) {
    if (Math.abs(la - lat) > 0.6) continue
    const d = haversineM(lat, lon, la, lo)
    if (d < best) best = d
  }
  return best === Infinity ? 99999 : best
}

export function nearestLA(meta, lat, lon) {
  let best = null
  let bd = Infinity
  for (const l of meta.la) {
    const d = (l.lat - lat) ** 2 + ((l.lon - lon) * Math.cos((lat * Math.PI) / 180)) ** 2
    if (d < bd) { bd = d; best = l }
  }
  return best
}

export function buildRow(s, meta) {
  const light = s.Light_Conditions
  const night = light.startsWith('Darkness') ? 1 : 0
  const urban = s.Urban_or_Rural_Area === 'Urban' ? 1 : 0
  const atJunction = s.Junction_Detail !== 'Not at junction or within 20 metres'
  const la = meta.la.find((l) => l.name === s.la_district) || meta.la[0]
  const ck = cellKey(s.lat, s.lon, meta.cell_deg)
  const c = meta.cells[ck]
  const n = c ? c[0] : 0
  const prior = meta.prior_ksi
  const m = 20
  const dist = nearestHotspotM(meta, s.lat, s.lon)
  const sl = s.speed_limit
  return {
    ...s,
    Police_Force: la.force,
    season: SEASON[s.month - 1],
    speed_band: sl <= 30 ? 'low' : sl <= 50 ? 'medium' : 'high',
    is_weekend: s.dow >= 5 ? 1 : 0,
    is_rush_hour: [7, 8, 9, 16, 17, 18, 19].includes(s.hour) ? 1 : 0,
    is_night: night,
    hour_sin: Math.sin((2 * Math.PI * s.hour) / 24),
    hour_cos: Math.cos((2 * Math.PI * s.hour) / 24),
    is_urban: urban,
    wet_surface: ['Wet or damp', 'Flood over 3cm. deep'].includes(s.Road_Surface_Conditions) ? 1 : 0,
    bad_weather: BAD_WEATHER.includes(s.Weather_Conditions) ? 1 : 0,
    dark_and_unlit: ['Darkness - no lighting', 'Darkness - lights unlit'].includes(light) ? 1 : 0,
    junction_at_night: atJunction && night ? 1 : 0,
    high_speed_rural: sl >= 60 && !urban ? 1 : 0,
    h_cell_n: n,
    h_cell_ksi_rate: ((c ? c[1] : 0) + m * prior) / (n + m),
    h_cell_score: c ? c[2] : 0,
    h_la_n: la.n,
    h_la_ksi_rate: la.ksi_rate,
    h_la_score: la.score,
    h_dist_hotspot_m: dist,
    h_in_hotspot: dist < 300 ? 1 : 0,
  }
}

export const PRESETS = [
  { name: 'Quiet daytime street', emoji: '☀️', patch: { speed_limit: 30, hour: 11, Light_Conditions: 'Daylight', Weather_Conditions: 'Fine no high winds', Road_Surface_Conditions: 'Dry', Urban_or_Rural_Area: 'Urban', Road_Type: 'Single carriageway', Junction_Detail: 'Not at junction or within 20 metres', n_vehicles: 1, '1st_Road_Class': 'Unclassified', lat: 51.5074, lon: -0.1278, la_district: 'Westminster' } },
  { name: 'Rural road at night', emoji: '🌙', patch: { speed_limit: 60, hour: 2, dow: 5, Light_Conditions: 'Darkness - no lighting', Weather_Conditions: 'Fine no high winds', Road_Surface_Conditions: 'Dry', Urban_or_Rural_Area: 'Rural', Road_Type: 'Single carriageway', Junction_Detail: 'Not at junction or within 20 metres', n_vehicles: 1, '1st_Road_Class': 'A', lat: 53.0, lon: -1.8, la_district: 'Staffordshire Moorlands' } },
  { name: 'Wet motorway rush hour', emoji: '🌧️', patch: { speed_limit: 70, hour: 8, dow: 1, Light_Conditions: 'Daylight', Weather_Conditions: 'Raining no high winds', Road_Surface_Conditions: 'Wet or damp', Urban_or_Rural_Area: 'Rural', Road_Type: 'Dual carriageway', Junction_Detail: 'Not at junction or within 20 metres', n_vehicles: 3, '1st_Road_Class': 'Motorway', lat: 52.9, lon: -1.3, la_district: 'Erewash' } },
  { name: 'Junction after dark', emoji: '🚦', patch: { speed_limit: 40, hour: 22, dow: 4, Light_Conditions: 'Darkness - lights lit', Weather_Conditions: 'Raining no high winds', Road_Surface_Conditions: 'Wet or damp', Urban_or_Rural_Area: 'Urban', Road_Type: 'Single carriageway', Junction_Detail: 'T or staggered junction', n_vehicles: 2, '1st_Road_Class': 'A', lat: 52.48, lon: -1.9, la_district: 'Birmingham' } },
]
