"use strict";

const TZ = "America/Los_Angeles";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? esc(u) : "#");
const fmtTime = (iso, withDay = true) => new Date(iso).toLocaleString("en-US", {
  timeZone: TZ, hour: "2-digit", minute: "2-digit", hour12: false,
  ...(withDay ? { month: "short", day: "numeric" } : {}),
});
const ago = (iso, at) => {
  const m = Math.round((at - new Date(iso)) / 60000);
  return m < 60 ? `${m} min before` : `${(m / 60).toFixed(1)} h before`;
};

// ---- map ------------------------------------------------------------------

const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([34.06, -118.45], 10);
const base = {
  "Streets": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "© OpenStreetMap contributors" }),
  "Satellite": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics" }),
};
base["Streets"].addTo(map);
L.control.layers(base, null, { position: "topleft" }).addTo(map);
L.control.scale({ imperial: true, metric: true }).addTo(map);

// Weather alerts sit under everything else.
map.createPane("alerts").style.zIndex = 350;

const LAYERS = [
  { key: "alerts", label: "Weather alerts (NWS)", color: "--alert-fire",
    note: "Warnings and advisories in effect, drawn over their forecast zones in the National Weather Service colours." },
  { key: "fires", label: "Wildfires", color: "--fire",
    note: "CAL FIRE and NIFC incidents, mapped perimeters, and NASA satellite heat detections (last 24 h; some are industrial sites)." },
  { key: "quakes", label: "Earthquakes (USGS)", color: "--quake" },
  { key: "alertcalifornia", label: "AlertCalifornia cameras", color: "--ac",
    note: "Wedge = view direction and field of view. Red = re-aimed in this window." },
  { key: "cctv", label: "Caltrans CCTV", color: "--cctv" },
  { key: "incidents", label: "CHP calls", color: "--chp" },
  { key: "news", label: "News (GDELT)", color: "--news" },
  { key: "weather", label: "Weather", color: "--weather" },
  { key: "beliefs", label: "Agent beliefs", color: "--belief",
    note: "Circles are where a belief applies, shaded by the agent's confidence. Pick one in the Beliefs tab to see lines to its evidence." },
];
const groups = {};
for (const l of LAYERS) {
  groups[l.key] = l.key === "news"
    ? L.markerClusterGroup({ maxClusterRadius: 40, spiderfyOnMaxZoom: true, showCoverageOnHover: false })
    : L.layerGroup();
  groups[l.key].addTo(map);
}

const layersEl = $("layers");
for (const l of LAYERS) {
  // Notes show on hover, so the list below keeps its room.
  const label = document.createElement("label");
  label.id = `layer-${l.key}`;
  label.title = l.note || "";
  label.innerHTML = `<input type="checkbox" checked data-layer="${l.key}">
    <span class="swatch" style="background:var(${l.color})"></span>
    <span>${l.label}</span><span class="count" id="count-${l.key}">–</span>`;
  layersEl.append(label);
}
layersEl.addEventListener("change", (e) => {
  const key = e.target.dataset.layer;
  if (!key) return;
  e.target.checked ? groups[key].addTo(map) : map.removeLayer(groups[key]);
});

// A wedge from the camera along its bearing, as wide as its field of view.
function wedge(lat, lon, bearing, fov, km) {
  const pts = [[lat, lon]];
  const half = Math.min(fov || 60, 120) / 2;
  for (let a = bearing - half; a <= bearing + half + 0.01; a += half / 6) {
    const r = (a * Math.PI) / 180;
    pts.push([lat + (km / 111) * Math.cos(r), lon + (km / (111 * Math.cos((lat * Math.PI) / 180))) * Math.sin(r)]);
  }
  pts.push([lat, lon]);
  return pts;
}
const ray = (lat, lon, bearing, km) => wedge(lat, lon, bearing, 1, km).slice(0, 2);

// ---- state ----------------------------------------------------------------

let range = null;       // {start, end} across sources, ms
let snapshot = null;
let fireData = { fires: [], perimeters: [], hotspots: [] };
let live = { chp: null, alerts: [], zones: {}, quakes: [], fetched: {} };
let beliefData = { beliefs: [] };
let runs = [];
let selectedBelief = null;
let tab = "incidents";
const markers = {};     // id -> marker, for the list

function selectedTime() {
  return $("live").checked ? new Date() : new Date(Number($("slider").value) * 1000);
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function loadRange() {
  const info = await getJSON("/api/range");
  const times = Object.values(info.sources).flatMap((s) => [new Date(s.start), new Date(s.end)]);
  if (!times.length) return;
  const now = new Date();
  range = { start: Math.min(...times), end: Math.max(now, ...times) };
  const slider = $("slider");
  slider.min = Math.floor(range.start / 1000);
  slider.max = Math.ceil(range.end / 1000);
  if ($("live").checked) slider.value = slider.max;
  try { runs = (await getJSON("/api/beliefs")).runs; } catch { runs = []; }
  const runSelect = $("run"), current = runSelect.value;
  runSelect.innerHTML = runs.length
    ? runs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}${r.agents.join(", ") !== r.run ? ` · ${esc(r.agents.join(", "))}` : ""} · ${r.beliefs} beliefs</option>`).join("")
    : `<option value="">No belief runs yet</option>`;
  if (runs.some((r) => r.run === current)) runSelect.value = current;
  $("status").textContent = `Data refreshed ${fmtTime(info.refreshed)} · times are Los Angeles`;
}

async function loadSnapshot() {
  const at = selectedTime();
  $("clock").textContent = fmtTime(at.toISOString());
  const hours = $("window").value;
  const atParam = encodeURIComponent(at.toISOString());
  const run = $("run").value;
  try {
    [snapshot, fireData, live, beliefData] = await Promise.all([
      getJSON(`/api/snapshot?at=${atParam}&hours=${hours}`),
      getJSON(`/api/fires?at=${atParam}&hours=${Math.max(hours, 24)}`).catch(() => fireData),
      getJSON(`/api/live?at=${atParam}&hours=${hours}`).catch(() => live),
      run ? getJSON(`/api/beliefs?run=${encodeURIComponent(run)}&at=${atParam}`).catch(() => ({ beliefs: [] }))
        : Promise.resolve({ beliefs: [] }),
    ]);
  } catch (err) {
    $("status").textContent = `Could not load data: ${err.message}`;
    return;
  }
  draw();
}

// ---- drawing --------------------------------------------------------------

function draw() {
  const s = snapshot;
  const at = new Date(s.at);
  for (const g of Object.values(groups)) g.clearLayers();
  for (const k of Object.keys(markers)) delete markers[k];

  for (const c of s.alertcalifornia) {
    const color = c.reaimed ? css("--reaim") : css("--ac");
    if (c.bearing != null) {
      for (const b of c.bearings) {
        if (Math.abs(b - c.bearing) > 2) {
          L.polyline(ray(c.lat, c.lon, b, 4), { color, weight: 1, opacity: 0.5, dashArray: "3 4", interactive: false })
            .addTo(groups.alertcalifornia);
        }
      }
      L.polygon(wedge(c.lat, c.lon, c.bearing, c.fov, 4), {
        color, weight: c.reaimed ? 2 : 1, fillColor: color, fillOpacity: 0.16, interactive: false,
      }).addTo(groups.alertcalifornia);
    }
    const m = L.circleMarker([c.lat, c.lon], { radius: 5, color: "#fff", weight: 1.5, fillColor: color, fillOpacity: 1 })
      .bindPopup(() => cameraPopup("alertcalifornia", c, at), { maxWidth: 420, minWidth: 320 })
      .addTo(groups.alertcalifornia);
    markers[`ac:${c.name}`] = m;
  }

  for (const c of s.cctv) {
    L.circleMarker([c.lat, c.lon], { radius: 3.5, color: css("--cctv"), weight: 1, fillColor: css("--cctv"), fillOpacity: 0.7 })
      .bindPopup(() => cameraPopup("cctv", c, at), { maxWidth: 380, minWidth: 300 })
      .addTo(groups.cctv);
  }

  // Live CHP dispatch calls when they were being collected; otherwise the daily file.
  const calls = chpCalls();
  if (live.chp) {
    $("layer-incidents").title = `Live CHP dispatch calls open at this time (fetched ${fmtTime(live.fetched["chp-live"])}).`;
    for (const c of calls) {
      const roadwork = c.kind === "roadwork" || c.kind === "other";
      const m = L.circleMarker([c.lat, c.lon], { radius: roadwork ? 4 : 6, color: "#fff", weight: 1.5,
        fillColor: css(`--chp-${c.kind}`), fillOpacity: 0.95 })
        .bindPopup(() => chpPopup(c), { maxWidth: 360 })
        .addTo(groups.incidents);
      markers[`chp:${c.id}`] = m;
    }
  } else {
    $("layer-incidents").title = "Live CHP calls were not being collected at this time; showing CHP's daily incident file, published the next morning.";
    for (const i of s.incidents) {
      const m = L.circleMarker([i.lat, i.lon], { radius: 6, color: "#fff", weight: 1.5, fillColor: css("--chp-injury"), fillOpacity: 0.95 })
        .bindPopup(`<div class="pop"><h3>${esc(i.description)}</h3>
          <div class="m">${fmtTime(i.time)} · CHP incident ${esc(i.incident)}${i.duration_minutes ? ` · lasted ${esc(i.duration_minutes)} min` : ""}</div></div>`)
        .addTo(groups.incidents);
      markers[i.id] = m;
    }
  }

  const alertsOnly = $("alertsOnly").checked;
  const news = s.news.filter((n) => !alertsOnly || n.alert);
  for (const n of news) {
    const color = n.alert ? css("--news-alert") : css("--news");
    const icon = L.divIcon({ className: "", iconSize: [12, 12], iconAnchor: [6, 6],
      html: `<span class="dot" style="background:${color}"></span>` });
    const m = L.marker([n.lat, n.lon], { icon })
      .bindPopup(`<div class="pop"><h3><a href="${safeUrl(n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a></h3>
        <div class="m">${fmtTime(n.time)} · ${esc(n.domain)}${n.outlets > 1 ? ` and ${n.outlets - 1} other outlets` : ""} · ${esc(n.place)}</div>
        ${n.categories.length ? `<div class="m" style="margin-top:4px">${esc(n.categories.join(", "))}</div>` : ""}</div>`, { maxWidth: 340 });
    groups.news.addLayer(m);
    markers[n.id] = m;
  }

  for (const w of s.weather) {
    const wind = w.wind_direction_degrees != null
      ? `<i style="transform:rotate(${(w.wind_direction_degrees + 180) % 360}deg)">↑</i>${w.wind_speed_mps.toFixed(1)} m/s` : "";
    L.marker([w.lat, w.lon], { icon: L.divIcon({ className: "", html: `<span class="wx">${Math.round(w.temperature_f)}°F ${wind}</span>`, iconAnchor: [30, 10] }) })
      .bindPopup(`<div class="pop"><h3>${esc(w.station)}</h3><div class="m">${fmtTime(w.time)}</div>
        ${esc(w.description)} · ${Math.round(w.temperature_f)}°F · humidity ${w.humidity}%<br>
        wind ${w.wind_speed_mps} m/s${w.wind_direction_degrees != null ? ` from ${w.wind_direction_degrees}°` : ""}</div>`)
      .addTo(groups.weather);
  }

  drawFires(at);
  drawAlerts();
  drawQuakes();
  drawBeliefs();

  const counts = { alerts: live.alerts.length, fires: fireData.fires.filter((f) => !f.minor).length, quakes: live.quakes.length,
    alertcalifornia: s.alertcalifornia.length, cctv: s.cctv.length, incidents: live.chp ? calls.length : s.incidents.length,
    news: news.length, weather: s.weather.length, beliefs: beliefData.beliefs.length };
  for (const [k, v] of Object.entries(counts)) $(`count-${k}`).textContent = v;
  drawList();
}

// ---- wildfires ----------------------------------------------------------------

const toRad = (d) => (d * Math.PI) / 180;
function distanceKm(a, b) {
  const dLat = toRad(b.lat - a.lat), dLon = toRad(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}
function bearingTo(a, b) {
  const y = Math.sin(toRad(b.lon - a.lon)) * Math.cos(toRad(b.lat));
  const x = Math.cos(toRad(a.lat)) * Math.sin(toRad(b.lat)) - Math.sin(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.cos(toRad(b.lon - a.lon));
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}
// AlertCalifornia cameras whose current view wedge covers the fire.
function camerasOn(fire) {
  return snapshot.alertcalifornia.filter((c) => {
    if (c.bearing == null) return false;
    const km = distanceKm(c, fire);
    if (km > 40) return false;
    const off = Math.abs(((bearingTo(c, fire) - c.bearing + 540) % 360) - 180);
    return km < 0.5 || off <= (c.fov || 60) / 2 + 3;
  }).map((c) => ({ name: c.name, km: distanceKm(c, fire) })).sort((a, b) => a.km - b.km);
}
const fmtAcres = (a) => (a == null ? "size unknown" : `${Number(a).toLocaleString("en-US", { maximumFractionDigits: 1 })} acres`);
const fmtContained = (c) => (c == null ? "" : ` · ${c}% contained`);

function newsFacts(fire) {
  const a = fire.news.find((n) => n.acres != null || n.contained != null);
  return a ? { ...a } : null;
}

function drawFires(at) {
  const g = groups.fires, fireColor = css("--fire");
  L.geoJSON({ type: "FeatureCollection", features: fireData.perimeters }, {
    style: { color: fireColor, weight: 2, fillColor: fireColor, fillOpacity: 0.3 },
  }).bindPopup((layer) => {
    const p = layer.feature.properties;
    return `<div class="pop"><h3>${esc(p.name)} Fire perimeter</h3><div class="m">${fmtAcres(p.acres)} mapped${p.mapped ? ` · ${fmtTime(p.mapped)}` : ""}</div>
      ${p.behavior ? `<div class="m">Behavior: ${esc(p.behavior)}</div>` : ""}${p.personnel ? `<div class="m">${p.personnel} personnel</div>` : ""}</div>`;
  }).addTo(g);
  for (const h of fireData.hotspots) {
    L.circleMarker([h.lat, h.lon], { radius: 3, color: css("--hot"), weight: 1, fillColor: css("--hot"), fillOpacity: 0.85 })
      .bindPopup(`<div class="pop"><h3>Satellite heat detection</h3><div class="m">${fmtTime(h.time)} · ${esc(h.satellite)} VIIRS · confidence ${esc(h.confidence)} · ${esc(h.frp)} MW</div></div>`)
      .addTo(g);
  }
  for (const f of fireData.fires) {
    const out = f.contained >= 100;
    const size = f.minor ? 10 : Math.round(18 + Math.min(Math.log10((f.acres || 1) + 1), 4) * 5);
    const icon = L.divIcon({ className: "", iconSize: [size, size], iconAnchor: [size / 2, size - 2],
      html: `<span class="flame${out ? " out" : ""}${f.minor ? " minor" : ""}" style="width:${size}px;height:${size}px"></span>` });
    const m = L.marker([f.lat, f.lon], { icon, zIndexOffset: f.minor ? 0 : 1000, title: f.name })
      .bindPopup(() => firePopup(f), { maxWidth: 380, minWidth: 300 })
      .addTo(g);
    markers[`fire:${f.key}`] = m;
    if (!f.minor && !out) {
      L.marker([f.lat, f.lon], { interactive: false, icon: L.divIcon({ className: "", iconAnchor: [-size / 2, size / 2 + 4],
        html: `<span class="fire-label">${esc(f.name.replace(/ Fire$/, ""))}</span>` }) }).addTo(g);
    }
  }
}

function firePopup(f) {
  const el = document.createElement("div");
  el.className = "pop";
  const facts = newsFacts(f);
  const cams = camerasOn(f);
  el.innerHTML = `<h3>${f.url ? `<a href="${safeUrl(f.url)}" target="_blank" rel="noopener">${esc(f.name)}</a>` : esc(f.name)}</h3>
    <div class="m">${esc([f.county && `${f.county} County`, f.location].filter(Boolean).join(" · "))}</div>
    <div class="facts"><b>${fmtAcres(f.acres)}${fmtContained(f.contained)}</b>
      <span class="m">${esc(f.source)}${f.updated ? `, updated ${fmtTime(f.updated)}` : ""}${f.started ? ` · started ${fmtTime(f.started)}` : ""}</span></div>
    ${facts ? `<div class="facts"><b>Per news: ${facts.acres != null ? fmtAcres(facts.acres) : ""}${facts.contained != null ? `${facts.acres != null ? " · " : ""}${facts.contained}% contained` : ""}</b>
      <span class="m">${esc(facts.domain)}, ${fmtTime(facts.time)}</span></div>` : ""}
    ${cams.length ? `<div class="sub">Cameras facing it</div><div class="cams">${cams.slice(0, 8).map((c) =>
      `<button type="button" data-cam="${esc(c.name)}">${esc(c.name)} <span class="m">${c.km.toFixed(1)} km</span></button>`).join("")}</div>` : ""}
    <div class="sub">In the news (${f.news.length})</div>
    ${f.news.length ? `<ul class="articles">${f.news.slice(0, 8).map((n) =>
      `<li><a href="${safeUrl(n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a> <span class="m">${esc(n.domain)} · ${fmtTime(n.time)}</span></li>`).join("")}</ul>`
      : `<div class="m">No collected articles name this fire.</div>`}`;
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-cam]");
    const m = b && markers[`ac:${b.dataset.cam}`];
    if (m) { map.setView(m.getLatLng(), Math.max(map.getZoom(), 12)); m.openPopup(); }
  });
  return el;
}

function chpCalls() {
  if (!live.chp) return [];
  return live.chp.filter((c) => c.lat != null && (!$("hideRoadwork").checked || (c.kind !== "roadwork" && c.kind !== "other")));
}

const CHP_LABEL = { fire: "Fire", injury: "Injury", closure: "Closure", collision: "Collision", roadwork: "Roadwork", other: "Other" };

function chpPopup(c) {
  return `<div class="pop"><h3>${esc(c.type)}</h3>
    <div class="m">${fmtTime(c.time)} · ${esc(c.area)} · CHP ${esc(c.center)} log ${esc(c.id)}${c.units ? ` · ${c.units} unit updates` : ""}</div>
    <div>${esc(c.location)}${c.where ? ` <span class="m">(${esc(c.where)})</span>` : ""}</div>
    ${c.details.length ? `<ul class="articles log">${c.details.slice().reverse().map((d) =>
      `<li><span class="m">${d.time ? fmtTime(d.time, false) : ""}</span> ${esc(d.text)}</li>`).join("")}</ul>` : ""}</div>`;
}

function incidentRows() {
  const rows = [];
  // Alerts, one row per kind of warning.
  const byEvent = {};
  for (const a of live.alerts) (byEvent[a.event] = byEvent[a.event] || []).push(a);
  const severity = { Extreme: 0, Severe: 1, Moderate: 2, Minor: 3 };
  for (const [event, alerts] of Object.entries(byEvent).sort((a, b) => (severity[a[1][0].severity] ?? 4) - (severity[b[1][0].severity] ?? 4))) {
    const ends = alerts.map((a) => a.ends).filter(Boolean).sort().pop();
    rows.push({ id: `alert:${event}`, tag: "Weather", tagClass: `alert-${alerts[0].kind}`, t: event,
      m: `${alerts.length} alert${alerts.length > 1 ? "s" : ""}${ends ? ` · until ${fmtTime(ends)}` : ""} · ${[...new Set(alerts.flatMap((a) => a.area.split("; ")))].slice(0, 4).join("; ")}` });
  }
  for (const f of fireData.fires.filter((f) => !f.minor)
    .sort((a, b) => (a.contained >= 100) - (b.contained >= 100) || b.news.length - a.news.length || (b.acres || 0) - (a.acres || 0))) {
    rows.push({ id: `fire:${f.key}`, tag: "Fire", tagClass: "fire", t: f.name,
      m: `${fmtAcres(f.acres)}${fmtContained(f.contained)} · ${f.news.length} articles · ${camerasOn(f).length} cameras facing${f.county ? ` · ${f.county}` : ""}` });
  }
  for (const c of chpCalls().filter((c) => c.kind === "fire")) {
    rows.push({ id: `chp:${c.id}`, tag: "CHP", tagClass: "chp-fire", t: c.type, m: `${fmtTime(c.time)} · ${c.location} · ${c.area}` });
  }
  for (const q of live.quakes.filter((q) => q.mag >= 2.5).sort((a, b) => b.mag - a.mag)) {
    rows.push({ id: `quake:${q.id}`, tag: "Quake", tagClass: "quake", t: `M${q.mag.toFixed(1)} ${q.place}`, m: `${fmtTime(q.time)} · ${q.depth_km.toFixed(0)} km deep` });
  }
  return rows;
}

function drawList() {
  const s = snapshot;
  const list = $("list");
  $("alertFilter").hidden = tab !== "news";
  $("roadworkFilter").hidden = tab !== "chp" || !live.chp;
  $("runPicker").hidden = tab !== "beliefs";
  $("detail").hidden = !(tab === "beliefs" && selectedBelief);
  list.hidden = !$("detail").hidden;
  let rows = [];
  if (tab === "incidents") {
    rows = incidentRows();
  } else if (tab === "news") {
    rows = s.news.filter((n) => !$("alertsOnly").checked || n.alert)
      .map((n) => ({ id: n.id, t: n.title,
        m: `${fmtTime(n.time)} · ${n.place} · ${n.outlets > 1 ? `${n.outlets} outlets` : n.domain}${n.categories.length ? ` · ${n.categories.join(", ")}` : ""}` }));
  } else if (tab === "chp") {
    rows = live.chp
      ? chpCalls().sort((a, b) => (b.time || "").localeCompare(a.time || ""))
        .map((c) => ({ id: `chp:${c.id}`, tag: CHP_LABEL[c.kind], tagClass: `chp-${c.kind}`, t: c.type, m: `${fmtTime(c.time)} · ${c.location} · ${c.area}` }))
      : s.incidents.map((i) => ({ id: i.id, t: i.description, m: `${fmtTime(i.time)} · ${i.lat.toFixed(3)}, ${i.lon.toFixed(3)}` }));
  } else if (tab === "beliefs") {
    if (selectedBelief) return drawBeliefDetail();
    return drawBeliefList();
  } else {
    rows = s.alertcalifornia.filter((c) => c.reaimed || c.bearings.length > 1)
      .map((c) => ({ id: `ac:${c.name}`, t: c.name, m: `now ${Math.round(c.bearing)}° · pointed ${c.bearings.map((b) => `${b}°`).join(", ")} in window` }));
  }
  list.innerHTML = rows.length
    ? rows.slice(0, 400).map((r) => `<li data-id="${esc(r.id)}"><div class="t">${r.tag ? `<span class="tag ${esc(r.tagClass)}">${esc(r.tag)}</span> ` : ""}${esc(r.t)}</div><div class="m">${esc(r.m)}</div></li>`).join("")
    : `<li class="empty">Nothing in this window.</li>`;
}

function focus(m) {
  if (!m) return;
  if (groups.news.hasLayer(m)) return groups.news.zoomToShowLayer(m, () => m.openPopup());
  if (m.getBounds) { map.fitBounds(m.getBounds(), { maxZoom: 11, padding: [20, 20] }); m.openPopup(m.getBounds().getCenter()); return; }
  map.setView(m.getLatLng(), Math.max(map.getZoom(), 13));
  m.openPopup();
}

$("list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-id]");
  if (!li) return;
  if (li.dataset.id.startsWith("belief:")) { selectedBelief = li.dataset.id.slice(7); drawBeliefs(); drawList(); return; }
  focus(markers[li.dataset.id]);
});

// ---- weather alerts and earthquakes -----------------------------------------

const ALERT_ORDER = { other: 0, water: 1, winter: 1, air: 2, heat: 3, wind: 4, fire: 5 };

function drawAlerts() {
  const byEvent = {};
  for (const a of live.alerts.slice().sort((a, b) => ALERT_ORDER[a.kind] - ALERT_ORDER[b.kind])) {
    const geometries = a.zones.map((z) => live.zones[z] && live.zones[z].geometry).filter(Boolean);
    if (!geometries.length) continue;
    const color = css(`--alert-${a.kind}`);
    const layer = L.geoJSON({ type: "GeometryCollection", geometries }, { pane: "alerts",
      style: { color, weight: a.kind === "fire" ? 2 : 1.5, fillColor: color, fillOpacity: 0.06, dashArray: a.severity === "Severe" || a.severity === "Extreme" ? null : "5 4" } })
      .bindPopup(() => alertPopup(a), { maxWidth: 380 })
      .addTo(groups.alerts);
    (byEvent[a.event] = byEvent[a.event] || []).push(layer);
  }
  for (const [event, layers] of Object.entries(byEvent)) markers[`alert:${event}`] = L.featureGroup(layers);
}

function alertPopup(a) {
  return `<div class="pop"><h3>${esc(a.event)}</h3>
    <div class="m">${esc(a.severity)} · ${a.onset ? `from ${fmtTime(a.onset)} ` : ""}${a.ends ? `until ${fmtTime(a.ends)}` : ""} · ${esc(a.sender)}</div>
    <div class="m">${esc(a.area)}</div>
    ${a.headline ? `<p>${esc(a.headline)}</p>` : ""}
    <details><summary>Full text</summary><p class="pre">${esc(a.description)}</p>${a.instruction ? `<p class="pre"><b>What to do:</b> ${esc(a.instruction)}</p>` : ""}</details></div>`;
}

function drawQuakes() {
  for (const q of live.quakes) {
    const m = L.circleMarker([q.lat, q.lon], { radius: 3 + Math.max(q.mag, 0) * 2.5, color: css("--quake"), weight: 1.5,
      fillColor: css("--quake"), fillOpacity: 0.25 })
      .bindPopup(`<div class="pop"><h3><a href="${safeUrl(q.url)}" target="_blank" rel="noopener">M${q.mag.toFixed(1)} ${esc(q.place)}</a></h3>
        <div class="m">${fmtTime(q.time)} · ${q.depth_km.toFixed(1)} km deep${q.felt ? ` · felt by ${q.felt} people` : ""} · ${esc(q.status)}</div></div>`)
      .addTo(groups.quakes);
    markers[`quake:${q.id}`] = m;
  }
}

// ---- agent beliefs ------------------------------------------------------------

const pct = (c) => `${Math.round(c * 100)}%`;
const STANCE = { supports: "For", contradicts: "Against", context: "Context" };
const beliefById = () => Object.fromEntries(beliefData.beliefs.map((b) => [b.id, b]));

function drawBeliefs() {
  const color = css("--belief");
  groups.beliefs.clearLayers();
  for (const b of beliefData.beliefs) {
    if (b.lat == null || b.lon == null) continue;
    const chosen = b.id === selectedBelief;
    const shape = L.circle([b.lat, b.lon], { radius: (b.radius_km || 1) * 1000, color, weight: chosen ? 3 : 1,
      opacity: b.status === "open" || b.status === "confirmed" ? 0.9 : 0.4, dashArray: b.status === "refuted" || b.status === "retracted" ? "4 4" : null,
      fillColor: color, fillOpacity: b.status === "open" || b.status === "confirmed" ? 0.04 + 0.22 * b.confidence : 0.02 })
      .bindTooltip(`${esc(b.statement)} · ${pct(b.confidence)}`, { sticky: true })
      .on("click", () => { selectedBelief = b.id; setTab("beliefs"); drawBeliefs(); drawList(); })
      .addTo(groups.beliefs);
    markers[`belief:${b.id}`] = shape;
  }
  const b = beliefById()[selectedBelief];
  if (!b || b.lat == null) return;
  const lines = L.layerGroup();
  for (const e of b.evidence) {
    if (e.lat == null || e.lon == null) continue;
    const c = css(`--stance-${e.stance}`);
    L.polyline([[b.lat, b.lon], [e.lat, e.lon]], { color: c, weight: 2, dashArray: e.stance === "context" ? "3 4" : null, interactive: false }).addTo(lines);
    L.circleMarker([e.lat, e.lon], { radius: 5, color: "#fff", weight: 1.5, fillColor: c, fillOpacity: 1 })
      .bindTooltip(`${STANCE[e.stance]}: ${esc(e.title || e.source)}`).addTo(lines);
  }
  lines.addTo(groups.beliefs);
}

function sparkline(history, width = 64, height = 18) {
  if (history.length < 2) return "";
  const t0 = new Date(history[0].time), t1 = new Date(history[history.length - 1].time);
  const x = (t) => ((new Date(t) - t0) / Math.max(t1 - t0, 1)) * width;
  const y = (c) => height - 1 - c * (height - 2);
  let d = `M0 ${y(history[0].confidence)}`;
  for (const h of history.slice(1)) d += ` H${x(h.time).toFixed(1)} V${y(h.confidence).toFixed(1)}`;
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true"><path d="${d}"/></svg>`;
}

function drawBeliefList() {
  const list = $("list");
  if (!runs.length) {
    list.innerHTML = `<li class="empty">No belief runs yet. An agent writes them to processing-output/beliefs as JSON Lines; see explorer/beliefs.py. <code>python -m explorer.example_beliefs</code> writes a rule-based example.</li>`;
    return;
  }
  if (!beliefData.beliefs.length) { list.innerHTML = `<li class="empty">This run had no beliefs yet at this time.</li>`; return; }
  const byTopic = {};
  for (const b of beliefData.beliefs) (byTopic[b.topic || "Other"] = byTopic[b.topic || "Other"] || []).push(b);
  const depth = (b, seen = new Set()) => b.parents.reduce((d, p) => {
    const parent = beliefById()[p];
    return parent && !seen.has(p) ? Math.max(d, 1 + depth(parent, new Set([...seen, p]))) : d;
  }, 0);
  // Topics with open beliefs first, most recently revised first.
  const rank = (beliefs) => [beliefs.some((b) => b.status === "open") ? 0 : 1, -Math.max(...beliefs.map((b) => new Date(b.time)))];
  const topics = Object.entries(byTopic).sort((a, b) => {
    const [x, y] = [rank(a[1]), rank(b[1])];
    return x[0] - y[0] || x[1] - y[1];
  });
  list.innerHTML = topics.map(([topic, beliefs]) => `<li class="group">${esc(topic)}</li>` +
    beliefs.sort((a, b) => depth(a) - depth(b)).map((b) => {
      const n = (s) => b.evidence.filter((e) => e.stance === s).length;
      return `<li data-id="belief:${esc(b.id)}" class="belief ${esc(b.status)}" style="--depth:${depth(b)}">
        <div class="t">${esc(b.statement)}</div>
        <div class="conf"><span class="bar"><i style="width:${pct(b.confidence)}"></i></span><b>${pct(b.confidence)}</b>${sparkline(b.history)}</div>
        <div class="m">${b.status !== "open" ? `<span class="status">${esc(b.status)}</span> · ` : ""}${n("supports")} for · ${n("contradicts")} against · revised ${fmtTime(b.time)}${b.revisions > 1 ? ` (${b.revisions} versions)` : ""}</div></li>`;
    }).join("")).join("");
}

function historyChart(b) {
  const W = 300, H = 80, pad = 4;
  const times = b.history.map((h) => new Date(h.time));
  const t0 = times[0], t1 = new Date(Math.max(times[times.length - 1], selectedTime()));
  const x = (t) => pad + ((t - t0) / Math.max(t1 - t0, 1)) * (W - 2 * pad);
  const y = (c) => H - pad - c * (H - 2 * pad);
  let d = `M${x(t0)} ${y(b.history[0].confidence)}`;
  b.history.forEach((h, i) => { if (i) d += ` H${x(times[i]).toFixed(1)} V${y(h.confidence).toFixed(1)}`; });
  d += ` H${x(t1).toFixed(1)}`;
  const dots = b.history.map((h, i) => `<circle cx="${x(times[i]).toFixed(1)}" cy="${y(h.confidence).toFixed(1)}" r="2.5"><title>${fmtTime(h.time)}: ${pct(h.confidence)} ${h.status}</title></circle>`).join("");
  return `<svg class="history" viewBox="0 0 ${W} ${H}" role="img" aria-label="Confidence over time">
    <line x1="${pad}" x2="${W - pad}" y1="${y(0.5)}" y2="${y(0.5)}" class="mid"/><text x="${pad}" y="${y(0.5) - 3}">50%</text>
    <path d="${d}"/>${dots}</svg>
    <div class="axis m"><span>${fmtTime(t0.toISOString())}</span><span>${fmtTime(t1.toISOString())}</span></div>`;
}

function drawBeliefDetail() {
  const el = $("detail"), byId = beliefById(), b = byId[selectedBelief];
  if (!b) {
    el.innerHTML = `<button type="button" class="back">← All beliefs</button><p class="m">This belief did not exist yet at the selected time.</p>`;
    return;
  }
  const link = (id) => byId[id] ? `<button type="button" class="blink" data-belief="${esc(id)}">${esc(byId[id].statement)} <span class="m">${pct(byId[id].confidence)}</span></button>` : `<span class="m">${esc(id)}</span>`;
  const section = (stance) => {
    const items = b.evidence.filter((e) => e.stance === stance);
    if (!items.length) return "";
    return `<div class="sub">${STANCE[stance]} (${items.length})</div><ul class="evidence">${items.map((e, i) => `
      <li class="${stance}" data-ev="${b.evidence.indexOf(e)}">
        <div><span class="tag src">${esc(e.source)}</span> ${e.url ? `<a href="${safeUrl(e.url)}" target="_blank" rel="noopener">${esc(e.title || e.url)}</a>` : esc(e.title || "")}</div>
        ${e.note ? `<div class="m">${esc(e.note)}</div>` : ""}
        <div class="m">${e.time ? fmtTime(e.time) : ""} · weight <span class="w"><i style="width:${pct(e.weight)}"></i></span>${e.lat != null ? ` · <button type="button" class="show">show on map</button>` : ""}</div>
        ${e.frame ? `<img loading="lazy" alt="${esc(e.title || "camera frame")}" src="/frame?p=${encodeURIComponent(e.frame)}">` : ""}
      </li>`).join("")}</ul>`;
  };
  el.innerHTML = `<button type="button" class="back">← All beliefs</button>
    <h2>${esc(b.statement)}</h2>
    <div class="m">${esc(b.kind || "belief")} · ${esc(b.status)} · ${esc(b.agent || "")} · ${esc(b.place || "")}</div>
    <div class="conf big"><span class="bar"><i style="width:${pct(b.confidence)}"></i></span><b>${pct(b.confidence)}</b></div>
    ${historyChart(b)}
    ${b.reasoning ? `<p class="reasoning">${esc(b.reasoning)}</p>` : ""}
    ${b.parents.length ? `<div class="sub">Builds on</div>${b.parents.map(link).join("")}` : ""}
    ${b.children.length ? `<div class="sub">Leads to</div>${b.children.map(link).join("")}` : ""}
    ${section("supports")}${section("contradicts")}${section("context")}
    ${b.evidence.length ? "" : `<p class="m">No evidence recorded.</p>`}`;
}

$("detail").addEventListener("click", (e) => {
  if (e.target.closest(".back")) { selectedBelief = null; drawBeliefs(); drawList(); return; }
  const other = e.target.closest("[data-belief]");
  if (other) { selectedBelief = other.dataset.belief; drawBeliefs(); drawList(); $("detail").scrollTop = 0; return; }
  const show = e.target.closest(".show");
  if (show) {
    const ev = beliefById()[selectedBelief].evidence[Number(show.closest("li").dataset.ev)];
    map.setView([ev.lat, ev.lon], Math.max(map.getZoom(), 12));
  }
});

// ---- camera popups with a frame scrubber -----------------------------------

function cameraPopup(source, c, at) {
  const el = document.createElement("div");
  el.className = "pop";
  const pointing = c.bearing != null
    ? `bearing ${Math.round(c.bearing)}°${c.tilt != null ? ` · tilt ${c.tilt.toFixed(1)}°` : ""}${c.zoom != null ? ` · zoom ${c.zoom}` : ""}` : "";
  el.innerHTML = `<h3>${esc(c.name)}</h3>
    <div class="m info">${fmtTime(c.time)} (${ago(c.time, at)}) ${pointing ? "· " + esc(pointing) : ""}</div>
    <img alt="Latest frame from ${esc(c.name)}" src="/frame?p=${encodeURIComponent(c.frame)}">
    <div class="scrub"><button type="button" aria-label="Play">▶</button><input type="range" min="0" max="0" value="0" aria-label="Frame"><span class="m n">${c.frames} frames</span></div>`;
  const img = el.querySelector("img"), slider = el.querySelector("input"), info = el.querySelector(".info"),
    play = el.querySelector("button"), n = el.querySelector(".n");
  let frames = [], timer = null;
  const show = (i) => {
    const f = frames[i];
    if (!f) return;
    img.src = `/frame?p=${encodeURIComponent(f.frame)}`;
    info.textContent = `${fmtTime(f.time)}${f.bearing != null ? ` · bearing ${Math.round(f.bearing)}°` : ""}${f.source === "1min-history" ? " · 1-min history" : ""}`;
    n.textContent = `${i + 1}/${frames.length}`;
  };
  const params = new URLSearchParams({ source, name: c.name, start: snapshot.start, end: snapshot.at });
  getJSON(`/api/frames?${params}`).then((list) => {
    frames = list;
    slider.max = Math.max(frames.length - 1, 0);
    slider.value = slider.max;
    n.textContent = `${frames.length}/${frames.length}`;
  });
  slider.addEventListener("input", () => show(Number(slider.value)));
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶"; return; }
    if (Number(slider.value) >= frames.length - 1) slider.value = 0;
    play.textContent = "❚❚";
    timer = setInterval(() => {
      const i = Number(slider.value) + 1;
      if (i >= frames.length) { clearInterval(timer); timer = null; play.textContent = "▶"; return; }
      slider.value = i; show(i);
    }, 250);
  });
  map.once("popupclose", () => timer && clearInterval(timer));
  return el;
}

// ---- controls ---------------------------------------------------------------

let pending = null, popupOpen = false;
map.on("popupopen", () => { popupOpen = true; });
map.on("popupclose", () => { popupOpen = false; });
const reload = () => { clearTimeout(pending); pending = setTimeout(loadSnapshot, 150); };
$("slider").addEventListener("input", () => {
  $("live").checked = false;
  $("clock").textContent = fmtTime(new Date(Number($("slider").value) * 1000).toISOString());
  reload();
});
$("live").addEventListener("change", () => { if ($("live").checked) $("slider").value = $("slider").max; reload(); });
$("window").addEventListener("change", reload);
$("alertsOnly").addEventListener("change", () => snapshot && draw());
$("hideRoadwork").addEventListener("change", () => snapshot && draw());
$("run").addEventListener("change", () => { selectedBelief = null; reload(); });
function setTab(name) {
  tab = name;
  for (const x of document.querySelectorAll(".tabs button")) x.setAttribute("aria-selected", String(x.dataset.tab === name));
  if (snapshot) drawList();
}
document.querySelector(".tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (b) setTab(b.dataset.tab);
});

(async function start() {
  try { await loadRange(); } catch (err) { $("status").textContent = `Could not reach the server: ${err.message}`; }
  await loadSnapshot();
  const wanted = /^#fire-([a-z0-9-]+)$/.exec(location.hash);
  const target = wanted && markers[`fire:${wanted[1].replace(/-/g, " ")}`];
  if (target) { map.setView(target.getLatLng(), 12); target.openPopup(); }
  const belief = /^#belief-(.+)$/.exec(location.hash);
  if (belief) { selectedBelief = decodeURIComponent(belief[1]); drawBeliefs(); }
  if (location.hash === "#beliefs" || belief) setTab("beliefs");
  setInterval(async () => {
    if (!$("live").checked) return;
    try { await loadRange(); } catch { /* keep the last view */ }
    if (!popupOpen) loadSnapshot();   // don't close an open popup
  }, 60000);
})();
