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

const LAYERS = [
  { key: "alertcalifornia", label: "AlertCalifornia cameras", color: "--ac",
    note: "Wedge = view direction and field of view. Red = re-aimed in this window." },
  { key: "cctv", label: "Caltrans CCTV", color: "--cctv" },
  { key: "incidents", label: "CHP incidents", color: "--chp" },
  { key: "news", label: "News (GDELT)", color: "--news" },
  { key: "weather", label: "Weather", color: "--weather" },
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
  const label = document.createElement("label");
  label.innerHTML = `<input type="checkbox" checked data-layer="${l.key}">
    <span class="swatch" style="background:var(${l.color})"></span>
    <span>${l.label}</span><span class="count" id="count-${l.key}">–</span>`;
  layersEl.append(label);
  const note = document.createElement("div");
  note.className = "note"; note.id = `note-${l.key}`; note.textContent = l.note || "";
  note.hidden = !l.note;
  layersEl.append(note);
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
let tab = "news";
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
  const chp = info.sources["pems-incidents"];
  $("note-incidents").hidden = !chp;
  if (chp) $("note-incidents").textContent = `CHP publishes a daily file the next morning; latest incident ${fmtTime(chp.end)}.`;
  $("status").textContent = `Data refreshed ${fmtTime(info.refreshed)} · times are Los Angeles`;
}

async function loadSnapshot() {
  const at = selectedTime();
  $("clock").textContent = fmtTime(at.toISOString());
  const hours = $("window").value;
  try {
    snapshot = await getJSON(`/api/snapshot?at=${encodeURIComponent(at.toISOString())}&hours=${hours}`);
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

  for (const i of s.incidents) {
    const m = L.circleMarker([i.lat, i.lon], { radius: 6, color: "#fff", weight: 1.5, fillColor: css("--chp"), fillOpacity: 0.95 })
      .bindPopup(`<div class="pop"><h3>${esc(i.description)}</h3>
        <div class="m">${fmtTime(i.time)} · CHP incident ${esc(i.incident)}${i.duration_minutes ? ` · lasted ${esc(i.duration_minutes)} min` : ""}</div></div>`)
      .addTo(groups.incidents);
    markers[i.id] = m;
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

  const counts = { alertcalifornia: s.alertcalifornia.length, cctv: s.cctv.length, incidents: s.incidents.length,
    news: news.length, weather: s.weather.length };
  for (const [k, v] of Object.entries(counts)) $(`count-${k}`).textContent = v;
  drawList();
}

function drawList() {
  const s = snapshot;
  const list = $("list");
  $("alertFilter").hidden = tab !== "news";
  let rows = [];
  if (tab === "news") {
    rows = s.news.filter((n) => !$("alertsOnly").checked || n.alert)
      .map((n) => ({ id: n.id, t: n.title,
        m: `${fmtTime(n.time)} · ${n.place} · ${n.outlets > 1 ? `${n.outlets} outlets` : n.domain}${n.categories.length ? ` · ${n.categories.join(", ")}` : ""}` }));
  } else if (tab === "incidents") {
    rows = s.incidents.map((i) => ({ id: i.id, t: i.description, m: `${fmtTime(i.time)} · ${i.lat.toFixed(3)}, ${i.lon.toFixed(3)}` }));
  } else {
    rows = s.alertcalifornia.filter((c) => c.reaimed || c.bearings.length > 1)
      .map((c) => ({ id: `ac:${c.name}`, t: c.name, m: `now ${Math.round(c.bearing)}° · pointed ${c.bearings.map((b) => `${b}°`).join(", ")} in window` }));
  }
  list.innerHTML = rows.length
    ? rows.slice(0, 400).map((r) => `<li data-id="${esc(r.id)}"><div class="t">${esc(r.t)}</div><div class="m">${esc(r.m)}</div></li>`).join("")
    : `<li class="empty">Nothing in this window.</li>`;
}

$("list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-id]");
  const m = li && markers[li.dataset.id];
  if (!m) return;
  if (groups.news.hasLayer(m)) groups.news.zoomToShowLayer(m, () => m.openPopup());
  else { map.setView(m.getLatLng(), Math.max(map.getZoom(), 13)); m.openPopup(); }
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
document.querySelector(".tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  tab = b.dataset.tab;
  for (const x of document.querySelectorAll(".tabs button")) x.setAttribute("aria-selected", String(x === b));
  if (snapshot) drawList();
});

(async function start() {
  try { await loadRange(); } catch (err) { $("status").textContent = `Could not reach the server: ${err.message}`; }
  await loadSnapshot();
  setInterval(async () => {
    if (!$("live").checked) return;
    try { await loadRange(); } catch { /* keep the last view */ }
    if (!popupOpen) loadSnapshot();   // don't close an open popup
  }, 60000);
})();
