"""Live public feeds for the map explorer.

Fetched on a schedule and saved as timestamped snapshots, so the time slider
can show each feed as it stood:

  chp-live       CHP computer-aided dispatch calls, every 2 minutes
  nws-alerts     National Weather Service watches, warnings and advisories
  quakes         USGS earthquakes of the past day
  calfire        CAL FIRE incident list
  perimeters     NIFC WFIGS mapped fire outlines
  incidents      NIFC WFIGS wildland fire reports, including small ones
  firms-*        NASA FIRMS VIIRS satellite heat detections (24 h)

None of these need an API key. Each feed is clipped to Southern California
before it is saved, and a snapshot is written only when the content changes.
Snapshots are kept for 14 days.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

# Wider than the camera area, so fires such as Henderson (San Diego County)
# that are in the news still show.
REGION = (32.5, -120.7, 35.8, -114.1)  # south, west, north, east
ENVELOPE = f"{REGION[1]},{REGION[0]},{REGION[3]},{REGION[2]}"
WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
WFIGS_QUERY = (f"/FeatureServer/0/query?where=1%3D1&geometry={ENVELOPE}&geometryType=esriGeometryEnvelope"
               "&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&outSR=4326&f=geojson")
FIRMS = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"
USER_AGENT = "urban-observation-explorer"
KEEP_DAYS = 14
STAMP = "%Y%m%dT%H%M%SZ"
LOS_ANGELES = ZoneInfo("America/Los_Angeles")


def in_region(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    south, west, north, east = REGION
    return south <= lat <= north and west <= lon <= east


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get(url: str, timeout: float = 60) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}), timeout=timeout) as response:
        return response.read()


# -- cleaning: raw response -> what is saved ---------------------------------------


def _json(content: bytes, store: "FeedStore") -> bytes:
    json.loads(content)  # refuse error pages
    return content


def _firms(content: bytes, store: "FeedStore") -> bytes:
    if not content.startswith(b"latitude"):
        raise ValueError("not a FIRMS CSV")
    rows = list(csv.reader(io.StringIO(content.decode("utf-8", errors="replace"))))
    header, body = rows[0], rows[1:]
    lat, lon = header.index("latitude"), header.index("longitude")
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(r for r in body if len(r) == len(header) and in_region(float(r[lat]), float(r[lon])))
    return out.getvalue().encode()


def _text(element: ET.Element | None) -> str:
    # Every CHP value comes wrapped in literal quotes.
    return (element.text or "").strip().strip('"').strip() if element is not None else ""


def _chp_time(value: str) -> str | None:
    try:
        local = datetime.strptime(value, "%b %d %Y %I:%M%p").replace(tzinfo=LOS_ANGELES)
    except ValueError:
        return None
    return iso(local)


def _chp(content: bytes, store: "FeedStore") -> bytes:
    """CHP's statewide XML -> the calls in the region, as JSON."""
    calls = []
    for dispatch in ET.fromstring(content).iter("Dispatch"):
        for log in dispatch.iter("Log"):
            try:
                north, west = _text(log.find("LATLON")).split(":")
                lat, lon = int(north) / 1e6, -int(west) / 1e6
            except ValueError:
                lat = lon = None
            if not in_region(lat, lon):
                continue
            details = [{"time": _chp_time(_text(d.find("DetailTime"))), "text": _text(d.find("IncidentDetail"))}
                       for d in log.iter("details")]
            calls.append({
                "id": log.get("ID"), "center": dispatch.get("ID"), "time": _chp_time(_text(log.find("LogTime"))),
                "type": _text(log.find("LogType")), "location": _text(log.find("Location")),
                "where": _text(log.find("LocationDesc")), "area": _text(log.find("Area")),
                "lat": lat, "lon": lon, "details": details[-12:], "units": len(list(log.iter("units"))),
            })
    calls.sort(key=lambda c: c["id"])  # stable order, so unchanged feeds hash the same
    return json.dumps(calls, separators=(",", ":")).encode()


def _quakes(content: bytes, store: "FeedStore") -> bytes:
    quakes = []
    for feature in json.loads(content).get("features", []):
        lon, lat, depth = (feature.get("geometry") or {}).get("coordinates", [None, None, None])[:3]
        props = feature.get("properties") or {}
        if not in_region(lat, lon) or props.get("type") != "earthquake":
            continue
        quakes.append({"id": feature.get("id"), "lat": lat, "lon": lon, "depth_km": depth, "mag": props.get("mag"),
                       "place": props.get("place"), "time": iso(datetime.fromtimestamp(props["time"] / 1000, timezone.utc)),
                       "felt": props.get("felt"), "url": props.get("url"), "status": props.get("status")})
    quakes.sort(key=lambda q: q["time"])
    return json.dumps(quakes, separators=(",", ":")).encode()


def _alerts(content: bytes, store: "FeedStore") -> bytes:
    """Statewide NWS alerts -> those whose zones reach the region, with zone ids."""
    alerts = []
    for feature in json.loads(content).get("features", []):
        props = feature.get("properties") or {}
        zones = [z.rstrip("/").split("/zones/")[-1] for z in props.get("affectedZones") or []]
        boxes = [store.zone_bbox(z) for z in zones]
        if not any(b and b[0] <= REGION[2] and b[2] >= REGION[0] and b[1] <= REGION[3] and b[3] >= REGION[1]
                   for b in boxes):
            continue
        alerts.append({
            "id": props.get("id"), "event": props.get("event"), "severity": props.get("severity"),
            "certainty": props.get("certainty"), "urgency": props.get("urgency"), "headline": props.get("headline"),
            "area": props.get("areaDesc"), "sender": props.get("senderName"),
            "sent": props.get("sent"), "onset": props.get("onset") or props.get("effective"),
            "ends": props.get("ends") or props.get("expires"),
            "description": (props.get("description") or "")[:2500], "instruction": (props.get("instruction") or "")[:1200],
            "zones": zones,
        })
    alerts.sort(key=lambda a: a["id"] or "")
    return json.dumps(alerts, separators=(",", ":")).encode()


@dataclass(frozen=True)
class Feed:
    url: str
    suffix: str
    every: float           # seconds between fetches
    clean: Callable[[bytes, "FeedStore"], bytes]


FEEDS = {
    "chp-live": Feed("https://media.chp.ca.gov/sa_xml/sa.xml", "json", 120, _chp),
    "nws-alerts": Feed("https://api.weather.gov/alerts/active?area=CA", "json", 300, _alerts),
    "quakes": Feed("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson", "json", 300, _quakes),
    "calfire": Feed("https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false", "json", 600, _json),
    "perimeters": Feed(WFIGS + "/WFIGS_Interagency_Perimeters_Current" + WFIGS_QUERY, "geojson", 600, _json),
    "incidents": Feed(WFIGS + "/WFIGS_Incident_Locations_Current" + WFIGS_QUERY, "geojson", 600, _json),
    "firms-noaa20": Feed(FIRMS + "/noaa-20-viirs-c2/csv/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv", "csv", 600, _firms),
    "firms-snpp": Feed(FIRMS + "/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv", "csv", 600, _firms),
}


def _thin(ring: list[list[float]], tolerance: float) -> list[list[float]]:
    """Douglas-Peucker on one ring or line."""
    if len(ring) < 4:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        first, last = stack.pop()
        (x1, y1), (x2, y2) = ring[first], ring[last]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        far, index = 0.0, None
        for i in range(first + 1, last):
            px, py = ring[i][0] - x1, ring[i][1] - y1
            # A closed ring starts and ends on the same point: measure from it.
            d = abs(dy * px - dx * py) / norm if norm else (px * px + py * py) ** 0.5
            if d > far:
                far, index = d, i
        if index is not None and far > tolerance:
            keep[index] = True
            stack += [(first, index), (index, last)]
    return [p for p, k in zip(ring, keep) if k]


def _simplify(coords: Any, tolerance: float = 0.002) -> Any:
    """Thin zone outlines to about 200 m; NWS draws them in great detail."""
    if coords and isinstance(coords[0][0], (int, float)):  # a ring
        return [[round(x, 4), round(y, 4)] for x, y in _thin([c[:2] for c in coords], tolerance)]
    return [_simplify(c, tolerance) for c in coords]


def _bbox(coords: Any, box: list[float] | None = None) -> list[float]:
    box = box or [90.0, 180.0, -90.0, -180.0]  # south, west, north, east
    if coords and isinstance(coords[0], (int, float)):
        lon, lat = coords[:2]
        return [min(box[0], lat), min(box[1], lon), max(box[2], lat), max(box[3], lon)]
    for item in coords:
        box = _bbox(item, box)
    return box


class FeedStore:
    def __init__(self, cache_dir: str | Path):
        self.cache = Path(cache_dir)
        self.last_fetch: dict[str, float] = {}
        self._zones: dict[str, dict[str, Any]] = {}

    # -- fetching ------------------------------------------------------------

    def fetch(self, now: datetime | None = None, only_due: bool = False) -> dict[str, str]:
        """Fetch the feeds (those due, with only_due); save changed ones. Returns statuses."""
        now = now or datetime.now(timezone.utc)
        statuses = {}
        for name, feed in FEEDS.items():
            if only_due and time.monotonic() - self.last_fetch.get(name, -1e9) < feed.every:
                continue
            self.last_fetch[name] = time.monotonic()
            try:
                content = feed.clean(get(feed.url), self)
            except (URLError, OSError, ValueError, ET.ParseError) as exc:
                statuses[name] = f"error: {exc}"
                continue
            statuses[name] = "saved" if self._save(name, feed.suffix, content, now) else "unchanged"
            # Snapshots are written only on change, so log every good fetch: it
            # tells "unchanged" apart from "collection was down".
            with (self.cache / name / "fetches.log").open("a") as log:
                log.write(now.strftime(STAMP) + "\n")
        self._prune(now)
        return statuses

    def _save(self, name: str, suffix: str, content: bytes, now: datetime) -> bool:
        folder = self.cache / name
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        marker = folder / "latest.sha256"
        if marker.exists() and marker.read_text() == digest:
            return False
        target = folder / f"{now.strftime(STAMP)}.{suffix}"
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        marker.write_text(digest)
        return True

    def _prune(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=KEEP_DAYS)).strftime(STAMP)
        for name in FEEDS:
            log = self.cache / name / "fetches.log"
            if log.exists() and log.read_text()[:16] < cutoff:
                log.write_text("".join(l + "\n" for l in log.read_text().splitlines() if l >= cutoff))
            for path in (self.cache / name).glob("2*"):
                if path.name[:16] < cutoff:
                    path.unlink(missing_ok=True)

    def zone(self, zone_id: str) -> dict[str, Any] | None:
        """An NWS zone outline ("forecast/CAZ554"), fetched once and kept."""
        if zone_id in self._zones:
            return self._zones[zone_id]
        path = self.cache / "nws-zones" / f"{zone_id.replace('/', '-')}.json"
        if path.exists():
            self._zones[zone_id] = json.loads(path.read_text())
            return self._zones[zone_id]
        if "/" not in zone_id or ".." in zone_id:
            return None
        try:
            feature = json.loads(get(f"https://api.weather.gov/zones/{zone_id}", timeout=30))
        except (URLError, OSError, ValueError):
            return None
        geometry = feature.get("geometry")
        if not geometry:
            return None
        zone = {"id": zone_id, "name": (feature.get("properties") or {}).get("name"),
                "geometry": {"type": geometry["type"], "coordinates": _simplify(geometry["coordinates"])}}
        zone["bbox"] = _bbox(zone["geometry"]["coordinates"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(zone, separators=(",", ":")))
        self._zones[zone_id] = zone
        return zone

    def zone_bbox(self, zone_id: str) -> list[float] | None:
        zone = self.zone(zone_id)
        return zone["bbox"] if zone else None

    # -- reading -------------------------------------------------------------

    def _fetched(self, feed: str, at: datetime, within: timedelta) -> bool:
        """Was the feed fetched successfully in the `within` before `at`?"""
        log = self.cache / feed / "fetches.log"
        if not log.exists():
            return False
        low, high = (at - within).strftime(STAMP), at.strftime(STAMP)
        return any(low <= line.strip() <= high for line in log.read_text().splitlines())

    def latest(self, feed: str, at: datetime, max_age: timedelta | None = None) -> tuple[Path, datetime] | None:
        """The newest snapshot of a feed taken at or before `at`."""
        folder = self.cache / feed
        if not folder.is_dir():
            return None
        best = None
        for path in folder.iterdir():
            try:
                stamp = datetime.strptime(path.name.split(".")[0], STAMP).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if stamp <= at and (best is None or stamp > best[1]):
                best = (path, stamp)
        if best and max_age is not None and at - best[1] > max_age and not self._fetched(feed, at, max_age):
            return None  # collection was down then
        return best

    def live(self, at: datetime, hours: float) -> dict[str, Any]:
        """CHP calls open at `at`, weather alerts in effect, and the window's earthquakes.

        Alerts name their zones; `zones` holds each outline once.

        `chp` is None when live CHP collection was not running at `at`, so the
        map can fall back to the daily CHP file.
        """
        result: dict[str, Any] = {"chp": None, "alerts": [], "zones": {}, "quakes": [], "fetched": {}}
        found = self.latest("chp-live", at, timedelta(minutes=30))
        if found:
            result["fetched"]["chp-live"] = iso(found[1])
            result["chp"] = [dict(c, kind=chp_kind(c["type"])) for c in json.loads(found[0].read_text())
                             if not c["time"] or c["time"] <= iso(at)]

        found = self.latest("nws-alerts", at, timedelta(hours=2))
        if found:
            result["fetched"]["nws-alerts"] = iso(found[1])
            for alert in json.loads(found[0].read_text()):
                onset, ends = alert.get("onset"), alert.get("ends")
                if (onset and parse(onset) > at) or (ends and parse(ends) <= at):
                    continue
                for zone_id in alert["zones"]:
                    zone = self.zone(zone_id)
                    if zone:
                        result["zones"][zone_id] = {"name": zone["name"], "geometry": zone["geometry"]}
                result["alerts"].append({**alert, "kind": alert_kind(alert.get("event") or "")})

        found = self.latest("quakes", at, timedelta(hours=2))
        if found:
            result["fetched"]["quakes"] = iso(found[1])
            start = iso(at - timedelta(hours=hours))
            result["quakes"] = [q for q in json.loads(found[0].read_text()) if start <= q["time"] <= iso(at)]
        return result


def chp_kind(log_type: str) -> str:
    """Group CHP call types for the map: fire, injury, closure, collision, roadwork or other."""
    upper = log_type.upper()
    if upper.startswith(("CZP", "MZP")) or "CONSTRUCTION" in upper or "MAINTENANCE" in upper:
        return "roadwork"
    if "FIRE" in upper:
        return "fire"
    if "1144" in upper or "FATAL" in upper or ("INJ" in upper and "NO INJ" not in upper) or "1141" in upper:
        return "injury"
    if "SIG" in upper or "CLOSURE" in upper or "1184" in upper:
        return "closure"
    if "COLLISION" in upper or "HIT AND RUN" in upper:
        return "collision"
    return "other"


def alert_kind(event: str) -> str:
    lower = event.lower()
    for kind, words in (("fire", ("red flag", "fire")), ("wind", ("wind",)), ("heat", ("heat",)),
                        ("air", ("air quality", "smoke", "dust")), ("water", ("flood", "surf", "rip", "marine", "beach")),
                        ("winter", ("winter", "freeze", "frost", "snow"))):
        if any(w in lower for w in words):
            return kind
    return "other"
