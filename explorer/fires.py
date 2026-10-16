"""Official wildfire feeds for the map explorer.

Fetched every few minutes and saved as timestamped snapshots, so the time
slider can show a fire as it was:

  CAL FIRE incident list     name, location, acres, containment, incident page
  NIFC WFIGS perimeters      mapped fire outlines
  NIFC WFIGS incident points every reported wildland fire, including small ones
  NASA FIRMS VIIRS (24 h)    satellite heat detections

None of these need an API key. Snapshots are written only when a feed's
content changes.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

from urllib.error import URLError
from urllib.request import Request, urlopen

# Southern California: wider than the camera area, so fires such as Henderson
# (San Diego County) or Norte (Orange County) that are in the news still show.
REGION = (32.5, -120.7, 35.8, -114.1)  # south, west, north, east
ENVELOPE = f"{REGION[1]},{REGION[0]},{REGION[3]},{REGION[2]}"
WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
WFIGS_QUERY = (f"/FeatureServer/0/query?where=1%3D1&geometry={ENVELOPE}&geometryType=esriGeometryEnvelope"
               "&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&outSR=4326&f=geojson")
FEEDS = {
    "calfire.json": "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false",
    "perimeters.geojson": WFIGS + "/WFIGS_Interagency_Perimeters_Current" + WFIGS_QUERY,
    "incidents.geojson": WFIGS + "/WFIGS_Incident_Locations_Current" + WFIGS_QUERY,
    "firms-noaa20.csv": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
    "firms-snpp.csv": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
}
KEEP_DAYS = 14
STAMP = "%Y%m%dT%H%M%SZ"


def in_region(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    south, west, north, east = REGION
    return south <= lat <= north and west <= lon <= east


def _ms(value: Any) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def fire_key(name: str) -> str:
    """"Bryson Fire " / "BRYSON" -> "bryson"; used to match feeds and news."""
    return re.sub(r"\s+fire\s*$", "", name.strip(), flags=re.I).strip().lower()


class FireFeeds:
    def __init__(self, cache_dir: str | Path):
        self.cache = Path(cache_dir)

    # -- fetching ------------------------------------------------------------

    def fetch(self, now: datetime | None = None) -> dict[str, str]:
        """Fetch every feed; save the ones whose content changed. Returns statuses."""
        now = now or datetime.now(timezone.utc)
        statuses = {}
        for name, url in FEEDS.items():
            try:
                with urlopen(Request(url, headers={"User-Agent": "urban-observation-explorer"}), timeout=60) as response:
                    content = response.read()
                if name.endswith("json"):
                    json.loads(content)  # refuse error pages
                elif not content.startswith(b"latitude"):
                    raise ValueError("not a FIRMS CSV")
            except (URLError, OSError, ValueError) as exc:
                statuses[name] = f"error: {exc}"
                continue
            if name.startswith("firms"):
                content = self._firms_region(content)
            statuses[name] = "saved" if self._save(name, content, now) else "unchanged"
        self._prune(now)
        return statuses

    @staticmethod
    def _firms_region(content: bytes) -> bytes:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8", errors="replace"))))
        header, body = rows[0], rows[1:]
        lat, lon = header.index("latitude"), header.index("longitude")
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(r for r in body if len(r) == len(header) and in_region(float(r[lat]), float(r[lon])))
        return out.getvalue().encode()

    def _save(self, name: str, content: bytes, now: datetime) -> bool:
        folder = self.cache / name.split(".")[0]
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        marker = folder / "latest.sha256"
        if marker.exists() and marker.read_text() == digest:
            return False
        target = folder / f"{now.strftime(STAMP)}.{name.split('.', 1)[1]}"
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        marker.write_text(digest)
        return True

    def _prune(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=KEEP_DAYS)).strftime(STAMP)
        for path in self.cache.glob("*/*"):
            if path.name[:16] < cutoff and path.name != "latest.sha256" and path.name[:1].isdigit():
                path.unlink(missing_ok=True)

    # -- reading -------------------------------------------------------------

    def _latest(self, feed: str, at: datetime) -> tuple[Path, datetime] | None:
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
        return best

    def snapshot(self, at: datetime) -> dict[str, Any]:
        """Fires, perimeters and heat detections as the feeds stood at `at`."""
        fires: dict[str, dict[str, Any]] = {}
        fetched = {}

        found = self._latest("calfire", at)
        if found:
            fetched["calfire"] = found[1]
            for item in json.loads(found[0].read_text()):
                lat, lon = item.get("Latitude"), item.get("Longitude")
                if not in_region(lat, lon):
                    continue
                started = item.get("Started")
                if started and started > _iso(at):
                    continue
                fires[fire_key(item["Name"])] = {
                    "name": item["Name"].strip(), "lat": lat, "lon": lon, "acres": item.get("AcresBurned"),
                    "contained": item.get("PercentContained"), "started": started, "updated": item.get("Updated"),
                    "county": item.get("County"), "location": item.get("Location"), "url": item.get("Url"),
                    "source": "CAL FIRE",
                }

        found = self._latest("incidents", at)
        if found:
            fetched["incidents"] = found[1]
            for feature in json.loads(found[0].read_text()).get("features", []):
                props = feature.get("properties") or {}
                lon, lat = (feature.get("geometry") or {}).get("coordinates", [None, None])[:2]
                name = props.get("IncidentName") or ""
                discovered = _ms(props.get("FireDiscoveryDateTime"))
                if not in_region(lat, lon) or (discovered and discovered > _iso(at)):
                    continue
                key = fire_key(name)
                if key in fires:  # CAL FIRE already has it; keep its numbers
                    fires[key].setdefault("nifc_size", props.get("IncidentSize"))
                    continue
                # Unnamed reports ("LAC-373571") are small dispatches; skip old ones.
                if discovered and parse(discovered) < at - timedelta(days=3) and not props.get("IncidentSize"):
                    continue
                fires[key] = {
                    "name": name.title() if not re.match(r"[A-Z]{3}-\d", name) else name,
                    "lat": lat, "lon": lon, "acres": props.get("IncidentSize"),
                    "contained": props.get("PercentContained"), "started": discovered,
                    "updated": _ms(props.get("ModifiedOnDateTime_dt")), "county": props.get("POOCounty"),
                    "location": props.get("IncidentShortDescription"), "url": None, "source": "NIFC",
                    # Dispatch numbers ("LAC-373571") and reports under 5 acres
                    # show as small dots and stay out of the list.
                    "minor": bool(re.match(r"[A-Z]{3}-\d", name)) or (props.get("IncidentSize") or 0) < 5,
                }

        perimeters = []
        found = self._latest("perimeters", at)
        if found:
            fetched["perimeters"] = found[1]
            for feature in json.loads(found[0].read_text()).get("features", []):
                props = feature.get("properties") or {}
                perimeters.append({"type": "Feature", "geometry": feature.get("geometry"), "properties": {
                    "name": (props.get("poly_IncidentName") or "").title(),
                    "acres": round(props.get("poly_Acres_AutoCalc") or 0, 1) or None,
                    "mapped": _ms(props.get("poly_DateCurrent")),
                    "behavior": ", ".join(filter(None, (props.get("attr_FireBehaviorGeneral"),
                                                        props.get("attr_FireBehaviorGeneral2")))),
                    "personnel": props.get("attr_TotalIncidentPersonnel"),
                }})

        hotspots = []
        for feed in ("firms-noaa20", "firms-snpp"):
            found = self._latest(feed, at)
            if not found:
                continue
            fetched[feed] = found[1]
            for row in csv.DictReader(io.StringIO(found[0].read_text())):
                try:
                    seen = datetime.strptime(row["acq_date"] + row["acq_time"].zfill(4), "%Y-%m-%d%H%M").replace(tzinfo=timezone.utc)
                    lat, lon = float(row["latitude"]), float(row["longitude"])
                except (KeyError, ValueError):
                    continue
                if seen <= at and at - seen <= timedelta(hours=24):
                    hotspots.append({"lat": lat, "lon": lon, "time": _iso(seen), "frp": row.get("frp"),
                                     "confidence": row.get("confidence"), "satellite": feed[6:].upper()})

        return {"fires": [dict(v, key=k) for k, v in fires.items()], "perimeters": perimeters,
                "hotspots": hotspots, "fetched": {k: _iso(v) for k, v in fetched.items()}}


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# -- news ----------------------------------------------------------------------

NAMED_FIRE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?) [Ff]ire\b")
ACRES = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*acres", re.I)
CONTAINED = re.compile(r"(\d{1,3})\s*(?:%|percent)\s*contain", re.I)
NOT_FIRES = {"the", "a", "brush", "wild", "house", "structure", "camp", "county", "los angeles", "angeles county",
             "southern california", "california", "red flag", "very high", "cal", "wildland", "metro", "orange county"}


def news_fires(title: str, body: str) -> dict[str, dict[str, Any]]:
    """Fires named in an article, with acreage and containment from sentences naming them.

    Returns {"bryson": {"acres": 200.0, "contained": 20}, ...}; numbers are
    the article's own claims, so the map labels them "per news".
    """
    text = f"{title}. {body or ''}"
    found: dict[str, dict[str, Any]] = {}
    for match in NAMED_FIRE.finditer(text):
        name = match.group(1)
        if name.split()[0] in ("The", "A", "An", "This", "That"):
            name = " ".join(name.split()[1:])
        key = fire_key(name)
        if not key or key in NOT_FIRES:
            continue
        found.setdefault(key, {})
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        named = [key for key in found if re.search(rf"\b{re.escape(key)}\b", sentence, re.I)]
        if len(named) != 1:  # skip sentences comparing several fires
            continue
        facts = found[named[0]]
        acres = ACRES.search(sentence)
        contained = CONTAINED.search(sentence)
        # Articles tell the story in order ("reported at two acres ... grew
        # to 101.5"), so keep the largest figure.
        if acres:
            facts["acres"] = max(facts.get("acres", 0), float(acres.group(1).replace(",", "")))
        if contained and int(contained.group(1)) <= 100:
            facts["contained"] = max(facts.get("contained", 0), int(contained.group(1)))
    return found
