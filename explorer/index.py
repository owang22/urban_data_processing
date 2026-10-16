"""In-memory index of located observations for the map explorer.

Built from the replay readers, so the explorer shows what replay would send.
Each (source, day) partition is re-read only when its folder changes.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import threading
from typing import Any

from replay.catalog import DataCatalog
from replay.readers import SOURCE_FOLDERS, observations

# Greater Los Angeles, with the Santa Monica Mountains and Orange County coast.
BBOX = (33.55, -119.2, 34.85, -117.4)  # south, west, north, east

SOURCES = ("alertcalifornia", "cctv", "pems-incidents", "gdelt", "weather")

# GDELT V1 themes grouped into the categories the map shows. Matching is
# exact: substrings catch too much ("KILL" is inside "SKILLS", and the generic
# CRISISLEX tags appear on most articles).
THEME_CATEGORIES = {
    "fire": {"DISASTER_FIRE", "NATURAL_DISASTER_WILDFIRE", "NATURAL_DISASTER_WILDFIRES",
             "NATURAL_DISASTER_BRUSH_FIRE", "NATURAL_DISASTER_FIRE", "NATURAL_DISASTER_FOREST_FIRE"},
    "evacuation": {"EVACUATION", "CRISISLEX_T09_DISPLACEDRELOCATEDEVACUATED"},
    "power outage": {"POWER_OUTAGE", "MANMADE_DISASTER_POWER_OUTAGES", "MANMADE_DISASTER_WITHOUT_POWER",
                     "MANMADE_DISASTER_WITHOUT_ELECTRICITY"},
    "wind": {"NATURAL_DISASTER_STRONG_WINDS", "NATURAL_DISASTER_HIGH_WINDS", "NATURAL_DISASTER_WINDSTORM"},
    "earthquake": {"NATURAL_DISASTER_EARTHQUAKE", "NATURAL_DISASTER_EARTHQUAKES"},
    "flood": {"NATURAL_DISASTER_FLOOD", "NATURAL_DISASTER_FLOODS", "NATURAL_DISASTER_FLOODING"},
    "casualties": {"CRISISLEX_T02_INJURED", "CRISISLEX_T03_DEAD"},
    "traffic": {"TRAFFIC", "MANMADE_DISASTER_TRAFFIC_ACCIDENT", "TRAFFIC_ACCIDENT"},
    "crime": {"ARREST", "KILL", "WOUND", "CRIME_COMMON_ROBBERY", "SHOOTING", "TERROR"},
}
# Casualties alone also tags obituaries and sports, so it is not an emergency by itself.
EMERGENCY = ("fire", "evacuation", "power outage", "wind", "earthquake", "flood")


def categories(themes: str) -> list[str]:
    names = set(themes.split(";"))
    return [name for name, wanted in THEME_CATEGORIES.items() if names & wanted]


# CAMEO root event codes, for GDELT event rows that have no article title.
CAMEO_ROOTS = {
    "01": "Public statement", "02": "Appeal", "03": "Intent to cooperate", "04": "Consult",
    "05": "Diplomatic cooperation", "06": "Material cooperation", "07": "Provide aid", "08": "Yield",
    "09": "Investigate", "10": "Demand", "11": "Disapprove", "12": "Reject", "13": "Threaten",
    "14": "Protest", "15": "Show of force", "16": "Reduce relations", "17": "Coerce", "18": "Assault",
    "19": "Fight", "20": "Mass violence",
}


def in_bbox(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    south, west, north, east = BBOX
    return south <= lat <= north and west <= lon <= east


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def gkg_location(fields: list[str], title: str = "") -> tuple[float, float, str] | None:
    """Pick the most specific Los Angeles-area place from a GKG V1Locations field.

    Entries are "type#name#country#adm1#lat#lon#feature" separated by ";".
    A place named in the headline wins; generic "Los Angeles" is used only
    when nothing more specific is named.
    """
    if len(fields) <= 9:
        return None
    fallback = None
    first = None
    headline = title.lower()
    for entry in fields[9].split(";"):
        parts = entry.split("#")
        if len(parts) < 6 or parts[0] in ("1", "2", "5"):  # country or state level
            continue
        try:
            lat, lon = float(parts[4]), float(parts[5])
        except ValueError:
            continue
        if not in_bbox(lat, lon):
            continue
        name = parts[1].replace(", United States", "")
        if name in ("Los Angeles, California", "Los Angeles County, California"):
            fallback = fallback or (lat, lon, name)
            continue
        if headline and name.split(",")[0].lower() in headline:
            return lat, lon, name
        first = first or (lat, lon, name)
    return first or fallback


def event_location(fields: list[str]) -> tuple[float, float, str] | None:
    """ActionGeo of a GDELT 2.0 event row (columns 52, 56, 57)."""
    if len(fields) <= 57:
        return None
    try:
        lat, lon = float(fields[56]), float(fields[57])
    except ValueError:
        return None
    if not in_bbox(lat, lon):
        return None
    return lat, lon, fields[52].replace(", United States", "")


@dataclass
class Frame:
    time: datetime
    path: str
    lat: float
    lon: float
    data: dict[str, Any]


@dataclass
class Item:
    time: datetime
    lat: float
    lon: float
    props: dict[str, Any]


@dataclass
class Partition:
    signature: tuple
    cameras: dict[str, list[Frame]] = field(default_factory=dict)
    items: list[Item] = field(default_factory=list)


def folder_signature(folder: Path) -> tuple:
    """Modification times of a day folder and its direct subfolders.

    Collectors add files inside per-camera or per-station subfolders, which
    changes the subfolder's time but not the day folder's.
    """
    try:
        times = [folder.stat().st_mtime_ns]
        with os.scandir(folder) as entries:
            times.extend(e.stat().st_mtime_ns for e in entries if e.is_dir())
        return (len(times), max(times))
    except OSError:
        return ()


class ExplorerIndex:
    def __init__(self, data_root: str | Path, assets: str | Path, days: int = 3):
        self.data_root = Path(data_root).resolve()
        self.catalog = DataCatalog(self.data_root, "/nonexistent")
        self.days = days
        self.options = {
            "weather_locations": str(Path(assets) / "owm_locations.txt"),
            "cctv_locations": str(Path(assets) / "cctv.kml"),
        }
        self.partitions: dict[tuple[str, str], Partition] = {}
        self.lock = threading.Lock()
        self.refreshed: datetime | None = None

    # -- loading -----------------------------------------------------------

    def recent_days(self, source: str) -> list[str]:
        return self.catalog.dates(SOURCE_FOLDERS[source])[-self.days:]

    def refresh(self) -> None:
        wanted = {(source, day) for source in SOURCES for day in self.recent_days(source)}
        for source, day in sorted(wanted):
            folder = self.data_root / SOURCE_FOLDERS[source] / day
            signature = folder_signature(folder)
            current = self.partitions.get((source, day))
            if current is not None and current.signature == signature:
                continue
            partition = self.load(source, day, signature)
            with self.lock:
                self.partitions[(source, day)] = partition
        with self.lock:
            for key in list(self.partitions):
                if key not in wanted:
                    del self.partitions[key]
            self.refreshed = datetime.now(timezone.utc)

    def load(self, source: str, day: str, signature: tuple) -> Partition:
        result = Partition(signature)
        stored = self.catalog.partition(SOURCE_FOLDERS[source], day)
        if stored is None:
            return result
        seen_urls: set[str] = set()
        for obs in observations(stored, source, **self.options):
            time = parse_time(obs.time)
            if source in ("alertcalifornia", "cctv"):
                if obs.latitude is None or not obs.files:
                    continue
                frame = Frame(time, obs.files[0].path, obs.latitude, obs.longitude, obs.data)
                result.cameras.setdefault(obs.sensor, []).append(frame)
            elif source == "pems-incidents":
                if in_bbox(obs.latitude, obs.longitude):
                    result.items.append(Item(time, obs.latitude, obs.longitude, {
                        "id": obs.id, "description": obs.data.get("description", ""),
                        "duration_minutes": obs.data.get("duration") or None,
                        "incident": obs.sensor}))
            elif source == "weather":
                if obs.latitude is not None:
                    result.items.append(Item(time, obs.latitude, obs.longitude,
                                             {"station": obs.sensor, **obs.data}))
            elif source == "gdelt":
                item = self.news_item(obs, time, seen_urls)
                if item is not None:
                    result.items.append(item)
        for frames in result.cameras.values():
            frames.sort(key=lambda f: f.time)
        result.items.sort(key=lambda i: i.time)
        return result

    @staticmethod
    def news_item(obs, time: datetime, seen_urls: set[str]) -> Item | None:
        fields = obs.data.get("provider_fields") or []
        url = obs.data.get("url", "")
        if obs.data.get("kind") == "gkg":
            title = obs.data.get("title") or ""
            place = gkg_location(fields, title)
            themes = fields[7] if len(fields) > 7 else ""
        else:
            place = event_location(fields)
            code = fields[28] if len(fields) > 28 else ""
            themes = "KILL" if code in ("18", "19", "20") else ""
            title = ""
            label = CAMEO_ROOTS.get(code.zfill(2), "Event")
        if place is None or not url or url in seen_urls:
            return None
        seen_urls.add(url)
        domain = url.split("/")[2] if url.count("/") >= 2 else url
        found = categories(themes)
        return Item(time, place[0], place[1], {
            "id": obs.id, "kind": obs.data.get("kind"), "url": url, "domain": domain,
            "title": title or (f"{label} ({domain})" if obs.data.get("kind") != "gkg" else domain),
            "place": place[2], "categories": found,
            "alert": any(name in EMERGENCY for name in found),
        })

    # -- queries -----------------------------------------------------------

    def _partitions(self, source: str) -> list[Partition]:
        with self.lock:
            return [p for (s, _), p in sorted(self.partitions.items()) if s == source]

    def time_range(self) -> dict[str, Any]:
        ranges = {}
        for source in SOURCES:
            times = []
            for part in self._partitions(source):
                times += [frames[-1].time for frames in part.cameras.values()]
                times += [frames[0].time for frames in part.cameras.values()]
                if part.items:
                    times += [part.items[0].time, part.items[-1].time]
            if times:
                ranges[source] = {"start": iso(min(times)), "end": iso(max(times))}
        return {"sources": ranges, "refreshed": iso(self.refreshed) if self.refreshed else None}

    def snapshot(self, at: datetime, hours: float) -> dict[str, Any]:
        start = at - timedelta(hours=hours)
        result: dict[str, Any] = {"at": iso(at), "start": iso(start)}
        for source in ("alertcalifornia", "cctv"):
            result[source] = self._cameras(source, start, at)
        result["incidents"] = self._items("pems-incidents", start, at)
        result["news"] = self.merge_copies(self._items("gdelt", start, at))
        result["weather"] = self._latest_weather(at)
        return result

    def _cameras(self, source: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        merged: dict[str, list[Frame]] = {}
        for part in self._partitions(source):
            for name, frames in part.cameras.items():
                merged.setdefault(name, []).extend(frames)
        cameras = []
        for name, frames in merged.items():
            times = [f.time for f in frames]
            last = bisect_right(times, end) - 1
            if last < 0 or frames[last].time < start:
                continue
            first = bisect_right(times, start)
            window = frames[first:last + 1]
            frame = frames[last]
            data = frame.data
            cameras.append({
                "name": name, "lat": frame.lat, "lon": frame.lon, "time": iso(frame.time),
                "frame": self.relative(frame.path), "frames": len(window),
                "bearing": data.get("direction_degrees"), "fov": data.get("field_of_view_degrees"),
                "tilt": data.get("tilt_degrees"), "zoom": data.get("zoom"),
                "camera_id": data.get("camera_id"),
                "reaimed": any(f.data.get("reaimed") for f in window),
                "bearings": sorted({round(f.data["direction_degrees"]) for f in window
                                    if f.data.get("direction_degrees") is not None}),
            })
        return cameras

    def frames(self, source: str, name: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        found = []
        for part in self._partitions(source):
            for frame in part.cameras.get(name, []):
                if start <= frame.time <= end:
                    found.append({"time": iso(frame.time), "frame": self.relative(frame.path),
                                  "bearing": frame.data.get("direction_degrees"),
                                  "source": frame.data.get("frame_source")})
        return sorted(found, key=lambda f: f["time"])

    def _items(self, source: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        found = []
        for part in self._partitions(source):
            for item in part.items:
                if start <= item.time <= end:
                    found.append({"time": iso(item.time), "lat": item.lat, "lon": item.lon, **item.props})
        return sorted(found, key=lambda i: i["time"], reverse=True)

    @staticmethod
    def merge_copies(news: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the newest copy of each wire story; count the other outlets."""
        merged: dict[str, dict[str, Any]] = {}
        for item in news:  # newest first
            key = " ".join(item["title"].lower().split()) if item["kind"] == "gkg" and item["title"] != item["domain"] else item["url"]
            if key in merged:
                merged[key]["outlets"] += 1
            else:
                merged[key] = {**item, "outlets": 1}
        return list(merged.values())

    def _latest_weather(self, at: datetime) -> list[dict[str, Any]]:
        latest: dict[str, Item] = {}
        for part in self._partitions("weather"):
            for item in part.items:
                if item.time <= at and at - item.time <= timedelta(hours=6):
                    latest[item.props["station"]] = item
        return [{"time": iso(i.time), "lat": i.lat, "lon": i.lon, **i.props} for i in latest.values()]

    def relative(self, path: str) -> str:
        return Path(path).resolve().relative_to(self.data_root).as_posix()

    def frame_path(self, relative: str) -> Path | None:
        """Resolve a frame path from the browser, refusing anything outside the data root."""
        path = (self.data_root / relative).resolve()
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png") or not path.is_file():
            return None
        if os.path.commonpath([path, self.data_root]) != str(self.data_root):
            return None
        return path
