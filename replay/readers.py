"""Pure adapters from stored provider formats to common observations."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from email.utils import parsedate_to_datetime
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Iterator
import xml.etree.ElementTree as ET
from zipfile import ZipFile
try:  # Python 3.10+ is supported; this fallback keeps older dev hosts usable.
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - exercised only on unsupported Python
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        value = gettz(name)
        if value is None:
            raise ValueError(f"Unknown timezone: {name}")
        return value

from .catalog import Partition, StoredFile
from .model import FileReference, Observation


SOURCE_FOLDERS = {
    "air": "air_data", "weather": "weather_data", "pems-stations": "pem_data_station_5min",
    "pems-incidents": "pem_data_chp_incidents_day", "cctv": "cctv",
    "alertcalifornia": "alertcalifornia", "citizen": "citizen_data",
    "twitter": "twitter_data", "gdelt": "gkg",
}


def _timestamp(value: Any, fallback_day: str, naive_timezone: str = "UTC") -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{10}(?:\.\d+)?", text):
        return datetime.fromtimestamp(float(text), timezone.utc).isoformat().replace("+00:00", "Z")
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
    if re.fullmatch(r"\d{14}", text):
        parsed = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo(naive_timezone))
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=ZoneInfo(naive_timezone))
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(naive_timezone))
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        try:
            parsed = parsedate_to_datetime(re.sub(r"\s*\([^)]*\)\s*$", "", text))
            if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=ZoneInfo(naive_timezone))
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError):
            parsed = datetime.strptime(fallback_day, "%Y%m%d").replace(tzinfo=ZoneInfo(naive_timezone))
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _id(source: str, *values: Any) -> str:
    digest = hashlib.sha256("\0".join(str(v) for v in values).encode()).hexdigest()[:24]
    return f"{source}:{digest}"


def _raw(partition: Partition, stored: StoredFile, fmt: str, row: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"format": fmt, "path": str(stored.path), "member": stored.member}
    if row is not None:
        value["row"] = row
    return value


def _csv_rows(stored: StoredFile) -> Iterator[list[str]]:
    yield from csv.reader(io.StringIO(stored.read_text()))


def _file_ref(stored: StoredFile, media_type: str) -> FileReference:
    return FileReference(str(stored.path), media_type, stored.member)


def read_air(partition: Partition, *, local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    for stored in partition.files():
        if stored.suffix != ".csv": continue
        time = _timestamp(stored.name.split(".")[0], partition.day, local_timezone)
        for index, row in enumerate(_csv_rows(stored)):
            if len(row) < 4 or not row[0].strip().replace(".", "", 1).isdigit(): continue
            sensor = row[0].strip()
            try: lat, lon, pm25 = float(row[1]), float(row[2]), float(row[3])
            except ValueError: continue
            yield Observation(_id("air", sensor, time), "air", time, sensor, {"pm25": pm25},
                              latitude=lat, longitude=lon, raw=_raw(partition, stored, "purpleair_csv.v1", index))


def _location_map(path: str | None) -> dict[str, tuple[float, float]]:
    result = {}
    if not path: return result
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4: result[f"{parts[0]}, {parts[1]}"] = (float(parts[2]), float(parts[3]))
    except OSError: pass
    return result


def read_weather(partition: Partition, *, weather_locations: str | None = None,
                 local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    locations = _location_map(weather_locations)
    for stored in partition.files():
        if stored.suffix != ".csv": continue
        parts = PurePosixPath(stored.relative_path).parts
        sensor = parts[-2].replace("  ", " ") if len(parts) > 1 else "unknown"
        lat, lon = locations.get(sensor, (None, None))
        # OpenWeather names files with the API observation's UTC timestamp
        # (weather_extract.py uses utcfromtimestamp), unlike collectors such as
        # CCTV and PurpleAir that name files in the host's local timezone.
        time = _timestamp(stored.name.split(".")[0], partition.day, "UTC")
        for index, row in enumerate(_csv_rows(stored)):
            if len(row) < 4: continue
            try:
                data = {"temperature_f": float(row[0]), "description": row[1],
                        "humidity": float(row[2]), "wind_speed_mps": float(row[3])}
                if len(row) > 4 and row[4] != "": data["wind_direction_degrees"] = float(row[4])
            except ValueError: continue
            yield Observation(_id("weather", sensor, time, index), "weather", time, sensor, data,
                              latitude=lat, longitude=lon, raw=_raw(partition, stored, "openweather_csv.v2" if len(row)>4 else "openweather_csv.v1", index))


def _email_value(row: dict[str, str], *names: str) -> str:
    lower = {str(k).strip().lower(): str(v or "").strip() for k, v in row.items()}
    return next((lower[n.lower()] for n in names if lower.get(n.lower())), "")


def read_email(partition: Partition, *, source: str, local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    for stored in partition.files():
        if stored.suffix != ".csv": continue
        rows = csv.DictReader(io.StringIO(stored.read_text()))
        for index, row in enumerate(rows):
            current = _email_value(row, "schema_version") == "email_raw.v1"
            received = _email_value(row, "received_at", "email_time", "original date", "timestamp", "date")
            event_time = _email_value(row, "start_time", "timestamp")
            if event_time.lower() in {"n/a", "na", "none", "null"}: event_time = ""
            time = _timestamp(event_time or received, partition.day, local_timezone)
            end = _email_value(row, "end_time")
            sensor = _email_value(row, "message_id", "imap_uid", "author") or stored.name
            data = {"sender": _email_value(row, "sender", "author"), "subject": _email_value(row, "subject"),
                    "body": _email_value(row, "body", "description")}
            if not current:
                data.update({"legacy_event_name": _email_value(row, "event name", "event"),
                             "legacy_event_type": _email_value(row, "event type", "event"),
                             "legacy_location": _email_value(row, "location")})
            fmt = "email_raw.v1" if current else f"{source}_email.legacy"
            yield Observation(_id(source, sensor, received, index), source, time, sensor, data,
                              end_time=_timestamp(end, partition.day, local_timezone) if end else None,
                              raw=_raw(partition, stored, fmt, index))


def _camera_locations(path: str | None) -> dict[str, tuple[float, float]]:
    result = {}
    if not path: return result
    try:
        root = ET.parse(path).getroot()
        for placemark in root.iter():
            if not placemark.tag.endswith("Placemark"): continue
            name = next((e.text for e in placemark.iter() if e.tag.endswith("name") and e.text), None)
            coordinates = next((e.text for e in placemark.iter() if e.tag.endswith("coordinates") and e.text), None)
            if name and coordinates:
                values = coordinates.strip().split(",")
                result[_camera_key(name)] = (float(values[1]), float(values[0]))
    except (OSError, ValueError, ET.ParseError): pass
    return result


def _camera_key(name: str) -> str:
    # Collected folder names are filesystem-safe variants of the KML names
    # ("I-10 : (101) Mcclure West" -> "I-10 - (101) Mcclure West",
    # "Griffith/San Pedro" -> "GriffithSan Pedro"), so match on letters and digits.
    return re.sub(r"[^a-z0-9]", "", name.lower())


def read_cameras(partition: Partition, *, source: str, cctv_locations: str | None = None,
                 local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    files = list(partition.files())
    by_rel = {item.relative_path: item for item in files}
    camera_locations = _camera_locations(cctv_locations) if source == "cctv" else {}
    for stored in files:
        if stored.suffix not in {".jpg", ".jpeg", ".png"}: continue
        parts = PurePosixPath(stored.relative_path).parts
        sensor = parts[-2] if len(parts) > 1 else "unknown"
        stem = PurePosixPath(stored.name).stem
        time = _timestamp(stem, partition.day, local_timezone)
        lat, lon = camera_locations.get(_camera_key(sensor), (None, None)); data: dict[str, Any] = {}; refs = [_file_ref(stored, "image/jpeg")]
        if source == "cctv":
            # From 16 Oct 2026 the collector writes "lat,lon" next to each image.
            sidecar = by_rel.get(str(PurePosixPath(stored.relative_path).with_suffix(".location")))
            if sidecar is not None:
                try:
                    lat, lon = (float(v) for v in sidecar.read_text().split(",")[:2])
                    refs.append(_file_ref(sidecar, "text/plain"))
                except ValueError:
                    pass
        if source == "alertcalifornia":
            sidecar = by_rel.get(str(PurePosixPath(stored.relative_path).with_suffix(".location")))
            # The collector writes the image before Selenium finishes resolving
            # its camera position. In follow mode, emitting that incomplete image
            # marks its stable ID as seen and the later sidecar is never observed.
            # Wait for a complete, parseable pair instead.
            if sidecar is None:
                continue
            try:
                values = [v.strip() for v in sidecar.read_text().split(",")]
                lat, lon = float(values[0]), float(values[1])
                direction = float(values[2]) if len(values) > 2 and values[2] else None
            except (ValueError, IndexError):
                continue
            refs.append(_file_ref(sidecar, "text/plain"))
            # Frames collected from 16 Oct 2026 on have a .json sidecar, and
            # their third .location field is a compass bearing. Older frames
            # hold a screen angle (0 = east, counter-clockwise); convert it.
            metadata = by_rel.get(str(PurePosixPath(stored.relative_path).with_suffix(".json")))
            if direction is not None:
                data["direction_degrees"] = direction if metadata is not None else round((90 - direction) % 360, 2)
            if metadata is not None:
                try:
                    meta = json.loads(metadata.read_text())
                    props = meta.get("properties") or {}
                    data.update({"camera_id": meta.get("camera_id"), "frame_source": meta.get("source"),
                                 "reaimed": bool(meta.get("reaimed_since_last_run")),
                                 "tilt_degrees": props.get("tilt_current"), "zoom": props.get("zoom_current"),
                                 "field_of_view_degrees": props.get("fov") or None})
                    data = {k: v for k, v in data.items() if v is not None}
                    refs.append(_file_ref(metadata, "application/json"))
                except ValueError:
                    pass
        yield Observation(_id(source, sensor, time, stored.relative_path), source, time, sensor, data,
                          latitude=lat, longitude=lon, files=tuple(refs), raw=_raw(partition, stored, f"{source}_image.v1"))


def _decompressed_lines(stored: StoredFile) -> Iterator[list[str]]:
    content = stored.read_bytes()
    if stored.name.endswith(".gz"): content = gzip.decompress(content)
    for row in csv.reader(io.StringIO(content.decode("utf-8", errors="replace"))): yield row


def read_pems_stations(partition: Partition, *, local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    for stored in partition.files():
        if not stored.name.endswith((".txt.gz", ".txt")): continue
        for index, row in enumerate(_decompressed_lines(stored)):
            if len(row) < 12: continue
            try:
                time = _timestamp(row[0], partition.day, local_timezone); sensor = row[1]
                occupancy = float(row[10]) if row[10] else None; speed = float(row[11]) if row[11] else None
            except ValueError: continue
            yield Observation(_id("pems-stations", sensor, time), "pems-stations", time, sensor,
                              {"avg_occupancy": occupancy, "avg_speed": speed}, raw=_raw(partition, stored, "pems_station_5min.v1", index))


def read_pems_incidents(partition: Partition, *, local_timezone: str = "America/Los_Angeles", **_: Any) -> Iterator[Observation]:
    for stored in partition.files():
        if not stored.name.endswith((".zip", ".csv", ".txt")): continue
        contents: list[tuple[str, bytes]]
        if stored.name.endswith(".zip"):
            with ZipFile(io.BytesIO(stored.read_bytes())) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/") and "_incident_det_" not in name]
                contents = []
                for name in names:
                    content = archive.read(name)
                    if name.endswith(".gz"): content = gzip.decompress(content)
                    contents.append((name, content))
        else: contents = [(stored.name, stored.read_bytes())]
        for inner, content in contents:
            for index, row in enumerate(csv.reader(io.StringIO(content.decode("utf-8", errors="replace")))):
                if len(row) < 20: continue
                try: lat, lon = float(row[9]), float(row[10])
                except ValueError: continue
                time = _timestamp(row[3], partition.day, local_timezone); sensor = row[0] or f"incident-{index}"
                data = {"description": row[4], "severity": row[18], "duration": row[19]}
                raw = _raw(partition, stored, "pems_incident.v1", index); raw["inner_file"] = inner
                yield Observation(_id("pems-incidents", sensor, time, index), "pems-incidents", time, sensor, data,
                                  latitude=lat, longitude=lon, raw=raw)


def read_gdelt(partition: Partition, **_: Any) -> Iterator[Observation]:
    files = list(partition.files()); text_by_id = {}; meta_by_id = {}
    for item in files:
        parent = PurePosixPath(item.relative_path).parent
        if ".gkg" not in parent.name: continue
        if item.suffix == ".txt": text_by_id[(parent.as_posix(), PurePosixPath(item.name).stem)] = item.read_text()
        elif item.suffix == ".json" and item.name != "manifest.json":
            try: meta_by_id[(parent.as_posix(), PurePosixPath(item.name).stem)] = json.loads(item.read_text())
            except json.JSONDecodeError: pass
    for stored in files:
        if stored.suffix != ".csv": continue
        kind = "gkg" if ".gkg" in stored.name else "events"
        for index, row in enumerate(_csv_rows(stored)):
            if not row or row[0] == "0": continue
            time_value = row[1] if len(row) > 1 else row[0]
            time = _timestamp(time_value, partition.day)
            url_index = 4 if kind == "gkg" else 60
            url = row[url_index] if len(row) > url_index else ""
            sensor = hashlib.sha256(url.encode()).hexdigest()[:24] if url else f"{stored.name}:{index}"
            data: dict[str, Any] = {"kind": kind, "url": url, "provider_fields": row}
            if kind == "gkg" and url:
                article_id = hashlib.sha256(url.encode()).hexdigest()[:24]
                parent = PurePosixPath(stored.relative_path).with_suffix("").as_posix()
                meta = meta_by_id.get((parent, article_id), {})
                body = text_by_id.get((parent, article_id))
                if body: data["body"] = body
                for key in ("title", "authors", "publish_date", "status", "http_status"):
                    if key in meta: data[key] = meta[key]
            yield Observation(_id("gdelt", stored.name, index, url), "gdelt", time, sensor, data,
                              raw=_raw(partition, stored, f"gdelt_{kind}.v1", index))


READERS = {
    "air": read_air, "weather": read_weather, "pems-stations": read_pems_stations,
    "pems-incidents": read_pems_incidents, "cctv": lambda p, **k: read_cameras(p, source="cctv", **k),
    "alertcalifornia": lambda p, **k: read_cameras(p, source="alertcalifornia", **k),
    "citizen": lambda p, **k: read_email(p, source="citizen", **k),
    "twitter": lambda p, **k: read_email(p, source="twitter", **k), "gdelt": read_gdelt,
}


def observations(partition: Partition, source: str, **options: Any) -> Iterable[Observation]:
    return READERS[source](partition, **options)
