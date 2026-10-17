"""Wildfires for the map explorer, from the saved feed snapshots (see feeds.py).

CAL FIRE's list is preferred; NIFC adds fires CAL FIRE does not run, mapped
perimeters come from NIFC and heat detections from NASA FIRMS. news_fires
reads fire names, acreage and containment out of news articles.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
import json
import re
from typing import Any

from .feeds import FeedStore, in_region, iso as _iso, parse


def _ms(value: Any) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def fire_key(name: str) -> str:
    """"Bryson Fire " / "BRYSON" -> "bryson"; used to match feeds and news."""
    return re.sub(r"\s+fire\s*$", "", name.strip(), flags=re.I).strip().lower()


def fire_snapshot(store: FeedStore, at: datetime) -> dict[str, Any]:
    """Fires, perimeters and heat detections as the feeds stood at `at`."""
    fires: dict[str, dict[str, Any]] = {}
    fetched = {}

    found = store.latest("calfire", at)
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

    found = store.latest("incidents", at)
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
    found = store.latest("perimeters", at)
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
        found = store.latest(feed, at)
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
