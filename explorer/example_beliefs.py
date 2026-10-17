"""Write an example belief run from the collected data, to try the belief viewer.

    python -m explorer.example_beliefs [--hours 24] [--server http://127.0.0.1:8095]

This is not an LLM agent: fixed rules turn the explorer's data into three
beliefs per active wildfire, revised every hour, so the viewer has a run
with genuine evidence to show. It needs the explorer running and costs
nothing. The run is written to processing-output/beliefs/example-rules.jsonl
(replaced each time) with agent "example-rules".

The rules, per fire:
  <fire> is burning      official listing, satellite heat, news, cameras on it
  <fire> grew in 6 h     official acreage then and now, news figures, containment
  fire weather at <fire> NWS warnings covering the spot, wind at the nearest station
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from .beliefs import BeliefLog, evidence

REPOSITORY = Path(__file__).parents[1]


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def distance_km(a, b) -> float:
    dlat, dlon = math.radians(b["lat"] - a["lat"]), math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def bearing_to(a, b) -> float:
    la, lb, dl = math.radians(a["lat"]), math.radians(b["lat"]), math.radians(b["lon"] - a["lon"])
    y = math.sin(dl) * math.cos(lb)
    x = math.cos(la) * math.sin(lb) - math.sin(la) * math.cos(lb) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def facing(camera, place) -> bool:
    if camera.get("bearing") is None or distance_km(camera, place) > 40:
        return False
    off = abs((bearing_to(camera, place) - camera["bearing"] + 540) % 360 - 180)
    return off <= (camera.get("fov") or 60) / 2 + 3


def inside(lon: float, lat: float, geometry: dict) -> bool:
    """Point in a Polygon or MultiPolygon (outer rings only)."""
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry.get("coordinates", [])
    for polygon in polygons:
        ring, hit = polygon[0], False
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                hit = not hit
        if hit:
            return True
    return False


def confidence(items: list[dict], prior: float = 0.0) -> float:
    """This example's rule: each item moves the log-odds by 3 x weight, for or against."""
    sign = {"supports": 1, "contradicts": -1, "context": 0}
    score = prior + sum(3 * sign[i["stance"]] * i["weight"] for i in items)
    return round(1 / (1 + math.exp(-score)), 2)


class Example:
    def __init__(self, server: str, log: BeliefLog):
        self.server, self.log = server, log
        self.last: dict[str, tuple] = {}

    def get(self, path: str, **params):
        with urlopen(f"{self.server}{path}?{urlencode(params)}", timeout=120) as response:
            return json.load(response)

    def record(self, at: datetime, belief_id: str, statement: str, items: list[dict], prior: float = 0.0, **fields):
        value = confidence(items, prior)
        status = fields.pop("status", "open")
        signature = (value, status, tuple(sorted((i["source"], i["stance"], i.get("title")) for i in items)))
        if self.last.get(belief_id) == signature:
            return  # nothing changed this hour
        self.last[belief_id] = signature
        self.log.record(belief_id, statement, value, time=at, evidence=items, status=status, **fields)

    def step(self, at: datetime) -> None:
        fires = self.get("/api/fires", at=iso(at), hours=24)
        before = {f["key"]: f for f in self.get("/api/fires", at=iso(at - timedelta(hours=6)), hours=24)["fires"]}
        snapshot = self.get("/api/snapshot", at=iso(at), hours=3)
        live = self.get("/api/live", at=iso(at), hours=3)
        for fire in fires["fires"]:
            if fire.get("minor") or not (fire.get("news") or (fire.get("acres") or 0) >= 50):
                continue
            self.fire(at, fire, before.get(fire["key"]), fires["hotspots"], snapshot, live)

    def fire(self, at, fire, before, hotspots, snapshot, live) -> None:
        key, name = fire["key"], fire["name"]
        where = {"place": fire.get("location") or fire.get("county"), "lat": fire["lat"], "lon": fire["lon"],
                 "topic": name}
        out = (fire.get("contained") or 0) >= 100

        # 1. It is burning.
        items = [evidence("contradicts" if out else "supports", fire["source"].lower().replace(" ", ""),
                          weight=0.6, time=fire.get("updated"), url=fire.get("url"), lat=fire["lat"], lon=fire["lon"],
                          title=f"{fire['source']} lists it: {fire.get('acres') or 0:g} acres, {fire.get('contained') or 0:g}% contained",
                          note="contained, so no longer spreading" if out else "official incident list")]
        near = sorted((h for h in hotspots if distance_km(h, fire) < 5), key=lambda h: h["time"], reverse=True)
        if near:
            items.append(evidence("supports", "firms", weight=0.4, time=near[0]["time"], lat=near[0]["lat"],
                                  lon=near[0]["lon"], title=f"{len(near)} satellite heat detections within 5 km",
                                  note=f"newest from {near[0]['satellite']}"))
        if fire.get("news"):
            # Prefer an article whose headline names this fire over one that mentions it in passing.
            article = next((n for n in fire["news"] if key in n["title"].lower()), fire["news"][0])
            items.append(evidence("supports", "gdelt", weight=0.3, time=article["time"], url=article["url"],
                                  title=article["title"], note=f"{len(fire['news'])} articles name it in 24 h"))
        cameras = sorted((c for c in snapshot["alertcalifornia"] if facing(c, fire)), key=lambda c: distance_km(c, fire))
        for camera in cameras[:2]:
            items.append(evidence("supports" if camera.get("reaimed") else "context", "alertcalifornia",
                                  weight=0.3 if camera.get("reaimed") else 0.1, time=camera["time"], frame=camera["frame"],
                                  lat=camera["lat"], lon=camera["lon"], ref=camera["name"],
                                  title=f"{camera['name']} faces it, {distance_km(camera, fire):.1f} km away",
                                  note="operators re-aimed it in the last 3 h" if camera.get("reaimed") else "view covers the fire"))
        self.record(at, f"{key}-burning", f"The {name} is burning", items, kind="observation",
                    status="refuted" if out else "open", radius_km=2, **where,
                    reasoning="Official listing and containment set the baseline; heat detections, articles and "
                              "re-aimed cameras add support.")

        # 2. It grew in the past 6 hours.
        items = []
        if before and before.get("acres") is not None and fire.get("acres") is not None:
            grew = fire["acres"] > before["acres"]
            items.append(evidence("supports" if grew else "contradicts", "calfire", weight=0.6 if grew else 0.4,
                                  url=fire.get("url"), time=fire.get("updated"),
                                  title=f"{before['acres']:g} acres 6 h ago, {fire['acres']:g} now",
                                  note=None if grew else "the official figure is updated only a few times a day"))
            if (fire.get("contained") or 0) > (before.get("contained") or 0):
                items.append(evidence("contradicts", "calfire", weight=0.3, time=fire.get("updated"),
                                      title=f"containment rose from {before.get('contained') or 0}% to {fire['contained']}%"))
        elif not before:
            items.append(evidence("supports", fire["source"].lower().replace(" ", ""), weight=0.5,
                                  title="not listed 6 h ago", time=fire.get("started")))
        reported = [n for n in fire.get("news", []) if n.get("acres")]
        if reported and fire.get("acres") is not None and reported[0]["acres"] > fire["acres"]:
            items.append(evidence("supports", "gdelt", weight=0.3, time=reported[0]["time"], url=reported[0]["url"],
                                  title=reported[0]["title"],
                                  note=f"reports {reported[0]['acres']:g} acres, ahead of the official {fire['acres']:g}"))
        if items:
            self.record(at, f"{key}-growing", f"The {name} grew in the past 6 hours", items, kind="hypothesis",
                        parents=[f"{key}-burning"], status="refuted" if out else "open", radius_km=4, **where,
                        reasoning="Compares the official acreage with 6 hours earlier; news figures larger than "
                                  "the official one suggest the official count lags.")

        # 3. Fire weather is pushing it.
        if "nws-alerts" not in live["fetched"] or out:
            return  # alerts were not being collected yet, or the fire is contained
        items = []
        for alert in live["alerts"]:
            if alert["kind"] not in ("fire", "wind"):
                continue
            if any(inside(fire["lon"], fire["lat"], live["zones"][z]["geometry"]) for z in alert["zones"] if z in live["zones"]):
                items.append(evidence("supports", "nws", weight=0.5 if alert["kind"] == "fire" else 0.3,
                                      time=alert.get("sent"), title=alert["headline"] or alert["event"],
                                      note=f"in effect until {alert.get('ends')}"))
        stations = sorted(snapshot["weather"], key=lambda w: distance_km(w, fire))
        if stations and distance_km(stations[0], fire) < 30 and stations[0].get("wind_speed_mps") is not None:
            w = stations[0]
            windy = w["wind_speed_mps"] >= 8
            items.append(evidence("supports" if windy else "contradicts", "weather", weight=0.3, time=w["time"],
                                  lat=w["lat"], lon=w["lon"], title=f"{w['station']}: wind {w['wind_speed_mps']} m/s",
                                  note=f"{distance_km(w, fire):.0f} km away; 8 m/s or more counts as strong"))
        if not items:
            return  # no alert covers it and no station is near enough to say
        self.record(at, f"{key}-fire-weather", f"Fire weather is helping the {name} spread", items, prior=-0.5,
                    kind="hypothesis", parents=[f"{key}-growing"], status="refuted" if out else "open",
                    radius_km=8, **where,
                    reasoning="Red Flag and wind warnings covering the fire, and wind at the nearest station. "
                              "Starts slightly doubtful without either.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8095")
    parser.add_argument("--hours", type=float, default=24, help="how far back to start")
    parser.add_argument("--out", default=str(REPOSITORY / "processing-output" / "beliefs" / "example-rules.jsonl"))
    args = parser.parse_args()

    out = Path(args.out)
    out.unlink(missing_ok=True)
    example = Example(args.server, BeliefLog(out, agent="example-rules"))
    now = datetime.now(timezone.utc)
    at = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=args.hours)
    while at < now:
        example.step(at)
        at += timedelta(hours=1)
    example.step(now)
    print(f"{out}: {sum(1 for _ in out.open())} revisions" if out.exists() else "no active fires; nothing written")


if __name__ == "__main__":
    main()
