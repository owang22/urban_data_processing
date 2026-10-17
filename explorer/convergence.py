"""Find places where several AlertCalifornia cameras turned to look at once.

    python -m explorer.convergence [--hours 72] [--server http://127.0.0.1:8095]

Operators re-aim cameras toward smoke and reported fires, and bearings
otherwise barely move. When cameras at two or more different sites turn
within half an hour of each other and their new sight lines cross, the
crossing point is a candidate for something that drew attention there:
a new fire, a fire being watched, or a coincidence.

Each candidate is checked against what the other sources say near that
point and time: fires (CAL FIRE, NIFC) and when they started, NASA heat
detections, and live CHP fire calls. A fire that started after the cameras
turned gives a lead time. No LLM is used; it costs nothing.

Writes processing-output/convergence/candidates-<end>.json and a belief run,
processing-output/beliefs/camera-convergence.jsonl, with three competing
hypotheses per candidate (new fire, known fire, routine) that resolve
against later feed data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen

from .beliefs import BeliefLog, evidence
from .feeds import FeedStore
from .fires import fire_snapshot

REPOSITORY = Path(__file__).parents[1]
MIN_TURN = 5            # degrees; smaller changes are jitter
MAX_FRAME_GAP = 40      # minutes; a turn between frames further apart has no usable time
PAIR_WINDOW = 30        # minutes between two cameras' turns
MIN_CROSSING = 15       # degrees between sight lines, or the crossing point is poorly fixed
MAX_RANGE = 40          # km from each camera
NEAR_KM = 6             # corroborating observations within this distance


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def angle(a: float, b: float) -> float:
    return abs(((b - a) + 540) % 360 - 180)


def km(a, b) -> float:
    dlat, dlon = math.radians(b["lat"] - a["lat"]), math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def site(name: str) -> str:
    """"Castro Peak 2" -> "Castro Peak": cameras on one mast cannot triangulate."""
    return re.sub(r"\s+\d+$", "", name.strip())


def crossing(a: dict, b: dict) -> dict | None:
    """Where two sight lines cross, in front of both cameras, or None."""
    lat0, lon0 = a["lat"], a["lon"]
    kx, ky = 111.32 * math.cos(math.radians(lat0)), 110.57
    bx, by = (b["lon"] - lon0) * kx, (b["lat"] - lat0) * ky
    da = (math.sin(math.radians(a["to"])), math.cos(math.radians(a["to"])))
    db = (math.sin(math.radians(b["to"])), math.cos(math.radians(b["to"])))
    det = da[0] * -db[1] - da[1] * -db[0]
    if abs(det) < 1e-9:
        return None
    s = (bx * -db[1] - by * -db[0]) / det   # distance along a's line
    u = (da[0] * by - da[1] * bx) / det     # distance along b's line
    cross = angle(a["to"], b["to"])
    if not (0.5 < s <= MAX_RANGE and 0.5 < u <= MAX_RANGE and MIN_CROSSING <= cross <= 180 - MIN_CROSSING):
        return None
    # A target can be anywhere across the view, so use a quarter of the field of view as the bearing error.
    error = math.radians(max(min(a.get("fov") or 60, b.get("fov") or 60) / 4, 2))
    return {"lat": lat0 + s * da[1] / ky, "lon": lon0 + s * da[0] / kx, "range_km": [round(s, 1), round(u, 1)],
            "crossing_deg": round(cross), "radius_km": round(max(s, u) * math.tan(error) / math.sin(math.radians(cross)), 1)}


class Detector:
    def __init__(self, server: str, feeds: FeedStore):
        self.server, self.feeds = server, feeds

    def get(self, path: str, **params):
        with urlopen(f"{self.server}{path}?{urlencode(params)}", timeout=180) as response:
            return json.load(response)

    def turns(self, start: datetime, end: datetime) -> list[dict]:
        events = []
        for camera in self.get("/api/snapshot", at=iso(end), hours=(end - start).total_seconds() / 3600)["alertcalifornia"]:
            frames = [f for f in self.get("/api/frames", source="alertcalifornia", name=camera["name"],
                                          start=iso(start), end=iso(end)) if f.get("bearing") is not None]
            for i, (prev, cur) in enumerate(zip(frames, frames[1:]), 1):
                gap = (parse(cur["time"]) - parse(prev["time"])).total_seconds() / 60
                if angle(prev["bearing"], cur["bearing"]) <= MIN_TURN or gap > MAX_FRAME_GAP:
                    continue
                earlier = frames[:i]
                # How often it looked this way before: a preset it often returns to is weak evidence.
                usual = sum(angle(f["bearing"], cur["bearing"]) <= MIN_TURN for f in earlier) / len(earlier)
                events.append({"camera": camera["name"], "site": site(camera["name"]), "lat": camera["lat"],
                               "lon": camera["lon"], "fov": camera.get("fov"), "from": prev["bearing"],
                               "to": cur["bearing"], "after": prev["time"], "time": cur["time"],
                               "frame_before": prev["frame"], "frame": cur["frame"], "usual_share": round(usual, 2)})
        return sorted(events, key=lambda e: e["time"])

    def candidates(self, events: list[dict]) -> list[dict]:
        found = []
        for i, a in enumerate(events):
            for b in events[i + 1:]:
                if (parse(b["time"]) - parse(a["time"])).total_seconds() / 60 > PAIR_WINDOW:
                    break
                if a["site"] == b["site"]:
                    continue
                point = crossing(a, b)
                if point:
                    found.append({**point, "time": min(a["time"], b["time"]), "turns": [a, b]})
        # Merge crossings of the same spot (several camera pairs) into one candidate.
        merged: list[dict] = []
        for c in sorted(found, key=lambda c: c["time"]):
            same = next((m for m in merged if km(m, c) <= max(m["radius_km"], c["radius_km"], 3)
                         and (parse(c["time"]) - parse(m["time"])).total_seconds() <= 3600), None)
            if same:
                names = {t["camera"] for t in same["turns"]}
                same["turns"] += [t for t in c["turns"] if t["camera"] not in names]
                same["radius_km"] = min(same["radius_km"], c["radius_km"])
            else:
                merged.append(c)
        return merged

    def context(self, c: dict) -> dict:
        """What the other sources say near the crossing, from 6 hours before to 12 after."""
        t = parse(c["time"])
        later = fire_snapshot(self.feeds, min(t + timedelta(hours=12), datetime.now(timezone.utc)))
        fires = []
        for f in later["fires"]:
            if f.get("started") and km(c, f) <= NEAR_KM + c["radius_km"]:
                minutes = round((parse(f["started"]) - t).total_seconds() / 60)
                fires.append({"name": f["name"], "km": round(km(c, f), 1), "started_minutes_after": minutes,
                              "acres": f.get("acres"), "source": f["source"], "minor": f.get("minor", False)})
        heat = [h for h in later["hotspots"] if km(c, h) <= NEAR_KM + c["radius_km"]
                and -3 * 3600 <= (parse(h["time"]) - t).total_seconds() <= 12 * 3600]
        chp = []
        for path in sorted((self.feeds.cache / "chp-live").glob("2*.json")):
            stamp = datetime.strptime(path.name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            if not -2 <= (stamp - t).total_seconds() / 3600 <= 6:
                continue
            for call in json.loads(path.read_text()):
                if call["lat"] and "FIRE" in call["type"].upper() and km(c, call) <= NEAR_KM + c["radius_km"]:
                    chp.append(call)
        chp = list({call["id"]: call for call in chp}.values())
        return {"fires": sorted(fires, key=lambda f: abs(f["started_minutes_after"])), "heat": len(heat),
                "first_heat": min((h["time"] for h in heat), default=None),
                "chp_fire_calls": [{"id": x["id"], "time": x["time"], "type": x["type"], "location": x["location"]} for x in chp],
                "chp_running": any((self.feeds.cache / "chp-live").glob(f"{t.strftime('%Y%m%d')}*"))}


def classify(c: dict) -> str:
    fires = [f for f in c["context"]["fires"] if not f["minor"]] or c["context"]["fires"]
    if any(f["started_minutes_after"] < -60 for f in fires):
        return "known fire"   # it was already burning: operators were watching it
    if any(-60 <= f["started_minutes_after"] <= 12 * 60 for f in fires):
        return "new fire"
    return "unexplained"


def record(log: BeliefLog, c: dict) -> None:
    t = parse(c["time"])
    key = f"conv-{t.strftime('%m%d-%H%M')}-{c['lat']:.2f}-{c['lon']:.2f}".replace("-1", "m1")
    cams = ", ".join(sorted({x["camera"] for x in c["turns"]}))
    items = [evidence("supports", "alertcalifornia", weight=0.3 if x["usual_share"] < 0.2 else 0.1, time=x["time"],
                      frame=x["frame"], lat=x["lat"], lon=x["lon"], ref=x["camera"],
                      title=f"{x['camera']} turned {x['from']:.0f}° → {x['to']:.0f}°",
                      note=f"looked this way in {x['usual_share']:.0%} of earlier frames")
             for x in c["turns"]]
    ctx = c["context"]
    where = {"lat": round(c["lat"], 4), "lon": round(c["lon"], 4), "radius_km": max(c["radius_km"], 1),
             "topic": f"Cameras converged {t.astimezone(timezone(timedelta(hours=-7))).strftime('%d %b %H:%M')} PDT"}
    resolves = {"by": iso(t + timedelta(hours=12)),
                "rule": "CAL FIRE or NIFC lists a fire within 6 km that started from 1 hour before to 12 hours after"}
    fire_items = [evidence("supports" if f["started_minutes_after"] >= -60 else "contradicts", f["source"].lower().replace(" ", ""),
                           weight=0.6, title=f"{f['name']}, {f['km']} km away, started {f['started_minutes_after']:+d} min",
                           note="started after the cameras turned" if f["started_minutes_after"] >= -60 else "already burning")
                  for f in ctx["fires"][:3]]
    heat = [evidence("supports", "firms", weight=0.4, time=ctx["first_heat"], title=f"{ctx['heat']} satellite heat detections nearby")] if ctx["heat"] else []
    calls = [evidence("supports", "chp", weight=0.4, time=x["time"], title=f"CHP: {x['type']} at {x['location']}") for x in ctx["chp_fire_calls"][:3]]
    verdict = classify(c)
    p_new = {"new fire": 0.8, "known fire": 0.05, "unexplained": 0.25 if heat or calls else 0.1}[verdict]
    p_known = 0.9 if verdict == "known fire" else 0.05
    ids = [f"{key}-new", f"{key}-known", f"{key}-routine"]
    common = dict(time=t, kind="hypothesis", **where)
    log.record(ids[0], f"A new fire started near this point ({cams})", p_new, evidence=items + fire_items + heat + calls,
               competes=ids[1:], resolves=resolves, status="confirmed" if verdict == "new fire" else "open",
               reasoning=f"{len(c['turns'])} cameras on different sites turned within {PAIR_WINDOW} min; their sight lines "
                         f"cross at {c['crossing_deg']}° about {c['range_km'][0]}–{c['range_km'][1]} km out (±{c['radius_km']} km).",
               **common)
    log.record(ids[1], "Operators were watching a fire that was already burning", p_known, evidence=fire_items,
               competes=[ids[0], ids[2]], status="confirmed" if verdict == "known fire" else "open", **common)
    log.record(ids[2], "The turns were routine or coincidental", round(max(0.0, 1 - p_new - p_known), 2), evidence=[
        evidence("supports", "alertcalifornia", weight=0.3, title=f"{x['camera']} often looks this way",
                 note=f"{x['usual_share']:.0%} of earlier frames") for x in c["turns"] if x["usual_share"] >= 0.2],
        competes=ids[:2], status="open", **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8095")
    parser.add_argument("--hours", type=float, default=72)
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    detector = Detector(args.server, FeedStore(REPOSITORY / "processing-output" / "live-feeds"))
    events = detector.turns(end - timedelta(hours=args.hours), end)
    candidates = detector.candidates(events)
    for c in candidates:
        c["context"] = detector.context(c)
        c["verdict"] = classify(c)
    out = REPOSITORY / "processing-output" / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"candidates-{end.strftime('%Y%m%dT%H%MZ')}.json").write_text(json.dumps(
        {"end": iso(end), "hours": args.hours, "turns": events, "candidates": candidates}, indent=1))
    beliefs = REPOSITORY / "processing-output" / "beliefs" / "camera-convergence.jsonl"
    beliefs.unlink(missing_ok=True)
    log = BeliefLog(beliefs, agent="convergence-detector")
    for c in candidates:
        record(log, c)
    print(f"{len(events)} turns of more than {MIN_TURN}° by {len({e['camera'] for e in events})} cameras; "
          f"{len(candidates)} places where cameras on different sites converged")
    for c in candidates:
        print(f"  {c['time']}  {c['lat']:.3f},{c['lon']:.3f} ±{c['radius_km']} km  {c['verdict']:<11} "
              f"{', '.join(sorted({t['camera'] for t in c['turns']}))}  fires={[(f['name'], f['started_minutes_after']) for f in c['context']['fires'][:2]]} "
              f"heat={c['context']['heat']} chp={len(c['context']['chp_fire_calls'])}")


if __name__ == "__main__":
    main()
