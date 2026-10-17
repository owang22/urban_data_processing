"""Build a published copy of the situation map: the same page, without a server.

    /usr/bin/python3 -m explorer.publish [--out processing-output/published/site]

Writes index.html (the page, with its styles and script inlined), data.json
(everything the page asks the server for, as of now, with camera thumbnails
embedded) and two basemap images stitched from OpenStreetMap tiles, since a
hosted page such as a claude.ai artifact cannot load tiles or reach this
machine. Rebuild and republish data.json and version.json to update it; the page
checks version.json every 5 minutes and fetches data.json only when it
changed. Needs the explorer running and Pillow (the system
Python has it; the processing venv does not).
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

from .example_beliefs import distance_km, facing

HERE = Path(__file__).parent
REPOSITORY = HERE.parent
USER_AGENT = "urban-observation-explorer/1.0"
# USGS National Map topo tiles: public domain, so saving a copy is allowed
# (OpenStreetMap's tile servers forbid bulk downloads).
TILE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
ATTRIBUTION = "Basemap: USGS The National Map"
# (south, west, north, east, zoom): the whole area, then the Santa Monica Mountains in detail.
# (south, west, north, east, zoom, map zoom from which it shows): an overview,
# the whole area in more detail, and the populated core (Malibu to Pomona) at street level.
BASEMAPS = [(33.25, -119.25, 34.85, -117.3, 10, 0), (33.25, -119.25, 34.85, -117.3, 12, 11),
            (33.70, -118.95, 34.35, -117.70, 14, 13)]
CHUNK_TILES = 8  # 2048-pixel images, loaded only when in view
WINDOW_HOURS = 3
SEQUENCE_STEP = timedelta(minutes=30)
SEQUENCE_HOURS = 24
SEQUENCE_CAMERAS = 3  # nearest cameras facing each active fire get a frame sequence
HISTORY_WIDTH = 240


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Builder:
    def __init__(self, server: str, data_root: Path, cache: Path):
        self.server, self.data_root, self.cache = server, data_root, cache
        self.frames: dict[str, str] = {}

    def get(self, path: str, **params):
        with urlopen(f"{self.server}{path}?{urlencode(params)}", timeout=180) as response:
            return json.load(response)

    def thumbnail(self, relative: str, width: int = 320) -> bool:
        """Embed a frame, resized; cached on disk so republishing is quick."""
        if relative in self.frames:
            return True
        cached = self.cache / "thumbs" / f"{hashlib.sha1(f'{relative}:{width}'.encode()).hexdigest()}.jpg"
        if not cached.exists():
            try:
                image = Image.open(self.data_root / relative).convert("RGB")
            except OSError:
                return False
            height = round(image.height * width / image.width)
            cached.parent.mkdir(parents=True, exist_ok=True)
            image.resize((width, height), Image.LANCZOS).save(cached, "JPEG", quality=62, optimize=True)
        self.frames[relative] = "data:image/jpeg;base64," + base64.b64encode(cached.read_bytes()).decode()
        return True

    def build(self, now: datetime) -> dict:
        at = iso(now)
        data = {
            "generated": at,
            "range": self.get("/api/range"),
            "snapshot": self.get("/api/snapshot", at=at, hours=WINDOW_HOURS),
            "fires": self.get("/api/fires", at=at, hours=24),
            "live": self.get("/api/live", at=at, hours=WINDOW_HOURS),
        }
        runs = self.get("/api/beliefs")["runs"]
        data["beliefs"] = {"runs": runs, "states": {r["run"]: self.get("/api/beliefs", run=r["run"], at=at) for r in runs}}

        snapshot = data["snapshot"]
        for camera in snapshot["alertcalifornia"]:
            self.thumbnail(camera["frame"])
        for camera in snapshot["cctv"]:
            self.thumbnail(camera["frame"], 240)
        for state in data["beliefs"]["states"].values():
            for belief in state["beliefs"]:
                for item in belief["evidence"]:
                    if item.get("frame"):
                        self.thumbnail(item["frame"])

        # A frame every 30 minutes since each active fire started, from the nearest cameras facing it.
        data["sequences"] = {}
        for fire in data["fires"]["fires"]:
            if fire.get("minor") or (fire.get("contained") or 0) >= 100 or not fire.get("started"):
                continue
            cameras = sorted((c for c in snapshot["alertcalifornia"] if facing(c, fire)), key=lambda c: distance_km(c, fire))
            start = max(parse(fire["started"]), now - timedelta(hours=SEQUENCE_HOURS))
            for camera in cameras[:SEQUENCE_CAMERAS]:
                key = f"alertcalifornia:{camera['name']}"
                if key in data["sequences"]:
                    continue
                picked, last = [], None
                for frame in self.get("/api/frames", source="alertcalifornia", name=camera["name"], start=iso(start), end=at):
                    t = parse(frame["time"])
                    if (last is None or t - last >= SEQUENCE_STEP) and self.thumbnail(frame["frame"]):
                        picked.append(frame)
                        last = t
                if picked:
                    data["sequences"][key] = picked

        # Every camera's last 3 hours: a frame every 30 minutes, and both sides
        # of each re-aim, so each direction it pointed has a view.
        for camera in snapshot["alertcalifornia"]:
            frames = self.get("/api/frames", source="alertcalifornia", name=camera["name"], start=snapshot["start"], end=at)
            keep = set()
            last = None
            for i, frame in enumerate(frames):
                t = parse(frame["time"])
                turned = i and not same_bearing(frames[i - 1].get("bearing"), frame.get("bearing"))
                if turned:
                    keep.update((i - 1, i))
                if last is None or t - last >= SEQUENCE_STEP or turned:
                    keep.add(i)
                    last = t
            if frames:
                keep.add(len(frames) - 1)
            picked = [frames[i] for i in sorted(keep) if self.thumbnail(frames[i]["frame"], HISTORY_WIDTH)]
            key = f"alertcalifornia:{camera['name']}"
            merged = {f["frame"]: f for f in data["sequences"].get(key, []) + picked}
            data["sequences"][key] = sorted(merged.values(), key=lambda f: f["time"])
        data["frames"] = self.frames
        return data


def same_bearing(a, b) -> bool:
    if a is None or b is None:
        return True
    return abs(((b - a) + 540) % 360 - 180) <= 2


def tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile_lat(y: float, zoom: int) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** zoom))))


def basemap(south, west, north, east, zoom, show_from, tiles: Path, out: Path) -> list[dict]:
    """Stitch tiles into chunk images (tiles are fetched once and kept)."""
    x0, y0 = (int(v) for v in tile_xy(north, west, zoom))
    x1, y1 = (int(v) for v in tile_xy(south, east, zoom))
    chunks = []
    for cx in range(x0, x1 + 1, CHUNK_TILES):
        for cy in range(y0, y1 + 1, CHUNK_TILES):
            xe, ye = min(cx + CHUNK_TILES - 1, x1), min(cy + CHUNK_TILES - 1, y1)
            name = f"basemap/{zoom}-{cx}-{cy}.jpg"
            if not (out / name).exists():
                (out / "basemap").mkdir(parents=True, exist_ok=True)
                tiles.mkdir(parents=True, exist_ok=True)
                canvas = Image.new("RGB", ((xe - cx + 1) * 256, (ye - cy + 1) * 256), "white")
                for x in range(cx, xe + 1):
                    for y in range(cy, ye + 1):
                        path = tiles / f"{zoom}-{x}-{y}.jpg"
                        if not path.exists():
                            with urlopen(Request(TILE_URL.format(z=zoom, x=x, y=y), headers={"User-Agent": USER_AGENT}),
                                         timeout=30) as response:
                                path.write_bytes(response.read())
                            time.sleep(0.05)  # be gentle with the public server
                        canvas.paste(Image.open(path).convert("RGB"), ((x - cx) * 256, (y - cy) * 256))
                canvas.save(out / name, "JPEG", quality=72, optimize=True, progressive=True)
            bounds = [[tile_lat(ye + 1, zoom), cx / 2 ** zoom * 360 - 180],
                      [tile_lat(cy, zoom), (xe + 1) / 2 ** zoom * 360 - 180]]
            chunks.append({"src": name, "bounds": bounds, "zoom": zoom, "from": show_from})
    return chunks


def page(basemaps: list[dict]) -> str:
    """The localhost page with its stylesheets and script inlined."""
    static = HERE / "static"
    html = (static / "index.html").read_text()
    title = re.search(r"<title>.*?</title>", html).group(0)
    body = html[html.index("<body>") + 6:html.index("<script")]
    styles = "\n".join((static / name).read_text() for name in (
        "vendor/leaflet.css", "vendor/MarkerCluster.min.css", "vendor/MarkerCluster.Default.min.css", "style.css"))
    scripts = "\n".join(re.findall(r'<script src="https://cdnjs[^"]+"></script>', html))
    settings = json.dumps({"data": "data.json", "version": "version.json", "basemaps": basemaps, "attribution": ATTRIBUTION})
    app = (static / "app.js").read_text()
    assert "</script" not in app
    return (f"{title}\n<style>\n{styles}\n</style>\n{body}{scripts}\n"
            f"<script>window.SITUATION_STATIC = {settings};</script>\n<script>\n{app}\n</script>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPOSITORY / "processing-output" / "published" / "site"))
    parser.add_argument("--server", default="http://127.0.0.1:8095")
    parser.add_argument("--data-root", help="defaults to paths.data_root in config.json")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out.parent
    data_root = Path(args.data_root or json.loads((REPOSITORY / "config.json").read_text())["paths"]["data_root"])

    basemaps = [chunk for b in BASEMAPS for chunk in basemap(*b, cache / "usgs-tiles", out)]
    (out / "index.html").write_text(page(basemaps))
    data = Builder(args.server, data_root, cache).build(datetime.now(timezone.utc))
    text = json.dumps(data, separators=(",", ":"))
    temporary = out / "data.json.part"
    temporary.write_text(text)
    temporary.replace(out / "data.json")
    (out / "version.json").write_text(json.dumps({"generated": data["generated"]}))
    print(f"{out}: data.json {len(text) / 1e6:.1f} MB, {len(data['frames'])} frames, "
          f"{sum(len(v) for v in data['sequences'].values())} sequence frames, {len(data['fires']['fires'])} fires, "
          f"{len(data['live']['chp'] or [])} CHP calls, {len(data['live']['alerts'])} alerts, "
          f"{len(data['snapshot']['news'])} news")


if __name__ == "__main__":
    main()
