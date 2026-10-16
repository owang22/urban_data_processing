"""Export a self-contained snapshot of the map explorer as one HTML file.

    python3 explorer/export.py --out fire-map.html [--server http://127.0.0.1:8095]

The page needs no server: the basemap (OpenStreetMap tiles stitched into two
images), camera thumbnails and all data are embedded. It is meant for sharing
a moment in time, such as a claude.ai artifact. Needs Pillow and a running
explorer (python -m explorer.server).
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image

HERE = Path(__file__).parent
REPOSITORY = HERE.parent
USER_AGENT = "urban-observation-explorer/1.0"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# (south, west, north, east, zoom): the whole area, then the Santa Monica Mountains in detail.
BASEMAPS = [(33.25, -119.25, 34.85, -117.3, 10), (33.98, -119.0, 34.22, -118.55, 12)]
THUMB_WIDTH = 320
SEQUENCE_STEP = timedelta(minutes=30)
SEQUENCE_CAMERAS = 3  # nearest cameras facing each active fire get a frame sequence


def get_json(server: str, path: str, **params) -> object:
    with urlopen(f"{server}{path}?{urlencode(params)}", timeout=120) as response:
        return json.load(response)


def tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_lat(y: float, zoom: int) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** zoom))))


def basemap(south, west, north, east, zoom, cache: Path) -> dict:
    x0, y0 = (int(v) for v in tile_xy(north, west, zoom))
    x1, y1 = (int(v) for v in tile_xy(south, east, zoom))
    canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            path = cache / f"{zoom}-{x}-{y}.png"
            if not path.exists():
                url = TILE_URL.format(z=zoom, x=x, y=y)
                with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
                    path.write_bytes(response.read())
            canvas.paste(Image.open(path).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
    bounds = [[tile_lat(y1 + 1, zoom), x0 / 2 ** zoom * 360 - 180],
              [tile_lat(y0, zoom), (x1 + 1) / 2 ** zoom * 360 - 180]]
    return {"image": jpeg(canvas, 72), "bounds": bounds, "zoom": zoom}


def jpeg(image: Image.Image, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def thumbnail(data_root: Path, relative: str, width: int = THUMB_WIDTH) -> str | None:
    try:
        image = Image.open(data_root / relative).convert("RGB")
    except OSError:
        return None
    height = round(image.height * width / image.width)
    return jpeg(image.resize((width, height), Image.LANCZOS), 62)


def distance_km(a, b) -> float:
    dlat, dlon = math.radians(b["lat"] - a["lat"]), math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def bearing_to(a, b) -> float:
    la, lb, dl = math.radians(a["lat"]), math.radians(b["lat"]), math.radians(b["lon"] - a["lon"])
    y = math.sin(dl) * math.cos(lb)
    x = math.cos(la) * math.sin(lb) - math.sin(la) * math.cos(lb) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def facing(camera, fire) -> bool:
    if camera.get("bearing") is None or distance_km(camera, fire) > 40:
        return False
    off = abs((bearing_to(camera, fire) - camera["bearing"] + 540) % 360 - 180)
    return off <= (camera.get("fov") or 60) / 2 + 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8095")
    parser.add_argument("--hours", type=float, default=24, help="news and incident history to include")
    parser.add_argument("--data-root", help="defaults to paths.data_root in config.json")
    args = parser.parse_args()

    data_root = Path(args.data_root or json.loads((REPOSITORY / "config.json").read_text())["paths"]["data_root"])
    cache = REPOSITORY / "processing-output" / "explorer-tiles"
    cache.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    snap = get_json(args.server, "/api/snapshot", hours=args.hours)
    fires = get_json(args.server, "/api/fires", hours=args.hours)
    ranges = get_json(args.server, "/api/range")["sources"]
    # CHP incidents arrive a day late; take the last day that has them.
    chp_end = ranges.get("pems-incidents", {}).get("end")
    incidents = get_json(args.server, "/api/snapshot", at=chp_end, hours=24)["incidents"] if chp_end else []

    cameras = []
    for source in ("alertcalifornia", "cctv"):
        for camera in snap[source]:
            if now - datetime.fromisoformat(camera["time"].replace("Z", "+00:00")) > timedelta(hours=3):
                continue  # stale; the snapshot is about now
            thumb = thumbnail(data_root, camera["frame"], THUMB_WIDTH if source == "alertcalifornia" else 240)
            if thumb:
                cameras.append({**{k: camera.get(k) for k in ("name", "lat", "lon", "time", "bearing", "fov", "tilt",
                                                               "zoom", "reaimed", "bearings")},
                                "source": source, "thumb": thumb})

    # A frame every 30 minutes since each active fire started, from the nearest cameras facing it.
    sequences = {}
    for fire in fires["fires"]:
        if fire.get("minor") or (fire.get("contained") or 0) >= 100 or not fire.get("started"):
            continue
        facing_cams = sorted((c for c in snap["alertcalifornia"] if facing(c, fire)), key=lambda c: distance_km(c, fire))
        start = max(datetime.fromisoformat(fire["started"].replace("Z", "+00:00")), now - timedelta(hours=36))
        for camera in facing_cams[:SEQUENCE_CAMERAS]:
            if camera["name"] in sequences:
                continue
            frames = get_json(args.server, "/api/frames", source="alertcalifornia", name=camera["name"],
                              start=start.isoformat().replace("+00:00", "Z"), end=now.isoformat().replace("+00:00", "Z"))
            picked, last = [], None
            for frame in frames:
                t = datetime.fromisoformat(frame["time"].replace("Z", "+00:00"))
                if last is None or t - last >= SEQUENCE_STEP:
                    thumb = thumbnail(data_root, frame["frame"])
                    if thumb:
                        picked.append({"time": frame["time"], "bearing": frame.get("bearing"), "thumb": thumb})
                        last = t
            if picked:
                sequences[camera["name"]] = picked

    payload = {
        "generated": now.isoformat().replace("+00:00", "Z"),
        "hours": args.hours,
        "basemaps": [basemap(*b, cache) for b in BASEMAPS],
        "cameras": cameras,
        "sequences": sequences,
        "fires": fires,
        "news": snap["news"],
        "incidents": incidents,
        "weather": snap["weather"],
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    template = (HERE / "export_template.html").read_text()
    leaflet_css = (HERE / "static" / "vendor" / "leaflet.css").read_text()
    html = template.replace("/*LEAFLET_CSS*/", leaflet_css).replace("{{DATA}}", data)
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.1f} MB, {len(cameras)} cameras, "
          f"{sum(len(v) for v in sequences.values())} sequence frames from {len(sequences)} cameras, "
          f"{len(fires['fires'])} fires, {len(snap['news'])} news, {len(incidents)} CHP incidents")


if __name__ == "__main__":
    main()
