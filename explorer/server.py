"""Serve the map explorer.

    python -m explorer.server [--port 8095] [--host 127.0.0.1] [--days 3]

Read-only toward the collectors' raw data folder. It fetches public wildfire
feeds (CAL FIRE, NIFC, NASA FIRMS) every 10 minutes and keeps snapshots in
processing-output/fire-feeds.
Binds to localhost by default; reach it from another machine with
    ssh -L 8095:localhost:8095 <user>@<this machine>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from .fires import FireFeeds
from .index import ExplorerIndex, SOURCES, parse_time

STATIC = Path(__file__).with_name("static")
REPOSITORY = Path(__file__).parents[1]


def make_handler(index: ExplorerIndex, fires: FireFeeds):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC), **kwargs)

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def send_json(self, value, status=200):
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/range":
                    return self.send_json(index.time_range())
                if url.path == "/api/snapshot":
                    at = parse_time(query["at"]) if query.get("at") else datetime.now(timezone.utc)
                    hours = min(max(float(query.get("hours", 3)), 0.25), 72)
                    return self.send_json(index.snapshot(at, hours))
                if url.path == "/api/fires":
                    at = parse_time(query["at"]) if query.get("at") else datetime.now(timezone.utc)
                    hours = min(max(float(query.get("hours", 24)), 1), 72)
                    result = fires.snapshot(at)
                    news = index.fire_news({f["key"] for f in result["fires"]}, at - timedelta(hours=hours), at)
                    for fire in result["fires"]:
                        fire["news"] = news.get(fire["key"], [])[:25]
                    return self.send_json(result)
                if url.path == "/api/frames":
                    if query.get("source") not in ("alertcalifornia", "cctv"):
                        return self.send_json({"error": "unknown source"}, 400)
                    end = parse_time(query["end"]) if query.get("end") else datetime.now(timezone.utc)
                    start = parse_time(query["start"]) if query.get("start") else end - timedelta(hours=3)
                    return self.send_json(index.frames(query["source"], query.get("name", ""), start, end))
                if url.path == "/frame":
                    path = index.frame_path(query.get("p", ""))
                    if path is None:
                        return self.send_error(404)
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png" if path.suffix == ".png" else "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "max-age=86400")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except (KeyError, ValueError) as exc:
                return self.send_json({"error": str(exc)}, 400)
            if url.path.startswith("/api/"):
                return self.send_error(404)
            return super().do_GET()

    return Handler


def refresh_forever(index: ExplorerIndex, interval: float) -> None:
    while True:
        time.sleep(interval)
        started = time.time()
        try:
            index.refresh()
            print(f"index refreshed in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:  # keep serving the last good index
            print(f"index refresh failed: {exc!r}", flush=True)


def fetch_fires_forever(fires: FireFeeds, interval: float) -> None:
    while True:
        try:
            statuses = fires.fetch()
            errors = {k: v for k, v in statuses.items() if v.startswith("error")}
            print(f"fire feeds: {sum(v == 'saved' for v in statuses.values())} updated"
                  + (f", errors: {errors}" if errors else ""), flush=True)
        except Exception as exc:
            print(f"fire feed fetch failed: {exc!r}", flush=True)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--days", type=int, default=3, help="most recent days of each source to load")
    parser.add_argument("--refresh-seconds", type=float, default=60)
    parser.add_argument("--data-root", help="defaults to paths.data_root in config.json")
    parser.add_argument("--fire-cache", default=str(REPOSITORY / "processing-output" / "fire-feeds"),
                        help="where fetched fire feed snapshots are kept (14 days)")
    parser.add_argument("--fire-seconds", type=float, default=600, help="how often to fetch the fire feeds")
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root:
        for name in ("config.json", "config.example.json"):
            try:
                data_root = json.loads((REPOSITORY / name).read_text())["paths"]["data_root"]
                break
            except (OSError, KeyError, ValueError):
                continue
    index = ExplorerIndex(data_root, REPOSITORY / "observation_pipeline" / "assets", days=args.days)
    print(f"loading {', '.join(SOURCES)} from {data_root} ...", flush=True)
    index.refresh()
    threading.Thread(target=refresh_forever, args=(index, args.refresh_seconds), daemon=True).start()
    fires = FireFeeds(args.fire_cache)
    threading.Thread(target=fetch_fires_forever, args=(fires, args.fire_seconds), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, fires))
    print(f"explorer on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
