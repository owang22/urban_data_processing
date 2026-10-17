"""Serve the map explorer.

    python -m explorer.server [--port 8095] [--host 127.0.0.1] [--days 3]

Read-only toward the collectors' raw data folder. It also collects public
live feeds (CHP dispatch calls, National Weather Service alerts, USGS
earthquakes, CAL FIRE, NIFC and NASA FIRMS; see feeds.py) into
processing-output/live-feeds, and shows agent belief logs from
processing-output/beliefs (see beliefs.py).
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

from .beliefs import BeliefRuns
from .feeds import FeedStore
from .fires import fire_snapshot
from .index import ExplorerIndex, SOURCES, parse_time

STATIC = Path(__file__).with_name("static")
REPOSITORY = Path(__file__).parents[1]


def make_handler(index: ExplorerIndex, feeds: FeedStore, beliefs: BeliefRuns):
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
                    result = fire_snapshot(feeds, at)
                    news = index.fire_news({f["key"] for f in result["fires"]}, at - timedelta(hours=hours), at)
                    for fire in result["fires"]:
                        fire["news"] = news.get(fire["key"], [])[:25]
                    return self.send_json(result)
                if url.path == "/api/live":
                    at = parse_time(query["at"]) if query.get("at") else datetime.now(timezone.utc)
                    hours = min(max(float(query.get("hours", 3)), 0.25), 72)
                    return self.send_json(feeds.live(at, hours))
                if url.path == "/api/beliefs":
                    if not query.get("run"):
                        return self.send_json({"runs": beliefs.runs()})
                    at = parse_time(query["at"]) if query.get("at") else None
                    try:
                        return self.send_json(beliefs.state(query["run"], at))
                    except KeyError:
                        return self.send_json({"error": "no such run"}, 404)
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


def fetch_feeds_forever(feeds: FeedStore) -> None:
    """Each feed is fetched on its own schedule (feeds.FEEDS); check every 30 s."""
    while True:
        try:
            statuses = feeds.fetch(only_due=True)
            errors = {k: v for k, v in statuses.items() if v.startswith("error")}
            if errors:
                print(f"live feed errors: {errors}", flush=True)
        except Exception as exc:
            print(f"live feed fetch failed: {exc!r}", flush=True)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--days", type=int, default=3, help="most recent days of each source to load")
    parser.add_argument("--refresh-seconds", type=float, default=60)
    parser.add_argument("--data-root", help="defaults to paths.data_root in config.json")
    parser.add_argument("--feed-cache", default=str(REPOSITORY / "processing-output" / "live-feeds"),
                        help="where live feed snapshots are kept (14 days)")
    parser.add_argument("--no-feeds", action="store_true", help="show saved feeds but do not fetch new ones")
    parser.add_argument("--beliefs", default=str(REPOSITORY / "processing-output" / "beliefs"),
                        help="folder of belief logs (*.jsonl)")
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
    feeds = FeedStore(args.feed_cache)
    if not args.no_feeds:
        threading.Thread(target=fetch_feeds_forever, args=(feeds,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, feeds, BeliefRuns(args.beliefs)))
    print(f"explorer on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
