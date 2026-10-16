"""Command-line interface for read-only Urban Observations replay."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import sys

from observation_pipeline.config import get_config
from .catalog import DataCatalog
from .engine import JSONLSink, follow, historical
from .protocol import (
    DEFAULT_RECEIVER_PORT,
    DEFAULT_RECEIVER_RETRIES,
    DEFAULT_RECEIVER_TIMEOUT,
    ObservationRejected,
    SocketJSONLSink,
    receiver_config,
)
from .readers import READERS


def _day(value: str) -> date:
    if value == "today":
        return date.today()
    try: return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc: raise argparse.ArgumentTypeError("expected YYYY-MM-DD or 'today'") from exc


def _near(value: str) -> tuple[float, float, float]:
    try:
        latitude, longitude, radius = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LAT,LON,KM") from exc
    return latitude, longitude, radius


def _instant(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO 8601 time with a UTC offset") from exc
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError("expected an ISO 8601 time with a UTC offset")
    return result


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _selected(item, args) -> bool:
    if args.near is not None:
        if item.latitude is None or item.longitude is None: return False
        if _distance_km(args.near[0], args.near[1], item.latitude, item.longitude) > args.near[2]: return False
    if args.between is not None:
        start, end = args.between
        if not start <= _instant(item.time) < end: return False
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Convert and replay already-collected observations as JSONL")
    result.add_argument("--config", help="configuration file (default: ./config.json)")
    result.add_argument("--from", dest="start",
                        help="inclusive YYYY-MM-DD, 'today', or 'now' with --follow")
    result.add_argument("--to", type=_day, help="exclusive YYYY-MM-DD; omit only with --follow")
    result.add_argument("--follow", action="store_true", help="continue monitoring collected files after catch-up")
    result.add_argument("--source", action="append", choices=sorted(READERS), help="repeat to select sources; default: all")
    result.add_argument("--data-root", help="override config save_folder")
    result.add_argument("--backup-root", help="override config backup_folder")
    result.add_argument("--weather-locations", help="OpenWeather coordinate inventory")
    result.add_argument("--cctv-locations", help="Caltrans CCTV KML inventory")
    result.add_argument("--poll-seconds", type=float, default=5.0)
    result.add_argument("--timezone", default="America/Los_Angeles", help="timezone for naive collector/PeMS timestamps")
    result.add_argument("--receiver-host", "--socket-host", dest="receiver_host",
                        help="send observations to this receiver instead of the configured host")
    result.add_argument("--no-receiver", "--no-socket", dest="no_receiver", action="store_true",
                        help="write JSONL to stdout instead of the configured receiver")
    result.add_argument("--receiver-port", "--socket-port", dest="receiver_port", type=int)
    result.add_argument("--receiver-timeout", "--ack-timeout", dest="receiver_timeout", type=float)
    result.add_argument("--receiver-retries", "--network-retries", dest="receiver_retries", type=int)
    result.add_argument("--skip-rejected", action="store_true",
                        help="log observations the receiver rejects to stderr and continue")
    result.add_argument("--limit", type=int, help="stop after sending this many historical observations")
    result.add_argument("--near", type=_near, metavar="LAT,LON,KM",
                        help="only observations with coordinates within KM of this point")
    result.add_argument("--between", type=_instant, nargs=2, metavar=("START", "END"),
                        help="only observations at or after START and before END (ISO 8601 with offset)")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.start: parser().error("--from is required")
    if not args.follow and args.to is None: parser().error("--to is required unless --follow is used")
    if args.start == "now" and not args.follow: parser().error("--from now requires --follow")
    config = get_config(args.config); paths = config.get("paths", {})
    legacy_replay = config.get("real_replay", {})
    endpoint = receiver_config(config, {
        key: value for key, value in {
            "host": legacy_replay.get("receiver_host"),
            "port": legacy_replay.get("receiver_port"),
        }.items() if value is not None
    })
    data_root = args.data_root or paths["data_root"]
    backup_root = args.backup_root or paths["backup_root"]
    repository = Path(__file__).parents[1]
    assets = repository / "observation_pipeline" / "assets"
    weather_locations = args.weather_locations or str(assets / "owm_locations.txt")
    cctv_locations = args.cctv_locations or str(assets / "cctv.kml")
    catalog = DataCatalog(data_root, backup_root); sources = args.source or sorted(READERS)
    seen: set[str] = set()
    receiver_host = None if args.no_receiver else (args.receiver_host or endpoint.get("host"))
    receiver_port = args.receiver_port or int(endpoint.get("port", DEFAULT_RECEIVER_PORT))
    receiver_timeout = (args.receiver_timeout if args.receiver_timeout is not None
                        else float(endpoint.get("timeout_seconds", DEFAULT_RECEIVER_TIMEOUT)))
    receiver_retries = (args.receiver_retries if args.receiver_retries is not None
                        else int(endpoint.get("retries", DEFAULT_RECEIVER_RETRIES)))
    sink = (SocketJSONLSink(receiver_host, receiver_port, timeout=receiver_timeout,
                           retries=receiver_retries)
            if receiver_host else JSONLSink(sys.stdout))
    skipped = 0

    def emit(item):
        nonlocal skipped
        try:
            sink.write(item)
        except ObservationRejected as exc:
            if not args.skip_rejected: raise
            skipped += 1
            print(json.dumps({"skipped": item.id, "error": str(exc)}), file=sys.stderr, flush=True)

    try:
        if args.start != "now":
            start = _day(args.start); end = args.to or (start + timedelta(days=1))
            for item in historical(catalog, sources, start, end, weather_locations=weather_locations,
                                   cctv_locations=cctv_locations, local_timezone=args.timezone):
                if not _selected(item, args): continue
                if args.limit is not None and len(seen) >= args.limit: break
                emit(item); seen.add(item.id)
        else:
            # Monitoring means records appearing after startup, not replaying today's
            # already-collected files. This baseline is deliberately not persisted.
            for item in historical(catalog, sources, date.today() - timedelta(days=1), date.today() + timedelta(days=1),
                                   weather_locations=weather_locations, cctv_locations=cctv_locations,
                                   local_timezone=args.timezone):
                seen.add(item.id)
        if args.follow:
            try: follow(catalog, sources, emit, poll_seconds=args.poll_seconds, seen=seen,
                        weather_locations=weather_locations, cctv_locations=cctv_locations,
                        local_timezone=args.timezone)
            except KeyboardInterrupt: return 0
        return 0
    finally:
        close = getattr(sink, "close", None)
        if close is not None: close()
        if skipped: print(f"skipped {skipped} rejected observation(s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
