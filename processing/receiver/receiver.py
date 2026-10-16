"""Receive, enrich, and durably store inline Urban Observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Protocol

from urban_observation_model import Observation, ObservationValidationError
from processing.enrichment import EnrichmentClient
from processing.enrichment.backends import BudgetExceeded


class ObservationHandler(Protocol):
    def handle(self, observation: Observation) -> None: ...


class JSONLHandler:
    """Append a validated observation and flush it before ACK."""

    def __init__(self, stream):
        self.stream = stream

    def handle(self, observation: Observation) -> None:
        self.stream.write(observation.to_json() + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())


class EnrichingHandler:
    """Run the shared gate/enrichment before the durable common sink."""

    def __init__(self, downstream: ObservationHandler, client: EnrichmentClient, *, force=False):
        self.downstream, self.client, self.force = downstream, client, force

    def handle(self, observation: Observation) -> None:
        self.downstream.handle(self.client.enrich(observation, force=self.force))


def _reply(writer, observation_id, accepted, *, error=None, retryable=None, stop=False):
    value = {"id": observation_id, "accepted": bool(accepted)}
    if error is not None:
        value["error"] = str(error)
    if retryable is not None:
        value["retryable"] = bool(retryable)
    if stop:
        # The receiver cannot accept anything more (for example, the OpenAI
        # budget is spent), so the sender should stop instead of skipping ahead.
        value["stop"] = True
    writer.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    writer.flush()


def handle_connection(connection, handler: ObservationHandler, *, max_message_bytes=25_000_000):
    accepted = 0
    with connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        try:
            while True:
                line = reader.readline(max_message_bytes + 1)
                if not line:
                    break
                if len(line.encode("utf-8")) > max_message_bytes or not line.endswith("\n"):
                    _reply(writer, None, False, error="message exceeds size limit", retryable=False)
                    break
                observation_id = None
                try:
                    raw = json.loads(line)
                    observation_id = raw.get("id") if isinstance(raw, dict) else None
                    observation = Observation.from_dict(raw)
                    handler.handle(observation)
                    _reply(writer, observation.id, True)
                    accepted += 1
                except BudgetExceeded as exc:
                    _reply(writer, observation_id, False, error=exc, retryable=False, stop=True)
                except (json.JSONDecodeError, ObservationValidationError, ValueError) as exc:
                    _reply(writer, observation_id, False, error=exc, retryable=False)
                except Exception as exc:
                    _reply(writer, observation_id, False, error=exc, retryable=True)
        except OSError:
            # The sender went away mid-reply; only this connection is affected.
            pass
        finally:
            for handle in (reader, writer):
                try:
                    handle.close()
                except OSError:
                    pass
    return accepted


def serve(host, port, handler: ObservationHandler, *, max_message_bytes=25_000_000, ready=None, stop_after=None):
    total = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        if ready is not None:
            ready(server.getsockname()[1])
        while stop_after is None or total < stop_after:
            connection, _ = server.accept()
            total += handle_connection(connection, handler, max_message_bytes=max_message_bytes)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Receive validated inline Urban Observations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-message-bytes", type=int, default=25_000_000)
    parser.add_argument("--enrichment-url", default="http://127.0.0.1:8770")
    parser.add_argument("--enrichment-timeout", type=float, default=180.0)
    parser.add_argument("--force-enrichment", action="store_true")
    parser.add_argument("--no-enrichment", action="store_true",
                        help="Store validated raw observations without calling enrichment")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        try:
            handler = JSONLHandler(stream)
            if not args.no_enrichment:
                handler = EnrichingHandler(
                    handler,
                    EnrichmentClient(args.enrichment_url, timeout=args.enrichment_timeout),
                    force=args.force_enrichment,
                )
            serve(args.host, args.port, handler, max_message_bytes=args.max_message_bytes)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
