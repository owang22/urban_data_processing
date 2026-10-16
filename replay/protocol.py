"""Inline JSONL wire protocol and stop-and-wait sender."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import time
from pathlib import PurePosixPath
from typing import Any

from .catalog import open_file_reference
from .model import Observation
from urban_observation_model import Observation as SharedObservation, SCHEMA_VERSION


DEFAULT_RECEIVER_PORT = 8766
DEFAULT_RECEIVER_TIMEOUT = 120.0
DEFAULT_RECEIVER_RETRIES = 3


def receiver_config(config: dict[str, Any], legacy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the shared replay endpoint with optional legacy overrides."""
    shared = config.get("replay", {}).get("receiver", {})
    return {**shared, **(legacy or {})}


class ProtocolError(RuntimeError):
    pass


class ObservationRejected(ProtocolError):
    """The receiver refused this one observation and will not accept it on retry."""


class ReceiverStopped(ProtocolError):
    """The receiver asked the sender to stop sending entirely."""


def inline_observation(observation: Observation) -> dict[str, Any]:
    """Return a portable observation with every referenced file embedded."""
    value = observation.to_dict()
    value["schema_version"] = SCHEMA_VERSION
    embedded = []
    for reference in observation.files:
        with open_file_reference(reference) as handle:
            content = handle.read()
        name = PurePosixPath(reference.member or reference.path).name
        embedded.append({
            "name": name,
            "media_type": reference.media_type,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "content_base64": base64.b64encode(content).decode("ascii"),
        })
    value["files"] = embedded
    return SharedObservation.from_dict(value).to_dict()


class SocketJSONLSink:
    """Send one inline observation and wait for its ACK before continuing."""

    def __init__(self, host: str, port: int = DEFAULT_RECEIVER_PORT, *,
                 timeout: float = DEFAULT_RECEIVER_TIMEOUT,
                 retries: int = DEFAULT_RECEIVER_RETRIES):
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        self.host, self.port = host, port
        self.timeout, self.retries = timeout, retries
        self._socket = self._reader = self._writer = None

    def _connect(self) -> None:
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = sock.makefile("w", encoding="utf-8", newline="\n")

    def write(self, observation: Observation) -> None:
        line = json.dumps(inline_observation(observation), ensure_ascii=False, separators=(",", ":"))
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self._writer is None:
                    self._connect()
                self._writer.write(line + "\n")
                self._writer.flush()
                response_line = self._reader.readline()
                if not response_line:
                    raise ConnectionError("receiver closed before acknowledging the observation")
                response = json.loads(response_line)
                if response.get("id") != observation.id:
                    raise ProtocolError(f"ACK ID mismatch: expected {observation.id!r}, got {response.get('id')!r}")
                if response.get("accepted") is True:
                    return
                if response.get("accepted") is False:
                    error = str(response.get("error") or "receiver rejected observation")
                    if response.get("stop"):
                        raise ReceiverStopped(error)
                    if not response.get("retryable", False):
                        raise ObservationRejected(error)
                    raise ConnectionError(error)
                raise ProtocolError(f"unexpected receiver response: {response!r}")
            except ProtocolError:
                raise
            except (OSError, ConnectionError, json.JSONDecodeError) as exc:
                last_error = exc
                self.close()
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        raise ProtocolError(f"observation {observation.id} was not acknowledged: {last_error}")

    def close(self) -> None:
        for handle in (self._writer, self._reader, self._socket):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._socket = self._reader = self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
