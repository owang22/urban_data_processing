import json
import queue
import socket
import threading
from types import SimpleNamespace

import pytest

from urban_observation_model import Observation as SharedObservation, SCHEMA_VERSION
from processing.enrichment.backends import BudgetExceeded, OpenAIBackend, UsageLedger
from processing.receiver.receiver import handle_connection, serve
from replay.model import Observation
from replay.protocol import ObservationRejected, ReceiverStopped, SocketJSONLSink


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=0))


class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=_usage(1_000_000, 0))


def _backend(replies, ledger):
    completions = FakeCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIBackend(client=client, ledger=ledger), completions


def test_backend_requests_json_mode_and_retries_one_malformed_reply(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl", budget_usd=5)
    backend, completions = _backend(['{"summary": "x",, }', '{"summary": "ok"}'], ledger)
    assert backend.annotate_news("text") == {"summary": "ok"}
    assert len(completions.calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in completions.calls)
    # Both calls are paid for, so both are counted.
    assert ledger.spent_usd == pytest.approx(0.30)


def test_ledger_refuses_calls_once_budget_is_spent_and_survives_restart(tmp_path):
    path = tmp_path / "usage.jsonl"
    backend, completions = _backend(['{"a": 1}'] * 5, UsageLedger(path, budget_usd=0.25))
    backend.annotate_text("one")  # $0.15
    backend.annotate_text("two")  # $0.30 total, past the limit
    with pytest.raises(BudgetExceeded):
        backend.annotate_text("three")
    assert len(completions.calls) == 2

    restarted, completions = _backend(['{"a": 1}'], UsageLedger(path, budget_usd=0.25))
    with pytest.raises(BudgetExceeded):
        restarted.annotate_text("four")
    assert completions.calls == []
    assert len(path.read_text().splitlines()) == 2


def test_ledger_refuses_unpriced_model(tmp_path):
    completions = FakeCompletions(['{"a": 1}'])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIBackend(client=client, text_model="unknown-model",
                            ledger=UsageLedger(tmp_path / "u.jsonl", budget_usd=5))
    with pytest.raises(BudgetExceeded):
        backend.annotate_text("x")
    assert completions.calls == []


def _value(observation_id="gdelt:1"):
    return {"schema_version": SCHEMA_VERSION, "id": observation_id, "source": "gdelt",
            "time": "2025-01-07T12:00:00Z", "sensor": "gdelt", "data": {}, "files": []}


def _exchange(handler, value):
    left, right = socket.socketpair()
    thread = threading.Thread(target=handle_connection, args=(left, handler), daemon=True)
    thread.start()
    reader = right.makefile("r"); writer = right.makefile("w")
    writer.write(json.dumps(value) + "\n"); writer.flush()
    response = json.loads(reader.readline())
    right.close(); thread.join(2)
    return response


def test_receiver_tells_sender_to_stop_when_budget_is_spent():
    class Handler:
        def handle(self, observation): raise BudgetExceeded("spent")
    response = _exchange(Handler(), _value())
    assert response == {"id": "gdelt:1", "accepted": False, "error": "spent",
                        "retryable": False, "stop": True}


def test_receiver_survives_client_that_disconnects_before_reply():
    try:
        socket.socket().close()
    except PermissionError:
        pytest.skip("sandbox does not permit local sockets")
    ports, release = queue.Queue(), threading.Event()

    class Handler:
        def handle(self, observation):
            release.wait(2)

    server = threading.Thread(
        target=serve, args=("127.0.0.1", 0, Handler()),
        kwargs={"ready": ports.put, "stop_after": 1}, daemon=True)
    server.start(); port = ports.get(timeout=2)

    # First client sends an observation and hangs up without reading the ACK.
    rude = socket.create_connection(("127.0.0.1", port))
    rude.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    rude.sendall((json.dumps(_value("gdelt:rude")) + "\n").encode()); rude.close()
    release.set()

    # The server must still be there for the next client.
    with SocketJSONLSink("127.0.0.1", port, timeout=2, retries=0) as sink:
        sink.write(Observation("gdelt:2", "gdelt", "2025-01-07T12:00:00Z", "gdelt", {}))
    server.join(2)
    assert not server.is_alive()


def _sender_against(reply):
    try:
        socket.socket().close()
    except PermissionError:
        pytest.skip("sandbox does not permit local sockets")
    ports = queue.Queue()

    def receive_one():
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0)); server.listen(); ports.put(server.getsockname()[1])
            connection, _ = server.accept()
            with connection:
                reader = connection.makefile("r"); writer = connection.makefile("w")
                value = json.loads(reader.readline())
                writer.write(json.dumps({"id": value["id"], **reply}) + "\n"); writer.flush()

    threading.Thread(target=receive_one, daemon=True).start()
    sink = SocketJSONLSink("127.0.0.1", ports.get(timeout=2), timeout=2, retries=0)
    return sink, Observation("gdelt:1", "gdelt", "2025-01-07T12:00:00Z", "gdelt", {})


def test_sender_distinguishes_rejected_observation_from_stop():
    sink, observation = _sender_against({"accepted": False, "error": "bad json", "retryable": False})
    with sink, pytest.raises(ObservationRejected):
        sink.write(observation)
    sink, observation = _sender_against({"accepted": False, "error": "spent", "retryable": False, "stop": True})
    with sink, pytest.raises(ReceiverStopped):
        sink.write(observation)
