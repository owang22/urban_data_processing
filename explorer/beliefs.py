"""What an agent believes about the world, and the evidence behind each belief.

An agent (an LLM loop, SIGMUS, a rule script or a person) records its
beliefs in a JSON Lines file, one file per run, in processing-output/beliefs.
Each line is a full revision of one belief; a belief's history is its lines
in order, so the map can replay how the agent's view changed:

    {"time": "2026-10-16T22:10:00Z",      when the agent made this revision
     "id": "bryson-growing",              stable across revisions
     "statement": "The Bryson Fire is still growing",
     "confidence": 0.8,                   the agent's own estimate, 0 to 1
     "status": "open",                    open | confirmed | refuted | retracted
     "kind": "hypothesis",                free text: hypothesis, prediction, ...
     "topic": "Bryson Fire",              optional; groups beliefs in the list
     "place": "Malibu", "lat": 34.09, "lon": -118.84, "radius_km": 3,
     "parents": ["bryson-active"],        beliefs this one builds on
     "reasoning": "Acreage rose from 147 to 250 in 4 hours ...",
     "evidence": [{"stance": "supports",  supports | contradicts | context
                   "weight": 0.6,         how much it moved the agent, 0 to 1
                   "source": "calfire",   any source name
                   "time": "...", "title": "...", "url": "...",
                   "frame": "alertcalifornia/20261016/Cam/20261016150000.jpg",
                   "lat": 34.1, "lon": -118.8, "ref": "observation id",
                   "note": "what this shows"}],
     "agent": "claude-opus-5-5"}

Evidence carries what the map needs to show it (title, link, camera frame,
place), so the viewer does not have to look anything up.

    log = BeliefLog("processing-output/beliefs/run-1.jsonl", agent="my-agent")
    log.record("bryson-growing", "The Bryson Fire is still growing", 0.8,
               evidence=[evidence("supports", "calfire", title="147 acres", weight=0.5)])
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

STATUSES = ("open", "confirmed", "refuted", "retracted")
STANCES = ("supports", "contradicts", "context")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def evidence(stance: str, source: str, *, weight: float = 0.5, **fields: Any) -> dict[str, Any]:
    """One piece of evidence; fields are time, title, url, frame, lat, lon, ref, note."""
    return {"stance": stance, "source": source, "weight": weight, **fields}


def validate(belief: dict[str, Any]) -> dict[str, Any]:
    """Check one revision; returns it with defaults filled in. Raises ValueError."""
    for key in ("id", "statement", "time"):
        if not belief.get(key):
            raise ValueError(f"belief needs {key!r}")
    datetime.fromisoformat(str(belief["time"]).replace("Z", "+00:00"))
    confidence = float(belief.get("confidence", 0.5))
    if not 0 <= confidence <= 1:
        raise ValueError(f"{belief['id']}: confidence {confidence} is outside 0-1")
    status = belief.get("status", "open")
    if status not in STATUSES:
        raise ValueError(f"{belief['id']}: status {status!r} is not one of {STATUSES}")
    items = []
    for item in belief.get("evidence") or []:
        if item.get("stance") not in STANCES:
            raise ValueError(f"{belief['id']}: evidence stance {item.get('stance')!r} is not one of {STANCES}")
        items.append({**item, "weight": min(max(float(item.get("weight", 0.5)), 0), 1)})
    return {**belief, "confidence": confidence, "status": status, "evidence": items,
            "parents": list(belief.get("parents") or [])}


class BeliefLog:
    """Append-only writer for one run."""

    def __init__(self, path: str | Path, agent: str):
        self.path = Path(path)
        self.agent = agent
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, belief_id: str, statement: str, confidence: float, *, time: datetime | None = None,
               evidence: list[dict[str, Any]] | None = None, **fields: Any) -> dict[str, Any]:
        belief = validate({"time": _iso(time or datetime.now(timezone.utc)), "id": belief_id,
                           "statement": statement, "confidence": confidence, "evidence": evidence or [],
                           "agent": self.agent, **fields})
        with self.path.open("a") as handle:
            handle.write(json.dumps(belief, separators=(",", ":")) + "\n")
        return belief


class BeliefRuns:
    """Reads every run in a folder for the map."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            revisions = self._read(path)
            if not revisions:
                continue
            out.append({"run": path.stem, "agents": sorted({r.get("agent") or "unknown" for r in revisions}),
                        "beliefs": len({r["id"] for r in revisions}), "revisions": len(revisions),
                        "start": revisions[0]["time"], "end": revisions[-1]["time"]})
        return out

    def state(self, run: str, at: datetime | None = None) -> dict[str, Any]:
        """Each belief's newest revision at or before `at`, with its confidence history."""
        if not RUN_NAME.match(run):
            raise ValueError("bad run name")
        path = self.folder / f"{run}.jsonl"
        if not path.exists():
            raise KeyError(run)
        cutoff = _iso(at) if at else None
        beliefs: dict[str, dict[str, Any]] = {}
        for revision in self._read(path):
            if cutoff and revision["time"] > cutoff:
                continue
            previous = beliefs.get(revision["id"])
            history = (previous["history"] if previous else []) + [
                {"time": revision["time"], "confidence": revision["confidence"], "status": revision["status"]}]
            beliefs[revision["id"]] = {**revision, "history": history, "revisions": len(history)}
        children: dict[str, list[str]] = {}
        for belief in beliefs.values():
            for parent in belief["parents"]:
                children.setdefault(parent, []).append(belief["id"])
        for belief in beliefs.values():
            belief["children"] = children.get(belief["id"], [])
        return {"run": run, "at": cutoff, "beliefs": sorted(beliefs.values(), key=lambda b: b["time"])}

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        revisions = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                revisions.append(validate(json.loads(line)))
            except (ValueError, TypeError) as exc:  # one bad line should not hide the run
                print(f"{path.name}:{number}: skipped: {exc}", flush=True)
        return sorted(revisions, key=lambda r: r["time"])
