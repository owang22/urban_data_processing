#!/usr/bin/env python3
"""Low-latency multimodal anomaly detection for incident sensing reports.

This module is designed to sit immediately before `observation_model.py`.
It emits candidate `observed_effects` and `possible_incidents` using the same
label style as the observation model, while keeping state for streaming
numeric sensors.

Typical usage from observation_model.py or a pipeline driver:

    from anomaly_detection import MultimodalAnomalyDetector

    detector = MultimodalAnomalyDetector(
        use_text_embeddings=True,
        use_image_clip=True,
        use_yolo=False,
        use_river=True,
    )

    anomaly = detector.detect_report(report)
    model_payload = anomaly.to_observation_model_payload(
        allowed_effects=allowed_effects,
        allowed_incidents=allowed_incidents,
    )

Optional dependencies:

    pip install numpy pandas rapidfuzz sentence-transformers pillow transformers torch river ultralytics

The module is deliberately dependency-tolerant: keyword/rule-based detection
and rolling robust time-series anomaly scoring work without the heavy optional
models. Embedding, CLIP, YOLO, and River are loaded lazily only when enabled.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import warnings
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # Optional but strongly recommended.
    import numpy as np
except Exception:  # pragma: no cover - fallback for minimal environments.
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Labels and default configuration
# ---------------------------------------------------------------------------

DEFAULT_INCIDENT_LABELS: List[str] = [
    "active shooter situation",
    "civil protest",
    "demonstration",
    "fire",
    "home crime",
    "road vehicle accident",
    "wildfire",
    "bomb threat",
    "dangerous person threat",
    "flood",
    "industrial crime",
    "road closure",
    "school closing",
    "terrorist incident",
]

# These are the observed-effect labels already used by the current observation
# model. Unknown labels are intentionally avoided so observation_model.py can
# consume this module's output with its existing vocabulary validation.
DEFAULT_EFFECT_LABELS: List[str] = [
    "possible_smoke_or_haze",
    "visible_flames_or_fire",
    "standing_or_flowing_water",
    "stopped_or_slow_traffic",
    "high_vehicle_density",
    "road_blockage_or_barricade",
    "high_pedestrian_density",
    "visible_damage_or_debris",
    "emergency_vehicle_presence",
]

LEAKY_KEYS = {
    "incident_id",
    "report_id",
    "data_file",
    "image_filepath",
    "filepath",
    "filename",
    "path",
}

TEXT_KEYS = {
    "text",
    "message",
    "body",
    "content",
    "description",
    "summary",
    "title",
    "caption",
    "mentioned_location",
    "location_name",
    "place",
    "sensor_name",
}

IMAGE_SENSOR_TYPES = {"cctv", "camera", "image", "traffic_camera"}
TEXT_SENSOR_TYPES = {
    "twitter", "x", "x.com", "social", "social_media", "citizen", "citizen_report",
    "news", "gdelt",
}
TIME_SERIES_SENSOR_TYPES = {
    "air", "air_quality", "weather", "california_traffic", "traffic", "road_sensor",
    "pems", "pems-stations", "purpleair", "openweather",
}

# Collector field names that are semantically equivalent to configured rules.
# Keeping this adapter here avoids changing raw observations or proliferating
# source-specific enrichment schemas.
TIME_SERIES_FIELD_ALIASES = {
    "pm2.5_atm": "pm2.5",
    "pm2.5_cf_1": "pm2.5",
    "pm2_5_atm": "pm2_5",
    "pm2_5_cf_1": "pm2_5",
}


def _escape_phrase(phrase: str) -> str:
    return re.escape(phrase).replace(r"\ ", r"\s+")


# Phrase lists are intentionally easy to edit and export. Regex patterns are
# generated from these phrases and supplemented with a few explicit patterns.
DEFAULT_CONFIG: Dict[str, Any] = {
    "effect_phrases": {
        "possible_smoke_or_haze": [
            "smoke",
            "smoky",
            "smoke column",
            "smoke plume",
            "haze",
            "hazy",
            "poor visibility",
            "reduced visibility",
            "ash in the air",
            "bad air quality",
            "unhealthy air",
            "thick air",
        ],
        "visible_flames_or_fire": [
            "flames",
            "visible flames",
            "fire",
            "visible fire",
            "blaze",
            "burning",
            "brush fire",
            "vegetation fire",
            "forest fire",
        ],
        "standing_or_flowing_water": [
            "flood",
            "flooding",
            "flash flood",
            "standing water",
            "water covering the road",
            "water over the road",
            "flowing water",
            "submerged street",
            "inundated",
        ],
        "stopped_or_slow_traffic": [
            "stopped traffic",
            "slow traffic",
            "traffic jam",
            "gridlock",
            "backup",
            "backed up",
            "cars barely moving",
            "traffic congestion",
            "standstill",
        ],
        "high_vehicle_density": [
            "heavy traffic",
            "dense traffic",
            "many cars",
            "packed freeway",
            "bumper to bumper",
            "vehicle queue",
            "long line of cars",
        ],
        "road_blockage_or_barricade": [
            "road closed",
            "road closure",
            "lane closed",
            "lanes closed",
            "freeway closed",
            "blocked road",
            "blocked lane",
            "barricade",
            "police barricade",
            "checkpoint",
        ],
        "high_pedestrian_density": [
            "crowd",
            "large crowd",
            "large group",
            "march",
            "rally",
            "protesters",
            "demonstrators",
            "people gathering",
        ],
        "visible_damage_or_debris": [
            "damage",
            "debris",
            "wreckage",
            "collapsed",
            "collapse",
            "destroyed",
            "shattered glass",
            "explosion damage",
            "burned structure",
            "crash debris",
        ],
        "emergency_vehicle_presence": [
            "emergency vehicle",
            "fire truck",
            "fire engine",
            "police car",
            "police vehicles",
            "ambulance",
            "paramedics",
            "responders",
            "firefighters",
        ],
    },
    "incident_phrases": {
        "active shooter situation": [
            "active shooter",
            "shots fired",
            "gunman",
            "shooting",
            "mass shooting",
            "shelter in place",
            "lockdown due to shooter",
        ],
        "civil protest": [
            "civil protest",
            "protest",
            "protesters",
            "rally",
            "march",
            "demonstration",
            "public demonstration",
        ],
        "demonstration": [
            "demonstration",
            "demonstrators",
            "march",
            "rally",
            "protest",
        ],
        "fire": [
            "fire",
            "flames",
            "blaze",
            "burning",
            "structure fire",
            "building fire",
            "vehicle fire",
        ],
        "home crime": [
            "home invasion",
            "break in",
            "break-in",
            "burglary",
            "robbery at home",
            "residential burglary",
        ],
        "road vehicle accident": [
            "car crash",
            "vehicle crash",
            "traffic collision",
            "collision",
            "multi vehicle crash",
            "overturned vehicle",
            "accident on the freeway",
            "crash debris",
        ],
        "wildfire": [
            "wildfire",
            "brush fire",
            "vegetation fire",
            "forest fire",
            "wildland fire",
            "fire in the hills",
            "red flag fire",
        ],
        "bomb threat": [
            "bomb threat",
            "suspicious package",
            "explosive device",
            "possible bomb",
            "device found",
        ],
        "dangerous person threat": [
            "dangerous person",
            "armed suspect",
            "person with a weapon",
            "threatening person",
            "suspect with a knife",
            "wanted suspect",
        ],
        "flood": [
            "flood",
            "flooding",
            "flash flood",
            "water over the road",
            "submerged street",
            "inundated",
        ],
        "industrial crime": [
            "industrial accident",
            "chemical spill",
            "hazmat",
            "hazardous material",
            "refinery fire",
            "factory fire",
            "warehouse explosion",
            "industrial explosion",
        ],
        "road closure": [
            "road closure",
            "road closed",
            "lane closure",
            "lanes closed",
            "freeway closed",
            "blocked road",
            "evacuation route closed",
        ],
        "school closing": [
            "school closing",
            "school closed",
            "schools closed",
            "classes canceled",
            "classes cancelled",
            "campus closed",
            "remote instruction today",
        ],
        "terrorist incident": [
            "terrorist incident",
            "terrorist attack",
            "terror attack",
            "terrorism",
            "coordinated attack",
            "bombing",
            "multiple explosions",
        ],
    },
    "effect_prototypes": {
        "possible_smoke_or_haze": [
            "smoke or haze is visible in the area",
            "there is a smoke plume or smoke column",
            "visibility is reduced because of smoke or ash",
            "air quality is poor because of smoke",
        ],
        "visible_flames_or_fire": [
            "flames are visible",
            "a fire is burning nearby",
            "there is a brush fire or vegetation fire",
            "a building or vehicle is on fire",
        ],
        "standing_or_flowing_water": [
            "water is standing or flowing across a road",
            "a street or area is flooded",
            "flood water is covering the roadway",
        ],
        "stopped_or_slow_traffic": [
            "traffic is stopped or moving very slowly",
            "cars are backed up in congestion",
            "there is gridlock on the road",
        ],
        "high_vehicle_density": [
            "there are many vehicles on the road",
            "traffic density is high",
            "lanes are filled with cars",
        ],
        "road_blockage_or_barricade": [
            "a road or lane is blocked",
            "a road closure or barricade is visible",
            "police or cones are blocking traffic",
        ],
        "high_pedestrian_density": [
            "there is a large crowd of people",
            "many pedestrians are gathered or marching",
            "a crowd or protest is visible",
        ],
        "visible_damage_or_debris": [
            "there is visible damage, debris, or wreckage",
            "a structure or vehicle appears damaged",
            "debris is scattered in the scene",
        ],
        "emergency_vehicle_presence": [
            "emergency vehicles or responders are present",
            "police cars, fire trucks, or ambulances are visible",
            "first responders are on scene",
        ],
    },
    "incident_prototypes": {
        "active shooter situation": [
            "an active shooter or shots fired situation is being reported",
            "police are responding to a gunman or shooting",
        ],
        "civil protest": [
            "a civil protest, march, or rally is occurring",
            "protesters are gathered in a public area",
        ],
        "demonstration": [
            "a public demonstration or march is occurring",
            "demonstrators are gathered in a crowd",
        ],
        "fire": [
            "a fire is burning and flames or smoke are present",
            "there is a structure, vehicle, or outdoor fire",
        ],
        "home crime": [
            "a burglary, home invasion, or residential crime is being reported",
        ],
        "road vehicle accident": [
            "a vehicle crash or traffic collision is blocking the road",
            "cars have crashed and debris may be present",
        ],
        "wildfire": [
            "a wildfire or brush fire is burning in vegetation or hills",
            "smoke is rising from a wildland or vegetation fire",
        ],
        "bomb threat": [
            "a bomb threat or suspicious package is being reported",
            "an explosive device may be present",
        ],
        "dangerous person threat": [
            "an armed or dangerous person is threatening public safety",
            "a suspect with a weapon is being reported",
        ],
        "flood": [
            "a flood or flash flood is affecting roads or buildings",
            "water is covering streets or property",
        ],
        "industrial crime": [
            "an industrial accident, chemical spill, or hazmat incident is occurring",
            "a factory, warehouse, refinery, or industrial site is involved",
        ],
        "road closure": [
            "a road or highway is closed or blocked",
            "traffic is being diverted because lanes are closed",
        ],
        "school closing": [
            "a school or campus is closed and classes are canceled",
        ],
        "terrorist incident": [
            "a terrorist attack, coordinated attack, bombing, or mass casualty emergency is being reported",
            "multiple attacks or explosions suggest terrorism",
        ],
    },
    "image_prompts": {
        "possible_smoke_or_haze": [
            "a CCTV image showing heavy smoke or haze",
            "a highway scene with smoke in the air",
            "a camera view with reduced visibility from smoke",
        ],
        "visible_flames_or_fire": [
            "a CCTV image showing visible flames or fire",
            "fire burning in vegetation near a road",
            "a scene with an active fire",
        ],
        "standing_or_flowing_water": [
            "a CCTV image showing flood water covering a road",
            "standing water or flowing water on a street",
        ],
        "stopped_or_slow_traffic": [
            "a CCTV image showing stopped traffic or gridlock",
            "cars backed up and moving slowly on a highway",
        ],
        "high_vehicle_density": [
            "a CCTV image showing many vehicles crowded on a highway",
            "dense traffic with many cars filling the lanes",
        ],
        "road_blockage_or_barricade": [
            "a CCTV image showing a road closure or blocked lanes",
            "traffic cones, barricades, or emergency vehicles blocking a road",
        ],
        "high_pedestrian_density": [
            "a CCTV image showing a large crowd of people",
            "many pedestrians gathered in a public place",
        ],
        "visible_damage_or_debris": [
            "a CCTV image showing visible damage, debris, or wreckage",
            "a damaged road, damaged vehicles, or debris in the street",
        ],
        "emergency_vehicle_presence": [
            "a CCTV image showing emergency vehicles or first responders",
            "police cars, fire trucks, or ambulances visible in a street scene",
        ],
        "wildfire": [
            "a CCTV image showing a wildfire or brush fire near a road",
            "smoke and flames from a vegetation fire in hills",
        ],
        "road closure": [
            "a CCTV image showing a highway or road closure",
            "vehicles stopped because lanes are closed",
        ],
        "road vehicle accident": [
            "a CCTV image showing a vehicle accident or crash on the road",
        ],
        "flood": [
            "a CCTV image showing flooding over a road or street",
        ],
        "civil protest": [
            "a CCTV image showing a protest march or public demonstration",
        ],
        "demonstration": [
            "a CCTV image showing demonstrators or a large rally",
        ],
        "terrorist incident": [
            "a CCTV image showing smoke, explosion damage, emergency vehicles, and a major attack scene",
        ],
    },
    "image_prefilter": {
        "enabled": True,
        "heartbeat_seconds": 3600.0,
        # Aggressive default: do not send every visually similar CCTV frame to CLIP.
        "change_threshold": 18.0,
        "resize_size": 48,
        # First frames are usually ordinary baselines. Run CLIP on a first frame
        # only when cheap color/contrast cues suggest visual interest.
        "run_on_first_frame": False,
        "run_first_frame_if_visually_interesting": True,
        "visual_interest_orange_fraction_min": 0.004,
        "visual_interest_dark_fraction_min": 0.10,
        "visual_interest_low_contrast_haze_std_max": 18.0,
        "visual_interest_low_contrast_haze_saturation_max": 38.0,
        "fail_open_on_error": False,
    },
    "traffic_prefilter": {
        "enabled": True,
        "sensor_type_patterns": ["pem", "pems", "traffic"],
        # Looser PeMS gate: the downstream hourly top-K selector is now the main
        # volume control, so this prefilter should keep enough moderate traffic
        # anomalies for ranking rather than only the most extreme stoppages.
        "min_candidate_score": 0.60,
        "severe_speed_mph": 5.0,
        "high_speed_mph": 15.0,
        "medium_speed_mph": 25.0,
        "moderate_speed_mph": 35.0,
        "medium_occupancy": 0.45,
        "high_occupancy": 0.70,
        "severe_occupancy": 0.85,
        # Keep isolated but meaningful PeMS anomalies. The hourly top-K archive
        # will still limit final saved/selected time-series examples.
        "require_consecutive": False,
        "consecutive_window_seconds": 3600.0,
        "severe_bypass_score": 0.98,
        # Allow at most roughly one candidate per station per hour. This avoids
        # one bad station flooding the candidate pool while still allowing a
        # persistent traffic incident to reappear in consecutive hourly buckets.
        "candidate_cooldown_seconds": 3600.0,
        # Add a weak gate for moderate congestion when speed and occupancy agree.
        "moderate_combined_score": 0.62,
        "medium_combined_score": 0.72,
    },
    # Time-series rules are both threshold detectors and effect/incident mappers.
    # Rolling robust z-score detection is applied in addition to these rules.
    "time_series_rules": {
        "aqi": {
            "direction": "high",
            "medium": 100.0,
            "high": 200.0,
            "effects": ["possible_smoke_or_haze"],
            "evidence": "Elevated AQI.",
        },
        "pm25": {
            "direction": "high",
            "medium": 55.0,
            "high": 150.0,
            "effects": ["possible_smoke_or_haze"],
            "evidence": "Elevated PM2.5.",
        },
        "pm2_5": {
            "direction": "high",
            "medium": 55.0,
            "high": 150.0,
            "effects": ["possible_smoke_or_haze"],
            "evidence": "Elevated PM2.5.",
        },
        "pm2.5": {
            "direction": "high",
            "medium": 55.0,
            "high": 150.0,
            "effects": ["possible_smoke_or_haze"],
            "evidence": "Elevated PM2.5.",
        },
        "pm10": {
            "direction": "high",
            "medium": 100.0,
            "high": 180.0,
            "effects": ["possible_smoke_or_haze"],
            "evidence": "Elevated PM10.",
        },
        "data_avg_speed": {
            "direction": "low",
            "medium": 30.0,
            "high": 15.0,
            "effects": ["stopped_or_slow_traffic"],
            "incidents": ["road closure", "road vehicle accident"],
            "evidence": "Traffic speed is unusually low.",
        },
        "avg_speed": {
            "direction": "low",
            "medium": 30.0,
            "high": 15.0,
            "effects": ["stopped_or_slow_traffic"],
            "incidents": ["road closure", "road vehicle accident"],
            "evidence": "Traffic speed is unusually low.",
        },
        "speed_mph": {
            "direction": "low",
            "medium": 30.0,
            "high": 15.0,
            "effects": ["stopped_or_slow_traffic"],
            "incidents": ["road closure", "road vehicle accident"],
            "evidence": "Traffic speed is unusually low.",
        },
        "data_avg_occupancy": {
            "direction": "high",
            "medium": 0.45,
            "high": 0.75,
            "effects": ["high_vehicle_density"],
            "incidents": ["road closure"],
            "evidence": "Traffic occupancy is elevated.",
        },
        "avg_occupancy": {
            "direction": "high",
            "medium": 0.45,
            "high": 0.75,
            "effects": ["high_vehicle_density"],
            "incidents": ["road closure"],
            "evidence": "Traffic occupancy is elevated.",
        },
        "occupancy": {
            "direction": "high",
            "medium": 0.45,
            "high": 0.75,
            "effects": ["high_vehicle_density"],
            "incidents": ["road closure"],
            "evidence": "Traffic occupancy is elevated.",
        },
        "precipitation_in": {
            "direction": "high",
            "medium": 0.25,
            "high": 0.75,
            "effects": ["standing_or_flowing_water"],
            "incidents": ["flood"],
            "evidence": "Precipitation is elevated enough to support flood monitoring.",
        },
        "wind_speed_mph": {
            "direction": "high",
            "medium": 25.0,
            "high": 45.0,
            "effects": [],
            "incidents": ["wildfire"],
            "evidence": "High wind can support fast fire spread when other fire evidence is present.",
            "context_only": True,
        },
        "temperature_f": {
            "direction": "high",
            "medium": 95.0,
            "high": 105.0,
            "effects": [],
            "incidents": ["wildfire"],
            "evidence": "High temperature can support wildfire risk when other fire evidence is present.",
            "context_only": True,
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a recursive merge where override values replace or extend base."""
    merged = json.loads(json.dumps(base))  # cheap deep copy for JSON-like config
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_anomaly_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load anomaly configuration, optionally overlaying a JSON file.

    The JSON file can contain any subset of DEFAULT_CONFIG, for example:

        {
          "incident_phrases": {
            "wildfire": ["spot fire", "ember cast"]
          }
        }

    Lists in the override replace the default list for that label. To append,
    load the default with `save_default_anomaly_config`, edit it, and pass it in.
    """
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    path = Path(path)
    with path.open("r", encoding="utf-8") as infile:
        override = json.load(infile)
    if not isinstance(override, dict):
        raise ValueError(f"Anomaly config must be a JSON object: {path}")
    return _deep_merge(DEFAULT_CONFIG, override)


def save_default_anomaly_config(path: str | Path) -> None:
    """Write the default editable label/phrase/prompt config to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(DEFAULT_CONFIG, outfile, indent=2, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def score_to_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@dataclass
class CandidateLabel:
    """One candidate observation effect or incident from the anomaly layer."""

    name: str
    score: float
    label_type: str  # "effect" or "incident"
    evidence: str = ""
    method: str = ""
    modality: str = ""

    def to_dict(self, *, include_score: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "name": self.name,
            "level": score_to_level(self.score),
            "evidence": self.evidence,
        }
        if include_score:
            value["score"] = round(clamp(self.score), 4)
        if self.method:
            value["method"] = self.method
        if self.modality:
            value["modality"] = self.modality
        return value


@dataclass
class AnomalyResult:
    """Multimodal anomaly output for one report or sensor row."""

    report_id: Optional[str] = None
    sensor_id: Optional[str] = None
    sensor_type: Optional[str] = None
    timestamp: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    anomaly_score: float = 0.0
    candidate_effects: List[CandidateLabel] = field(default_factory=list)
    candidate_incidents: List[CandidateLabel] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "AnomalyResult") -> "AnomalyResult":
        self.candidate_effects = dedupe_candidates(self.candidate_effects + other.candidate_effects)
        self.candidate_incidents = dedupe_candidates(self.candidate_incidents + other.candidate_incidents)
        self.anomaly_score = max(self.anomaly_score, other.anomaly_score)
        self.diagnostics.update(other.diagnostics)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "location": self.location,
            "modality": self.modality,
            "anomaly_score": round(clamp(self.anomaly_score), 4),
            "candidate_effects": [item.to_dict() for item in self.candidate_effects],
            "candidate_incidents": [item.to_dict() for item in self.candidate_incidents],
            "diagnostics": self.diagnostics,
        }

    def to_observation_model_payload(
        self,
        *,
        allowed_effects: Optional[Sequence[str]] = None,
        allowed_incidents: Optional[Sequence[str]] = None,
        min_effect_score: float = 0.30,
        min_incident_score: float = 0.30,
        include_score: bool = True,
    ) -> Dict[str, Any]:
        """Return JSON that `observation_model.validate_model_payload` can consume."""
        effect_set = set(allowed_effects) if allowed_effects is not None else None
        incident_set = set(allowed_incidents) if allowed_incidents is not None else None

        effects = [
            item.to_dict(include_score=include_score)
            for item in self.candidate_effects
            if item.score >= min_effect_score and (effect_set is None or item.name in effect_set)
        ]
        incidents = [
            item.to_dict(include_score=include_score)
            for item in self.candidate_incidents
            if item.score >= min_incident_score and (incident_set is None or item.name in incident_set)
        ]
        return {
            "observed_effects": effects,
            "possible_incidents": incidents,
            "anomaly": {
                "score": round(clamp(self.anomaly_score), 4),
                "modality": self.modality,
                "diagnostics": self.diagnostics,
            },
        }


def dedupe_candidates(candidates: Sequence[CandidateLabel]) -> List[CandidateLabel]:
    """Deduplicate by (type, name), keeping the highest score and merging evidence."""
    best: Dict[Tuple[str, str], CandidateLabel] = {}
    for item in candidates:
        item.score = clamp(item.score)
        key = (item.label_type, item.name)
        old = best.get(key)
        if old is None or item.score > old.score:
            best[key] = item
        elif old is not None and item.evidence and item.evidence not in old.evidence:
            old.evidence = (old.evidence + "; " + item.evidence).strip("; ")
    return sorted(best.values(), key=lambda x: (-x.score, x.label_type, x.name))


def _add_candidate(
    candidates: List[CandidateLabel],
    *,
    name: str,
    score: float,
    label_type: str,
    evidence: str,
    method: str,
    modality: str,
) -> None:
    candidates.append(
        CandidateLabel(
            name=name,
            score=clamp(score),
            label_type=label_type,
            evidence=evidence,
            method=method,
            modality=modality,
        )
    )


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def report_data(report: Mapping[str, Any]) -> Dict[str, Any]:
    data = report.get("data")
    if isinstance(data, Mapping):
        return dict(data)
    # Also support raw CSV rows, where the row itself is the data object.
    return {str(k): v for k, v in report.items() if str(k) != "data"}


def report_sensor_type(report: Mapping[str, Any]) -> str:
    return str(report.get("sensor_type") or "").strip().lower()


def report_modality(report: Mapping[str, Any]) -> str:
    metadata = report.get("metadata") or {}
    if isinstance(metadata, Mapping):
        modality = str(metadata.get("modality") or "").strip().lower()
        if modality:
            return modality
    sensor_type = report_sensor_type(report)
    data = report_data(report)
    if "image_filepath" in data or sensor_type in IMAGE_SENSOR_TYPES:
        return "image"
    if sensor_type in TEXT_SENSOR_TYPES:
        return "text"
    if sensor_type in TIME_SERIES_SENSOR_TYPES or any(_is_number(v) for v in data.values()):
        return "timeseries"
    if any(key in data for key in TEXT_KEYS):
        return "text"
    return "unknown"


def is_image_report(report: Mapping[str, Any]) -> bool:
    data = report_data(report)
    return report_modality(report) == "image" or "image_filepath" in data


def is_text_report(report: Mapping[str, Any]) -> bool:
    return report_modality(report) == "text"


def is_time_series_report(report: Mapping[str, Any]) -> bool:
    return report_modality(report) == "timeseries"


def result_metadata_from_report(report: Mapping[str, Any], modality: Optional[str] = None) -> Dict[str, Any]:
    data = report_data(report)
    timestamp = report.get("report_date") or report.get("timestamp") or data.get("timestamp") or data.get("time")
    location = report.get("location")
    if not isinstance(location, dict):
        lat = data.get("latitude") or data.get("lat")
        lon = data.get("longitude") or data.get("lon")
        if _is_number(lat) and _is_number(lon):
            location = {"latitude": float(lat), "longitude": float(lon)}
        else:
            location = None
    return {
        "report_id": report.get("report_id"),
        "sensor_id": report.get("sensor_id") or data.get("sensor_id"),
        "sensor_type": report.get("sensor_type") or data.get("sensor_type"),
        "timestamp": str(timestamp) if timestamp is not None else None,
        "location": location,
        "modality": modality or report_modality(report),
    }


def _walk_text(value: Any, *, key_hint: str = "") -> Iterable[str]:
    """Yield non-leaky text fields from nested data/report structures."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            if key_str.lower() in LEAKY_KEYS:
                continue
            yield from _walk_text(child, key_hint=key_str)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_text(item, key_hint=key_hint)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return
        # Prefer natural-language fields, but also allow short string metadata
        # such as mentioned_location and descriptions.
        if key_hint.lower() in TEXT_KEYS or len(text.split()) >= 2:
            yield text


def extract_text_for_detection(report_or_row: Mapping[str, Any]) -> str:
    """Flatten useful non-leaky text from a report or raw CSV row."""
    pieces: List[str] = []
    # For normalized reports, include top-level sensor_name and data.
    sensor_name = report_or_row.get("sensor_name")
    if isinstance(sensor_name, str) and sensor_name.strip():
        pieces.append(sensor_name.strip())
    data = report_or_row.get("data")
    if isinstance(data, Mapping):
        pieces.extend(_walk_text(data))
    else:
        pieces.extend(_walk_text(report_or_row))
    # De-duplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for piece in pieces:
        norm = piece.lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(piece)
    return "\n".join(unique)


def _is_number(value: Any) -> bool:
    try:
        if value is None or value == "":
            return False
        number = float(value)
        return math.isfinite(number)
    except Exception:
        return False


def numeric_features_from_mapping(row: Mapping[str, Any]) -> Dict[str, float]:
    """Extract numeric features from a report or raw CSV row."""
    data = row.get("data") if isinstance(row.get("data"), Mapping) else row
    features: Dict[str, float] = {}
    for key, value in data.items():
        key_str = str(key)
        if key_str.lower() in LEAKY_KEYS:
            continue
        if _is_number(value):
            features[key_str] = float(value)
    return features


# ---------------------------------------------------------------------------
# Text anomaly detection
# ---------------------------------------------------------------------------


class TextEmbeddingScorer:
    """Sentence-transformers similarity scorer for labels/prototypes."""

    def __init__(
        self,
        prototypes: Mapping[str, Sequence[str]],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.prototypes = {label: list(texts) for label, texts in prototypes.items()}
        self.model_name = model_name
        self.model: Any = None
        self.prototype_embeddings: Dict[str, Any] = {}
        self.available = False
        self.error: Optional[str] = None

    def _load(self) -> None:
        if self.model is not None or self.error:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(self.model_name)
            all_texts: List[str] = []
            offsets: Dict[str, Tuple[int, int]] = {}
            start = 0
            for label, texts in self.prototypes.items():
                all_texts.extend(texts)
                offsets[label] = (start, start + len(texts))
                start += len(texts)
            if not all_texts:
                self.available = True
                return
            embeddings = self.model.encode(all_texts, normalize_embeddings=True)
            for label, (lo, hi) in offsets.items():
                self.prototype_embeddings[label] = embeddings[lo:hi]
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency.
            self.error = f"sentence-transformers unavailable: {exc}"
            self.available = False

    def score(self, text: str) -> Dict[str, Tuple[float, str]]:
        """Return {label: (score, evidence)} based on max prototype similarity."""
        self._load()
        if not self.available or not text.strip() or self.model is None:
            return {}
        query = self.model.encode([text], normalize_embeddings=True)[0]
        out: Dict[str, Tuple[float, str]] = {}
        for label, embeddings in self.prototype_embeddings.items():
            if np is not None:
                sims = np.asarray(embeddings) @ np.asarray(query)
                idx = int(np.argmax(sims))
                sim = float(sims[idx])
            else:  # fallback dot product without numpy
                sims = [sum(float(a) * float(b) for a, b in zip(vec, query)) for vec in embeddings]
                idx = max(range(len(sims)), key=sims.__getitem__)
                sim = float(sims[idx])
            # Cosine similarity can be negative. Map useful similarity region to [0,1].
            # all-MiniLM scores around 0.35-0.45 are often weakly topical; >0.55 is strong.
            score = clamp((sim - 0.30) / 0.35)
            out[label] = (score, f"embedding similarity {sim:.3f} to prototype: {self.prototypes[label][idx]}")
        return out


class TextAnomalyDetector:
    """Keyword/fuzzy/embedding detector for short text reports."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        use_embeddings: bool = True,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        keyword_score: float = 0.88,
        fuzzy_score: float = 0.72,
        embedding_min_score: float = 0.34,
        max_labels: int = 8,
    ) -> None:
        self.config = dict(config or load_anomaly_config())
        self.use_embeddings = use_embeddings
        self.keyword_score = keyword_score
        self.fuzzy_score = fuzzy_score
        self.embedding_min_score = embedding_min_score
        self.max_labels = max_labels
        self.effect_phrases: Dict[str, List[str]] = {
            k: list(v) for k, v in self.config.get("effect_phrases", {}).items()
        }
        self.incident_phrases: Dict[str, List[str]] = {
            k: list(v) for k, v in self.config.get("incident_phrases", {}).items()
        }
        self.effect_patterns = self._compile_phrase_patterns(self.effect_phrases)
        self.incident_patterns = self._compile_phrase_patterns(self.incident_phrases)

        self.effect_embedder: Optional[TextEmbeddingScorer] = None
        self.incident_embedder: Optional[TextEmbeddingScorer] = None
        if use_embeddings:
            self.effect_embedder = TextEmbeddingScorer(
                self.config.get("effect_prototypes", {}), model_name=embedding_model_name
            )
            self.incident_embedder = TextEmbeddingScorer(
                self.config.get("incident_prototypes", {}), model_name=embedding_model_name
            )

        try:
            from rapidfuzz import fuzz  # type: ignore

            self._fuzz = fuzz
            self.fuzzy_available = True
        except Exception:  # pragma: no cover - optional dependency.
            self._fuzz = None
            self.fuzzy_available = False

    @staticmethod
    def _compile_phrase_patterns(phrase_map: Mapping[str, Sequence[str]]) -> Dict[str, List[re.Pattern[str]]]:
        patterns: Dict[str, List[re.Pattern[str]]] = {}
        for label, phrases in phrase_map.items():
            compiled: List[re.Pattern[str]] = []
            for phrase in phrases:
                # Word-boundary-ish match for alphanumeric phrases; allow punctuation around phrase.
                escaped = _escape_phrase(phrase.lower())
                compiled.append(re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE))
            patterns[label] = compiled
        return patterns

    def _keyword_candidates(
        self,
        text: str,
        patterns: Mapping[str, Sequence[re.Pattern[str]]],
        phrase_map: Mapping[str, Sequence[str]],
        *,
        label_type: str,
    ) -> List[CandidateLabel]:
        candidates: List[CandidateLabel] = []
        if not text.strip():
            return candidates
        lower_text = text.lower()
        for label, compiled in patterns.items():
            matched: Optional[str] = None
            for pattern in compiled:
                match = pattern.search(lower_text)
                if match:
                    matched = match.group(0)
                    break
            if matched is not None:
                _add_candidate(
                    candidates,
                    name=label,
                    score=self.keyword_score,
                    label_type=label_type,
                    evidence=f"matched phrase: {matched}",
                    method="keyword",
                    modality="text",
                )
                continue
            # Optional fuzzy match catches minor typos and variants without hand-writing every form.
            if self.fuzzy_available and self._fuzz is not None:
                best_phrase = ""
                best_ratio = 0.0
                for phrase in phrase_map.get(label, []):
                    # Avoid very short fuzzy phrases such as "fire" matching substrings
                    # like "firefighting". Short phrases are handled by regex boundaries.
                    if len(re.sub(r"[^a-z0-9]", "", phrase.lower())) < 6:
                        continue
                    ratio = float(self._fuzz.partial_ratio(phrase.lower(), lower_text)) / 100.0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_phrase = phrase
                if best_ratio >= 0.92:
                    _add_candidate(
                        candidates,
                        name=label,
                        score=max(self.fuzzy_score, best_ratio * 0.85),
                        label_type=label_type,
                        evidence=f"fuzzy match {best_ratio:.2f}: {best_phrase}",
                        method="rapidfuzz",
                        modality="text",
                    )
        return candidates

    def _embedding_candidates(
        self,
        text: str,
        scorer: Optional[TextEmbeddingScorer],
        *,
        label_type: str,
    ) -> List[CandidateLabel]:
        if scorer is None or not self.use_embeddings or not text.strip():
            return []
        candidates: List[CandidateLabel] = []
        for label, (score, evidence) in scorer.score(text).items():
            if score >= self.embedding_min_score:
                _add_candidate(
                    candidates,
                    name=label,
                    score=score,
                    label_type=label_type,
                    evidence=evidence,
                    method="sentence-transformers",
                    modality="text",
                )
        return candidates

    def detect_text(self, text: str, *, metadata: Optional[Mapping[str, Any]] = None) -> AnomalyResult:
        metadata = dict(metadata or {})
        candidates: List[CandidateLabel] = []
        candidates.extend(
            self._keyword_candidates(
                text, self.effect_patterns, self.effect_phrases, label_type="effect"
            )
        )
        candidates.extend(
            self._keyword_candidates(
                text, self.incident_patterns, self.incident_phrases, label_type="incident"
            )
        )
        candidates.extend(self._embedding_candidates(text, self.effect_embedder, label_type="effect"))
        candidates.extend(self._embedding_candidates(text, self.incident_embedder, label_type="incident"))
        candidates = dedupe_candidates(candidates)

        effects = [c for c in candidates if c.label_type == "effect"][: self.max_labels]
        incidents = [c for c in candidates if c.label_type == "incident"][: self.max_labels]
        anomaly_score = max([c.score for c in candidates], default=0.0)
        diagnostics = {
            "text_length": len(text),
            "fuzzy_available": self.fuzzy_available,
            "embeddings_enabled": self.use_embeddings,
        }
        if self.effect_embedder and self.effect_embedder.error:
            diagnostics["effect_embedding_error"] = self.effect_embedder.error
        if self.incident_embedder and self.incident_embedder.error:
            diagnostics["incident_embedding_error"] = self.incident_embedder.error

        metadata = {**metadata, "modality": "text"}
        return AnomalyResult(
            **metadata,
            anomaly_score=anomaly_score,
            candidate_effects=effects,
            candidate_incidents=incidents,
            diagnostics=diagnostics,
        )

    def detect_report(self, report: Mapping[str, Any]) -> AnomalyResult:
        text = extract_text_for_detection(report)
        return self.detect_text(text, metadata=result_metadata_from_report(report, "text"))


# ---------------------------------------------------------------------------
# Image anomaly detection: CLIP zero-shot prompts + optional YOLO counts
# ---------------------------------------------------------------------------


class ImageCLIPScorer:
    """CLIP prompt scorer using transformers."""

    def __init__(
        self,
        image_prompts: Mapping[str, Sequence[str]],
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        self.image_prompts = {label: list(prompts) for label, prompts in image_prompts.items()}
        self.model_name = model_name
        self.device = device
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.available = False
        self.error: Optional[str] = None
        self.prompts: List[str] = []
        self.prompt_labels: List[str] = []
        for label, prompts in self.image_prompts.items():
            for prompt in prompts:
                self.prompt_labels.append(label)
                self.prompts.append(prompt)

    def _load(self) -> None:
        if self.model is not None or self.error:
            return
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self.torch = torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency.
            self.error = f"transformers/CLIP unavailable: {exc}"
            self.available = False

    def score_image(self, image_path: str | Path) -> Dict[str, Tuple[float, str]]:
        self._load()
        if not self.available or self.model is None or self.processor is None or not self.prompts:
            return {}
        try:
            from PIL import Image  # type: ignore

            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=self.prompts, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0).detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - optional dependency/runtime.
            self.error = f"CLIP scoring failed: {exc}"
            return {}

        best_by_label: Dict[str, Tuple[float, str]] = {}
        for label, prompt, prob in zip(self.prompt_labels, self.prompts, probs):
            old = best_by_label.get(label)
            if old is None or float(prob) > old[0]:
                best_by_label[label] = (float(prob), prompt)
        return {label: (score, f"CLIP prompt probability {score:.3f}: {prompt}") for label, (score, prompt) in best_by_label.items()}


class YOLOCountScorer:
    """Optional Ultralytics YOLO scorer for people/vehicle density."""

    VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}
    PERSON_CLASS_NAMES = {"person"}

    def __init__(
        self,
        *,
        model_name: str = "yolov8n.pt",
        vehicle_count_high: int = 20,
        vehicle_count_medium: int = 8,
        person_count_high: int = 30,
        person_count_medium: int = 12,
        confidence: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.vehicle_count_high = vehicle_count_high
        self.vehicle_count_medium = vehicle_count_medium
        self.person_count_high = person_count_high
        self.person_count_medium = person_count_medium
        self.confidence = confidence
        self.model: Any = None
        self.available = False
        self.error: Optional[str] = None

    def _load(self) -> None:
        if self.model is not None or self.error:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.model_name)
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency.
            self.error = f"ultralytics unavailable: {exc}"
            self.available = False

    @staticmethod
    def _score_count(count: int, medium: int, high: int) -> float:
        if count >= high:
            return 0.95
        if count >= medium:
            return 0.55 + 0.35 * ((count - medium) / max(1, high - medium))
        return 0.0

    def score_image(self, image_path: str | Path) -> List[CandidateLabel]:
        self._load()
        if not self.available or self.model is None:
            return []
        try:
            results = self.model.predict(str(image_path), conf=self.confidence, verbose=False)
        except Exception as exc:  # pragma: no cover - optional runtime.
            self.error = f"YOLO prediction failed: {exc}"
            return []

        vehicle_count = 0
        person_count = 0
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "cls", None) is None:
                continue
            classes = boxes.cls.detach().cpu().tolist()
            for class_id in classes:
                name = str(names.get(int(class_id), "")).lower()
                if name in self.VEHICLE_CLASS_NAMES:
                    vehicle_count += 1
                elif name in self.PERSON_CLASS_NAMES:
                    person_count += 1

        candidates: List[CandidateLabel] = []
        vehicle_score = self._score_count(vehicle_count, self.vehicle_count_medium, self.vehicle_count_high)
        if vehicle_score > 0:
            _add_candidate(
                candidates,
                name="high_vehicle_density",
                score=vehicle_score,
                label_type="effect",
                evidence=f"YOLO counted {vehicle_count} vehicles.",
                method="ultralytics-yolo",
                modality="image",
            )
        person_score = self._score_count(person_count, self.person_count_medium, self.person_count_high)
        if person_score > 0:
            _add_candidate(
                candidates,
                name="high_pedestrian_density",
                score=person_score,
                label_type="effect",
                evidence=f"YOLO counted {person_count} people.",
                method="ultralytics-yolo",
                modality="image",
            )
        return candidates


@dataclass
class ImageCameraState:
    """State for cheap per-camera image prefiltering.

    The signature is a small grayscale thumbnail stored as bytes.  It is used
    only to decide whether an expensive image model should run; it is not an
    observation label by itself.
    """

    last_signature: Optional[bytes] = None
    last_signature_timestamp: Optional[datetime] = None
    first_signature_timestamp: Optional[datetime] = None
    last_model_timestamp: Optional[datetime] = None
    frames_seen: int = 0
    model_runs: int = 0
    skipped_no_change: int = 0


_IMAGE_PREFILTER_TIME_FORMATS = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y%m%d_%H%M%S",
    "%Y%m%d%H%M%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
]


def parse_image_prefilter_datetime(value: Any) -> Optional[datetime]:
    """Parse report timestamps for image heartbeat gating.

    Naive datetimes are treated as local/stream time and compared as naive
    datetimes.  Timezone-aware values are converted to UTC and then made naive
    so the rest of the pipeline can keep using local ISO strings.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        iso_text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso_text)
        except ValueError:
            dt = None  # type: ignore[assignment]
            for fmt in _IMAGE_PREFILTER_TIME_FORMATS:
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0)


def report_timestamp_for_prefilter(report_or_metadata: Mapping[str, Any]) -> datetime:
    """Return the best timestamp available for prefilter state comparisons."""
    data = report_or_metadata.get("data") if isinstance(report_or_metadata.get("data"), Mapping) else {}
    candidates = [
        report_or_metadata.get("timestamp"),
        report_or_metadata.get("report_date"),
        data.get("timestamp") if isinstance(data, Mapping) else None,
        data.get("time") if isinstance(data, Mapping) else None,
    ]
    for value in candidates:
        dt = parse_image_prefilter_datetime(value)
        if dt is not None:
            return dt
    return datetime.utcnow().replace(microsecond=0)


def mean_abs_signature_difference(a: bytes, b: bytes) -> float:
    """Mean absolute pixel difference between two equal-length grayscale signatures."""
    if not a or not b:
        return float("inf")
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    return sum(abs(a[i] - b[i]) for i in range(n)) / float(n)


def image_signature_bytes(image_path: str | Path, *, resize_size: int = 64) -> bytes:
    """Return a cheap grayscale thumbnail signature for no-change filtering."""
    from PIL import Image  # type: ignore

    size = max(8, int(resize_size))
    with Image.open(image_path) as image:
        image = image.convert("L").resize((size, size))
        return image.tobytes()


def cheap_image_interest_metrics(image_path: str | Path, *, resize_size: int = 48) -> Dict[str, Any]:
    """Return cheap image statistics used to decide whether CLIP is worth running.

    These are deliberately simple and fast. They are not final labels; they only
    decide whether an expensive image model should inspect the frame.
    """
    from PIL import Image, ImageStat  # type: ignore

    size = max(16, int(resize_size))
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((size, size))
        gray = rgb.convert("L")
        stat = ImageStat.Stat(gray)
        gray_mean = float(stat.mean[0])
        gray_std = float(stat.stddev[0])
        pixels = list(rgb.getdata())

    n = max(1, len(pixels))
    orange = 0
    dark = 0
    low_sat = 0
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        if r >= 125 and r > g * 1.10 and r > b * 1.35 and (r - b) >= 45:
            orange += 1
        if mx <= 70:
            dark += 1
        if (mx - mn) <= 35:
            low_sat += 1

    return {
        "gray_mean": round(gray_mean, 4),
        "gray_std": round(gray_std, 4),
        "orange_fraction": round(orange / float(n), 6),
        "dark_fraction": round(dark / float(n), 6),
        "low_saturation_fraction": round(low_sat / float(n), 6),
    }


def cheap_image_is_visually_interesting(
    metrics: Mapping[str, Any],
    *,
    orange_fraction_min: float,
    dark_fraction_min: float,
    haze_std_max: float,
    haze_saturation_max: float,
) -> Tuple[bool, str]:
    """Cheap first-frame gate for obvious fire/smoke-like visual cues."""
    orange = float(metrics.get("orange_fraction") or 0.0)
    dark = float(metrics.get("dark_fraction") or 0.0)
    gray_std = float(metrics.get("gray_std") or 0.0)
    low_sat = float(metrics.get("low_saturation_fraction") or 0.0)

    if orange >= float(orange_fraction_min):
        return True, "orange_fire_like_pixels"
    if dark >= float(dark_fraction_min) and low_sat >= 0.45:
        return True, "dark_low_saturation_smoke_like_pixels"
    if gray_std <= float(haze_std_max) and low_sat * 255.0 >= float(haze_saturation_max):
        return True, "low_contrast_haze_like_frame"
    return False, "no_cheap_visual_interest"


class ImageAnomalyDetector:
    """CCTV/image anomaly detector with cheap no-change gating before CLIP/YOLO.

    Per-camera prefilter behavior:
      * first frame for a camera runs the image model;
      * changed frames run the image model;
      * unchanged frames skip the image model until the heartbeat is due;
      * heartbeat forces one image-model scan every `heartbeat_seconds`.

    The returned AnomalyResult always records whether the image model ran in
    diagnostics["image_prefilter"].  Downstream observation gating can treat a
    skipped image as filtered, while a CLIP/YOLO candidate can be sent onward to
    the full observation model.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        use_clip: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_min_score: float = 0.35,
        use_yolo: bool = False,
        yolo_model_name: str = "yolov8n.pt",
        max_labels: int = 8,
        use_no_change_filter: bool = True,
        heartbeat_seconds: float = 3600.0,
        change_threshold: float = 18.0,
        signature_resize_size: int = 48,
        run_on_first_frame: bool = False,
        fail_open_on_prefilter_error: bool = True,
    ) -> None:
        self.config = dict(config or load_anomaly_config())
        image_prefilter_config = dict(self.config.get("image_prefilter", {}) or {})

        self.use_clip = use_clip
        self.clip_min_score = clip_min_score
        self.max_labels = max_labels
        self.clip_scorer: Optional[ImageCLIPScorer] = None
        if use_clip:
            self.clip_scorer = ImageCLIPScorer(
                self.config.get("image_prompts", {}), model_name=clip_model_name
            )
        self.yolo_scorer: Optional[YOLOCountScorer] = None
        if use_yolo:
            self.yolo_scorer = YOLOCountScorer(model_name=yolo_model_name)

        self.use_no_change_filter = bool(image_prefilter_config.get("enabled", use_no_change_filter))
        self.heartbeat_seconds = float(image_prefilter_config.get("heartbeat_seconds", heartbeat_seconds))
        self.change_threshold = float(image_prefilter_config.get("change_threshold", change_threshold))
        self.signature_resize_size = int(image_prefilter_config.get("resize_size", signature_resize_size))
        self.run_on_first_frame = bool(image_prefilter_config.get("run_on_first_frame", run_on_first_frame))
        self.run_first_frame_if_visually_interesting = bool(
            image_prefilter_config.get("run_first_frame_if_visually_interesting", True)
        )
        self.visual_interest_orange_fraction_min = float(
            image_prefilter_config.get("visual_interest_orange_fraction_min", 0.004)
        )
        self.visual_interest_dark_fraction_min = float(
            image_prefilter_config.get("visual_interest_dark_fraction_min", 0.10)
        )
        self.visual_interest_low_contrast_haze_std_max = float(
            image_prefilter_config.get("visual_interest_low_contrast_haze_std_max", 18.0)
        )
        self.visual_interest_low_contrast_haze_saturation_max = float(
            image_prefilter_config.get("visual_interest_low_contrast_haze_saturation_max", 38.0)
        )
        self.fail_open_on_prefilter_error = bool(
            image_prefilter_config.get("fail_open_on_error", fail_open_on_prefilter_error)
        )
        self.camera_states: Dict[str, ImageCameraState] = defaultdict(ImageCameraState)

    def reset_state(self) -> None:
        """Clear per-camera image prefilter state."""
        self.camera_states.clear()

    @staticmethod
    def _label_type_from_image_label(label: str) -> str:
        return "effect" if label in DEFAULT_EFFECT_LABELS else "incident"

    @staticmethod
    def _camera_key_from_report(report: Mapping[str, Any], image_path: str | Path) -> str:
        data = report_data(report)
        return str(
            report.get("sensor_id")
            or data.get("sensor_id")
            or report.get("sensor_name")
            or Path(image_path).parent.name
            or "unknown_camera"
        )

    def _should_run_image_model(
        self,
        *,
        camera_key: str,
        image_path: str | Path,
        timestamp: datetime,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return whether CLIP/YOLO should run for this image report."""
        diagnostics: Dict[str, Any] = {
            "enabled": self.use_no_change_filter,
            "camera_key": camera_key,
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "heartbeat_seconds": self.heartbeat_seconds,
            "change_threshold_mean_abs_pixel": self.change_threshold,
            "signature_resize_size": self.signature_resize_size,
            "clip_yolo_will_run": True,
        }

        if not self.use_no_change_filter:
            diagnostics.update({"decision": "run", "reason": "prefilter_disabled"})
            return True, diagnostics

        state = self.camera_states[camera_key]
        state.frames_seen += 1

        try:
            signature = image_signature_bytes(image_path, resize_size=self.signature_resize_size)
            interest_metrics = cheap_image_interest_metrics(image_path, resize_size=self.signature_resize_size)
            visually_interesting, visual_interest_reason = cheap_image_is_visually_interesting(
                interest_metrics,
                orange_fraction_min=self.visual_interest_orange_fraction_min,
                dark_fraction_min=self.visual_interest_dark_fraction_min,
                haze_std_max=self.visual_interest_low_contrast_haze_std_max,
                haze_saturation_max=self.visual_interest_low_contrast_haze_saturation_max,
            )
            diagnostics["cheap_visual_interest"] = {
                **interest_metrics,
                "visually_interesting": visually_interesting,
                "reason": visual_interest_reason,
            }
        except Exception as exc:
            diagnostics["signature_error"] = repr(exc)
            if self.fail_open_on_prefilter_error:
                diagnostics.update({
                    "decision": "run",
                    "reason": "signature_failed_fail_open",
                    "clip_yolo_will_run": True,
                })
                return True, diagnostics
            diagnostics.update({
                "decision": "skip",
                "reason": "signature_failed_fail_closed",
                "clip_yolo_will_run": False,
            })
            return False, diagnostics

        if state.last_signature is None:
            state.last_signature = signature
            state.last_signature_timestamp = timestamp
            state.first_signature_timestamp = timestamp
            if self.run_on_first_frame:
                diagnostics.update({
                    "decision": "run",
                    "reason": "first_frame_for_camera",
                    "frames_seen_for_camera": state.frames_seen,
                    "clip_yolo_will_run": True,
                })
                return True, diagnostics
            if self.run_first_frame_if_visually_interesting and visually_interesting:
                diagnostics.update({
                    "decision": "run",
                    "reason": f"first_frame_{visual_interest_reason}",
                    "frames_seen_for_camera": state.frames_seen,
                    "clip_yolo_will_run": True,
                })
                return True, diagnostics
            diagnostics.update({
                "decision": "skip",
                "reason": "first_frame_skipped_no_cheap_visual_interest",
                "frames_seen_for_camera": state.frames_seen,
                "clip_yolo_will_run": False,
            })
            return False, diagnostics

        mean_abs_diff = mean_abs_signature_difference(signature, state.last_signature)
        heartbeat_elapsed_seconds: Optional[float] = None
        if state.last_model_timestamp is not None:
            heartbeat_elapsed_seconds = max(0.0, (timestamp - state.last_model_timestamp).total_seconds())
            heartbeat_due = heartbeat_elapsed_seconds >= self.heartbeat_seconds
        elif state.first_signature_timestamp is not None:
            # The model has never run for this camera. Count from its first
            # frame: the previous frame is always one interval ago, so counting
            # from it would never reach the heartbeat.
            heartbeat_elapsed_seconds = max(0.0, (timestamp - state.first_signature_timestamp).total_seconds())
            heartbeat_due = heartbeat_elapsed_seconds >= self.heartbeat_seconds
        else:
            heartbeat_due = False

        changed = mean_abs_diff >= self.change_threshold

        # Always advance the signature so the gate compares consecutive frames.
        state.last_signature = signature
        state.last_signature_timestamp = timestamp

        diagnostics.update({
            "frames_seen_for_camera": state.frames_seen,
            "mean_abs_pixel_diff_from_previous_frame": round(float(mean_abs_diff), 4),
            "frame_changed": bool(changed),
            "heartbeat_due": bool(heartbeat_due),
            "seconds_since_last_image_model_run": (
                round(float(heartbeat_elapsed_seconds), 3)
                if heartbeat_elapsed_seconds is not None
                else None
            ),
        })

        if changed:
            diagnostics.update({
                "decision": "run",
                "reason": "frame_changed",
                "clip_yolo_will_run": True,
            })
            return True, diagnostics

        if heartbeat_due:
            diagnostics.update({
                "decision": "run",
                "reason": "hourly_heartbeat_due",
                "clip_yolo_will_run": True,
            })
            return True, diagnostics

        state.skipped_no_change += 1
        diagnostics.update({
            "decision": "skip",
            "reason": "no_change_and_heartbeat_not_due",
            "skipped_no_change_for_camera": state.skipped_no_change,
            "clip_yolo_will_run": False,
        })
        return False, diagnostics

    def _mark_image_model_run(self, camera_key: str, timestamp: datetime) -> None:
        state = self.camera_states[camera_key]
        state.last_model_timestamp = timestamp
        state.model_runs += 1

    def detect_image(
        self,
        image_path: str | Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        prefilter_diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> AnomalyResult:
        metadata = dict(metadata or {})
        candidates: List[CandidateLabel] = []
        diagnostics: Dict[str, Any] = {"image_path": str(image_path)}
        if prefilter_diagnostics is not None:
            diagnostics["image_prefilter"] = dict(prefilter_diagnostics)

        if self.clip_scorer is not None:
            scores = self.clip_scorer.score_image(image_path)
            if self.clip_scorer.error:
                diagnostics["clip_error"] = self.clip_scorer.error
            for label, (score, evidence) in scores.items():
                # CLIP prompt probabilities are relative over the prompt set; thresholds should be lower than
                # text cosine thresholds. Top labels still receive their raw probability as score.
                if score >= self.clip_min_score:
                    _add_candidate(
                        candidates,
                        name=label,
                        score=score,
                        label_type=self._label_type_from_image_label(label),
                        evidence=evidence,
                        method="clip-zero-shot",
                        modality="image",
                    )

        if self.yolo_scorer is not None:
            candidates.extend(self.yolo_scorer.score_image(image_path))
            if self.yolo_scorer.error:
                diagnostics["yolo_error"] = self.yolo_scorer.error

        candidates = dedupe_candidates(candidates)
        effects = [c for c in candidates if c.label_type == "effect"][: self.max_labels]
        incidents = [c for c in candidates if c.label_type == "incident"][: self.max_labels]
        anomaly_score = max([c.score for c in candidates], default=0.0)
        metadata = {**metadata, "modality": "image"}
        return AnomalyResult(
            **metadata,
            anomaly_score=anomaly_score,
            candidate_effects=effects,
            candidate_incidents=incidents,
            diagnostics=diagnostics,
        )

    def detect_report(self, report: Mapping[str, Any]) -> AnomalyResult:
        data = report_data(report)
        image_path = data.get("image_filepath") or report.get("image_filepath")
        metadata = result_metadata_from_report(report, "image")
        if not image_path:
            return AnomalyResult(
                **metadata,
                anomaly_score=0.0,
                diagnostics={"error": "No image_filepath found in report/data."},
            )

        camera_key = self._camera_key_from_report(report, image_path)
        timestamp = report_timestamp_for_prefilter({**dict(report), **metadata})
        should_run, prefilter_diagnostics = self._should_run_image_model(
            camera_key=camera_key,
            image_path=str(image_path),
            timestamp=timestamp,
        )

        if not should_run:
            return AnomalyResult(
                **metadata,
                anomaly_score=0.0,
                candidate_effects=[],
                candidate_incidents=[],
                diagnostics={
                    "image_path": str(image_path),
                    "image_prefilter": prefilter_diagnostics,
                },
            )

        result = self.detect_image(
            str(image_path),
            metadata=metadata,
            prefilter_diagnostics=prefilter_diagnostics,
        )
        self._mark_image_model_run(camera_key, timestamp)
        image_prefilter = result.diagnostics.setdefault("image_prefilter", {})
        if isinstance(image_prefilter, dict):
            image_prefilter["model_runs_for_camera"] = self.camera_states[camera_key].model_runs
        return result



# ---------------------------------------------------------------------------
# Time-series anomaly detection
# ---------------------------------------------------------------------------


@dataclass
class RollingRobustState:
    """Rolling robust stats state for one sensor variable."""

    window_size: int = 96
    min_samples: int = 8
    values: Deque[float] = field(default_factory=deque)
    last_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.values.maxlen is None:
            self.values = deque(self.values, maxlen=self.window_size)
        elif self.values.maxlen != self.window_size:
            self.values = deque(self.values, maxlen=self.window_size)

    def stats(self) -> Optional[Dict[str, float]]:
        if len(self.values) < self.min_samples:
            return None
        vals = list(self.values)
        median = statistics.median(vals)
        deviations = [abs(v - median) for v in vals]
        mad = statistics.median(deviations)
        mean = statistics.fmean(vals)
        # statistics.pstdev returns 0.0 for constant arrays.
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return {"median": median, "mad": mad, "mean": mean, "std": std}

    def score(self, value: float, *, direction: str = "both") -> Tuple[float, str]:
        details: List[str] = []
        stats = self.stats()
        score = 0.0
        if stats is not None:
            median = stats["median"]
            mad = stats["mad"]
            robust_scale = max(1e-6, 1.4826 * mad)
            signed_z = (value - median) / robust_scale
            if direction == "high":
                z = max(0.0, signed_z)
            elif direction == "low":
                z = max(0.0, -signed_z)
            else:
                z = abs(signed_z)
            # z≈3 is suspicious, z≈6 is very strong.
            score = max(score, clamp((z - 3.0) / 3.0))
            details.append(f"robust_z={signed_z:.2f} median={median:.3f} mad={mad:.3f}")
        if self.last_value is not None:
            delta = value - self.last_value
            details.append(f"delta={delta:.3f}")
        if not details:
            details.append(f"warmup n={len(self.values)}/{self.min_samples}")
        return score, ", ".join(details)

    def update(self, value: float) -> None:
        self.values.append(value)
        self.last_value = value


class RiverOnlineScorer:
    """Optional River online anomaly scorer, one model per sensor stream."""

    def __init__(self, *, seed: int = 42, window_size: int = 250) -> None:
        self.seed = seed
        self.window_size = window_size
        self.models: Dict[str, Any] = {}
        self.available = False
        self.error: Optional[str] = None
        try:
            from river import anomaly  # type: ignore

            self._river_anomaly = anomaly
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency.
            self._river_anomaly = None
            self.error = f"river unavailable: {exc}"

    def _model_for(self, stream_id: str) -> Any:
        if stream_id not in self.models:
            if not self.available or self._river_anomaly is None:
                return None
            self.models[stream_id] = self._river_anomaly.HalfSpaceTrees(
                n_trees=15,
                height=8,
                window_size=self.window_size,
                seed=self.seed,
            )
        return self.models[stream_id]

    def score_and_update(self, stream_id: str, features: Mapping[str, float]) -> Tuple[float, str]:
        if not self.available or not features:
            return 0.0, self.error or "river disabled"
        model = self._model_for(stream_id)
        if model is None:
            return 0.0, self.error or "river model unavailable"
        try:
            raw = float(model.score_one(dict(features)))
            model.learn_one(dict(features))
            # HalfSpaceTrees scores are not calibrated probabilities. This smooth squashing
            # makes them usable as a weak auxiliary signal.
            score = clamp(1.0 - math.exp(-max(0.0, raw)))
            return score, f"river_half_space_trees raw_score={raw:.4f}"
        except Exception as exc:  # pragma: no cover - optional runtime.
            self.error = f"river scoring failed: {exc}"
            return 0.0, self.error


def _traffic_sensor_type_matches(sensor_type: Any, patterns: Sequence[str]) -> bool:
    text = str(sensor_type or "").strip().lower()
    return any(pattern.lower() in text for pattern in patterns)


def _parse_optional_datetime_for_prefilter(value: Any) -> Optional[datetime]:
    return parse_image_prefilter_datetime(value)


def _traffic_gate_score(features: Mapping[str, float], cfg: Mapping[str, Any]) -> Tuple[float, str]:
    """Return a PeMS traffic anomaly score from speed/occupancy.

    This is intentionally looser than the earlier aggressive gate. The job of
    this function is to keep a reasonably broad pool of PeMS anomalies for the
    hourly top-K selector; it should filter clearly normal freeway flow, not
    decide the final candidate budget by itself.
    """
    speed = None
    for key in ("data_avg_speed", "avg_speed", "speed_mph"):
        if key in features:
            speed = float(features[key])
            break
    occupancy = None
    for key in ("data_avg_occupancy", "avg_occupancy", "occupancy"):
        if key in features:
            occupancy = float(features[key])
            break

    severe_speed = float(cfg.get("severe_speed_mph", 5.0))
    high_speed = float(cfg.get("high_speed_mph", 15.0))
    medium_speed = float(cfg.get("medium_speed_mph", 25.0))
    moderate_speed = float(cfg.get("moderate_speed_mph", 35.0))
    medium_occ = float(cfg.get("medium_occupancy", 0.45))
    high_occ = float(cfg.get("high_occupancy", 0.70))
    severe_occ = float(cfg.get("severe_occupancy", 0.85))
    moderate_combined_score = float(cfg.get("moderate_combined_score", 0.62))
    medium_combined_score = float(cfg.get("medium_combined_score", 0.72))

    score = 0.0
    reasons: List[str] = []

    if speed is not None:
        if speed <= severe_speed:
            score = max(score, 1.0)
            reasons.append(f"speed<={severe_speed:g}")
        elif speed <= high_speed:
            score = max(score, 0.88)
            reasons.append(f"speed<={high_speed:g}")
        elif speed <= medium_speed:
            if occupancy is None:
                # Low speed alone is still useful, but less trustworthy than a
                # speed+occupancy combination because detector noise or ramp
                # stations can produce isolated low speeds.
                score = max(score, 0.64)
                reasons.append(f"speed<={medium_speed:g}")
            elif occupancy >= medium_occ:
                score = max(score, medium_combined_score)
                reasons.append(f"speed<={medium_speed:g}_and_occupancy>={medium_occ:g}")
        elif occupancy is not None and speed <= moderate_speed and occupancy >= high_occ:
            score = max(score, moderate_combined_score)
            reasons.append(f"speed<={moderate_speed:g}_and_occupancy>={high_occ:g}")

    if occupancy is not None:
        if occupancy >= severe_occ:
            score = max(score, 0.95)
            reasons.append(f"occupancy>={severe_occ:g}")
        elif occupancy >= high_occ:
            score = max(score, 0.72)
            reasons.append(f"occupancy>={high_occ:g}")
        elif speed is not None and occupancy >= medium_occ and speed <= medium_speed:
            score = max(score, medium_combined_score)
            reasons.append(f"occupancy>={medium_occ:g}_and_speed<={medium_speed:g}")

    if not reasons:
        reasons.append("traffic_not_anomalous_enough")
    return clamp(score), ";".join(reasons)


class TimeSeriesAnomalyDetector:
    """Online anomaly detector for weather, air-quality, and traffic streams."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        window_size: int = 96,
        min_samples: int = 8,
        update_before_scoring: bool = False,
        use_river: bool = True,
        context_incident_max_score: float = 0.25,
        use_traffic_prefilter: bool = True,
    ) -> None:
        self.config = dict(config or load_anomaly_config())
        self.rules: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in self.config.get("time_series_rules", {}).items()
        }
        self.window_size = window_size
        self.min_samples = min_samples
        self.update_before_scoring = update_before_scoring
        self.context_incident_max_score = context_incident_max_score
        self.traffic_prefilter_config = dict(self.config.get("traffic_prefilter", {}) or {})
        self.use_traffic_prefilter = bool(self.traffic_prefilter_config.get("enabled", use_traffic_prefilter))
        self.traffic_pending_anomaly_timestamp: Dict[str, datetime] = {}
        self.traffic_last_candidate_timestamp: Dict[str, datetime] = {}
        self.states: Dict[Tuple[str, str], RollingRobustState] = defaultdict(
            lambda: RollingRobustState(window_size=self.window_size, min_samples=self.min_samples)
        )
        self.river = RiverOnlineScorer(window_size=max(window_size, 250)) if use_river else None
        self.text_detector = TextAnomalyDetector(config=self.config, use_embeddings=False)

    @staticmethod
    def _threshold_score(value: float, rule: Mapping[str, Any]) -> float:
        direction = str(rule.get("direction", "high"))
        medium = float(rule.get("medium", 0.0))
        high = float(rule.get("high", medium))
        if direction == "high":
            if value >= high:
                return 0.95
            if value >= medium:
                return 0.45 + 0.45 * ((value - medium) / max(1e-6, high - medium))
            return 0.0
        if direction == "low":
            # For low anomalies, high is the more severe/lower threshold.
            if value <= high:
                return 0.95
            if value <= medium:
                return 0.45 + 0.45 * ((medium - value) / max(1e-6, medium - high))
            return 0.0
        # both-sided threshold support is not used by defaults but kept extensible.
        return 0.0

    def _apply_rule(
        self,
        *,
        variable: str,
        value: float,
        score: float,
        evidence: str,
        modality: str = "timeseries",
    ) -> List[CandidateLabel]:
        rule = self.rules.get(variable) or self.rules.get(
            TIME_SERIES_FIELD_ALIASES.get(variable.lower(), "")
        )
        if not rule or score <= 0.0:
            return []
        candidates: List[CandidateLabel] = []
        context_only = bool(rule.get("context_only"))
        effective_incident_score = min(score, self.context_incident_max_score) if context_only else score
        for effect in rule.get("effects", []) or []:
            _add_candidate(
                candidates,
                name=str(effect),
                score=score,
                label_type="effect",
                evidence=f"{rule.get('evidence', 'Numeric anomaly')} {variable}={value:.3f}. {evidence}",
                method="rolling-threshold",
                modality=modality,
            )
        for incident in rule.get("incidents", []) or []:
            if effective_incident_score <= 0.0:
                continue
            _add_candidate(
                candidates,
                name=str(incident),
                score=effective_incident_score,
                label_type="incident",
                evidence=f"{rule.get('evidence', 'Numeric anomaly')} {variable}={value:.3f}. {evidence}",
                method="rolling-threshold-context" if context_only else "rolling-threshold",
                modality=modality,
            )
        return candidates

    def detect_row(
        self,
        row: Mapping[str, Any],
        *,
        sensor_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        timestamp: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        data = row.get("data") if isinstance(row.get("data"), Mapping) else row
        features = numeric_features_from_mapping(row)
        sensor_id = sensor_id or str(row.get("sensor_id") or data.get("sensor_id") or "unknown_sensor")
        sensor_type = sensor_type or str(row.get("sensor_type") or data.get("sensor_type") or "unknown")
        timestamp_value = timestamp or row.get("report_date") or row.get("timestamp") or data.get("timestamp") or data.get("time")
        if location is None:
            lat = data.get("latitude") or data.get("lat")
            lon = data.get("longitude") or data.get("lon")
            if _is_number(lat) and _is_number(lon):
                location = {"latitude": float(lat), "longitude": float(lon)}

        candidates: List[CandidateLabel] = []
        diagnostics: Dict[str, Any] = {"features": features, "state_window_size": self.window_size}

        for variable, value in features.items():
            rule = self.rules.get(variable) or self.rules.get(
                TIME_SERIES_FIELD_ALIASES.get(variable.lower(), "")
            )
            direction = str(rule.get("direction", "both")) if rule else "both"
            state = self.states[(sensor_id, variable)]
            if self.update_before_scoring:
                state.update(value)
            rolling_score, rolling_evidence = state.score(value, direction=direction)
            threshold_score = self._threshold_score(value, rule) if rule else 0.0
            combined_score = max(rolling_score, threshold_score)
            evidence_parts = [rolling_evidence]
            if threshold_score > 0:
                evidence_parts.append(f"threshold_score={threshold_score:.2f}")
            if rolling_score > 0:
                evidence_parts.append(f"rolling_score={rolling_score:.2f}")
            if rule:
                candidates.extend(
                    self._apply_rule(
                        variable=variable,
                        value=value,
                        score=combined_score,
                        evidence=", ".join(evidence_parts),
                    )
                )
            # If the variable has no label mapping but is unusual, preserve a diagnostic only.
            diagnostics[f"{variable}_rolling"] = {
                "score": round(rolling_score, 4),
                "evidence": rolling_evidence,
                "threshold_score": round(threshold_score, 4),
            }
            if not self.update_before_scoring:
                state.update(value)

        if self.river is not None:
            river_score, river_evidence = self.river.score_and_update(sensor_id, features)
            diagnostics["river"] = {"score": round(river_score, 4), "evidence": river_evidence}
        else:
            river_score = 0.0

        # Time-series rows often have a short description. Treat it as another text cue,
        # but with no embedding model by default to keep the numeric path fast.
        description_text = extract_text_for_detection(row)
        if description_text:
            text_result = self.text_detector.detect_text(
                description_text,
                metadata={
                    "report_id": str(row.get("report_id")) if row.get("report_id") else None,
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "timestamp": str(timestamp_value) if timestamp_value is not None else None,
                    "location": location,
                    "modality": "timeseries_text",
                },
            )
            # Downweight text descriptions in numeric sensor rows slightly. This prevents simulator
            # descriptions from overwhelming numeric evidence, while still supporting real descriptions.
            for item in text_result.candidate_effects + text_result.candidate_incidents:
                item.score *= 0.80
                item.method = f"timeseries-description/{item.method}"
                item.modality = "timeseries"
                candidates.append(item)

        candidates = dedupe_candidates(candidates)
        effects = [c for c in candidates if c.label_type == "effect"]
        incidents = [c for c in candidates if c.label_type == "incident"]
        anomaly_score = max([c.score for c in candidates] + [river_score], default=0.0)

        result = AnomalyResult(
            report_id=str(row.get("report_id")) if row.get("report_id") else None,
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            timestamp=str(timestamp_value) if timestamp_value is not None else None,
            location=location,
            modality="timeseries",
            anomaly_score=anomaly_score,
            candidate_effects=effects,
            candidate_incidents=incidents,
            diagnostics=diagnostics,
        )

        if self.use_traffic_prefilter and _traffic_sensor_type_matches(
            sensor_type,
            self.traffic_prefilter_config.get("sensor_type_patterns", ["pem", "pems", "traffic"]),
        ):
            result = self._apply_aggressive_traffic_prefilter(result, features, timestamp_value)

        return result

    def _apply_aggressive_traffic_prefilter(
        self,
        result: AnomalyResult,
        features: Mapping[str, float],
        timestamp_value: Any,
    ) -> AnomalyResult:
        cfg = self.traffic_prefilter_config
        min_score = float(cfg.get("min_candidate_score", 0.85))
        severe_bypass_score = float(cfg.get("severe_bypass_score", 0.98))
        require_consecutive = bool(cfg.get("require_consecutive", True))
        consecutive_window_seconds = float(cfg.get("consecutive_window_seconds", 7200.0))
        cooldown_seconds = float(cfg.get("candidate_cooldown_seconds", 21600.0))

        gate_score, gate_reason = _traffic_gate_score(features, cfg)
        timestamp = _parse_optional_datetime_for_prefilter(timestamp_value) or datetime.utcnow().replace(microsecond=0)
        sensor_key = str(result.sensor_id or "unknown_sensor")

        diag = {
            "enabled": True,
            "gate_score": round(gate_score, 4),
            "gate_reason": gate_reason,
            "min_candidate_score": min_score,
            "require_consecutive": require_consecutive,
            "candidate_cooldown_seconds": cooldown_seconds,
            "decision": "candidate",
        }

        if gate_score < min_score:
            diag.update({"decision": "filtered", "reason": "below_aggressive_traffic_gate"})
            result.candidate_effects = []
            result.candidate_incidents = []
            result.anomaly_score = 0.0
            result.diagnostics["traffic_prefilter"] = diag
            return result

        if require_consecutive and gate_score < severe_bypass_score:
            previous = self.traffic_pending_anomaly_timestamp.get(sensor_key)
            self.traffic_pending_anomaly_timestamp[sensor_key] = timestamp
            if previous is None or (timestamp - previous).total_seconds() > consecutive_window_seconds:
                diag.update({"decision": "filtered", "reason": "waiting_for_consecutive_traffic_anomaly"})
                result.candidate_effects = []
                result.candidate_incidents = []
                result.anomaly_score = 0.0
                result.diagnostics["traffic_prefilter"] = diag
                return result
            diag["consecutive_previous_timestamp"] = previous.isoformat(timespec="seconds")

        last = self.traffic_last_candidate_timestamp.get(sensor_key)
        if last is not None:
            elapsed = (timestamp - last).total_seconds()
            if elapsed < cooldown_seconds:
                diag.update({
                    "decision": "filtered",
                    "reason": "traffic_candidate_cooldown",
                    "seconds_since_last_candidate": round(float(elapsed), 3),
                })
                result.candidate_effects = []
                result.candidate_incidents = []
                result.anomaly_score = 0.0
                result.diagnostics["traffic_prefilter"] = diag
                return result

        self.traffic_last_candidate_timestamp[sensor_key] = timestamp
        result.anomaly_score = max(result.anomaly_score, gate_score)
        for item in result.candidate_effects + result.candidate_incidents:
            item.score = max(item.score, gate_score)
            item.method = f"aggressive-traffic-prefilter/{item.method}"
        result.diagnostics["traffic_prefilter"] = diag
        return result

    def detect_report(self, report: Mapping[str, Any]) -> AnomalyResult:
        metadata = result_metadata_from_report(report, "timeseries")
        return self.detect_row(
            report,
            sensor_id=metadata.get("sensor_id"),
            sensor_type=metadata.get("sensor_type"),
            timestamp=metadata.get("timestamp"),
            location=metadata.get("location"),
        )


# ---------------------------------------------------------------------------
# Unified multimodal detector
# ---------------------------------------------------------------------------


class MultimodalAnomalyDetector:
    """Unified detector with persistent time-series state.

    Instantiate this once per pipeline process so rolling statistics are kept per
    sensor. Reusing the same object is important for traffic/weather/air-quality
    anomaly scoring.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        use_text_embeddings: bool = True,
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_image_clip: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_yolo: bool = False,
        yolo_model_name: str = "yolov8n.pt",
        use_image_no_change_filter: bool = True,
        image_heartbeat_seconds: float = 3600.0,
        image_change_threshold: float = 18.0,
        image_signature_resize_size: int = 48,
        use_river: bool = True,
        use_traffic_prefilter: bool = True,
        time_window_size: int = 96,
        time_min_samples: int = 8,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("Pass either config or config_path, not both.")
        self.config = dict(config or load_anomaly_config(config_path))
        self.text = TextAnomalyDetector(
            self.config,
            use_embeddings=use_text_embeddings,
            embedding_model_name=text_embedding_model_name,
        )
        self.image = ImageAnomalyDetector(
            self.config,
            use_clip=use_image_clip,
            clip_model_name=clip_model_name,
            use_yolo=use_yolo,
            yolo_model_name=yolo_model_name,
            use_no_change_filter=use_image_no_change_filter,
            heartbeat_seconds=image_heartbeat_seconds,
            change_threshold=image_change_threshold,
            signature_resize_size=image_signature_resize_size,
        )
        self.timeseries = TimeSeriesAnomalyDetector(
            self.config,
            window_size=time_window_size,
            min_samples=time_min_samples,
            use_river=use_river,
            use_traffic_prefilter=use_traffic_prefilter,
        )

    def reset_state(self) -> None:
        """Clear online detector state while keeping loaded models/config.

        This is useful between replayed incidents/runs.  It resets rolling
        time-series baselines, River models, and the per-camera image
        no-change/heartbeat state.
        """
        try:
            self.timeseries.states.clear()
            self.timeseries.traffic_pending_anomaly_timestamp.clear()
            self.timeseries.traffic_last_candidate_timestamp.clear()
        except Exception:
            pass
        try:
            if self.timeseries.river is not None:
                self.timeseries.river.models.clear()
        except Exception:
            pass
        try:
            self.image.reset_state()
        except Exception:
            pass

    def reset(self) -> None:
        """Alias used by full_pipeline.py when resetting per-incident state."""
        self.reset_state()

    def detect_report(self, report: Mapping[str, Any]) -> AnomalyResult:
        modality = report_modality(report)
        if modality == "image":
            return self.image.detect_report(report)
        if modality == "text":
            return self.text.detect_report(report)
        if modality == "timeseries":
            return self.timeseries.detect_report(report)

        # Unknown reports may contain both text and numeric fields; run cheap detectors and merge.
        metadata = result_metadata_from_report(report, "unknown")
        metadata = {**metadata, "modality": "unknown"}
        result = AnomalyResult(**metadata)
        text = extract_text_for_detection(report)
        if text:
            result.merge(self.text.detect_text(text, metadata=metadata))
        if numeric_features_from_mapping(report):
            result.merge(self.timeseries.detect_report(report))
        if is_image_report(report):
            result.merge(self.image.detect_report(report))
        return result

    def detect_many(self, reports: Iterable[Mapping[str, Any]]) -> Iterable[AnomalyResult]:
        for report in reports:
            yield self.detect_report(report)


def anomaly_payload_for_observation_model(
    report: Mapping[str, Any],
    *,
    detector: Optional[MultimodalAnomalyDetector] = None,
    allowed_effects: Optional[Sequence[str]] = None,
    allowed_incidents: Optional[Sequence[str]] = None,
    min_effect_score: float = 0.30,
    min_incident_score: float = 0.30,
) -> Dict[str, Any]:
    """Convenience function for direct use inside observation_model.py.

    Note: For streaming time-series data, pass a persistent detector instead of
    relying on the default here, otherwise rolling state resets every call.
    """
    detector = detector or MultimodalAnomalyDetector()
    result = detector.detect_report(report)
    return result.to_observation_model_payload(
        allowed_effects=allowed_effects,
        allowed_incidents=allowed_incidents,
        min_effect_score=min_effect_score,
        min_incident_score=min_incident_score,
    )


# ---------------------------------------------------------------------------
# Optional CLI for smoke tests on JSONL reports or raw CSV rows
# ---------------------------------------------------------------------------


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _read_csv_rows(path: str | Path) -> Iterable[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("CSV input requires pandas: pip install pandas") from exc
    df = pd.read_csv(path)
    for row in df.to_dict(orient="records"):
        yield row


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run multimodal anomaly detection on JSONL reports or CSV rows.")
    parser.add_argument("input", help="Path to .jsonl or .csv input file")
    parser.add_argument("--config", help="Optional anomaly config JSON")
    parser.add_argument("--write-default-config", help="Write the default config JSON to this path and exit")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers text embeddings")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP image scoring")
    parser.add_argument("--yolo", action="store_true", help="Enable Ultralytics YOLO vehicle/person counts")
    parser.add_argument("--no-image-prefilter", action="store_true", help="Disable per-camera no-change/heartbeat image prefiltering")
    parser.add_argument("--image-heartbeat-seconds", type=float, default=3600.0, help="Force an image model scan at least this often per camera")
    parser.add_argument("--image-change-threshold", type=float, default=18.0, help="Mean absolute grayscale pixel-difference threshold for changed-frame detection")
    parser.add_argument("--image-signature-resize", type=int, default=48, help="Thumbnail side length used by the image no-change filter")
    parser.add_argument("--no-river", action="store_true", help="Disable River online anomaly scorer")
    parser.add_argument("--no-traffic-prefilter", action="store_true", help="Disable aggressive PeMS/traffic candidate filtering")
    args = parser.parse_args()

    if args.write_default_config:
        save_default_anomaly_config(args.write_default_config)
        print(f"Wrote default anomaly config to {args.write_default_config}")
        return 0

    detector = MultimodalAnomalyDetector(
        config_path=args.config,
        use_text_embeddings=not args.no_embeddings,
        use_image_clip=not args.no_clip,
        use_yolo=args.yolo,
        use_image_no_change_filter=not args.no_image_prefilter,
        image_heartbeat_seconds=args.image_heartbeat_seconds,
        image_change_threshold=args.image_change_threshold,
        image_signature_resize_size=args.image_signature_resize,
        use_river=not args.no_river,
        use_traffic_prefilter=not args.no_traffic_prefilter,
    )
    path = Path(args.input)
    if path.suffix.lower() == ".csv":
        rows = _read_csv_rows(path)
    else:
        rows = _read_jsonl(path)

    for row in rows:
        print(json.dumps(detector.detect_report(row).to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raise SystemExit(_main())
