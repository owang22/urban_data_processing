"""Optional external services used only after the anomaly gate passes."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


# USD per million tokens (input, cached input, output). A model missing here is
# refused while a budget is set, so spending can never go uncounted.
OPENAI_PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


class BudgetExceeded(RuntimeError):
    """The configured OpenAI spending limit has been reached."""


class UsageLedger:
    """Append-only record of OpenAI token usage with a hard spending limit.

    The total is rebuilt from the file on startup, so restarting the service
    does not reset what has already been spent.
    """

    def __init__(self, path: str | Path, *, budget_usd: float):
        self.path, self.budget_usd = Path(path), float(budget_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.spent_usd = 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.spent_usd += float(json.loads(line).get("cost_usd", 0.0))

    @staticmethod
    def cost(model: str, usage) -> float:
        if model not in OPENAI_PRICES:
            raise BudgetExceeded(f"no price known for model {model!r}; refusing to call it")
        price_in, price_cached, price_out = OPENAI_PRICES[model]
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return ((prompt - cached) * price_in + cached * price_cached + completion * price_out) / 1_000_000

    def check(self, model: str) -> None:
        if model not in OPENAI_PRICES:
            raise BudgetExceeded(f"no price known for model {model!r}; refusing to call it")
        with self.lock:
            if self.spent_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"OpenAI budget of ${self.budget_usd:.2f} reached (spent ${self.spent_usd:.4f})")

    def record(self, model: str, kind: str, usage) -> float:
        cost = self.cost(model, usage)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model, "kind": kind,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cost_usd": round(cost, 8),
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self.spent_usd += cost
        return cost


def _object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start: raise ValueError("model did not return a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict): raise ValueError("model response is not an object")
    return value


class OpenAIBackend:
    """Factual text/image annotations with optional Google geocoding."""
    def __init__(self, *, api_key=None, text_model="gpt-4o-mini", vision_model="gpt-4o-mini",
                 base_url=None, google_api_key=None, ledger: UsageLedger | None = None, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.client = client
        self.text_model, self.vision_model = text_model, vision_model
        self.provider = "openai"
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_PLACES_API_KEY")
        self.ledger = ledger

    def _complete(self, model, kind, messages):
        # JSON mode keeps the reply parseable; one retry covers the rare reply
        # that is still cut off or malformed.
        for attempt in range(2):
            if self.ledger is not None: self.ledger.check(model)
            response = self.client.chat.completions.create(
                model=model, temperature=0, messages=messages,
                response_format={"type": "json_object"})
            if self.ledger is not None and response.usage is not None:
                self.ledger.record(model, kind, response.usage)
            try:
                return _object(response.choices[0].message.content or "")
            except ValueError:
                if attempt == 1: raise

    @staticmethod
    def prompt():
        return ("Return one factual JSON object that completes all ordinary enrichment in one call. "
                "Allowed keys: summary, event, location, entities, relations, effects, incidents. "
                "event may contain name, type, start_time, end_time, and description. location may "
                "contain only text. entities are arrays of {name,type,description,location}. relations "
                "are arrays of {subject,predicate,object}. effects/incidents are arrays of {name,score}, "
                "where score is a number from 0 to 1. "
                "Use null or omit a field when unsupported. Do not invent facts.")

    @classmethod
    def news_prompt(cls):
        return (
            cls.prompt()
            + " For a news article, incidents must describe specific real-world event instances, "
              "not generic categories. Use a concise incident name that distinguishes the event "
              "with its supported event kind and place, plus a date or year when the article "
              "supports one (for example, '2026 Los Angeles Downtown Warehouse Fire'). The event "
              "type may remain generic. Do not create an incident when the article does not "
              "support a particular event."
        )

    @classmethod
    def image_prompt(cls):
        return (
            cls.prompt()
            + " For a camera frame, report only effects and incidents visible in the frame. Do not "
              "report stopped or slow traffic when no vehicles are visible, and do not infer an "
              "incident from darkness, glare, rain on the lens, or an empty road. Give a score "
              "above 0.5 only when the evidence is clearly visible."
        )

    def annotate_text(self, text):
        return self._complete(self.text_model, "text",
            [{"role": "system", "content": self.prompt()}, {"role": "user", "content": text}])

    def annotate_news(self, text):
        return self._complete(self.text_model, "news",
            [{"role": "system", "content": self.news_prompt()}, {"role": "user", "content": text}])

    def annotate_image(self, content, media_type):
        url = f"data:{media_type};base64,{base64.b64encode(content).decode('ascii')}"
        return self._complete(self.vision_model, "image",
            [{"role": "user", "content": [{"type": "text", "text": self.image_prompt()},
                                          {"type": "image_url", "image_url": {"url": url}}]}])

    @property
    def geocode_entities(self):
        # The free OpenStreetMap fallback allows one request per second, so it
        # only places each report's main location, not every entity it names.
        return bool(self.google_api_key)

    def geocode(self, location):
        if not self.google_api_key: return nominatim_geocode(location)
        import requests
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                                params={"address": location, "key": self.google_api_key}, timeout=20)
        response.raise_for_status(); results = response.json().get("results") or []
        if not results: return None
        result = results[0]; point = result["geometry"]["location"]
        return {"formatted_address": result.get("formatted_address"), "latitude": point["lat"],
                "longitude": point["lng"], "provider": "google", "provider_place_id": result.get("place_id")}

    def reverse_geocode(self, latitude, longitude):
        if not self.google_api_key: return None
        import requests
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                                params={"latlng": f"{latitude},{longitude}", "key": self.google_api_key},
                                timeout=20)
        response.raise_for_status(); results = response.json().get("results") or []
        if not results: return None
        result = results[0]; point = result["geometry"]["location"]
        return {"formatted_address": result.get("formatted_address"), "latitude": point["lat"],
                "longitude": point["lng"], "provider": "google", "provider_place_id": result.get("place_id")}


_nominatim_lock = threading.Lock()
_nominatim_last = 0.0


def nominatim_geocode(location: str, *, min_interval: float = 1.1):
    """Free OpenStreetMap geocoding, throttled to Nominatim's one request per second.

    Failures return None so that a missing place never fails an observation.
    """
    global _nominatim_last
    import requests
    with _nominatim_lock:
        wait = _nominatim_last + min_interval - time.monotonic()
        if wait > 0: time.sleep(wait)
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": "urban-observation-enrichment/0.1 (UCLA NESL research demo)"},
                timeout=20)
            response.raise_for_status(); results = response.json()
        except (requests.RequestException, ValueError):
            return None
        finally:
            _nominatim_last = time.monotonic()
    if not results: return None
    result = results[0]
    return {"formatted_address": result.get("display_name"), "latitude": float(result["lat"]),
            "longitude": float(result["lon"]), "provider": "openstreetmap",
            "provider_place_id": f"{result.get('osm_type')}:{result.get('osm_id')}"}


def _readable(page: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page, "html.parser")
    # Headlines often sit inside <header>, which is dropped below with the site
    # chrome; keep them so the text still names the story.
    og_title = soup.find("meta", attrs={"property": "og:title"})
    heading = soup.find("h1")
    headlines = [soup.title.get_text(" ", strip=True) if soup.title else "",
                 og_title.get("content", "").strip() if og_title else "",
                 heading.get_text(" ", strip=True) if heading else ""]
    headlines = list(dict.fromkeys(line for line in headlines if line))
    for item in soup(["script", "style", "nav", "footer", "header"]): item.decompose()
    text = "\n".join(part.strip() for part in soup.get_text("\n").splitlines() if part.strip())
    if not text: raise ValueError("article contained no readable text")
    return "\n".join(headlines + [text])


def retrieve_article(url: str) -> str:
    import requests
    response = requests.get(url, headers={"User-Agent": "urban-observation-enrichment/0.1"}, timeout=20)
    response.raise_for_status()
    return _readable(response.text)


def retrieve_archived_article(url: str, before: str) -> tuple[str, str] | None:
    """The Internet Archive copy of a page saved at or before an ISO time.

    Returns (archive URL, readable text), or None when no copy exists.
    """
    import requests
    stamp = "".join(ch for ch in before if ch.isdigit())[:14]
    response = requests.get("http://web.archive.org/cdx/search/cdx", timeout=30, params={
        "url": url, "to": stamp, "output": "json", "filter": "statuscode:200", "limit": "-1"})
    response.raise_for_status()
    rows = response.json() if response.text.strip() else []
    if len(rows) < 2: return None
    snapshot = f"http://web.archive.org/web/{rows[-1][1]}id_/{url}"
    page = requests.get(snapshot, headers={"User-Agent": "urban-observation-enrichment/0.1"}, timeout=30)
    page.raise_for_status()
    return snapshot, _readable(page.text)
