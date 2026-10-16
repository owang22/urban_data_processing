"""Small synchronous HTTP process for shared enrichment."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from urllib.parse import parse_qs, urlparse

from urban_observation_model import Observation, ObservationValidationError
from .backends import OpenAIBackend, retrieve_archived_article, retrieve_article
from .cache import EnrichmentCache
from .enricher import Enricher, NoModelBackend


def handler_class(enricher: Enricher):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/enrich": self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                observation = Observation.from_json(self.rfile.read(length).decode("utf-8"))
                force = parse_qs(parsed.query).get("force", ["false"])[0].lower() == "true"
                result = enricher.enrich(observation, force=force).to_json().encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result))); self.end_headers(); self.wfile.write(result)
            except (ValueError, ObservationValidationError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, format, *args):
            return
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared anomaly-gated observation enrichment service")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--anomaly-threshold", type=float, default=0.25)
    parser.add_argument("--anomaly-only", action="store_true",
                        default=os.environ.get("URBAN_ENRICHMENT_ANOMALY_ONLY", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--text-model", default="gpt-4o-mini"); parser.add_argument("--vision-model", default="gpt-4o-mini")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--cache", default=os.environ.get("URBAN_ENRICHMENT_CACHE", "/cache/enrichment.sqlite3"))
    args = parser.parse_args(argv)
    backend = NoModelBackend() if args.anomaly_only else OpenAIBackend(
        text_model=args.text_model, vision_model=args.vision_model, base_url=args.openai_base_url)
    cache = EnrichmentCache(args.cache, version=Enricher.VERSION) if args.cache else None
    enricher = Enricher(backend, anomaly_threshold=args.anomaly_threshold,
                        article_retriever=retrieve_article,
                        archive_retriever=retrieve_archived_article, cache=cache)
    server = ThreadingHTTPServer((args.host, args.port), handler_class(enricher))
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
