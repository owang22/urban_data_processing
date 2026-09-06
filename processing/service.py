"""Run receiving, enrichment, and durable JSONL storage in one process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .enrichment.backends import OpenAIBackend, retrieve_article
from .enrichment.cache import EnrichmentCache
from .enrichment.enricher import Enricher, NoModelBackend
from .receiver.receiver import EnrichingHandler, JSONLHandler, serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Receive and enrich Urban Observations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-message-bytes", type=int, default=25_000_000)
    parser.add_argument("--anomaly-threshold", type=float, default=0.25)
    parser.add_argument(
        "--anomaly-only", action="store_true",
        default=os.environ.get("URBAN_ENRICHMENT_ANOMALY_ONLY", "").lower()
        in {"1", "true", "yes"},
    )
    parser.add_argument("--text-model", default="gpt-4o-mini")
    parser.add_argument("--vision-model", default="gpt-4o-mini")
    parser.add_argument("--openai-base-url")
    parser.add_argument(
        "--cache",
        default=os.environ.get("URBAN_ENRICHMENT_CACHE", "/cache/enrichment.sqlite3"),
    )
    args = parser.parse_args(argv)

    backend = NoModelBackend() if args.anomaly_only else OpenAIBackend(
        text_model=args.text_model,
        vision_model=args.vision_model,
        base_url=args.openai_base_url,
    )
    cache = EnrichmentCache(args.cache, version=Enricher.VERSION) if args.cache else None
    enricher = Enricher(
        backend,
        anomaly_threshold=args.anomaly_threshold,
        article_retriever=retrieve_article,
        cache=cache,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        handler = EnrichingHandler(JSONLHandler(stream), enricher)
        try:
            serve(args.host, args.port, handler, max_message_bytes=args.max_message_bytes)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
