import hashlib

from urban_observation_model import InlineFile, Observation, SCHEMA_VERSION
from processing.enrichment import Enricher
from processing.enrichment.cache import EnrichmentCache


class Result:
    def __init__(self, score): self.score = score
    def to_observation_model_payload(self):
        return {"anomaly": {"score": self.score, "modality": "test", "diagnostics": {}},
                "observed_effects": [], "possible_incidents": []}


class Detector:
    def __init__(self, score): self.score, self.reports = score, []
    def detect_report(self, report): self.reports.append(report); return Result(self.score)


class Backend:
    def __init__(self):
        self.text_calls = []; self.news_calls = []; self.image_calls = []; self.locations = []; self.coordinates = []
    def annotate_text(self, value):
        self.text_calls.append(value)
        return {"summary": "annotated", "location": {"text": "Los Angeles"},
                "incidents": [{"name": "fire", "score": .9}]}
    def annotate_news(self, value):
        self.news_calls.append(value)
        return {"summary": "news annotated", "location": {"text": "Los Angeles"},
                "incidents": [{"name": "2026 Downtown Los Angeles Fire", "score": .9}]}
    def annotate_image(self, content, media_type):
        self.image_calls.append((content, media_type)); return {"summary": "image annotated"}
    def geocode(self, location):
        self.locations.append(location); return {"latitude": 34.05, "longitude": -118.24}
    def reverse_geocode(self, latitude, longitude):
        self.coordinates.append((latitude, longitude))
        return {"formatted_address": "Los Angeles, CA, USA", "latitude": latitude,
                "longitude": longitude, "provider": "google", "provider_place_id": "place-1"}


def observation(source="citizen", data=None, files=None, observation_id="one"):
    return Observation.from_dict({"schema_version": SCHEMA_VERSION, "id": observation_id,
        "source": source, "time": "2026-09-02T12:00:00Z", "sensor": "sensor",
        "data": data or {}, "files": files or []})


def test_low_anomaly_skips_expensive_text_annotation():
    backend = Backend()
    result = Enricher(backend, detector=Detector(.1)).enrich(
        observation(data={"body": "ordinary update"}))
    assert backend.text_calls == []
    assert result.value["annotations"]["enrichment"]["status"] == "skipped_by_anomaly"


def test_low_anomaly_still_reverse_geocodes_coordinates():
    backend = Backend()
    item = observation(data={"body": "ordinary update"}).to_dict()
    item["latitude"], item["longitude"] = 34.05, -118.24

    result = Enricher(backend, detector=Detector(.1)).enrich(Observation.from_dict(item))

    assert backend.text_calls == []
    assert backend.coordinates == [(34.05, -118.24)]
    assert result.value["annotations"]["location"]["formatted_address"] == "Los Angeles, CA, USA"


def test_high_anomaly_uses_text_model_and_geocoder():
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8)).enrich(
        observation(data={"subject": "Fire reported downtown"}))
    annotations = result.value["annotations"]
    assert backend.text_calls == ["Fire reported downtown"]
    assert annotations["summary"] == "annotated"
    assert annotations["incidents"] == [{"name": "fire", "score": .9}]
    assert annotations["location"]["latitude"] == 34.05


def test_coordinate_only_observation_is_reverse_geocoded():
    class CoordinateOnlyBackend(Backend):
        def annotate_text(self, value):
            self.text_calls.append(value)
            return {"summary": "coordinate-only report"}

    backend = CoordinateOnlyBackend()
    item = observation(source="cctv", data={"description": "Smoke visible"}).to_dict()
    item["latitude"], item["longitude"] = 34.05, -118.24

    result = Enricher(backend, detector=Detector(.8)).enrich(Observation.from_dict(item))

    assert backend.coordinates == [(34.05, -118.24)]
    assert result.value["annotations"]["location"]["formatted_address"] == "Los Angeles, CA, USA"


def test_model_label_with_null_score_is_normalized():
    class NullScoreBackend(Backend):
        def annotate_text(self, value):
            return {"incidents": [{"name": "urban fire", "score": None}]}

    result = Enricher(NullScoreBackend(), detector=Detector(.8)).enrich(
        observation(data={"description": "Smoke is visible"}))

    assert result.value["annotations"]["incidents"] == [
        {"name": "urban fire", "score": 0.0}
    ]


def test_high_anomaly_sends_inline_image_to_vision_backend():
    content = b"jpeg bytes"
    inline = InlineFile("camera.jpg", "image/jpeg", len(content),
                        hashlib.sha256(content).hexdigest(), content)
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8)).enrich(
        observation(source="cctv", files=[inline.to_dict()]))
    assert backend.image_calls == [(content, "image/jpeg")]
    assert result.value["annotations"]["summary"] == "image annotated"


def test_image_vlm_clears_weak_detector_incident_and_sensor_location_wins():
    class ImageResult(Result):
        def to_observation_model_payload(self):
            return {"anomaly": {"score": .8, "modality": "image", "diagnostics": {}},
                    "observed_effects": [],
                    "possible_incidents": [{"name": "wildfire", "score": .8}]}

    class ImageDetector:
        def detect_report(self, report): return ImageResult(.8)

    class ImageBackend(Backend):
        def annotate_image(self, content, media_type):
            return {"summary": "ordinary mountain view", "location": {"text": "UC San Diego"}}
        def reverse_geocode(self, latitude, longitude):
            # Provider result coordinates may be an address centroid; the
            # original sensor coordinates must remain authoritative.
            return {"formatted_address": "Los Angeles, CA, USA", "latitude": 1, "longitude": 2}

    content = b"jpeg bytes"
    inline = InlineFile("camera.jpg", "image/jpeg", len(content),
                        hashlib.sha256(content).hexdigest(), content)
    item = observation(source="alertcalifornia", files=[inline.to_dict()]).to_dict()
    item["latitude"], item["longitude"] = 34.13, -118.32
    result = Enricher(ImageBackend(), detector=ImageDetector()).enrich(Observation.from_dict(item))
    annotations = result.value["annotations"]

    assert annotations["incidents"] == []
    assert annotations["summary"] == "ordinary mountain view"
    assert annotations["location"]["formatted_address"] == "Los Angeles, CA, USA"
    assert annotations["location"]["latitude"] == 34.13


def test_link_only_news_is_retrieved_before_detection_and_annotation():
    backend, detector = Backend(), Detector(.8)
    result = Enricher(backend, detector=detector,
                      article_retriever=lambda url: "Downloaded article body").enrich(
        observation(source="gdelt", data={"url": "https://example.test/story"}))
    assert backend.text_calls == []
    assert backend.news_calls == ["Downloaded article body"]
    assert result.value["annotations"]["incidents"] == []
    assert result.value["annotations"]["news_incidents"] == [
        {"name": "2026 Downtown Los Angeles Fire", "score": .9}
    ]
    assert result.value["annotations"]["article_retrieval"]["status"] == "ok"
    assert detector.reports[0]["sensor_type"] == "gdelt"


def test_synthetic_news_fields_use_news_annotation():
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8)).enrich(observation(
        source="news",
        data={"headline": "Fire contained", "article_text": "Crews remain on scene."},
    ))

    assert backend.text_calls == []
    assert backend.news_calls == ["Fire contained\n\nCrews remain on scene."]
    assert result.value["annotations"]["news_incidents"] == [
        {"name": "2026 Downtown Los Angeles Fire", "score": .9}
    ]


def test_force_bypasses_low_anomaly_gate():
    backend = Backend()
    result = Enricher(backend, detector=Detector(0)).enrich(
        observation(data={"body": "manual hypothesis"}), force=True)
    assert backend.text_calls == ["manual hypothesis"]
    enrichment = result.value["annotations"]["enrichment"]
    assert enrichment["status"] == "completed"
    assert enrichment["forced"] is True
    assert enrichment["version"] == Enricher.VERSION


def test_durable_cache_avoids_repeating_model_call_after_restart(tmp_path):
    item = observation(data={"body": "Fire reported"}, observation_id="cached")
    first_backend = Backend()
    first = Enricher(first_backend, detector=Detector(.8),
                     cache=EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION))
    first.enrich(item)
    assert len(first_backend.text_calls) == 1

    second_backend = Backend()
    second = Enricher(second_backend, detector=Detector(.8),
                      cache=EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION))
    result = second.enrich(item)
    assert second_backend.text_calls == []
    assert result.value["annotations"]["summary"] == "annotated"


def test_geocoding_is_reused_across_different_observations(tmp_path):
    cache_path = tmp_path / "cache.sqlite3"
    first_backend = Backend()
    Enricher(first_backend, detector=Detector(.8),
             cache=EnrichmentCache(cache_path, version=Enricher.VERSION)).enrich(
        observation(data={"body": "first fire"}, observation_id="first"))
    assert first_backend.locations == ["Los Angeles"]

    second_backend = Backend()
    Enricher(second_backend, detector=Detector(.8),
             cache=EnrichmentCache(cache_path, version=Enricher.VERSION)).enrich(
        observation(data={"body": "second fire"}, observation_id="second"))
    assert second_backend.text_calls == ["second fire"]
    assert second_backend.locations == []


def test_cached_result_without_coordinates_is_placed_without_a_new_model_call(tmp_path):
    value = {"schema_version": SCHEMA_VERSION, "id": "gdelt:9", "source": "gdelt",
             "time": "2025-01-07T19:00:00Z", "sensor": "gdelt",
             "data": {"headline": "Fire in Pacific Palisades"}, "files": []}
    cache = EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION)
    source = Observation.from_dict(value)
    cache.put(source, False, Observation.from_dict(
        {**value, "annotations": {"location": {"text": "Pacific Palisades"}}}))

    class NoGeocoderBackend(Backend):
        def geocode(self, location): self.locations.append(location); return None

    unplaced = Enricher(NoGeocoderBackend(), detector=Detector(.9), cache=cache).enrich(source)
    assert "latitude" not in unplaced.value["annotations"]["location"]

    backend = Backend()
    placed = Enricher(backend, detector=Detector(.9), cache=cache).enrich(source)
    assert placed.value["annotations"]["location"]["latitude"] == 34.05
    assert backend.news_calls == [] and backend.locations == ["Pacific Palisades"]
    # The placed result is saved, so the next run neither calls the model nor geocodes.
    again = Backend()
    Enricher(again, detector=Detector(.9), cache=cache).enrich(source)
    assert again.locations == [] and again.news_calls == []


def test_entities_are_not_geocoded_when_backend_opts_out(tmp_path):
    class Limited(Backend):
        geocode_entities = False
        def annotate_news(self, value):
            return {"location": {"text": "Los Angeles"},
                    "entities": [{"name": "LAFD", "location": "Los Angeles City Hall"}]}
    backend = Limited()
    value = {"schema_version": SCHEMA_VERSION, "id": "gdelt:10", "source": "gdelt",
             "time": "2025-01-07T19:00:00Z", "sensor": "gdelt",
             "data": {"headline": "Fire"}, "files": []}
    Enricher(backend, detector=Detector(.9)).enrich(Observation.from_dict(value))
    assert backend.locations == ["Los Angeles"]


def test_scores_on_other_scales_are_rescaled_to_unit_range():
    class ScaleBackend(Backend):
        def annotate_news(self, value):
            return {"effects": [{"name": "Increased travel time", "score": 8},
                                {"name": "road closure", "score": 0.4}],
                    "incidents": [{"name": "2025 Palisades Fire", "score": 95}]}
    annotations = Enricher(ScaleBackend(), detector=Detector(.8)).enrich(
        observation(source="news", data={"headline": "Fire"})).value["annotations"]
    assert annotations["effects"][0] == {"name": "Increased travel time", "score": .8, "raw_score": 8.0}
    assert annotations["news_incidents"][0]["score"] == .95


def test_cached_scores_on_other_scales_are_rescaled(tmp_path):
    cache = EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION)
    source = observation(data={"body": "Fire"})
    stored = source.to_dict()
    stored["annotations"] = {"effects": [{"name": "smoke", "score": 7.0}]}
    cache.put(source, False, Observation.from_dict(stored))
    annotations = Enricher(Backend(), detector=Detector(.8), cache=cache).enrich(
        source).value["annotations"]
    assert annotations["effects"] == [{"name": "smoke", "score": .7, "raw_score": 7.0}]


GKG_EXTRAS = ("<PAGE_PRECISEPUBTIMESTAMP>20250107061000</PAGE_PRECISEPUBTIMESTAMP>"
              "<PAGE_TITLE>Strongest winds in over a decade could increase fire risk | Newsday</PAGE_TITLE>")


def gdelt_story():
    return observation(source="gdelt", data={"url": "https://example.test/story",
                                             "provider_fields": ["1", "20250107091500", GKG_EXTRAS]})


def test_unchanged_page_is_used_as_downloaded():
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8), article_retriever=lambda url: (
        "Newsday\nStrongest winds in over a decade could increase fire risk\nForecasters say...")
    ).enrich(gdelt_story())
    assert backend.news_calls[0].startswith("Newsday")
    assert result.value["annotations"]["article_retrieval"] == {"status": "ok", "title_checked": True}


def test_rewritten_page_uses_archived_copy_from_before_observation():
    backend, asked = Backend(), []
    def archive(url, before):
        asked.append(before); return ("http://web.archive.org/web/1id_/x", "Windstorm warning text")
    result = Enricher(backend, detector=Detector(.8),
                      article_retriever=lambda url: "Palisades Fire destroys homes",
                      archive_retriever=archive).enrich(gdelt_story())
    assert asked == ["2026-09-02T12:00:00Z"]
    assert backend.news_calls == ["Windstorm warning text"]
    assert result.value["annotations"]["article_retrieval"]["status"] == "archived"


def test_rewritten_page_without_archive_uses_gdelt_title_only():
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8),
                      article_retriever=lambda url: "Palisades Fire destroys homes",
                      archive_retriever=lambda url, before: None).enrich(gdelt_story())
    assert backend.news_calls == [
        "Strongest winds in over a decade could increase fire risk | Newsday"]
    assert result.value["annotations"]["article_retrieval"]["status"] == "page_changed_title_only"


def test_cached_summary_of_rewritten_page_is_redone(tmp_path):
    cache = EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION)
    source = gdelt_story()
    stored = source.to_dict()
    stored["annotations"] = {"summary": "Palisades Fire destroys homes",
                             "article_retrieval": {"status": "ok"},
                             "enrichment": {"status": "completed"}}
    cache.put(source, False, Observation.from_dict(stored))
    backend = Backend()
    result = Enricher(backend, detector=Detector(.8), cache=cache,
                      article_retriever=lambda url: "Palisades Fire destroys homes",
                      archive_retriever=lambda url, before: None).enrich(source)
    assert len(backend.news_calls) == 1
    assert result.value["annotations"]["article_retrieval"]["status"] == "page_changed_title_only"


def test_cached_summary_of_unchanged_page_is_kept_and_marked(tmp_path):
    cache = EnrichmentCache(tmp_path / "cache.sqlite3", version=Enricher.VERSION)
    source = gdelt_story()
    stored = source.to_dict()
    stored["annotations"] = {"summary": "windstorm", "article_retrieval": {"status": "ok"},
                             "enrichment": {"status": "completed"}}
    cache.put(source, False, Observation.from_dict(stored))
    backend, fetches = Backend(), []
    def fetch(url):
        fetches.append(url); return "Strongest winds in over a decade could increase fire risk"
    enricher = Enricher(backend, detector=Detector(.8), cache=cache, article_retriever=fetch)
    assert enricher.enrich(source).value["annotations"]["summary"] == "windstorm"
    assert Enricher(backend, detector=Detector(.8), cache=cache,
                    article_retriever=fetch).enrich(source).value["annotations"]["summary"] == "windstorm"
    assert backend.news_calls == [] and len(fetches) == 1
