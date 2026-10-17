from datetime import datetime, timedelta, timezone
import json

from explorer.index import ExplorerIndex, categories, gkg_location, in_bbox


def _frame(root, day, name, stamp, bearing, reaimed=False):
    folder = root / "alertcalifornia" / day / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stamp}.jpg").write_bytes(b"\xff\xd8jpeg")
    (folder / f"{stamp}.json").write_text(json.dumps({
        "camera_id": "Axis-Test", "source": "latest-frame", "reaimed_since_last_run": reaimed,
        "properties": {"tilt_current": 1.0, "zoom_current": 1.0, "fov": 60}}))
    (folder / f"{stamp}.location").write_text(f"34.05,-118.8,{bearing}")


def test_gkg_location_prefers_headline_place_over_generic_los_angeles():
    locations = ("3#Los Angeles, California, United States#US#USCA#34.0522#-118.244#1662328;"
                 "3#Azusa, California, United States#US#USCA#34.13#-117.90#1;"
                 "3#Malibu, California, United States#US#USCA#34.03#-118.78#2;"
                 "3#Paris, Ile-de-France, France#FR#FR00#48.8#2.3#3")
    fields = [""] * 9 + [locations]
    assert gkg_location(fields)[2] == "Azusa, California"
    assert gkg_location(fields, "Malibu fire grows")[2] == "Malibu, California"
    only_la = [""] * 9 + ["3#Los Angeles, California, United States#US#USCA#34.0522#-118.244#1"]
    assert gkg_location(only_la)[2] == "Los Angeles, California"
    assert gkg_location([""] * 9 + ["2#California, United States#US#USCA#36.17#-119.7#CA"]) is None


def test_categories_match_whole_themes_only():
    assert categories("DISASTER_FIRE;EVACUATION") == ["fire", "evacuation"]
    assert categories("WB_2131_EMPLOYABILITY_SKILLS_AND_JOBS;CRISISLEX_CRISISLEXREC") == []


def test_in_bbox():
    assert in_bbox(34.03, -118.78)
    assert not in_bbox(37.77, -122.42)
    assert not in_bbox(None, -118.0)


def test_snapshot_reports_latest_frame_and_reaim(tmp_path):
    _frame(tmp_path, "20261016", "Test Cam", "20261016100000", 90)
    _frame(tmp_path, "20261016", "Test Cam", "20261016101500", 200, reaimed=True)
    index = ExplorerIndex(tmp_path, tmp_path, days=3)
    index.refresh()
    at = datetime(2026, 10, 16, 17, 20, tzinfo=timezone.utc)  # 10:20 in Los Angeles
    cameras = index.snapshot(at, 1)["alertcalifornia"]
    assert len(cameras) == 1
    camera = cameras[0]
    assert camera["bearing"] == 200 and camera["reaimed"] and camera["bearings"] == [90, 200]
    assert camera["frame"] == "alertcalifornia/20261016/Test Cam/20261016101500.jpg"
    # Before the re-aim, only the first frame is in view.
    early = index.snapshot(at - timedelta(minutes=10), 1)["alertcalifornia"][0]
    assert early["bearing"] == 90 and not early["reaimed"]
    assert len(index.frames("alertcalifornia", "Test Cam", at - timedelta(hours=1), at)) == 2


def test_frame_path_stays_inside_data_root(tmp_path):
    _frame(tmp_path / "raw", "20261016", "Cam", "20261016100000", 0)
    (tmp_path / "secret.jpg").write_bytes(b"x")
    index = ExplorerIndex(tmp_path / "raw", tmp_path)
    assert index.frame_path("alertcalifornia/20261016/Cam/20261016100000.jpg") is not None
    assert index.frame_path("../secret.jpg") is None
    assert index.frame_path("alertcalifornia/20261016/Cam/20261016100000.json") is None


def test_merge_copies_counts_outlets():
    news = [{"kind": "gkg", "title": "Fire burns homes", "domain": "a.com", "url": "https://a.com/1"},
            {"kind": "gkg", "title": "Fire  burns homes", "domain": "b.com", "url": "https://b.com/1"},
            {"kind": "events", "title": "Assault (c.com)", "domain": "c.com", "url": "https://c.com/1"}]
    merged = ExplorerIndex.merge_copies(news)
    assert [(n["domain"], n["outlets"]) for n in merged] == [("a.com", 2), ("c.com", 1)]


def test_news_fires_reads_largest_acreage_and_skips_comparisons():
    from explorer.fires import news_fires
    found = news_fires("Bryson Fire grows",
                       "The Bryson Fire was reported at two acres. The Bryson fire grew to 101.5 acres. "
                       "The Bryson Fire is at about 20 percent containment. "
                       "The Bryson and Peak fires together burned 500 acres. The Peak Fire burned 43 acres.")
    assert found["bryson"] == {"acres": 101.5, "contained": 20}
    assert found["peak"] == {"acres": 43.0}


def test_fire_snapshot_merges_feeds_as_of_time(tmp_path):
    from explorer.feeds import FeedStore
    from explorer.fires import fire_snapshot
    feeds = FeedStore(tmp_path)
    (tmp_path / "calfire").mkdir()
    (tmp_path / "calfire" / "20261016T120000Z.json").write_text(json.dumps([
        {"Name": "Bryson Fire ", "Latitude": 34.09, "Longitude": -118.84, "AcresBurned": 147, "PercentContained": 17,
         "Started": "2026-10-16T00:00:49Z", "Url": "https://www.fire.ca.gov/x"},
        {"Name": "Far Fire", "Latitude": 40.0, "Longitude": -121.0, "AcresBurned": 9, "Started": "2026-10-16T00:00:00Z"}]))
    (tmp_path / "incidents").mkdir()
    (tmp_path / "incidents" / "20261016T120000Z.geojson").write_text(json.dumps({"features": [
        {"properties": {"IncidentName": "BRYSON", "IncidentSize": 200, "FireDiscoveryDateTime": 1792107679000},
         "geometry": {"coordinates": [-118.84, 34.09]}},
        {"properties": {"IncidentName": "AVA", "IncidentSize": None, "FireDiscoveryDateTime": 1792107679000},
         "geometry": {"coordinates": [-118.76, 34.14]}}]}))
    at = datetime(2026, 10, 16, 13, tzinfo=timezone.utc)
    fires = {f["key"]: f for f in fire_snapshot(feeds, at)["fires"]}
    assert set(fires) == {"bryson", "ava"}
    assert fires["bryson"]["acres"] == 147 and fires["bryson"]["source"] == "CAL FIRE"
    assert fire_snapshot(feeds, datetime(2026, 10, 16, 11, tzinfo=timezone.utc))["fires"] == []


CHP_XML = b"""<?xml version="1.0" ?><State><Center ID = "LAHB"><Dispatch ID = "LACC">
<Log ID = "261017LA0005"><LogTime>"Oct 17 2026 12:00AM"</LogTime><LogType>"1179-Trfc Collision-1141 Enrt"</LogType>
<Location>"I10 W / N Azusa Ave"</Location><LocationDesc>"WB JEO"</LocationDesc><Area>"Baldwin Park"</Area>
<LATLON>"34072223:117907517"</LATLON><LogDetails><details><DetailTime>"Oct 17 2026 12:00AM"</DetailTime>
<IncidentDetail>"[1] 2 VEHS"</IncidentDetail></details><units><UnitTime>"x"</UnitTime></units></LogDetails></Log>
<Log ID = "261017LA0006"><LogTime>"Oct 17 2026 12:01AM"</LogTime><LogType>"1125-Traffic Hazard"</LogType>
<LATLON>"38683821:121456859"</LATLON></Log></Dispatch></Center></State>"""


def test_chp_calls_are_clipped_and_read_as_of_time(tmp_path):
    from explorer.feeds import FeedStore, _chp
    calls = json.loads(_chp(CHP_XML, None))
    assert len(calls) == 1  # the Sacramento call is outside the region
    call = calls[0]
    assert (call["lat"], call["lon"]) == (34.072223, -117.907517)
    assert call["time"] == "2026-10-17T07:00:00Z" and call["location"] == "I10 W / N Azusa Ave"
    assert call["details"] == [{"time": "2026-10-17T07:00:00Z", "text": "[1] 2 VEHS"}] and call["units"] == 1

    store = FeedStore(tmp_path)
    store._save("chp-live", "json", json.dumps(calls).encode(), datetime(2026, 10, 17, 7, 2, tzinfo=timezone.utc))
    live = store.live(datetime(2026, 10, 17, 7, 10, tzinfo=timezone.utc), 3)
    assert [c["kind"] for c in live["chp"]] == ["injury"]
    # An hour later with no newer snapshot, collection was down: no live calls.
    assert store.live(datetime(2026, 10, 17, 8, 10, tzinfo=timezone.utc), 3)["chp"] is None
    # Unchanged but still fetched: the old snapshot still holds.
    (tmp_path / "chp-live" / "fetches.log").write_text("20261017T080500Z\n")
    assert len(store.live(datetime(2026, 10, 17, 8, 10, tzinfo=timezone.utc), 3)["chp"]) == 1


def test_zone_outlines_keep_their_shape():
    from explorer.feeds import _simplify
    ring = [[0, 0], [1, 0.0001], [2, 0], [2, 2], [1, 2.0001], [0, 2], [0, 0]]
    assert _simplify([ring]) == [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]


def test_belief_history_replays_as_of_time(tmp_path):
    from explorer.beliefs import BeliefLog, BeliefRuns, evidence
    log = BeliefLog(tmp_path / "run-1.jsonl", agent="test")
    t = datetime(2026, 10, 16, 20, tzinfo=timezone.utc)
    log.record("burning", "The fire is burning", 0.6, time=t)
    log.record("growing", "The fire is growing", 0.7, time=t + timedelta(hours=1), parents=["burning"],
               evidence=[evidence("supports", "calfire", title="147 acres", weight=0.6)])
    log.record("burning", "The fire is burning", 0.9, time=t + timedelta(hours=2))
    runs = BeliefRuns(tmp_path)
    assert runs.runs()[0]["beliefs"] == 2
    now = {b["id"]: b for b in runs.state("run-1")["beliefs"]}
    assert [h["confidence"] for h in now["burning"]["history"]] == [0.6, 0.9]
    assert now["burning"]["children"] == ["growing"]
    earlier = {b["id"]: b for b in runs.state("run-1", t + timedelta(minutes=30))["beliefs"]}
    assert set(earlier) == {"burning"} and earlier["burning"]["confidence"] == 0.6


def test_belief_validation_and_run_names(tmp_path):
    import pytest
    from explorer.beliefs import BeliefLog, BeliefRuns
    log = BeliefLog(tmp_path / "r.jsonl", agent="test")
    with pytest.raises(ValueError):
        log.record("x", "Too sure", 1.5)
    with pytest.raises(ValueError):
        log.record("x", "Odd stance", 0.5, evidence=[{"stance": "maybe", "source": "a"}])
    (tmp_path / "r.jsonl").write_text('{"id": "x"}\n' + json.dumps(
        {"id": "y", "statement": "ok", "time": "2026-10-16T20:00:00Z"}) + "\n")
    assert [b["id"] for b in BeliefRuns(tmp_path).state("r")["beliefs"]] == ["y"]  # bad line skipped
    with pytest.raises(ValueError):
        BeliefRuns(tmp_path).state("../r")
