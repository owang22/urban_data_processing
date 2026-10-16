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
