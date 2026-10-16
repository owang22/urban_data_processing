from datetime import datetime, timedelta

import pytest

Image = pytest.importorskip("PIL.Image")

from processing.enrichment.anomaly import ImageAnomalyDetector


def test_heartbeat_fires_for_camera_whose_first_frame_was_skipped(tmp_path):
    # A saturated green frame is not haze-, smoke- or fire-like, and it is identical
    # every time, so only the hourly heartbeat can send it to the image model.
    frame = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (40, 160, 60)).save(frame)
    detector = ImageAnomalyDetector(use_clip=False, heartbeat_seconds=3600.0)
    start = datetime(2025, 1, 7, 10, 0, 0)
    decisions = []
    for step in range(5):  # 10:00 to 11:00 every 15 minutes
        run, diagnostics = detector._should_run_image_model(
            camera_key="cam", image_path=frame, timestamp=start + timedelta(minutes=15 * step))
        decisions.append((run, diagnostics["reason"]))
    assert decisions[0] == (False, "first_frame_skipped_no_cheap_visual_interest")
    assert [run for run, _ in decisions[1:4]] == [False, False, False]
    assert decisions[4] == (True, "hourly_heartbeat_due")  # 11:00, one hour after the first frame
