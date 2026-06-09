import numpy as np
import pytest

from omg.tracking.holomotion.video import camera_azimuth, draw_overlay, yaw_degrees_from_wxyz


def test_camera_azimuth_views():
    assert camera_azimuth("back") == pytest.approx(0.0)
    assert camera_azimuth("side") == pytest.approx(90.0)
    assert camera_azimuth("iso") == pytest.approx(135.0)
    assert camera_azimuth("front") == pytest.approx(180.0)


def test_yaw_degrees_from_wxyz():
    half = np.deg2rad(45.0 / 2.0)
    quat = np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
    assert yaw_degrees_from_wxyz(quat) == pytest.approx(45.0)


def test_yaw_degrees_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        yaw_degrees_from_wxyz(np.zeros(4, dtype=np.float32))


def test_draw_overlay_preserves_frame_shape():
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    rendered = draw_overlay(frame, ["text: walk", "replan: 0 frame: 1/60"])
    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype
    assert rendered.sum() > 0
