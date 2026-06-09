from __future__ import annotations

import pytest

from omg.realtime.sim_stream import MujocoSimStream
from omg.realtime.sim_stream import parse_host_port
from omg.realtime.sim_stream import SimStreamConfig


def test_parse_host_port_accepts_host_port_and_http_url() -> None:
    assert parse_host_port("127.0.0.1:7870") == ("127.0.0.1", 7870)
    assert parse_host_port(":7870") == ("127.0.0.1", 7870)
    assert parse_host_port("http://localhost:7871/video.mjpg") == ("localhost", 7871)


def test_parse_host_port_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        parse_host_port("127.0.0.1")
    with pytest.raises(ValueError):
        parse_host_port("127.0.0.1:70000")


def test_sim_stream_state_reports_latest_qpos(monkeypatch) -> None:
    import threading

    import numpy as np

    class _FakeRenderer:
        def close(self) -> None:
            pass

    class _FakeCv2:
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def imencode(*_args, **_kwargs):
            return True, np.asarray([1, 2, 3], dtype=np.uint8)

    def _fake_init(self, config: SimStreamConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg = None
        self._qpos_36 = None
        self._overlay_lines = []
        self._frame_index = -1
        self._render_count = 0
        self._last_render = 0.0
        self._last_update_wall_time = 0.0
        self._closed = False
        self._renderer = _FakeRenderer()
        self._cv2 = _FakeCv2()

    monkeypatch.setattr(MujocoSimStream, "__init__", _fake_init)
    monkeypatch.setattr(MujocoSimStream, "_render_frame", lambda self, qpos_36, overlay_lines: np.zeros((2, 2, 3), dtype=np.uint8))

    stream = MujocoSimStream(SimStreamConfig(bind="127.0.0.1:1", fps=1000.0))
    qpos = np.arange(36, dtype=np.float32)
    stream.update(qpos, frame_index=12, overlay_lines=["prompt: stand"])
    state = stream.state()
    assert state["frame_index"] == 12
    assert state["overlay_lines"] == ["prompt: stand"]
    np.testing.assert_allclose(state["qpos_36"], qpos)
