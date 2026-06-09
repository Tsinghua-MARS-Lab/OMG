from pathlib import Path

import numpy as np
import pytest

from omg.tracking.holomotion.io import load_reference_motion
from omg.tracking.holomotion.reference import resample_qpos


def _qpos(frames: int = 4) -> np.ndarray:
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.arange(frames, dtype=np.float32)
    return qpos


def test_load_reference_npz_with_fps(tmp_path: Path):
    path = tmp_path / "motion.npz"
    np.savez(path, qpos_36=_qpos(), fps=np.asarray(30.0, dtype=np.float32), text=np.asarray("walk"))
    ref = load_reference_motion(path)
    assert ref.qpos_36.shape == (4, 36)
    assert ref.fps == pytest.approx(30.0)
    assert ref.metadata["text"] == "walk"


def test_load_reference_npy_requires_fps(tmp_path: Path):
    path = tmp_path / "motion.npy"
    np.save(path, _qpos())
    with pytest.raises(ValueError, match="fps"):
        load_reference_motion(path)
    ref = load_reference_motion(path, fps=50.0)
    assert ref.fps == pytest.approx(50.0)


def test_load_reference_rejects_bad_shape(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, qpos_36=np.zeros((3, 35), dtype=np.float32), fps=30.0)
    with pytest.raises(ValueError, match="qpos_36"):
        load_reference_motion(path)


def test_resample_qpos_normalizes_quaternion():
    qpos = _qpos(frames=3)
    qpos[:, 3] = 2.0
    out = resample_qpos(qpos, source_fps=30.0, target_fps=30.0)
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0, atol=1e-6)
