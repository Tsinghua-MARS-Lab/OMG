from pathlib import Path

import numpy as np
import pytest

from omg.tracking.holomotion.export_clip import (
    HoloMotionClipExportConfig,
    build_holomotion_deployment_clip,
    export_holomotion_deployment_clip,
)


def _qpos(frames: int = 4) -> np.ndarray:
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.linspace(0.0, 0.1, frames, dtype=np.float32)
    return qpos


@pytest.mark.parametrize("frames", [1, 4])
def test_build_holomotion_deployment_clip_schema(frames: int):
    clip = build_holomotion_deployment_clip(_qpos(frames), fps=50.0)
    assert clip["ref_dof_pos"].shape == (frames, 29)
    assert clip["ref_dof_vel"].shape == (frames, 29)
    assert clip["ref_global_translation"].shape == (frames, 30, 3)
    assert clip["ref_global_rotation_quat"].shape == (frames, 30, 4)
    assert clip["ref_global_velocity"].shape == (frames, 30, 3)
    assert clip["ref_global_angular_velocity"].shape == (frames, 30, 3)
    assert float(clip["fps"][0]) == pytest.approx(50.0)
    assert clip["joint_names"].shape == (29,)
    assert clip["body_names"].shape == (30,)
    np.testing.assert_allclose(np.linalg.norm(clip["ref_global_rotation_quat"], axis=-1), 1.0, atol=1e-5)


def test_export_holomotion_deployment_clip_resamples_and_writes(tmp_path: Path):
    reference = tmp_path / "reference.npz"
    output = tmp_path / "clip.npz"
    np.savez(reference, qpos_36=_qpos(6), fps=np.asarray([30.0], dtype=np.float32))

    result = export_holomotion_deployment_clip(
        HoloMotionClipExportConfig(reference=reference, output=output, target_fps=50.0)
    )

    assert result.output_path == output.resolve()
    assert result.frames == 9
    with np.load(output, allow_pickle=False) as data:
        assert data["ref_dof_pos"].shape == (9, 29)
        assert data["ref_global_translation"].shape == (9, 30, 3)
        assert str(data["metadata"][0])
