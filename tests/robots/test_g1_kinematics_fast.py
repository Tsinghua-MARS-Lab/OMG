from __future__ import annotations

import torch
import torch.nn.functional as F

from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics


def _sample_qpos(batch_shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(1234)
    qpos = torch.randn(*batch_shape, 36, generator=generator)
    qpos[..., 3:7] = F.normalize(qpos[..., 3:7], dim=-1)
    qpos[..., 7:] = qpos[..., 7:].clamp(-1.0, 1.0)
    return qpos


def test_forward_body_positions_matches_full_fk():
    kinematics = G1Kinematics()
    qpos = _sample_qpos((4, 8))

    full = kinematics.forward_kinematics_full(qpos)
    fast_body_pos = kinematics.forward_body_positions(qpos)
    fast_fk = kinematics.forward_kinematics(qpos)

    torch.testing.assert_close(fast_body_pos, full["body_pos_w"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_fk["body_pos_w"], full["body_pos_w"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_fk["body_quat_w"], full["body_quat_w"], atol=1e-6, rtol=1e-6)


def test_prev_state_features_accept_pos_only_fk():
    kinematics = G1Kinematics()
    codec = G1MotionFeatureCodec(kinematics, num_prev_states=2)
    qpos = _sample_qpos((4, 2))
    fk = kinematics.forward_kinematics(qpos)
    fast_body_pos = kinematics.forward_body_positions(qpos)
    fps = torch.full((4,), 30.0)

    baseline, baseline_root_pos, baseline_root_quat = codec.prev_state_features_from_history(
        qpos,
        fk["body_pos_w"],
        fk["body_quat_w"],
        fps=fps,
    )
    fast, fast_root_pos, fast_root_quat = codec.prev_state_features_from_history(
        qpos,
        fast_body_pos,
        None,
        fps=fps,
    )

    torch.testing.assert_close(fast, baseline, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_root_pos, baseline_root_pos)
    torch.testing.assert_close(fast_root_quat, baseline_root_quat)
