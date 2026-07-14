import torch

from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    rotation_6d_to_matrix,
    rotation_6d_to_matrix_canonical_gradient,
)


def _codec(rotation_representation: str) -> G1MotionFeatureCodec:
    return G1MotionFeatureCodec(
        G1Kinematics(kinematics_path="assets/robots/g1/g1_kinematics.json"),
        num_prev_states=10,
        canonical_frame_idx=9,
        rotation_representation=rotation_representation,
    )


def test_root_rotation_representations_have_expected_feature_dims():
    assert _codec("quat").feature_dim == 123
    assert _codec("rot6d").feature_dim == 125


def test_root_rotation_representation_roundtrip_to_quaternion():
    axis_angle = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [-0.4, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    quat = axis_angle_to_quaternion(axis_angle)
    quat = quat / quat.norm(dim=-1, keepdim=True)

    for representation in ("quat", "rot6d"):
        codec = _codec(representation)
        features = codec.rotation_quat_to_features(quat)
        recovered = codec.rotation_features_to_quat(features)
        alignment = (quat * recovered).sum(dim=-1).abs()
        assert torch.allclose(alignment, torch.ones_like(alignment), atol=1e-5)


def test_canonical_gradient_matches_vanilla_at_canonical_representative():
    axis_angle = torch.tensor([[0.2, -0.1, 0.3]])
    canonical = axis_angle_to_matrix(axis_angle)[..., :2, :].reshape(1, 6)
    upstream = torch.randn(1, 3, 3)

    vanilla = canonical.clone().requires_grad_(True)
    (rotation_6d_to_matrix(vanilla) * upstream).sum().backward()
    canonical_mode = canonical.clone().requires_grad_(True)
    (rotation_6d_to_matrix_canonical_gradient(canonical_mode) * upstream).sum().backward()

    torch.testing.assert_close(canonical_mode.grad, vanilla.grad, atol=1e-6, rtol=1e-6)


def test_canonical_gradient_is_bounded_and_linear_at_degenerate_input():
    candidate = torch.tensor([[1.0e-7, 0.0, 0.0, 1.0e-7, 1.0e-9, 0.0]])
    upstream = torch.randn(1, 3, 3)

    first = candidate.clone().requires_grad_(True)
    (rotation_6d_to_matrix_canonical_gradient(first) * upstream).sum().backward()
    scaled = candidate.clone().requires_grad_(True)
    (0.3 * rotation_6d_to_matrix_canonical_gradient(scaled) * upstream).sum().backward()

    assert torch.isfinite(first.grad).all()
    assert first.grad.norm() < 10.0
    torch.testing.assert_close(scaled.grad, 0.3 * first.grad, atol=1e-6, rtol=1e-6)
