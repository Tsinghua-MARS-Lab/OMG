import torch

from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import axis_angle_to_quaternion


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
