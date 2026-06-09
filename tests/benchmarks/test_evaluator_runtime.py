import torch

from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
from omg.benchmarks.evaluator.representation import canonical_body_positions_from_qpos
from omg.benchmarks.evaluator.retrieval import batch_r_precision
from omg.robots.g1.kinematics import G1Kinematics


def test_batch_r_precision_accepts_explicit_positive_mask():
    logits = torch.tensor(
        [
            [0.1, 3.0, -1.0],
            [3.0, 0.1, -1.0],
            [-1.0, 0.1, 3.0],
        ]
    )
    positives = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )
    assert batch_r_precision(logits, top_k=1, positive_mask=positives).tolist() == [1.0]


def test_canonical_body_positions_from_qpos_keeps_anchor_root_at_origin():
    qpos = torch.zeros(2, 4, 36)
    qpos[..., 3] = 1.0
    qpos[:, :, 0] = torch.arange(4, dtype=torch.float32)
    body_pos = canonical_body_positions_from_qpos(qpos, G1Kinematics())
    assert body_pos.shape[:3] == (2, 4, 30)
    torch.testing.assert_close(body_pos[:, 0, 0], torch.zeros(2, 3))


def test_motion_encoder_accepts_body_position_sequences():
    encoder = MotionEncoder(
        input_dim=90,
        movement_dim=32,
        hidden_dim=32,
        output_dim=16,
        movement_mode="linear",
        num_layers=1,
        num_heads=4,
    )
    valid = torch.ones(2, 5, dtype=torch.bool)
    embedding = encoder(torch.randn(2, 5, 30, 3), valid_mask=valid)
    assert embedding.shape == (2, 16)
