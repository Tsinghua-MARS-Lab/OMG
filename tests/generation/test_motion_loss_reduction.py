import torch

from omg.generation.losses.motion import _masked_element_mean, _rotation_chordal_sq
from omg.utils.rotation_conversions import axis_angle_to_matrix


def test_masked_element_mean_counts_trailing_event_dimensions():
    values = torch.tensor(
        [
            [[1.0, 3.0, 5.0], [100.0, 100.0, 100.0]],
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
        ]
    )
    valid = torch.tensor([[True, False], [True, True]])

    actual = _masked_element_mean(values, valid)

    # Each sample is reduced over all of its valid scalar elements first, so
    # sequence length does not change that sample's weight in the batch.
    expected = torch.tensor(((1.0 + 3.0 + 5.0) / 3.0 + (2.0 + 4.0 + 6.0 + 8.0 + 10.0 + 12.0) / 6.0) / 2.0)
    torch.testing.assert_close(actual, expected)


def test_masked_element_mean_excludes_samples_without_valid_values():
    values = torch.tensor([[[100.0, 100.0]], [[2.0, 4.0]]])
    valid = torch.tensor([[False], [True]])

    actual = _masked_element_mean(values, valid)

    torch.testing.assert_close(actual, torch.tensor(3.0))


def test_rotation_chordal_loss_is_locally_angle_squared():
    angle = torch.tensor([[1.0e-3, -2.0e-3, 3.0e-3]])
    pred = axis_angle_to_matrix(angle)
    target = torch.eye(3).unsqueeze(0)

    actual = _rotation_chordal_sq(pred, target)

    torch.testing.assert_close(actual, angle.square().sum(dim=-1), atol=1e-8, rtol=1e-5)


def test_rotation_chordal_gradient_is_finite_at_pi_cut_locus():
    angle = torch.tensor([[torch.pi, 0.0, 0.0]], requires_grad=True)
    pred = axis_angle_to_matrix(angle)
    target = torch.eye(3).unsqueeze(0)

    _rotation_chordal_sq(pred, target).sum().backward()

    assert torch.isfinite(angle.grad).all()
