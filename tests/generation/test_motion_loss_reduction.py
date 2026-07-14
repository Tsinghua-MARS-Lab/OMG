import torch

from omg.generation.losses.motion import _masked_element_mean


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
