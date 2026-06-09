import torch

from omg.generation.diffusion.guided import GuidedDiffusion


def test_min_snr_gamma_uses_x0_weighting():
    diffusion = GuidedDiffusion(
        timesteps=10,
        test_timestep_respacing="10",
        zero_terminal_snr=True,
        loss_weighting="min_snr_gamma",
        snr_gamma=5.0,
    )
    timesteps = torch.tensor([0, 5, 9], dtype=torch.long)
    weights = diffusion._loss_weights(timesteps, target_ndim=1, dtype=torch.float32)
    expected = diffusion.base_snr[timesteps].clamp(min=0.0, max=5.0)

    assert torch.allclose(weights, expected)
    assert weights[-1] < 1e-5
