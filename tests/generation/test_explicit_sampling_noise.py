import pytest
import torch

from omg.generation.diffusion.forcing import DiffusionForcingProcess
from omg.generation.diffusion.guided import GuidedDiffusion


class _ZeroDenoiser(torch.nn.Module):
    def forward(self, x, timesteps, conditions, valid_mask=None):
        del timesteps, conditions, valid_mask
        return torch.zeros_like(x)


@pytest.mark.parametrize(
    "diffusion",
    [
        GuidedDiffusion(timesteps=8, test_timestep_respacing="ddim4", ddim_eta=0.0),
        DiffusionForcingProcess(timesteps=8, sampling_steps=4, ddim_eta=0.0),
    ],
)
def test_explicit_initial_noise_batches_match_individual_samples(diffusion):
    denoiser = _ZeroDenoiser()
    noise = torch.randn(3, 5, 7)
    conditions = {"conditioning": torch.zeros(3, 1, 1)}

    batched = diffusion.sample(
        denoiser,
        tuple(noise.shape),
        conditions,
        initial_noise=noise,
    )
    individual = torch.cat(
        [
            diffusion.sample(
                denoiser,
                (1, noise.shape[1], noise.shape[2]),
                {"conditioning": conditions["conditioning"][index : index + 1]},
                initial_noise=noise[index : index + 1],
            )
            for index in range(noise.shape[0])
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, individual, rtol=0.0, atol=0.0)


def test_explicit_initial_noise_rejects_wrong_shape():
    diffusion = GuidedDiffusion(
        timesteps=8,
        test_timestep_respacing="ddim4",
        ddim_eta=0.0,
    )
    with pytest.raises(ValueError, match="initial_noise has shape"):
        diffusion.sample(
            _ZeroDenoiser(),
            (2, 5, 7),
            {"conditioning": torch.zeros(2, 1, 1)},
            initial_noise=torch.zeros(1, 5, 7),
        )
