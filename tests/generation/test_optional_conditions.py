import sys
import types

import torch
import torch.nn as nn

if "pytorch_lightning" not in sys.modules:
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = nn.Module
    pl.LightningDataModule = object
    sys.modules["pytorch_lightning"] = pl
if "hydra" not in sys.modules:
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda cfg: cfg
    sys.modules["hydra"] = hydra
    sys.modules["hydra.utils"] = hydra_utils
if "omegaconf" not in sys.modules:
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    sys.modules["omegaconf"] = omegaconf

from omg.data.datamodule import motion_collate_fn


def _sample(has_audio: bool, has_human_motion: bool):
    valid = torch.tensor([True, True, False])
    return {
        "length": torch.tensor(2),
        "fps": torch.tensor(30.0),
        "caption": "walk",
        "has_text": torch.tensor(True),
        "motion_features": torch.zeros(3, 123),
        "audio_features": torch.ones(3, 35) if has_audio else None,
        "human_motion": torch.ones(3, 66) if has_human_motion else None,
        "mask": {
            "valid": valid,
            "has_audio": valid if has_audio else torch.zeros(3, dtype=torch.bool),
            "has_human_motion": valid if has_human_motion else torch.zeros(3, dtype=torch.bool),
        },
        "meta": {"sequence_name": "seq"},
    }


def test_optional_condition_collate_zero_fills_missing_tensors():
    batch = motion_collate_fn([_sample(True, False), _sample(False, True)])

    assert batch["audio_features"].shape == (2, 3, 35)
    assert batch["human_motion"].shape == (2, 3, 66)
    assert batch["mask"]["has_audio"].tolist() == [[True, True, False], [False, False, False]]
    assert batch["mask"]["has_human_motion"].tolist() == [[False, False, False], [True, True, False]]
    assert torch.all(batch["audio_features"][1] == 0)
    assert torch.all(batch["human_motion"][0] == 0)
