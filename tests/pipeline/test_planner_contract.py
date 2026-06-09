from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from omg.runtime.onnx_providers import DEFAULT_DIFFUSION_ONNX_PROVIDERS, DEFAULT_TENSORRT_ONNX_PROVIDERS
from omg.pipeline.planner import (
    DiffusionContinuationState,
    MotionPlan,
    OnnxDiffusionPlanner,
    _continuation_start_step,
    _coerce_seed_qpos,
    _providers,
    save_motion_plan,
)
from omg.cli.pipeline.utils import (
    _append_executed_history,
    _source_cursor_from_tracker_cursor,
    _tracking_error_stats,
)


def test_provider_parser_accepts_comma_list():
    assert _providers("CUDAExecutionProvider,CPUExecutionProvider") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_provider_parser_defaults_to_cuda_for_standard_diffusion():
    assert _providers(None) == list(DEFAULT_DIFFUSION_ONNX_PROVIDERS)
    assert _providers(None, {"tensorrt_compatible": False}) == list(DEFAULT_DIFFUSION_ONNX_PROVIDERS)


def test_provider_parser_defaults_to_tensorrt_for_tensorrt_diffusion():
    assert _providers(None, {"tensorrt_compatible": True}) == list(DEFAULT_TENSORRT_ONNX_PROVIDERS)


def test_coerce_seed_qpos_validates_shape():
    with pytest.raises(ValueError):
        _coerce_seed_qpos(np.zeros((4, 35), dtype=np.float32))
    qpos = _coerce_seed_qpos(np.zeros((4, 36), dtype=np.float32))
    assert qpos.shape == (4, 36)
    assert qpos.dtype == np.float32


def test_save_motion_plan_contract(tmp_path):
    plan = MotionPlan(
        qpos_36=np.zeros((3, 36), dtype=np.float32),
        motion_features=np.zeros((3, 123), dtype=np.float32),
        fps=30.0,
        metadata={"mode": "diffusion-only"},
    )
    out = save_motion_plan(plan, tmp_path / "sample", extra_metadata={"text": "walk"})
    assert (out / "qpos_36.npy").exists()
    assert (out / "reference_motion.npz").exists()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "diffusion-only"
    assert metadata["text"] == "walk"


def test_source_cursor_from_tracker_cursor_uses_next_source_frame():
    assert _source_cursor_from_tracker_cursor(0, source_fps=30.0, tracker_fps=50.0) == 0
    assert _source_cursor_from_tracker_cursor(59, source_fps=30.0, tracker_fps=50.0) == 36
    assert _source_cursor_from_tracker_cursor(60, source_fps=30.0, tracker_fps=50.0) == 36


def test_append_executed_history_uses_tracker_rollout_tail():
    seed = np.zeros((10, 36), dtype=np.float32)
    seed[:, 0] = np.arange(10, dtype=np.float32)
    seed[:, 3] = 1.0
    executed = np.zeros((50, 36), dtype=np.float32)
    executed[:, 0] = 100.0 + np.arange(50, dtype=np.float32)
    executed[:, 3] = 1.0
    updated = _append_executed_history(
        seed,
        executed,
        executed_fps=50.0,
        target_fps=30.0,
        history_frames=10,
    )
    assert updated.shape == (10, 36)
    assert np.all(updated[:, 0] >= 100.0)
    assert np.all(np.diff(updated[:, 0]) > 0.0)


def test_tracking_error_stats_reports_root_and_joint_error():
    reference = np.zeros((4, 36), dtype=np.float32)
    executed = reference.copy()
    executed[:, 0] = 0.3
    executed[:, 7:36] = 0.2
    stats = _tracking_error_stats(executed, reference)
    assert stats["root_xy_error_mean"] == pytest.approx(0.3)
    assert stats["root_xy_error_max"] == pytest.approx(0.3)
    assert stats["joint_abs_error_mean"] == pytest.approx(0.2)
    assert stats["joint_abs_error_max"] == pytest.approx(0.2)


def test_continuation_start_step_counts_remaining_denoise_steps():
    assert _continuation_start_step(50, 8) == 7
    assert _continuation_start_step(50, 40) == 39
    with pytest.raises(ValueError, match="continuation_steps"):
        _continuation_start_step(50, 0)
    with pytest.raises(ValueError, match="continuation_steps"):
        _continuation_start_step(50, 51)


def test_diffusion_continuation_state_stays_out_of_saved_metadata(tmp_path):
    state = DiffusionContinuationState(
        latent_states=np.zeros((2, 3, 4), dtype=np.float32),
        valid_steps=np.ones((2,), dtype=bool),
        sample_timestep_map=np.arange(2, dtype=np.int64),
        canonical_root_pos=np.zeros((1, 1, 3), dtype=np.float32),
        canonical_root_quat=np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    plan = MotionPlan(
        qpos_36=np.zeros((3, 36), dtype=np.float32),
        motion_features=np.zeros((3, 123), dtype=np.float32),
        fps=30.0,
        metadata={"mode": "diffusion-only"},
        continuation_state=state,
    )
    out = save_motion_plan(plan, tmp_path / "sample")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "continuation_state" not in metadata



def test_onnx_cfg_pred_batches_cond_and_null_branches():
    planner = object.__new__(__import__("omg.pipeline.planner", fromlist=["OnnxDiffusionPlanner"]).OnnxDiffusionPlanner)
    planner.batch_size = 2
    planner.use_audio = False
    planner.use_human_motion = False
    calls = []

    def fake_onnx_pred(
        x,
        model_timestep,
        valid_mask,
        history_features,
        text_inputs,
        audio_inputs=None,
        human_motion_inputs=None,
    ):
        calls.append(
            {
                "x_shape": x.shape,
                "text_context": text_inputs["text_context"].copy(),
                "text_mask": text_inputs["text_mask"].copy(),
            }
        )
        return np.asarray([[[3.0]], [[1.0]]], dtype=np.float32)

    planner._onnx_pred = fake_onnx_pred
    pred = planner._onnx_cfg_pred(
        np.zeros((1, 1, 1), dtype=np.float32),
        0,
        np.ones((1, 1), dtype=bool),
        np.zeros((1, 1, 1), dtype=np.float32),
        {"text_context": np.asarray([[[10.0]]], dtype=np.float32), "text_mask": np.asarray([[True]])},
        {"text_context": np.asarray([[[0.0]]], dtype=np.float32), "text_mask": np.asarray([[False]])},
        2.5,
    )
    assert len(calls) == 1
    assert calls[0]["x_shape"] == (2, 1, 1)
    assert calls[0]["text_context"][:, 0, 0].tolist() == [10.0, 0.0]
    assert calls[0]["text_mask"][:, 0].tolist() == [True, False]
    assert pred.shape == (1, 1, 1)
    assert pred[0, 0, 0] == pytest.approx(6.0)


def test_pipeline_async_defaults_to_tensorrt_fp16_and_cache(tmp_path):
    from omg.cli.pipeline.main import _diffusion_dit_cache, _diffusion_tensorrt_cache_path, _diffusion_tensorrt_fp16

    args = type(
        "Args",
        (),
        {"mode": "async", "tensorrt_fp16": None, "tensorrt_engine_cache_path": None, "dit_cache": None},
    )()
    assert _diffusion_tensorrt_fp16(args) is True
    assert _diffusion_tensorrt_cache_path(args, tmp_path) == tmp_path / "tensorrt_engine_cache"
    assert _diffusion_dit_cache(args) is True

    args.mode = "sync"
    assert _diffusion_tensorrt_fp16(args) is False
    assert _diffusion_tensorrt_cache_path(args, tmp_path) is None
    assert _diffusion_dit_cache(args) is False

    args.dit_cache = False
    args.mode = "async"
    assert _diffusion_dit_cache(args) is False

    args.tensorrt_fp16 = False
    args.tensorrt_engine_cache_path = "custom_cache"
    assert _diffusion_tensorrt_fp16(args) is False
    assert _diffusion_tensorrt_cache_path(args, tmp_path) == Path("custom_cache").expanduser()



def _dit_cache_test_planner(*, enabled: bool):
    planner = object.__new__(OnnxDiffusionPlanner)
    planner.rng = np.random.default_rng(0)
    planner.sequence_length = 2
    planner.feat_dim = 1
    planner.sample_timestep_map = np.arange(5, dtype=np.int64)
    planner.sample_alphas_cumprod = np.linspace(0.2, 0.9, 5, dtype=np.float64)
    planner.sample_alphas_cumprod_prev = np.asarray([1.0, 0.2, 0.375, 0.55, 0.725], dtype=np.float64)
    planner.ddim_eta = 0.0
    planner.dit_cache = enabled
    planner.dit_cache_threshold = 0.99
    planner.dit_cache_warmup_steps = 1
    planner.dit_cache_max_consecutive = 2
    planner.calls = []
    planner._dit_cache_similarity_seconds = 0.0
    planner._diffusion_update_seconds = 0.0

    def fake_cfg_pred(
        x,
        model_timestep,
        valid_mask,
        history_np,
        cond_text,
        null_text,
        cfg_scale,
        **kwargs,
    ):
        planner.calls.append(int(model_timestep))
        return np.ones_like(x, dtype=np.float32)

    planner._onnx_cfg_pred = fake_cfg_pred
    return planner


def test_dit_cache_skips_after_warmup_and_similarity_threshold():
    planner = _dit_cache_test_planner(enabled=True)
    _, _, stats = planner._sample_chunk(
        torch.zeros((1, 1, 1), dtype=torch.float32),
        torch.zeros((1, 1, 3), dtype=torch.float32),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        {"text_context": np.zeros((1, 1, 1), dtype=np.float32), "text_mask": np.ones((1, 1), dtype=bool)},
        {"text_context": np.zeros((1, 1, 1), dtype=np.float32), "text_mask": np.ones((1, 1), dtype=bool)},
        2.5,
    )
    assert stats.enabled is True
    assert stats.executed_steps == 3
    assert stats.skipped_steps == 2
    assert planner.calls == [4, 3, 0]


def test_dit_cache_disabled_executes_all_steps():
    planner = _dit_cache_test_planner(enabled=False)
    _, _, stats = planner._sample_chunk(
        torch.zeros((1, 1, 1), dtype=torch.float32),
        torch.zeros((1, 1, 3), dtype=torch.float32),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        {"text_context": np.zeros((1, 1, 1), dtype=np.float32), "text_mask": np.ones((1, 1), dtype=bool)},
        {"text_context": np.zeros((1, 1, 1), dtype=np.float32), "text_mask": np.ones((1, 1), dtype=bool)},
        2.5,
    )
    assert stats.enabled is False
    assert stats.executed_steps == 5
    assert stats.skipped_steps == 0
    assert planner.calls == [4, 3, 2, 1, 0]
