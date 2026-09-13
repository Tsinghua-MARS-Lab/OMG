# Inference and deployment contract

Generation and ONNX export resolve history position encoding, self/cross QK
normalization and frame conditioning from the checkpoint architecture contract.
An explicit conflicting override is an error. Do not globally enable sinusoidal
PE: old checkpoints can require `none`. ONNX deployment uses the exported metadata.

For a checkpoint without a contract, recover its training configuration and pass
both `--legacy-attention-contract` and `model.history_pos_encoding=...`, along
with matching `denoiser.self_attention_qk_norm` and
`denoiser.cross_attention_qk_norm` overrides. Version-1 contracts without history
PE also need an explicit PE override. Shape-compatible loading is not evidence of
matching normalization semantics.

Pure-text separate CFG uses the same batched conditional/null path as joint CFG.
With batch 2, 50 uncached diffusion steps therefore require 50 ONNX calls, not
100 duplicated calls. True multi-modal separate guidance retains its independent
branches. Repeated text conditions are cached automatically, up to 128 prompts.

DiT cache is **enabled by default**, including synchronous inference. This is an
approximation and can change motion quality. Use `--no-dit-cache` (or
`dit_cache=False` in the planner API) for uncached comparisons. Defaults remain
threshold 0.995, warmup 4, maximum consecutive skips 2. This does not turn on
FP16, change diffusion steps, or enable continuation. Training and native
PyTorch generation do not use this ONNX planner cache.

Export parity validation runs in IEEE FP32 and restores the caller's precision
settings, including on failure. The numerical acceptance thresholds are unchanged.
MP4 rendering requires H.264/yuv420p/faststart; packaged imageio-ffmpeg is used
when no system ffmpeg exists. Encoding failures are surfaced, not silently
reported as successful portable videos.

Inspect each plan's metadata: architecture, active providers, CFG routing, text
cache state, TensorRT precision, cache-executed/skipped steps, ONNX call count,
and timing breakdown. An engine cache only saves engine rebuilds; it is not a
DiT cache. Async latency depends on hardware and scheduling, not only checkpoint
size. Changing measured latency changes activation times and executed histories,
so closed-loop videos need not match even for algebraically equivalent CFG.
