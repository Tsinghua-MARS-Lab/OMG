# Installation

OMG targets Python 3.10 and CUDA-capable Linux machines for training,
ONNX export, TensorRT inference, and HoloMotion tracking. macOS can be used for
lightweight code review and documentation work, but GPU execution is expected on
Linux.

## Python Environment

Use a repo-local virtual environment.

```bash
cd /path/to/OMG
curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if uv is already installed
uv venv --python 3.10 .venv
source .venv/bin/activate
```

Install PyTorch first, then install OMG.

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[all]"
```

The same setup is available through the Makefile:

```bash
make venv
source .venv/bin/activate
make install
```

For China mainland networks:

```bash
make install-cn
```

For a smaller install, choose extras by task:

```bash
uv pip install -e ".[train]"
uv pip install -e ".[render]"
uv pip install -e ".[tracking]"
uv pip install -e ".[export]"
uv pip install -e ".[realtime]"
uv pip install -e ".[benchmark]"
```

Common runtime environment:

```bash
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

## Data and Model Roots

The release configs default to:

```text
data/OMG-Data
models/
```

Override them when using external disks or shared storage:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models
```

Text-conditioned training and generation require the Hugging Face `t5-base`
text encoder. The default config loads it from:

```text
${OMG_MODELS_ROOT}/t5-base-local
```

Place a local `t5-base` copy there for offline or cluster runs, or override
`model.text_encoder.model_name` with a different local path or Hugging Face
model id.

If the text encoder is already cached and the machine should not access the
network:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## HoloMotion Dependencies

Offline tracker and realtime deployment require:

- HoloMotion motion tracking ONNX model.
- G1 MuJoCo XML or the default HoloMotion G1 scene.
- `onnxruntime-gpu` and TensorRT runtime libraries.
- MuJoCo for offline simulation and rendering.

Download the HoloMotion G1 motion-tracking ONNX model from the official
[HoloMotion repository](https://github.com/HorizonRobotics/HoloMotion) or the
[HoloMotion Hugging Face artifacts](https://huggingface.co/HorizonRobotics/HoloMotion_models).
OMG does not redistribute HoloMotion weights. The recommended local path is:

```text
models/holomotion/motion_tracking/model.onnx
```

Pass this path explicitly with `--holomotion-onnx` when running tracking or
pipeline modes.

For real-robot deployment, prefer the HoloMotion velocity-tracking model and
place it at:

```text
models/holomotion/velocity_tracking/model.onnx
```

See [Generation](generation.md), [Tracking](tracking.md), and
[Realtime G1 Deployment](realtime_g1.md) for task-specific runtime commands.
