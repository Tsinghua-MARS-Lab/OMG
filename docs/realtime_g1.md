# Realtime G1 Deployment

Realtime deployment uses three processes:

1. HoloMotion deployment on the G1 Orin.
2. OMG realtime planner server on a GPU workstation.
3. OMG real bridge on the G1 Orin.

The planner server owns diffusion inference. The real bridge reads robot
lowstate, builds history, sends condition-sequence requests, receives planned
future motion, and publishes HoloMotion `obs65` reference packets.

## Runtime Environment

Realtime deployment uses two machines:

- GPU workstation: runs `omg.cli.realtime.planner_server`.
- G1 Orin: runs HoloMotion deployment and `omg.cli.realtime.holomotion_real_bridge`.

The G1 Orin environment must provide Unitree ROS messages, including
`unitree_hg.msg.LowState`. Run the real bridge inside the HoloMotion deployment
environment or container where those ROS packages are sourced.

## Network

The G1 Orin must reach the workstation planner bind address. Use wired Ethernet
when available. Wi-Fi can work, but planner latency and jitter should be checked
before live tests.

Example workstation planner address:

```text
tcp://10.0.20.14:5571
```

## Terminal 1: HoloMotion on G1

Run inside the G1 HoloMotion deployment directory:

```bash
cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
./launch_holomotion_29dof_docker.sh
```

The active launch profile must configure HoloMotion to subscribe to OMG
latest-obs ZMQ:

```yaml
latest_obs_zmq_uri: tcp://127.0.0.1:6001
latest_obs_zmq_topic: obs65
latest_obs_zmq_mode: connect
enable_teleop_reference: true
```

Keep runtime and deployment fields in the launch profile, not in the robot
config YAML.

## Terminal 2: Planner Server on Workstation

```bash
cd /path/to/OMG
source .venv/bin/activate

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m omg.cli.realtime.planner_server \
  --bind tcp://0.0.0.0:5571 \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --tensorrt-engine-cache-path tensorrt_engine_cache/realtime_planner \
  --dit-cache \
  --log-jsonl outputs_realtime/planner.jsonl
```

The planner server does not own the prompt. Conditions come from bridge request
metadata. This keeps runtime condition changes on the robot-side bridge command.

## Terminal 3: Real Bridge on G1

Run inside the HoloMotion deployment environment or container where Unitree ROS
messages are available:

```bash
cd /home/unitree/OMG

PYTHONPATH=src:$PYTHONPATH /root/miniconda3/envs/holomotion_deploy/bin/python \
  -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://10.0.20.14:5571 \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --continuous \
  --replan-remaining-frames 40 \
  --condition-sequence "text: walk forward" \
  --holomotion-config /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof/src/config/g1_29dof_holomotion.yaml \
  --publish-bind tcp://*:6001 \
  --activation-mode remote-b \
  --sleep \
  --status-jsonl /home/unitree/OMG/outputs_realtime/real_demo/status.jsonl \
  --output /home/unitree/OMG/outputs_realtime/real_demo/bridge.npz
```

With `--activation-mode remote-b`, the bridge waits for the Unitree remote B
button before it starts sending active replans. The first active replan uses
live lowstate history.

## Dry Run

Before live tests, run a dry bridge against the planner:

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_dry_run \
  --connect tcp://10.0.20.14:5571 \
  --seed-motion /path/to/seed_motion.npz \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --continuous \
  --replan-remaining-frames 40 \
  --condition-sequence "text: walk forward" \
  --publish-bind tcp://*:6001 \
  --sleep \
  --output outputs_realtime/dry_run/bridge.npz
```

Add `--sim-stream-bind 127.0.0.1:7870` to view a local MuJoCo stream at
`http://127.0.0.1:7870/video.mjpg`.

## Logs

Planner log lines include:

```text
[replan 0001] request=... frame=... buffer=... prompt='walk forward' latency=...
```

Bridge log lines include:

```text
[real-bridge replan 0001] request=... append=... history=lowstate ...
```

Check:

- `history=lowstate` after activation.
- lowstate age stays small.
- bridge latency is close to server latency plus network/request overhead.
- reference buffer does not drain to zero.

## Safety

Test HoloMotion standalone before realtime diffusion. Keep the Unitree remote
available and verify emergency stop behavior before pressing B for active
realtime rollout.
