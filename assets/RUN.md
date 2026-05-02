# Running DDRL-Nav2D

This repository currently supports five practical execution paths:

1. **Cluster smoke test**
   - one coordinator
   - one learner
   - one or more neural workers

2. **Online BC**
   - live expert rollouts
   - neural policy trained from scratch with supervised loss only

3. **RL With BC Anchor**
   - neural workers for RL
   - expert workers for supervised anchoring

4. **Pure RL**
   - resume from a checkpoint and train with V-trace only

5. **Standalone ONNX playback**
   - export a checkpoint
   - run one worker without learner/coordinator streaming

Optional experimental path:

- offline BC pretrain from recorded demos

## Quick Start

### Pull Prebuilt Docker Images

If the images have already been published, the simplest public run path is:

```bash
docker compose pull
docker compose up coordinator learner worker
```

This uses the image names already defined in [docker-compose.yml](../docker-compose.yml). The default compose worker runs headless so the cluster smoke test works without X11 setup.

Notes:

- `coordinator` listens on port `10800`
- `learner` serves worker gRPC policy/rollout traffic on port `6600`
- optional overrides can be placed in `.env`; see [`.env.example`](../.env.example)

For Linux/X11 visualization, allow local X11 access and run the optional visualization profile:

```bash
xhost +si:localuser:root
docker compose --profile viz up coordinator learner worker-viz
xhost -si:localuser:root
```

### Native Local Build

For Ubuntu 24.04, install the native C++ worker dependencies and the pinned ONNX Runtime SDK:

```bash
chmod +x scripts/install.sh
scripts/install.sh
```

This installs the system packages used by the C++ worker build (`cmake`, `ninja`, `yaml-cpp`, `spdlog`, `CLI11`, `gRPC`, `Protobuf`, `VTK`, Qt/OpenGL/X11 development packages) and downloads ONNX Runtime into `third_party/onnxruntime`.

Then create the Python environment and build:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/learner/requirements.txt
pip install -r apps/coordinator/requirements.txt

cmake --preset dev
cmake --build --preset dev
```

If protobufs changed:

```bash
python scripts/proto_gen.py
```

## Configuration

This repo uses four configuration layers with precedence:

> code defaults < YAML config < environment / launcher overrides < CLI flags

### 1. Code defaults

Examples:

- [run_local_stack.py](/home/ttd/Projects/CourseWork/ddrl-nav2d/scripts/run_local_stack.py)
- [main.py](/home/ttd/Projects/CourseWork/ddrl-nav2d/apps/learner/main.py)
- worker config parsing in [config.cpp](/home/ttd/Projects/CourseWork/ddrl-nav2d/apps/worker/src/config.cpp)

These are the fallback values when nothing else overrides them.

### 2. YAML config files

Examples:

- [config/policy_arch.yaml](/home/ttd/Projects/CourseWork/ddrl-nav2d/config/policy_arch.yaml)
- [config/rl_reward.yaml](/home/ttd/Projects/CourseWork/ddrl-nav2d/config/rl_reward.yaml)
- [config/worker.yaml](/home/ttd/Projects/CourseWork/ddrl-nav2d/config/worker.yaml)
- [config/worker_expert.yaml](/home/ttd/Projects/CourseWork/ddrl-nav2d/config/worker_expert.yaml)
- [config/simulator.yaml](/home/ttd/Projects/CourseWork/ddrl-nav2d/config/simulator.yaml)

These define the base runtime behavior for a component.

### 3. Environment variables

These are typically injected by:

- `scripts/run_local_stack.py`
- shell exports in your terminal
- Docker / compose

Examples:

- `DDRL_WORKER_NUM_SIM_INSTANCES`
- `DDRL_WORKER_ENABLE_VISUALIZATION`
- `DDRL_WORKER_REAL_TIME_FACTOR`
- `DDRL_COORDINATOR_TARGET`
- `DDRL_LEARNER_DEVICE`

For worker launch, environment variables often override YAML config because they are applied at runtime before the worker starts.

### 4. CLI flags

Examples:

- learner:
  - `--trainer-mode`
  - `--resume-checkpoint`
  - `--coordinator-target`
- worker:
  - `--config`
  - `--policy-onnx`
  - `--standalone-run-enable`

CLI flags override config-file values for the same setting.

## Cluster Smoke Test

This is the fastest native sanity check. Mainly used for integration test.

### Terminal 1: Coordinator

```bash
python scripts/run_local_stack.py --service coordinator
```

### Terminal 2: Learner

```bash
DDRL_LEARNER_DEVICE=auto \
python scripts/run_local_stack.py --service learner
```

### Terminal 3: Worker 1

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=4 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=1.0 \
python scripts/run_local_stack.py --service worker --worker-identity worker-01
```

### Terminal 4: Worker 2

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=4 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=1.0 \
python scripts/run_local_stack.py --service worker --worker-identity worker-02
```

## Training Workflows

### A. Online BC From Scratch

Use this when you want the current learner to mimic the old online BC behavior:

- live expert rollouts
- no RL loss
- no warm start required

This is the cleanest imitation-only path in the current codebase:

- supervised actor training from expert-tagged rollouts
- no V-trace loss

Terminal 1: Coordinator

```bash
python scripts/run_local_stack.py --service coordinator
```

Terminal 2: Learner

```bash
python3 apps/learner/main.py \
  --trainer-mode online_bc \
  --grpc-bind 0.0.0.0:6600 \
  --grpc-advertise 127.0.0.1:6600 \
  --coordinator-target 127.0.0.1:10800 \
  --checkpoint-dir learner_checkpoints/online_bc_mlp
```

Terminal 3+: Expert worker(s)

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=6 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=4.0 \
build/dev/apps/worker/ddrl_worker \
  --config config/worker_expert.yaml
```

Important:

- `online_bc` expects expert-tagged rollouts
- if you mix neural workers into this learner session, the learner will ignore those non-expert rollouts

### B. RL With BC Anchor

Use this when:

- pure RL drifts too quickly from online BC expert.
- you want RL optimization while keeping expert supervision in the loop

This is the closest replacement for the older “BC coefficient during RL” workflow.

Terminal 1: Coordinator

```bash
python scripts/run_local_stack.py --service coordinator
```

Terminal 2: Learner

Resume from a checkpoint produced by `online_bc` or offline BC:

```bash
python3 apps/learner/main.py \
  --trainer-mode rl_bc_anchor \
  --bc-coef 1.0 \
  --bc-decay-updates 0 \
  --grpc-bind 0.0.0.0:6600 \
  --grpc-advertise 127.0.0.1:6600 \
  --coordinator-target 127.0.0.1:10800 \
  --checkpoint-dir learner_checkpoints/rl_bc_anchor_mlp \
  --resume-checkpoint learner_checkpoints/online_bc_mlp/policy_v00600.pt
```

Terminal 3: Neural worker(s)

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=4 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=-1.0 \
build/dev/apps/worker/ddrl_worker \
  --config config/worker.yaml
```

Terminal 4: Expert worker(s)

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=4 \
DDRL_WORKER_ENABLE_VISUALIZATION=false \
DDRL_WORKER_REAL_TIME_FACTOR=4.0 \
build/dev/apps/worker/ddrl_worker \
  --config config/worker_expert.yaml
```

What this corresponds to in `README.md`:

- V-trace actor-critic on neural rollouts
- BC loss on expert-tagged rollouts
- combined loss inside the same learner step

### C. Pure RL / RL finetune

Use this after the anchored stage if you want to remove supervised pressure and let RL continue alone.

#### Learner

```bash
python3 apps/learner/main.py \
  --trainer-mode rl \
  --grpc-bind 0.0.0.0:6600 \
  --grpc-advertise 127.0.0.1:6600 \
  --coordinator-target 127.0.0.1:10800 \
  --checkpoint-dir learner_checkpoints/rl_mlp \
  --resume-checkpoint learner_checkpoints/rl_bc_anchor_mlp/policy_v12200.pt
```

#### Worker(s)

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=4 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=-1.0 \
build/dev/apps/worker/ddrl_worker \
  --config config/worker.yaml
```

## Optional Paths

### Offline BC Pretrain

Use this when you want a clean supervised checkpoint before any online learner loop.

#### Record expert demo batches

```bash
DDRL_WORKER_NUM_SIM_INSTANCES=9 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=1.0 \
build/dev/apps/worker/ddrl_worker \
  --config config/worker_expert_record.yaml \
  --seed -1
```

This writes `rollout_batch_*.pb` files under `data/demos_bc/`.

#### Train offline BC

```bash
python apps/learner/tools/bc_pretrain.py \
  --input-dir data/demos_bc \
  --output-dir learner_checkpoints/bc_pretrain \
  --policy-arch-config config/policy_arch.yaml \
  --epochs 10 \
  --batch-size 1024 \
  --lr 1e-4 \
  --device auto
```

Outputs:

- `learner_checkpoints/bc_pretrain/model_1.pt`
- `learner_checkpoints/bc_pretrain/model_2.pt`
- ...
- `learner_checkpoints/bc_pretrain/model_best.pt`

## Standalone ONNX Playback

Use this to inspect a trained policy without coordinator or learner streaming.

```bash
python apps/learner/tools/export_checkpoint_onnx.py \
  learner_checkpoints/rl_bc_anchor_mlp/policy_v12200.pt \
  --output learner_checkpoints/rl_bc_anchor_mlp/policy_v12200.onnx

DDRL_WORKER_NUM_SIM_INSTANCES=6 \
DDRL_WORKER_ENABLE_VISUALIZATION=true \
DDRL_WORKER_REAL_TIME_FACTOR=8.0 \
build/dev/apps/worker/ddrl_worker \
  --standalone-run-enable true \
  --policy-onnx learner_checkpoints/rl_bc_anchor_mlp/policy_v12200.onnx
```

In standalone mode:

- the worker does not need a learner
- the ONNX policy is evaluated locally
- this is the easiest way to compare BC / RL checkpoints visually
