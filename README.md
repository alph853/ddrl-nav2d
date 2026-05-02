# DDRL-Nav2D

## 1. Overview

Distributed deep reinforcement learning for 2.5D robot navigation. The repository combines a C++ simulation worker, a Python actor-learner stack, ONNX Runtime policy deployment, and tooling for scenario generation and visualization. In summary:

- navigation in dynamic scenes
- asynchronous distributed training
- policy deployment into a native runtime
- reproducible simulation and evaluation
- reward design and imitation-learning support

![DDRL-Nav2D distributed simulator demo](./assets/ddrl.gif)

Distributed worker visualization: multiple C++ simulator workers connect to coordinator and learner(s).

For commands and running workflows, see [RUN.md](./assets/RUN.md).

### Features

- Distributed actor-learner training with multiple C++ rollout workers and Python learner
- IMPALA/V-trace training loop for asynchronous off-policy correction
- Online BC, RL + BC anchor, and offline BC warm start support
- Continuous Ackermann control with ONNX export and worker-side deployment
- 2.5D simulator with BVH/SAT collision, raycasting, LiDAR-like observations, and VTK visualization
- Reward shaping for progress, goal completion, collisions, reverse usage, steering persistence, and stall behavior
- Docker and native local flows for reproducible demos

| Component | Stack | Responsibility |
| --- | --- | --- |
| Worker | C++23, CMake, gRPC, ONNX Runtime | Runs simulator instances, evaluates policies, emits rollouts |
| Learner | Python, PyTorch, gRPC | Optimizes policy/value losses and exports ONNX checkpoints |
| Coordinator | Python, gRPC | Tracks active learners and workers, heartbeats, endpoints |
| Simulator | C++23 | 2.5D geometry, collision, raycasting, dynamic actors, reward telemetry |
| Config Generator | Python, PyQt, PyVista | Builds YAML maps and scenario profiles |

## 2. Workflows

The repository supports three learning modes:

1. **Online BC**
   - expert workers generate `is_expert=1` rollouts
   - learner trains the neural actor directly from expert actions
   - no RL loss is used

2. **RL + BC Anchor**
   - neural workers provide RL rollouts
   - expert workers provide expert-tagged rollouts
   - learner optimizes RL loss plus a BC term on expert trajectories

3. **Pure RL**
   - learner runs V-trace actor-critic updates on neural-worker rollouts only

### Policy And Value Outputs

The policy produces continuous Ackermann actions (speed, steering). The actor outputs Gaussian parameters in latent space ($\mu$, $\log \sigma$):

The worker maps latent actions to bounded control:

$$
u \sim \mathcal{N}(\mu, \sigma), \qquad
a = \tanh(u) \odot s_{\text{action}}
$$

The learner also predicts a scalar value estimate `V(s)`.

### V-trace

The RL path uses the standard IMPALA correction:

$$
\rho_t = \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}
$$

with clipped importance weights:

$$
\bar{\rho}_t = \min(\rho_t, \rho_{\max}), \qquad
c_t = \min(\rho_t, c_{\max})
$$

Temporal-difference correction:

$$
\delta_t=\bar{\rho}_t \left(r_t + \gamma V(s_{t+1}) - V(s_t)\right)
$$

Recursive target:

$$
v_t=V(s_t) + \delta_t + \gamma c_t \left(v_{t+1} - V(s_{t+1})\right)
$$

Policy-gradient advantage:

$$
A_t=\bar{\rho}_t \left(r_t + \gamma v_{t+1} - V(s_t)\right)
$$

- workers act asynchronously, so the behavior policy that generated a rollout can be stale by the time the learner updates
- V-trace keeps those off-policy rollouts usable by correcting the target with clipped importance weights
- this is the key reason the system can scale to multiple independent C++ workers without requiring strict on-policy synchronization

### RL Loss

The learner optimizes:

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{actor}} + c_v \mathcal{L}_{\text{value}} - c_e \mathcal{L}_{\text{entropy}} + c_{bc}\mathcal{L}_{bc}
$$

where:

- $\mathcal{L}_{\text{actor}}$ is the policy-gradient term built from V-trace advantages
- $\mathcal{L}_{\text{value}}$ regresses the critic toward corrected value targets
- $\mathcal{L}_{\text{entropy}}$ prevents the policy from collapsing too early
- $\mathcal{L}_{bc}$ is only active in modes that include supervised anchoring

### Behavior Cloning

Offline BC and online BC both regress bounded applied actions, not latent Gaussian density:

$$
\mathcal{L}_{bc}=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert a^{(i)}_{\text{pred}} - a^{(i)}_{\text{expert}} \right\rVert_2^2
$$

This was chosen because it is easier to debug and matched the project’s Ackermann control behavior better than latent-space likelihood fitting.

Intuition:

- the expert controller already encodes a workable maneuver policy
- BC turns that controller into a neural policy by supervised regression on expert state-action pairs
- using bounded applied actions instead of latent density fitting makes it easier to reason about what the policy is actually being asked to match

In this repo, BC serves two purposes:

1. **policy initialization**
   - get a neural policy that already drives in roughly expert-like ways

2. **policy anchoring**
   - keep RL from drifting too quickly into reward-exploiting but qualitatively bad behaviors such as orbiting

### RL + BC Anchor

When `rl_bc_anchor` is used, the learner mixes RL improvement and imitation:

$$
\mathcal{L}_{\text{anchor}}=\mathcal{L}_{\text{RL}} + c_{bc}\mathcal{L}_{bc}
$$

This mode exists because:

- pure RL improves return, but it can exploit weak spots in the reward
- BC anchoring keeps the policy near the expert manifold while still letting RL optimize the task
- in practice, this is useful when the expert has good local behavior but the reward alone is not enough to preserve it
