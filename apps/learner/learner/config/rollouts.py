from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from proto import rollouts_pb2
from array import array


@dataclass
class RolloutStep:
    action: List[float]
    raw_action: List[float]
    reward: float
    reward_terms: Dict[str, Dict[str, float]]
    terminal: bool
    behaviour_log_prob: float
    timestamp: float
    observation: List[float] = field(default_factory=list)


@dataclass
class RolloutSample:
    sim_id: int
    rollout_id: int
    policy_version: int
    episode_return: float
    episode_terminal: bool
    steps: List[RolloutStep]
    bootstrap_observation: Optional[List[float]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    recurrent_hidden: Optional[List[float]] = None
    recurrent_cell: Optional[List[float]] = None
    recurrent_shape: Optional[List[int]] = None


def rollouts_from_proto(batch: rollouts_pb2.RolloutBatch) -> List[RolloutSample]:
    parsed: List[RolloutSample] = []
    for rollout_msg in batch.rollouts:
        steps: List[RolloutStep] = []
        for step_msg in rollout_msg.steps:
            actions = list(step_msg.action.continuous)
            observation = array("f", step_msg.observation.features)
            steps.append(
                RolloutStep(
                    action=actions,
                    raw_action=list(step_msg.raw_action),
                    reward=step_msg.reward,
                    reward_terms={
                        key: {
                            "raw": float(value.raw),
                            "weighted": float(value.weighted),
                        }
                        for key, value in step_msg.reward_terms.items()
                    },
                    terminal=step_msg.terminal,
                    behaviour_log_prob=step_msg.action.log_prob,
                    timestamp=step_msg.timestamp_sec,
                    observation=observation,
                )
            )
        if not steps:
            continue
        metrics: Dict[str, float] = {}
        for key, value in rollout_msg.metrics.items():
            metrics[key] = float(value)
        recurrent_hidden: Optional[List[float]] = None
        recurrent_cell: Optional[List[float]] = None
        recurrent_shape: Optional[List[int]] = None
        if rollout_msg.HasField("initial_recurrent_state"):
            recurrent = rollout_msg.initial_recurrent_state
            recurrent_hidden = list(recurrent.hidden)
            recurrent_cell = list(recurrent.cell)
            recurrent_shape = list(recurrent.shape)
        parsed.append(
            RolloutSample(
                sim_id=rollout_msg.actor.sim_id,
                rollout_id=rollout_msg.actor.rollout_id,
                policy_version=rollout_msg.policy_version,
                episode_return=rollout_msg.episode_return,
                episode_terminal=rollout_msg.episode_terminal,
                steps=steps,
                metrics=metrics,
                recurrent_hidden=recurrent_hidden,
                recurrent_cell=recurrent_cell,
                recurrent_shape=recurrent_shape,
            )
        )
    return parsed
