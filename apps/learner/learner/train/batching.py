from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from learner.config.learner import LearnerConfig
from learner.config.rollouts import RolloutSample


@dataclass(frozen=True)
class LearnerBatch:
    observation: torch.Tensor
    actions: torch.Tensor
    bc_actions: torch.Tensor
    bc_action_mask: torch.Tensor
    rl_action_mask: torch.Tensor
    raw_actions: torch.Tensor
    raw_action_mask: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    behaviour_log_probs: torch.Tensor
    lengths: list[int]
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None


def prepare_minibatch(
    rollouts: Sequence[RolloutSample],
    config: LearnerConfig,
    device: torch.device,
) -> LearnerBatch:
    """Convert a minibatch of rollouts into padded tensors on the learner device."""
    batch_size = len(rollouts)
    lengths = [len(rollout.steps) for rollout in rollouts]
    max_len = max(lengths, default=0)
    action_dim = max(int(config.action_dim), 0)

    observation = np.zeros(
        (batch_size, max_len, config.observation_dim), dtype=np.float32
    )
    actions = np.zeros((batch_size, max_len, action_dim), dtype=np.float32)
    bc_actions = np.zeros((batch_size, max_len, action_dim), dtype=np.float32)
    bc_action_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    rl_action_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    raw_actions = np.zeros((batch_size, max_len, action_dim), dtype=np.float32)
    raw_action_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    rewards = np.zeros((batch_size, max_len), dtype=np.float32)
    terminals = np.zeros((batch_size, max_len), dtype=np.float32)
    behaviour_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
    recurrent_shape = _infer_recurrent_shape(rollouts)
    recurrent_hidden: np.ndarray | None = None
    recurrent_cell: np.ndarray | None = None
    if recurrent_shape is not None:
        layers, hidden_dim = recurrent_shape
        recurrent_hidden = np.zeros((layers, batch_size, hidden_dim), dtype=np.float32)
        recurrent_cell = np.zeros((layers, batch_size, hidden_dim), dtype=np.float32)

    for idx, rollout in enumerate(rollouts):
        rollout_is_expert = float(rollout.metrics.get("is_expert", 0.0)) >= 0.5
        if (
            recurrent_shape is not None
            and recurrent_hidden is not None
            and recurrent_cell is not None
            and rollout.recurrent_hidden is not None
            and rollout.recurrent_cell is not None
        ):
            layers, hidden_dim = recurrent_shape
            expected = layers * hidden_dim
            hidden = np.asarray(rollout.recurrent_hidden, dtype=np.float32)
            cell = np.asarray(rollout.recurrent_cell, dtype=np.float32)
            if hidden.size == expected and cell.size == expected:
                recurrent_hidden[:, idx, :] = hidden.reshape(layers, hidden_dim)
                recurrent_cell[:, idx, :] = cell.reshape(layers, hidden_dim)
        for t, step in enumerate(rollout.steps):
            dim = config.observation_dim
            arr = np.zeros(dim, dtype=np.float32)
            if step.observation:
                data = np.asarray(step.observation, dtype=np.float32)
                length = min(dim, data.shape[0])
                arr[:length] = data[:length]
            observation[idx, t] = arr

            action = np.asarray(step.action, dtype=np.float32)
            if action.size > 0:
                length = min(action_dim, action.shape[0])
                actions[idx, t, :length] = action[:length]
                if rollout_is_expert:
                    bc_actions[idx, t, :length] = action[:length]
                    bc_action_mask[idx, t] = 1.0
                else:
                    rl_action_mask[idx, t] = 1.0

            raw_action = np.asarray(step.raw_action, dtype=np.float32)
            if raw_action.size >= action_dim:
                raw_actions[idx, t, :action_dim] = raw_action[:action_dim]
                raw_action_mask[idx, t] = 1.0

            rewards[idx, t] = float(step.reward)
            terminals[idx, t] = 1.0 if step.terminal else 0.0
            behaviour_log_probs[idx, t] = float(step.behaviour_log_prob)

    reward_clip = max(float(config.reward_clip_abs), 0.0)
    if reward_clip > 0.0:
        np.clip(rewards, -reward_clip, reward_clip, out=rewards)

    recurrent_state = None
    if recurrent_hidden is not None and recurrent_cell is not None:
        recurrent_state = (
            torch.from_numpy(recurrent_hidden).to(device),
            torch.from_numpy(recurrent_cell).to(device),
        )

    return LearnerBatch(
        observation=torch.from_numpy(observation).to(device),
        actions=torch.from_numpy(actions).to(device),
        bc_actions=torch.from_numpy(bc_actions).to(device),
        bc_action_mask=torch.from_numpy(bc_action_mask).to(device),
        rl_action_mask=torch.from_numpy(rl_action_mask).to(device),
        raw_actions=torch.from_numpy(raw_actions).to(device),
        raw_action_mask=torch.from_numpy(raw_action_mask).to(device),
        rewards=torch.from_numpy(rewards).to(device),
        terminals=torch.from_numpy(terminals).to(device),
        behaviour_log_probs=torch.from_numpy(behaviour_log_probs).to(device),
        lengths=lengths,
        recurrent_state=recurrent_state,
    )


def _infer_recurrent_shape(
    rollouts: Sequence[RolloutSample],
) -> tuple[int, int] | None:
    for rollout in rollouts:
        shape = rollout.recurrent_shape
        if shape is None or len(shape) != 2:
            continue
        layers, hidden_dim = int(shape[0]), int(shape[1])
        if layers <= 0 or hidden_dim <= 0:
            continue
        expected = layers * hidden_dim
        if (
            rollout.recurrent_hidden is not None
            and rollout.recurrent_cell is not None
            and len(rollout.recurrent_hidden) == expected
            and len(rollout.recurrent_cell) == expected
        ):
            return layers, hidden_dim
    return None
