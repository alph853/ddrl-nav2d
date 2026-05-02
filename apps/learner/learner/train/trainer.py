from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import torch
from loguru import logger

from proto import policy_pb2
from learner.config.learner import LearnerConfig
from learner.config.rollouts import RolloutSample
from learner.policy import PolicyModule, build_policy, export_policy_to_onnx
from learner.train.batching import LearnerBatch, prepare_minibatch
from learner.train.vtrace import VTraceConfig, vtrace_from_logits


class PolicyTrainer:
    """Encapsulates the neural policy, optimizer, and loss computation."""

    def __init__(self, config: LearnerConfig, device: torch.device) -> None:
        self._config = config
        self._device = device
        self._action_dim = max(int(config.action_dim), 0)
        if self._action_dim <= 0:
            raise ValueError(f"Invalid learner action_dim={config.action_dim}.")
        self._model: PolicyModule = build_policy(
            config, self._action_dim, device=self._device
        ).to(self._device)
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.lr, eps=1e-5
        )
        self._logger = logger.bind(logger_name="ddrl.trainer")
        self._opt_steps = 0
        self._bc_decay_steps = 0
        self._perf_log_every = 50
        self._logger.info(
            "initialized model (policy_model={}, observation_dim={}, action_dim={}, hidden={}, device={})",
            self._config.policy_model,
            self._config.observation_dim,
            self._action_dim,
            self._config.lstm_hidden_dim
            if self._config.policy_model == "lstm"
            else self._config.mlp_hidden_dim,
            self._device,
        )

    def optimize(self, rollouts: Sequence[RolloutSample]) -> Dict[str, float]:
        """Accumulate gradients from a batch of rollouts and apply one optimizer step."""
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_bc = 0.0
        total_steps = 0
        rho_stats: List[float] = []
        max_rho = 0.0
        bc_coef_used = 0.0

        t0 = time.perf_counter()
        batch = prepare_minibatch(rollouts, self._config, self._device)
        self._log_prepared_batch(batch)
        t1 = time.perf_counter()
        self._update_observation_stats(batch.observation, batch.lengths)

        recurrent_state = batch.recurrent_state or self._model.initial_recurrent_state(
            len(rollouts), self._device
        )
        mean, log_std, values, _recurrent_out = self._model(
            batch.observation,
            recurrent_state,
        )
        t2 = time.perf_counter()
        self._assert_finite("mean", mean)
        self._assert_finite("log_std", log_std)
        self._assert_finite("values", values)

        is_terminal = torch.tensor(
            [
                bool(rollout.steps and rollout.steps[-1].terminal)
                for rollout in rollouts
            ],
            dtype=torch.bool,
            device=self._device,
        )

        batch_size, max_len = values.shape
        if max_len == 0 or batch_size == 0:
            total_loss = values.new_tensor(0.0)
            actor_loss = values.new_tensor(0.0)
            value_loss = values.new_tensor(0.0)
            entropy_loss = values.new_tensor(0.0)
            bc_loss = values.new_tensor(0.0)
            total_steps = 0
            mean_rho = 0.0
            max_rho = 0.0
            entropy_val = 0.0
        else:
            lengths_tensor = torch.tensor(batch.lengths, device=values.device)
            step_idx = torch.arange(max_len, device=values.device)[None, :]
            mask = (step_idx < lengths_tensor[:, None]).float()
            rl_mask = mask * batch.rl_action_mask
            rl_valid = rl_mask.sum()

            target_log_probs, entropy = self._model.action_log_prob(
                mean,
                log_std,
                batch.actions,
                batch.raw_actions,
                batch.raw_action_mask,
            )
            entropy = entropy * rl_mask

            actor_loss = values.new_tensor(0.0)
            value_loss = values.new_tensor(0.0)
            entropy_loss = values.new_tensor(0.0)
            mean_rho = 0.0
            max_rho = 0.0
            entropy_val = 0.0

            if float(rl_valid.item()) > 0.0:
                last_idx = torch.clamp(lengths_tensor - 1, min=0)
                batch_idx = torch.arange(batch_size, device=values.device)
                last_values = values[batch_idx, last_idx]
                bootstrap_value = torch.where(
                    is_terminal, values.new_zeros(batch_size), last_values.detach()
                )

                discounts = self._config.gamma * (1.0 - batch.terminals) * rl_mask
                vtrace = vtrace_from_logits(
                    rewards=batch.rewards * rl_mask,
                    discounts=discounts,
                    values=values,
                    bootstrap_value=bootstrap_value,
                    behaviour_log_probs=batch.behaviour_log_probs,
                    target_log_probs=target_log_probs,
                    mask=rl_mask,
                    config=VTraceConfig(
                        clip_rho=self._config.clip_rho,
                        clip_c=self._config.clip_c,
                    ),
                )
                vs = vtrace.values
                pg_adv = vtrace.policy_advantages
                rhos = vtrace.rhos
                self._assert_finite("vtrace.vs", vs)
                self._assert_finite("vtrace.pg_adv", pg_adv)
                self._assert_finite("vtrace.rhos", rhos)

                valid = rl_valid.clamp(min=1.0)
                actor_loss = (
                    -(pg_adv.detach() * target_log_probs * rl_mask).sum() / valid
                )
                value_loss = (
                    0.5 * ((vs.detach() - values).pow(2) * rl_mask).sum() / valid
                )
                entropy_loss = entropy.sum() / valid

                rhos_masked = rhos * rl_mask
                mean_rho = float(rhos_masked.sum().item() / valid.item())
                max_rho = float(rhos_masked.masked_select(rl_mask > 0.5).max().item())
                entropy_val = float(entropy_loss.detach().item())

            bc_loss = values.new_tensor(0.0)
            bc_coef = (
                self._current_bc_coef()
                if self._config.trainer_mode == "rl_bc_anchor"
                else 0.0
            )
            bc_coef_used = bc_coef
            bc_valid = batch.bc_action_mask.sum()
            if bc_coef > 0.0 and float(bc_valid.item()) > 0.0:
                pred_actions = self._model.mean_to_action(mean)
                per_step_mse = (pred_actions - batch.bc_actions).pow(2).mean(dim=-1)
                bc_loss = (per_step_mse * batch.bc_action_mask).sum() / bc_valid
                self._assert_finite("bc_loss", bc_loss)

            total_loss = (
                actor_loss
                + self._config.value_loss_coef * value_loss
                - self._config.entropy_coef * entropy_loss
                + bc_coef * bc_loss
            )

            total_steps = int(mask.sum().item())

        if not total_loss.requires_grad:
            total_loss = total_loss + values.sum() * 0.0
        self._assert_finite("loss.total", total_loss)
        total_actor = total_actor + actor_loss.detach()
        total_value = total_value + value_loss.detach()
        total_entropy = total_entropy + entropy_loss.detach()
        total_bc = total_bc + bc_loss.detach()
        rho_stats.append(mean_rho)
        t3 = time.perf_counter()

        total_loss.backward()
        grad_norm_sq = 0.0
        grad_params = 0
        max_abs_grad = 0.0
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                self._assert_finite(f"grad:{name}", param.grad)
                grad = param.grad.detach()
                grad_norm_sq += float(torch.sum(grad * grad).item())
                grad_params += grad.numel()
                max_abs_grad = max(max_abs_grad, float(grad.abs().max().item()))
        if self._config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self._config.grad_clip
            )
        self._optimizer.step()
        t4 = time.perf_counter()
        self._opt_steps += 1
        if (
            self._config.trainer_mode == "rl_bc_anchor"
            and int(self._config.bc_decay_updates) > 0
        ):
            self._bc_decay_steps += 1
        if self._opt_steps % self._perf_log_every == 0:
            prep_ms = (t1 - t0) * 1000.0
            fwd_ms = (t2 - t1) * 1000.0
            loss_ms = (t3 - t2) * 1000.0
            bwd_ms = (t4 - t3) * 1000.0
            self._logger.info(
                "perf update={} prep_ms={:.2f} fwd_ms={:.2f} loss_ms={:.2f} bwd_ms={:.2f} rollouts={} steps={}",
                self._opt_steps,
                prep_ms,
                fwd_ms,
                loss_ms,
                bwd_ms,
                len(rollouts),
                total_steps,
            )
            with torch.no_grad():
                lengths_tensor = torch.tensor(batch.lengths, device=self._device)
                max_len = mean.shape[1]
                step_idx = torch.arange(max_len, device=self._device)[None, :]
                mask = step_idx < lengths_tensor[:, None]
                if mask.any():
                    speed_mu = mean[..., 0][mask].mean().item()
                    steer_mu = (
                        mean[..., 1][mask].mean().item() if mean.shape[-1] > 1 else 0.0
                    )
                    speed_log_std = log_std[..., 0][mask].mean().item()
                    steer_log_std = (
                        log_std[..., 1][mask].mean().item()
                        if log_std.shape[-1] > 1
                        else 0.0
                    )
                    actions = batch.actions
                    action_speed = actions[..., 0][mask].mean().item()
                    action_steer = (
                        actions[..., 1][mask].mean().item()
                        if actions.shape[-1] > 1
                        else 0.0
                    )
                    self._logger.info(
                        "policy_stats update={} mean_speed_mu={:.3f} mean_steer_mu={:.3f} "
                        "mean_speed_log_std={:.3f} mean_steer_log_std={:.3f} "
                        "mean_action_speed={:.3f} mean_action_steer={:.3f} "
                        "entropy={:.3f}",
                        self._opt_steps,
                        speed_mu,
                        steer_mu,
                        speed_log_std,
                        steer_log_std,
                        action_speed,
                        action_steer,
                        entropy_val,
                    )

        actor = total_actor.item()
        critic = total_value.item()
        entropy = total_entropy.item()
        bc = total_bc.item()
        bc_speed_mae = 0.0
        bc_steer_mae = 0.0
        bc_valid_for_mae = batch.bc_action_mask.sum().clamp(min=1.0)
        if float(batch.bc_action_mask.sum().item()) > 0.0:
            with torch.no_grad():
                pred_actions = self._model.mean_to_action(mean)
                bc_speed_mae = float(
                    (
                        (pred_actions[..., 0] - batch.bc_actions[..., 0]).abs()
                        * batch.bc_action_mask
                    )
                    .sum()
                    .div(bc_valid_for_mae)
                    .item()
                )
                if pred_actions.shape[-1] > 1:
                    bc_steer_mae = float(
                        (
                            (pred_actions[..., 1] - batch.bc_actions[..., 1]).abs()
                            * batch.bc_action_mask
                        )
                        .sum()
                        .div(bc_valid_for_mae)
                        .item()
                    )
        rl_steps = float(batch.rl_action_mask.sum().item())
        bc_steps = float(batch.bc_action_mask.sum().item())
        expert_rollouts = sum(
            1
            for rollout in rollouts
            if float(rollout.metrics.get("is_expert", 0.0)) >= 0.5
        )
        neural_rollouts = len(rollouts) - expert_rollouts
        behaviour_valid = batch.rl_action_mask > 0.5
        behaviour_mean = 0.0
        behaviour_min = 0.0
        behaviour_max = 0.0
        if bool(behaviour_valid.any()):
            behaviour_vals = batch.behaviour_log_probs[behaviour_valid]
            behaviour_mean = float(behaviour_vals.mean().item())
            behaviour_min = float(behaviour_vals.min().item())
            behaviour_max = float(behaviour_vals.max().item())
        action_speed_mean = (
            float((batch.actions[..., 0]).mean().item())
            if batch.actions.numel() > 0
            else 0.0
        )
        action_steer_mean = (
            float((batch.actions[..., 1]).mean().item())
            if batch.actions.shape[-1] > 1
            else 0.0
        )
        raw_speed_mean = (
            float((batch.raw_actions[..., 0]).mean().item())
            if batch.raw_actions.numel() > 0
            else 0.0
        )
        raw_steer_mean = (
            float((batch.raw_actions[..., 1]).mean().item())
            if batch.raw_actions.shape[-1] > 1
            else 0.0
        )
        reward_mean = (
            float(batch.rewards.mean().item()) if batch.rewards.numel() > 0 else 0.0
        )
        reward_min = (
            float(batch.rewards.min().item()) if batch.rewards.numel() > 0 else 0.0
        )
        reward_max = (
            float(batch.rewards.max().item()) if batch.rewards.numel() > 0 else 0.0
        )
        policy_mean_speed = 0.0
        policy_mean_steer = 0.0
        policy_sigma_speed = 0.0
        policy_sigma_steer = 0.0
        with torch.no_grad():
            lengths_tensor = torch.tensor(batch.lengths, device=self._device)
            max_len = mean.shape[1]
            step_idx = torch.arange(max_len, device=self._device)[None, :]
            policy_mask = step_idx < lengths_tensor[:, None]
            if bool(policy_mask.any()):
                policy_mean_speed = float(mean[..., 0][policy_mask].mean().item())
                policy_sigma_speed = float(
                    log_std[..., 0][policy_mask].exp().mean().item()
                )
                if mean.shape[-1] > 1:
                    policy_mean_steer = float(mean[..., 1][policy_mask].mean().item())
                    policy_sigma_steer = float(
                        log_std[..., 1][policy_mask].exp().mean().item()
                    )
        grad_norm = math.sqrt(max(grad_norm_sq, 0.0))
        self._logger.debug(
            "loss={:.4f} actor={:.4f} value={:.4f} entropy={:.4f} bc={:.4f} steps={} rho={:.3f}",
            total_loss.item(),
            actor,
            critic,
            entropy,
            bc,
            total_steps,
            sum(rho_stats) / max(len(rho_stats), 1),
        )

        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor),
            "value_loss": float(critic),
            "entropy": float(entropy),
            "bc_loss": float(bc),
            "bc_coef": float(bc_coef_used),
            "bc_decay_steps": float(self._bc_decay_steps),
            "steps": float(total_steps),
            "mean_rho": float(sum(rho_stats) / max(len(rho_stats), 1)),
            "max_rho": float(max_rho),
            "expert_rollouts": float(expert_rollouts),
            "neural_rollouts": float(neural_rollouts),
            "rl_steps": float(rl_steps),
            "bc_steps": float(bc_steps),
            "behaviour_log_prob_mean": float(behaviour_mean),
            "behaviour_log_prob_min": float(behaviour_min),
            "behaviour_log_prob_max": float(behaviour_max),
            "action_speed_mean": float(action_speed_mean),
            "action_steer_mean": float(action_steer_mean),
            "raw_speed_mean": float(raw_speed_mean),
            "raw_steer_mean": float(raw_steer_mean),
            "reward_mean": float(reward_mean),
            "reward_min": float(reward_min),
            "reward_max": float(reward_max),
            "grad_norm": float(grad_norm),
            "max_abs_grad": float(max_abs_grad),
            "grad_params": float(grad_params),
            "bc_speed_mae": float(bc_speed_mae),
            "bc_steer_mae": float(bc_steer_mae),
            "policy_mean_speed": float(policy_mean_speed),
            "policy_mean_steer": float(policy_mean_steer),
            "policy_sigma_speed": float(policy_sigma_speed),
            "policy_sigma_steer": float(policy_sigma_steer),
        }

    def _current_bc_coef(self) -> float:
        coef = max(0.0, float(self._config.bc_coef))
        decay_updates = int(self._config.bc_decay_updates)
        if coef == 0.0 or decay_updates <= 0:
            return coef
        floor = min(coef, max(0.0, float(self._config.bc_floor_coef)))
        progress = min(self._bc_decay_steps, decay_updates) / float(decay_updates)
        return max(floor, coef * max(0.0, 1.0 - progress))

    def optimize_bc(self, rollouts: Sequence[RolloutSample]) -> Dict[str, float]:
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        batch = prepare_minibatch(rollouts, self._config, self._device)
        self._log_prepared_batch(batch)
        self._assert_finite("observation", batch.observation)
        self._assert_finite("bc_actions", batch.bc_actions)
        if float(batch.bc_action_mask.sum().item()) <= 0.0:
            raise ValueError(
                "online BC batch did not contain any expert rollout actions"
            )
        self._update_observation_stats(batch.observation, batch.lengths)
        recurrent_state = batch.recurrent_state or self._model.initial_recurrent_state(
            len(rollouts), self._device
        )
        mean, log_std, _values, _recurrent_out = self._model(
            batch.observation,
            recurrent_state,
        )
        self._assert_finite("mean", mean)
        pred_actions = self._model.mean_to_action(mean)
        mask = batch.bc_action_mask
        valid = mask.sum().clamp(min=1.0)
        per_step_mse = (pred_actions - batch.bc_actions).pow(2).mean(dim=-1)
        loss = (per_step_mse * mask).sum() / valid
        self._assert_finite("bc_loss", loss)
        loss.backward()
        if self._config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self._config.grad_clip
            )
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                self._assert_finite(f"grad:{name}", param.grad)
        self._optimizer.step()
        self._opt_steps += 1

        expert_rollouts = sum(
            1
            for rollout in rollouts
            if float(rollout.metrics.get("is_expert", 0.0)) >= 0.5
        )
        neural_rollouts = len(rollouts) - expert_rollouts
        grad_norm_sq = 0.0
        grad_params = 0
        max_abs_grad = 0.0
        for param in self._model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad_norm_sq += float(torch.sum(grad * grad).item())
            grad_params += grad.numel()
            max_abs_grad = max(max_abs_grad, float(grad.abs().max().item()))
        grad_norm = math.sqrt(max(grad_norm_sq, 0.0))

        speed_mae = (
            ((pred_actions[..., 0] - batch.bc_actions[..., 0]).abs()) * mask
        ).sum() / valid
        steer_mae = (
            ((pred_actions[..., 1] - batch.bc_actions[..., 1]).abs()) * mask
        ).sum() / valid
        policy_mean_speed = 0.0
        policy_mean_steer = 0.0
        policy_sigma_speed = 0.0
        policy_sigma_steer = 0.0
        with torch.no_grad():
            lengths_tensor = torch.tensor(batch.lengths, device=self._device)
            max_len = mean.shape[1]
            step_idx = torch.arange(max_len, device=self._device)[None, :]
            policy_mask = step_idx < lengths_tensor[:, None]
            if bool(policy_mask.any()):
                policy_mean_speed = float(mean[..., 0][policy_mask].mean().item())
                policy_sigma_speed = float(
                    log_std[..., 0][policy_mask].exp().mean().item()
                )
                if mean.shape[-1] > 1:
                    policy_mean_steer = float(mean[..., 1][policy_mask].mean().item())
                    policy_sigma_steer = float(
                        log_std[..., 1][policy_mask].exp().mean().item()
                    )
        return {
            "loss": float(loss.item()),
            "actor_loss": float(loss.item()),
            "value_loss": 0.0,
            "entropy": 0.0,
            "bc_loss": float(loss.item()),
            "bc_coef": 0.0,
            "bc_decay_steps": float(self._bc_decay_steps),
            "steps": float(valid.item()),
            "mean_rho": 0.0,
            "max_rho": 0.0,
            "expert_rollouts": float(expert_rollouts),
            "neural_rollouts": float(neural_rollouts),
            "rl_steps": 0.0,
            "bc_steps": float(valid.item()),
            "bc_speed_mae": float(speed_mae.item()),
            "bc_steer_mae": float(steer_mae.item()),
            "grad_norm": float(grad_norm),
            "max_abs_grad": float(max_abs_grad),
            "grad_params": float(grad_params),
            "policy_mean_speed": float(policy_mean_speed),
            "policy_mean_steer": float(policy_mean_steer),
            "policy_sigma_speed": float(policy_sigma_speed),
            "policy_sigma_steer": float(policy_sigma_steer),
        }

    def build_policy_push(
        self, version: int, stats: Dict[str, float]
    ) -> policy_pb2.PolicyPush:
        """Serialize the policy state dict and metrics into a PolicyPush message."""
        push = policy_pb2.PolicyPush()
        push.version = version
        with torch.no_grad():
            push.onnx_model = export_policy_to_onnx(self._model, self._config)
        return push

    @property
    def model_state(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    @property
    def optimizer_state(self) -> Dict[str, torch.Tensor]:
        return self._optimizer.state_dict()

    @property
    def bc_decay_steps(self) -> int:
        return self._bc_decay_steps

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def opt_steps(self) -> int:
        return self._opt_steps

    def restore(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, torch.Tensor],
        opt_steps: int = 0,
        restore_optimizer: bool = True,
    ) -> None:
        """Restore model, optimizer, and optimizer-step state from a checkpoint."""
        self._model.load_state_dict(model_state)
        reset_action_std = bool(self._config.reset_action_std_on_resume)
        if reset_action_std:
            with torch.no_grad():
                init_std = max(float(self._config.action_init_std), 1e-6)
                self._model.log_std.fill_(math.log(init_std))
        self._bc_decay_steps = 0
        if not restore_optimizer:
            self._opt_steps = 0
            self._logger.info(
                "restored learner model weights with fresh optimizer (action_dim={}, reset_action_std={}, action_init_std={})",
                self._action_dim,
                reset_action_std,
                self._config.action_init_std,
            )
            return
        restored_optimizer = True
        try:
            self._optimizer.load_state_dict(optimizer_state)
            if reset_action_std:
                log_std_state = self._optimizer.state.get(self._model.log_std)
                if log_std_state is not None:
                    log_std_state.clear()
        except ValueError as exc:
            restored_optimizer = False
            self._logger.warning(
                "checkpoint optimizer state is incompatible with current learner optimizer; "
                "using fresh optimizer instead ({})",
                exc,
            )
        self._opt_steps = max(0, int(opt_steps)) if restored_optimizer else 0
        if restored_optimizer:
            self._logger.info(
                "restored learner state (action_dim={}, opt_steps={}, reset_action_std={}, action_init_std={})",
                self._action_dim,
                self._opt_steps,
                reset_action_std,
                self._config.action_init_std,
            )
        else:
            self._logger.info(
                "restored learner model weights with fresh optimizer (action_dim={}, reset_action_std={}, action_init_std={})",
                self._action_dim,
                reset_action_std,
                self._config.action_init_std,
            )

    # ------------------------------------------------------------------ internals
    def _log_prepared_batch(self, batch: LearnerBatch) -> None:
        if batch.observation.ndim != 3:
            return
        if batch.observation.shape[-1] != self._config.observation_dim:
            raise ValueError(
                f"observation_tensor dim mismatch: expected {self._config.observation_dim}, "
                f"got {batch.observation.shape[-1]}"
            )
        if batch.observation.shape[0] == 0 or batch.observation.shape[1] == 0:
            return
        sample_obs = (
            batch.observation[0, 0, : min(8, batch.observation.shape[-1])]
            .detach()
            .cpu()
            .numpy()
        )
        self._logger.bind(logger_name="ddrl.learner").opt(lazy=True).debug(
            "observation_tensor sample = {}",
            lambda: sample_obs.tolist(),
        )

    def _update_observation_stats(
        self, observation: torch.Tensor, lengths: List[int]
    ) -> None:
        if observation.ndim != 3:
            return
        max_len = observation.shape[1]
        lengths_tensor = torch.tensor(lengths, device=observation.device)
        step_idx = torch.arange(max_len, device=observation.device)[None, :]
        mask = (step_idx < lengths_tensor[:, None]).float()
        self._model.update_observation_stats(observation, mask)

    def _assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if torch.isfinite(tensor).all():
            return
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_vals = detached[finite_mask]
        min_val = (
            float(finite_vals.min().item()) if finite_vals.numel() > 0 else float("nan")
        )
        max_val = (
            float(finite_vals.max().item()) if finite_vals.numel() > 0 else float("nan")
        )
        raise ValueError(
            f"non-finite tensor '{name}' detected: shape={tuple(detached.shape)} "
            f"finite={int(finite_mask.sum().item())}/{detached.numel()} "
            f"min={min_val} max={max_val}"
        )
