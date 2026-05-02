#include "worker/rollout_recorder.hpp"

#include <chrono>
#include <cstddef>
#include <format>
#include <fstream>
#include <functional>
#include <optional>

#include "rollouts.pb.h"
#include "worker/observation_encoder.hpp"

namespace ddrl::worker {

namespace {

void fill_recurrent_state_proto(
  const std::optional<RecurrentStateSnapshot>& state, ddrl::comm::Rollout* proto
)
{
  if (!state || proto == nullptr || state->hidden.empty() || state->cell.empty()) {
    return;
  }
  auto* recurrent = proto->mutable_initial_recurrent_state();
  for (float value : state->hidden) {
    recurrent->add_hidden(value);
  }
  for (float value : state->cell) {
    recurrent->add_cell(value);
  }
  for (int32_t value : state->shape) {
    recurrent->add_shape(value);
  }
}

ddrl::comm::RolloutBatch to_proto_batch(
  const std::vector<Rollout>& rollouts, uint32_t worker_id, uint64_t batch_sequence,
  const config::RLPolicyConfig::Architecture& architecture
)
{
  ddrl::comm::RolloutBatch batch_msg;
  batch_msg.set_batch_id(batch_sequence);
  const auto created_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
  )
                            .count();
  batch_msg.set_created_unix_ms(static_cast<uint64_t>(created_ms));

  for (const auto& rollout : rollouts) {
    auto* proto = batch_msg.add_rollouts();
    auto* actor = proto->mutable_actor();
    actor->set_worker_id(worker_id);
    actor->set_sim_id(rollout.sim_id);
    actor->set_rollout_id(rollout.rollout_id);

    proto->set_policy_version(rollout.policy_version);
    proto->set_episode_return(rollout.episode_return);
    proto->set_episode_terminal(rollout.is_terminal);
    fill_recurrent_state_proto(rollout.initial_recurrent_state, proto);

    for (const auto& step : rollout.steps) {
      auto* step_proto = proto->add_steps();
      auto* obs = step_proto->mutable_observation();

      const auto& features = !step.encoded_observation_features.empty()
                               ? step.encoded_observation_features
                               : encode_observation(step.state, architecture).features;
      for (float v : features) {
        obs->add_features(v);
      }

      auto* action = step_proto->mutable_action();
      action->add_continuous(static_cast<float>(step.action.normalized_speed_action));
      action->add_continuous(static_cast<float>(step.action.normalized_steer_action));
      action->set_log_prob(step.behaviour_log_prob);
      if (step.action.has_raw_policy_action) {
        step_proto->add_raw_action(static_cast<float>(step.action.raw_speed_action));
        step_proto->add_raw_action(static_cast<float>(step.action.raw_steer_action));
      }

      step_proto->set_reward(step.reward);
      step_proto->set_terminal(step.is_terminal);
      step_proto->set_timestamp_sec(step.action.timestamp);

      for (const auto& [name, value] : step.reward_terms) {
        auto& proto_term = (*step_proto->mutable_reward_terms())[name];
        proto_term.set_raw(value.raw);
        proto_term.set_weighted(value.weighted);
      }
    }

    (*proto->mutable_metrics())["steps"] = static_cast<double>(rollout.steps.size());
    (*proto->mutable_metrics())["is_expert"] = rollout.is_expert ? 1.0 : 0.0;
  }

  return batch_msg;
}

} // namespace

RolloutRecorder::RolloutRecorder(Params params)
    : output_dir_(std::move(params.output_dir)),
      max_files_(params.max_files),
      logger_(std::move(params.logger))
{
  std::error_code ec;
  std::filesystem::create_directories(output_dir_, ec);
  if (ec && logger_) {
    DDRL_LOG_WARN(logger_, "Failed to create demo output dir '{}': {}", output_dir_.string(), ec.message());
  }
}

void RolloutRecorder::record_batch(
  const std::vector<Rollout>& rollouts, uint32_t worker_id, uint64_t batch_sequence,
  const config::RLPolicyConfig::Architecture& architecture
)
{
  if (rollouts.empty()) {
    return;
  }
  if (max_files_ > 0 && written_ >= max_files_) {
    return;
  }

  auto msg = to_proto_batch(rollouts, worker_id, batch_sequence, architecture);

  const auto ts = msg.created_unix_ms();
  const auto filename =
    std::format("rollout_batch_{:06d}_{}.pb", static_cast<int>(msg.batch_id()), ts);
  const auto path = output_dir_ / filename;

  std::ofstream out(path, std::ios::binary);
  if (!out.good()) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Failed to open demo file '{}'", path.string());
    }
    return;
  }

  if (!msg.SerializeToOstream(&out)) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Failed to serialize demo batch to '{}'", path.string());
    }
    return;
  }
  ++written_;
}

} // namespace ddrl::worker
