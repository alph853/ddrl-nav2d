#include "worker/rollout_stream_client.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <functional>
#include <numeric>
#include <optional>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <utility>

#include "worker/observation_encoder.hpp"

namespace ddrl::worker {

namespace {

constexpr int kMaxGrpcMessageBytes = 128 * 1024 * 1024;

void fill_step_proto(
  const TrajectoryStep& step, ddrl::comm::TrajectoryStep* proto,
  const config::RLPolicyConfig::Architecture& architecture
)
{
  if (proto == nullptr) {
    return;
  }

  auto* obs = proto->mutable_observation();

  const auto& features = !step.encoded_observation_features.empty()
                           ? step.encoded_observation_features
                           : encode_observation(step.state, architecture).features;
  for (float v : features) {
    obs->add_features(v);
  }

  auto* action = proto->mutable_action();
  action->add_continuous(static_cast<float>(step.action.normalized_speed_action));
  action->add_continuous(static_cast<float>(step.action.normalized_steer_action));
  action->set_log_prob(step.behaviour_log_prob);
  if (step.action.has_raw_policy_action) {
    proto->add_raw_action(static_cast<float>(step.action.raw_speed_action));
    proto->add_raw_action(static_cast<float>(step.action.raw_steer_action));
  }

  proto->set_reward(step.reward);
  proto->set_terminal(step.is_terminal);
  proto->set_timestamp_sec(step.action.timestamp);

  for (const auto& [name, value] : step.reward_terms) {
    auto& proto_term = (*proto->mutable_reward_terms())[name];
    proto_term.set_raw(value.raw);
    proto_term.set_weighted(value.weighted);
  }
}

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
      fill_step_proto(step, proto->add_steps(), architecture);
    }

    (*proto->mutable_metrics())["steps"] = static_cast<double>(rollout.steps.size());
    (*proto->mutable_metrics())["is_expert"] = rollout.is_expert ? 1.0 : 0.0;
  }

  return batch_msg;
}

} // namespace

RolloutStreamClient::RolloutStreamClient(Params params)
    : target_(std::move(params.target)),
      worker_id_(std::move(params.worker_id)),
      session_token_(std::move(params.session_token)),
      logger_(std::move(params.logger)),
      architecture_(std::move(params.architecture))
{
  stats_window_start_ = std::chrono::steady_clock::now();
}

void RolloutStreamClient::set_target(std::string target)
{
  std::lock_guard lock(mutex_);
  if (target_ == target) {
    return;
  }
  target_ = std::move(target);
  reset_stream_locked();
}

void RolloutStreamClient::set_architecture(config::RLPolicyConfig::Architecture architecture)
{
  std::lock_guard lock(mutex_);
  architecture_ = std::move(architecture);
}

bool RolloutStreamClient::send(const std::vector<Rollout>& batch)
{
  if (batch.empty()) {
    return true;
  }

  std::lock_guard lock(mutex_);
  if (target_.empty()) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Rollout stream target is empty; dropping batch");
    }
    return false;
  }
  if (!ensure_stream_locked()) {
    return false;
  }

  const auto batch_id  = ++batch_sequence_;
  auto       batch_msg = to_proto_batch(
    batch,
    static_cast<uint32_t>(std::hash<std::string>{}(worker_id_)),
    batch_id,
    architecture_
  );

  ddrl::comm::RolloutStreamRequest request;
  request.set_worker_id(worker_id_);
  request.set_session_token(session_token_);
  *request.mutable_batch() = std::move(batch_msg);
  const auto payload_bytes = static_cast<uint64_t>(request.ByteSizeLong());
  const auto steps_total =
    std::accumulate(batch.begin(), batch.end(), uint64_t{0}, [](uint64_t acc, const Rollout& r) {
      return acc + static_cast<uint64_t>(r.steps.size());
    });

  if (!stream_->Write(request)) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Rollout stream write failed; reconnecting");
    }
    reset_stream_locked();
    return false;
  }

  stats_window_batches_ += 1;
  stats_window_rollouts_ += static_cast<uint64_t>(batch.size());
  stats_window_steps_ += steps_total;
  stats_window_bytes_ += payload_bytes;

  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration<double>(now - stats_window_start_).count();
  constexpr double kStatsPeriodSeconds = 5.0;
  if (elapsed >= kStatsPeriodSeconds && logger_) {
    const double mb_per_sec = (static_cast<double>(stats_window_bytes_) / (1024.0 * 1024.0)) /
                              std::max(elapsed, 1e-6);
    const double rollouts_per_sec =
      static_cast<double>(stats_window_rollouts_) / std::max(elapsed, 1e-6);
    const double steps_per_sec =
      static_cast<double>(stats_window_steps_) / std::max(elapsed, 1e-6);
    const double avg_batch_bytes =
      stats_window_batches_ > 0 ? static_cast<double>(stats_window_bytes_) /
                                    static_cast<double>(stats_window_batches_)
                                : 0.0;
    const double avg_step_bytes =
      stats_window_steps_ > 0 ? static_cast<double>(stats_window_bytes_) /
                                   static_cast<double>(stats_window_steps_)
                               : 0.0;

    DDRL_LOG_INFO_THROTTLE(
      logger_,
      5000,
      "Rollout stream stats: batches={} rollouts={} steps={} serialized_bytes={} "
      "throughput={:.2f}MB/s rollouts/s={:.1f} steps/s={:.1f} "
      "avg_batch_bytes={:.0f} avg_step_bytes={:.1f}",
      stats_window_batches_,
      stats_window_rollouts_,
      stats_window_steps_,
      stats_window_bytes_,
      mb_per_sec,
      rollouts_per_sec,
      steps_per_sec,
      avg_batch_bytes,
      avg_step_bytes
    );
    stats_window_start_ = now;
    stats_window_batches_ = 0;
    stats_window_rollouts_ = 0;
    stats_window_steps_ = 0;
    stats_window_bytes_ = 0;
  }

  return true;
}

bool RolloutStreamClient::ensure_stream_locked()
{
  if (stream_) {
    return true;
  }

  if (target_.empty()) {
    return false;
  }

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(kMaxGrpcMessageBytes);
  args.SetMaxSendMessageSize(kMaxGrpcMessageBytes);

  channel_ = grpc::CreateCustomChannel(target_, grpc::InsecureChannelCredentials(), args);
  stub_    = ddrl::comm::RolloutStream::NewStub(channel_);
  context_ = std::make_unique<grpc::ClientContext>();
  stream_  = stub_->StreamRollouts(context_.get());
  if (!stream_) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "Failed to open rollout stream to {}", target_);
    }
    reset_stream_locked();
    return false;
  }

  if (logger_) {
    DDRL_LOG_INFO(logger_, "Connected rollout stream to {}", target_);
  }

  return true;
}

void RolloutStreamClient::reset_stream_locked()
{
  if (stream_) {
    stream_->WritesDone();
    stream_.reset();
  }
  if (context_) {
    context_.reset();
  }
  stub_.reset();
  channel_.reset();
}

} // namespace ddrl::worker
