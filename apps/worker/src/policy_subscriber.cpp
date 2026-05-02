#include "worker/policy_subscriber.hpp"

#include <chrono>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <utility>

#include "config/rl.hpp"
#include "policy.grpc.pb.h"
#include "logging/logging.hpp"
#include "worker/policy_model.hpp"

namespace ddrl::worker {

namespace {
constexpr std::chrono::milliseconds kReconnectDelay{1500};
}

PolicySubscriber::PolicySubscriber(Params params)
    : worker_id_(std::move(params.worker_id)),
      session_token_(std::move(params.session_token)),
      logger_(logging::get_logger("PolicySubscriber"))
{
}

PolicySubscriber::~PolicySubscriber()
{
  stop();
}

void PolicySubscriber::start()
{
  if (thread_.joinable()) {
    return;
  }
  stop_.store(false);
  thread_ = std::thread([this]() { this->run_loop(); });
}

void PolicySubscriber::stop()
{
  stop_.store(true);
  request_reconnect();
  target_cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void PolicySubscriber::set_target(std::string target)
{
  {
    std::lock_guard lock(target_mutex_);
    target_ = std::move(target);
  }
  target_cv_.notify_all();
  request_reconnect();
}

uint64_t PolicySubscriber::last_version() const
{
  return last_version_.load(std::memory_order_relaxed);
}

void PolicySubscriber::set_policy_config(const config::RLPolicyConfig* policy_config)
{
  policy_config_   = policy_config;
  warn_no_config_.store(false, std::memory_order_relaxed);
}

void PolicySubscriber::run_loop()
{
  while (!stop_.load()) {
    std::string target;
    {
      std::unique_lock lock(target_mutex_);
      target_cv_.wait(lock, [this]() { return stop_.load() || !target_.empty(); });
      if (stop_.load()) {
        break;
      }
      target = target_;
    }

    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    auto stub    = ddrl::comm::PolicyService::NewStub(channel);

    grpc::ClientContext ctx;
    {
      std::lock_guard lock(context_mutex_);
      active_context_ = &ctx;
    }

    if (logger_) {
      DDRL_LOG_INFO(logger_, "Connecting to learner at {} for policy stream", target);
    }

    auto stream = stub->StreamPolicyUpdates(&ctx);
    if (!stream) {
      if (logger_) {
        DDRL_LOG_ERROR(logger_, "Failed to create policy stream for {}", target);
      }
      {
        std::lock_guard lock(context_mutex_);
        active_context_ = nullptr;
      }
      std::this_thread::sleep_for(kReconnectDelay);
      continue;
    }

    ddrl::comm::PolicyStreamRequest subscribe_req;

    auto* sub = subscribe_req.mutable_subscribe();
    sub->set_worker_id(worker_id_);
    sub->set_session_token(session_token_);
    sub->set_last_version(last_version_.load(std::memory_order_relaxed));

    if (!stream->Write(subscribe_req)) {
      if (logger_) {
        DDRL_LOG_ERROR(logger_, "Failed to write subscription request to {}", target);
      }
      stream->WritesDone();
      auto status = stream->Finish();
      (void)status;
      {
        std::lock_guard lock(context_mutex_);
        active_context_ = nullptr;
      }
      std::this_thread::sleep_for(kReconnectDelay);
      continue;
    }

    ddrl::comm::PolicyPush push;
    while (!stop_.load() && stream->Read(&push)) {
      auto payload = build_payload(push);
      last_version_.store(payload.version, std::memory_order_relaxed);
      if (policy_config_ != nullptr) {
        apply_update(payload, *policy_config_, logger_);
      } else if (!warn_no_config_.exchange(true, std::memory_order_relaxed) && logger_) {
        DDRL_LOG_WARN(logger_, "Policy update received before policy config is available");
      }

      ddrl::comm::PolicyStreamRequest ack_req;
      auto*                           ack = ack_req.mutable_ack();
      ack->set_worker_id(worker_id_);
      ack->set_version(payload.version);
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
      )
                            .count();
      ack->set_received_unix_ms(static_cast<uint64_t>(now_ms));
      stream->Write(ack_req);
    }

    stream->WritesDone();
    auto status = stream->Finish();
    {
      std::lock_guard lock(context_mutex_);
      active_context_ = nullptr;
    }

    if (stop_.load()) {
      break;
    }

    if (!status.ok() && logger_) {
      DDRL_LOG_WARN(
        logger_,
        "Policy stream closed (status={}): {}",
        static_cast<int>(status.error_code()),
        status.error_message()
      );
    } else if (logger_) {
      DDRL_LOG_INFO(logger_, "Policy stream ended, reconnecting...");
    }

    std::this_thread::sleep_for(kReconnectDelay);
  }
}

void PolicySubscriber::request_reconnect()
{
  std::lock_guard lock(context_mutex_);
  if (active_context_ != nullptr) {
    active_context_->TryCancel();
  }
}

PolicyPayload PolicySubscriber::build_payload(const ddrl::comm::PolicyPush& push)
{
  PolicyPayload payload;
  payload.version = push.version();
  payload.onnx_model.assign(push.onnx_model().begin(), push.onnx_model().end());

  return payload;
}

std::shared_ptr<const PolicyModel> PolicySubscriber::model() const
{
  return std::atomic_load(&model_);
}

std::shared_ptr<const PolicyModel>
PolicySubscriber::model_or_warn(const std::shared_ptr<ddrl::logging::Logger>& logger) const
{
  auto current = model();
  if (current) {
    return current;
  }
  if (!warn_missing_.exchange(true, std::memory_order_relaxed) && logger) {
    DDRL_LOG_WARN(
      logger,
      "Policy type set to 'neural_network' but no learner weights received yet. Falling back to "
      "random commands."
    );
  }
  return nullptr;
}

uint64_t PolicySubscriber::version() const
{
  const auto current = model();
  if (current) {
    return current->version();
  }
  return version_.load(std::memory_order_relaxed);
}

void PolicySubscriber::apply_update(
  const PolicyPayload& payload, const config::RLPolicyConfig& policy_cfg,
  const std::shared_ptr<ddrl::logging::Logger>& logger
)
{
  auto built = PolicyModel::from_payload(payload, policy_cfg, logger);
  if (!built) {
    if (logger) {
      DDRL_LOG_WARN(logger, "Failed to build ONNX policy model from payload v{}", payload.version);
    }
    return;
  }

  std::atomic_store(&model_, built);
  version_.store(payload.version, std::memory_order_relaxed);
  warn_missing_.store(false, std::memory_order_relaxed);
  if (logger) {
    DDRL_LOG_INFO(
      logger,
      "Applied policy update v{} (onnx={} bytes)",
      payload.version,
      payload.onnx_model.size()
    );
  }
}

} // namespace ddrl::worker
