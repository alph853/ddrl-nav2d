#include "worker/coordinator_client.hpp"

#include "core/base/result.hpp"
#include <chrono>
#include <format>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ddrl::worker {
constexpr std::chrono::milliseconds kRetryBackoff{1000};

CoordinatorClient::CoordinatorClient(Params params)
    : config_(std::move(params.config)), worker_version_(std::move(params.worker_version))
{
  if (!config_) {
    throw std::runtime_error("CoordinatorClient requires non-null config");
  }

  if (config_->hostname.empty()) {
    config_->hostname = config_->worker_identity;
  }

  channel_ = grpc::CreateChannel(config_->target, grpc::InsecureChannelCredentials());
  stub_    = ddrl::comm::CoordinatorControl::NewStub(channel_);
  register_worker(); // throws on failure
}

CoordinatorClient::~CoordinatorClient()
{
  stop();
}

void CoordinatorClient::register_worker()
{
  if (!stub_) {
    throw std::runtime_error("Coordinator stub not initialized");
  }
  ddrl::comm::WorkerHello hello;
  hello.set_hostname(config_->hostname);
  hello.set_version(worker_version_);

  // Capabilities
  for (const auto& [key, value] : config_->capabilities) {
    if (key.empty()) {
      continue;
    }
    auto* cap = hello.add_capabilities();
    cap->set_name(key);
    cap->set_value(value);
  }

  // Labels (include identity hints)
  for (const auto& [key, value] : config_->labels) {
    (*hello.mutable_labels())[key] = value;
  }
  if (!config_->worker_identity.empty()) {
    auto* labels = hello.mutable_labels();
    if (!labels->contains("identity")) {
      (*labels)["identity"] = config_->worker_identity;
    }
    if (!labels->contains("worker_identity")) {
      (*labels)["worker_identity"] = config_->worker_identity;
    }
  }

  grpc::ClientContext       ctx;
  ddrl::comm::WorkerSession session_msg;
  const auto                status = stub_->RegisterWorker(&ctx, hello, &session_msg);
  if (!status.ok()) {
    if (logger_) {
      DDRL_LOG_ERROR(
        logger_,
        "Coordinator register failed (addr={}, status={}): {}",
        config_->target,
        static_cast<int>(status.error_code()),
        status.error_message()
      );
    }
    throw std::runtime_error(
      std::format(
        "Coordinator register failed: {} ({})",
        status.error_message(),
        static_cast<int>(status.error_code())
      )
    );
  }

  session_.assigned_identity  = session_msg.worker_id();
  session_.session_token      = session_msg.session_token();
  session_.learner_target     = session_msg.learner_target();
  session_.heartbeat_interval_ms =
    session_msg.heartbeat_interval_ms() > 0 ? session_msg.heartbeat_interval_ms() : 2000;

  if (logger_) {
    DDRL_LOG_INFO(
      logger_,
      "Coordinator assigned worker_id={} heartbeat={}ms learner={}",
      session_.assigned_identity,
      session_.heartbeat_interval_ms,
      session_.learner_target
    );
  }
}

core::Result<void> CoordinatorClient::start_heartbeat(
  const StatusProvider& status_provider, const DirectiveHandler& handler
)
{
  if (!stub_) {
    return std::unexpected(
      core::make_error(
        core::ErrorCode::CONFIG,
        "Coordinator stub not initialized",
        core::make_context("CoordinatorClient::start_heartbeat")
      )
    );
  }
  if (running_.exchange(true)) {
    return {};
  }
  stop_.store(false);

  auto status_provider_clone = status_provider;
  auto handler_clone         = handler;
  heartbeat_thread_          = std::thread([this,
                                   status_provider_copy = std::move(status_provider_clone),
                                   handler_copy         = std::move(handler_clone)]() mutable {
    this->heartbeat_loop(status_provider_copy, handler_copy);
  });

  return {};
}

void CoordinatorClient::stop()
{
  stop_.store(true);
  if (!running_.load()) {
    return;
  }
  if (heartbeat_thread_.joinable()) {
    heartbeat_thread_.join();
  }
  running_.store(false);
}

std::string CoordinatorClient::effective_identity() const
{
  if (!session_.assigned_identity.empty()) {
    return session_.assigned_identity;
  }
  if (!config_->worker_identity.empty()) {
    return config_->worker_identity;
  }
  return config_->hostname;
}

uint32_t CoordinatorClient::effective_hash() const
{
  return static_cast<uint32_t>(std::hash<std::string>{}(effective_identity()));
}

void CoordinatorClient::heartbeat_loop(
  const StatusProvider& status_provider, const DirectiveHandler& handler
)
{
  while (!stop_.load()) {
    SessionInfo session = session_;
    const auto  heartbeat_interval =
      std::chrono::milliseconds(std::max<uint64_t>(session.heartbeat_interval_ms, 500));

    grpc::ClientContext ctx;
    auto                stream = stub_->WorkerHeartbeatStream(&ctx);
    if (!stream) {
      if (logger_) {
        DDRL_LOG_ERROR(
          logger_, "Failed to open heartbeat stream to coordinator at {}", config_->target
        );
      }
      std::this_thread::sleep_for(kRetryBackoff);
      continue;
    }

    std::atomic<bool> writer_running{true};
    std::thread       writer(
      [this, &stream, &status_provider, heartbeat_interval, session, &writer_running]() mutable {
        while (!stop_.load()) {
          const auto snapshot = status_provider ? status_provider() : HeartbeatSnapshot{};
          ddrl::comm::WorkerHeartbeat heartbeat;
          heartbeat.set_worker_id(session.assigned_identity);
          heartbeat.set_session_token(session.session_token);
          heartbeat.set_unix_ms(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()
            )
                                    .count())
          );
          auto* status = heartbeat.mutable_status();
          status->set_phase(snapshot.phase);
          status->set_active_simulators(snapshot.active_simulators);
          status->set_detail(snapshot.detail);
          for (const auto& [key, value] : snapshot.metrics) {
            (*heartbeat.mutable_metrics())[key] = value;
          }

          if (!stream->Write(heartbeat)) {
            break;
          }

          std::this_thread::sleep_for(heartbeat_interval);
        }

        stream->WritesDone();
        writer_running.store(false);
      }
    );

    ddrl::comm::ControlDirective directive;
    while (!stop_.load() && stream->Read(&directive)) {
      if (handler) {
        handler(directive);
      }
    }

    if (writer_running.load()) {
      stream->WritesDone();
    }
    if (writer.joinable()) {
      writer.join();
    }

    const auto status = stream->Finish();
    if (!status.ok() && !stop_.load() && logger_) {
      DDRL_LOG_WARN(
        logger_,
        "WorkerHeartbeat stream closed (status={}): {}",
        static_cast<int>(status.error_code()),
        status.error_message()
      );
    }

    if (!stop_.load() && (status.error_code() == grpc::StatusCode::UNAUTHENTICATED ||
                          status.error_code() == grpc::StatusCode::NOT_FOUND)) {
      if (logger_) {
        DDRL_LOG_INFO(
          logger_, "Re-registering worker after session failure (status={})", status.error_message()
        );
      }
      try {
        register_worker();
        continue;
      } catch (const std::exception& e) {
        if (logger_) {
          DDRL_LOG_WARN(logger_, "Re-registration failed; retrying after backoff: {}", e.what());
        }
      }
    }

    if (!stop_.load()) {
      std::this_thread::sleep_for(kRetryBackoff);
    }
  }

  running_.store(false, std::memory_order_relaxed);
}

} // namespace ddrl::worker
