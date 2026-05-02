#pragma once

#include "core/base/result.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include "coordinator.grpc.pb.h"
#include "logging/logging.hpp"
#include "worker/config.hpp"

namespace ddrl::worker {

struct HeartbeatSnapshot {
  std::string                   phase;
  uint32_t                      active_simulators{0};
  std::string                   detail;
  std::map<std::string, double> metrics;
};

class CoordinatorClient
{
public:
  struct SessionInfo {
    std::string session_token;
    std::string learner_target;
    uint64_t    heartbeat_interval_ms{2000};
    std::string assigned_identity; // from coordinator
  };

  struct Params {
    std::shared_ptr<WorkerConfig::CoordinatorCfg> config;
    std::string                                   worker_version;
  };

  using StatusProvider   = std::function<HeartbeatSnapshot()>;
  using DirectiveHandler = std::function<void(const ddrl::comm::ControlDirective&)>;

  CoordinatorClient(Params params);
  ~CoordinatorClient();

  CoordinatorClient(const CoordinatorClient&)            = delete;
  CoordinatorClient& operator=(const CoordinatorClient&) = delete;

  core::Result<void>
       start_heartbeat(const StatusProvider& status_provider, const DirectiveHandler& handler);
  void stop();

  [[nodiscard]] bool        running() const { return running_.load(std::memory_order_relaxed); }
  [[nodiscard]] std::string effective_identity() const;
  [[nodiscard]] uint32_t    effective_hash() const;
  [[nodiscard]] std::string session_token() const { return session_.session_token; }
  [[nodiscard]] std::string learner_target() const { return session_.learner_target; }
  [[nodiscard]] std::string assigned_identity() const { return session_.assigned_identity; }
  [[nodiscard]] uint64_t    heartbeat_interval_ms() const { return session_.heartbeat_interval_ms; }
  [[nodiscard]] bool        has_session() const { return !session_.session_token.empty(); }

private:
  void register_worker();

  void heartbeat_loop(const StatusProvider& provider, const DirectiveHandler& handler);

  SessionInfo session_;

  std::shared_ptr<WorkerConfig::CoordinatorCfg>         config_;
  std::string                                           worker_version_;
  std::shared_ptr<grpc::Channel>                        channel_;
  std::unique_ptr<ddrl::comm::CoordinatorControl::Stub> stub_;

  std::thread       heartbeat_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};

  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("CoordinatorClient");
};

} // namespace ddrl::worker
