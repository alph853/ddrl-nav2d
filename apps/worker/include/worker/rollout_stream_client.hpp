#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <grpcpp/channel.h>

#include "config/rl.hpp"
#include "logging/logging.hpp"
#include "rollouts.grpc.pb.h"
#include "worker/rollout_publisher.hpp"

namespace ddrl::worker {

class RolloutStreamClient
{
public:
  struct Params {
    std::string                      target;
    std::string                      worker_id;
    std::string                      session_token;
    config::RLPolicyConfig::Architecture architecture;
    std::shared_ptr<logging::Logger> logger;
  };

  explicit RolloutStreamClient(Params params);
  RolloutStreamClient(const RolloutStreamClient&)            = delete;
  RolloutStreamClient& operator=(const RolloutStreamClient&) = delete;

  void set_target(std::string target);
  void set_architecture(config::RLPolicyConfig::Architecture architecture);
  bool send(const std::vector<Rollout>& batch);

private:
  bool ensure_stream_locked();
  void reset_stream_locked();

  std::mutex                                     mutex_;
  std::string                                    target_;
  std::string                                    worker_id_;
  std::string                                    session_token_;
  std::shared_ptr<logging::Logger>               logger_;
  std::shared_ptr<grpc::Channel>                 channel_;
  std::unique_ptr<ddrl::comm::RolloutStream::Stub> stub_;
  std::unique_ptr<grpc::ClientContext>           context_;
  std::unique_ptr<grpc::ClientReaderWriter<
    ddrl::comm::RolloutStreamRequest,
    ddrl::comm::RolloutStreamResponse>> stream_;
  uint64_t                                       batch_sequence_{0};
  config::RLPolicyConfig::Architecture           architecture_{};
  std::chrono::steady_clock::time_point          stats_window_start_;
  uint64_t                                       stats_window_batches_{0};
  uint64_t                                       stats_window_rollouts_{0};
  uint64_t                                       stats_window_steps_{0};
  uint64_t                                       stats_window_bytes_{0};
};

} // namespace ddrl::worker
