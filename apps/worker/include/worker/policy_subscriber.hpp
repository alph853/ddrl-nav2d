#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace grpc {
class ClientContext;
}

namespace ddrl::logging {
class Logger;
}

namespace ddrl::comm {
class PolicyPush;
}

namespace ddrl::config {
class RLPolicyConfig;
}

namespace ddrl::worker {

class PolicyModel;

struct PolicyPayload {
  uint64_t             version{0};
  std::vector<uint8_t> onnx_model; // preferred ONNX export
};

class PolicySubscriber
{
public:
  struct Params {
    std::string                      worker_id;
    std::string                      session_token;
  };

  /**
   * @brief Bi-directional client that subscribes to learner policy pushes.
   *
   * Opens a gRPC stream to the learner, receives policy tensors, and invokes
   * the supplied handler on each update. Automatically reconnects when the
   * learner endpoint changes or the stream drops.
   */
  explicit PolicySubscriber(Params params);
  ~PolicySubscriber();

  PolicySubscriber(const PolicySubscriber&)            = delete;
  PolicySubscriber& operator=(const PolicySubscriber&) = delete;

  /// Begin streaming policy updates in a background thread.
  void start();
  /// Stop streaming and join the background thread.
  void stop();
  /// Update the learner target endpoint; triggers a reconnect.
  void set_target(std::string target);
  /// Last successfully applied policy version.
  [[nodiscard]] uint64_t last_version() const;
  void set_policy_config(const config::RLPolicyConfig* policy_config);

  std::shared_ptr<const PolicyModel> model() const;
  std::shared_ptr<const PolicyModel>
  model_or_warn(const std::shared_ptr<ddrl::logging::Logger>& logger) const;
  uint64_t version() const;
  void apply_update(
    const PolicyPayload& payload, const config::RLPolicyConfig& policy_cfg,
    const std::shared_ptr<ddrl::logging::Logger>& logger
  );

private:
  void run_loop();
  void request_reconnect();

  [[nodiscard]] static PolicyPayload build_payload(const ddrl::comm::PolicyPush& push);

  std::string worker_id_;
  std::string session_token_;

  std::thread       thread_;
  std::atomic<bool> stop_{false};

  std::atomic<uint64_t> last_version_{0};
  std::atomic<std::shared_ptr<const PolicyModel>> model_{nullptr};
  std::atomic<uint64_t>     version_{0};
  mutable std::atomic<bool> warn_missing_{false};
  std::atomic<bool>         warn_no_config_{false};

  const config::RLPolicyConfig* policy_config_{nullptr};

  std::string             target_;
  std::mutex              target_mutex_;
  std::condition_variable target_cv_;

  std::mutex           context_mutex_;
  grpc::ClientContext* active_context_{nullptr};

  std::shared_ptr<ddrl::logging::Logger> logger_;
};

} // namespace ddrl::worker
