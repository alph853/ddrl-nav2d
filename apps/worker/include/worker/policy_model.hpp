#pragma once

#include "core/rl/rl.hpp"
#include <algorithm>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <vector>

#include "config/rl.hpp"
#include "logging/logging.hpp"
#include "worker/policy_subscriber.hpp"

namespace ddrl::worker {

/**
 * @brief ONNX Runtime-backed policy inference wrapper.
 */
class PolicyModel final
{
public:
  /**
   * @brief Per-simulator mutable state reused between evaluations.
   */
  struct PolicyContext {
    std::vector<float> hidden_state;
    std::vector<float> cell_state;
    uint64_t           policy_version{0};

    // Scratch buffers to avoid per-step allocations.
    std::vector<float> mean;
    std::vector<float> log_std;
  };

  /**
   * @brief Build a policy model from a learner payload containing an ONNX export.
   */
  static std::shared_ptr<PolicyModel> from_payload(
    const PolicyPayload& payload, const config::RLPolicyConfig& policy_config,
    const std::shared_ptr<logging::Logger>& logger
  );

  /**
   * @brief Run inference for the observation and update recurrent context.
   */
  [[nodiscard]] bool evaluate(
    const core::rl::Observation& obs, PolicyContext& context, std::vector<float>& mean_out,
    std::vector<float>& log_std_out
  ) const;
  [[nodiscard]] bool evaluate_encoded(
    std::span<const float> encoded_observation, PolicyContext& context, std::vector<float>& mean_out,
    std::vector<float>& log_std_out
  ) const;

  [[nodiscard]] uint64_t version() const { return version_; }

  static double map_to_range(double v, double min, double max)
  {
    const double alpha = 0.5 * (std::clamp(v, -1.0, 1.0) + 1.0);
    return min + alpha * (max - min);
  }
  static Ort::SessionOptions default_session_options();

private:
  PolicyModel(config::RLPolicyConfig::Architecture arch, std::shared_ptr<logging::Logger> logger);

  bool load_onnx(const PolicyPayload& payload);
  void allocate_context(PolicyContext& ctx) const;

  static Ort::Env& ort_env();

  uint64_t version_{0};

  config::RLPolicyConfig::Architecture arch_{};

  std::shared_ptr<logging::Logger> logger_;

  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo               memory_info_{nullptr};

  size_t observation_dim_{0};
  size_t action_dim_{0};
  bool   recurrent_{false};
  size_t recurrent_hidden_dim_{0};
};

} // namespace ddrl::worker
