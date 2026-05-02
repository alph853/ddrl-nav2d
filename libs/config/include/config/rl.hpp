#pragma once

#include "core/rl/reward.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ddrl::config {

struct RLPolicyConfig {
  enum class Type : uint8_t { RANDOM, NEURAL_NETWORK, EXPERT };

  struct Architecture {
    uint32_t azimuth_bins{32};
    std::string policy_model{"mlp"};
    std::vector<std::pair<double, double>> z_bands{
      {0.0, 0.4},
      {0.4, 1.6},
      {1.6, 3.0},
    };
    double   range_max_m{35.0};
    uint32_t state_dim{5};
    uint32_t action_dim{2};
    uint32_t mlp_hidden_dim{256};
    uint32_t lstm_hidden_dim{256};

    [[nodiscard]] uint32_t observation_dim() const
    {
      return azimuth_bins * static_cast<uint32_t>(z_bands.size()) + state_dim;
    }
  };

  Type         type{Type::RANDOM};
  Architecture architecture{};
};

struct RLRewardConfig {
  std::string                     reward_type{"default"};
  bool                            terminate_on_goal{true};
  core::rl::RewardFunctionWeights weights;
};

struct RLConfig {
  RLPolicyConfig policy{};
  RLRewardConfig reward{};
};

} // namespace ddrl::config
