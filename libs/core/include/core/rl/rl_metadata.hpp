#pragma once

#include "core/rl/reward.hpp"

#include <cstddef>

namespace ddrl::core::rl {

/**
 * @brief Shared metadata describing reinforcement learning episode progress.
 */
struct RLMetadata {
  std::size_t episode_index{0};
  std::size_t episode_step_count{0};
  double      step_reward{0.0};
  double      episode_return{0.0};
  bool        goal_reached{false};
  RewardTermValues reward_terms{};
};

} // namespace ddrl::core::rl
