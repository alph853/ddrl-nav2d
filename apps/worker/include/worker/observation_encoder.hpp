#pragma once

#include <vector>

#include "config/rl.hpp"
#include "core/math/primitives.hpp"
#include "core/rl/rl.hpp"

namespace ddrl::worker {

struct EncodedObservation {
  std::vector<float> features;
  std::vector<core::math::Point3> representative_points_world;
};

/**
 * @brief Convert a simulator observation into the compact tensors consumed by the learner/policy.
 *
 * The active encoding is a robot-relative polar grid:
 * - azimuth bins partition the horizontal field around the base link
 * - z bands partition point height in the active world-frame sensor cloud
 * - each cell stores the 5th percentile planar range in [0, range_max_m], normalized to [0, 1]
 *   with empty cells filled to 1.0
 * - low-dimensional ego-motion and waypoint features are appended after the grid:
 *   [vx_ego, vy_ego, yaw_rate, goal_distance, heading_error]
 */
EncodedObservation
encode_observation(
  const core::rl::Observation& obs, const config::RLPolicyConfig::Architecture& arch
);

} // namespace ddrl::worker
