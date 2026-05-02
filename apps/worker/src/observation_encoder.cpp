#include "worker/observation_encoder.hpp"

#include "core/math/numeric.hpp"
#include <cmath>
#include <numbers>
#include "core/rl/observation_encoding.hpp"

namespace ddrl::worker {

namespace {

template <typename T>
T finite_or(T value, T fallback)
{
  return std::isfinite(value) ? value : fallback;
}

} // namespace

EncodedObservation
encode_observation(const core::rl::Observation& obs, const config::RLPolicyConfig::Architecture& arch)
{
  EncodedObservation enc;
  const auto scan = core::rl::encode_scan_grid(
    obs,
    core::rl::ScanGridEncodingConfig{
      .azimuth_bins = arch.azimuth_bins,
      .z_bands = arch.z_bands,
      .range_max_m = arch.range_max_m,
    }
  );
  enc.features.reserve(scan.normalized_ranges.size() + 5);
  enc.features.insert(
    enc.features.end(), scan.normalized_ranges.begin(), scan.normalized_ranges.end()
  );
  enc.representative_points_world = scan.representative_points_world;

  const double cos_yaw = std::cos(-obs.est_pose.yaw);
  const double sin_yaw = std::sin(-obs.est_pose.yaw);

  const double vx_world = obs.est_vel.linear.x;
  const double vy_world = obs.est_vel.linear.y;
  const double vx_ego   = cos_yaw * vx_world - sin_yaw * vy_world;
  const double vy_ego   = sin_yaw * vx_world + cos_yaw * vy_world;

  const double gx             = obs.waypoint_pos_world.x - obs.est_pose.pos.x;
  const double gy             = obs.waypoint_pos_world.y - obs.est_pose.pos.y;
  const double distance       = std::hypot(gx, gy);
  const double heading_error = core::math::normalize_angle(obs.heading_to_waypoint - obs.est_pose.yaw);
  const double inv_range_max = 1.0 / std::max(arch.range_max_m, 1e-3);

  enc.features.push_back(static_cast<float>(finite_or(vx_ego, 0.0)));
  enc.features.push_back(static_cast<float>(finite_or(vy_ego, 0.0)));
  enc.features.push_back(static_cast<float>(finite_or(obs.est_vel.angular, 0.0)));
  enc.features.push_back(static_cast<float>(finite_or(distance * inv_range_max, 0.0)));
  enc.features.push_back(static_cast<float>(finite_or(heading_error / std::numbers::pi, 0.0)));

  return enc;
}

} // namespace ddrl::worker
