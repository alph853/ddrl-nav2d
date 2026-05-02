#pragma once

#include "core/math/pose.hpp"
#include "core/math/twist.hpp"
#include "core/vision/point_cloud.hpp"
#include <memory>

namespace ddrl::core::sim {

/// Snapshot of a single dynamic object state
struct ObjectState {
  std::uint32_t      id;       ///< Unique entity identifier
  std::string        name;     ///< Human-readable name for debugging
  std::string        model_id; ///< Reference to ModelProfile for geometry lookup
  core::math::Pose2  pose;
  core::math::Twist2 velocity;
  std::string        state_name; ///< FSM state for debugging (IDLE, RUN, STOP, etc.)
};

/// Snapshot of robot state
struct RobotState {
  std::string        model_id; ///< Reference to ModelProfile for geometry lookup
  core::math::Pose2  pose;
  core::math::Twist2 velocity;

  double steering_angle{0.0};
  double speed{0.0};
  bool   is_collided{false};
};

/// Complete world state snapshot for visualization
struct WorldStateSnapshot {
  RobotState               robot;
  std::vector<ObjectState> dynamic_objects;
  double                   sim_time{0.0};
  std::string              current_map; ///< Name of the current map
  std::string              active_route_name;
  uint32_t                 active_waypoint_index{0};
  uint32_t                 active_waypoint_count{0};

  std::vector<std::shared_ptr<const core::vision::PointCloud>> sensor_clouds;
};

} // namespace ddrl::core::sim

std::string to_string(const ddrl::core::sim::RobotState& state);

std::string to_string(const ddrl::core::sim::ObjectState& state);

std::string to_string(std::span<const ddrl::core::sim::ObjectState> states);

std::string to_string(const ddrl::core::sim::WorldStateSnapshot& snapshot);

std::string to_string(std::span<const ddrl::core::sim::WorldStateSnapshot> snapshot_ptr);
