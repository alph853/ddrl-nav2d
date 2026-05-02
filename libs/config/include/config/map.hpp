/**
 * @file map.hpp
 * @brief Map configuration schema containing static geometry and robot routes
 */

#pragma once

#include "core/geom/aabb.hpp"
#include <string>
#include <vector>

namespace ddrl::config {

struct StaticInstanceCfg {
  std::string       name;     // unique in world (auto-deduplicated)
  std::string       model_id; // key into model_library
  core::math::Pose2 pose_world;
};

struct RobotWaypoint {
  std::string        name;
  core::math::Point2 pos;
  double             tolerance{1.0};
};

struct RobotRoute {
  std::string                name;
  core::math::Pose2          start;
  std::vector<RobotWaypoint> waypoints;
};

/**
 * @brief Map configuration containing static geometry and robot routes
 */
struct MapConfig {
  std::string version;
  std::string name;
  std::string description;

  std::vector<StaticInstanceCfg> static_instances;

  std::vector<RobotRoute>  robot_routes;

  core::geom::AABB2 bounds;
};

} // namespace ddrl::config
