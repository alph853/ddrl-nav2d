#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "dynamic_profile.hpp"
#include "map.hpp"
#include "model_profile.hpp"
#include "robot_profile.hpp"

namespace ddrl::config {

struct DynamicInstanceCfg {
  std::string       name;               // unique in world
  std::string       model_id;           // visual model to use
  std::string       dynamic_profile_id; // lookup in profile_library
  core::math::Pose2 pose_world;         // initial pose
};

struct DeterministicWorldCfg {
  core::math::Pose2               robot_pose_world{};
  std::vector<DynamicInstanceCfg> dynamic_instances;
};

// ---------- Random world config ----------

struct RandomDynamicInstancesCfg {
  std::string name;               // generate ids like "*_0001, *_0002, ..."
  std::string model_id;           // visual/collision model to use
  std::string dynamic_profile_id; // lookup in profile_library
  size_t      count;              // how many to spawn
};

struct RandomWorldCfg {
  std::vector<RandomDynamicInstancesCfg> random_dynamic_instances;
};

// ---------- World root ----------

// Collision categories
struct CollisionFilter {
  std::string              category;      // e.g., "car"
  std::vector<std::string> collides_with; // e.g., ["car", "pedestrian", "wall"]
};

struct RobotInstanceCfg {
  std::string name;
  std::string robot_profile_id;

  double action_delay_s;
};

struct WorldConfig {
  std::string version;

  // Profile libraries
  std::unordered_map<std::string, ModelProfile>   model_profiles;
  std::unordered_map<std::string, DynamicProfile> dynamic_profiles;
  std::unordered_map<std::string, RobotProfile>   robot_profiles;

  // World type: deterministic or random
  std::variant<DeterministicWorldCfg, RandomWorldCfg> world;

  // Collision filtering rules
  std::vector<CollisionFilter> collision_categories;

  // Robot configuration (references zones from active_map)
  RobotInstanceCfg robot;
};

} // namespace ddrl::config
