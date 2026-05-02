#pragma once

#include "core/kinematics/ackermann.hpp"
#include "core/math/pose.hpp"
#include "core/math/twist.hpp"
#include "core/vision/point_cloud.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ddrl::core::rl {

/// Action command for the robot
struct Action {
  double                             timestamp{0.0}; // t_cmd in sim seconds
  core::kinematics::AckermannCommand cmd{};          // desired command
  bool                               has_raw_policy_action{false};
  double normalized_speed_action{0.0}; // tanh policy/expert speed action in [-1, 1]
  double normalized_steer_action{0.0}; // tanh policy/expert steer action in [-1, 1]
  double raw_speed_action{0.0};        // pre-squash policy sample/logit for speed
  double raw_steer_action{0.0};        // pre-squash policy sample/logit for steer
};

/// Single sensor observation with zero-copy shared ownership
struct SensorObservation {
  std::string sensor_id; ///< Sensor identifier
  std::shared_ptr<const core::vision::PointCloud>
         point_cloud;    ///< Shared point cloud data (zero-copy)
  double timestamp{0.0}; ///< Sensor capture time
};

/// Complete observation from robot
struct Observation {
  core::math::Pose2  est_pose{}; ///< Robot pose in world frame
  core::math::Twist2 est_vel;    ///< Robot velocity in world frame
  std::vector<SensorObservation>
                           sensor_data; ///< All sensor observations (shared_ptr avoids copies)
  std::vector<std::string> collision_objects;  ///< Names of objects involved in collisions
  double                   reward{0.0};        ///< Accumulated reward from previous decision tick
  bool                     is_terminal{false}; ///< Episode ended (collision, success, timeout)
  core::math::Point2       waypoint_pos_world;   ///< Active route waypoint in world frame
  double                   distance_to_waypoint{0.0};
  double                   heading_to_waypoint{0.0};
  bool                     waypoint_reached{false};
  bool                     final_goal_reached{false};
  uint32_t                 waypoint_index{0};
  uint32_t                 waypoint_count{0};
  bool                     reverse_gear{false};
  double                   gear_hold_time_s{0.0};

  /// Check if robot is currently in collision (convenience helper)
  [[nodiscard]] bool is_collided() const { return !collision_objects.empty(); }
};

} // namespace ddrl::core::rl
