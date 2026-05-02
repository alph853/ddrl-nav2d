#pragma once
#include "core/control/pid.hpp"
#include "core/control/pure_pursuit.hpp"
#include "core/kinematics/ackermann.hpp"
#include "core/math/pose.hpp"
#include <variant>
#include <vector>

namespace ddrl::config {

struct Waypoint {
  core::math::Pose2 p_local; // in the object’s local frame at spawn
  double            wait_s;  // optional dwell time on arrival
};

struct WaypointList {
  std::vector<Waypoint> points;

  double reached_pos_tol; // m, distance to consider "reached"
  double reached_yaw_tol; // rad, only if heading is enforced
  bool   loop;            // 0->1->…->N->…->1->0
};

struct LinearConstVelCfg {
  core::kinematics::MotionCaps caps;

  // Pick a random speed from this range at every waypoint
  double speed_min;     // m/s
  double speed_max;     // m/s
  bool   brake_to_stop; // obey max_decel on stop
};

struct LinearPIDCfg {
  core::kinematics::MotionCaps caps;

  // Controllers
  core::control::PIDParams speed_pid;
  core::control::PIDParams lateral_pid; // error is signed distance to centerline

  // Pick a random speed from this range at every waypoint
  double cruise_speed_min;  // m/s
  double cruise_speed_max;  // m/s
  double max_lateral_error; // m, for clamping integral/derivative action
};

struct AckermannFollowCfg {
  core::kinematics::AckermannCaps acker_caps;

  // Controllers
  core::control::PurePursuitCaps pp;
  core::control::PIDParams       speed_pid;

  // Pick a random speed from this range at every waypoint
  double cruise_speed_min;        // m/s
  double cruise_speed_max;        // m/s
  double cornering_speed_min;     // clamp when curvature high
  double arrival_stop_decel_bias; // multiply max_decel near last meters
};

using FollowCfg = std::variant<LinearConstVelCfg, LinearPIDCfg, AckermannFollowCfg>;

struct DynamicProfile {
  std::string  name;
  FollowCfg    follow_cfg;
  WaypointList waypoints_local;     // Waypoints in object coordinates
};

} // namespace ddrl::config
