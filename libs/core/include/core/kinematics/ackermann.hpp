#pragma once
#include "motion.hpp"

namespace ddrl::core::kinematics {

struct AckermannCaps {
  MotionCaps caps_fwd; // forward motion limits
  MotionCaps caps_rev; // reverse motion limits

  double wheel_base;      // m
  double max_steer_angle; // rad
  double max_steer_rate;  // rad/s
  double min_turn_radius; // m
  double gear_hold_time_s{0.0}; // minimum time to hold a gear before switching
};

struct AckermannCommand {
  double steer{0.0}; // rad
  double speed{0.0}; // m/s (magnitude)
  bool   reverse_gear{false};
};

} // namespace ddrl::core::kinematics
