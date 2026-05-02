#pragma once

namespace ddrl::core::kinematics {

struct MotionCaps {
  double max_speed; // m/s (magnitude, forward or reverse)
  double max_accel; // m/s^2 (throttle)
  double max_decel; // m/s^2 (brake, positive number)

  // Execution characteristics
  double cmd_rate_hz; // control/actuation update rate
  double cmd_delay_s; // actuation/comm latency

  // Noise / imperfections
  double accel_noise_std; // m/s^2
  double steer_noise_std; // rad
};

} // namespace ddrl::core::kinematics
