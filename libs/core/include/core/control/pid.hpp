#pragma once

namespace ddrl::core::control {

struct PIDParams {
  double kp, ki, kd;
  double integral_windup_limit;
  double d_filter_hz; // 0 => no filtering
};

} // namespace ddrl::core::control
