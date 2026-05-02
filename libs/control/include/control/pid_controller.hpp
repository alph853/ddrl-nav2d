#pragma once
#include "core/control/pid.hpp"
#include "core/math/primitives.hpp"
#include <algorithm>
#include <cmath>

namespace ddrl::control {

class PIDController
{
public:
  // ...existing code...
  PIDController(const core::control::PIDParams& cfg) noexcept : config_(cfg) { reset(); }

  void reset()
  {
    integral_   = 0.0;
    prev_error_ = 0.0;
    d_filtered_ = 0.0;
    has_prev_   = false;
  }

  void set_config(const core::control::PIDParams& cfg) { config_ = cfg; }
  [[nodiscard]] const core::control::PIDParams& config() const { return config_; }

  void                 set_setpoint(double sp) { setpoint_ = sp; }
  [[nodiscard]] double setpoint() const { return setpoint_; }

  // Update using a measured value (error = setpoint - measurement).
  double update(double measurement, double dt)
  {
    const double error = setpoint_ - measurement;
    return update_with_error(error, dt);
  }

  // Update when caller directly supplies error.
  double update_with_error(double error, double dt)
  {
    if (dt <= 0.0 || !std::isfinite(dt)) {
      dt = core::math::kEps;
    }

    // Proportional
    const double p_term = config_.kp * error;

    // Integral with clamp (anti-windup)
    integral_ += error * dt;
    const double i_max = std::max(0.0, config_.integral_windup_limit);
    if (i_max > 0.0) {
      integral_ = std::min(integral_, i_max);
      integral_ = std::max(integral_, -i_max);
    }
    const double i_term = config_.ki * integral_;

    // Derivative on error with optional low-pass filter
    double d_term = 0.0;
    if (has_prev_) {
      const double d_raw = (error - prev_error_) / dt;
      double       d     = d_raw;
      if (config_.d_filter_hz > 0.0) {
        const double w     = 2.0 * M_PI * config_.d_filter_hz;
        const double alpha = (w * dt) / (1.0 + w * dt); // 1st-order PT1
        d_filtered_        = (1.0 - alpha) * d_filtered_ + alpha * d_raw;
        d                  = d_filtered_;
      }
      d_term = config_.kd * d;
    } else {
      // Initialize derivative state
      d_filtered_ = 0.0;
      has_prev_   = true;
    }

    prev_error_ = error;
    return p_term + i_term + d_term;
  }

private:
  core::control::PIDParams config_;

  double setpoint_{0.0};
  double integral_{0.0};
  double prev_error_{0.0};
  double d_filtered_{0.0};
  bool   has_prev_{false};
};

} // namespace ddrl::control
