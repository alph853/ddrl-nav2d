/**
 * @file ackermann.cpp
 * @brief Ackermann steering kinematics implementation
 */

#include "kinematics/ackermann.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "logging/logging.hpp"

namespace ddrl::kinematics {

AckermannModel::AckermannModel(const core::kinematics::AckermannCaps& cfg) noexcept
    : caps_(cfg), state_{}
{
}

void AckermannModel::reset(const AckermannState& state) noexcept
{
  state_ = state;
  reverse_gear_          = state_.speed < 0.0;
  gear_hold_remaining_s_ = 0.0;
}

const AckermannState& AckermannModel::state() const noexcept
{
  return state_;
}

const core::kinematics::AckermannCaps& AckermannModel::params() const noexcept
{
  return caps_;
}

bool AckermannModel::reverse_gear() const noexcept
{
  return reverse_gear_;
}

double AckermannModel::gear_hold_time_remaining() const noexcept
{
  return std::max(0.0, gear_hold_remaining_s_);
}

void AckermannModel::step(std::chrono::duration<double>             dt,
                           const core::kinematics::AckermannCommand& cmd) noexcept
{
  // Apply speed command with acceleration limits
  apply_speed_command(dt, cmd.speed, cmd.reverse_gear);

  // Apply steering command with rate limits
  apply_steering_command(dt, cmd.steer);

  // Integrate pose using current speed and steering
  integrate_pose(dt);
}

void AckermannModel::apply_speed_command(
  std::chrono::duration<double> dt, double desired_speed, bool reverse_gear_cmd
) noexcept
{
  const double dt_s = dt.count();

  const auto& caps_fwd = caps_.caps_fwd;
  const auto& caps_rev = caps_.caps_rev;

  desired_speed = std::max(0.0, desired_speed);

  if (gear_hold_remaining_s_ > 0.0) {
    gear_hold_remaining_s_ = std::max(0.0, gear_hold_remaining_s_ - dt_s);
  }

  const bool desired_reverse = reverse_gear_cmd;
  if (desired_speed > 0.0 && desired_reverse != reverse_gear_) {
    if (gear_hold_remaining_s_ <= 0.0) {
      reverse_gear_          = desired_reverse;
      gear_hold_remaining_s_ = std::max(0.0, caps_.gear_hold_time_s);
    }
  }

  // Clamp desired speed to forward/reverse envelopes
  const double signed_desired = reverse_gear_ ? -desired_speed : desired_speed;
  const double target_speed = (signed_desired >= 0.0)
                                ? std::clamp(signed_desired, 0.0, caps_fwd.max_speed)
                                : std::clamp(signed_desired, -caps_rev.max_speed, 0.0);

  // Compute speed error
  const double speed_error = target_speed - state_.speed;

  // Determine acceleration/deceleration based on current direction
  const double accel_limit_pos = (state_.speed >= 0.0) ? caps_fwd.max_accel : caps_rev.max_decel;
  const double accel_limit_neg = (state_.speed >= 0.0) ? -caps_fwd.max_decel : -caps_rev.max_accel;
  const double accel           = std::clamp(speed_error / dt_s, accel_limit_neg, accel_limit_pos);

  // Update speed with acceleration
  state_.speed += accel * dt_s;

  // Clamp to ensure no overshoot
  if (state_.speed >= 0.0) {
    state_.speed = std::clamp(state_.speed, 0.0, caps_fwd.max_speed);
  } else {
    state_.speed = std::clamp(state_.speed, -caps_rev.max_speed, 0.0);
  }
}

void AckermannModel::apply_steering_command(std::chrono::duration<double> dt,
                                             double                        desired_steering) noexcept
{
  const double dt_s = dt.count();

  // Clamp desired steering to physical limits
  const double target_steering =
      std::clamp(desired_steering, -caps_.max_steer_angle, caps_.max_steer_angle);

  // Compute steering error
  const double steer_error = target_steering - state_.steering;

  // Apply steering rate limit
  const double max_delta_steer = caps_.max_steer_rate * dt_s;
  const double delta_steer     = std::clamp(steer_error, -max_delta_steer, max_delta_steer);

  // Update steering
  state_.steering += delta_steer;

  // Clamp to ensure within limits
  state_.steering = std::clamp(state_.steering, -caps_.max_steer_angle, caps_.max_steer_angle);
}

void AckermannModel::integrate_pose(std::chrono::duration<double> dt) noexcept
{
  const double dt_s = dt.count();

  const double v     = state_.speed;
  const double theta = state_.pose.yaw;
  const double delta = state_.steering;
  const double l     = caps_.wheel_base;

  // Handle small steering angles (avoid tan singularity)
  constexpr double kSmallAngle = 1e-6;
  double           omega       = 0.0; // angular velocity

  if (std::abs(delta) < kSmallAngle) {
    // Straight line motion
    omega = 0.0;
  } else {
    // Curved motion
    omega = (v / l) * std::tan(delta);
  }

  // Integrate position using midpoint method for better accuracy
  const double theta_mid = theta + 0.5 * omega * dt_s;
  const double dx        = v * std::cos(theta_mid) * dt_s;
  const double dy        = v * std::sin(theta_mid) * dt_s;
  const double dtheta    = omega * dt_s;

  // Update pose
  state_.pose.pos.x += dx;
  state_.pose.pos.y += dy;
  state_.pose.yaw += dtheta;

  // Normalize yaw to [-pi, pi]
  constexpr double kPi = std::numbers::pi;
  while (state_.pose.yaw > kPi) {
    state_.pose.yaw -= 2.0 * kPi;
  }
  while (state_.pose.yaw < -kPi) {
    state_.pose.yaw += 2.0 * kPi;
  }
}

} // namespace ddrl::kinematics
