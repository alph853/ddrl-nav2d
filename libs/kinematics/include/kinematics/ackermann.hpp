#pragma once

#include "core/kinematics/ackermann.hpp"
#include "core/math/pose.hpp"
#include <chrono>

namespace ddrl::kinematics {

struct AckermannState {
  core::math::Pose2 pose;

  double speed;
  double steering;
};

class AckermannModel
{
public:
  explicit AckermannModel(const core::kinematics::AckermannCaps& cfg) noexcept;

  void                                                 reset(const AckermannState& state) noexcept;
  [[nodiscard]] const AckermannState&                  state() const noexcept;
  [[nodiscard]] const core::kinematics::AckermannCaps& params() const noexcept;
  [[nodiscard]] bool                                   reverse_gear() const noexcept;
  [[nodiscard]] double                                 gear_hold_time_remaining() const noexcept;

  void step(std::chrono::duration<double>             dt,
            const core::kinematics::AckermannCommand& cmd) noexcept;

private:
  void apply_speed_command(
    std::chrono::duration<double> dt, double desired_speed, bool reverse_gear_cmd
  ) noexcept;
  void apply_steering_command(std::chrono::duration<double> dt, double desired_steering) noexcept;
  void integrate_pose(std::chrono::duration<double> dt) noexcept;

  core::kinematics::AckermannCaps caps_;
  AckermannState                  state_;
  bool                            reverse_gear_{false};
  double                          gear_hold_remaining_s_{0.0};
};

} // namespace ddrl::kinematics
