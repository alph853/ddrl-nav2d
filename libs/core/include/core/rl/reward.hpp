#pragma once

#include "core/kinematics/ackermann.hpp"
#include "core/math/pose.hpp"
#include "core/math/twist.hpp"
#include "core/sim/collision.hpp"
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "logging/logging.hpp"

namespace ddrl::core::rl {

struct RewardTermValue {
  double raw{0.0};
  double weighted{0.0};
};

using RewardTermValues = std::unordered_map<std::string, RewardTermValue>;

struct RewardStepResult {
  double           total{0.0};
  RewardTermValues terms;
};

/// Telemetry data from the simulation world for reward computation
struct RewardTelemetry {
  core::math::Pose2                  pose;
  core::math::Twist2                 velocity;
  core::kinematics::AckermannCommand command;
  bool                               has_raw_policy_action{false};
  double                             normalized_speed_action{0.0};
  double                             normalized_steer_action{0.0};
  double                             raw_speed_action{0.0};
  double                             raw_steer_action{0.0};

  core::math::Point2 waypoint_position;
  bool               reverse_gear{false};
  double             gear_hold_time_s{0.0};
  bool               requested_reverse_gear{false};

  double   distance_to_waypoint{0.0};
  double   heading_to_waypoint{0.0};
  bool     waypoint_reached{false};
  bool     final_goal_reached{false};
  uint32_t waypoint_index{0};
  uint32_t waypoint_count{0};
  bool     time_limit_reached{false};
  uint32_t episode_step_count{0};
  double   forward_clearance_m{std::numeric_limits<double>::infinity()};
};

/// Abstract interface for reward computation
class RewardFunction
{
public:
  virtual ~RewardFunction() = default;

  /// Compute reward for a single time step
  virtual RewardStepResult compute(
    const RewardTelemetry& telemetry, double dt, const core::sim::CollisionSeverity& collision,
    bool collision_active
  ) = 0;

  /// Reset internal state (e.g., for episodic calculations)
  virtual void reset() = 0;

  /// Get human-readable description of reward function
  [[nodiscard]] virtual std::string description() const = 0;

  /// Optional termination hint (default: never terminate)
  [[nodiscard]] virtual bool should_terminate() const { return false; }
};

struct RewardFunctionWeights {
  double progress{10.0};                  ///< Normalized goal-directed progress (per second)
  double progress_min{10.0};              ///< Minimum progress weight after decay
  double progress_decay_steps{0.0};       ///< Linear decay steps for progress weight (0 disables)
  double effort_penalty{-0.2};            ///< Penalty per meter traveled (progress-per-effort)
  double control_smoothness{-0.001};      ///< Legacy jerk penalty used by deprecated reward
  double time_penalty{-0.01};             ///< Small penalty per time step (encourage speed)
  double waypoint_reached_bonus{50.0};    ///< Bonus for reaching an intermediate waypoint
  double goal_reached_bonus{50.0};        ///< Bonus for reaching goal
  double goal_reached_bonus_min{50.0};    ///< Minimum goal bonus after decay
  double goal_reached_bonus_decay_steps{0.0}; ///< Linear decay steps for goal bonus (0 disables)
  double time_limit_penalty{-20.0};       ///< Penalty when max_sim_seconds reached
  double yaw_rate_penalty{-0.02};         ///< Penalty proportional to yaw rate magnitude
  double reverse_penalty{-1.0};           ///< Penalty on reverse-hold EMA (per second)
  double reverse_penalty_tau_s{0.75};     ///< EMA time constant for reverse penalty ramp-up
  double reverse_speed_penalty{-1.0};     ///< Penalty per second while reversing
  double reverse_hold_penalty{-0.5};      ///< Penalty per second while reversing continuously
  double gear_switch_penalty{-0.3};       ///< Penalty per gear switch
  double heading_alignment_bonus{0.1}; ///< Legacy alias used by deprecated reward
  double heading_to_goal_reward{0.1}; ///< Reward for facing the active goal direction (per second)
  double proximity_reward{-0.5}; ///< Reward/penalty driven by nearby forward obstacles (per second)
  double yaw_rate_near_goal_dist{4.0}; ///< Apply yaw penalty when closer than this distance
  double time_penalty_scale{-0.0001}; ///< Linear time-scaled penalty coefficient
  double time_penalty_scale_decay_steps{0.0}; ///< Linear decay steps for time penalty scale (0 disables)
  double steer_penalty{-0.1}; ///< Penalty on persistent same-direction steer EMA (per second)
  double steer_penalty_tau_s{0.75}; ///< EMA time constant for steer persistence ramp-up
  double steer_hold_penalty{-0.02}; ///< Penalty per second while steering continuously
  double steer_hold_threshold{0.2}; ///< Threshold to count as steering (radians)
  double steer_hold_terminate_s{0.0}; ///< Terminate if steering continuously this long (0 disables)
  double steer_hold_terminal_penalty{-50.0}; ///< Penalty when steer-hold termination triggers
  double steer_hold_decay_s{1.0}; ///< Decay time constant for steering hold timer (s)
  double collision_penalty_scale{-500.0}; ///< Collision penalty scale (per second at score=1.0)
  double progress_norm_speed_m_s{8.0}; ///< Normalization speed for progress projection (m/s)
  double progress_gate_epsilon_m_s{0.1}; ///< Minimum net progress rate (m/s) to earn progress reward
  double progress_steer_gate_min{1.0}; ///< Minimum fraction of progress reward kept at persistent steer=1
  double progress_steer_gate_power{1.0}; ///< Shape exponent for steer-persistence progress attenuation
  double steer_norm_abs{0.5}; ///< Normalization for steer magnitude penalty (rad)
  double smoothness_speed_jerk_norm{10.0}; ///< Legacy jerk norm used by deprecated reward
  double smoothness_steer_jerk_norm{2.0}; ///< Legacy jerk norm used by deprecated reward
  double clearance_gate_dist_min_m{1.0}; ///< Forward clearance below this => gate is 0
  double clearance_gate_dist_max_m{4.0}; ///< Forward clearance above this => gate is 1
  double clearance_gate_half_angle_rad{0.45}; ///< Forward cone half-angle used for scan-like clearance

  double stall_penalty_scale{-8.0};   ///< Penalty on stall EMA (negative)
  double stall_penalty_growth{0.5};   ///< Exponential growth rate (deprecated reward only)
  double stall_progress_epsilon{0.2}; ///< Progress rate to count as progress (m/s)
  double stall_window_s{2.5};         ///< Window size for net progress check (s)
  double stall_tau_s{1.5};            ///< EMA time constant for stall ramp-up
  double stall_limit{0.98};           ///< Hard terminate when stall EMA reaches this value
  double stall_terminal_penalty{-50.0}; ///< Penalty when stall termination triggers
};

/// Minimal, normalized default reward (recommended).
class DefaultRewardFunction : public RewardFunction
{
public:
  explicit DefaultRewardFunction(RewardFunctionWeights weights = RewardFunctionWeights())
      : weights_(weights)
  {
  }

  RewardStepResult compute(
    const RewardTelemetry& telemetry, double dt, const core::sim::CollisionSeverity& collision,
    bool collision_active
  ) override;
  void                      reset() override;
  [[nodiscard]] bool        should_terminate() const override { return stall_terminal_; }
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] double      stall_ema() const { return stall_ema_; }

private:
  RewardFunctionWeights                             weights_;

  bool has_prev_requested_reverse_{false};
  bool prev_requested_reverse_{false};

  double             reverse_penalty_ema_{0.0};
  double             steer_persistence_ema_{0.0};
  double             stall_ema_{0.0};
  bool               stall_terminal_{false};
  std::deque<double> goal_dist_hist_;
  std::deque<core::math::Point2> motion_pos_hist_;
  std::optional<uint32_t> last_waypoint_index_;
};

/// Legacy reward function kept for comparison and ablations.
class DeprecatedRewardFunction : public RewardFunction
{
public:
  explicit DeprecatedRewardFunction(RewardFunctionWeights weights = RewardFunctionWeights())
      : weights_(weights)
  {
  }

  RewardStepResult compute(
    const RewardTelemetry& telemetry, double dt, const core::sim::CollisionSeverity& collision,
    bool collision_active
  ) override;
  void                      reset() override;
  [[nodiscard]] bool        should_terminate() const override { return stall_terminal_; }
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] double      stall_ema() const { return stall_ema_; }

private:
  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("core.rl.reward");

  RewardFunctionWeights                             weights_;
  std::optional<core::kinematics::AckermannCommand> prev_cmd_;

  double             stall_time_s_{0.0};
  double             stall_ema_{0.0};
  bool               stall_terminal_{false};
  double             strike_timer_s_{0.0};
  bool               prev_reverse_gear_{false};
  bool               has_prev_gear_{false};
  double             elapsed_time_s_{0.0};
  std::deque<double> gear_switch_times_;
  double             steer_hold_time_s_{0.0};
  double             reverse_hold_time_s_{0.0};

  bool has_prev_goal_dist_ = false;
  double prev_goal_dist_ = 0.0;
  std::deque<double> goal_dist_hist_;
};

/// Factory function to create reward functions from config
std::unique_ptr<RewardFunction>
create_reward_function(const std::string& type, RewardFunctionWeights weights = {});

} // namespace ddrl::core::rl
