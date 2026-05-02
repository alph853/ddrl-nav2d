#include "core/rl/reward.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <cmath>
#include <string_view>
// #include <print>

namespace ddrl::core::rl {

namespace {

void add_term(RewardStepResult& result, std::string_view name, double raw, double weighted)
{
  auto& term = result.terms[std::string(name)];
  term.raw += raw;
  term.weighted += weighted;
  result.total += weighted;
}

} // namespace

RewardStepResult DeprecatedRewardFunction::compute(
  const RewardTelemetry& telemetry, double dt, const core::sim::CollisionSeverity& collision,
  bool collision_active
)
{
  const double safe_dt = std::max(dt, 1e-6);
  const auto ema_update = [safe_dt](double prev, double sample, double tau_s) {
    const double tau   = std::max(tau_s, safe_dt);
    const double alpha = 1.0 - std::exp(-safe_dt / tau);
    return prev + alpha * (sample - prev);
  };
  RewardStepResult result;

  // --------- components (for logging) ----------
  double time_penalty_reward        = 0.0;
  double time_scaled_penalty_reward = 0.0;
  double time_limit_reward          = 0.0;

  double progress_reward          = 0.0;
  double effort_penalty_reward    = 0.0;
  double heading_alignment_reward = 0.0;
  double yaw_rate_reward          = 0.0;

  double stall_reward              = 0.0;
  double stall_terminal_reward     = 0.0;
  double reverse_speed_reward       = 0.0;
  double reverse_hold_reward        = 0.0;
  double gear_switch_reward         = 0.0;

  double collision_severity_reward  = 0.0;
  double control_smoothness_reward  = 0.0;
  double goal_reached_reward        = 0.0;
  double steer_penalty_reward       = 0.0;
  double steer_hold_reward          = 0.0;
  double steer_hold_terminal_reward = 0.0;

  // ------------------------------------------------------------
  // 0) Basic geometry
  // ------------------------------------------------------------
  const core::math::Point2 position      = telemetry.pose.pos;
  const core::math::Point2 waypoint_position = telemetry.waypoint_position;
  const core::math::Vec2   to_goal           = waypoint_position - position;
  const double             goal_dist     = to_goal.length();

  // ------------------------------------------------------------
  // 1) Time penalties
  // ------------------------------------------------------------
  time_penalty_reward = weights_.time_penalty * dt;
  add_term(result, "time_penalty", dt, time_penalty_reward);

  elapsed_time_s_ += dt;

  double time_scale = weights_.time_penalty_scale;
  if (weights_.time_penalty_scale_decay_steps > 0.0) {
    const double decay =
      std::max(0.0, 1.0 - (telemetry.episode_step_count / weights_.time_penalty_scale_decay_steps));
    time_scale *= decay;
  }
  time_scaled_penalty_reward = time_scale * elapsed_time_s_ * dt;
  add_term(
    result,
    "time_penalty_scale",
    elapsed_time_s_ * dt,
    time_scaled_penalty_reward
  );

  if (telemetry.time_limit_reached) {
    time_limit_reward = weights_.time_limit_penalty;
    add_term(result, "time_limit_penalty", 1.0, time_limit_reward);
  }

  // ------------------------------------------------------------
  // 2) Progress: use distance delta
  // ------------------------------------------------------------
  // Keep prev distance (initialize on first call)
  if (!has_prev_goal_dist_) {
    prev_goal_dist_     = goal_dist;
    has_prev_goal_dist_ = true;
  }

  const double delta_d = prev_goal_dist_ - goal_dist; // + if closer
  prev_goal_dist_      = goal_dist;

  double progress_weight = weights_.progress;
  if (weights_.progress_decay_steps > 0.0) {
    const double decay =
      std::max(0.0, 1.0 - (telemetry.episode_step_count / weights_.progress_decay_steps));
    progress_weight = weights_.progress_min + (weights_.progress - weights_.progress_min) * decay;
  }
  // reward only forward progress to goal
  // const double prog_clip_m = 0.25; // meters per step clip (tune)
  // const double progress_m  = std::clamp(delta_d, 0.0, prog_clip_m);
  // progress_reward          = progress_weight * progress_m; // already per-step (NOT *dt)
  // reward += progress_reward;
  const double progress_m  = std::max(0.0, delta_d);
  progress_reward          = progress_weight * progress_m; // already per-step (NOT *dt)
  // Apply a small progress scale when holding steer or reverse far from the goal.
  // const bool steer_hold_now = std::abs(telemetry.command.steer) > weights_.steer_hold_threshold;
  // const bool reverse_hold_now = telemetry.command.reverse_gear;
  // const bool far_from_goal = goal_dist >= 20.0;
  // const auto scale_steer = (steer_hold_time_s_ > 6.0 && far_from_goal) ? (1.0 - steer_hold_time_s_ / 30.0) : 1.0;
  // const auto scale_reverse = (reverse_hold_time_s_ > 6.0 && far_from_goal) ? (1.0 - reverse_hold_time_s_ / 30.0) : 1.0;
  // const auto scale = scale_steer * scale_reverse;
  // progress_reward *= scale;
  add_term(
    result,
    "progress",
    progress_weight != 0.0 ? progress_reward / progress_weight : progress_m,
    progress_reward
  );

  // Progress-per-effort: penalize path length to discourage orbiting
  const double speed_m_s = telemetry.velocity.linear.length();
  effort_penalty_reward  = weights_.effort_penalty * speed_m_s * dt;
  add_term(result, "effort_penalty", speed_m_s * dt, effort_penalty_reward);

  // ------------------------------------------------------------
  // 3) Windowed stall: based on net distance progress
  // ------------------------------------------------------------
  // Maintain a window of goal distances
  const double max_window_s = std::max(weights_.stall_window_s, safe_dt);
  const int    max_window   = std::max(2, static_cast<int>(std::ceil(max_window_s / safe_dt)));
  goal_dist_hist_.push_back(goal_dist);
  if (static_cast<int>(goal_dist_hist_.size()) > max_window) {
    goal_dist_hist_.pop_front();
  }

  // Compute net progress over window (positive if moved closer overall)
  double net_progress_w = 0.0;
  if (goal_dist_hist_.size() >= 2) {
    net_progress_w = goal_dist_hist_.front() - goal_dist_hist_.back();
  }

  // Define "stalled" if net progress is too small over window
  // Use meters, not m/s. This avoids orbit spikes resetting it.
  const double eps_net_progress_m = weights_.stall_progress_epsilon;
  const bool   is_stalled = (goal_dist_hist_.size() >= 2) && (net_progress_w < eps_net_progress_m);

  if (!is_stalled) {
    stall_terminal_    = false;
    strike_timer_s_      = 0.0;
  }
  stall_time_s_ = is_stalled ? (stall_time_s_ + dt) : 0.0;

  // ------------------------------------------------------------
  // 4) Stall penalty + termination
  // ------------------------------------------------------------
  const double stall_sample = is_stalled ? 1.0 : 0.0;
  stall_ema_                = ema_update(stall_ema_, stall_sample, weights_.stall_tau_s);
  const double stall_growth = std::exp(weights_.stall_penalty_growth * stall_ema_);
  stall_reward              = weights_.stall_penalty_scale * stall_growth * stall_ema_ * dt;
  add_term(
    result,
    "stall_penalty",
    (stall_growth != 0.0 && weights_.stall_penalty_scale != 0.0)
      ? (stall_reward / weights_.stall_penalty_scale)
      : (stall_ema_ * dt),
    stall_reward
  );
  if (!stall_terminal_ && weights_.stall_limit > 0.0 && stall_ema_ >= weights_.stall_limit) {
    stall_terminal_reward = weights_.stall_terminal_penalty;
    stall_terminal_       = true;
    add_term(result, "stall_terminal_penalty", 1.0, stall_terminal_reward);
  }

  // ------------------------------------------------------------
  // 5) Reverse penalty: allow reverse ESCAPE when stalled
  // ------------------------------------------------------------
  // Idea: make reverse less costly when stalled, but don't make it "free"
  if (telemetry.reverse_gear) {
    // When stalled, reduce penalty to encourage backing out;
    // when not stalled, keep penalty to avoid reverse-only policy.
    const double scale   = is_stalled ? 0.25 : 1.0;
    reverse_speed_reward = weights_.reverse_speed_penalty * dt * scale;
    add_term(
      result,
      "reverse_speed_penalty",
      weights_.reverse_speed_penalty != 0.0 ? (reverse_speed_reward / weights_.reverse_speed_penalty) : (dt * scale),
      reverse_speed_reward
    );
  }
  if (telemetry.reverse_gear) {
    reverse_hold_time_s_ += dt;
  } else {
    reverse_hold_time_s_ = 0.0;
  }
  reverse_hold_reward = weights_.reverse_hold_penalty * reverse_hold_time_s_;
  add_term(
    result,
    "reverse_hold_penalty",
    weights_.reverse_hold_penalty != 0.0
      ? (reverse_hold_reward / weights_.reverse_hold_penalty)
      : reverse_hold_time_s_,
    reverse_hold_reward
  );

  // ------------------------------------------------------------
  // 6) Gear switch penalty (same as yours)
  // ------------------------------------------------------------
  if (has_prev_gear_ && telemetry.requested_reverse_gear != prev_reverse_gear_) {
    gear_switch_times_.push_back(elapsed_time_s_);
    const double window_s = 10.0;
    while (!gear_switch_times_.empty() && (elapsed_time_s_ - gear_switch_times_.front()) > window_s
    ) {
      gear_switch_times_.pop_front();
    }
    const auto switch_count      = static_cast<double>(gear_switch_times_.size());
    double     gear_switch_value = 1.0;
    if (switch_count >= 2.0) {
      gear_switch_value = std::pow(2.0, switch_count - 1.0);
      gear_switch_value = std::min(gear_switch_value, 4.0); // clamp (optional)
    }
    gear_switch_reward = weights_.gear_switch_penalty * gear_switch_value;
    add_term(
      result,
      "gear_switch_penalty",
      weights_.gear_switch_penalty != 0.0
        ? (gear_switch_reward / weights_.gear_switch_penalty)
        : gear_switch_value,
      gear_switch_reward
    );
  }
  prev_reverse_gear_ = telemetry.requested_reverse_gear;
  has_prev_gear_     = true;

  // ------------------------------------------------------------
  // 7) Collisions
  // ------------------------------------------------------------
  if (collision_active) {
    collision_severity_reward = weights_.collision_penalty_scale * collision.score * dt;
    // std::println("collision weight={} score={} reward={}", weights_.collision_penalty_scale, collision.score, collision_severity_reward);
    add_term(
      result,
      "collision",
      weights_.collision_penalty_scale != 0.0
        ? (collision_severity_reward / weights_.collision_penalty_scale)
        : (collision.score * dt),
      collision_severity_reward
    );
  }

  // ------------------------------------------------------------
  // 8) Control smoothness (your version, unchanged)
  // ------------------------------------------------------------
  if (prev_cmd_) {
    const double curr_speed =
      telemetry.command.reverse_gear ? -telemetry.command.speed : telemetry.command.speed;
    const double prev_speed = prev_cmd_->reverse_gear ? -prev_cmd_->speed : prev_cmd_->speed;
    const double speed_jerk = std::abs(curr_speed - prev_speed) / safe_dt;
    const double steer_jerk = std::abs(telemetry.command.steer - prev_cmd_->steer) / safe_dt;

    const double jerk_cap                 = 50.0;
    const double control_smoothness_value = std::min(speed_jerk + steer_jerk, jerk_cap);

    control_smoothness_reward = weights_.control_smoothness * control_smoothness_value * dt;
    add_term(
      result,
      "control_smoothness",
      weights_.control_smoothness != 0.0
        ? (control_smoothness_reward / weights_.control_smoothness)
        : (control_smoothness_value * dt),
      control_smoothness_reward
    );
  }
  prev_cmd_ = telemetry.command;

  // ------------------------------------------------------------
  // 9) Steer magnitude penalty (linear) + hold penalty
  // ------------------------------------------------------------
  const double steer_value = telemetry.command.steer;
  steer_penalty_reward     = weights_.steer_penalty * std::abs(steer_value) * dt;
  add_term(result, "steer_penalty", std::abs(steer_value) * dt, steer_penalty_reward);

  bool steer_holding = false;
  if (std::abs(steer_value) > weights_.steer_hold_threshold) {
    steer_hold_time_s_ += dt;
    steer_holding = true;
  } else {
    const double decay = dt / std::max(1e-6, weights_.steer_hold_decay_s);
    steer_hold_time_s_ = std::max(0.0, steer_hold_time_s_ - decay);
  }
  if (steer_holding) {
    steer_hold_reward = weights_.steer_hold_penalty * steer_hold_time_s_;
    add_term(
      result,
      "steer_hold_penalty",
      weights_.steer_hold_penalty != 0.0
        ? (steer_hold_reward / weights_.steer_hold_penalty)
        : steer_hold_time_s_,
      steer_hold_reward
    );
    if (weights_.steer_hold_terminate_s > 0.0 &&
        steer_hold_time_s_ >= weights_.steer_hold_terminate_s) {
      stall_terminal_            = true;
      steer_hold_terminal_reward = weights_.steer_hold_terminal_penalty;
      add_term(result, "steer_hold_terminal_penalty", 1.0, steer_hold_terminal_reward);
    }
  }

  // ------------------------------------------------------------
  // 10) Yaw rate + heading alignment: gate them to stalled / near goal
  // ------------------------------------------------------------
  // Penalize yaw rate mainly when it isn't producing progress (orbit/spin)
  const bool near_goal_yaw = goal_dist > 1e-3 && goal_dist < weights_.yaw_rate_near_goal_dist;
  if (is_stalled || near_goal_yaw) {
    const double yaw_rate_value = std::abs(telemetry.velocity.angular);
    yaw_rate_reward             = weights_.yaw_rate_penalty * yaw_rate_value * dt;
    add_term(result, "yaw_rate_penalty", std::abs(telemetry.velocity.angular) * dt, yaw_rate_reward);
  }

  // Heading alignment bonus only near goal (otherwise detours are normal)
  const double heading_gate_dist = 10.0; // meters (tune)
  if (goal_dist > 1e-3 && goal_dist < heading_gate_dist) {
    const double goal_heading = std::atan2(to_goal.y, to_goal.x);
    const double heading_error =
      std::abs(core::math::normalize_angle(goal_heading - telemetry.pose.yaw));
    const double heading_alignment = std::max(0.0, 1.0 - (heading_error / std::numbers::pi));
    heading_alignment_reward       = weights_.heading_alignment_bonus * heading_alignment * dt;
    add_term(
      result,
      "heading_alignment",
      weights_.heading_alignment_bonus != 0.0
        ? (heading_alignment_reward / weights_.heading_alignment_bonus)
        : heading_alignment * dt,
      heading_alignment_reward
    );
  }

  // ------------------------------------------------------------
  // 11) Goal reached bonus
  // ------------------------------------------------------------
  if (telemetry.final_goal_reached) {
    double goal_bonus = weights_.goal_reached_bonus;
    if (weights_.goal_reached_bonus_decay_steps > 0.0) {
      const double decay = std::max(
        0.0, 1.0 - (telemetry.episode_step_count / weights_.goal_reached_bonus_decay_steps)
      );
      goal_bonus = weights_.goal_reached_bonus_min +
                   (weights_.goal_reached_bonus - weights_.goal_reached_bonus_min) * decay;
    }
    goal_reached_reward = goal_bonus;
    add_term(result, "goal_bonus", 1.0, goal_reached_reward);
  }

  return result;
}

void DeprecatedRewardFunction::reset()
{
  prev_cmd_.reset();
  has_prev_goal_dist_ = false;
  prev_goal_dist_     = 0.0;
  goal_dist_hist_.clear();
  stall_time_s_      = 0.0;
  stall_ema_         = 0.0;
  stall_terminal_    = false;
  strike_timer_s_      = 0.0;
  prev_reverse_gear_   = false;
  has_prev_gear_       = false;
  elapsed_time_s_      = 0.0;
  gear_switch_times_.clear();
  steer_hold_time_s_   = 0.0;
  reverse_hold_time_s_ = 0.0;
}

std::string DeprecatedRewardFunction::description() const
{
  return "Deprecated reward function with progress, collision penalties, lane adherence, "
         "danger-zone avoidance, control smoothness, yaw stability, goal bonus, and stall "
         "penalties/termination";
}

} // namespace ddrl::core::rl
