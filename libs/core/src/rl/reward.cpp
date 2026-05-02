#include "core/rl/reward.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <cmath>
#include <string_view>

#include "logging/logging.hpp"

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

RewardStepResult DefaultRewardFunction::compute(
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
  double time_raw               = dt;
  double time_limit_raw         = 0.0;
  double progress_raw           = 0.0;
  double progress_steer_gate    = 1.0;
  double effort_raw             = 0.0;
  double steer_raw              = 0.0;
  double heading_raw            = 0.0;
  double proximity_raw          = 0.0;
  double collision_raw          = 0.0;
  double reverse_raw            = 0.0;
  double gear_switch_raw        = 0.0;
  double stall_raw              = 0.0;
  double stall_terminal_raw     = 0.0;
  double waypoint_raw           = 0.0;
  double goal_raw               = 0.0;

  if (!last_waypoint_index_ || *last_waypoint_index_ != telemetry.waypoint_index) {
    goal_dist_hist_.clear();
    motion_pos_hist_.clear();
    reverse_penalty_ema_ = 0.0;
    steer_persistence_ema_ = 0.0;
    stall_ema_           = 0.0;
    stall_terminal_      = false;
    last_waypoint_index_ = telemetry.waypoint_index;
  }

  // Geometry
  const core::math::Point2 position          = telemetry.pose.pos;
  const core::math::Point2 waypoint_position = telemetry.waypoint_position;
  const core::math::Vec2   to_goal           = waypoint_position - position;
  const double             goal_dist         = telemetry.distance_to_waypoint;
  const double clearance_min_m               = std::max(0.0, weights_.clearance_gate_dist_min_m);
  const double clearance_max_m =
    std::max(clearance_min_m + 1e-6, weights_.clearance_gate_dist_max_m);
  const double forward_clearance_m = telemetry.forward_clearance_m;
  const double clearance_gate =
    std::isfinite(forward_clearance_m)
      ? std::clamp(
          (forward_clearance_m - clearance_min_m) / (clearance_max_m - clearance_min_m), 0.0, 1.0
        )
      : 1.0;
  const double proximity_signal = 1.0 - clearance_gate;

  // Maintain goal-distance history for net progress checks.
  const double window_s     = std::max(weights_.stall_window_s, safe_dt);
  const int    max_samples  = std::max(2, static_cast<int>(std::ceil(window_s / safe_dt)));
  goal_dist_hist_.push_back(goal_dist);
  if (static_cast<int>(goal_dist_hist_.size()) > max_samples) {
    goal_dist_hist_.pop_front();
  }
  motion_pos_hist_.push_back(position);
  if (static_cast<int>(motion_pos_hist_.size()) > max_samples) {
    motion_pos_hist_.pop_front();
  }
  double net_progress_m = 0.0;
  if (goal_dist_hist_.size() >= 2) {
    net_progress_m = goal_dist_hist_.front() - goal_dist_hist_.back();
  }
  const double net_progress_rate_m_s = net_progress_m / window_s;
  double       net_motion_m          = 0.0;
  if (motion_pos_hist_.size() >= 2) {
    const auto& oldest = motion_pos_hist_.front();
    const auto& newest = motion_pos_hist_.back();
    net_motion_m = std::hypot(newest.x - oldest.x, newest.y - oldest.y);
  }
  const double net_motion_rate_m_s = net_motion_m / window_s;

  add_term(result, "time_penalty", time_raw, weights_.time_penalty * time_raw);
  if (telemetry.time_limit_reached) {
    time_limit_raw = 1.0;
    add_term(result, "time_limit_penalty", time_limit_raw, weights_.time_limit_penalty);
  }

  {
    const double eps = std::max(0.0, weights_.progress_gate_epsilon_m_s);
    double gated_rate = 0.0;
    if (net_progress_rate_m_s > eps) {
      gated_rate = net_progress_rate_m_s;
    } else if (net_progress_rate_m_s < -eps) {
      gated_rate = net_progress_rate_m_s; // penalize moving away
    }
    const double denom  = std::max(1e-6, weights_.progress_norm_speed_m_s);
    const double v_norm = std::clamp(gated_rate / denom, -1.0, 1.0);
    const double steer_gate_min =
      std::clamp(weights_.progress_steer_gate_min, 0.0, 1.0);
    const double steer_gate_power = std::max(weights_.progress_steer_gate_power, 1e-6);
    const double steer_gate_shape =
      std::pow(std::clamp(std::abs(steer_persistence_ema_), 0.0, 1.0), steer_gate_power);
    progress_steer_gate = 1.0 - (1.0 - steer_gate_min) * steer_gate_shape;
    progress_raw = v_norm * progress_steer_gate * dt;
    add_term(result, "progress", progress_raw, weights_.progress * progress_raw);
  }

  {
    const core::math::Vec2 v{telemetry.velocity.linear.x, telemetry.velocity.linear.y};
    const double           speed_m_s = std::hypot(v.x, v.y);
    double                 v_toward_inst = 0.0;
    if (goal_dist > 1e-6 && speed_m_s > 1e-6) {
      const core::math::Vec2 goal_dir = to_goal * (1.0 / goal_dist);
      v_toward_inst = v.x * goal_dir.x + v.y * goal_dir.y;
    }
    const double v_toward_pos = std::clamp(v_toward_inst, 0.0, speed_m_s);
    const double lateral_m_s  = std::max(0.0, speed_m_s - v_toward_pos);
    const double denom        = std::max(1e-6, weights_.progress_norm_speed_m_s);
    const double lateral_norm = std::clamp(lateral_m_s / denom, 0.0, 1.0);
    effort_raw = lateral_norm * dt;
    add_term(result, "effort_penalty", effort_raw, weights_.effort_penalty * effort_raw);
  }

  {
    const double denom = std::max(1e-6, weights_.steer_norm_abs);
    const double steer_signed_norm =
      std::clamp(telemetry.command.steer / denom, -1.0, 1.0) * clearance_gate;
    steer_persistence_ema_ =
      ema_update(steer_persistence_ema_, steer_signed_norm, weights_.steer_penalty_tau_s);
    steer_raw = std::abs(steer_persistence_ema_) * dt;
    add_term(result, "steer_penalty", steer_raw, weights_.steer_penalty * steer_raw);
  }

  {
    const double heading_error =
      core::math::normalize_angle(telemetry.heading_to_waypoint - telemetry.pose.yaw);
    heading_raw = std::max(0.0, std::cos(heading_error)) * dt;
    add_term(result, "heading_to_goal", heading_raw, weights_.heading_to_goal_reward * heading_raw);
  }

  proximity_raw = proximity_signal * dt;
  add_term(result, "proximity", proximity_raw, weights_.proximity_reward * proximity_raw);

  if (collision_active) {
    const double score = std::clamp(collision.score, 0.0, 1.0);
    collision_raw = score * dt;
    add_term(result, "collision", collision_raw, weights_.collision_penalty_scale * collision_raw);
  }

  {
    const double reverse_sample = (telemetry.requested_reverse_gear ? 1.0 : 0.0) * clearance_gate;
    reverse_penalty_ema_ =
      ema_update(reverse_penalty_ema_, reverse_sample, weights_.reverse_penalty_tau_s);
    reverse_raw = reverse_penalty_ema_ * dt;
    add_term(result, "reverse_penalty", reverse_raw, weights_.reverse_penalty * reverse_raw);
  }
  if (has_prev_requested_reverse_ && telemetry.requested_reverse_gear != prev_requested_reverse_) {
    gear_switch_raw = 1.0;
    add_term(result, "gear_switch_penalty", gear_switch_raw, weights_.gear_switch_penalty);
  }
  prev_requested_reverse_ = telemetry.requested_reverse_gear;
  has_prev_requested_reverse_ = true;

  const double progress_eps = std::max(weights_.stall_progress_epsilon, 1e-6);
  const double stall_sample =
    std::clamp((progress_eps - net_motion_rate_m_s) / progress_eps, 0.0, 1.0) * clearance_gate;
  stall_ema_    = ema_update(stall_ema_, stall_sample, weights_.stall_tau_s);
  stall_raw = stall_ema_ * dt;
  add_term(result, "stall_penalty", stall_raw, weights_.stall_penalty_scale * stall_raw);
  if (!stall_terminal_ && weights_.stall_limit > 0.0 && stall_ema_ >= weights_.stall_limit) {
    stall_terminal_ = true;
    stall_terminal_raw = 1.0;
    add_term(result, "stall_terminal_penalty", stall_terminal_raw, weights_.stall_terminal_penalty);
  }

  if (telemetry.waypoint_reached && !telemetry.final_goal_reached) {
    waypoint_raw = 1.0;
    add_term(result, "waypoint_bonus", waypoint_raw, weights_.waypoint_reached_bonus);
  }
  if (telemetry.final_goal_reached) {
    goal_raw = 1.0;
    add_term(result, "goal_bonus", goal_raw, weights_.goal_reached_bonus);
  }

  return result;
}

void DefaultRewardFunction::reset()
{
  has_prev_requested_reverse_ = false;
  prev_requested_reverse_ = false;
  reverse_penalty_ema_   = 0.0;
  goal_dist_hist_.clear();
  motion_pos_hist_.clear();
  last_waypoint_index_.reset();
  steer_persistence_ema_ = 0.0;
  stall_ema_           = 0.0;
  stall_terminal_      = false;
}

std::string DefaultRewardFunction::description() const
{
  return "Minimal default reward function with normalized progress, collision/danger/time penalties, "
         "clearance-gated persistent-steer/reverse/stall EMA penalties, proximity shaping, "
         "heading-to-goal shaping, and hard stall termination";
}

std::unique_ptr<RewardFunction>
create_reward_function(const std::string& type, RewardFunctionWeights weights)
{
  const std::string t = type.empty() ? "default" : type;
  if (t == "default") {
    return std::make_unique<DefaultRewardFunction>(weights);
  }
  if (t == "deprecated" || t == "legacy") {
    return std::make_unique<DeprecatedRewardFunction>(weights);
  }
  // Unknown types fall back to the default minimal reward.
  return std::make_unique<DefaultRewardFunction>(weights);
}

} // namespace ddrl::core::rl
