/**
 * @file object.cpp
 * @brief Clean implementation of static and dynamic world objects
 *
 * Grand Refactoring: Removed all respawn logic (now owned by World/SpawnManager)
 * Simplified FSM: ACTIVE, STOPPED, INACTIVE (was 8 states)
 * Objects focus purely on motion and behavior
 */

#include "sim/object.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <cmath>
#include <control/pid_controller.hpp>
#include <control/pure_pursuit.hpp>
#include <core/base/variant_overload.hpp>
#include <random>
#include <vector>

#include "logging/logging.hpp"

namespace ddrl::sim {

using core::Result;
using core::math::normalize_angle;

// ============================================================================
// StaticObject Implementation (unchanged)
// ============================================================================

class StaticObject::Impl
{
public:
  explicit Impl(Params&& params) : object_id_(params.object_id), config_(std::move(params.config))
  {
  }

  [[nodiscard]] uint32_t                         get_id() const { return object_id_; }
  [[nodiscard]] const config::StaticInstanceCfg& get_config() const { return *config_; }

private:
  uint32_t                                         object_id_;
  std::shared_ptr<const config::StaticInstanceCfg> config_;
};

StaticObject::StaticObject(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}
StaticObject::~StaticObject() = default;

StaticObject::StaticObject(StaticObject&&) noexcept            = default;
StaticObject& StaticObject::operator=(StaticObject&&) noexcept = default;

const config::StaticInstanceCfg& StaticObject::get_config() const
{
  return impl_->get_config();
}

uint32_t StaticObject::get_id() const
{
  return impl_->get_id();
}

// ============================================================================
// DynamicObject Implementation (REFACTORED - Clean & Minimal)
// ============================================================================

class DynamicObject::Impl
{
private:
  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("DynamicObject");

  uint32_t                                          object_id_{0};
  std::shared_ptr<const config::DynamicInstanceCfg> config_;
  std::shared_ptr<const config::DynamicProfile>     profile_;

  // Object state (simplified)
  DynamicObjectState state_;
  core::math::Pose2  current_pose_;
  core::math::Twist2 current_vel_;

  // Motion state
  double                          sim_time_{0.0};
  double                          stop_start_time_{-1.0};
  size_t                          current_waypoint_idx_{0};
  double                          current_speed_{0.0};
  double                          target_speed_{0.0};
  std::vector<core::math::Point2> waypoints_world_;
  double                          waypoint_reached_tol_{0.0};
  bool                            waypoints_loop_{false};

  // Collision state
  bool   in_collision_{false};
  double collision_start_time_{0.0};
  double stop_dwell_random_{1.5};
  double stop_dwell_collision_{3.0};

  // Randomness (for stochastic stops)
  uint64_t     rng_seed_{0};
  std::mt19937 rng_;
  double       next_random_stop_time_{-1.0};

  enum class FollowType : uint8_t { LINEAR_CONST, LINEAR_PID, ACKERMANN };
  FollowType                        follow_type_{FollowType::LINEAR_CONST};
  const config::LinearConstVelCfg*  linear_const_cfg_{nullptr};
  const config::LinearPIDCfg*       linear_pid_cfg_{nullptr};
  const config::AckermannFollowCfg* ackermann_cfg_{nullptr};

public:
  explicit Impl(Params&& params);

  Result<void> update(double dt);
  void         on_collision();
  void         teleport(const core::math::Pose2& new_pose);
  void         deactivate();
  void         activate();
  void         override_waypoints(std::vector<core::math::Point2> waypoints_world, bool loop);

  [[nodiscard]] uint32_t                          get_id() const { return object_id_; }
  [[nodiscard]] const config::DynamicInstanceCfg& get_config() const { return *config_; }
  [[nodiscard]] const core::math::Pose2&          get_pose() const { return current_pose_; }
  [[nodiscard]] const core::math::Twist2&         get_velocity() const { return current_vel_; }
  [[nodiscard]] DynamicObjectState                get_state() const { return state_; }
  [[nodiscard]] bool is_active() const { return state_ != DynamicObjectState::INACTIVE; }
  [[nodiscard]] bool is_stuck() const { return state_ == DynamicObjectState::STOPPED; }
  [[nodiscard]] bool was_collision_stopped() const;

private:
  // Core motion update
  void update_motion(double dt);
  void integrate_motion(double dt);

  // Waypoint following
  void update_waypoint_following(double dt);
  bool check_waypoint_reached();
  void advance_to_next_waypoint();

  // Helper functions
  void sample_new_target_speed();
  void schedule_next_random_stop();

  [[nodiscard]] double        motion_caps_max_speed() const;
  [[nodiscard]] double        motion_caps_max_accel() const;
  [[nodiscard]] double        motion_caps_max_decel() const;
  [[nodiscard]] static double clamp_heading_rate(double desired_rate);
};

// ============================================================================
// Constructor
// ============================================================================

DynamicObject::Impl::Impl(Params&& params)
    : object_id_(params.object_id),
      config_(std::move(params.config)),
      profile_(std::move(params.profile)),
      state_(DynamicObjectState::INACTIVE),
      current_pose_(config_->pose_world),
      current_vel_({}, 0.0),
      rng_seed_(static_cast<uint64_t>(object_id_)),
      rng_(rng_seed_)
{
  waypoint_reached_tol_ = profile_->waypoints_local.reached_pos_tol;
  waypoints_loop_       = profile_->waypoints_local.loop;
  waypoints_world_.reserve(profile_->waypoints_local.points.size());
  for (const auto& waypoint : profile_->waypoints_local.points) {
    waypoints_world_.push_back(config_->pose_world.transform_point(waypoint.p_local.pos));
  }

  std::visit(
    core::Overloaded{
      [&](const config::LinearConstVelCfg& cfg) {
        follow_type_      = FollowType::LINEAR_CONST;
        linear_const_cfg_ = &cfg;
      },
      [&](const config::LinearPIDCfg& cfg) {
        follow_type_    = FollowType::LINEAR_PID;
        linear_pid_cfg_ = &cfg;
      },
      [&](const config::AckermannFollowCfg& cfg) {
        follow_type_   = FollowType::ACKERMANN;
        ackermann_cfg_ = &cfg;
      }
    },
    profile_->follow_cfg
  );

  sample_new_target_speed();
  schedule_next_random_stop();
}

// ============================================================================
// Main Update Loop
// ============================================================================

Result<void> DynamicObject::Impl::update(double dt)
{
  sim_time_ += dt;

  // Inactive objects don't move
  if (state_ == DynamicObjectState::INACTIVE) {
    current_vel_   = core::math::Twist2({}, 0.0);
    current_speed_ = 0.0;
    return {};
  }

  // STOPPED objects just decelerate to zero and stay stopped (waiting for despawn)
  if (state_ == DynamicObjectState::STOPPED) {
    const double decel    = std::max(15.0, motion_caps_max_decel());
    current_speed_        = std::max(0.0, current_speed_ - decel * dt);
    current_vel_.linear.x = current_speed_ * std::cos(current_pose_.yaw);
    current_vel_.linear.y = current_speed_ * std::sin(current_pose_.yaw);
    current_vel_.angular  = 0.0;

    integrate_motion(dt);

    const double dwell = in_collision_ ? stop_dwell_collision_ : stop_dwell_random_;
    if (stop_start_time_ >= 0.0 && sim_time_ - stop_start_time_ >= dwell) {
      state_           = DynamicObjectState::ACTIVE;
      in_collision_    = false;
      stop_start_time_ = -1.0;
      if (current_waypoint_idx_ >= waypoints_world_.size()) {
        current_waypoint_idx_ = 0;
      }
      sample_new_target_speed();
      // Rebuild velocity to align with current heading
      current_vel_.linear.x = current_speed_ * std::cos(current_pose_.yaw);
      current_vel_.linear.y = current_speed_ * std::sin(current_pose_.yaw);
    }
    return {};
  }

  // ACTIVE state: normal motion
  update_motion(dt);
  integrate_motion(dt);

  return {};
}

// ============================================================================
// Motion Update (ACTIVE state)
// ============================================================================

void DynamicObject::Impl::update_motion(double dt)
{
  const double accel = motion_caps_max_accel();
  const double decel = motion_caps_max_decel();

  // Check for random stop event
  if (next_random_stop_time_ > 0.0 && sim_time_ >= next_random_stop_time_) {
    state_           = DynamicObjectState::STOPPED;
    in_collision_    = false;
    stop_start_time_ = sim_time_;
    schedule_next_random_stop();
    return;
  }

  // Update waypoint following
  update_waypoint_following(dt);

  // Accelerate/decelerate toward target speed
  if (current_speed_ < target_speed_) {
    current_speed_ = std::min(target_speed_, current_speed_ + accel * dt);
  } else if (current_speed_ > target_speed_) {
    current_speed_ = std::max(target_speed_, current_speed_ - decel * dt);
  }

  // Check if waypoint reached
  if (check_waypoint_reached()) {
    advance_to_next_waypoint();
  }
}

// ============================================================================
// Waypoint Following
// ============================================================================

void DynamicObject::Impl::update_waypoint_following(double /*dt*/)
{
  if (waypoints_world_.empty()) {
    // No waypoints: just move forward
    current_vel_.linear.x = current_speed_ * std::cos(current_pose_.yaw);
    current_vel_.linear.y = current_speed_ * std::sin(current_pose_.yaw);
    current_vel_.angular  = 0.0;
    return;
  }

  if (current_waypoint_idx_ >= waypoints_world_.size()) {
    // Reached end
    current_vel_   = core::math::Twist2({}, 0.0);
    current_speed_ = 0.0;
    return;
  }

  const auto& wp_world = waypoints_world_[current_waypoint_idx_];

  // Calculate direction to waypoint
  double dx         = wp_world.x - current_pose_.pos.x;
  double dy         = wp_world.y - current_pose_.pos.y;
  double target_yaw = std::atan2(dy, dx);
  double yaw_error  = normalize_angle(target_yaw - current_pose_.yaw);

  // Controller based on profile type
  switch (follow_type_) {
    case FollowType::LINEAR_CONST:
    case FollowType::LINEAR_PID: {
      current_vel_.angular  = clamp_heading_rate(yaw_error * 2.0);
      current_vel_.linear.x = current_speed_ * std::cos(current_pose_.yaw);
      current_vel_.linear.y = current_speed_ * std::sin(current_pose_.yaw);
      break;
    }
    case FollowType::ACKERMANN: {
      const auto& cfg = *ackermann_cfg_;
      double      steering =
        std::clamp(yaw_error, -cfg.acker_caps.max_steer_angle, cfg.acker_caps.max_steer_angle);
      double turn_radius =
        (std::abs(steering) > 1e-6) ? cfg.acker_caps.wheel_base / std::tan(steering) : 1e6;
      current_vel_.angular  = (std::abs(turn_radius) > 1e-6) ? current_speed_ / turn_radius : 0.0;
      current_vel_.linear.x = current_speed_ * std::cos(current_pose_.yaw);
      current_vel_.linear.y = current_speed_ * std::sin(current_pose_.yaw);
      break;
    }
  }
}

bool DynamicObject::Impl::check_waypoint_reached()
{
  if (waypoints_world_.empty() || current_waypoint_idx_ >= waypoints_world_.size()) {
    return false;
  }

  const auto& wp_world = waypoints_world_[current_waypoint_idx_];
  double      dist     = core::math::distance(current_pose_.pos, wp_world);
  return dist < waypoint_reached_tol_;
}

void DynamicObject::Impl::advance_to_next_waypoint()
{
  current_waypoint_idx_++;

  if (current_waypoint_idx_ >= waypoints_world_.size()) {
    if (waypoints_loop_) {
      // Loop back to start
      current_waypoint_idx_ = 0;
      sample_new_target_speed(); // Vary speed each loop
    } else {
      // Reached end, deactivate
      state_ = DynamicObjectState::INACTIVE;
    }
  }
}

// ============================================================================
// Motion Integration
// ============================================================================

void DynamicObject::Impl::integrate_motion(double dt)
{
  // Simple Euler integration
  current_pose_.pos.x += current_vel_.linear.x * dt;
  current_pose_.pos.y += current_vel_.linear.y * dt;
  current_pose_.yaw += current_vel_.angular * dt;
  current_pose_.yaw = normalize_angle(current_pose_.yaw);
}

// ============================================================================
// Collision Handling (Simplified)
// ============================================================================

void DynamicObject::Impl::on_collision()
{
  if (state_ == DynamicObjectState::INACTIVE) {
    return;
  }

  if (state_ == DynamicObjectState::ACTIVE) {
    state_                = DynamicObjectState::STOPPED;
    in_collision_         = true;
    collision_start_time_ = sim_time_;
    stop_start_time_      = sim_time_;
  }
}

bool DynamicObject::Impl::was_collision_stopped() const
{
  return in_collision_;
}

// ============================================================================
// Spawn/Despawn Interface (called by World/SpawnManager)
// ============================================================================

void DynamicObject::Impl::teleport(const core::math::Pose2& new_pose)
{
  current_pose_         = new_pose;
  current_vel_          = core::math::Twist2({}, 0.0);
  current_speed_        = 0.0;
  current_waypoint_idx_ = 0;
  in_collision_         = false;
  stop_start_time_      = -1.0;

  sample_new_target_speed();
  schedule_next_random_stop();
}

void DynamicObject::Impl::deactivate()
{
  state_           = DynamicObjectState::INACTIVE;
  current_vel_     = core::math::Twist2({}, 0.0);
  current_speed_   = 0.0;
  stop_start_time_ = -1.0;

}

void DynamicObject::Impl::activate()
{
  state_                = DynamicObjectState::ACTIVE;
  current_waypoint_idx_ = 0;
  in_collision_         = false;
  stop_start_time_      = -1.0;

  sample_new_target_speed();
  schedule_next_random_stop();

}

void DynamicObject::Impl::override_waypoints(
  std::vector<core::math::Point2> waypoints_world, bool loop
)
{
  waypoints_world_      = std::move(waypoints_world);
  waypoints_loop_       = loop;
  current_waypoint_idx_ = 0;
  state_                = DynamicObjectState::ACTIVE;
  stop_start_time_      = -1.0;
  in_collision_         = false;
  current_speed_        = 0.0;

  sample_new_target_speed();
  schedule_next_random_stop();
}

// ============================================================================
// Helper Functions
// ============================================================================

void DynamicObject::Impl::sample_new_target_speed()
{
  switch (follow_type_) {
    case FollowType::LINEAR_CONST: {
      const auto&                      cfg = *linear_const_cfg_;
      std::uniform_real_distribution<> dist(cfg.speed_min, cfg.speed_max);
      target_speed_ = dist(rng_);
      break;
    }
    case FollowType::LINEAR_PID: {
      const auto&                      cfg = *linear_pid_cfg_;
      std::uniform_real_distribution<> dist(cfg.cruise_speed_min, cfg.cruise_speed_max);
      target_speed_ = dist(rng_);
      break;
    }
    case FollowType::ACKERMANN: {
      const auto&                      cfg = *ackermann_cfg_;
      std::uniform_real_distribution<> dist(cfg.cruise_speed_min, cfg.cruise_speed_max);
      target_speed_ = dist(rng_);
      break;
    }
  }
}

void DynamicObject::Impl::schedule_next_random_stop()
{
  // Schedule random stop with 5% probability per second
  std::exponential_distribution<> dist(0.05); // Mean time = 20 seconds
  next_random_stop_time_ = sim_time_ + dist(rng_);
}

double DynamicObject::Impl::motion_caps_max_speed() const
{
  switch (follow_type_) {
    case FollowType::LINEAR_CONST:
      return linear_const_cfg_->caps.max_speed;
    case FollowType::LINEAR_PID:
      return linear_pid_cfg_->caps.max_speed;
    case FollowType::ACKERMANN:
      return ackermann_cfg_->acker_caps.caps_fwd.max_speed;
  }
  return 0.0;
}

double DynamicObject::Impl::motion_caps_max_accel() const
{
  switch (follow_type_) {
    case FollowType::LINEAR_CONST:
      return linear_const_cfg_->caps.max_accel;
    case FollowType::LINEAR_PID:
      return linear_pid_cfg_->caps.max_accel;
    case FollowType::ACKERMANN:
      return ackermann_cfg_->acker_caps.caps_fwd.max_accel;
  }
  return 0.0;
}

double DynamicObject::Impl::motion_caps_max_decel() const
{
  switch (follow_type_) {
    case FollowType::LINEAR_CONST:
      return linear_const_cfg_->caps.max_decel;
    case FollowType::LINEAR_PID:
      return linear_pid_cfg_->caps.max_decel;
    case FollowType::ACKERMANN:
      return ackermann_cfg_->acker_caps.caps_fwd.max_decel;
  }
  return 0.0;
}

double DynamicObject::Impl::clamp_heading_rate(double desired_rate)
{
  // Simple rate limiting
  constexpr double kMaxRate = 1.0; // rad/s
  return std::clamp(desired_rate, -kMaxRate, kMaxRate);
}

// ============================================================================
// DynamicObject Public API (delegates to Impl)
// ============================================================================

DynamicObject::DynamicObject(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

DynamicObject::~DynamicObject() = default;

DynamicObject::DynamicObject(DynamicObject&&) noexcept            = default;
DynamicObject& DynamicObject::operator=(DynamicObject&&) noexcept = default;

Result<void> DynamicObject::update(double dt)
{
  return impl_->update(dt);
}

void DynamicObject::on_collision()
{
  impl_->on_collision();
}

void DynamicObject::teleport(const core::math::Pose2& new_pose)
{
  impl_->teleport(new_pose);
}

void DynamicObject::deactivate()
{
  impl_->deactivate();
}

void DynamicObject::activate()
{
  impl_->activate();
}

void DynamicObject::override_waypoints(std::vector<core::math::Point2> waypoints_world, bool loop)
{
  impl_->override_waypoints(std::move(waypoints_world), loop);
}

const config::DynamicInstanceCfg& DynamicObject::get_config() const
{
  return impl_->get_config();
}

uint32_t DynamicObject::get_id() const
{
  return impl_->get_id();
}

const core::math::Pose2& DynamicObject::get_pose() const
{
  return impl_->get_pose();
}

const core::math::Twist2& DynamicObject::get_velocity() const
{
  return impl_->get_velocity();
}

DynamicObjectState DynamicObject::get_state() const
{
  return impl_->get_state();
}

bool DynamicObject::is_active() const
{
  return impl_->is_active();
}

bool DynamicObject::is_stuck() const
{
  return impl_->is_stuck();
}

bool DynamicObject::was_collision_stopped() const
{
  return impl_->was_collision_stopped();
}

} // namespace ddrl::sim

// ============================================================================
// Utility Functions
// ============================================================================

std::string to_string(ddrl::sim::DynamicObjectState state)
{
  switch (state) {
    case ddrl::sim::DynamicObjectState::ACTIVE:
      return "ACTIVE";
    case ddrl::sim::DynamicObjectState::STOPPED:
      return "STOPPED";
    case ddrl::sim::DynamicObjectState::INACTIVE:
      return "INACTIVE";
    default:
      return "UNKNOWN";
  }
}
