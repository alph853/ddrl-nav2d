#include "sim/robot.hpp"

#include "core/rl/rl.hpp"
#include <chrono>
#include <cmath>
#include <core/base/variant_overload.hpp>
#include <core/math/pose.hpp>
#include <core/math/twist.hpp>
#include <kinematics/ackermann.hpp>
#include <logging/logging.hpp>
#include <queue>
#include <span>

namespace ddrl::sim {

using config::RobotProfile;
using core::ErrorCode;
using core::make_error;
using core::make_error_from;
using core::Overloaded;
using core::Result;

using core::sim::CollisionContact;

class Robot::Impl
{
private:
  core::math::Pose2                  current_pose_;
  core::math::Twist2                 current_vel_;
  core::kinematics::AckermannCommand current_cmd_;
  core::rl::Action                   last_requested_action_;
  core::kinematics::AckermannCommand last_requested_cmd_;
  core::kinematics::AckermannCommand last_applied_cmd_;

  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("Robot");

  // Ackermann kinematics model
  std::unique_ptr<kinematics::AckermannModel> ackermann_model_;
  std::shared_ptr<const config::RLConfig>     rl_config_;
  std::shared_ptr<const config::RobotProfile> profile_;

  // Simulation dependencies
  std::shared_ptr<collision::TLAS2Pair>           scene_trees_;
  std::shared_ptr<threading::ThreadPool>          thread_pool_;
  std::shared_ptr<core::tf::TfBuffer>             tf_buffer_;
  std::shared_ptr<const config::RobotInstanceCfg> config_;
  std::shared_ptr<const config::ModelProfile>     model_profile_;

  std::string base_frame_id_;
  uint32_t    entity_id_{0};
  uint32_t    sim_id_{0};
  double      sim_time_{0.0};

  // Sensors with cooperative scheduling
  std::vector<SensorType> sensors_;

  StepUpdateInfo           last_update_info_;
  std::vector<std::string> last_collision_contacts_;

  // Delayed action queue (moved from Simulator)
  struct DelayedAction {
    double           apply_time_s; ///< Sim time when action should be applied
    core::rl::Action action;       ///< The action to apply
  };
  struct DelayedActionSooner {
    bool operator()(const DelayedAction& lhs, const DelayedAction& rhs) const
    {
      return lhs.apply_time_s > rhs.apply_time_s; // min-heap (earliest at top)
    }
  };
  std::priority_queue<DelayedAction, std::vector<DelayedAction>, DelayedActionSooner>
    delayed_actions_;

public:
  explicit Impl(Robot::Params&& params);
  ~Impl();

  Result<void> update(double dt);
  Result<void> set_actuator(const core::kinematics::AckermannCommand& cmd, double dt);
  Result<core::rl::Observation>
               build_observation(double tick_time, double tick_dt, double reward, bool is_terminal);
  Result<void> set_action(const core::rl::Action& action);

  Result<void> notify_collision(std::span<const CollisionContact> contacts, double timestamp);
  void         clear_collision_state();

  [[nodiscard]] const core::kinematics::AckermannCommand& last_applied_command() const
  {
    return last_applied_cmd_;
  }
  [[nodiscard]] const core::kinematics::AckermannCommand& current_command() const
  {
    return current_cmd_;
  }
  [[nodiscard]] const core::kinematics::AckermannCommand& requested_command() const
  {
    return last_requested_cmd_;
  }
  [[nodiscard]] const core::rl::Action& requested_action() const
  {
    return last_requested_action_;
  }
  [[nodiscard]] bool reverse_gear() const
  {
    return ackermann_model_ ? ackermann_model_->reverse_gear() : false;
  }
  [[nodiscard]] double gear_hold_time_remaining() const
  {
    return ackermann_model_ ? ackermann_model_->gear_hold_time_remaining() : 0.0;
  }

  [[nodiscard]] const core::math::Pose2&  get_pose() const { return current_pose_; }
  [[nodiscard]] const core::math::Twist2& get_velocity() const { return current_vel_; }

  [[nodiscard]] const StepUpdateInfo& last_update_info() const { return last_update_info_; }
  [[nodiscard]] StepUpdateInfo&       last_update_info() { return last_update_info_; }

  [[nodiscard]] uint32_t sim_id() const { return sim_id_; }
  [[nodiscard]] uint32_t get_id() const { return entity_id_; }

  [[nodiscard]] const config::ModelProfile& get_model_profile() const { return *model_profile_; }

private:
  void process_delayed_actions();
  [[nodiscard]] core::rl::Action scale_normalized_action(core::rl::Action action) const;
};

/// Initialize robot with profile and simulation dependencies
Robot::Impl::Impl(Robot::Params&& params)
    : current_pose_(params.initial_pose),
      profile_(params.profile),
      rl_config_(std::move(params.rl_config)),
      scene_trees_(params.scene_trees),
      thread_pool_(params.thread_pool),
      tf_buffer_(params.tf_buffer),
      config_(std::move(params.config)),
      model_profile_(std::move(params.model_profile)),
      base_frame_id_(params.base_frame_id),
      entity_id_(params.entity_id),
      sim_id_(params.sim_id)
{
  if (!profile_) {
    throw std::runtime_error("Robot profile is null");
  }

  DDRL_LOG_DEBUG(
    logger_,
    "Initializing robot '{}' with model '{}', entity_id {}, sim_id {}",
    config_->name,
    profile_->model_id,
    entity_id_,
    sim_id_
  );

  // Initialize sensors from profile
  sensors_.clear();
  sensors_.reserve(profile_->sensors.size());

  for (const auto& sensor_cfg : profile_->sensors) {
    // Create sensor parameters
    SensorBase::Params sensor_params;
    sensor_params.config      = std::make_shared<const core::sensors::SensorConfig>(sensor_cfg);
    sensor_params.scene_trees = scene_trees_;
    sensor_params.thread_pool = thread_pool_;
    sensor_params.tf_buffer   = tf_buffer_;
    sensor_params.robot_base_frame = base_frame_id_;

    // Create sensor using factory (automatically calls constructor)
    auto sensor_result = create_sensor(sensor_cfg, sensor_params);
    if (!sensor_result.has_value()) {
      throw std::runtime_error("Failed to create sensor: " + sensor_result.error().msg);
    }

    sensors_.push_back(std::move(sensor_result.value()));

    // Register sensor frame in TF tree if we have a TF buffer
    if (tf_buffer_) {
      auto base_cfg = std::visit(
        [](const auto& cfg) -> const core::sensors::SensorBase& { return cfg.base; }, sensor_cfg
      );

      // Set static transform from base_link to sensor frame
      tf_buffer_->set_static_transform(base_frame_id_, base_cfg.frame_id, base_cfg.pose_rel_base);
    }
  }

  kinematics::AckermannState initial_state;
  initial_state.pose     = current_pose_;
  initial_state.speed    = 0.0;
  initial_state.steering = 0.0;

  auto caps = profile_->kinematics;

  ackermann_model_ = std::make_unique<kinematics::AckermannModel>(caps);
  ackermann_model_->reset(initial_state);

  last_update_info_.est_pose    = current_pose_;
  last_update_info_.est_vel     = current_vel_;
  last_update_info_.sensors     = &sensors_;
  last_update_info_.is_collided = false;

  sim_time_ = 0.0;
}

Robot::Impl::~Impl() = default;

/// Update robot state using Ackermann kinematics and cooperative sensor
/// scheduler
core::Result<void> Robot::Impl::update(double dt)
{
  if (!profile_) {
    return std::unexpected(make_error(ErrorCode::SIM, "Robot profile not set", "Robot::update"));
  }

  // Update simulation time
  sim_time_ += dt;

  // Process any delayed actions whose time has come
  process_delayed_actions();

  // Check if robot is colliding - limit forward motion to prevent tunneling
  // but allow backing out to enable collision recovery
  core::kinematics::AckermannCommand effective_cmd = current_cmd_;
  if (last_update_info_.is_collided) {
    // Allow reverse for collision recovery; block forward motion.
    if (!effective_cmd.reverse_gear && effective_cmd.speed > 0.0) {
      effective_cmd.speed = 0.0;
      DDRL_LOG_DEBUG_THROTTLE(
        logger_,
        1000,
        "Robot in collision - blocking forward motion (v={:.2f} -> 0.0), allowing reverse",
        current_cmd_.speed
      );
    }
    // Allow negative speed (backing out) and steering for recovery
  }
  last_applied_cmd_ = effective_cmd;

  // Step Ackermann kinematics model
  if (ackermann_model_) {
    ackermann_model_->step(std::chrono::duration<double>(dt), effective_cmd);

    // Update pose and velocity from Ackermann state
    const auto& state     = ackermann_model_->state();
    current_pose_         = state.pose;
    current_vel_.linear.x = state.speed * std::cos(state.pose.yaw);
    current_vel_.linear.y = state.speed * std::sin(state.pose.yaw);
    current_vel_.angular =
      (state.speed / profile_->kinematics.wheel_base) * std::tan(state.steering);

    const bool applied_reverse = ackermann_model_->reverse_gear();
    const double hold_remaining = ackermann_model_->gear_hold_time_remaining();
    DDRL_LOG_DEBUG_THROTTLE(
      logger_,
      2000,
      "ackermann_state cmd_speed={:.3f} cmd_reverse={} applied_reverse={} hold_s={:.2f} "
      "state_speed={:.3f} state_steer={:.3f} yaw_rate={:.3f}",
      effective_cmd.speed,
      effective_cmd.reverse_gear,
      applied_reverse,
      hold_remaining,
      state.speed,
      state.steering,
      current_vel_.angular
    );
  }

  // Update base_link transform in TF tree
  if (tf_buffer_) {
    const auto pose3d = current_pose_.to_pose3();
    tf_buffer_->push_dynamic("world", base_frame_id_, sim_time_, pose3d);
  }

  // Cooperative sensor scheduler: update all sensors
  for (auto& sensor : sensors_) {
    auto result = sensor->update(dt, sim_time_);
    if (!result.has_value()) {
      // Log error but continue with other sensors
      // In production, you might want to handle this differently
      continue;
    }
  }

  // Update step info
  last_update_info_.est_pose = current_pose_;
  last_update_info_.est_vel  = current_vel_;
  // sensors_ reference is already set in constructor
  // last_update_info_.is_collided will be set by collision detection

  return {};
}

core::Result<void>
Robot::Impl::set_actuator(const core::kinematics::AckermannCommand& cmd, double dt)
{
  (void)dt;
  current_cmd_ = cmd;
  return {};
}

Result<core::rl::Observation>
Robot::Impl::build_observation(double tick_time, double tick_dt, double reward, bool is_terminal)
{
  (void)tick_dt;

  DDRL_LOG_DEBUG_THROTTLE(
    logger_,
    2000,
    "build_observation at t={:.3f}, reward={:.3f}, terminal={}",
    tick_time,
    reward,
    is_terminal
  );

  core::rl::Observation obs;

  obs.reward            = reward;
  obs.is_terminal       = is_terminal;
  obs.est_pose          = last_update_info_.est_pose;
  obs.est_vel           = last_update_info_.est_vel;
  obs.collision_objects = last_collision_contacts_;
  obs.reverse_gear      = reverse_gear();
  obs.gear_hold_time_s  = gear_hold_time_remaining();

  obs.sensor_data.clear();
  obs.sensor_data.reserve(sensors_.size());

  for (const auto& sensor : sensors_) {
    if (sensor && sensor->has_data()) {
      core::rl::SensorObservation sensor_obs;
      sensor_obs.sensor_id   = sensor->id();
      sensor_obs.timestamp   = tick_time;
      sensor_obs.point_cloud = sensor->get_point_cloud();
      obs.sensor_data.push_back(std::move(sensor_obs));
    }
  }

  return obs;
}

core::Result<void> Robot::Impl::set_action(const core::rl::Action& action)
{
  // Calculate when the action should be applied
  const double apply_time = action.timestamp + config_->action_delay_s;
  auto applied_action = scale_normalized_action(action);

  last_requested_cmd_ = applied_action.cmd;
  last_requested_action_ = applied_action;
  delayed_actions_.push(DelayedAction{apply_time, std::move(applied_action)});

  return {};
}

core::rl::Action Robot::Impl::scale_normalized_action(core::rl::Action action) const
{
  const auto& caps = profile_->kinematics;
  const double normalized_speed =
    action.cmd.reverse_gear ? -action.cmd.speed : action.cmd.speed;
  const double speed_scale =
    normalized_speed >= 0.0 ? caps.caps_fwd.max_speed : caps.caps_rev.max_speed;
  const double signed_speed = std::clamp(normalized_speed, -1.0, 1.0) * speed_scale;
  action.cmd.reverse_gear = signed_speed < 0.0;
  action.cmd.speed        = std::abs(signed_speed);
  action.cmd.steer        =
    std::clamp(action.cmd.steer, -1.0, 1.0) * caps.max_steer_angle;
  return action;
}

void Robot::Impl::process_delayed_actions()
{
  // Apply all delayed actions whose time has come
  while (!delayed_actions_.empty() && delayed_actions_.top().apply_time_s <= sim_time_) {
    const auto action = delayed_actions_.top().action;

    // Apply the action to the actuator
    current_cmd_ = action.cmd;

    delayed_actions_.pop();
  }
}

Result<void>
Robot::Impl::notify_collision(std::span<const CollisionContact> contacts, double timestamp)
{
  const bool was_collided       = last_update_info_.is_collided;
  last_update_info_.is_collided = true;

  last_collision_contacts_.clear();
  last_collision_contacts_.reserve(contacts.size());
  for (const auto& contact : contacts) {
    last_collision_contacts_.push_back(contact.other_name);
  }

  if (!was_collided) {
    DDRL_LOG_DEBUG(
      logger_,
      "Collision detected at t={:.3f} with contacts={}",
      timestamp,
      ::to_string(std::span<const CollisionContact>(contacts))
    );
  }
  return {};
}

void Robot::Impl::clear_collision_state()
{
  if (!last_update_info_.is_collided && last_collision_contacts_.empty()) {
    return;
  }
  last_update_info_.is_collided = false;
  last_collision_contacts_.clear();
}

// =============================================================================
// Robot Public APIs (delegate to impl_)
// =============================================================================

Robot::Robot(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

Robot::~Robot() = default;

Robot::Robot(Robot&&) noexcept            = default;
Robot& Robot::operator=(Robot&&) noexcept = default;

core::Result<void> Robot::update(double dt)
{
  return impl_->update(dt);
}

core::Result<void> Robot::set_actuator(const core::kinematics::AckermannCommand& cmd, double dt)
{
  return impl_->set_actuator(cmd, dt);
}

core::Result<core::rl::Observation>
Robot::build_observation(double tick_time, double tick_dt, double reward, bool is_terminal)
{
  return impl_->build_observation(tick_time, tick_dt, reward, is_terminal);
}

core::Result<void> Robot::set_action(const core::rl::Action& action)
{
  return impl_->set_action(action);
}

const core::math::Pose2& Robot::get_pose() const
{
  return impl_->get_pose();
}

const core::math::Twist2& Robot::get_velocity() const
{
  return impl_->get_velocity();
}

const core::kinematics::AckermannCommand& Robot::last_applied_command() const
{
  return impl_->last_applied_command();
}

const core::kinematics::AckermannCommand& Robot::current_command() const
{
  return impl_->current_command();
}

const core::kinematics::AckermannCommand& Robot::requested_command() const
{
  return impl_->requested_command();
}

const core::rl::Action& Robot::requested_action() const
{
  return impl_->requested_action();
}

bool Robot::reverse_gear() const
{
  return impl_->reverse_gear();
}

double Robot::gear_hold_time_remaining() const
{
  return impl_->gear_hold_time_remaining();
}

const StepUpdateInfo& Robot::last_update_info() const
{
  return impl_->last_update_info();
}

StepUpdateInfo& Robot::last_update_info()
{
  return impl_->last_update_info();
}

uint32_t Robot::sim_id() const
{
  return impl_->sim_id();
}

core::Result<void>
Robot::notify_collision(std::span<const CollisionContact> contacts, double timestamp)
{
  return impl_->notify_collision(contacts, timestamp);
}

void Robot::clear_collision_state()
{
  impl_->clear_collision_state();
}

const config::ModelProfile& Robot::get_model_profile() const
{
  return impl_->get_model_profile();
}

} // namespace ddrl::sim
