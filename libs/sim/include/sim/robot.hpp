#pragma once
#include <collision/bvh.hpp>
#include <core/base/result.hpp>
#include <core/kinematics/ackermann.hpp>
#include <core/math/pose.hpp>
#include <core/math/twist.hpp>
#include <core/rl/rl.hpp>
#include <core/sim/collision.hpp>
#include <core/transform/tf.hpp>
#include <core/vision/point_cloud.hpp>
#include <kinematics/ackermann.hpp>
#include <memory>
#include <threading/thread_pool.hpp>
#include <vector>

#include "config/robot_profile.hpp"
#include "config/rl.hpp"
#include "config/world.hpp"
#include "sensors.hpp"

namespace ddrl::sim {

/// Robot with Ackermann steering kinematics

struct StepUpdateInfo {
  core::math::Pose2  est_pose;
  core::math::Twist2 est_vel;
  bool               is_collided = false;

  const std::vector<SensorType>* sensors{nullptr};
};

class Robot
{
public:
  struct Params {
    std::shared_ptr<const config::RobotInstanceCfg> config;
    std::shared_ptr<const config::RobotProfile>     profile;
    std::shared_ptr<threading::ThreadPool>          thread_pool;
    std::shared_ptr<const config::RLConfig>         rl_config;

    std::shared_ptr<collision::TLAS2Pair>       scene_trees;
    std::shared_ptr<core::tf::TfBuffer>         tf_buffer;
    std::shared_ptr<const config::ModelProfile> model_profile;

    std::string base_frame_id{"base_link"};
    uint32_t    entity_id{0}; ///< Collision entity ID
    uint32_t    sim_id{0};    ///< Simulator instance ID

    core::math::Pose2 initial_pose{}; ///< Initial pose in world frame
  };

  explicit Robot(Params params);
  ~Robot();

  Robot(const Robot&)            = delete;
  Robot& operator=(const Robot&) = delete;
  Robot(Robot&&) noexcept;
  Robot& operator=(Robot&&) noexcept;

  /// Update robot dynamics for one timestep
  core::Result<void> update(double dt);
  core::Result<void> set_actuator(const core::kinematics::AckermannCommand& cmd, double dt);

  /// Called at decision intervals to publish observation with reward
  /// @param reward Accumulated reward since last decision tick
  /// @param is_terminal Whether episode has ended
  core::Result<core::rl::Observation> build_observation(
    double tick_time, double tick_dt, double reward = 0.0, bool is_terminal = false
  );

  /// Set action from event bus (called by Worker via event bus)
  core::Result<void> set_action(const core::rl::Action& action);

  [[nodiscard]] const core::math::Pose2&  get_pose() const;
  [[nodiscard]] const core::math::Twist2& get_velocity() const;
  [[nodiscard]] const core::kinematics::AckermannCommand& last_applied_command() const;
  [[nodiscard]] const core::kinematics::AckermannCommand& current_command() const;
  [[nodiscard]] const core::kinematics::AckermannCommand& requested_command() const;
  [[nodiscard]] const core::rl::Action& requested_action() const;
  [[nodiscard]] bool reverse_gear() const;
  [[nodiscard]] double gear_hold_time_remaining() const;

  [[nodiscard]] const StepUpdateInfo& last_update_info() const;
  [[nodiscard]] StepUpdateInfo&       last_update_info();

  [[nodiscard]] uint32_t sim_id() const;
  [[nodiscard]] uint32_t get_id() const;

  [[nodiscard]] const config::ModelProfile& get_model_profile() const;

  core::Result<void>
       notify_collision(std::span<const core::sim::CollisionContact> contacts, double timestamp);
  void clear_collision_state();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim
