#pragma once
#include "core/base/result.hpp"
#include "core/math/pose.hpp"
#include "core/math/primitives.hpp"
#include "core/math/twist.hpp"
#include <memory>
#include <vector>

#include "config/dynamic_profile.hpp"
#include "config/world.hpp"

namespace ddrl::sim {

/// Static world geometry (buildings, walls, etc.)
class StaticObject
{
public:
  struct Params {
    uint32_t object_id{0};

    std::shared_ptr<const config::StaticInstanceCfg> config;
  };
  StaticObject(Params params);
  ~StaticObject();

  StaticObject(const StaticObject&)            = delete;
  StaticObject& operator=(const StaticObject&) = delete;
  StaticObject(StaticObject&&) noexcept;
  StaticObject& operator=(StaticObject&&) noexcept;

  /// Get object configuration
  [[nodiscard]] const config::StaticInstanceCfg& get_config() const;
  [[nodiscard]] uint32_t                         get_id() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Dynamic object behavior states (simplified FSM)
enum class DynamicObjectState : uint8_t {
  ACTIVE,  // Normal operation (moving along waypoints)
  STOPPED, // Temporarily stopped (collision or random pause)
  INACTIVE, // Deactivated (ready for despawn/respawn)
};

/// Dynamic object (vehicles, pedestrians, movable obstacles)
class DynamicObject
{
public:
  struct Params {
    uint32_t object_id{0};

    std::shared_ptr<const config::DynamicInstanceCfg> config;
    std::shared_ptr<const config::DynamicProfile>     profile;
  };
  DynamicObject(Params params);
  ~DynamicObject();

  DynamicObject(const DynamicObject&)            = delete;
  DynamicObject& operator=(const DynamicObject&) = delete;

  DynamicObject(DynamicObject&&) noexcept;
  DynamicObject& operator=(DynamicObject&&) noexcept;

  /// Update object motion for one timestep
  core::Result<void> update(double dt);

  /// Notify object of collision (may pause movement)
  void on_collision();

  /// Teleport object to new pose (called by SpawnManager)
  void teleport(const core::math::Pose2& new_pose);

  /// Deactivate object (ready for despawn)
  void deactivate();

  /// Reactivate object (after spawn/respawn)
  void activate();

  /// Override waypoint list in world coordinates and resume motion immediately
  void override_waypoints(std::vector<core::math::Point2> waypoints_world, bool loop);

  [[nodiscard]] const config::DynamicInstanceCfg& get_config() const;
  [[nodiscard]] uint32_t                          get_id() const;
  [[nodiscard]] const core::math::Pose2&          get_pose() const;
  [[nodiscard]] const core::math::Twist2&         get_velocity() const;
  [[nodiscard]] DynamicObjectState                get_state() const;

  /// Check if object is active
  [[nodiscard]] bool is_active() const;

  /// Check if object has been stuck in collision
  [[nodiscard]] bool is_stuck() const;

  /// Returns true if the object is currently stopped because of a collision event
  [[nodiscard]] bool was_collision_stopped() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim

std::string to_string(ddrl::sim::DynamicObjectState state);
