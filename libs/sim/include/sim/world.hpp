#pragma once
#include "core/base/result.hpp"
#include "core/rl/reward.hpp"
#include "core/rl/rl.hpp"
#include "core/sim/collision.hpp"
#include "core/sim/world_state.hpp"
#include <collision/broad_phase.hpp>
#include <core/kinematics/ackermann.hpp>
#include <core/math/pose.hpp>
#include <core/math/twist.hpp>
#include <events/event_bus.hpp>
#include <memory>
#include <optional>
#include <threading/thread_pool.hpp>

#include "config/map.hpp"
#include "config/rl.hpp"
#include "config/simulator.hpp"
#include "config/world.hpp"

namespace ddrl::sim {

/// World state containing all static/dynamic objects and robot
class World
{
public:
  struct Params {
    std::shared_ptr<const config::WorldConfig>      config;
    std::shared_ptr<const config::MapConfig>        map;
    std::shared_ptr<const config::SpawnManagerConfig> spawn_config;
    std::shared_ptr<threading::ThreadPool>          thread_pool;
    std::shared_ptr<const config::RLConfig>         rl_config;
    uint64_t                                        rng_seed{0};
    uint32_t                                        sim_id;
  };

  explicit World(Params params);
  ~World();

  /// Update all objects for some timesteps
  core::Result<void> update(double dt);

  /// Signal robot to run decision tick, i.e send observation
  /// @param reward Accumulated reward since last decision tick
  /// @param is_terminal Whether episode has ended
  core::Result<core::rl::Observation>
  on_decision_tick(double tick_time, double tick_dt, double reward = 0.0, bool is_terminal = false);

  /// Build initial observation without advancing simulation time
  core::Result<core::rl::Observation> build_initial_observation(double sim_time);

  /// Apply an action command to the robot (handled internally with actuator delay)
  core::Result<void> apply_action(const core::rl::Action& action);

  /// Fetch the latest world snapshot (rebuilding lazily when outdated)
  [[nodiscard]] std::shared_ptr<const core::sim::WorldStateSnapshot> capture_snapshot();

  /// Consume and clear the most recent collision report, if any
  [[nodiscard]] std::optional<core::sim::CollisionReport> consume_collision_report();

  /// Retrieve the latest telemetry needed for reward computation
  [[nodiscard]] core::rl::RewardTelemetry reward_telemetry();

  /// Get the map object
  [[nodiscard]] std::shared_ptr<const config::MapConfig> get_map() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim
