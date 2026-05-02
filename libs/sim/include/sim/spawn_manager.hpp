#pragma once

#include "core/geom/aabb.hpp"
#include <cstddef>
#include <memory>
#include <span>

#include "config/map.hpp"
#include "config/simulator.hpp"
#include "sim/object.hpp"

namespace ddrl::sim {

/// Priority level for objects in spawn queue
enum class SpawnPriority : std::uint8_t {
  HIGH,   // Recently despawned (keep encounter rate high)
  NORMAL, // Regular queue
  LOW     // Never spawned yet
};

/// Tracks why an object was despawned (for debugging/metrics)
struct DespawnReason {
  enum class Type : std::uint8_t {
    COLLISION,
    TOO_FAR,
    TOO_CLOSE,
    STUCK,
    MANUAL,
    INITIAL // First-time spawn
  };

  Type   type{Type::INITIAL};
  double timestamp{0.0};
};

/// SpawnManager: Handles ego-centric spawning of dynamic objects around robot
class SpawnManager
{
public:
  struct Params {
    std::shared_ptr<const config::SpawnManagerConfig> config;
    std::span<DynamicObject>                          objects;
    std::shared_ptr<const config::MapConfig>          map;
    uint64_t                                          rng_seed{0};
  };
  explicit SpawnManager(Params params);
  ~SpawnManager();

  /// Update spawn manager: check despawns, attempt spawns from queue
  void update(double dt, const core::math::Pose2& robot_pose, std::size_t episode_count);

  /// Get current spawn statistics
  struct Stats {
    std::size_t active_count{0};
    std::size_t queued_count{0};
    std::size_t total_spawns{0};
    std::size_t total_despawns{0};
    std::size_t failed_spawn_attempts{0};
    double      current_min_spawn_dist{0.0};
    double      current_max_spawn_dist{0.0};
  };
  [[nodiscard]] Stats get_stats() const;

private:
  struct QueuedObject {
    std::size_t   object_index;
    SpawnPriority priority;
    double        queued_since;
    DespawnReason last_despawn_reason;
  };

  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim
