#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ddrl::config {

/// Spawn manager configuration for ego-centric dynamic object spawning
struct SpawnManagerConfig {
  struct AngularBias {
    double front_angle_deg{120.0}; // width of frontal sector
    double flank_angle_deg{60.0};  // width for each flank sector
    double rear_angle_deg{120.0};  // width of rear sector
    double front_weight{3.0};      // sampling weight for frontal sector
    double flank_weight{1.0};      // sampling weight for each flank
    double rear_weight{0.2};       // sampling weight for rear sector
  };

  struct DistanceBias {
    double near_distance{25.0};   // upper bound of near band
    double far_distance{55.0};    // lower bound of far band
    double near_weight_slow{1.2}; // near band weight when robot is slow
    double near_weight_fast{0.4}; // near band weight when robot is fast
    double far_weight_slow{0.6};  // far band weight when robot is slow
    double far_weight_fast{1.4};  // far band weight when robot is fast
    double slow_speed_mps{2.0};   // speed breakpoint for "slow"
    double fast_speed_mps{10.0};  // speed breakpoint for "fast"
  };

  struct OcclusionSettings {
    bool   allow_in_fov{false};         // allow spawns inside FOV if occluded
    double static_blocker_radius{3.0};  // radius used for static occluders
    double dynamic_blocker_radius{2.0}; // radius used for dynamic occluders
  };

  // Spawning distances
  double min_spawn_distance{30.0}; // meters from robot
  double max_spawn_distance{60.0};
  double min_inter_object_distance{8.0}; // minimum separation between objects

  // Perception envelope (derived by simulator from robot sensors)
  double fov_angle_deg{120.0};
  double fov_distance{60.0};
  double max_sensor_range{80.0};

  // Despawn triggers
  double max_despawn_distance{80.0}; // Despawn if too far
  double stuck_timeout{5.0};         // Despawn if stuck for N seconds

  // Field of view avoidance
  bool              avoid_fov{true};
  AngularBias       angular_bias{};
  DistanceBias      distance_bias{};
  OcclusionSettings occlusion{};

  // Map geometry (if enabled, check against static obstacles)
  bool respect_map_geometry{false};

  // Per-frame performance limits
  size_t max_spawns_per_frame{3};
  size_t max_spawn_attempts_per_frame{10};

  // Active object count management
  size_t min_active_objects{5};
  size_t max_active_objects{20};

  // Spawn attempt parameters
  size_t max_position_samples{50}; // Per spawn attempt
};

struct PhysicsConfig {
  double fixed_dt{0.005};          // 200 Hz internal physics
  size_t max_substeps_per_step{8}; // clamp per call to step(dt)
};

struct DecisionConfig {
  double decision_dt{0.05}; // 20 Hz “control/observation” tick
  // Action application between decision ticks:
  double action_delay_s{0.0};    // comms/actuator latency
  double first_order_tau_s{0.0}; // only used if FirstOrderLag
};

struct EpisodeConfig {
  size_t max_decision_steps{10000}; // 0 = no limit
  double max_sim_seconds{0.0};      // 0 = no limit
  double collision_timeout_s{3.0};  // 0 = disabled; time allowed in collision before termination
};

/**
 * @brief Simulator configuration
 */
struct SimConfig {
  PhysicsConfig      physics{};
  DecisionConfig     decision{};
  EpisodeConfig      episode{};
  SpawnManagerConfig spawn_manager{};

  struct CurriculumPhase {
    std::string              name;
    std::vector<std::string> maps;
    std::optional<double>    promote_on_success_rate{0.8};
    size_t                   success_rate_window{20};

    struct SpawnOverrides {
      std::optional<double> min_spawn_distance;
      std::optional<double> max_spawn_distance;
      std::optional<size_t> min_active_objects;
      std::optional<size_t> max_active_objects;
    } spawn;
  };

  struct Curriculum {
    bool                         enabled{false};
    std::vector<CurriculumPhase> phases;
  } curriculum;

  uint64_t rng_seed{1234567}; // reproducibility for resets/noise
};

} // namespace ddrl::config
