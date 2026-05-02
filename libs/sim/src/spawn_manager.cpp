#include "sim/spawn_manager.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "logging/logging.hpp"

namespace ddrl::sim {

using core::math::Point2;
using core::math::Pose2;

class SpawnManager::Impl
{
private:
  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("sim.spawn_manager");
  std::shared_ptr<const config::SpawnManagerConfig> config_;
  std::shared_ptr<const config::MapConfig>          map_;

  using QueueContainer = std::deque<QueuedObject>;

  std::span<DynamicObject>                objects_;
  std::array<QueueContainer, 3>           priority_queues_{};
  std::unordered_set<std::size_t>         active_objects_;
  std::unordered_map<std::size_t, double> stuck_since_;
  std::unordered_map<std::size_t, double> reroute_next_attempt_;

  // Statistics
  size_t total_spawns_{0};
  size_t total_despawns_{0};
  size_t failed_spawn_attempts_{0};
  double queue_clock_{0.0};

  // Curriculum learning state
  double current_min_spawn_dist_{0.0};
  double current_max_spawn_dist_{0.0};
  Pose2  last_robot_pose_{};
  bool   has_last_robot_pose_{false};
  double robot_speed_mps_{0.0};

  std::mt19937 rng_;

  struct SectorSample {
    double center;
    double half_angle;
    double weight;
  };

  struct DistanceBand {
    double min;
    double max;
    double weight;
  };

  struct SpawnBiasCache {
    std::vector<SectorSample>                              sectors;
    std::vector<DistanceBand>                              bands;
    std::optional<std::discrete_distribution<std::size_t>> sector_dist;
    std::optional<std::discrete_distribution<std::size_t>> band_dist;
  };

  SpawnBiasCache bias_cache_;

  static constexpr std::size_t priority_index(SpawnPriority priority)
  {
    return static_cast<std::size_t>(priority);
  }

public:
  explicit Impl(Params&& params)
      : config_(std::move(params.config)),
        map_(params.map),
        objects_(params.objects),
        rng_(static_cast<std::mt19937::result_type>(params.rng_seed))
  {
    if (!config_) {
      throw std::runtime_error("SpawnManager requires spawn configuration");
    }
    if (!map_) {
      throw std::runtime_error("SpawnManager requires map configuration");
    }

    current_min_spawn_dist_ = config_->min_spawn_distance;
    current_max_spawn_dist_ = config_->max_spawn_distance;

    for (auto& queue : priority_queues_) {
      queue.clear();
    }
    active_objects_.clear();
    stuck_since_.clear();
    reroute_next_attempt_.clear();
    total_spawns_          = 0;
    total_despawns_        = 0;
    failed_spawn_attempts_ = 0;
    queue_clock_           = 0.0;

    // Add all objects to queue with LOW priority (never spawned)
    for (std::size_t i = 0; i < objects_.size(); ++i) {
      enqueue_object(i, SpawnPriority::LOW, DespawnReason::Type::INITIAL);

      // Move inactive objects far outside map bounds to prevent sensor detection
      move_object_out_of_bounds(i);
    }

    DDRL_LOG_DEBUG(logger_, "SpawnManager initialized with {} objects in queue", total_queued());
  }

  Stats get_stats() const
  {
    Stats stats;
    stats.active_count           = active_objects_.size();
    stats.queued_count           = total_queued();
    stats.total_spawns           = total_spawns_;
    stats.total_despawns         = total_despawns_;
    stats.failed_spawn_attempts  = failed_spawn_attempts_;
    stats.current_min_spawn_dist = current_min_spawn_dist_;
    stats.current_max_spawn_dist = current_max_spawn_dist_;
    return stats;
  }

  void update(double dt, const Pose2& robot_pose, std::size_t episode_count)
  {
    if (objects_.empty()) {
      return;
    }

    update_robot_motion(dt, robot_pose);

    // Update dynamic spawn distances (curriculum handled by simulator overrides)
    (void)episode_count;
    update_curriculum_distances(0);

    queue_clock_ += dt;

    // 1. Check and despawn objects that meet despawn conditions
    check_and_despawn_objects(robot_pose);

    // Refresh spawn bias cache based on latest robot pose/settings
    prepare_spawn_bias(robot_pose);

    // 2. Try spawning from queue (respecting per-frame limits)
    spawn_from_queue(
      robot_pose, config_->max_spawns_per_frame, config_->max_spawn_attempts_per_frame
    );

    // 3. Ensure minimum active count if possible
    ensure_minimum_active(robot_pose);
  }

private:
  void update_robot_motion(double dt, const Pose2& robot_pose)
  {
    if (has_last_robot_pose_ && dt > 1e-4) {
      const double dist = core::math::distance(robot_pose.pos, last_robot_pose_.pos);
      robot_speed_mps_  = dist / dt;
    } else {
      robot_speed_mps_ = 0.0;
    }
    last_robot_pose_     = robot_pose;
    has_last_robot_pose_ = true;
  }

  void update_curriculum_distances(std::size_t /*episode_count*/)
  {
    current_min_spawn_dist_ = config_->min_spawn_distance;
    current_max_spawn_dist_ = config_->max_spawn_distance;
  }

  bool is_visible_to_robot(const Pose2& robot_pose, const Pose2& obj_pose) const
  {
    const bool in_fov   = is_in_fov(robot_pose, obj_pose.pos);
    const bool occluded = is_occluded(robot_pose, obj_pose.pos);
    return in_fov && !occluded;
  }

  bool try_reroute_object(DynamicObject& obj, const Pose2& robot_pose)
  {
    const auto   obj_pose = obj.get_pose();
    const double rel_x    = obj_pose.pos.x - robot_pose.pos.x;
    const double rel_y    = obj_pose.pos.y - robot_pose.pos.y;
    const double rel_n    = std::hypot(rel_x, rel_y);

    core::math::Point2 side_dir{-rel_y, rel_x};
    const double       side_n = std::hypot(side_dir.x, side_dir.y);
    if (side_n < 1e-4 || rel_n < 1e-4) {
      return false;
    }
    side_dir.x /= side_n;
    side_dir.y /= side_n;

    core::math::Point2 forward_dir{rel_x / rel_n, rel_y / rel_n};

    constexpr double kLateralOffset = 2.5; // meters
    constexpr double kForwardOffset = 5.0; // meters

    // Try both lateral directions to improve the odds of finding a valid bypass
    for (double lateral_sign : std::array<double, 2>{1.0, -1.0}) {
      core::math::Point2 wp1{
        obj_pose.pos.x + lateral_sign * side_dir.x * kLateralOffset,
        obj_pose.pos.y + lateral_sign * side_dir.y * kLateralOffset
      };
      core::math::Point2 wp2{
        wp1.x + forward_dir.x * kForwardOffset, wp1.y + forward_dir.y * kForwardOffset
      };

      if (!map_->bounds.contains(wp1) || !map_->bounds.contains(wp2)) {
        continue;
      }

      if (check_static_collision(wp1) || check_static_collision(wp2)) {
        continue;
      }

      if (!check_separation(wp1) || !check_separation(wp2)) {
        continue;
      }

      obj.override_waypoints({wp1, wp2}, false);
      return true;
    }

    return false;
  }

  void check_and_despawn_objects(const Pose2& robot_pose)
  {
    std::vector<std::pair<std::size_t, DespawnReason::Type>> to_despawn;
    const bool trace_enabled = logger_->should_log(logging::LogLevel::TRACE);

    // Conservative estimate of robot's maximum sensor range from robot profile
    const double k_max_sensor_range = std::max(1.0, config_->max_sensor_range);

    const auto robot_forward =
      core::math::Point2{std::cos(robot_pose.yaw), std::sin(robot_pose.yaw)};

    for (auto idx : active_objects_) {
      if (idx >= objects_.size()) {
        continue;
      }
      auto& obj = objects_[idx];

      const auto obj_pose = obj.get_pose();
      const bool visible  = is_visible_to_robot(robot_pose, obj_pose);

      if (!obj.is_active() && !visible) {
        to_despawn.emplace_back(idx, DespawnReason::Type::MANUAL);
        stuck_since_.erase(idx);
        continue;
      }

      // Only despawn if too far (to save resources) or stuck (to respawn elsewhere)
      // Objects near the robot are KEPT so it can learn to avoid them!
      const double dist = core::math::distance(obj_pose.pos, robot_pose.pos);
      if (dist > config_->max_despawn_distance && !visible) {
        to_despawn.emplace_back(idx, DespawnReason::Type::TOO_FAR);
        if (trace_enabled) {
          DDRL_LOG_TRACE(
            logger_,
            "Object {} too far ({:.2f}m > {:.2f}m)",
            idx,
            dist,
            config_->max_despawn_distance
          );
        }
        continue;
      }

      // const double min_pass_dist = std::max(0.0, config_->min_distance_from_robot);
      // if (min_pass_dist > 1e-3 && dist <= min_pass_dist) {
      //   const auto rel =
      //     core::math::Point2{obj_pose.pos.x - robot_pose.pos.x, obj_pose.pos.y -
      //     robot_pose.pos.y};
      //   const double along_track = rel.x * robot_forward.x + rel.y * robot_forward.y;
      //   if (along_track < 0.0) {
      //     to_despawn.emplace_back(idx, DespawnReason::Type::TOO_CLOSE);
      //     if (trace_enabled) {
      //       DDRL_LOG_TRACE(
      //         logger_,
      //         "Object {} passed by robot (dist={:.2f}m, along_track={:.2f})",
      //         idx,
      //         dist,q
      //         along_track
      //       );
      //     }
      //     continue;
      //   }
      // }

      const bool is_stuck = obj.is_stuck();
      if (!is_stuck) {
        stuck_since_.erase(idx);
        reroute_next_attempt_.erase(idx);
        continue;
      }

      auto [stuck_entry, inserted] = stuck_since_.try_emplace(idx, queue_clock_);
      if (inserted) {
        stuck_entry->second = queue_clock_;
      }
      const auto reroute_entry = reroute_next_attempt_.find(idx);
      const bool reroute_ready =
        reroute_entry == reroute_next_attempt_.end() || queue_clock_ >= reroute_entry->second;
      const double stuck_duration = std::max(0.0, queue_clock_ - stuck_entry->second);
      const double timeout        = std::max(0.0, config_->stuck_timeout);
      const bool   timed_out      = timeout > 0.0 && stuck_duration >= timeout;

      constexpr double kRerouteStartDelay = 0.35; // let collision settle before rerouting
      if (stuck_duration >= kRerouteStartDelay && reroute_ready && visible && dist <= 30.0) {
        if (try_reroute_object(obj, robot_pose)) {
          reroute_next_attempt_.erase(idx);
          continue;
        }
        constexpr double kRerouteRetryInterval = 1.0; // seconds between reroute attempts
        reroute_next_attempt_[idx]             = queue_clock_ + kRerouteRetryInterval;
      }
      // Only despawn stuck objects if they're OUTSIDE sensor range
      // This prevents objects from "teleporting" while visible in LiDAR/cameras
      if (dist > k_max_sensor_range || (timed_out && !visible)) {
        to_despawn.emplace_back(idx, DespawnReason::Type::STUCK);
        if (trace_enabled) {
          DDRL_LOG_TRACE(
            logger_, "Object {} is stuck (duration={:.2f}s, dist={:.2f})", idx, stuck_duration, dist
          );
        }
      }
    }

    // Despawn objects with their respective reasons
    for (const auto& [idx, reason] : to_despawn) {
      despawn_object_internal(idx, reason);
    }
  }

  void prepare_spawn_bias(const Pose2& robot_pose)
  {
    (void)robot_pose; // currently unused but kept for future extensions

    bias_cache_.sectors = build_sector_bias();
    bias_cache_.bands   = build_distance_bands();

    bias_cache_.sector_dist.reset();
    if (!bias_cache_.sectors.empty()) {
      std::vector<double> weights;
      weights.reserve(bias_cache_.sectors.size());
      for (const auto& sector : bias_cache_.sectors) {
        weights.push_back(std::max(0.0, sector.weight));
      }
      if (auto dist = build_distribution(weights)) {
        bias_cache_.sector_dist = dist;
      }
    }

    bias_cache_.band_dist.reset();
    if (!bias_cache_.bands.empty()) {
      std::vector<double> weights;
      weights.reserve(bias_cache_.bands.size());
      for (const auto& band : bias_cache_.bands) {
        weights.push_back(std::max(0.0, band.weight));
      }
      if (auto dist = build_distribution(weights)) {
        bias_cache_.band_dist = dist;
      }
    }
  }

  void spawn_from_queue(const Pose2& robot_pose, std::size_t max_spawns, std::size_t max_attempts)
  {
    if (max_spawns == 0 || max_attempts == 0) {
      return;
    }

    const bool trace_enabled = logger_->should_log(logging::LogLevel::TRACE);
    const bool debug_enabled = logger_->should_log(logging::LogLevel::DEBUG);
    if (active_objects_.size() >= config_->max_active_objects || total_queued() == 0) {
      return;
    }

    if (!bias_cache_.band_dist) {
      return;
    }

    std::size_t spawns_this_frame   = 0;
    std::size_t attempts_this_frame = 0;

    auto try_priority_queue = [&](QueueContainer& queue) {
      while (!queue.empty() && spawns_this_frame < max_spawns &&
             attempts_this_frame < max_attempts &&
             active_objects_.size() < config_->max_active_objects) {
        auto entry = queue.front();
        queue.pop_front();
        attempts_this_frame++;

        bool spawned = false;
        if (entry.object_index < objects_.size()) {
          if (auto pose = find_spawn_pose(robot_pose)) {
            auto& obj = objects_[entry.object_index];
            obj.teleport(*pose);
            obj.activate();
            active_objects_.insert(entry.object_index);
            total_spawns_++;
            spawns_this_frame++;
            spawned = true;

            if (trace_enabled) {
              DDRL_LOG_TRACE(
                logger_,
                "Spawned object {} at [{:.2f}, {:.2f}] (priority={}, queued={:.2f}s)",
                entry.object_index,
                pose->pos.x,
                pose->pos.y,
                static_cast<int>(entry.priority),
                queued_age(entry)
              );
            }
          } else {
            failed_spawn_attempts_++;
          }
        }

        if (!spawned) {
          queue.push_back(entry);
        }
      }
    };

    for (SpawnPriority priority :
         {SpawnPriority::HIGH, SpawnPriority::NORMAL, SpawnPriority::LOW}) {
      try_priority_queue(priority_queues_[priority_index(priority)]);
      if (spawns_this_frame >= max_spawns || attempts_this_frame >= max_attempts ||
          active_objects_.size() >= config_->max_active_objects) {
        break;
      }
    }

    if (spawns_this_frame > 0 && debug_enabled) {
      DDRL_LOG_DEBUG(
        logger_,
        "Spawned {} objects this frame ({} attempts, {} active, {} queued)",
        spawns_this_frame,
        attempts_this_frame,
        active_objects_.size(),
        total_queued()
      );
    }
  }

  void ensure_minimum_active(const Pose2& robot_pose)
  {
    if (active_objects_.size() >= config_->min_active_objects) {
      return;
    }

    if (total_queued() == 0) {
      return;
    }

    const std::size_t needed = config_->min_active_objects - active_objects_.size();
    if (needed == 0) {
      return;
    }

    const std::size_t base_attempts =
      std::max<std::size_t>(1, config_->max_spawn_attempts_per_frame);
    const std::size_t extra_attempts = std::max<std::size_t>(needed * 20, base_attempts);
    spawn_from_queue(robot_pose, needed, extra_attempts);
  }

  std::optional<Pose2> find_spawn_pose(const Pose2& robot_pose)
  {
    if (!bias_cache_.band_dist || bias_cache_.bands.empty()) {
      return std::nullopt;
    }

    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    std::uniform_real_distribution<double> fallback_angle(-std::numbers::pi, std::numbers::pi);

    for (std::size_t attempt = 0; attempt < config_->max_position_samples; ++attempt) {
      double relative_angle = fallback_angle(rng_);
      if (bias_cache_.sector_dist && !bias_cache_.sectors.empty()) {
        const auto                             sector_idx = (*bias_cache_.sector_dist)(rng_);
        const auto&                            sector     = bias_cache_.sectors[sector_idx];
        std::uniform_real_distribution<double> offset(-sector.half_angle, sector.half_angle);
        relative_angle = sector.center + offset(rng_);
      }

      const auto   band_idx = (*bias_cache_.band_dist)(rng_);
      const auto&  band     = bias_cache_.bands[band_idx];
      const double radius   = band.min + (band.max - band.min) * unit01(rng_);

      Pose2        candidate;
      const double world_angle = core::math::normalize_angle(robot_pose.yaw + relative_angle);
      candidate.pos.x          = robot_pose.pos.x + radius * std::cos(world_angle);
      candidate.pos.y          = robot_pose.pos.y + radius * std::sin(world_angle);

      // Check map bounds
      if (!map_->bounds.contains(candidate.pos)) {
        continue;
      }

      // Check collision with static obstacles
      if (check_static_collision(candidate.pos)) {
        continue;
      }

      const bool candidate_in_fov = config_->avoid_fov && is_in_fov(robot_pose, candidate.pos);
      if (candidate_in_fov) {
        const bool occluded = is_occluded(robot_pose, candidate.pos);
        if (!(config_->occlusion.allow_in_fov && occluded)) {
          continue;
        }
      }

      // Check minimum separation from other active objects
      if (!check_separation(candidate.pos)) {
        continue;
      }

      // Orient spawned object toward the robot so it faces the ego vehicle immediately
      candidate.yaw =
        std::atan2(robot_pose.pos.y - candidate.pos.y, robot_pose.pos.x - candidate.pos.x);
      return candidate;
    }

    return std::nullopt;
  }

  std::vector<SectorSample> build_sector_bias() const
  {
    std::vector<SectorSample> sectors;
    const auto&               bias = config_->angular_bias;

    const auto clamp_angle = [](double deg) { return std::clamp(deg, 0.0, 360.0); };
    const auto deg_to_rad  = [](double deg) { return deg * std::numbers::pi / 180.0; };

    const double front_half = deg_to_rad(clamp_angle(bias.front_angle_deg)) * 0.5;
    const double flank_half = deg_to_rad(clamp_angle(bias.flank_angle_deg)) * 0.5;
    const double rear_half  = deg_to_rad(clamp_angle(bias.rear_angle_deg)) * 0.5;

    const double left_center  = front_half + flank_half;
    const double right_center = -(front_half + flank_half);

    const auto push_sector = [&](double center, double half_angle, double weight) {
      if (half_angle <= 0.0 || weight <= 0.0) {
        return;
      }
      sectors.push_back(SectorSample{center, half_angle, weight});
    };

    push_sector(0.0, front_half, bias.front_weight);
    push_sector(left_center, flank_half, bias.flank_weight);
    push_sector(right_center, flank_half, bias.flank_weight);
    push_sector(std::numbers::pi, rear_half, bias.rear_weight);

    return sectors;
  }

  double blend_weight(double slow_weight, double fast_weight) const
  {
    const auto&  dist_cfg = config_->distance_bias;
    const double slow     = dist_cfg.slow_speed_mps;
    const double fast     = std::max(dist_cfg.fast_speed_mps, slow + 1.0);

    if (robot_speed_mps_ <= slow) {
      return slow_weight;
    }
    if (robot_speed_mps_ >= fast) {
      return fast_weight;
    }

    const double t = (robot_speed_mps_ - slow) / (fast - slow);
    return slow_weight + t * (fast_weight - slow_weight);
  }

  std::vector<DistanceBand> build_distance_bands() const
  {
    std::vector<DistanceBand> bands;
    const double              min_dist = current_min_spawn_dist_;
    const double              max_dist = current_max_spawn_dist_;
    if (max_dist <= min_dist) {
      return bands;
    }

    const auto&  dist_cfg = config_->distance_bias;
    const double near_max = std::clamp(dist_cfg.near_distance, min_dist, max_dist);
    const double far_min =
      std::max(std::clamp(dist_cfg.far_distance, min_dist, max_dist), near_max);

    const double near_weight = blend_weight(dist_cfg.near_weight_slow, dist_cfg.near_weight_fast);
    const double far_weight  = blend_weight(dist_cfg.far_weight_slow, dist_cfg.far_weight_fast);

    const auto add_band = [&](double min_val, double max_val, double weight) {
      if (max_val - min_val <= 1e-3 || weight <= 0.0) {
        return;
      }
      bands.push_back(DistanceBand{min_val, max_val, weight});
    };

    add_band(min_dist, near_max, near_weight);
    add_band(near_max, far_min, 1.0);
    add_band(far_min, max_dist, far_weight);

    if (bands.empty()) {
      bands.push_back(DistanceBand{min_dist, max_dist, 1.0});
    }

    return bands;
  }

  static std::optional<std::discrete_distribution<std::size_t>>
  build_distribution(const std::vector<double>& weights)
  {
    if (weights.empty()) {
      return std::nullopt;
    }

    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0.0) {
      return std::nullopt;
    }

    return std::discrete_distribution<std::size_t>(weights.begin(), weights.end());
  }

  bool is_occluded(const Pose2& robot_pose, const Point2& candidate_pos) const
  {
    if (!config_->occlusion.allow_in_fov) {
      return false;
    }

    const double radius_static = std::max(0.0, config_->occlusion.static_blocker_radius);
    const double radius_dyn    = std::max(0.0, config_->occlusion.dynamic_blocker_radius);
    if (radius_static <= 0.0 && radius_dyn <= 0.0) {
      return false;
    }

    if (radius_static > 0.0 && config_->respect_map_geometry) {
      for (const auto& geom : map_->static_instances) {
        if (core::math::segment_intersects_circle(
              robot_pose.pos, candidate_pos, geom.pose_world.pos, radius_static
            )) {
          return true;
        }
      }
    }

    if (radius_dyn > 0.0) {
      for (auto idx : active_objects_) {
        if (idx >= objects_.size()) {
          continue;
        }
        const auto& obj = objects_[idx];
        if (!obj.is_active()) {
          continue;
        }
        if (core::math::segment_intersects_circle(
              robot_pose.pos, candidate_pos, obj.get_pose().pos, radius_dyn
            )) {
          return true;
        }
      }
    }

    return false;
  }

  bool is_in_fov(const Pose2& robot_pose, const Point2& point) const
  {
    const double dx   = point.x - robot_pose.pos.x;
    const double dy   = point.y - robot_pose.pos.y;
    const double dist = std::sqrt(dx * dx + dy * dy);

    // Only check FOV if within FOV distance
    if (dist > config_->fov_distance) {
      return false;
    }

    // Calculate angle to point relative to robot heading
    const double angle_to_point = std::atan2(dy, dx);
    const double angle_diff =
      std::abs(core::math::normalize_angle(angle_to_point - robot_pose.yaw));

    const double half_fov = (config_->fov_angle_deg * std::numbers::pi / 180.0) * 0.5;
    return angle_diff < half_fov;
  }

  bool check_separation(const Point2& candidate_pos) const
  {
    return std::ranges::all_of(active_objects_, [&](auto idx) {
      if (idx >= objects_.size()) {
        return true;
      }

      const auto& obj = objects_[idx];
      if (!obj.is_active()) {
        return true;
      }

      const auto   obj_pos = obj.get_pose().pos;
      const double dist    = core::math::distance(candidate_pos, obj_pos);

      return dist >= config_->min_inter_object_distance;
    });
  }

  bool check_static_collision(const Point2& candidate_pos) const
  {
    if (!config_->respect_map_geometry) {
      return false;
    }

    // Check collision with all static objects using conservative radius-based clearance
    // This ensures objects don't spawn inside or too close to static obstacles
    constexpr double kMinSafetyRadius = 2.5; // 2.5 meters minimum clearance from static objects

    return std::ranges::any_of(map_->static_instances, [&](const auto& geom) {
      const auto   static_pos = geom.pose_world.pos;
      const double dist       = core::math::distance(candidate_pos, static_pos);

      return dist < kMinSafetyRadius; // Return true if TOO CLOSE (collision)
    });
  }

  /// Move an object far outside map bounds to prevent sensor detection
  void move_object_out_of_bounds(std::size_t object_index)
  {
    if (object_index >= objects_.size()) {
      return;
    }

    // Calculate a position far outside map bounds (10x map size away)
    const auto&  bounds       = map_->bounds;
    const double map_diagonal = std::sqrt(
      std::pow(bounds.max.x - bounds.min.x, 2) + std::pow(bounds.max.y - bounds.min.y, 2)
    );

    // Place object 10x the map diagonal away in a consistent direction
    const double offset = map_diagonal * 10.0;

    Pose2 out_of_bounds_pose;
    out_of_bounds_pose.pos.x = bounds.max.x + offset;
    out_of_bounds_pose.pos.y = bounds.max.y + offset;
    out_of_bounds_pose.yaw   = 0.0;

    objects_[object_index].teleport(out_of_bounds_pose);
  }

  void despawn_object_internal(std::size_t object_index, DespawnReason::Type reason)
  {
    if (object_index >= objects_.size()) {
      return;
    }

    auto it = active_objects_.find(object_index);
    if (it == active_objects_.end()) {
      return; // Not active
    }

    // Move object out of bounds before deactivating to prevent sensor detection
    move_object_out_of_bounds(object_index);

    // Deactivate object
    objects_[object_index].deactivate();
    active_objects_.erase(it);
    stuck_since_.erase(object_index);
    reroute_next_attempt_.erase(object_index);
    total_despawns_++;

    // Add back to queue with HIGH priority (recently despawned)
    enqueue_object(object_index, SpawnPriority::HIGH, reason);

    if (logger_->should_log(logging::LogLevel::TRACE)) {
      DDRL_LOG_TRACE(
        logger_,
        "Despawned object {} (reason={}, {} active, {} queued)",
        object_index,
        static_cast<int>(reason),
        active_objects_.size(),
        total_queued()
      );
    }
  }

  std::size_t total_queued() const
  {
    std::size_t total = 0;
    for (const auto& queue : priority_queues_) {
      total += queue.size();
    }
    return total;
  }

  double queued_age(const QueuedObject& obj) const
  {
    return std::max(0.0, queue_clock_ - obj.queued_since);
  }

  void enqueue_object(std::size_t object_index, SpawnPriority priority, DespawnReason::Type reason)
  {
    QueuedObject entry{
      .object_index        = object_index,
      .priority            = priority,
      .queued_since        = queue_clock_,
      .last_despawn_reason = {.type = reason, .timestamp = queue_clock_}
    };
    priority_queues_[priority_index(priority)].push_back(entry);
  }
};

// =============================================================================
// SpawnManager Public API
// =============================================================================

SpawnManager::SpawnManager(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

SpawnManager::~SpawnManager() = default;

void SpawnManager::update(double dt, const core::math::Pose2& robot_pose, std::size_t episode_count)
{
  impl_->update(dt, robot_pose, episode_count);
}

SpawnManager::Stats SpawnManager::get_stats() const
{
  return impl_->get_stats();
}

} // namespace ddrl::sim
