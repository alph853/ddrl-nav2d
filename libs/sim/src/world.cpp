#include "sim/world.hpp"

#include "core/base/variant_overload.hpp"
#include "core/geom/aabb.hpp"
#include "core/geom/shape.hpp"
#include "core/math/primitives.hpp"
#include "core/sim/collision.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "collision/broad_phase.hpp"
#include "collision/narrow_phase.hpp"
#include "logging/logging.hpp"
#include "sim/object.hpp"
#include "sim/robot.hpp"
#include "sim/spawn_manager.hpp"

namespace ddrl::sim {

using config::DeterministicWorldCfg;
using config::DynamicInstanceCfg;
using config::RandomWorldCfg;
using config::RobotProfile;
using config::StaticInstanceCfg;

using core::Error;
using core::ErrorCode;
using core::make_error;
using core::make_error_from;
using core::Result;
using core::geom::AABB2;
using core::math::Pose2;

using collision::BroadPhase;
using collision::CollisionResult;
using collision::TLAS2;
using collision::TLAS2Pair;

using core::sim::CollisionContact;
using core::sim::CollisionReport;
using core::sim::CollisionSeverity;
using core::sim::WorldStateSnapshot;

using core::rl::Observation;
using core::rl::RewardTelemetry;

namespace {

double compute_forward_clearance_scan_like(
  const std::vector<SensorType>* sensors, const core::math::Pose2& robot_pose, double half_angle_rad
)
{
  if (sensors == nullptr || sensors->empty()) {
    return std::numeric_limits<double>::infinity();
  }

  constexpr std::size_t kScanBins = 72;
  const double cos_yaw = std::cos(-robot_pose.yaw);
  const double sin_yaw = std::sin(-robot_pose.yaw);
  const double rx = robot_pose.pos.x;
  const double ry = robot_pose.pos.y;
  const double cone = std::clamp(half_angle_rad, 1e-3, std::numbers::pi);
  std::vector<double> scan_ranges(kScanBins, std::numeric_limits<double>::infinity());

  for (const auto& sensor : *sensors) {
    if (!sensor || !sensor->has_data()) {
      continue;
    }
    const auto cloud = sensor->get_point_cloud();
    if (!cloud) {
      continue;
    }
    for (const auto& pt : cloud->points) {
      const double dx = pt.x - rx;
      const double dy = pt.y - ry;
      const double ex = cos_yaw * dx - sin_yaw * dy;
      const double ey = sin_yaw * dx + cos_yaw * dy;
      const double range = std::hypot(ex, ey);
      if (range <= 1e-6) {
        continue;
      }
      const double azimuth = std::atan2(ey, ex);
      const double wrapped = (azimuth + std::numbers::pi) / (2.0 * std::numbers::pi);
      const auto bin = static_cast<std::size_t>(std::clamp(
        static_cast<long>(std::floor(wrapped * static_cast<double>(kScanBins))),
        0L,
        static_cast<long>(kScanBins - 1)
      ));
      scan_ranges[bin] = std::min(scan_ranges[bin], range);
    }
  }

  double min_forward = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < scan_ranges.size(); ++i) {
    const double center =
      -std::numbers::pi +
      (static_cast<double>(i) + 0.5) *
        (2.0 * std::numbers::pi / static_cast<double>(kScanBins));
    if (std::abs(center) > cone) {
      continue;
    }
    min_forward = std::min(min_forward, scan_ranges[i]);
  }
  return min_forward;
}

} // namespace

class World::Impl
{
  struct EntityRef {
    enum class Type : uint8_t { STATIC, DYNAMIC, ROBOT };
    Type        type;
    std::size_t index; // index into static_objects_ or dynamic_objects_ (unused for Robot)
    std::string name;  // for logging/debugging only
  };

private:
  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("sim.world");

  std::shared_ptr<const config::WorldConfig>        config_;
  std::shared_ptr<const config::MapConfig>          map_;
  std::shared_ptr<const config::SpawnManagerConfig> spawn_config_;
  std::shared_ptr<const config::RLConfig>           rl_config_;
  std::shared_ptr<threading::ThreadPool>            thread_pool_;
  uint64_t                                          rng_seed_{0};

  uint32_t sim_id_;
  double   sim_time_{0.0};
  double   last_update_dt_{0.0};

  std::shared_ptr<WorldStateSnapshot> last_world_snapshot_;
  std::shared_ptr<core::tf::TfBuffer> tf_buffer_;
  bool                                snapshot_dirty_{true};

  std::unique_ptr<BroadPhase> broad_phase_;
  std::shared_ptr<TLAS2Pair>  scene_trees_;

  // Unified entity registry stored densely by entity_id
  std::vector<std::optional<EntityRef>> entity_registry_;

  // Cached TLAS instance lookup keyed by entity id to avoid re-hashing every frame
  std::vector<const TLAS2::Instance*> instance_lookup_;
  std::vector<std::uint32_t>          instance_lookup_touched_;

  std::uint32_t next_entity_id_{0};
  std::uint32_t robot_entity_id_{std::numeric_limits<std::uint32_t>::max()};

  std::unordered_map<std::string, std::uint32_t> name_suffix_counts_;
  std::unordered_set<std::string>                used_names_;

  std::vector<StaticObject>  static_objects_;
  std::vector<DynamicObject> dynamic_objects_;
  std::optional<Robot>
    robot_; // Always has value after construction; using optional for deferred init

  std::optional<CollisionReport>                           pending_collision_report_;
  core::math::Pose2                                        robot_start_pose_;
  core::math::Pose2                                        active_route_start_pose_;
  struct ActiveWaypoint {
    std::string        name;
    core::math::Point2 pos;
    double             tolerance{1.0};
  };
  std::string                 active_route_name_;
  std::vector<ActiveWaypoint> active_route_;
  std::size_t                 active_waypoint_index_{0};
  bool                        final_goal_reached_{false};

  // Spawn management
  std::unique_ptr<SpawnManager> spawn_manager_;
  std::size_t                   episode_count_{0};

public:
  explicit Impl(Params&& params);
  ~Impl();

  core::Result<void> update(double dt);
  core::Result<void> reset();
  core::Result<Observation>
  on_decision_tick(double tick_time, double tick_dt, double reward, bool is_terminal);
  core::Result<Observation>                 build_initial_observation(double sim_time);
  core::Result<void>                        apply_action(const core::rl::Action& action);
  std::shared_ptr<const WorldStateSnapshot> capture_snapshot();
  std::optional<CollisionReport>            consume_collision_report();
  RewardTelemetry                           reward_telemetry();

  [[nodiscard]] std::shared_ptr<const config::MapConfig> get_map() const { return map_; }

private:
  Result<void> update_scene_trees();
  Result<void> run_collisions_detection();
  void         rebuild_world_snapshot();

  void                                 register_existing_name(const std::string& name);
  std::string                          make_instance_name(const std::string& desired);
  void                                 ensure_entity_slot(std::uint32_t entity_id);
  void                                 register_entity(std::uint32_t entity_id, EntityRef entity);
  [[nodiscard]] const EntityRef*       find_entity(std::uint32_t entity_id) const;
  void                                 reset_instance_cache();
  void                                 cache_instance(const TLAS2::Instance& instance);
  [[nodiscard]] const TLAS2::Instance* instance_for(std::uint32_t entity_id) const;

  // =============================================================================
  // Collision Utilities
  // =============================================================================

  [[nodiscard]] static core::geom::AABB2
  model_bounds(const config::ModelProfile& model, const Pose2& world_pose);

  [[nodiscard]] static CollisionResult narrow_phase_intersect_check(
    const config::ModelProfile& a_model, const Pose2& a_pose, const config::ModelProfile& b_model,
    const Pose2& b_pose
  );

  core::Result<void> resolve_robot_collisions();
  core::Result<void> resolve_dynamic_object_collisions();

  [[nodiscard]] static CollisionSeverity compute_collision_severity(
    const std::vector<CollisionContact>& contacts,
    const std::vector<CollisionResult>& collision_results, const core::math::Twist2& robot_velocity,
    const core::math::Pose2& robot_pose
  );

  void select_route(std::mt19937& rng, bool randomize);
  [[nodiscard]] const ActiveWaypoint& current_waypoint() const;
  [[nodiscard]] bool                  is_current_waypoint_reached(
                     const core::math::Point2& point
                   ) const;
  void populate_route_observation(core::rl::Observation& obs) const;

};

World::Impl::Impl(Params&& params)
    : config_(std::move(params.config)),
      map_(std::move(params.map)),
      spawn_config_(std::move(params.spawn_config)),
      rl_config_(std::move(params.rl_config)),
      thread_pool_(std::move(params.thread_pool)),
      rng_seed_(params.rng_seed),
      sim_id_(params.sim_id),
      tf_buffer_(std::make_shared<core::tf::TfBuffer>())
{
  DDRL_LOG_DEBUG(logger_, "Initializing world for sim instance {}", sim_id_);

  if (!config_ || !map_ || !thread_pool_ || !spawn_config_) {
    throw std::runtime_error("World requires config, map, spawn config, and thread pool");
  }

  DDRL_LOG_DEBUG(logger_, "Loading map: '{}' ({})", map_->name, map_->description);

  tf_buffer_->register_frame("", "world");

  // =============================================================================
  // Load static objects from map
  // =============================================================================

  for (const auto& static_cfg : map_->static_instances) {
    static_objects_.emplace_back(StaticObject::Params{
      .object_id = next_entity_id_++,
      .config    = std::shared_ptr<const config::StaticInstanceCfg>(config_, &static_cfg)
    });
    DDRL_LOG_DEBUG(
      logger_,
      "Added static object '{}' from map '{}' (entity_id={})",
      static_objects_.back().get_config().name,
      map_->name,
      static_objects_.back().get_id()
    );

    register_entity(
      static_objects_.back().get_id(),
      EntityRef{
        .type  = EntityRef::Type::STATIC,
        .index = static_objects_.size() - 1,
        .name  = static_cfg.name
      }
    );

    register_existing_name(static_cfg.name);
  }

  std::mt19937 rng(static_cast<std::mt19937::result_type>(rng_seed_));

  const auto& robot_cfg = config_->robot;

  auto robot_prf_it = config_->robot_profiles.find(robot_cfg.robot_profile_id);
  if (robot_prf_it == config_->robot_profiles.end()) {
    throw std::runtime_error("Robot profile not found: " + robot_cfg.robot_profile_id);
  }
  const auto& robot_profile     = robot_prf_it->second;
  auto        robot_profile_ptr = std::shared_ptr<const RobotProfile>(config_, &robot_profile);

  auto robot_model_it = config_->model_profiles.find(robot_profile.model_id);
  if (robot_model_it == config_->model_profiles.end()) {
    throw std::runtime_error("Robot model profile not found: " + robot_profile.model_id);
  }
  const auto& robot_model = robot_model_it->second;
  Pose2       robot_initial_pose;

  std::visit(
    core::Overloaded{
      [&](const DeterministicWorldCfg& det_world) {
        select_route(rng, false);
        robot_initial_pose = active_route_start_pose_;

        for (const auto& dyn_cfg : det_world.dynamic_instances) {
          auto profile_it = config_->dynamic_profiles.find(dyn_cfg.dynamic_profile_id);
          if (profile_it == config_->dynamic_profiles.end()) {
            throw std::runtime_error("Dynamic profile not found: " + dyn_cfg.dynamic_profile_id);
          }

          dynamic_objects_.emplace_back(DynamicObject::Params{
            .object_id = next_entity_id_++,
            .config    = std::shared_ptr<const DynamicInstanceCfg>(config_, &dyn_cfg),
            .profile   = std::shared_ptr<const config::DynamicProfile>(config_, &profile_it->second)
          });
          DDRL_LOG_DEBUG(logger_, "Added dynamic object '{}'", dyn_cfg.name);

          register_entity(
            dynamic_objects_.back().get_id(),
            EntityRef{
              .type  = EntityRef::Type::DYNAMIC,
              .index = dynamic_objects_.size() - 1,
              .name  = dyn_cfg.name
            }
          );

          register_existing_name(dyn_cfg.name);
        }

        DDRL_LOG_DEBUG(
          logger_,
          "[Sim {}] Assigned deterministic route '{}' with {} waypoints",
          sim_id_,
          active_route_name_,
          active_route_.size()
        );

        DDRL_LOG_DEBUG(
          logger_,
          "Initialized deterministic world with {} dynamic objects",
          det_world.dynamic_instances.size()
        );
      },

      [&](const RandomWorldCfg& rand_world) {
        // Create dynamic objects (without initial spawning - SpawnManager handles that)
        for (const auto& rdi_cfg : rand_world.random_dynamic_instances) {
          // Look up the dynamic profile
          auto profile_it = config_->dynamic_profiles.find(rdi_cfg.dynamic_profile_id);
          if (profile_it == config_->dynamic_profiles.end()) {
            throw std::runtime_error("Dynamic profile not found: " + rdi_cfg.dynamic_profile_id);
          }

          // Create all objects for this group
          for (size_t i = 0; i < rdi_cfg.count; ++i) {
            // Create dynamic instance config with default pose (will be set by SpawnManager)
            DynamicInstanceCfg dyn_inst;
            dyn_inst.name     = make_instance_name(std::format("{}_{:04d}", rdi_cfg.name, i));
            dyn_inst.model_id = rdi_cfg.model_id;
            dyn_inst.dynamic_profile_id = rdi_cfg.dynamic_profile_id;
            dyn_inst.pose_world = Pose2{}; // Default pose, SpawnManager will set actual pose

            // Create object in INACTIVE state (SpawnManager will activate when ready)
            dynamic_objects_.emplace_back(DynamicObject::Params{
              .object_id = next_entity_id_++,
              .config    = std::make_shared<const DynamicInstanceCfg>(std::move(dyn_inst)),
              .profile = std::shared_ptr<const config::DynamicProfile>(config_, &profile_it->second)
            });

            register_entity(
              dynamic_objects_.back().get_id(),
              EntityRef{
                .type  = EntityRef::Type::DYNAMIC,
                .index = dynamic_objects_.size() - 1,
                .name  = dyn_inst.name
              }
            );

            register_existing_name(dyn_inst.name);
          }

          DDRL_LOG_DEBUG(
            logger_,
            "Created {} dynamic objects for '{}' (will be spawned by SpawnManager)",
            rdi_cfg.count,
            rdi_cfg.name
          );
        }
        select_route(rng, true);
        robot_initial_pose = active_route_start_pose_;
        DDRL_LOG_DEBUG(
          logger_,
          "[Sim {}] Selected route '{}' with {} waypoints and start [{:.2f}, {:.2f}, {:.2f}]",
          sim_id_,
          active_route_name_,
          active_route_.size(),
          robot_initial_pose.pos.x,
          robot_initial_pose.pos.y,
          robot_initial_pose.yaw
        );
      }
    },
    config_->world
  );

  robot_start_pose_ = robot_initial_pose;

  // Get robot model profile for geometry checking
  DDRL_LOG_DEBUG(logger_, "Building scene trees for sim_id={}", sim_id_);

  std::vector<BroadPhase::Instance> instances;
  instances.reserve(static_objects_.size() + dynamic_objects_.size() + 1);

  for (const auto& static_obj : static_objects_) {
    const auto& cfg      = static_obj.get_config();
    auto        model_it = config_->model_profiles.find(cfg.model_id);
    if (model_it == config_->model_profiles.end()) {
      DDRL_LOG_WARN(logger_, "Model profile not found for static object '{}'", cfg.model_id);
      continue;
    }
    if (model_it->second.name.starts_with("tile_")) {
      DDRL_LOG_DEBUG(
        logger_, "Skipping static object '{}' with tile model from broad phase", cfg.name
      );
      continue;
    }

    BroadPhase::Instance inst;
    inst.name          = cfg.name;
    inst.pose_world    = cfg.pose_world;
    inst.is_static     = true;
    inst.entity_id     = static_obj.get_id();
    inst.model_profile = std::shared_ptr<const config::ModelProfile>(config_, &model_it->second);
    instances.push_back(std::move(inst));

    DDRL_LOG_DEBUG(
      logger_,
      "Added static object '{}' to broad phase (entity_id={})",
      cfg.name,
      static_obj.get_id()
    );
  }

  for (const auto& dynamic_object : dynamic_objects_) {
    const auto& cfg      = dynamic_object.get_config();
    auto        model_it = config_->model_profiles.find(cfg.model_id);
    if (model_it == config_->model_profiles.end()) {
      DDRL_LOG_WARN(logger_, "Model profile not found for dynamic object '{}'", cfg.model_id);
      continue;
    }

    BroadPhase::Instance inst;
    inst.name          = cfg.name;
    inst.pose_world    = cfg.pose_world;
    inst.is_static     = false;
    inst.entity_id     = dynamic_object.get_id();
    inst.model_profile = std::shared_ptr<const config::ModelProfile>(config_, &model_it->second);
    instances.push_back(std::move(inst));
  }

  auto robot_model_profile = std::shared_ptr<const config::ModelProfile>(config_, &robot_model);
  robot_entity_id_         = next_entity_id_++;

  register_entity(
    robot_entity_id_, EntityRef{.type = EntityRef::Type::ROBOT, .index = 0, .name = robot_cfg.name}
  );

  BroadPhase::Instance robot_inst;
  robot_inst.name          = robot_cfg.name;
  robot_inst.pose_world    = robot_initial_pose;
  robot_inst.is_static     = false;
  robot_inst.entity_id     = robot_entity_id_;
  robot_inst.model_profile = robot_model_profile;
  instances.push_back(std::move(robot_inst));

  broad_phase_ = std::make_unique<BroadPhase>(BroadPhase::Params{
    .instances = std::move(instances), .collision_masks = config_->collision_categories
  });
  broad_phase_->build();

  scene_trees_ = std::shared_ptr<TLAS2Pair>(
    std::shared_ptr<TLAS2Pair>{}, const_cast<TLAS2Pair*>(&broad_phase_->trees())
  );

  robot_.emplace(Robot::Params{
    .config        = std::shared_ptr<const config::RobotInstanceCfg>(config_, &config_->robot),
    .profile       = robot_profile_ptr,
    .thread_pool   = thread_pool_,
    .rl_config     = rl_config_,
    .scene_trees   = scene_trees_,
    .tf_buffer     = tf_buffer_,
    .model_profile = robot_model_profile,
    .base_frame_id = "base_link",
    .entity_id     = robot_entity_id_,
    .sim_id        = sim_id_,
    .initial_pose  = robot_initial_pose,
  });

  last_world_snapshot_                    = std::make_shared<WorldStateSnapshot>();
  last_world_snapshot_->current_map       = map_->name;
  last_world_snapshot_->robot.model_id    = robot_model_profile->name;
  last_world_snapshot_->robot.pose        = robot_->get_pose();
  last_world_snapshot_->robot.velocity    = robot_->get_velocity();
  last_world_snapshot_->robot.is_collided = false;

  for (const auto& dynamic_object : dynamic_objects_) {
    const auto& cfg = dynamic_object.get_config();

    core::sim::ObjectState obj_state;
    obj_state.id         = dynamic_object.get_id();
    obj_state.name       = cfg.name;
    obj_state.model_id   = cfg.model_id;
    obj_state.pose       = dynamic_object.get_pose();
    obj_state.velocity   = dynamic_object.get_velocity();
    obj_state.state_name = to_string(dynamic_object.get_state());

    last_world_snapshot_->dynamic_objects.push_back(std::move(obj_state));
  }

  snapshot_dirty_ = false;

  // Initialize spawn manager from config
  spawn_manager_ = std::make_unique<SpawnManager>(SpawnManager::Params{
    .config = spawn_config_,
    .objects = dynamic_objects_,
    .map = map_,
    .rng_seed = rng_seed_,
  });

  DDRL_LOG_DEBUG(
    logger_,
    "SpawnManager initialized: {} objects with custom spawn config",
    dynamic_objects_.size()
  );
}

World::Impl::~Impl()
{
}

core::Result<void> World::Impl::update(double dt)
{
  sim_time_ += dt;
  last_update_dt_ = dt;

  // Update spawn manager (check despawns, attempt spawns)
  if (spawn_manager_ && robot_) {
    spawn_manager_->update(dt, robot_->get_pose(), episode_count_);
  }

  for (auto& obj : dynamic_objects_) {
    auto result = obj.update(dt);
    if (!result) {
      return std::unexpected(
        make_error_from(result.error(), ErrorCode::WORLD, core::make_context("World::Impl::update"))
      );
    }
  }

  auto result = robot_->update(dt).and_then([this]() { return update_scene_trees(); }
  ).and_then([this]() { return run_collisions_detection(); });

  if (!result) {
    return std::unexpected(
      make_error_from(result.error(), ErrorCode::WORLD, core::make_context("World::Impl::update"))
    );
  }

  snapshot_dirty_ = true;
  return {};
}

core::Result<Observation>
World::Impl::on_decision_tick(double tick_time, double tick_dt, double reward, bool is_terminal)
{
  auto obs_res = robot_->build_observation(tick_time, tick_dt, reward, is_terminal);
  if (!obs_res) {
    return obs_res;
  }
  auto obs = std::move(*obs_res);
  populate_route_observation(obs);
  return obs;
}

core::Result<Observation> World::Impl::build_initial_observation(double sim_time)
{
  auto obs_res = robot_->build_observation(sim_time, 0.0, 0.0, false);
  if (!obs_res) {
    return obs_res;
  }
  auto obs = std::move(*obs_res);
  populate_route_observation(obs);
  return obs;
}

core::Result<void> World::Impl::apply_action(const core::rl::Action& action)
{
  return robot_->set_action(action);
}

std::shared_ptr<const WorldStateSnapshot> World::Impl::capture_snapshot()
{
  if (snapshot_dirty_) {
    rebuild_world_snapshot();
    snapshot_dirty_ = false;
  }
  return last_world_snapshot_;
}

std::optional<CollisionReport> World::Impl::consume_collision_report()
{
  if (!pending_collision_report_) {
    return std::nullopt;
  }
  auto report = std::move(*pending_collision_report_);
  pending_collision_report_.reset();
  return report;
}

RewardTelemetry World::Impl::reward_telemetry()
{
  RewardTelemetry telemetry;
  telemetry.pose                   = robot_->get_pose();
  telemetry.velocity               = robot_->get_velocity();
  telemetry.command                = robot_->last_applied_command();
  telemetry.has_raw_policy_action  = robot_->requested_action().has_raw_policy_action;
  telemetry.normalized_speed_action = robot_->requested_action().normalized_speed_action;
  telemetry.normalized_steer_action = robot_->requested_action().normalized_steer_action;
  telemetry.raw_speed_action       = robot_->requested_action().raw_speed_action;
  telemetry.raw_steer_action       = robot_->requested_action().raw_steer_action;
  telemetry.reverse_gear           = robot_->reverse_gear();
  telemetry.gear_hold_time_s       = robot_->gear_hold_time_remaining();
  telemetry.requested_reverse_gear = robot_->requested_command().reverse_gear;

  const auto& waypoint              = current_waypoint();
  telemetry.waypoint_position       = waypoint.pos;
  telemetry.distance_to_waypoint    = core::math::distance(telemetry.pose.pos, waypoint.pos);
  telemetry.heading_to_waypoint     = std::atan2(
    waypoint.pos.y - telemetry.pose.pos.y, waypoint.pos.x - telemetry.pose.pos.x
  );
  telemetry.waypoint_index          = static_cast<uint32_t>(active_waypoint_index_);
  telemetry.waypoint_count          = static_cast<uint32_t>(active_route_.size());
  telemetry.final_goal_reached      = final_goal_reached_;
  telemetry.forward_clearance_m     = compute_forward_clearance_scan_like(
    robot_->last_update_info().sensors,
    telemetry.pose,
    rl_config_ ? rl_config_->reward.weights.clearance_gate_half_angle_rad : 0.45
  );

  if (!final_goal_reached_ && is_current_waypoint_reached(telemetry.pose.pos)) {
    telemetry.waypoint_reached = true;
    if (active_waypoint_index_ + 1 >= active_route_.size()) {
      final_goal_reached_           = true;
      telemetry.final_goal_reached  = true;
    } else {
      ++active_waypoint_index_;
      DDRL_LOG_DEBUG(
        logger_,
        "[Sim {}] Advanced route '{}' to waypoint {}/{} '{}'",
        sim_id_,
        active_route_name_,
        active_waypoint_index_ + 1,
        active_route_.size(),
        current_waypoint().name
      );
    }
  }

  return telemetry;
}

void World::Impl::select_route(std::mt19937& rng, bool randomize)
{
  if (map_->robot_routes.empty()) {
    throw std::runtime_error("Map requires at least one robot route");
  }

  std::size_t route_index = 0;
  if (randomize && map_->robot_routes.size() > 1) {
    std::uniform_int_distribution<std::size_t> dist(0, map_->robot_routes.size() - 1);
    route_index = dist(rng);
  }

  const auto& route = map_->robot_routes.at(route_index);
  if (route.waypoints.empty()) {
    throw std::runtime_error("Robot route has no waypoints: " + route.name);
  }

  active_route_name_ = route.name;
  active_route_start_pose_ = route.start;
  if (randomize) {
    std::uniform_real_distribution<double> yaw_dist(-std::numbers::pi, std::numbers::pi);
    active_route_start_pose_.yaw = yaw_dist(rng);
  }
  active_route_.clear();
  active_route_.reserve(route.waypoints.size());
  for (const auto& waypoint : route.waypoints) {
    active_route_.push_back(ActiveWaypoint{
      .name = waypoint.name, .pos = waypoint.pos, .tolerance = waypoint.tolerance
    });
  }
  active_waypoint_index_ = 0;
  final_goal_reached_    = false;
}

const World::Impl::ActiveWaypoint& World::Impl::current_waypoint() const
{
  if (active_route_.empty()) {
    throw std::runtime_error("Active robot route is empty");
  }
  return active_route_.at(std::min(active_waypoint_index_, active_route_.size() - 1));
}

bool World::Impl::is_current_waypoint_reached(const core::math::Point2& point) const
{
  const auto& waypoint = current_waypoint();
  return core::math::distance(point, waypoint.pos) <= waypoint.tolerance;
}

void World::Impl::populate_route_observation(core::rl::Observation& obs) const
{
  const auto& waypoint       = current_waypoint();
  obs.waypoint_pos_world     = waypoint.pos;
  obs.distance_to_waypoint   = core::math::distance(obs.est_pose.pos, waypoint.pos);
  obs.heading_to_waypoint    = std::atan2(
    waypoint.pos.y - obs.est_pose.pos.y, waypoint.pos.x - obs.est_pose.pos.x
  );
  obs.waypoint_reached       = final_goal_reached_ || is_current_waypoint_reached(obs.est_pose.pos);
  obs.final_goal_reached     = final_goal_reached_;
  obs.waypoint_index         = static_cast<uint32_t>(active_waypoint_index_);
  obs.waypoint_count         = static_cast<uint32_t>(active_route_.size());
}

Result<void> World::Impl::update_scene_trees()
{
  if (!broad_phase_) {
    return std::unexpected(make_error(
      ErrorCode::WORLD,
      "Broad phase is not initialized for updating scene trees",
      "World::Impl::update_scene_trees"
    ));
  }

  for (const auto& dynamic_object : dynamic_objects_) {
    const auto& entity_id = dynamic_object.get_id();
    broad_phase_->update_dynamics(entity_id, dynamic_object.get_pose());
  }

  if (robot_entity_id_ != std::numeric_limits<std::uint32_t>::max()) {
    broad_phase_->update_dynamics(robot_entity_id_, robot_->get_pose());
  }

  broad_phase_->refit_dynamics();
  return {};
}

void World::Impl::rebuild_world_snapshot()
{
  auto snapshot = std::make_shared<WorldStateSnapshot>();
  if (map_) {
    snapshot->current_map = map_->name;
  }
  snapshot->active_route_name    = active_route_name_;
  snapshot->active_waypoint_index = static_cast<uint32_t>(active_waypoint_index_);
  snapshot->active_waypoint_count = static_cast<uint32_t>(active_route_.size());

  snapshot->robot.model_id =
    last_world_snapshot_ ? last_world_snapshot_->robot.model_id : std::string{};
  snapshot->robot.pose        = robot_->get_pose();
  snapshot->robot.velocity    = robot_->get_velocity();
  snapshot->robot.is_collided = robot_->last_update_info().is_collided;

  snapshot->dynamic_objects.reserve(dynamic_objects_.size());
  for (const auto& obj : dynamic_objects_) {
    const auto&            cfg = obj.get_config();
    core::sim::ObjectState snapshot_obj;
    snapshot_obj.id         = obj.get_id();
    snapshot_obj.name       = cfg.name;
    snapshot_obj.model_id   = cfg.model_id;
    snapshot_obj.pose       = obj.get_pose();
    snapshot_obj.velocity   = obj.get_velocity();
    snapshot_obj.state_name = to_string(obj.get_state());
    snapshot->dynamic_objects.push_back(std::move(snapshot_obj));
  }

  const auto& sensors = robot_->last_update_info().sensors;
  if (sensors != nullptr) {
    snapshot->sensor_clouds.reserve(sensors->size());
    for (const auto& sensor : *sensors) {
      if (sensor && sensor->has_data()) {
        auto cloud_ptr = sensor->get_point_cloud();
        snapshot->sensor_clouds.push_back(std::move(cloud_ptr));
      }
    }
  }

  snapshot->sim_time   = sim_time_;
  last_world_snapshot_ = std::move(snapshot);
}

Result<void> World::Impl::run_collisions_detection()
{
  if (!broad_phase_) {
    return std::unexpected(make_error(
      ErrorCode::WORLD,
      "Broad phase is not initialized for collision detection",
      "World::Impl::run_collisions_detection"
    ));
  }

  const auto static_instances  = broad_phase_->static_instances();
  const auto dynamic_instances = broad_phase_->dynamic_instances();

  reset_instance_cache();
  for (const auto& inst : static_instances) {
    cache_instance(inst);
  }
  for (const auto& inst : dynamic_instances) {
    cache_instance(inst);
  }

  auto result = resolve_robot_collisions();
  if (!result) {
    return std::unexpected(make_error_from(
      result.error(), ErrorCode::WORLD, core::make_context("World::Impl::run_collisions_detection")
    ));
  }

  auto dyn_result = resolve_dynamic_object_collisions();
  if (!dyn_result) {
    return std::unexpected(make_error_from(
      dyn_result.error(),
      ErrorCode::WORLD,
      core::make_context("World::Impl::run_collisions_detection")
    ));
  }

  return {};
}

Result<void> World::Impl::resolve_robot_collisions()
{
  const auto robot_pose = robot_->get_pose();

  core::geom::AABB2 robot_query;
  robot_query = model_bounds(robot_->get_model_profile(), robot_pose);

  auto candidate_ids = broad_phase_->query_entity_ids(robot_query, true, true);

  std::vector<CollisionContact> contacts;
  std::vector<CollisionResult>  collision_results;
  contacts.reserve(candidate_ids.size());
  collision_results.reserve(candidate_ids.size());

  bool robot_collided = false;

  for (std::uint32_t candidate_entity_id : candidate_ids) {
    if (candidate_entity_id == robot_entity_id_) {
      continue;
    }

    const auto* entity = find_entity(candidate_entity_id);
    if (entity == nullptr) {
      continue;
    }

    const auto* inst = instance_for(candidate_entity_id);
    if ((inst == nullptr) || !inst->model) {
      continue;
    }

    Pose2 other_pose;
    bool  is_dynamic = false;

    switch (entity->type) {
      case EntityRef::Type::STATIC: {
        if (entity->index >= static_objects_.size()) {
          continue;
        }
        const auto& static_cfg = static_objects_[entity->index].get_config();
        other_pose             = static_cfg.pose_world;
        is_dynamic             = false;
        break;
      }
      case EntityRef::Type::DYNAMIC: {
        if (entity->index >= dynamic_objects_.size()) {
          continue;
        }
        auto& dyn_obj = dynamic_objects_[entity->index];
        if (!dyn_obj.is_active()) {
          continue;
        }
        other_pose = dyn_obj.get_pose();
        is_dynamic = true;
        break;
      }
      case EntityRef::Type::ROBOT:
        continue;
    }

    auto collision_result = narrow_phase_intersect_check(
      robot_->get_model_profile(), robot_pose, *inst->model, other_pose
    );

    if (collision_result.intersects) {
      robot_collided = true;

      if (is_dynamic && entity->index < dynamic_objects_.size()) {
        dynamic_objects_[entity->index].on_collision();
      }

      CollisionContact contact{};
      contact.other_name       = entity->name;
      contact.other_pose       = other_pose;
      contact.other_is_dynamic = is_dynamic;
      contacts.push_back(std::move(contact));
      collision_results.push_back(collision_result);
    }
  }

  if (robot_collided) {
    auto notify_result = robot_->notify_collision(contacts, sim_time_);
    if (!notify_result) {
      return std::unexpected(make_error_from(
        notify_result.error(),
        ErrorCode::WORLD,
        core::make_context("World::Impl::run_collisions_detection")
      ));
    }

    const auto robot_velocity = robot_->get_velocity();
    const auto severity =
      compute_collision_severity(contacts, collision_results, robot_velocity, robot_pose);

    CollisionReport report;
    report.timestamp = sim_time_;
    report.severity  = severity;
    report.contacts  = std::move(contacts);

    pending_collision_report_ = std::move(report);
  } else {
    robot_->clear_collision_state();
    pending_collision_report_.reset();
  }
  return {};
}

Result<void> World::Impl::resolve_dynamic_object_collisions()
{
  for (std::size_t i = 0; i < dynamic_objects_.size(); ++i) {
    if (!dynamic_objects_[i].is_active()) {
      continue;
    }

    auto self_entity_id = dynamic_objects_[i].get_id();

    const auto* self_instance = instance_for(self_entity_id);
    if ((self_instance == nullptr) || !self_instance->model) {
      DDRL_LOG_WARN(
        logger_,
        "Dynamic object entity_id={} has no collision instance in scene tree",
        self_entity_id
      );
      continue;
    }

    const auto& self_model = *self_instance->model;
    const auto  self_pose  = dynamic_objects_[i].get_pose();

    const auto candidates = broad_phase_->query_entity_ids(self_instance->bounds, false, true);

    for (std::uint32_t candidate_entity_id : candidates) {
      if (candidate_entity_id == self_entity_id) {
        continue;
      }

      const auto* entity = find_entity(candidate_entity_id);
      if (entity == nullptr) {
        continue;
      }

      const auto j = entity->index;
      if (j >= dynamic_objects_.size() || j <= i) {
        continue;
      }
      if (!dynamic_objects_[j].is_active()) {
        continue;
      }

      const auto* other_instance = instance_for(candidate_entity_id);
      if ((other_instance == nullptr) || !other_instance->model) {
        continue;
      }

      const auto  other_pose  = dynamic_objects_[j].get_pose();
      const auto& other_model = *other_instance->model;

      auto collision_result =
        narrow_phase_intersect_check(self_model, self_pose, other_model, other_pose);

      if (collision_result.intersects) {
        dynamic_objects_[i].on_collision();
        dynamic_objects_[j].on_collision();
      }
    }
  }

  for (auto& dynamic_object : dynamic_objects_) {
    if (!dynamic_object.is_active()) {
      continue;
    }

    auto self_entity_id = dynamic_object.get_id();

    const auto* self_instance = instance_for(self_entity_id);
    if ((self_instance == nullptr) || !self_instance->model) {
      DDRL_LOG_WARN(
        logger_,
        "Dynamic object entity_id={} has no collision instance in scene tree",
        self_entity_id
      );
      continue;
    }

    const auto& self_model = *self_instance->model;
    const auto  self_pose  = dynamic_object.get_pose();

    const auto candidates = broad_phase_->query_entity_ids(self_instance->bounds, true, false);

    for (std::uint32_t candidate_entity_id : candidates) {
      const auto* entity = find_entity(candidate_entity_id);
      if (entity == nullptr) {
        continue;
      }

      const auto* other_instance = instance_for(candidate_entity_id);
      if ((other_instance == nullptr) || !other_instance->model) {
        continue;
      }
      const auto  other_pose  = static_objects_[entity->index].get_config().pose_world;
      const auto& other_model = *other_instance->model;

      auto collision_result =
        narrow_phase_intersect_check(self_model, self_pose, other_model, other_pose);

      if (collision_result.intersects) {
        dynamic_object.on_collision();
      }
    }
  }
  return {};
}

void World::Impl::register_existing_name(const std::string& name)
{
  if (name.empty()) {
    return;
  }
  used_names_.emplace(name);
  name_suffix_counts_.try_emplace(name, 1);
}

std::string World::Impl::make_instance_name(const std::string& desired)
{
  std::string base   = desired.empty() ? std::string("object") : desired;
  auto [_, inserted] = used_names_.emplace(base);
  if (inserted) {
    name_suffix_counts_.try_emplace(base, 1);
    return base;
  }

  auto& counter = name_suffix_counts_[base];
  if (counter == 0) {
    counter = 1;
  }

  std::string candidate;
  while (true) {
    candidate = std::format("{}-{}", base, counter++);
    if (used_names_.emplace(candidate).second) {
      name_suffix_counts_.try_emplace(candidate, 1);
      return candidate;
    }
  }
}

void World::Impl::ensure_entity_slot(std::uint32_t entity_id)
{
  const auto required_size = static_cast<std::size_t>(entity_id) + 1;
  if (entity_registry_.size() < required_size) {
    entity_registry_.resize(required_size);
  }
  if (instance_lookup_.size() < required_size) {
    instance_lookup_.resize(required_size, nullptr);
  }
}

void World::Impl::register_entity(std::uint32_t entity_id, EntityRef entity)
{
  ensure_entity_slot(entity_id);
  entity_registry_[entity_id] = std::move(entity);
}

const World::Impl::EntityRef* World::Impl::find_entity(std::uint32_t entity_id) const
{
  if (entity_id >= entity_registry_.size()) {
    return nullptr;
  }
  const auto& slot = entity_registry_[entity_id];
  return slot ? std::addressof(*slot) : nullptr;
}

void World::Impl::reset_instance_cache()
{
  for (auto entity_id : instance_lookup_touched_) {
    if (entity_id < instance_lookup_.size()) {
      instance_lookup_[entity_id] = nullptr;
    }
  }
  instance_lookup_touched_.clear();
}

void World::Impl::cache_instance(const TLAS2::Instance& instance)
{
  ensure_entity_slot(instance.entity_id);
  instance_lookup_[instance.entity_id] = &instance;
  instance_lookup_touched_.push_back(instance.entity_id);
}

const TLAS2::Instance* World::Impl::instance_for(std::uint32_t entity_id) const
{
  if (entity_id >= instance_lookup_.size()) {
    return nullptr;
  }
  return instance_lookup_[entity_id];
}

core::geom::AABB2
World::Impl::model_bounds(const config::ModelProfile& model, const Pose2& world_pose)
{
  if (model.parts.empty()) {
    return core::geom::AABB2{world_pose.pos, world_pose.pos};
  }

  bool              initialized = false;
  core::geom::AABB2 bounds{};
  for (const auto& part : model.parts) {
    const auto local_bounds = core::geom::get_bounds(part.geom.base);
    const auto part_pose    = world_pose * part.rel_pose;
    const auto world_bounds = local_bounds.transform(part_pose);
    if (!initialized) {
      bounds      = world_bounds;
      initialized = true;
    } else {
      bounds.merge_inplace(world_bounds);
    }
  }
  return bounds;
}

CollisionResult World::Impl::narrow_phase_intersect_check(
  const config::ModelProfile& a_model, const Pose2& a_pose, const config::ModelProfile& b_model,
  const Pose2& b_pose
)
{
  for (const auto& a_part : a_model.parts) {
    const auto a_part_pose = a_pose * a_part.rel_pose;

    for (const auto& b_part : b_model.parts) {
      const auto b_part_pose = b_pose * b_part.rel_pose;
      auto result = collision::collide_shapes(a_part.geom, a_part_pose, b_part.geom, b_part_pose);
      if (result.intersects) {
        return result;
      }
    }
  }
  return CollisionResult{false};
}

[[nodiscard]] CollisionSeverity World::Impl::compute_collision_severity(
  const std::vector<CollisionContact>& contacts,
  const std::vector<CollisionResult>& collision_results, const core::math::Twist2& robot_velocity,
  const core::math::Pose2& robot_pose
)
{
  CollisionSeverity severity;
  severity.contact_count = contacts.size();

  if (contacts.empty()) {
    return severity;
  }

  // Configuration parameters (tunable based on robot characteristics)
  constexpr double kPenetrationThresholdMinor    = 0.01; // 1cm
  constexpr double kPenetrationThresholdModerate = 0.05; // 5cm
  constexpr double kPenetrationThresholdSevere   = 0.15; // 15cm

  constexpr double kSpeedThresholdMinor    = 0.5; // 0.5 m/s
  constexpr double kSpeedThresholdModerate = 1.5; // 1.5 m/s
  constexpr double kSpeedThresholdSevere   = 3.0; // 3.0 m/s

  constexpr double kImpactVelThresholdMinor    = 0.3; // 0.3 m/s
  constexpr double kImpactVelThresholdModerate = 1.0; // 1.0 m/s
  constexpr double kImpactVelThresholdSevere   = 2.0; // 2.0 m/s

  // Convert robot velocity from local frame to world frame
  const double     cos_yaw = std::cos(robot_pose.yaw);
  const double     sin_yaw = std::sin(robot_pose.yaw);
  core::math::Vec2 robot_vel_world{
    robot_velocity.linear.x * cos_yaw - robot_velocity.linear.y * sin_yaw,
    robot_velocity.linear.x * sin_yaw + robot_velocity.linear.y * cos_yaw
  };

  // Compute robot speed magnitude
  severity.robot_speed =
    std::sqrt(robot_vel_world.x * robot_vel_world.x + robot_vel_world.y * robot_vel_world.y);

  // Analyze each collision to find worst-case metrics
  double max_penetration     = 0.0;
  double max_impact_velocity = 0.0;

  for (size_t i = 0; i < std::min(contacts.size(), collision_results.size()); ++i) {
    const auto& contact = contacts[i];
    const auto& result  = collision_results[i];

    if (contact.other_is_dynamic) {
      severity.has_dynamic_contact = true;
    }

    // Track maximum penetration
    max_penetration = std::max(max_penetration, result.penetration_depth);

    // Compute impact velocity (velocity component along collision normal)
    // Negative dot product means robot is moving INTO the collision
    const double impact_vel =
      robot_vel_world.x * result.normal.x + robot_vel_world.y * result.normal.y;

    // Only consider velocity moving INTO the collision (negative direction)
    if (impact_vel < 0) {
      max_impact_velocity = std::max(max_impact_velocity, std::abs(impact_vel));
    }
  }

  severity.penetration_depth = max_penetration;
  severity.impact_velocity   = max_impact_velocity;

  // ==========================================================================
  // Compute individual factor scores (each normalized 0-1)
  // ==========================================================================

  double penetration_score = 0.0;
  if (max_penetration >= kPenetrationThresholdSevere) {
    penetration_score = 1.0;
  } else if (max_penetration >= kPenetrationThresholdModerate) {
    penetration_score = 0.6;
  } else if (max_penetration >= kPenetrationThresholdMinor) {
    penetration_score = 0.3;
  } else {
    penetration_score = std::min(1.0, max_penetration / kPenetrationThresholdMinor * 0.3);
  }

  double speed_score = 0.0;
  if (severity.robot_speed >= kSpeedThresholdSevere) {
    speed_score = 1.0;
  } else if (severity.robot_speed >= kSpeedThresholdModerate) {
    speed_score = 0.6;
  } else if (severity.robot_speed >= kSpeedThresholdMinor) {
    speed_score = 0.3;
  } else {
    speed_score = std::min(1.0, severity.robot_speed / kSpeedThresholdMinor * 0.3);
  }

  double impact_score = 0.0;
  if (max_impact_velocity >= kImpactVelThresholdSevere) {
    impact_score = 1.0;
  } else if (max_impact_velocity >= kImpactVelThresholdModerate) {
    impact_score = 0.6;
  } else if (max_impact_velocity >= kImpactVelThresholdMinor) {
    impact_score = 0.3;
  } else if (max_impact_velocity > 0) {
    impact_score = std::min(1.0, max_impact_velocity / kImpactVelThresholdMinor * 0.3);
  }

  double contact_score = 0.0;
  if (severity.contact_count >= 3) {
    contact_score = 1.0;
  } else if (severity.contact_count == 2) {
    contact_score = 0.5;
  } else {
    contact_score = 0.2;
  }

  double dynamic_score = severity.has_dynamic_contact ? 1.0 : 0.0;

  severity.score = 0.30 * penetration_score + // 30% weight on penetration depth
                   0.25 * impact_score +      // 25% weight on impact velocity
                   0.20 * speed_score +       // 20% weight on robot speed
                   0.15 * contact_score +     // 15% weight on number of contacts
                   0.10 * dynamic_score;      // 10% weight on dynamic objects

  // Classify severity level based on continuous score
  if (severity.score >= 0.75) {
    severity.severity_level = CollisionSeverity::Level::CRITICAL;
  } else if (severity.score >= 0.50) {
    severity.severity_level = CollisionSeverity::Level::SEVERE;
  } else if (severity.score >= 0.30) {
    severity.severity_level = CollisionSeverity::Level::MODERATE;
  } else if (severity.score >= 0.10) {
    severity.severity_level = CollisionSeverity::Level::MINOR;
  } else {
    severity.severity_level = CollisionSeverity::Level::NONE;
  }

  return severity;
}

// =============================================================================
// World Public APIs
// =============================================================================

World::World(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

World::~World() = default;

core::Result<void> World::update(double dt)
{
  return impl_->update(dt);
}

core::Result<Observation>
World::on_decision_tick(double tick_time, double tick_dt, double reward, bool is_terminal)
{
  return impl_->on_decision_tick(tick_time, tick_dt, reward, is_terminal);
}

core::Result<Observation> World::build_initial_observation(double sim_time)
{
  return impl_->build_initial_observation(sim_time);
}

core::Result<void> World::apply_action(const core::rl::Action& action)
{
  return impl_->apply_action(action);
}

std::shared_ptr<const core::sim::WorldStateSnapshot> World::capture_snapshot()
{
  return impl_->capture_snapshot();
}

std::optional<CollisionReport> World::consume_collision_report()
{
  return impl_->consume_collision_report();
}

RewardTelemetry World::reward_telemetry()
{
  return impl_->reward_telemetry();
}

std::shared_ptr<const config::MapConfig> World::get_map() const
{
  return impl_->get_map();
}

} // namespace ddrl::sim
