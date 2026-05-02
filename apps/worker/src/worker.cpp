#include "worker/worker.hpp"

#include "core/base/real_time_pacer.hpp"
#include "core/math/numeric.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "config/parsing.hpp"
#include "config/rl.hpp"
#include "config/simulator.hpp"
#include "config/visualizer.hpp"
#include "control/pure_pursuit.hpp"
#include "logging/logging.hpp"
#include "sim/map_manager.hpp"
#include "sim/simulator.hpp"
#include "visualizer/provider/event_bus_provider.hpp"
#include "visualizer/visualizer.hpp"
#include "worker/config.hpp"
#include "worker/policy_model.hpp"

namespace ddrl::worker {

using core::Error;
using core::ErrorCode;
using core::make_context;
using core::make_error;
using core::make_error_from;
using core::Result;
using PolicyType = config::RLPolicyConfig::Type;

namespace {

double signed_speed(const core::kinematics::AckermannCommand& cmd)
{
  return cmd.reverse_gear ? -cmd.speed : cmd.speed;
}

core::kinematics::AckermannCommand make_signed_command(double signed_speed_cmd, double steer_cmd)
{
  core::kinematics::AckermannCommand cmd;
  cmd.reverse_gear = signed_speed_cmd < 0.0;
  cmd.speed        = std::abs(signed_speed_cmd);
  cmd.steer        = steer_cmd;
  return cmd;
}

void set_normalized_action(
  core::rl::Action& action, double speed_max_fwd, double speed_max_rev, double steer_max_abs
)
{
  const double signed_speed_cmd = signed_speed(action.cmd);
  const double speed_scale = signed_speed_cmd >= 0.0 ? speed_max_fwd : speed_max_rev;
  action.normalized_speed_action = std::clamp(
    signed_speed_cmd / std::max(speed_scale, 1e-6),
    -1.0,
    1.0
  );
  action.normalized_steer_action = std::clamp(
    action.cmd.steer / std::max(steer_max_abs, 1e-6),
    -1.0,
    1.0
  );
}

void log_rollout_summary(
  const std::shared_ptr<logging::Logger>& logger, uint32_t sim_id, const Rollout& rollout
)
{
  if (!logger || rollout.steps.empty() || rollout.is_expert) {
    return;
  }

  double sum_reward     = 0.0;
  double min_reward     = std::numeric_limits<double>::infinity();
  double max_reward     = -std::numeric_limits<double>::infinity();
  double mean_speed     = 0.0;
  double mean_abs_steer = 0.0;
  double mean_raw_speed = 0.0;
  double mean_raw_steer = 0.0;
  double mean_log_prob  = 0.0;
  size_t raw_count      = 0;

  for (const auto& step : rollout.steps) {
    sum_reward += step.reward;
    min_reward = std::min(min_reward, step.reward);
    max_reward = std::max(max_reward, step.reward);
    mean_speed += signed_speed(step.action.cmd);
    mean_abs_steer += std::abs(step.action.cmd.steer);
    mean_log_prob += step.behaviour_log_prob;
    if (step.action.has_raw_policy_action) {
      mean_raw_speed += step.action.raw_speed_action;
      mean_raw_steer += step.action.raw_steer_action;
      ++raw_count;
    }
  }

  const double denom = static_cast<double>(rollout.steps.size());
  mean_speed /= denom;
  mean_abs_steer /= denom;
  mean_log_prob /= denom;
  if (raw_count > 0) {
    const double raw_denom = static_cast<double>(raw_count);
    mean_raw_speed /= raw_denom;
    mean_raw_steer /= raw_denom;
  }

  DDRL_LOG_INFO_THROTTLE(
    logger,
    1000,
    "[Sim {}] neural_rollout policy_v={} rollout_id={} steps={} return={:.3f} terminal={} "
    "reward(sum={:.3f} min={:.3f} max={:.3f}) action(mean_speed={:.3f} mean_abs_steer={:.3f}) "
    "raw(mean_speed={:.3f} mean_steer={:.3f} raw_count={}) log_prob(mean={:.3f})",
    sim_id,
    rollout.policy_version,
    rollout.rollout_id,
    rollout.steps.size(),
    rollout.episode_return,
    rollout.is_terminal,
    sum_reward,
    min_reward,
    max_reward,
    mean_speed,
    mean_abs_steer,
    mean_raw_speed,
    mean_raw_steer,
    raw_count,
    mean_log_prob
  );
}

} // namespace

Worker::Worker(Params params)
    : config_path_(std::move(params.config_path)),
      worker_version_(std::move(params.version)),
      config_parser_(std::make_shared<config::ConfigParser>())
{
  auto config_result = parse_worker_config(config_path_);
  if (!config_result) {
    throw std::runtime_error(
      std::format("Failed to load worker config:\n{}", to_string(config_result.error()))
    );
  }
  config_ = std::make_shared<WorkerConfig>(std::move(*config_result));
  apply_cli_config_overrides(*config_, params.overrides);
  logger_ = logging::get_logger("worker");

  if (config_->coordinator.worker_identity.empty()) {
    config_->coordinator.worker_identity = make_default_worker_identity();
  }
  DDRL_LOG_INFO(logger_, "Worker identity '{}'", config_->coordinator.worker_identity);
}

Worker::~Worker() = default;

core::Result<void> Worker::init()
{
  auto ctx = core::make_context("Worker::init");

  if (initialized_) {
    return std::unexpected(make_error(ErrorCode::WORKER, "Worker already initialized!", ctx));
  }

  const uint32_t hw          = std::max(1u, std::thread::hardware_concurrency());
  const bool     viz_enabled = config_->enable_visualization;

  uint32_t n_sim_thr = config_->num_sim_instances;
  uint32_t n_ray_thr = config_->num_raycast_threads;

  // Reserve cores: 1 for OS/background, 1 for VTK (if on), 1 for main
  const uint32_t reserve_os   = 1;
  const uint32_t reserve_viz  = viz_enabled ? 1u : 0u;
  const uint32_t reserve_main = 1;
  const uint32_t reserved     = std::min(hw, reserve_os + reserve_viz + reserve_main);

  const uint32_t avail = (hw > reserved) ? (hw - reserved) : 1;

  // Case A: both auto -> split 80/20 (sim/ray), at least 1 each
  if (n_sim_thr == 0 && n_ray_thr == 0) {
    n_sim_thr = std::max(1u, static_cast<uint32_t>(std::floor(avail * 0.8)));
    n_ray_thr = std::max(1u, avail - n_sim_thr);
  }
  // Case B: sims pinned, rays auto =>  leave whatever is left to rays
  else if (n_sim_thr > 0 && n_ray_thr == 0) {
    n_ray_thr = std::max(1u, (hw > reserved + n_sim_thr) ? (hw - reserved - n_sim_thr) : 1u);
  }
  // Case C: rays pinned, sims auto
  else if (n_ray_thr > 0 && n_sim_thr == 0) {
    n_sim_thr = std::max(1u, (hw > reserved + n_ray_thr) ? (hw - reserved - n_ray_thr) : 1u);
  }

  // Final clamps: never exceed available, keep both >=1
  if (n_sim_thr + n_ray_thr > avail) {
    // prefer keeping at least 1 ray thread to avoid sensor stalls
    n_ray_thr = std::max(1u, std::min(n_ray_thr, avail - 1));
    n_sim_thr = std::max(1u, avail - n_ray_thr);
  }

  n_ray_thr = std::min(8u, n_ray_thr);

  DDRL_LOG_INFO(logger_, "Detected {} hardware threads", hw);
  DDRL_LOG_INFO(logger_, "Spawning {} simulator instances", n_sim_thr);
  DDRL_LOG_INFO(logger_, "Creating raycast thread pool with {} threads", n_ray_thr);

  const bool record_demos = config_->demos.mode != WorkerConfig::DemoCfg::Mode::OFF;
  if (record_demos) {
    rollout_recorder_ = std::make_unique<RolloutRecorder>(RolloutRecorder::Params{
      .output_dir = std::filesystem::path(config_->demos.output_dir),
      .max_files  = config_->demos.max_files,
      .logger     = logger_,
    });
    DDRL_LOG_INFO(
      logger_,
      "Demo recording enabled (mode={}, output_dir={})",
      static_cast<int>(config_->demos.mode),
      config_->demos.output_dir
    );
  }

  const bool stream_enabled = (!config_->standalone_run_enable) &&
                              (config_->demos.mode != WorkerConfig::DemoCfg::Mode::RECORD_ONLY);

  if (!stream_enabled) {
    DDRL_LOG_INFO(
      logger_,
      "Rollout streaming disabled (standalone_run_enable={} demos.mode={})",
      config_->standalone_run_enable,
      static_cast<int>(config_->demos.mode)
    );
  } else {
    if (auto cp_result = setup_control_plane(n_sim_thr); !cp_result) {
      return cp_result;
    }
  }

  // Create rollout publisher whenever we either stream or record.
  // NOTE: RolloutPublisher currently gates its background thread + enqueue logic
  // on `enable_streaming`, so we must enable it when recording-only as well.
  if (stream_enabled || record_demos) {
    RolloutPublisher::Params params{
      .enable_streaming = (stream_enabled || record_demos),
      .max_queue_depth  = 4096,
      .max_batch_rollouts = config_->max_batch_rollouts,
      .flush_interval   = std::chrono::milliseconds(5),
      .logger           = logger_,
    };
    RolloutPublisher::PublishHandler handler = [this](const auto& batch) {
      this->publish_rollout_batch(batch);
    };
    rollout_publisher_ = std::make_unique<RolloutPublisher>(std::move(params), std::move(handler));
  }

  event_bus_    = std::make_shared<events::EventBus>();
  raycast_pool_ = std::make_shared<ddrl::threading::ThreadPool>(n_ray_thr, "raycast");

  config::RLConfig rl_cfg;
  if (!config_->rl_config_path.empty()) {
    auto rl_result = config_parser_->parse_rl_config(config_->rl_config_path);
    if (!rl_result) {
      return std::unexpected(rl_result.error());
    }
    rl_cfg = std::move(*rl_result);
  } else {
    auto arch_result = config_parser_->parse_policy_arch_config(config_->policy_arch_config_path);
    if (!arch_result) {
      return std::unexpected(arch_result.error());
    }
    auto reward_result = config_parser_->parse_reward_config(config_->reward_config_path);
    if (!reward_result) {
      return std::unexpected(reward_result.error());
    }
    rl_cfg.policy.type         = config_->policy_type;
    rl_cfg.policy.architecture = std::move(*arch_result);
    rl_cfg.reward              = std::move(*reward_result);
  }
  rl_config_ = std::make_shared<config::RLConfig>(std::move(rl_cfg));
  DDRL_LOG_INFO(
    logger_,
    "Loaded RL config: policy.type={} reward.type={}",
    static_cast<int>(rl_config_->policy.type),
    rl_config_->reward.reward_type
  );
  if (policy_subscriber_) {
    policy_subscriber_->set_policy_config(&rl_config_->policy);
  }
  if (rollout_stream_client_) {
    rollout_stream_client_->set_architecture(rl_config_->policy.architecture);
  }
  if (auto load_result = load_offline_policy(); !load_result) {
    return load_result;
  }

  // Load simulator and world configs using parsing utilities
  auto sim_config_result = config_parser_->parse_simulator_config(config_->sim_config_path);
  if (!sim_config_result) {
    return std::unexpected(sim_config_result.error());
  }
  sim_config_ = std::make_shared<config::SimConfig>(sim_config_result.value());
  if (cli_overrides_.seed) {
    if (*cli_overrides_.seed == -1) {
      const auto random_seed = static_cast<uint64_t>(
        std::random_device{}() ^ (static_cast<unsigned>(::getpid()) + 0x9e3779b9U)
      );
      sim_config_->rng_seed = random_seed;
      DDRL_LOG_INFO(logger_, "Using random simulator seed {}", sim_config_->rng_seed);
    } else {
      sim_config_->rng_seed = static_cast<uint64_t>(*cli_overrides_.seed);
      DDRL_LOG_INFO(logger_, "Using overridden simulator seed {}", sim_config_->rng_seed);
    }
  }

  auto world_config_result = config_parser_->parse_world_config(config_->world_config_path);
  if (!world_config_result) {
    return std::unexpected(world_config_result.error());
  }
  world_config_ = std::make_shared<config::WorldConfig>(std::move(world_config_result.value()));

  map_manager_ = std::make_shared<sim::MapManager>(sim::MapManager::Params{
    .map_directory = config_->map_directory,
    .parser        = config_parser_,
    .world_config  = world_config_,
  });

  auto available_maps = map_manager_->get_available_map_names();
  if (!available_maps.empty()) {
    std::ostringstream oss;
    for (size_t i = 0; i < available_maps.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << available_maps[i];
    }
    DDRL_LOG_INFO(logger_, "Available maps: {}", oss.str());
  }

  if (viz_enabled) {
    auto viz_cfg_result = config_parser_->parse_visualizer_config(config_->visualizer_config_path);
    if (!viz_cfg_result) {
      return std::unexpected(viz_cfg_result.error());
    }
    viz_config_ = std::make_shared<config::VizConfig>(viz_cfg_result.value());
  }

  // Create simulator instances
  sim_instances_.reserve(n_sim_thr);
  for (uint32_t sim_id = 0; sim_id < n_sim_thr; ++sim_id) {
    sim::Simulator::Params sim_params{
      .config                = sim_config_,
      .world_config          = world_config_,
      .rl_config             = rl_config_,
      .thread_pool           = raycast_pool_,
      .event_bus             = event_bus_,
      .initial_map           = nullptr,
      .map_manager           = map_manager_,
      .visualization_rate_hz = viz_enabled ? static_cast<double>(viz_config_->target_fps) : 0.0,
      .sim_id                = sim_id
    };

    auto simulator = std::make_shared<sim::Simulator>(sim_params);

    SimInstance instance(config_->unroll_length);
    instance.sim_id    = sim_id;
    instance.simulator = std::move(simulator);
    sim_instances_.push_back(std::move(instance));

    DDRL_LOG_INFO(logger_, "Prepared simulator instance {}", sim_id);
  }

  if (viz_enabled) {
    if (!viz_config_) {
      return std::unexpected(make_error(
        ErrorCode::WORKER, "Visualizer configuration not available", make_context("Worker::init")
      ));
    }
    // Collect simulator instance IDs
    std::vector<uint32_t> sim_ids;
    sim_ids.reserve(sim_instances_.size());
    for (const auto& sim : sim_instances_) {
      sim_ids.push_back(sim.sim_id);
    }

    viz_.set(std::make_unique<visualizer::Visualizer>(visualizer::Visualizer::Params{
      .provider      = std::make_shared<visualizer::EventBusStateProvider>(event_bus_),
      .config        = viz_config_,
      .sim_ids       = std::move(sim_ids),
      .world_config  = world_config_,
      .robot_profile = get_robot_profile(),
      .maps          = map_manager_->get_available_maps(),
      .on_shutdown =
        [this]() {
          DDRL_LOG_INFO(logger_, "Visualizer requested worker shutdown");
          this->request_stop();
        },
    }));

    DDRL_LOG_INFO(
      logger_, "Started visualization with {} simulator instances", sim_instances_.size()
    );
  }

  DDRL_LOG_INFO(logger_, "Worker initialization complete");

  initialized_ = true;
  return {};
}

Result<void> Worker::run()
{
  if (!initialized_) {
    DDRL_LOG_ERROR(logger_, "Worker not started - call init() first");
    return std::unexpected(make_error(
      ErrorCode::WORKER, "Worker not started - call init() first", core::make_context("Worker::run")
    ));
  }
  if (running_.load(std::memory_order_relaxed)) {
    return std::unexpected(
      make_error(ErrorCode::WORKER, "Worker already running!", core::make_context("Worker::run"))
    );
  }

  DDRL_LOG_INFO(logger_, "Worker starting...");

  stop_requested_.store(false, std::memory_order_relaxed);
  running_.store(true, std::memory_order_relaxed);

  if (coordinator_client_) {
    auto status_provider   = [this]() { return this->build_heartbeat_snapshot(); };
    auto directive_handler = [this](const ddrl::comm::ControlDirective& directive) {
      this->handle_control_directive(directive);
    };
    if (auto hb = coordinator_client_->start_heartbeat(status_provider, directive_handler); !hb) {
      DDRL_LOG_ERROR(logger_, "Failed to start coordinator heartbeat: {}", to_string(hb.error()));
    }
  }

  if (rollout_publisher_) {
    rollout_publisher_->start();
  }

  if (policy_subscriber_) {
    policy_subscriber_->start();
  }

  for (auto& instance : sim_instances_) {
    const uint32_t sim_id = instance.sim_id;
    instance.thread = std::jthread([this, sim_id](const std::stop_token& stoken) {
      sim_thread_main(stoken, sim_id);
    });
    DDRL_LOG_INFO(logger_, "Started simulator thread for instance {}", instance.sim_id);
  }

  if (viz_.enabled()) {
    viz_.launch();
    DDRL_LOG_INFO(logger_, "Started visualization thread");
  }

  DDRL_LOG_INFO(logger_, "Worker running...");

  while (!stop_requested_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  DDRL_LOG_INFO(logger_, "Worker::run loop exited");
  return {};
}

void Worker::request_stop()
{
  stop_requested_.store(true, std::memory_order_relaxed);
  for (auto& instance : sim_instances_) {
    instance.thread.request_stop();
  }
}

Result<void> Worker::stop()
{
  if (!running_.load(std::memory_order_relaxed)) {
    DDRL_LOG_INFO(logger_, "Worker stop requested but worker is not running");
    return {};
  }
  DDRL_LOG_INFO(logger_, "Stopping worker...");

  try {
    for (auto& instance : sim_instances_) {
      instance.thread.request_stop();
      if (instance.thread.joinable()) {
        instance.thread.join();
      }
      DDRL_LOG_INFO(logger_, "Simulator instance/thread {} stopped", instance.sim_id);
    }

    DDRL_LOG_INFO(logger_, "All simulator threads stopped");

    if (policy_subscriber_) {
      policy_subscriber_->stop();
      DDRL_LOG_INFO(logger_, "Policy subscriber stopped");
    }

    if (coordinator_client_) {
      coordinator_client_->stop();
      DDRL_LOG_INFO(logger_, "Coordinator heartbeat stopped");
    }

    if (rollout_publisher_) {
      rollout_publisher_->stop();
      DDRL_LOG_INFO(logger_, "Rollout publisher stopped");
    }

    if (viz_.enabled()) {
      viz_.stop();
      DDRL_LOG_INFO(logger_, "Visualization thread stopped");
    }

    running_.store(false, std::memory_order_relaxed);

    DDRL_LOG_INFO(logger_, "Worker stopped");
  } catch (const std::exception& e) {
    return std::unexpected(make_error(
      ErrorCode::WORKER,
      std::format("Error stopping worker: {}", e.what()),
      core::make_context("Worker::stop")
    ));
  }

  return {};
}

bool Worker::is_running() const
{
  return running_.load(std::memory_order_relaxed);
}

Worker::PolicyEval Worker::generate_action(
  uint32_t sim_id, const core::rl::Observation& obs, const EncodedObservation& encoded_obs,
  double sim_time
)
{
  const auto policy_type = rl_config_ ? rl_config_->policy.type : PolicyType::RANDOM;
  if (policy_type == PolicyType::NEURAL_NETWORK) {
    return run_neural_policy(sim_id, encoded_obs.features, sim_time);
  }
  if (policy_type == PolicyType::EXPERT) {
    return run_expert_policy(sim_id, obs, sim_time);
  }
  return run_random_policy(sim_time);
}

Worker::PolicyEval Worker::run_random_policy(double sim_time) const
{
  static thread_local std::mt19937 rng{std::random_device{}()};

  std::uniform_real_distribution<double> action_dist(-1.0, 1.0);

  PolicyEval eval;
  eval.action.timestamp         = sim_time;
  eval.action.normalized_speed_action = action_dist(rng);
  eval.action.normalized_steer_action = action_dist(rng);
  eval.action.cmd = make_signed_command(
    eval.action.normalized_speed_action,
    eval.action.normalized_steer_action
  );
  eval.log_prob                 = 0.0;
  return eval;
}

Worker::PolicyEval
Worker::run_expert_policy(uint32_t sim_id, const core::rl::Observation& obs, double sim_time)
{
  if (sim_id >= sim_instances_.size()) {
    return run_random_policy(sim_time);
  }
  auto& expert_state = sim_instances_[sim_id].expert;

  double speed_scale_fwd = 1.0;
  double speed_scale_rev = 1.0;
  double steer_scale     = 1.0;
  double wheel_base      = 2.5;
  if (const auto profile = get_robot_profile()) {
    const auto& kin = profile->kinematics;
    speed_scale_fwd = std::max(speed_scale_fwd, kin.caps_fwd.max_speed);
    speed_scale_rev = std::max(speed_scale_rev, kin.caps_rev.max_speed);
    steer_scale     = std::max(steer_scale, kin.max_steer_angle);
    wheel_base      = kin.wheel_base > 0.0 ? kin.wheel_base : wheel_base;
  }
  speed_scale_fwd = speed_scale_fwd > 0.0 ? speed_scale_fwd : 1.0;
  speed_scale_rev = speed_scale_rev > 0.0 ? speed_scale_rev : 1.0;
  steer_scale     = steer_scale > 0.0 ? steer_scale : 1.0;

  // Scripted expert:
  // - steer toward the active route waypoint
  // - go forward at a moderate speed when the path ahead is clear
  // - if an obstacle is too close in front (or we are in collision), stop and wait
  const double dx            = obs.waypoint_pos_world.x - obs.est_pose.pos.x;
  const double dy            = obs.waypoint_pos_world.y - obs.est_pose.pos.y;
  const double goal_distance = std::hypot(dx, dy);
  const double goal_heading  = std::atan2(dy, dx);
  const double heading_error = core::math::normalize_angle(goal_heading - obs.est_pose.yaw);
  const double current_speed = std::hypot(obs.est_vel.linear.x, obs.est_vel.linear.y);

  // Stop if currently colliding.
  bool should_stop = obs.is_collided();

  // Stop if obstacles are detected in a forward cone.
  // NOTE: point clouds are world-frame; we convert to ego for the check.
  // Heuristics:
  // - ignore near/self returns
  // - ignore low-height (ground) points
  // - require multiple close points before stopping (single point is too noisy)
  constexpr double stop_dist_m      = 2.5;
  constexpr double cone_half_fov    = 0.5; // radians (~57 deg)
  constexpr double ignore_near_m    = 1.0; // ignore self returns very close to the robot
  constexpr double min_obstacle_z_m = 0.2; // ignore ground / low clutter
  constexpr int    stop_point_count = 8;   // points within stop_dist_m to stop
  constexpr double emergency_stop_m = 1.2; // immediate stop if obstacle is this close
  constexpr int    emergency_count  = 2;
  const double     cos_yaw          = std::cos(-obs.est_pose.yaw);
  const double     sin_yaw          = std::sin(-obs.est_pose.yaw);
  const double     goal_x_body      = cos_yaw * dx - sin_yaw * dy;
  const double     rx               = obs.est_pose.pos.x;
  const double     ry               = obs.est_pose.pos.y;
  double           min_front_dist   = std::numeric_limits<double>::infinity();
  int              close_points     = 0;
  for (const auto& sensor_obs : obs.sensor_data) {
    const auto cloud = sensor_obs.point_cloud;
    if (!cloud) {
      continue;
    }
    for (const auto& pt : cloud->points) {
      if (pt.z < min_obstacle_z_m) {
        continue;
      }
      const double dxp = pt.x - rx;
      const double dyp = pt.y - ry;
      const double ex  = cos_yaw * dxp - sin_yaw * dyp;
      const double ey  = sin_yaw * dxp + cos_yaw * dyp;
      if (ex <= 0.0) {
        continue;
      }
      const double angle = std::atan2(ey, ex);
      if (std::abs(angle) > cone_half_fov) {
        continue;
      }
      const double dist = std::hypot(ex, ey);
      if (dist < ignore_near_m) {
        continue;
      }
      min_front_dist = std::min(min_front_dist, dist);
      if (dist < stop_dist_m) {
        ++close_points;
      }
    }
  }
  if ((min_front_dist < emergency_stop_m && close_points >= emergency_count) ||
      (close_points >= stop_point_count)) {
    should_stop = true;
  }

  const std::vector<core::math::Point2> local_path{
    obs.est_pose.pos,
    obs.waypoint_pos_world,
  };
  core::control::PurePursuitCaps pp_cfg{
    .lookahead_m                = std::clamp(goal_distance, 0.75, 2.5),
    .min_lookahead_m            = 0.75,
    .max_lookahead_m            = 3.0,
    .speed_coupled_to_curvature = true,
    .kv                         = 1.5,
  };
  const control::PurePursuitController pp(pp_cfg);
  constexpr double                     kExpertForwardNominalSpeedScale = 0.5;
  const double nominal_speed = kExpertForwardNominalSpeedScale * speed_scale_fwd;
  const auto   pp_out        = pp.track(obs.est_pose, local_path, current_speed, nominal_speed);
  const double pp_steer      = std::clamp(
    control::PurePursuitController::curvature_to_steering(pp_out.curvature, wheel_base),
    -steer_scale,
    steer_scale
  );
  constexpr double kRecoveryEnterGoalXBody = -0.25;
  constexpr double kRecoveryExitGoalXBody  = 0.75;
  constexpr double kRecoveryEnterHeading   = 2.2;
  constexpr double kRecoveryExitHeading    = 0.7;
  constexpr double kRecoveryMinDistance    = 0.75;
  const bool       waypoint_behind_enter   = goal_x_body < kRecoveryEnterGoalXBody;
  const bool       waypoint_forward_exit   = goal_x_body > kRecoveryExitGoalXBody;
  const bool       strong_misalignment     = std::abs(heading_error) > kRecoveryEnterHeading;
  const bool       mild_misalignment       = std::abs(heading_error) < kRecoveryExitHeading;

  if (should_stop || goal_distance <= kRecoveryMinDistance) {
    expert_state.reverse_recovery_active = false;
  } else if (!expert_state.reverse_recovery_active) {
    expert_state.reverse_recovery_active = waypoint_behind_enter || strong_misalignment;
  } else if (waypoint_forward_exit && mild_misalignment) {
    expert_state.reverse_recovery_active = false;
  }
  const bool use_reverse_recovery = expert_state.reverse_recovery_active;

  double speed_cmd = should_stop ? 0.0 : (pp_out.valid ? pp_out.target_speed : nominal_speed);
  double steer_cmd = should_stop ? 0.0 : (pp_out.valid ? pp_steer : 0.0);
  if (use_reverse_recovery) {
    constexpr double kReverseSteer = 0.8;
    speed_cmd                      = -std::max(0.25 * speed_scale_fwd, 0.6);
    // Reverse yaw response flips sign for the same steering angle.
    steer_cmd = std::clamp(-kReverseSteer * heading_error, -steer_scale, steer_scale);
  }

  PolicyEval eval;
  eval.action.timestamp = sim_time;
  eval.action.cmd       = make_signed_command(speed_cmd, steer_cmd);
  set_normalized_action(eval.action, speed_scale_fwd, speed_scale_rev, steer_scale);
  eval.action.cmd = make_signed_command(
    eval.action.normalized_speed_action,
    eval.action.normalized_steer_action
  );
  eval.log_prob         = 0.0;

  return eval;
}

Worker::PolicyEval
Worker::run_neural_policy(uint32_t sim_id, std::span<const float> encoded_features, double sim_time)
{
  std::optional<PolicyEval> fallback_eval;
  const auto                get_fallback = [this, sim_time, &fallback_eval]() -> PolicyEval& {
    if (!fallback_eval) {
      fallback_eval = run_random_policy(sim_time);
    }
    return *fallback_eval;
  };

  std::shared_ptr<const PolicyModel> model;
  if (policy_subscriber_) {
    model = policy_subscriber_->model();
  }
  if (!model && offline_policy_model_) {
    model = offline_policy_model_;
  }
  if (!model && policy_subscriber_) {
    model = policy_subscriber_->model_or_warn(logger_);
  }
  if (!model) {
    return get_fallback();
  }

  if (sim_id >= sim_instances_.size()) {
    return get_fallback();
  }
  auto&      instance        = sim_instances_[sim_id];
  auto&      ctx             = instance.policy;
  const auto current_version = model->version();

  if (!rl_config_) {
    return get_fallback();
  }
  const bool needs_init = ctx.policy_version != current_version;
  if (needs_init) {
    reset_policy_context(instance, current_version);
  }

  if (!model->evaluate_encoded(encoded_features, ctx, ctx.mean, ctx.log_std)) {
    return get_fallback();
  }

  auto neural_gaussian_rng = []() -> std::mt19937& {
    thread_local std::mt19937 rng = [] {
      std::random_device rd;
      return std::mt19937{rd()};
    }();
    return rng;
  };
  auto& rng = neural_gaussian_rng();

  std::normal_distribution<double> normal(0.0, 1.0);
  struct TanhGaussianSample {
    double x{0.0};
    double log_prob{0.0};
    double raw_u{0.0};
  };
  const auto clamp_log_std_for_dim = [](int action_idx, double log_std) {
    constexpr double kMinLogStd  = -5.0;
    const double     max_log_std = (action_idx == 0) ? -3.0 : -2.0;
    return std::clamp(log_std, kMinLogStd, max_log_std);
  };
  const auto sample_tanh_gaussian = [&](double mu, double log_std, int action_idx) {
    const auto clamped_log_std = clamp_log_std_for_dim(action_idx, log_std);
    const auto sigma           = std::exp(clamped_log_std);
    const auto u               = mu + sigma * normal(rng);
    const auto x               = std::tanh(u);
    const auto diff            = u - mu;
    const auto var             = std::max(sigma * sigma, 1e-9);
    const auto base_lp =
      -0.5 * ((diff * diff) / var + 2.0 * clamped_log_std + std::log(2.0 * std::numbers::pi));
    const auto log_det = std::log(std::max(1.0 - x * x, 1e-12));
    return TanhGaussianSample{x, base_lp - log_det, u};
  };

  PolicyEval eval;
  eval.action.timestamp             = sim_time;
  eval.action.has_raw_policy_action = true;
  double     log_prob               = 0.0;
  const bool deterministic_eval =
    config_ && (config_->standalone_run_enable || config_->deterministic_policy_eval);
  if (ctx.mean.size() >= 2 && ctx.log_std.size() >= 2) {
    // Normalized signed speed action in [-1, 1] via tanh.
    {
      const auto mu       = static_cast<double>(ctx.mean[0]);
      double     action_x = 0.0;
      double     raw_u    = mu;
      if (deterministic_eval) {
        action_x = std::tanh(mu);
      } else {
        const auto log_sd  = static_cast<double>(ctx.log_std[0]);
        const auto sample  = sample_tanh_gaussian(mu, log_sd, 0);
        action_x           = sample.x;
        raw_u              = sample.raw_u;
        log_prob += sample.log_prob;
      }
      eval.action.cmd               = make_signed_command(action_x, eval.action.cmd.steer);
      eval.action.normalized_speed_action = action_x;
      eval.action.raw_speed_action  = raw_u;
    }

    {
      const auto mu       = static_cast<double>(ctx.mean[1]);
      double     action_x = 0.0;
      double     raw_u    = mu;
      if (deterministic_eval) {
        action_x = std::tanh(mu);
      } else {
        const auto log_sd = static_cast<double>(ctx.log_std[1]);
        const auto sample = sample_tanh_gaussian(mu, log_sd, 1);
        action_x          = sample.x;
        raw_u             = sample.raw_u;
        log_prob += sample.log_prob;
      }
      eval.action.cmd.steer        = action_x;
      eval.action.normalized_steer_action = action_x;
      eval.action.raw_steer_action = raw_u;
    }
  }

  eval.log_prob = deterministic_eval ? 0.0 : log_prob;

  return eval;
}

void Worker::reset_policy_context(SimInstance& instance, uint64_t version) const
{
  auto&      ctx = instance.policy;
  ctx.hidden_state.clear();
  ctx.cell_state.clear();
  ctx.policy_version = version;
}

void Worker::handle_rollout_ready(
  uint32_t sim_id, std::vector<TrajectoryStep>&& steps, double episode_return, bool is_terminal
)
{
  if (!rollout_publisher_) {
    return;
  }
  if (sim_id >= sim_instances_.size()) {
    return;
  }
  if (steps.empty()) {
    return;
  }

  auto&   instance = sim_instances_[sim_id];
  Rollout rollout;
  rollout.sim_id         = sim_id;
  rollout.rollout_id     = instance.rollout_sequence++;
  rollout.policy_version = current_policy_version();
  rollout.is_expert      = rl_config_ && rl_config_->policy.type == PolicyType::EXPERT;
  rollout.episode_return = episode_return;
  rollout.is_terminal    = is_terminal;
  rollout.timestamp      = std::chrono::system_clock::now();
  rollout.steps          = std::move(steps);
  rollout.initial_recurrent_state = instance.trajectory_buffer.extract_initial_recurrent_state();

  log_rollout_summary(logger_, sim_id, rollout);

  if (!rollout_publisher_->enqueue(std::move(rollout))) {
    DDRL_LOG_WARN(
      logger_, "[Sim {}] Failed to enqueue rollout batch {}", sim_id, instance.rollout_sequence
    );
  }
}

void Worker::publish_rollout_batch(const std::vector<Rollout>& batch)
{
  if (batch.empty()) {
    return;
  }

  // Always record if enabled.
  if (rollout_recorder_) {
    const auto worker_id_str = coordinator_client_ ? coordinator_client_->effective_identity()
                                                   : config_->coordinator.worker_identity;
    const auto worker_id     = static_cast<uint32_t>(std::hash<std::string>{}(worker_id_str));
    rollout_recorder_->record_batch(
      batch,
      worker_id,
      ++rollout_batch_sequence_,
      rl_config_ ? rl_config_->policy.architecture : config::RLPolicyConfig::Architecture{}
    );
  }

  const bool stream_enabled = (!config_->standalone_run_enable) &&
                              (config_->demos.mode != WorkerConfig::DemoCfg::Mode::RECORD_ONLY);
  if (!stream_enabled) {
    return;
  }

  if (!rollout_stream_client_) {
    DDRL_LOG_WARN(logger_, "No rollout stream client configured; dropping batch");
    return;
  }
  if (!rollout_stream_client_->send(batch)) {
    DDRL_LOG_WARN(logger_, "Failed to stream rollout batch to learner");
    return;
  }
}

void Worker::sim_thread_main(const std::stop_token& stoken, uint32_t sim_index)
{
  // ============================================================
  // ====== init ======
  // ============================================================

  if (sim_index >= sim_instances_.size()) {
    DDRL_LOG_ERROR(
      logger_,
      "[Sim {}] Thread received invalid simulator index; instance count={}",
      sim_index,
      sim_instances_.size()
    );
    return;
  }

  auto& instance  = sim_instances_[sim_index];
  auto& simulator = *instance.simulator;

  core::RealTimePacer pacer(config_->real_time_factor);
  const bool neural_policy = rl_config_ && rl_config_->policy.type == PolicyType::NEURAL_NETWORK;

  const auto id = instance.sim_id;
  DDRL_LOG_INFO(logger_, "[Sim {}] Thread started", id);

  auto& buffer = instance.trajectory_buffer;
  buffer.clear();
  if (neural_policy) {
    reset_policy_context(instance, current_policy_version());
  }

  const auto snapshot_recurrent_state_if_fragment_start = [&]() {
    if (!neural_policy || !buffer.empty() || buffer.has_pending()) {
      return;
    }
    const auto& ctx = instance.policy;
    if (!ctx.hidden_state.empty() && ctx.hidden_state.size() == ctx.cell_state.size()) {
      buffer.set_initial_recurrent_state(RecurrentStateSnapshot{
        .hidden = ctx.hidden_state,
        .cell   = ctx.cell_state,
        .shape  = {1, static_cast<int32_t>(ctx.hidden_state.size())},
      });
      return;
    }
    if (
      rl_config_ && rl_config_->policy.architecture.policy_model == "lstm" &&
      rl_config_->policy.architecture.lstm_hidden_dim > 0
    ) {
      const auto hidden_dim =
        static_cast<std::size_t>(rl_config_->policy.architecture.lstm_hidden_dim);
      buffer.set_initial_recurrent_state(RecurrentStateSnapshot{
        .hidden = std::vector<float>(hidden_dim, 0.0f),
        .cell   = std::vector<float>(hidden_dim, 0.0f),
        .shape  = {1, static_cast<int32_t>(hidden_dim)},
      });
    }
  };

  // ============================================================
  // ====== run ======
  // ============================================================

  std::optional<PolicyEval> next_action;
  const auto reseed_until_ready = [&]() -> bool {
    auto result = simulator.reset({});
    while (!stoken.stop_requested()) {
      if (!result) {
        DDRL_LOG_ERROR(logger_, "[Sim {}] Reset failed: {}", id, to_string(result.error()));
        return false;
      }
      auto res         = std::move(*result);
      auto obs         = std::move(res.initial_observation.observation);
      auto encoded_obs = encode_observation(
        obs, rl_config_ ? rl_config_->policy.architecture : config::RLPolicyConfig::Architecture{}
      );

      const double sim_time = res.initial_observation.sim_time;
      if (!obs.is_terminal) {
        snapshot_recurrent_state_if_fragment_start();
        auto                      eval = generate_action(id, obs, encoded_obs, sim_time);
        if (viz_.enabled()) {
          auto debug_cloud    = std::make_shared<core::vision::PointCloud>();
          debug_cloud->points = encoded_obs.representative_points_world;
          viz_.get()->update_observation_debug_cloud(id, std::move(debug_cloud));
        }
        buffer.set_pending_step(
          std::move(obs),
          std::move(encoded_obs.features),
          eval.action,
          eval.log_prob
        );
        next_action = eval;
        pacer.reset(sim_time);
        return true;
      }
      DDRL_LOG_WARN(logger_, "[Sim {}] Initial observation is terminal", id);
      next_action.reset();
      result = simulator.reset({});
    }
    return false;
  };

  if (!reseed_until_ready()) {
    return;
  }

  while (!stoken.stop_requested()) {
    sim::Simulator::StepRequest request;
    request.action = next_action->action;

    auto step_result = simulator.step(request);
    if (!step_result) {
      DDRL_LOG_ERROR(logger_, "[Sim {}] Step failed: {}", id, to_string(step_result.error()));
      break;
    }
    auto step        = std::move(step_result.value());
    auto obs         = std::move(step.observation);
    auto encoded_obs = encode_observation(
      obs, rl_config_ ? rl_config_->policy.architecture : config::RLPolicyConfig::Architecture{}
    );

    const auto& rl_info     = step.rl_metadata;
    const bool  is_terminal = obs.is_terminal;

    pacer.pace(step.sim_time);

    // Complete any pending step (S_t, A_t) with the received (R_t)
    auto completed_step =
      buffer.complete_pending_step(obs.reward, is_terminal, rl_info.reward_terms);
    buffer.push_step(std::move(completed_step));
    if (viz_.enabled()) {
      viz_.get()->update_rl_info(id, rl_info);
      auto debug_cloud    = std::make_shared<core::vision::PointCloud>();
      debug_cloud->points = encoded_obs.representative_points_world;
      viz_.get()->update_observation_debug_cloud(id, std::move(debug_cloud));
    }

    // Check if rollouts should be published
    if (buffer.is_full() || is_terminal) {
      if (!buffer.empty()) {
        auto rollout_steps = buffer.extract();
        handle_rollout_ready(id, std::move(rollout_steps), rl_info.episode_return, is_terminal);
      }
    }

    if (is_terminal) {
      next_action.reset();
      buffer.clear();
      if (neural_policy) {
        reset_policy_context(instance, current_policy_version());
      }

      if (!reseed_until_ready()) {
        break;
      }
    } else {
      snapshot_recurrent_state_if_fragment_start();
      auto                      eval = generate_action(id, obs, encoded_obs, step.sim_time);
      buffer.set_pending_step(
        std::move(obs),
        std::move(encoded_obs.features),
        eval.action,
        eval.log_prob
      );
      next_action = eval;
    }
  }

  DDRL_LOG_INFO(logger_, "[Sim {}] Thread stopped", id);
}

core::Result<void> Worker::setup_control_plane(uint32_t num_sim_instances)
{
  (void)num_sim_instances;

  if (config_->coordinator.target.empty()) {
    return std::unexpected(make_error(
      ErrorCode::CONFIG,
      "Coordinator address is required when standalone_run_enable is false",
      make_context("Worker::setup_control_plane")
    ));
  }

  DDRL_LOG_INFO(logger_, "Connecting to coordinator at {}...", config_->coordinator.target);

  try {
    coordinator_client_ = std::make_unique<CoordinatorClient>(CoordinatorClient::Params{
      .config = std::shared_ptr<WorkerConfig::CoordinatorCfg>(config_, &config_->coordinator),
      .worker_version = worker_version_,
    });
  } catch (const std::exception& e) {
    return std::unexpected(make_error(
      ErrorCode::WORKER,
      std::format("Failed to set up coordinator client: {}", e.what()),
      make_context("Worker::setup_control_plane")
    ));
  }

  const std::string subscriber_id = coordinator_client_->assigned_identity().empty()
                                      ? coordinator_client_->effective_identity()
                                      : coordinator_client_->assigned_identity();

  policy_subscriber_ = std::make_unique<PolicySubscriber>(PolicySubscriber::Params{
    .worker_id     = subscriber_id,
    .session_token = coordinator_client_->session_token(),
  });

  std::string initial_target = coordinator_client_->learner_target();
  if (initial_target.empty()) {
    return std::unexpected(make_error(
      ErrorCode::WORKER,
      "Coordinator returned empty learner target; start a learner before starting workers",
      make_context("Worker::setup_control_plane")
    ));
  }
  policy_subscriber_->set_target(initial_target);

  rollout_stream_client_ = std::make_unique<RolloutStreamClient>(RolloutStreamClient::Params{
    .target        = initial_target,
    .worker_id     = subscriber_id,
    .session_token = coordinator_client_->session_token(),
    .architecture =
      rl_config_ ? rl_config_->policy.architecture : config::RLPolicyConfig::Architecture{},
    .logger = logger_,
  });

  return {};
}

HeartbeatSnapshot Worker::build_heartbeat_snapshot() const
{
  HeartbeatSnapshot snapshot;
  snapshot.phase             = stop_requested_.load(std::memory_order_relaxed) ? "stopping"
                               : running_.load(std::memory_order_relaxed)      ? "running"
                                                                               : "initializing";
  snapshot.active_simulators = static_cast<uint32_t>(sim_instances_.size());
  snapshot.detail =
    coordinator_client_ ? coordinator_client_->effective_identity() : "<unregistered>";
  return snapshot;
}

void Worker::handle_control_directive(const ddrl::comm::ControlDirective& directive)
{
  switch (directive.type()) {
    case ddrl::comm::CONTROL_SWITCH_LEARNER:
      if (policy_subscriber_) {
        policy_subscriber_->set_target(directive.learner_target());
      }
      if (rollout_stream_client_) {
        rollout_stream_client_->set_target(directive.learner_target());
      }
      if (logger_) {
        DDRL_LOG_INFO(
          logger_,
          "Switching learner endpoint to {} (reason: {})",
          directive.learner_target(),
          directive.message()
        );
      }
      break;
    case ddrl::comm::CONTROL_DRAIN:
      if (logger_) {
        DDRL_LOG_INFO(logger_, "Received drain directive: {}", directive.message());
      }
      break;
    case ddrl::comm::CONTROL_SHUTDOWN:
      if (logger_) {
        DDRL_LOG_WARN(logger_, "Received shutdown directive: {}", directive.message());
      }
      request_stop();
      break;
    default:
      if (logger_) {
        DDRL_LOG_INFO(logger_, "Unhandled control directive: {}", directive.message());
      }
      break;
  }
}

core::Result<void> Worker::load_offline_policy()
{
  if (!config_ || config_->policy_onnx_path.empty()) {
    return {};
  }
  if (!rl_config_) {
    return std::unexpected(make_error(
      ErrorCode::WORKER,
      "RL config must be loaded before loading offline policy",
      make_context("Worker::load_offline_policy")
    ));
  }

  std::filesystem::path onnx_path{config_->policy_onnx_path};
  std::ifstream         input(onnx_path, std::ios::binary);
  if (!input) {
    return std::unexpected(make_error(
      ErrorCode::CONFIG,
      std::format("Failed to open ONNX policy file: {}", onnx_path.string()),
      make_context("Worker::load_offline_policy")
    ));
  }

  std::vector<uint8_t> bytes(
    (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
  );
  if (bytes.empty()) {
    return std::unexpected(make_error(
      ErrorCode::CONFIG,
      std::format("ONNX policy file is empty: {}", onnx_path.string()),
      make_context("Worker::load_offline_policy")
    ));
  }

  PolicyPayload payload;
  payload.version    = 1;
  payload.onnx_model = std::move(bytes);

  auto model = PolicyModel::from_payload(payload, rl_config_->policy, logger_);
  if (!model) {
    return std::unexpected(make_error(
      ErrorCode::WORKER,
      std::format("Failed to build policy model from ONNX: {}", onnx_path.string()),
      make_context("Worker::load_offline_policy")
    ));
  }
  offline_policy_model_ = model;
  DDRL_LOG_INFO(
    logger_,
    "Loaded offline ONNX policy from {} ({} bytes)",
    onnx_path.string(),
    payload.onnx_model.size()
  );
  return {};
}

uint64_t Worker::current_policy_version() const
{
  if (policy_subscriber_) {
    return policy_subscriber_->version();
  }
  if (offline_policy_model_) {
    return offline_policy_model_->version();
  }
  return 0;
}

std::string Worker::make_default_worker_identity()
{
  std::array<char, 128> hostname{};
  if (::gethostname(hostname.data(), hostname.size()) != 0) {
    return std::format("worker-{}", static_cast<uint32_t>(::getpid()));
  }
  hostname.back() = '\0';
  return std::format("{}-{}", hostname.data(), static_cast<uint32_t>(::getpid()));
}

std::shared_ptr<const config::RobotProfile> Worker::get_robot_profile() const
{
  if (!world_config_) {
    return nullptr;
  }
  auto robot_model_id = world_config_->robot.robot_profile_id;
  auto it             = world_config_->robot_profiles.find(robot_model_id);
  if (it == world_config_->robot_profiles.end()) {
    return nullptr;
  }
  return {world_config_, &it->second};
}

} // namespace ddrl::worker
