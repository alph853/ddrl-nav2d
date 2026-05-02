#include "sim/simulator.hpp"

#include "core/base/variant_overload.hpp"
#include "core/rl/reward.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "events/sim_models.hpp"
#include "logging/logging.hpp"
#include "sim/map_manager.hpp"
#include "sim/world.hpp"

namespace ddrl::sim {

using config::WorldConfig;
using core::ErrorCode;
using core::make_context;
using core::make_error;
using core::make_error_from;
using core::Result;

static constexpr size_t kStepTimingLogInterval = 120;

namespace {

uint64_t mix_seed(uint64_t base, uint64_t salt)
{
  uint64_t x = base + 0x9e3779b97f4a7c15ULL + (salt << 6) + (salt >> 2);
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

} // namespace

class Simulator::Impl
{
  friend class Simulator;

private:
  std::shared_ptr<const config::SimConfig>   config_;
  std::shared_ptr<const config::WorldConfig> world_config_;
  std::shared_ptr<const config::RLConfig>    rl_config_;
  std::shared_ptr<threading::ThreadPool>     thread_pool_;
  std::shared_ptr<events::EventBus>          event_bus_;
  std::shared_ptr<MapManager>                map_manager_;

  std::unique_ptr<World> world_;

  std::shared_ptr<const config::MapConfig>          current_map_;
  std::shared_ptr<const config::SpawnManagerConfig> active_spawn_config_;

  std::unique_ptr<core::rl::RewardFunction> reward_fn_;
  std::shared_ptr<logging::Logger>          logger_;

  uint32_t     sim_id_{0};
  std::mt19937 rng_{std::random_device{}()};

  double sim_time_s_{0.0};
  double viz_publish_period_{0.05};
  double next_viz_publish_time_{0.0};
  bool   visualization_enabled_{true};

  double                    accumulated_reward_{0.0};
  core::rl::RewardTermValues accumulated_reward_terms_;
  bool   episode_terminal_{false};
  bool   initialized_{false};
  double collision_timeout_s_{0.0};
  double collision_dwell_time_{0.0};

  double episode_return_{0.0};
  size_t episode_step_count_{0};
  size_t episode_index_{0};
  bool   last_episode_success_{false};
  bool   have_last_episode_result_{false};

  size_t           current_phase_idx_{0};
  size_t           episodes_in_phase_{0};
  size_t           successes_in_phase_{0};
  std::deque<bool> success_window_;

  double step_time_window_accum_s_{0.0};
  size_t step_time_window_samples_{0};

  core::sim::CollisionSeverity             step_collision_severity_{};
  std::vector<core::sim::CollisionContact> step_collision_contacts_;
  bool                                     step_waypoint_reached_{false};
  bool                                     step_goal_reached_{false};

public:
  Impl(Simulator::Params&& params);
  ~Impl();

  core::Result<ResetResult> reset(ResetParams params);
  core::Result<StepResult>  step(StepRequest request);

private:
  core::Result<void> apply_action_if_requested(const StepRequest& request);
  core::Result<void> integrate_until(double target_time);
  core::Result<void> publish_world_snapshot(double current_time, bool force_publish);

  void record_episode_outcome(bool success);
  void maybe_advance_curriculum();

  [[nodiscard]] const config::SimConfig::CurriculumPhase* current_phase() const;
  std::shared_ptr<const config::MapConfig>
  pick_phase_map(const config::SimConfig::CurriculumPhase* phase);
  std::shared_ptr<const config::SpawnManagerConfig>
  build_spawn_config(const config::SimConfig::CurriculumPhase* phase);

  void reset_episode_counters();
  void log_step_timing(double last_step_seconds);

  static void derive_perception_from_profile(
    const config::RobotProfile& profile, config::SpawnManagerConfig& spawn_cfg
  );
};

Simulator::Impl::Impl(Simulator::Params&& params)
    : config_(std::move(params.config)),
      world_config_(std::move(params.world_config)),
      rl_config_(std::move(params.rl_config)),
      thread_pool_(std::move(params.thread_pool)),
      event_bus_(std::move(params.event_bus)),
      map_manager_(std::move(params.map_manager)),
      current_map_(std::move(params.initial_map)),
      logger_(logging::get_logger("sim.simulator")),
      sim_id_(params.sim_id)
{
  if (!config_ || !world_config_) {
    throw std::runtime_error("Simulator configuration or world configuration is null");
  }
  if (!rl_config_) {
    throw std::runtime_error("Simulator RL configuration is null");
  }

  if (!thread_pool_) {
    throw std::runtime_error("Thread pool is null for Simulator");
  }

  if (!event_bus_) {
    throw std::runtime_error("Event bus is null for Simulator");
  }

  if (event_bus_) {
    auto result = event_bus_->register_queue<events::WorldStateEvent>(sim_id_, 1024, 1);
    if (!result && result.error().code != ErrorCode::EVENT_BUS_REGISTERED) {
      throw std::runtime_error(
        "Failed to register world state event queue for sim_id " + std::to_string(sim_id_) + ": " +
        core::to_string(result.error())
      );
    }
  }

  if (!map_manager_) {
    DDRL_LOG_WARN(
      logger_, "Simulator constructed without MapManager; resets will reuse existing map"
    );
  }

  visualization_enabled_ = params.visualization_rate_hz > 0.0;
  rng_.seed(static_cast<std::mt19937::result_type>(mix_seed(config_->rng_seed, sim_id_)));
  if (visualization_enabled_) {
    viz_publish_period_ = 1.0 / params.visualization_rate_hz;
  } else {
    viz_publish_period_ = std::numeric_limits<double>::infinity();
  }
  const auto& rl_cfg = rl_config_->reward;
  reward_fn_ = core::rl::create_reward_function(rl_cfg.reward_type, rl_cfg.weights);
  collision_timeout_s_ = config_->episode.collision_timeout_s;
}

Simulator::Impl::~Impl()
{
  if (event_bus_) {
    auto result = event_bus_->unregister_queue<events::WorldStateEvent>(sim_id_);
    if (!result) {
      DDRL_LOG_ERROR(
        logger_,
        "Failed to unregister world state event queue for sim_id {}: {}",
        sim_id_,
        to_string(result.error())
      );
    }
  }
}

core::Result<Simulator::ResetResult> Simulator::Impl::reset(ResetParams params)
{
  (void)params;
  if (have_last_episode_result_) {
    record_episode_outcome(last_episode_success_);
    maybe_advance_curriculum();
    have_last_episode_result_ = false;
  }

  const auto* phase = current_phase();

  auto map = pick_phase_map(phase);
  if (!map) {
    return std::unexpected(
      make_error(ErrorCode::SIM, "Map configuration is null", make_context("Simulator::reset"))
    );
  }

  active_spawn_config_ = build_spawn_config(phase);
  if (!active_spawn_config_) {
    return std::unexpected(make_error(
      ErrorCode::SIM, "Failed to build spawn manager config", make_context("Simulator::reset")
    ));
  }

  try {
    const uint64_t world_seed =
      mix_seed(static_cast<uint64_t>(rng_()), static_cast<uint64_t>(episode_index_ + 1));
    world_ = std::make_unique<World>(World::Params{
      .config       = world_config_,
      .map          = map,
      .spawn_config = std::move(active_spawn_config_),
      .thread_pool  = thread_pool_,
      .rl_config    = rl_config_,
      .rng_seed     = world_seed,
      .sim_id       = sim_id_
    });
  } catch (const std::exception& e) {
    return std::unexpected(make_error(
      ErrorCode::SIM,
      std::format("Failed to create world: {}", e.what()),
      make_context("Simulator::create_world")
    ));
  }
  current_map_ = std::move(map);

  sim_time_s_              = 0.0;
  next_viz_publish_time_   = 0.0;
  accumulated_reward_      = 0.0;
  accumulated_reward_terms_.clear();
  episode_terminal_        = false;
  step_collision_severity_ = {};
  step_collision_contacts_.clear();
  step_waypoint_reached_ = false;
  step_goal_reached_ = false;
  collision_dwell_time_ = 0.0;
  if (reward_fn_) {
    reward_fn_->reset();
  }

  reset_episode_counters();

  auto obs = world_->build_initial_observation(sim_time_s_);
  if (!obs) {
    return std::unexpected(
      make_error_from(obs.error(), ErrorCode::SIM, make_context("Simulator::reset"))
    );
  }

  auto publish_res = publish_world_snapshot(sim_time_s_, true);
  if (!publish_res) {
    return std::unexpected(publish_res.error());
  }

  ResetResult result;
  result.initial_observation.observation  = std::move(obs.value());
  result.initial_observation.sim_time     = sim_time_s_;
  result.initial_observation.episode_done = result.initial_observation.observation.is_terminal;
  result.initial_observation.rl_metadata.episode_index      = episode_index_;
  result.initial_observation.rl_metadata.episode_step_count = episode_step_count_;
  result.initial_observation.rl_metadata.step_reward        = 0.0;
  result.initial_observation.rl_metadata.episode_return     = episode_return_;
  result.initial_observation.rl_metadata.goal_reached       = false;
  result.initial_observation.rl_metadata.reward_terms.clear();
  result.episode_index                                      = episode_index_;

  return result;
}

core::Result<Simulator::StepResult> Simulator::Impl::step(StepRequest request)
{
  const auto step_start = std::chrono::steady_clock::now();

  if (!world_) {
    return std::unexpected(
      make_error(ErrorCode::SIM, "Simulator world not initialized", make_context("Simulator::step"))
    );
  }

  if (auto res = apply_action_if_requested(request); !res) {
    return std::unexpected(res.error());
  }

  // Clear collision state for this decision step
  step_collision_severity_ = {};
  step_collision_contacts_.clear();
  step_waypoint_reached_ = false;
  step_goal_reached_ = false;

  const double decision_dt = config_->decision.decision_dt;
  const double target_time = sim_time_s_ + decision_dt;

  if (auto res = integrate_until(target_time); !res) {
    return std::unexpected(res.error());
  }

  const bool step_limit_reached =
    (config_->episode.max_decision_steps > 0 &&
     episode_step_count_ + 1 >= config_->episode.max_decision_steps);
  const bool time_limit_reached =
    (config_->episode.max_sim_seconds > 0.0 &&
     target_time >= config_->episode.max_sim_seconds - core::math::kEps);
  if (step_limit_reached || time_limit_reached) {
    episode_terminal_ = true;
  }

  auto obs =
    world_->on_decision_tick(target_time, decision_dt, accumulated_reward_, episode_terminal_);
  if (!obs) {
    return std::unexpected(
      make_error_from(obs.error(), ErrorCode::SIM, make_context("Simulator::step"))
    );
  }

  StepResult result;
  result.observation  = std::move(obs.value());
  result.sim_time     = target_time;
  result.episode_done = result.observation.is_terminal;
  result.time_limit_reached = time_limit_reached;

  episode_terminal_   = result.episode_done;
  result.rl_metadata.reward_terms = std::move(accumulated_reward_terms_);
  accumulated_reward_ = 0.0;
  accumulated_reward_terms_.clear();

  episode_return_ += result.observation.reward;
  ++episode_step_count_;

  result.rl_metadata.episode_index      = episode_index_;
  result.rl_metadata.episode_step_count = episode_step_count_;
  result.rl_metadata.step_reward        = result.observation.reward;
  result.rl_metadata.episode_return     = episode_return_;
  result.rl_metadata.goal_reached       = step_goal_reached_;

  if (result.episode_done) {
    last_episode_success_     = step_goal_reached_;
    have_last_episode_result_ = true;

    DDRL_LOG_INFO_THROTTLE(
      logger_,
      1000,
      "[Sim {}] episode_done success={} return={:.2f} steps={} time_limit={} collision_level={} collision_score={:.2f}",
      sim_id_,
      last_episode_success_ ? 1 : 0,
      episode_return_,
      episode_step_count_,
      time_limit_reached ? 1 : 0,
      to_string(step_collision_severity_.severity_level),
      step_collision_severity_.score
    );
  }

  const double step_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
  log_step_timing(step_seconds);

  return result;
}

core::Result<void> Simulator::Impl::apply_action_if_requested(const StepRequest& request)
{
  if (!request.action) {
    return {};
  }

  auto action = *request.action;
  if (action.timestamp <= 0.0) {
    action.timestamp = sim_time_s_;
  }

  auto res = world_->apply_action(action);
  if (!res) {
    return std::unexpected(make_error_from(
      res.error(), ErrorCode::SIM, make_context("Simulator::apply_action_if_requested")
    ));
  }
  return {};
}

core::Result<void> Simulator::Impl::integrate_until(double target_time)
{
  const double fixed_dt = config_->physics.fixed_dt;
  if (fixed_dt <= 0.0) {
    return std::unexpected(make_error(
      ErrorCode::SIM,
      "Physics fixed_dt must be positive",
      make_context("Simulator::integrate_until")
    ));
  }

  const auto& rl_cfg = rl_config_->reward;

  while (sim_time_s_ + core::math::kEps < target_time) {
    const double remaining = target_time - sim_time_s_;
    const double dt        = std::min(fixed_dt, remaining);

    if (auto update_res = world_->update(dt); !update_res) {
      return std::unexpected(make_error_from(
        update_res.error(), ErrorCode::SIM, make_context("Simulator::integrate_until")
      ));
    }

    sim_time_s_ += dt;

    // Check for collision events and update tracking state
    if (auto collision = world_->consume_collision_report()) {
      // Track the worst collision severity across the decision step
      if (step_collision_contacts_.empty() ||
          collision->severity.severity_level > step_collision_severity_.severity_level) {
        step_collision_severity_ = collision->severity;
      }
      step_collision_contacts_ = std::move(collision->contacts);

      const auto& severity = collision->severity;

      // Severe collisions terminate immediately
      if (severity.severity_level >= core::sim::CollisionSeverity::Level::SEVERE) {
        episode_terminal_ = true;
      }

      // Track collision dwell time for timeout logic
      if (severity.is_active()) {
        collision_dwell_time_ += dt;
        if (collision_timeout_s_ > 0.0 && collision_dwell_time_ >= collision_timeout_s_) {
          DDRL_LOG_DEBUG_THROTTLE(
            logger_,
            3000,
            "[Sim {}] Collision timeout exceeded ({:.2f}s >= {:.2f}s); terminating episode",
            sim_id_,
            collision_dwell_time_,
            collision_timeout_s_
          );
          episode_terminal_ = true;
        }
      } else {
        collision_dwell_time_ = 0.0;
      }
    } else {
      collision_dwell_time_ = 0.0;
    }

    // Compute reward for this physics timestep
    if (reward_fn_) {
      auto telemetry = world_->reward_telemetry();
      telemetry.episode_step_count = static_cast<uint32_t>(episode_step_count_);
      if (config_->episode.max_sim_seconds > 0.0) {
        const double max_s = config_->episode.max_sim_seconds;
        telemetry.time_limit_reached = (sim_time_s_ + dt >= max_s - core::math::kEps);
      }

      if (telemetry.waypoint_reached) {
        step_waypoint_reached_ = true;
      }

      // Check for final-goal based termination
      if (telemetry.final_goal_reached) {
        step_goal_reached_ = true;
        if (rl_cfg.terminate_on_goal) {
          episode_terminal_ = true;
        }
      }

      auto reward_result = reward_fn_->compute(
        telemetry, dt, step_collision_severity_, step_collision_severity_.is_active()
      );
      accumulated_reward_ += reward_result.total;
      for (auto& [name, value] : reward_result.terms) {
        auto& total = accumulated_reward_terms_[name];
        total.raw += value.raw;
        total.weighted += value.weighted;
      }

      if (!episode_terminal_ && reward_fn_->should_terminate()) {
        if (auto* default_reward = dynamic_cast<core::rl::DefaultRewardFunction*>(reward_fn_.get())) {
          DDRL_LOG_INFO_THROTTLE(
            logger_,
            3000,
            "[Sim {}] Terminating episode due to stall limit (ema {:.3f} / limit {:.3f}).",
            sim_id_,
            default_reward->stall_ema(),
            rl_cfg.weights.stall_limit
          );
        } else {
          DDRL_LOG_INFO_THROTTLE(
            logger_,
            3000,
            "[Sim {}] Terminating episode due to reward function stall/termination hint.",
            sim_id_
          );
        }
        episode_terminal_ = true;
      }
    }

    if (auto publish_res = publish_world_snapshot(sim_time_s_, false); !publish_res) {
      return publish_res;
    }
  }

  return {};
}

core::Result<void> Simulator::Impl::publish_world_snapshot(double current_time, bool force_publish)
{
  if (!visualization_enabled_) {
    return {};
  }

  if (!event_bus_ || !world_) {
    return {};
  }

  if (!force_publish && current_time + core::math::kEps < next_viz_publish_time_) {
    return {};
  }

  auto snapshot = world_->capture_snapshot();
  if (!snapshot) {
    return {};
  }

  events::WorldStateEvent event{sim_id_, current_time, snapshot};
  auto                    publish_res = event_bus_->publish(sim_id_, std::move(event));
  if (!publish_res) {
    return std::unexpected(make_error_from(
      publish_res.error(), ErrorCode::SIM, make_context("Simulator::publish_world_snapshot")
    ));
  }

  next_viz_publish_time_ = current_time + viz_publish_period_;
  return {};
}

void Simulator::Impl::log_step_timing(double last_step_seconds)
{
  step_time_window_accum_s_ += last_step_seconds;
  ++step_time_window_samples_;

  if (step_time_window_samples_ < kStepTimingLogInterval) {
    return;
  }

  const double avg_ms =
    (step_time_window_accum_s_ / static_cast<double>(step_time_window_samples_)) * 1000.0;
  const double last_ms = last_step_seconds * 1000.0;

  DDRL_LOG_DEBUG_THROTTLE(
    logger_,
    5000,
    "[Sim {}] Step timing: avg {:.3f} ms over {} steps (last {:.3f} ms)",
    sim_id_,
    avg_ms,
    step_time_window_samples_,
    last_ms
  );

  step_time_window_accum_s_ = 0.0;
  step_time_window_samples_ = 0;
}

void Simulator::Impl::reset_episode_counters()
{
  if (initialized_) {
    ++episode_index_;
  } else {
    initialized_ = true;
  }

  episode_return_     = 0.0;
  episode_step_count_ = 0;
}

void Simulator::Impl::record_episode_outcome(bool success)
{
  ++episodes_in_phase_;
  if (success) {
    ++successes_in_phase_;
  }

  // Maintain rolling window
  const auto* phase = current_phase();
  size_t      window =
    ((phase != nullptr) && phase->success_rate_window > 0) ? phase->success_rate_window : size_t(0);
  if (window > 0) {
    success_window_.push_back(success);
    while (success_window_.size() > window) {
      success_window_.pop_front();
    }
  } else {
    success_window_.clear();
  }
}

void Simulator::Impl::maybe_advance_curriculum()
{
  const auto& curr = config_->curriculum;
  if (!curr.enabled || curr.phases.empty()) {
    return;
  }
  if (current_phase_idx_ + 1 >= curr.phases.size()) {
    return;
  }
  const auto& phase = curr.phases[current_phase_idx_];

  if (phase.promote_on_success_rate && phase.success_rate_window > 0 &&
      success_window_.size() >= phase.success_rate_window) {
    const double wins =
      static_cast<double>(std::count(success_window_.begin(), success_window_.end(), true));
    const double rate = wins / static_cast<double>(success_window_.size());
    if (rate >= phase.promote_on_success_rate.value()) {
      ++current_phase_idx_;
      episodes_in_phase_  = 0;
      successes_in_phase_ = 0;
      success_window_.clear();

      DDRL_LOG_INFO(
        logger_,
        "[Sim {}] Advancing curriculum to phase {} ({})",
        sim_id_,
        current_phase_idx_,
        curr.phases[current_phase_idx_].name
      );
    }
  }
}

const config::SimConfig::CurriculumPhase* Simulator::Impl::current_phase() const
{
  const auto& curr = config_->curriculum;
  if (!curr.enabled || curr.phases.empty() || current_phase_idx_ >= curr.phases.size()) {
    return nullptr;
  }
  return &curr.phases[current_phase_idx_];
}

std::shared_ptr<const config::MapConfig>
Simulator::Impl::pick_phase_map(const config::SimConfig::CurriculumPhase* phase)
{
  if (phase == nullptr) {
    DDRL_LOG_ERROR(logger_, "[Sim {}] No curriculum phase", sim_id_);
    return {};
  }
  if (phase->maps.empty()) {
    DDRL_LOG_ERROR(
      logger_, "[Sim {}] Curriculum phase '{}' has no maps defined", sim_id_, phase->name
    );
    return {};
  }

  const size_t idx    = std::uniform_int_distribution<size_t>(0, phase->maps.size() - 1)(rng_);
  const auto&  map_id = phase->maps[idx];

  return map_manager_->get_map_by_name(map_id);

  return {};
}

std::shared_ptr<const config::SpawnManagerConfig>
Simulator::Impl::build_spawn_config(const config::SimConfig::CurriculumPhase* phase)
{
  auto cfg = std::make_shared<config::SpawnManagerConfig>(config_->spawn_manager);
  // Derive perception once from robot profile
  const auto& robot_profile_id = world_config_->robot.robot_profile_id;
  if (auto it = world_config_->robot_profiles.find(robot_profile_id);
      it != world_config_->robot_profiles.end()) {
    derive_perception_from_profile(it->second, *cfg);
  } else {
    DDRL_LOG_WARN(
      logger_,
      "[Sim {}] Robot profile '{}' not found when deriving perception; using defaults",
      sim_id_,
      robot_profile_id
    );
  }
  if (phase == nullptr) {
    return cfg;
  }

  const auto& sp = phase->spawn;
  if (sp.min_spawn_distance) {
    cfg->min_spawn_distance = *sp.min_spawn_distance;
  }
  if (sp.max_spawn_distance) {
    cfg->max_spawn_distance = *sp.max_spawn_distance;
  }
  if (sp.min_active_objects) {
    cfg->min_active_objects = *sp.min_active_objects;
  }
  if (sp.max_active_objects) {
    cfg->max_active_objects = *sp.max_active_objects;
  }

  return cfg;
}

void Simulator::Impl::derive_perception_from_profile(
  const config::RobotProfile& profile, config::SpawnManagerConfig& spawn_cfg
)
{
  double max_range = 0.0;
  double max_hfov  = 0.0;

  for (const auto& sensor_cfg : profile.sensors) {
    std::visit(
      core::Overloaded{
        [&](const core::sensors::Lidar3D& lidar) {
          max_range = std::max<double>(max_range, static_cast<double>(lidar.base.max_range));
          max_hfov  = std::max<double>(max_hfov, lidar.hfov);
        },
        [&](const core::sensors::Lidar2D& lidar) {
          max_range = std::max<double>(max_range, static_cast<double>(lidar.base.max_range));
          max_hfov  = std::max<double>(max_hfov, lidar.fov);
        },
        [&](const core::sensors::DepthCamera& cam) {
          max_range     = std::max<double>(max_range, static_cast<double>(cam.base.max_range));
          const auto fx = static_cast<double>(cam.fx);
          const auto w  = static_cast<double>(cam.width);
          if (fx > 1e-6) {
            const double hfov = 2.0 * std::atan(w / (2.0 * fx));
            max_hfov          = std::max<double>(max_hfov, hfov);
          }
        }
      },
      sensor_cfg
    );
  }

  if (max_range > 1e-3) {
    spawn_cfg.fov_distance     = max_range;
    spawn_cfg.max_sensor_range = max_range;
  }
  if (max_hfov > 1e-3) {
    spawn_cfg.fov_angle_deg = std::clamp(max_hfov * 180.0 / std::numbers::pi, 0.0, 360.0);
  }
}

// =============================================================================
// Simulator Public APIs
// =============================================================================

Simulator::Simulator(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

Simulator::~Simulator() = default;

Simulator::Simulator(Simulator&&) noexcept            = default;
Simulator& Simulator::operator=(Simulator&&) noexcept = default;

core::Result<Simulator::ResetResult> Simulator::reset(ResetParams params)
{
  return impl_->reset(params);
}

core::Result<Simulator::StepResult> Simulator::step(StepRequest request)
{
  return impl_->step(request);
}

} // namespace ddrl::sim
