/**
 * @file parsing.cpp
 * @brief Implementation of configuration parsing utilities
 */

#include "config/parsing.hpp"

#include "core/base/result.hpp"
#include "core/base/string_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

#include "logging/logging.hpp"

namespace ddrl::config {

using core::Error;
using core::ErrorCode;
using core::make_context;
using core::make_error;

namespace fs = std::filesystem;

namespace {

// Logger for config parsing
auto g_logger = logging::get_logger("config.parsing");

/**
 * @brief Expand glob patterns (e.g., "config/\*.yaml") to list of matching file paths
 * @param pattern The glob pattern to expand
 * @return Vector of expanded file paths, or empty vector if no matches or pattern is invalid
 */
std::vector<std::string> expand_glob_pattern(std::string_view pattern)
{
  namespace fs = std::filesystem;
  std::vector<std::string> result;

  // Check if pattern contains wildcard
  if (pattern.find('*') == std::string::npos && pattern.find('?') == std::string::npos) {
    // No wildcard, return as-is
    result.emplace_back(pattern);
    return result;
  }

  // Split pattern into directory and filename parts
  fs::path    pattern_path(pattern);
  fs::path    dir_path         = pattern_path.parent_path();
  std::string filename_pattern = pattern_path.filename().string();

  // If no directory specified, use current directory
  if (dir_path.empty()) {
    dir_path = ".";
  }

  // Check if directory exists
  if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
    DDRL_LOG_WARN(
      g_logger,
      "Glob pattern directory does not exist: '{}' in pattern '{}'",
      dir_path.string(),
      pattern
    );
    return result;
  }

  // Convert glob pattern to regex
  auto glob_to_regex = [](const std::string& glob) -> std::string {
    std::string regex;
    regex.reserve(glob.size() * 2);
    regex += "^";
    for (char ch : glob) {
      switch (ch) {
        case '*':
          regex += ".*";
          break;
        case '?':
          regex += ".";
          break;
        case '.':
          regex += "\\.";
          break;
        case '+':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
        case '^':
        case '$':
        case '|':
        case '\\':
          regex += '\\';
          regex += ch;
          break;
        default:
          regex += ch;
      }
    }
    regex += "$";
    return regex;
  };

  std::string regex_pattern = glob_to_regex(filename_pattern);
  std::regex  filename_regex(regex_pattern, std::regex::icase);

  // Iterate through directory and match files
  try {
    for (const auto& entry : fs::directory_iterator(dir_path)) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, filename_regex)) {
          result.push_back(entry.path().string());
        }
      }
    }
  } catch (const fs::filesystem_error& e) {
    DDRL_LOG_ERROR(g_logger, "Filesystem error while expanding glob '{}': {}", pattern, e.what());
    return result;
  }

  // Sort for deterministic order
  std::ranges::sort(result);

  DDRL_LOG_DEBUG(g_logger, "Expanded glob pattern '{}' to {} file(s)", pattern, result.size());

  return result;
}

core::math::Pose2 parse_pose2(const YAML::Node& node)
{
  core::math::Pose2 pose;
  if (node["pos"]) {
    const auto& p = node["pos"];
    pose.pos.x    = p[0].as<double>();
    pose.pos.y    = p[1].as<double>();
  }
  if (node["yaw"]) {
    pose.yaw = node["yaw"].as<double>();
  }
  return pose;
}

core::math::Point2 parse_point2(const YAML::Node& node)
{
  core::math::Point2 point;
  point.x = node[0].as<double>();
  point.y = node[1].as<double>();
  return point;
}

core::geom::Polygon2 parse_polygon2(const YAML::Node& node)
{
  core::geom::Polygon2 poly;
  for (const auto& pt : node) {
    poly.v.emplace_back(pt[0].as<double>(), pt[1].as<double>());
  }
  return poly;
}

// ============================================================================
// NameGenerator - Handles unique name generation with suffix management
// ============================================================================

class NameGenerator
{
public:
  NameGenerator(std::string context, std::string fallback)
      : context_(std::move(context)), fallback_(std::move(fallback))
  {
  }

  std::string operator()(const std::string& desired)
  {
    std::string base   = desired.empty() ? fallback_ : desired;
    auto [_, inserted] = used_.emplace(base);

    if (inserted) {
      suffix_.try_emplace(base, 1);
      return base;
    }

    auto& counter = suffix_[base];
    if (counter == 0) {
      counter = 1;
    }

    std::string candidate;
    do {
      candidate = std::format("{}-{}", base, counter++);
    } while (!used_.emplace(candidate).second);

    suffix_.try_emplace(candidate, 1);
    DDRL_LOG_WARN(
      g_logger, "Duplicate {} name '{}' detected, renamed to '{}'.", context_, base, candidate
    );
    return candidate;
  }

private:
  std::string                                    context_;
  std::string                                    fallback_;
  std::unordered_set<std::string>                used_;
  std::unordered_map<std::string, std::uint32_t> suffix_;
};

} // anonymous namespace

// ============================================================================
// ConfigParser::Impl - Hidden implementation class
// ============================================================================

class ConfigParser::Impl
{
private:
  NameGenerator map_name_gen_{"map", "default_map"};

public:
  // ============================================================================
  // Simulator Config
  // ============================================================================

  [[nodiscard]] core::Result<SimConfig> parse_simulator_config(std::string_view path) const
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));
      SimConfig  cfg;

      // Physics
      if (yaml["physics"]) {
        const auto& p                     = yaml["physics"];
        cfg.physics.fixed_dt              = p["fixed_dt"].as<double>(0.005);
        cfg.physics.max_substeps_per_step = p["max_substeps_per_step"].as<size_t>(8);
      }

      // Decision
      if (yaml["decision"]) {
        const auto& d                  = yaml["decision"];
        cfg.decision.decision_dt       = d["decision_dt"].as<double>(0.05);
        cfg.decision.action_delay_s    = d["action_delay_s"].as<double>(0.0);
        cfg.decision.first_order_tau_s = d["first_order_tau_s"].as<double>(0.0);
      }

      // Episode
      if (yaml["episode"]) {
        const auto& e                   = yaml["episode"];
        cfg.episode.max_decision_steps  = e["max_decision_steps"].as<size_t>(10000);
        cfg.episode.max_sim_seconds     = e["max_sim_seconds"].as<double>(0.0);
        cfg.episode.collision_timeout_s = e["collision_timeout_s"].as<double>(3.0);
      }

      // RNG seed
      cfg.rng_seed = yaml["rng_seed"].as<uint64_t>(1234567);

      const auto& spawn_node = yaml["spawn_manager"];
      if (!spawn_node || !spawn_node.IsMap()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "Missing or invalid 'spawn_manager' section in simulator config",
          make_context("ConfigParser::parse_simulator_config")
        ));
      }

      auto& sm = cfg.spawn_manager;

      if (spawn_node["min_spawn_distance"]) {
        sm.min_spawn_distance = spawn_node["min_spawn_distance"].as<double>();
      }
      if (spawn_node["max_spawn_distance"]) {
        sm.max_spawn_distance = spawn_node["max_spawn_distance"].as<double>();
      }
      if (spawn_node["min_inter_object_distance"]) {
        sm.min_inter_object_distance = spawn_node["min_inter_object_distance"].as<double>();
      }
      if (spawn_node["avoid_fov"]) {
        sm.avoid_fov = spawn_node["avoid_fov"].as<bool>();
      }
      if (const auto& bias_node = spawn_node["angular_bias"]; bias_node && bias_node.IsMap()) {
        if (bias_node["front_angle_deg"]) {
          sm.angular_bias.front_angle_deg = bias_node["front_angle_deg"].as<double>();
        }
        if (bias_node["flank_angle_deg"]) {
          sm.angular_bias.flank_angle_deg = bias_node["flank_angle_deg"].as<double>();
        }
        if (bias_node["rear_angle_deg"]) {
          sm.angular_bias.rear_angle_deg = bias_node["rear_angle_deg"].as<double>();
        }
        if (bias_node["front_weight"]) {
          sm.angular_bias.front_weight = bias_node["front_weight"].as<double>();
        }
        if (bias_node["flank_weight"]) {
          sm.angular_bias.flank_weight = bias_node["flank_weight"].as<double>();
        }
        if (bias_node["rear_weight"]) {
          sm.angular_bias.rear_weight = bias_node["rear_weight"].as<double>();
        }
      }
      if (const auto& dist_node = spawn_node["distance_bias"]; dist_node && dist_node.IsMap()) {
        if (dist_node["near_distance"]) {
          sm.distance_bias.near_distance = dist_node["near_distance"].as<double>();
        }
        if (dist_node["far_distance"]) {
          sm.distance_bias.far_distance = dist_node["far_distance"].as<double>();
        }
        if (dist_node["near_weight_slow"]) {
          sm.distance_bias.near_weight_slow = dist_node["near_weight_slow"].as<double>();
        }
        if (dist_node["near_weight_fast"]) {
          sm.distance_bias.near_weight_fast = dist_node["near_weight_fast"].as<double>();
        }
        if (dist_node["far_weight_slow"]) {
          sm.distance_bias.far_weight_slow = dist_node["far_weight_slow"].as<double>();
        }
        if (dist_node["far_weight_fast"]) {
          sm.distance_bias.far_weight_fast = dist_node["far_weight_fast"].as<double>();
        }
        if (dist_node["slow_speed_mps"]) {
          sm.distance_bias.slow_speed_mps = dist_node["slow_speed_mps"].as<double>();
        }
        if (dist_node["fast_speed_mps"]) {
          sm.distance_bias.fast_speed_mps = dist_node["fast_speed_mps"].as<double>();
        }
      }
      if (const auto& occ_node = spawn_node["occlusion"]; occ_node && occ_node.IsMap()) {
        if (occ_node["allow_in_fov"]) {
          sm.occlusion.allow_in_fov = occ_node["allow_in_fov"].as<bool>();
        }
        if (occ_node["static_blocker_radius"]) {
          sm.occlusion.static_blocker_radius = occ_node["static_blocker_radius"].as<double>();
        }
        if (occ_node["dynamic_blocker_radius"]) {
          sm.occlusion.dynamic_blocker_radius = occ_node["dynamic_blocker_radius"].as<double>();
        }
      }
      if (spawn_node["respect_map_geometry"]) {
        sm.respect_map_geometry = spawn_node["respect_map_geometry"].as<bool>();
      }
      if (spawn_node["max_spawns_per_frame"]) {
        sm.max_spawns_per_frame = spawn_node["max_spawns_per_frame"].as<size_t>();
      }
      if (spawn_node["max_spawn_attempts_per_frame"]) {
        sm.max_spawn_attempts_per_frame = spawn_node["max_spawn_attempts_per_frame"].as<size_t>();
      }
      if (spawn_node["min_active_objects"]) {
        sm.min_active_objects = spawn_node["min_active_objects"].as<size_t>();
      }
      if (spawn_node["max_active_objects"]) {
        sm.max_active_objects = spawn_node["max_active_objects"].as<size_t>();
      }
      if (spawn_node["max_distance_from_robot"]) {
        sm.max_despawn_distance = spawn_node["max_distance_from_robot"].as<double>();
      }
      if (spawn_node["stuck_timeout"]) {
        sm.stuck_timeout = spawn_node["stuck_timeout"].as<double>();
      }
      if (spawn_node["fov_angle_deg"]) {
        sm.fov_angle_deg = spawn_node["fov_angle_deg"].as<double>();
      }
      if (spawn_node["fov_distance"]) {
        sm.fov_distance = spawn_node["fov_distance"].as<double>();
      }
      if (spawn_node["max_sensor_range"]) {
        sm.max_sensor_range = spawn_node["max_sensor_range"].as<double>();
      }
      if (spawn_node["max_position_samples"]) {
        sm.max_position_samples = spawn_node["max_position_samples"].as<size_t>();
      }

      const auto& curr_node = spawn_node["curriculum"];
      if (!curr_node && !curr_node.IsMap()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "Missing or invalid 'curriculum' section in spawn manager config",
          make_context("ConfigParser::parse_simulator_config")
        ));
      }

      cfg.curriculum.enabled = curr_node["enabled"].as<bool>(false);
      if (cfg.curriculum.enabled && (!curr_node["phases"] || !curr_node["phases"].IsSequence())) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "Curriculum enabled but 'phases' is missing or not a list",
          "parse_simulator_config"
        ));
      }
      if (const auto& phases = curr_node["phases"]; phases && phases.IsSequence()) {
        if (cfg.curriculum.enabled && phases.size() == 0) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG, "Curriculum enabled but no phases provided", "parse_simulator_config"
          ));
        }
        for (size_t i = 0; i < phases.size(); ++i) {
          const auto& phase_node = phases[i];
          if (!phase_node.IsMap()) {
            continue;
          }
          SimConfig::CurriculumPhase phase;
          phase.name = phase_node["name"].as<std::string>(std::format("phase_{}", i));
          if (const auto& maps = phase_node["maps"]; maps && maps.IsSequence()) {
            for (const auto& m : maps) {
              phase.maps.push_back(m.as<std::string>());
            }
          }
          if (phase_node["promote_on_success_rate"]) {
            phase.promote_on_success_rate = phase_node["promote_on_success_rate"].as<double>();
          }
          if (phase_node["success_rate_window"]) {
            phase.success_rate_window = phase_node["success_rate_window"].as<size_t>(0);
          }
          if (phase_node["maps"]) {
            phase.maps = phase_node["maps"].as<std::vector<std::string>>();
          } else if (cfg.curriculum.enabled) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("Curriculum phase '{}' is missing 'maps' list", phase.name),
              "parse_simulator_config"
            ));
          }
          if (cfg.curriculum.enabled && phase.maps.empty()) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("Curriculum phase '{}' has an empty 'maps' list", phase.name),
              "parse_simulator_config"
            ));
          }
          if (const auto& spawn = phase_node["spawn"]; spawn && spawn.IsMap()) {
            auto& sp = phase.spawn;
            if (spawn["min_spawn_distance"]) {
              sp.min_spawn_distance = spawn["min_spawn_distance"].as<double>();
            }
            if (spawn["max_spawn_distance"]) {
              sp.max_spawn_distance = spawn["max_spawn_distance"].as<double>();
            }
            if (spawn["min_active_objects"]) {
              sp.min_active_objects = spawn["min_active_objects"].as<size_t>();
            }
            if (spawn["max_active_objects"]) {
              sp.max_active_objects = spawn["max_active_objects"].as<size_t>();
            }
          }
          cfg.curriculum.phases.push_back(std::move(phase));
        }
      }

      log_simulator_config(cfg, path);
      return cfg;
    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_simulator_config"));
    } catch (const std::exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_simulator_config"));
    }
  }

  [[nodiscard]] core::Result<RLConfig> parse_rl_config(std::string_view path) const
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));
      RLConfig   cfg;

      const auto policy = yaml["policy"];
      if (!policy) {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Missing 'policy' section", "parse_rl_config")
        );
      }

      const auto type = policy["type"].as<std::string>("random");
      if (type == "random") {
        cfg.policy.type = RLPolicyConfig::Type::RANDOM;
      } else if (type == "neural_network") {
        cfg.policy.type = RLPolicyConfig::Type::NEURAL_NETWORK;
      } else if (type == "expert") {
        cfg.policy.type = RLPolicyConfig::Type::EXPERT;
      } else {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Unknown policy.type: " + type, "parse_rl_config")
        );
      }

      const auto arch = policy["architecture"];
      if (!arch) {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Missing policy.architecture", "parse_rl_config")
        );
      }
      cfg.policy.architecture.policy_model =
        arch["policy_model"].as<std::string>(cfg.policy.architecture.policy_model);
      cfg.policy.architecture.azimuth_bins =
        arch["azimuth_bins"].as<uint32_t>(cfg.policy.architecture.azimuth_bins);
      cfg.policy.architecture.range_max_m =
        arch["range_max_m"].as<double>(cfg.policy.architecture.range_max_m);
      cfg.policy.architecture.state_dim =
        arch["state_dim"].as<uint32_t>(cfg.policy.architecture.state_dim);
      cfg.policy.architecture.action_dim =
        arch["action_dim"].as<uint32_t>(cfg.policy.architecture.action_dim);
      cfg.policy.architecture.mlp_hidden_dim =
        arch["mlp_hidden_dim"].as<uint32_t>(cfg.policy.architecture.mlp_hidden_dim);
      cfg.policy.architecture.lstm_hidden_dim =
        arch["lstm_hidden_dim"].as<uint32_t>(cfg.policy.architecture.lstm_hidden_dim);
      if (const auto z_bands = arch["z_bands"]) {
        cfg.policy.architecture.z_bands.clear();
        for (const auto& entry : z_bands) {
          if (!entry.IsSequence() || entry.size() != 2) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              "policy.architecture.z_bands entries must be [z_min, z_max] pairs",
              "parse_rl_config"
            ));
          }
          cfg.policy.architecture.z_bands.emplace_back(
            entry[0].as<double>(), entry[1].as<double>()
          );
        }
      }
      if (cfg.policy.architecture.z_bands.empty()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "policy.architecture.z_bands must not be empty",
          "parse_rl_config"
        ));
      }

      const auto reward = yaml["reward"];
      if (!reward) {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Missing 'reward' section", "parse_rl_config")
        );
      }

      cfg.reward.reward_type = reward["reward_type"].as<std::string>(cfg.reward.reward_type);
      cfg.reward.terminate_on_goal =
        reward["terminate_on_goal"].as<bool>(cfg.reward.terminate_on_goal);

      if (const auto weights = reward["weights"]) {
        auto& w             = cfg.reward.weights;
        w.progress = weights["progress"].as<double>(w.progress);
        w.progress_min = weights["progress_min"].as<double>(w.progress_min);
        w.progress_decay_steps =
          weights["progress_decay_steps"].as<double>(w.progress_decay_steps);
        w.progress_norm_speed_m_s =
          weights["progress_norm_speed_m_s"].as<double>(w.progress_norm_speed_m_s);
        w.progress_gate_epsilon_m_s =
          weights["progress_gate_epsilon_m_s"].as<double>(w.progress_gate_epsilon_m_s);
        w.progress_steer_gate_min =
          weights["progress_steer_gate_min"].as<double>(w.progress_steer_gate_min);
        w.progress_steer_gate_power =
          weights["progress_steer_gate_power"].as<double>(w.progress_steer_gate_power);
        w.steer_norm_abs = weights["steer_norm_abs"].as<double>(w.steer_norm_abs);
        w.effort_penalty = weights["effort_penalty"].as<double>(w.effort_penalty);
        w.time_penalty        = weights["time_penalty"].as<double>(w.time_penalty);
        w.waypoint_reached_bonus =
          weights["waypoint_reached_bonus"].as<double>(w.waypoint_reached_bonus);
        w.goal_reached_bonus  = weights["goal_reached_bonus"].as<double>(w.goal_reached_bonus);
        w.goal_reached_bonus_min =
          weights["goal_reached_bonus_min"].as<double>(w.goal_reached_bonus_min);
        w.goal_reached_bonus_decay_steps =
          weights["goal_reached_bonus_decay_steps"].as<double>(w.goal_reached_bonus_decay_steps);
        w.yaw_rate_penalty    = weights["yaw_rate_penalty"].as<double>(w.yaw_rate_penalty);
        w.reverse_penalty     = weights["reverse_penalty"].as<double>(w.reverse_penalty);
        w.reverse_penalty_tau_s =
          weights["reverse_penalty_tau_s"].as<double>(w.reverse_penalty_tau_s);
        w.reverse_speed_penalty =
          weights["reverse_speed_penalty"].as<double>(w.reverse_speed_penalty);
        w.gear_switch_penalty =
          weights["gear_switch_penalty"].as<double>(w.gear_switch_penalty);
        w.heading_alignment_bonus =
          weights["heading_alignment_bonus"].as<double>(w.heading_alignment_bonus);
        w.heading_to_goal_reward =
          weights["heading_to_goal_reward"].as<double>(w.heading_to_goal_reward);
        w.proximity_reward =
          weights["proximity_reward"].as<double>(w.proximity_reward);
        if (weights["heading_to_goal_reward"]) {
          w.heading_alignment_bonus = w.heading_to_goal_reward;
        }
        w.time_penalty_scale =
          weights["time_penalty_scale"].as<double>(w.time_penalty_scale);
        w.time_penalty_scale_decay_steps =
          weights["time_penalty_scale_decay_steps"].as<double>(w.time_penalty_scale_decay_steps);
        w.time_limit_penalty =
          weights["time_limit_penalty"].as<double>(w.time_limit_penalty);
        w.steer_penalty = weights["steer_penalty"].as<double>(w.steer_penalty);
        w.steer_penalty_tau_s =
          weights["steer_penalty_tau_s"].as<double>(w.steer_penalty_tau_s);
        w.steer_hold_penalty =
          weights["steer_hold_penalty"].as<double>(w.steer_hold_penalty);
        w.steer_hold_threshold =
          weights["steer_hold_threshold"].as<double>(w.steer_hold_threshold);
        w.steer_hold_terminate_s =
          weights["steer_hold_terminate_s"].as<double>(w.steer_hold_terminate_s);
        w.steer_hold_terminal_penalty =
          weights["steer_hold_terminal_penalty"].as<double>(w.steer_hold_terminal_penalty);
        w.steer_hold_decay_s =
          weights["steer_hold_decay_s"].as<double>(w.steer_hold_decay_s);
        w.yaw_rate_near_goal_dist =
          weights["yaw_rate_near_goal_dist"].as<double>(w.yaw_rate_near_goal_dist);
        w.reverse_hold_penalty =
          weights["reverse_hold_penalty"].as<double>(w.reverse_hold_penalty);
        if (weights["reverse_speed_penalty"]) {
          w.reverse_penalty =
            weights["reverse_speed_penalty"].as<double>(w.reverse_penalty);
        }
        if (weights["reverse_hold_penalty"]) {
          w.reverse_penalty =
            weights["reverse_hold_penalty"].as<double>(w.reverse_penalty);
        }
        w.collision_penalty_scale =
          weights["collision_penalty_scale"].as<double>(w.collision_penalty_scale);
        w.clearance_gate_dist_min_m =
          weights["clearance_gate_dist_min_m"].as<double>(w.clearance_gate_dist_min_m);
        w.clearance_gate_dist_max_m =
          weights["clearance_gate_dist_max_m"].as<double>(w.clearance_gate_dist_max_m);
        w.clearance_gate_half_angle_rad =
          weights["clearance_gate_half_angle_rad"].as<double>(w.clearance_gate_half_angle_rad);
        w.stall_penalty_scale =
          weights["stall_penalty_scale"].as<double>(w.stall_penalty_scale);
        w.stall_penalty_growth =
          weights["stall_penalty_growth"].as<double>(w.stall_penalty_growth);
        w.stall_progress_epsilon =
          weights["stall_progress_epsilon"].as<double>(w.stall_progress_epsilon);
        w.stall_window_s = weights["stall_window_s"].as<double>(w.stall_window_s);
        w.stall_tau_s    = weights["stall_tau_s"].as<double>(w.stall_tau_s);
        w.stall_limit    = weights["stall_limit"].as<double>(w.stall_limit);
        w.stall_terminal_penalty =
          weights["stall_terminal_penalty"].as<double>(w.stall_terminal_penalty);

        if (weights["stagnation_penalty_scale"]) {
          w.stall_penalty_scale =
            weights["stagnation_penalty_scale"].as<double>(w.stall_penalty_scale);
        }
        if (weights["stagnation_penalty_growth"]) {
          w.stall_penalty_growth =
            weights["stagnation_penalty_growth"].as<double>(w.stall_penalty_growth);
        }
        if (weights["stagnation_progress_epsilon"]) {
          w.stall_progress_epsilon =
            weights["stagnation_progress_epsilon"].as<double>(w.stall_progress_epsilon);
        }
        if (weights["stagnation_window_s"]) {
          w.stall_window_s = weights["stagnation_window_s"].as<double>(w.stall_window_s);
        }
        if (weights["stagnation_ramp_s"]) {
          w.stall_tau_s = weights["stagnation_ramp_s"].as<double>(w.stall_tau_s);
        }
        if (weights["stagnation_patience_s"]) {
          w.stall_tau_s = weights["stagnation_patience_s"].as<double>(w.stall_tau_s);
        }
        if (weights["stagnation_strike_duration_s"]) {
          w.stall_tau_s =
            weights["stagnation_strike_duration_s"].as<double>(w.stall_tau_s);
        }
        if (weights["stagnation_max_strikes"]) {
          const auto max_strikes =
            weights["stagnation_max_strikes"].as<double>(w.stall_limit);
          w.stall_limit = std::max(1.0, max_strikes);
        }
        if (weights["stagnation_terminal_penalty"]) {
          w.stall_terminal_penalty =
            weights["stagnation_terminal_penalty"].as<double>(w.stall_terminal_penalty);
        }
      }

      DDRL_LOG_DEBUG(g_logger, "Parsed RL config from '{}':", path);
      DDRL_LOG_DEBUG(g_logger, "  policy.type: {}", type);
      DDRL_LOG_DEBUG(
        g_logger,
        "  architecture: model={} azimuth_bins={} z_bands={} observation_dim={} state_dim={} action={} "
        "mlp_hidden={} lstm_hidden={} range_max_m={}",
        cfg.policy.architecture.policy_model,
        cfg.policy.architecture.azimuth_bins,
        cfg.policy.architecture.z_bands.size(),
        cfg.policy.architecture.observation_dim(),
        cfg.policy.architecture.state_dim,
        cfg.policy.architecture.action_dim,
        cfg.policy.architecture.mlp_hidden_dim,
        cfg.policy.architecture.lstm_hidden_dim,
        cfg.policy.architecture.range_max_m
      );
      DDRL_LOG_DEBUG(g_logger, "  reward.type: {}", cfg.reward.reward_type);
      DDRL_LOG_DEBUG(g_logger, "  reward.terminate_on_goal: {}", cfg.reward.terminate_on_goal);
      DDRL_LOG_DEBUG(
        g_logger,
        "  weights: prog={} effort_penalty={} time={} waypoint={} goal={} yaw={} "
        "reverse_penalty={} reverse_tau={} gear_switch={} heading_to_goal={} proximity_reward={} "
        "time_scale={} time_limit={} "
        "steer_penalty={} steer_tau={} progress_steer_gate_min={} progress_steer_gate_power={} "
        "steer_hold_penalty={} steer_hold_threshold={} "
        "steer_hold_terminate_s={} "
        "steer_hold_terminal_penalty={} steer_hold_decay_s={} yaw_rate_near_goal_dist={} "
        "time_scale_decay_steps={} "
        "progress_min={} progress_decay_steps={} goal_bonus_min={} goal_bonus_decay_steps={} "
        "collision_penalty_scale={} "
        "clearance_gate_dist_min_m={} clearance_gate_dist_max_m={} clearance_gate_half_angle_rad={} "
        "stall_penalty={} stall_growth={} stall_eps={} stall_window={} stall_tau={} stall_limit={} "
        "stall_terminal_penalty={}",
        cfg.reward.weights.progress,
        cfg.reward.weights.effort_penalty,
        cfg.reward.weights.time_penalty,
        cfg.reward.weights.waypoint_reached_bonus,
        cfg.reward.weights.goal_reached_bonus,
        cfg.reward.weights.yaw_rate_penalty,
        cfg.reward.weights.reverse_penalty,
        cfg.reward.weights.reverse_penalty_tau_s,
        cfg.reward.weights.gear_switch_penalty,
        cfg.reward.weights.heading_to_goal_reward,
        cfg.reward.weights.proximity_reward,
        cfg.reward.weights.time_penalty_scale,
        cfg.reward.weights.time_limit_penalty,
        cfg.reward.weights.steer_penalty,
        cfg.reward.weights.steer_penalty_tau_s,
        cfg.reward.weights.progress_steer_gate_min,
        cfg.reward.weights.progress_steer_gate_power,
        cfg.reward.weights.steer_hold_penalty,
        cfg.reward.weights.steer_hold_threshold,
        cfg.reward.weights.steer_hold_terminate_s,
        cfg.reward.weights.steer_hold_terminal_penalty,
        cfg.reward.weights.steer_hold_decay_s,
        cfg.reward.weights.yaw_rate_near_goal_dist,
        cfg.reward.weights.time_penalty_scale_decay_steps,
        cfg.reward.weights.progress_min,
        cfg.reward.weights.progress_decay_steps,
        cfg.reward.weights.goal_reached_bonus_min,
        cfg.reward.weights.goal_reached_bonus_decay_steps,
        cfg.reward.weights.collision_penalty_scale,
        cfg.reward.weights.clearance_gate_dist_min_m,
        cfg.reward.weights.clearance_gate_dist_max_m,
        cfg.reward.weights.clearance_gate_half_angle_rad,
        cfg.reward.weights.stall_penalty_scale,
        cfg.reward.weights.stall_penalty_growth,
        cfg.reward.weights.stall_progress_epsilon,
        cfg.reward.weights.stall_window_s,
        cfg.reward.weights.stall_tau_s,
        cfg.reward.weights.stall_limit,
        cfg.reward.weights.stall_terminal_penalty
      );

      return cfg;
    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_rl_config"));
    }
  }

  [[nodiscard]] core::Result<RLPolicyConfig::Architecture> parse_policy_arch_config(
    std::string_view path
  ) const
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));
      auto       arch = yaml["architecture"];
      if (!arch && yaml["policy"]) {
        arch = yaml["policy"]["architecture"];
      }
      if (!arch) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "Missing 'architecture' or 'policy.architecture' section",
          "parse_policy_arch_config"
        ));
      }

      RLPolicyConfig::Architecture cfg;
      cfg.policy_model = arch["policy_model"].as<std::string>(cfg.policy_model);
      cfg.azimuth_bins = arch["azimuth_bins"].as<uint32_t>(cfg.azimuth_bins);
      cfg.range_max_m  = arch["range_max_m"].as<double>(cfg.range_max_m);
      cfg.state_dim    = arch["state_dim"].as<uint32_t>(cfg.state_dim);
      cfg.action_dim   = arch["action_dim"].as<uint32_t>(cfg.action_dim);
      cfg.mlp_hidden_dim =
        arch["mlp_hidden_dim"].as<uint32_t>(cfg.mlp_hidden_dim);
      cfg.lstm_hidden_dim =
        arch["lstm_hidden_dim"].as<uint32_t>(cfg.lstm_hidden_dim);
      if (const auto z_bands = arch["z_bands"]) {
        cfg.z_bands.clear();
        for (const auto& entry : z_bands) {
          if (!entry.IsSequence() || entry.size() != 2) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              "architecture.z_bands entries must be [z_min, z_max] pairs",
              "parse_policy_arch_config"
            ));
          }
          cfg.z_bands.emplace_back(entry[0].as<double>(), entry[1].as<double>());
        }
      }
      if (cfg.z_bands.empty()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          "architecture.z_bands must not be empty",
          "parse_policy_arch_config"
        ));
      }

      DDRL_LOG_DEBUG(
        g_logger,
        "Parsed policy architecture config from '{}': model={} azimuth_bins={} z_bands={} "
        "observation_dim={} state_dim={} action={} mlp_hidden={} lstm_hidden={} range_max_m={}",
        path,
        cfg.policy_model,
        cfg.azimuth_bins,
        cfg.z_bands.size(),
        cfg.observation_dim(),
        cfg.state_dim,
        cfg.action_dim,
        cfg.mlp_hidden_dim,
        cfg.lstm_hidden_dim,
        cfg.range_max_m
      );
      return cfg;
    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_policy_arch_config"));
    }
  }

  [[nodiscard]] core::Result<RLRewardConfig> parse_reward_config(std::string_view path) const
  {
    try {
      YAML::Node yaml   = YAML::LoadFile(std::string(path));
      auto       reward = yaml["reward"] ? yaml["reward"] : yaml;

      RLRewardConfig cfg;
      cfg.reward_type       = reward["reward_type"].as<std::string>(cfg.reward_type);
      cfg.terminate_on_goal = reward["terminate_on_goal"].as<bool>(cfg.terminate_on_goal);

      if (const auto weights = reward["weights"]) {
        auto& w = cfg.weights;
        w.progress = weights["progress"].as<double>(w.progress);
        w.progress_min = weights["progress_min"].as<double>(w.progress_min);
        w.progress_decay_steps =
          weights["progress_decay_steps"].as<double>(w.progress_decay_steps);
        w.progress_norm_speed_m_s =
          weights["progress_norm_speed_m_s"].as<double>(w.progress_norm_speed_m_s);
        w.progress_gate_epsilon_m_s =
          weights["progress_gate_epsilon_m_s"].as<double>(w.progress_gate_epsilon_m_s);
        w.progress_steer_gate_min =
          weights["progress_steer_gate_min"].as<double>(w.progress_steer_gate_min);
        w.progress_steer_gate_power =
          weights["progress_steer_gate_power"].as<double>(w.progress_steer_gate_power);
        w.steer_norm_abs = weights["steer_norm_abs"].as<double>(w.steer_norm_abs);
        w.effort_penalty = weights["effort_penalty"].as<double>(w.effort_penalty);
        w.time_penalty   = weights["time_penalty"].as<double>(w.time_penalty);
        w.waypoint_reached_bonus =
          weights["waypoint_reached_bonus"].as<double>(w.waypoint_reached_bonus);
        w.goal_reached_bonus = weights["goal_reached_bonus"].as<double>(w.goal_reached_bonus);
        w.goal_reached_bonus_min =
          weights["goal_reached_bonus_min"].as<double>(w.goal_reached_bonus_min);
        w.goal_reached_bonus_decay_steps =
          weights["goal_reached_bonus_decay_steps"].as<double>(w.goal_reached_bonus_decay_steps);
        w.yaw_rate_penalty = weights["yaw_rate_penalty"].as<double>(w.yaw_rate_penalty);
        w.reverse_penalty  = weights["reverse_penalty"].as<double>(w.reverse_penalty);
        w.reverse_penalty_tau_s =
          weights["reverse_penalty_tau_s"].as<double>(w.reverse_penalty_tau_s);
        w.reverse_speed_penalty =
          weights["reverse_speed_penalty"].as<double>(w.reverse_speed_penalty);
        w.gear_switch_penalty =
          weights["gear_switch_penalty"].as<double>(w.gear_switch_penalty);
        w.heading_alignment_bonus =
          weights["heading_alignment_bonus"].as<double>(w.heading_alignment_bonus);
        w.heading_to_goal_reward =
          weights["heading_to_goal_reward"].as<double>(w.heading_to_goal_reward);
        w.proximity_reward = weights["proximity_reward"].as<double>(w.proximity_reward);
        if (weights["heading_to_goal_reward"]) {
          w.heading_alignment_bonus = w.heading_to_goal_reward;
        }
        w.time_penalty_scale = weights["time_penalty_scale"].as<double>(w.time_penalty_scale);
        w.time_penalty_scale_decay_steps =
          weights["time_penalty_scale_decay_steps"].as<double>(w.time_penalty_scale_decay_steps);
        w.time_limit_penalty = weights["time_limit_penalty"].as<double>(w.time_limit_penalty);
        w.steer_penalty      = weights["steer_penalty"].as<double>(w.steer_penalty);
        w.steer_penalty_tau_s =
          weights["steer_penalty_tau_s"].as<double>(w.steer_penalty_tau_s);
        w.steer_hold_penalty =
          weights["steer_hold_penalty"].as<double>(w.steer_hold_penalty);
        w.steer_hold_threshold =
          weights["steer_hold_threshold"].as<double>(w.steer_hold_threshold);
        w.steer_hold_terminate_s =
          weights["steer_hold_terminate_s"].as<double>(w.steer_hold_terminate_s);
        w.steer_hold_terminal_penalty =
          weights["steer_hold_terminal_penalty"].as<double>(w.steer_hold_terminal_penalty);
        w.steer_hold_decay_s =
          weights["steer_hold_decay_s"].as<double>(w.steer_hold_decay_s);
        w.yaw_rate_near_goal_dist =
          weights["yaw_rate_near_goal_dist"].as<double>(w.yaw_rate_near_goal_dist);
        w.reverse_hold_penalty =
          weights["reverse_hold_penalty"].as<double>(w.reverse_hold_penalty);
        if (weights["reverse_speed_penalty"]) {
          w.reverse_penalty = weights["reverse_speed_penalty"].as<double>(w.reverse_penalty);
        }
        if (weights["reverse_hold_penalty"]) {
          w.reverse_penalty = weights["reverse_hold_penalty"].as<double>(w.reverse_penalty);
        }
        w.collision_penalty_scale =
          weights["collision_penalty_scale"].as<double>(w.collision_penalty_scale);
        w.clearance_gate_dist_min_m =
          weights["clearance_gate_dist_min_m"].as<double>(w.clearance_gate_dist_min_m);
        w.clearance_gate_dist_max_m =
          weights["clearance_gate_dist_max_m"].as<double>(w.clearance_gate_dist_max_m);
        w.clearance_gate_half_angle_rad =
          weights["clearance_gate_half_angle_rad"].as<double>(w.clearance_gate_half_angle_rad);
        w.stall_penalty_scale =
          weights["stall_penalty_scale"].as<double>(w.stall_penalty_scale);
        w.stall_penalty_growth =
          weights["stall_penalty_growth"].as<double>(w.stall_penalty_growth);
        w.stall_progress_epsilon =
          weights["stall_progress_epsilon"].as<double>(w.stall_progress_epsilon);
        w.stall_window_s = weights["stall_window_s"].as<double>(w.stall_window_s);
        w.stall_tau_s    = weights["stall_tau_s"].as<double>(w.stall_tau_s);
        w.stall_limit    = weights["stall_limit"].as<double>(w.stall_limit);
        w.stall_terminal_penalty =
          weights["stall_terminal_penalty"].as<double>(w.stall_terminal_penalty);
      }

      DDRL_LOG_DEBUG(
        g_logger,
        "Parsed reward config from '{}': type={} terminate_on_goal={}",
        path,
        cfg.reward_type,
        cfg.terminate_on_goal
      );
      return cfg;
    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_reward_config"));
    }
  }

  // ============================================================================
  // World Config
  // ============================================================================

  [[nodiscard]] core::Result<WorldConfig> parse_world_config(std::string_view path)
  {
    try {
      YAML::Node  yaml = YAML::LoadFile(std::string(path));
      WorldConfig cfg;

      cfg.version = yaml["version"].as<std::string>("1.0");

      // Initialize name generators for deduplication
      NameGenerator gen_instance("instance", "object");
      NameGenerator gen_robot("robot", "robot");
      NameGenerator gen_zone("zone", "zone");
      NameGenerator gen_category("collision category", "category");
      NameGenerator gen_group("dynamic group", "group");
      NameGenerator gen_model_profile("model profile", "model");
      NameGenerator gen_dynamic_profile("dynamic profile", "dynamic_profile");
      NameGenerator gen_robot_profile("robot profile", "robot_profile");

      // Parse each section
      if (auto res = parse_collision_categories(yaml["collision_categories"], cfg, gen_category);
          !res) {
        return std::unexpected(res.error());
      }

      fs::path parent_dir = fs::canonical(fs::absolute(path)).parent_path();

      if (auto res = parse_profile_library(
            yaml["model_profiles"],
            &ConfigParser::Impl::parse_model_profile,
            cfg.model_profiles,
            "model_profiles",
            gen_model_profile,
            parent_dir
          );
          res) {
        return std::unexpected(std::move(*res));
      }

      if (auto res = parse_profile_library(
            yaml["dynamic_profiles"],
            &ConfigParser::Impl::parse_dynamic_profile,
            cfg.dynamic_profiles,
            "dynamic_profiles",
            gen_dynamic_profile,
            parent_dir
          );
          res) {
        return std::unexpected(std::move(*res));
      }

      if (auto res = parse_profile_library(
            yaml["robot_profiles"],
            &ConfigParser::Impl::parse_robot_profile,
            cfg.robot_profiles,
            "robot_profiles",
            gen_robot_profile,
            parent_dir
          );
          res) {
        return std::unexpected(std::move(*res));
      }

      std::unordered_map<std::string, std::string> zone_aliases;
      if (auto res = parse_robot_config(yaml["robot"], cfg, gen_robot, zone_aliases); !res) {
        return std::unexpected(res.error());
      }

      if (auto res = parse_world_section(yaml["world"], cfg, gen_instance, gen_group); !res) {
        return std::unexpected(res.error());
      }

      log_world_config(cfg, path);
      return cfg;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_world_config"));
    } catch (const std::exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_world_config"));
    }
  }

  // ============================================================================
  // Model Profile
  // ============================================================================

  core::Result<ModelProfile> parse_model_profile(std::string_view path)
  {
    try {
      YAML::Node   yaml = YAML::LoadFile(std::string(path));
      ModelProfile profile;

      if (!yaml["name"]) {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Model profile missing 'name' field", "parse_model_profile")
        );
      }
      if (!yaml["parts"]) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG, "Model profile missing 'parts' field", "parse_model_profile"
        ));
      }

      profile.name     = yaml["name"].as<std::string>();
      profile.category = yaml["category"].as<std::string>("generic");

      for (const auto& part_node : yaml["parts"]) {
        auto part_res = parse_model_part(part_node);
        if (!part_res) {
          return std::unexpected(part_res.error());
        }
        profile.parts.emplace_back(std::move(*part_res));
      }

      log_model_profile(profile, path);
      return profile;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_model_profile"));
    }
  }

  // ============================================================================
  // Dynamic Profile
  // ============================================================================

  core::Result<DynamicProfile> parse_dynamic_profile(std::string_view path)
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));

      DynamicProfile profile;
      profile.name = yaml["name"].as<std::string>("unnamed");

      // Parse motion controller cfg
      if (yaml["cfg"]) {
        const auto& cfg_node = yaml["cfg"];
        auto        type     = cfg_node["type"].as<std::string>();

        if (type == "linear_const_vel") {
          LinearConstVelCfg cfg;
          if (cfg_node["caps"]) {
            const auto& c            = cfg_node["caps"];
            cfg.caps.max_speed       = c["max_speed"].as<double>(10.0);
            cfg.caps.max_accel       = c["max_accel"].as<double>(3.0);
            cfg.caps.max_decel       = c["max_decel"].as<double>(5.0);
            cfg.caps.cmd_rate_hz     = c["cmd_rate_hz"].as<double>(50.0);
            cfg.caps.cmd_delay_s     = c["cmd_delay_s"].as<double>(0.0);
            cfg.caps.accel_noise_std = c["accel_noise_std"].as<double>(0.0);
            cfg.caps.steer_noise_std = c["steer_noise_std"].as<double>(0.0);
          }
          cfg.speed_min      = cfg_node["speed_min"].as<double>(1.0);
          cfg.speed_max      = cfg_node["speed_max"].as<double>(2.0);
          cfg.brake_to_stop  = cfg_node["brake_to_stop"].as<bool>(true);
          profile.follow_cfg = cfg;
        } else if (type == "linear_pid") {
          LinearPIDCfg cfg;
          if (cfg_node["caps"]) {
            const auto& c            = cfg_node["caps"];
            cfg.caps.max_speed       = c["max_speed"].as<double>(10.0);
            cfg.caps.max_accel       = c["max_accel"].as<double>(3.0);
            cfg.caps.max_decel       = c["max_decel"].as<double>(5.0);
            cfg.caps.cmd_rate_hz     = c["cmd_rate_hz"].as<double>(50.0);
            cfg.caps.cmd_delay_s     = c["cmd_delay_s"].as<double>(0.0);
            cfg.caps.accel_noise_std = c["accel_noise_std"].as<double>(0.0);
            cfg.caps.steer_noise_std = c["steer_noise_std"].as<double>(0.0);
          }
          if (cfg_node["speed_pid"]) {
            const auto& pid                     = cfg_node["speed_pid"];
            cfg.speed_pid.kp                    = pid["kp"].as<double>(1.0);
            cfg.speed_pid.ki                    = pid["ki"].as<double>(0.1);
            cfg.speed_pid.kd                    = pid["kd"].as<double>(0.05);
            cfg.speed_pid.integral_windup_limit = pid["integral_windup_limit"].as<double>(10.0);
            cfg.speed_pid.d_filter_hz           = pid["d_filter_hz"].as<double>(10.0);
          }
          if (cfg_node["lateral_pid"]) {
            const auto& pid                       = cfg_node["lateral_pid"];
            cfg.lateral_pid.kp                    = pid["kp"].as<double>(1.0);
            cfg.lateral_pid.ki                    = pid["ki"].as<double>(0.1);
            cfg.lateral_pid.kd                    = pid["kd"].as<double>(0.05);
            cfg.lateral_pid.integral_windup_limit = pid["integral_windup_limit"].as<double>(10.0);
            cfg.lateral_pid.d_filter_hz           = pid["d_filter_hz"].as<double>(10.0);
          }
          cfg.cruise_speed_min  = cfg_node["cruise_speed_min"].as<double>(5.0);
          cfg.cruise_speed_max  = cfg_node["cruise_speed_max"].as<double>(10.0);
          cfg.max_lateral_error = cfg_node["max_lateral_error"].as<double>(2.0);
          profile.follow_cfg    = cfg;
        } else if (type == "ackermann_follow") {
          AckermannFollowCfg cfg;
          if (cfg_node["kin"]) {
            const auto& k          = cfg_node["kin"];
            const auto  parse_caps = [](const YAML::Node& c, core::kinematics::MotionCaps& caps) {
              caps.max_speed       = c["max_speed"].as<double>(10.0);
              caps.max_accel       = c["max_accel"].as<double>(3.0);
              caps.max_decel       = c["max_decel"].as<double>(5.0);
              caps.cmd_rate_hz     = c["cmd_rate_hz"].as<double>(50.0);
              caps.cmd_delay_s     = c["cmd_delay_s"].as<double>(0.0);
              caps.accel_noise_std = c["accel_noise_std"].as<double>(0.0);
              caps.steer_noise_std = c["steer_noise_std"].as<double>(0.0);
            };

            if (k["caps_fwd"]) {
              parse_caps(k["caps_fwd"], cfg.acker_caps.caps_fwd);
            }
            if (k["caps_rev"]) {
              parse_caps(k["caps_rev"], cfg.acker_caps.caps_rev);
            }
            cfg.acker_caps.wheel_base      = k["wheel_base"].as<double>(2.5);
            cfg.acker_caps.max_steer_angle = k["max_steer_angle"].as<double>(0.7);
            cfg.acker_caps.max_steer_rate  = k["max_steer_rate"].as<double>(1.0);
            cfg.acker_caps.gear_hold_time_s = k["gear_hold_time_s"].as<double>(0.0);
            cfg.acker_caps.min_turn_radius = k["min_turn_radius"].as<double>(5.0);
          }
          if (cfg_node["pp"]) {
            const auto& pp                    = cfg_node["pp"];
            cfg.pp.lookahead_m                = pp["lookahead_m"].as<double>(2.0);
            cfg.pp.min_lookahead_m            = pp["min_lookahead_m"].as<double>(1.0);
            cfg.pp.max_lookahead_m            = pp["max_lookahead_m"].as<double>(5.0);
            cfg.pp.speed_coupled_to_curvature = pp["speed_coupled_to_curvature"].as<bool>(true);
            cfg.pp.kv                         = pp["kv"].as<double>(0.5);
          }
          if (cfg_node["speed_pid"]) {
            const auto& pid                     = cfg_node["speed_pid"];
            cfg.speed_pid.kp                    = pid["kp"].as<double>(1.0);
            cfg.speed_pid.ki                    = pid["ki"].as<double>(0.1);
            cfg.speed_pid.kd                    = pid["kd"].as<double>(0.05);
            cfg.speed_pid.integral_windup_limit = pid["integral_windup_limit"].as<double>(10.0);
            cfg.speed_pid.d_filter_hz           = pid["d_filter_hz"].as<double>(10.0);
          }
          cfg.cruise_speed_min        = cfg_node["cruise_speed_min"].as<double>(8.0);
          cfg.cruise_speed_max        = cfg_node["cruise_speed_max"].as<double>(12.0);
          cfg.cornering_speed_min     = cfg_node["cornering_speed_min"].as<double>(3.0);
          cfg.arrival_stop_decel_bias = cfg_node["arrival_stop_decel_bias"].as<double>(1.2);
          profile.follow_cfg          = cfg;
        } else {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("Unknown dynamic profile cfg type: {}", type),
            "parse_dynamic_profile"
          ));
        }
      }

      // Parse waypoints_local
      if (yaml["waypoints_local"]) {
        const auto& wp_node                     = yaml["waypoints_local"];
        profile.waypoints_local.reached_pos_tol = wp_node["reached_pos_tol"].as<double>(0.5);
        profile.waypoints_local.reached_yaw_tol = wp_node["reached_yaw_tol"].as<double>(0.1);
        profile.waypoints_local.loop            = wp_node["loop"].as<bool>(false);

        if (wp_node["points"]) {
          for (const auto& pt_node : wp_node["points"]) {
            Waypoint wp;
            wp.wait_s = pt_node["wait_s"].as<double>(0.0);
            if (pt_node["p_local"]) {
              wp.p_local = parse_pose2(pt_node["p_local"]);
            }
            profile.waypoints_local.points.push_back(wp);
          }
        }
      }

      DDRL_LOG_DEBUG(
        g_logger,
        "Parsed dynamic profile from '{}': {} (waypoints: {})",
        path,
        profile.name,
        profile.waypoints_local.points.size()
      );

      return profile;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_dynamic_profile"));
    }
  }

  // ============================================================================
  // Robot Profile
  // ============================================================================

  core::Result<RobotProfile> parse_robot_profile(std::string_view path)
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));

      RobotProfile profile;
      profile.name     = yaml["name"].as<std::string>("unnamed");
      profile.model_id = yaml["model_id"].as<std::string>();

      // Parse kinematics
      if (yaml["kinematics"]) {
        const auto& k = yaml["kinematics"];

        // Motion caps
        const auto parse_caps = [](const YAML::Node& c, core::kinematics::MotionCaps& caps) {
          caps.max_speed       = c["max_speed"].as<double>(10.0);
          caps.max_accel       = c["max_accel"].as<double>(3.0);
          caps.max_decel       = c["max_decel"].as<double>(5.0);
          caps.cmd_rate_hz     = c["cmd_rate_hz"].as<double>(50.0);
          caps.cmd_delay_s     = c["cmd_delay_s"].as<double>(0.0);
          caps.accel_noise_std = c["accel_noise_std"].as<double>(0.0);
          caps.steer_noise_std = c["steer_noise_std"].as<double>(0.0);
        };

        if (k["caps_fwd"]) {
          parse_caps(k["caps_fwd"], profile.kinematics.caps_fwd);
        }
        if (k["caps_rev"]) {
          parse_caps(k["caps_rev"], profile.kinematics.caps_rev);
        }

        // Ackermann-specific
        profile.kinematics.wheel_base       = k["wheel_base"].as<double>(2.5);
        profile.kinematics.max_steer_angle  = k["max_steer_angle"].as<double>(0.7);
        profile.kinematics.max_steer_rate   = k["max_steer_rate"].as<double>(1.0);
        profile.kinematics.min_turn_radius  = k["min_turn_radius"].as<double>(5.0);
        profile.kinematics.gear_hold_time_s = k["gear_hold_time_s"].as<double>(0.0);
      }

      // Parse sensors
      if (yaml["sensors"]) {
        for (const auto& sensor_node : yaml["sensors"]) {
          std::string sensor_type = sensor_node["type"].as<std::string>();

          core::sensors::SensorBase base;
          base.id           = sensor_node["id"].as<std::string>();
          base.frame_id     = sensor_node["id"].as<std::string>() + "_frame";
          base.min_range    = sensor_node["min_range"].as<float>(0.1f);
          base.max_range    = sensor_node["max_range"].as<float>(100.0f);
          base.noise_std    = sensor_node["noise_std"].as<float>(0.0f);
          base.dropout_prob = sensor_node["dropout_prob"].as<float>(0.0f);
          base.update_hz    = sensor_node["update_hz"].as<float>(10.0f);
          base.latency_s    = sensor_node["latency_s"].as<float>(0.0f);

          // Parse pose_rel_base
          if (sensor_node["pose_rel_base"]) {
            const auto& pose_node = sensor_node["pose_rel_base"];
            if (pose_node["t"]) {
              const auto& t          = pose_node["t"];
              base.pose_rel_base.t.x = t[0].as<double>(0.0);
              base.pose_rel_base.t.y = t[1].as<double>(0.0);
              base.pose_rel_base.t.z = t[2].as<double>(0.0);
            }
            if (pose_node["q"]) {
              const auto& q          = pose_node["q"];
              base.pose_rel_base.q.w = q[0].as<double>(1.0);
              base.pose_rel_base.q.x = q[1].as<double>(0.0);
              base.pose_rel_base.q.y = q[2].as<double>(0.0);
              base.pose_rel_base.q.z = q[3].as<double>(0.0);
            }
          }

          if (sensor_type == "lidar_3d") {
            core::sensors::Lidar3D lidar;
            lidar.base        = base;
            lidar.hfov        = sensor_node["hfov"].as<double>(6.283185);
            lidar.angular_res = sensor_node["angular_res"].as<double>(0.017453);
            lidar.vfov        = sensor_node["vfov"].as<double>(0.523599);
            lidar.num_rings   = sensor_node["num_rings"].as<size_t>(32);
            profile.sensors.push_back(lidar);
          } else if (sensor_type == "lidar_2d") {
            core::sensors::Lidar2D lidar;
            lidar.base        = base;
            lidar.fov         = sensor_node["fov"].as<double>(3.14159);
            lidar.angular_res = sensor_node["angular_res"].as<double>(0.017453);
            profile.sensors.push_back(lidar);
          } else if (sensor_type == "depth_camera") {
            core::sensors::DepthCamera camera;
            camera.base   = base;
            camera.fx     = sensor_node["fx"].as<float>(500.0f);
            camera.fy     = sensor_node["fy"].as<float>(500.0f);
            camera.cx     = sensor_node["cx"].as<float>(320.0f);
            camera.cy     = sensor_node["cy"].as<float>(240.0f);
            camera.width  = sensor_node["width"].as<uint16_t>(640);
            camera.height = sensor_node["height"].as<uint16_t>(480);
            camera.near_z = sensor_node["near_z"].as<double>(0.1);
            camera.far_z  = sensor_node["far_z"].as<double>(50.0);
            profile.sensors.push_back(camera);
          }
        }
      }

      log_robot_profile(profile, path);
      return profile;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_robot_profile"));
    }
  }

  // ============================================================================
  // Visualizer Config
  // ============================================================================

  core::Result<VizConfig> parse_visualizer_config(std::string_view path)
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));
      VizConfig  cfg;

      // Backend (default: VTK)
      if (yaml["backend"]) {
        auto backend_str = yaml["backend"].as<std::string>("vtk");
        if (backend_str == "vtk" || backend_str == "VTK") {
          cfg.backend = VisualizerBackend::VTK;
        } else {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("Unknown visualizer backend: {}", backend_str),
            "parse_visualizer_config"
          ));
        }
      } else {
        cfg.backend = VisualizerBackend::VTK;
      }

      // Window settings
      cfg.window_width  = yaml["window_width"].as<uint32_t>(1920);
      cfg.window_height = yaml["window_height"].as<uint32_t>(1080);
      cfg.window_title  = yaml["window_title"].as<std::string>("DDRL Visualizer");

      // Performance settings
      cfg.target_fps = yaml["target_fps"].as<double>(30.0);

      // Interactive style configuration
      if (yaml["interactive_style"]) {
        const auto& style                = yaml["interactive_style"];
        cfg.interactor_style.sensitivity = style["sensitivity"].as<double>(1.0);

        if (style["zoom"]) {
          const auto& z                          = style["zoom"];
          cfg.interactor_style.zoom.max_zoom_in  = z["max_zoom_in"].as<double>(0.8);
          cfg.interactor_style.zoom.max_zoom_out = z["max_zoom_out"].as<double>(5.0);
        }

        if (style["pan"]) {
          const auto& p                    = style["pan"];
          cfg.interactor_style.pan.max_pan = p["max_pan"].as<double>(1.5);
        }

        if (style["rotate"]) {
          const auto& r                         = style["rotate"];
          cfg.interactor_style.rotate.max_pitch = r["max_pitch"].as<double>(80.0);
          cfg.interactor_style.rotate.min_pitch = r["min_pitch"].as<double>(-80.0);
        }
      }

      // Camera follow settings
      if (yaml["camera_settings"]) {
        const auto& cam                = yaml["camera_settings"];
        cfg.camera_follow.smooth_alpha = cam["smooth_alpha"].as<double>(0.15);

        if (cam["chase"]) {
          const auto& chase                     = cam["chase"];
          cfg.camera_follow.chase.back_distance = chase["back_distance"].as<double>(6.0);
          cfg.camera_follow.chase.up_distance   = chase["up_distance"].as<double>(3.0);
          cfg.camera_follow.chase.focal_height  = chase["focal_height"].as<double>(0.5);
        }

        if (cam["top_down"]) {
          const auto& top_down              = cam["top_down"];
          cfg.camera_follow.top_down.height = top_down["height"].as<double>(20.0);
        }

        if (cam["orbit"]) {
          const auto& orbit                    = cam["orbit"];
          cfg.camera_follow.orbit.radius       = orbit["radius"].as<double>(10.0);
          cfg.camera_follow.orbit.height       = orbit["height"].as<double>(6.0);
          cfg.camera_follow.orbit.speed        = orbit["speed"].as<double>(0.6);
          cfg.camera_follow.orbit.focal_height = orbit["focal_height"].as<double>(0.5);
        }

        if (cam["first_person"]) {
          const auto& fp                                = cam["first_person"];
          cfg.camera_follow.first_person.eye_height     = fp["eye_height"].as<double>(1.5);
          cfg.camera_follow.first_person.forward_offset = fp["forward_offset"].as<double>(0.3);
          cfg.camera_follow.first_person.focal_distance = fp["focal_distance"].as<double>(10.0);
        }
      }

      log_visualizer_config(cfg, path);
      return cfg;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_visualizer_config"));
    } catch (const std::exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_visualizer_config"));
    }
  }

  // ============================================================================
  // Map Config
  // ============================================================================

  core::Result<MapConfig> parse_map_config(std::string_view path)
  {
    try {
      YAML::Node yaml = YAML::LoadFile(std::string(path));
      MapConfig  cfg;

      cfg.version     = yaml["version"].as<std::string>("1.0");
      cfg.name        = map_name_gen_(yaml["name"].as<std::string>("unnamed_map"));
      cfg.description = yaml["description"].as<std::string>("");

      // Initialize name generator for static instances
      NameGenerator gen_instance("instance", "object");

      // Parse static instances
      if (yaml["static_instances"]) {
        const auto& static_node = yaml["static_instances"];
        if (!static_node.IsSequence()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG, "'static_instances' must be a sequence", "parse_map_config"
          ));
        }

        for (size_t i = 0; i < static_node.size(); ++i) {
          const auto&       inst_node = static_node[i];
          StaticInstanceCfg inst;

          if (inst_node["name"]) {
            inst.name = inst_node["name"].as<std::string>();
          } else if (inst_node["id"]) {
            inst.name = inst_node["id"].as<std::string>();
          } else {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("static_instances[{}] missing 'name'", i),
              "parse_map_config"
            ));
          }

          inst.name = gen_instance(inst.name);

          if (!inst_node["model_id"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("static instance '{}' missing 'model_id'", inst.name),
              "parse_map_config"
            ));
          }
          inst.model_id = inst_node["model_id"].as<std::string>();

          if (!inst_node["pose_world"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("static instance '{}' missing 'pose_world'", inst.name),
              "parse_map_config"
            ));
          }
          inst.pose_world = parse_pose2(inst_node["pose_world"]);

          cfg.static_instances.emplace_back(std::move(inst));
        }
      }

      // Parse optional bounds
      if (yaml["bounds"]) {
        const auto&       bounds_node = yaml["bounds"];
        core::geom::AABB2 bounds;
        if (bounds_node["min"]) {
          const auto& min = bounds_node["min"];
          bounds.min.x    = min[0].as<double>();
          bounds.min.y    = min[1].as<double>();
        }
        if (bounds_node["max"]) {
          const auto& max = bounds_node["max"];
          bounds.max.x    = max[0].as<double>();
          bounds.max.y    = max[1].as<double>();
        }
        cfg.bounds = bounds;

      } else {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "Map 'bounds' definition is required", "parse_map_config")
        );
      }

      if (!yaml["robot_routes"]) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG, "Map must define 'robot_routes'", "parse_map_config"
        ));
      }

      const auto& routes_node = yaml["robot_routes"];
      if (!routes_node.IsSequence()) {
        return std::unexpected(
          make_error(ErrorCode::CONFIG, "'robot_routes' must be a sequence", "parse_map_config")
        );
      }

      NameGenerator gen_route("route", "route");
      for (size_t route_idx = 0; route_idx < routes_node.size(); ++route_idx) {
        const auto& route_node = routes_node[route_idx];
        if (!route_node["name"]) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("robot_routes[{}] missing 'name'", route_idx),
            "parse_map_config"
          ));
        }
        if (!route_node["start"] || !route_node["start"].IsMap()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("robot route '{}' missing 'start' pose", route_node["name"].as<std::string>()),
            "parse_map_config"
          ));
        }
        if (!route_node["waypoints"] || !route_node["waypoints"].IsSequence()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("robot route '{}' missing waypoint sequence", route_node["name"].as<std::string>()),
            "parse_map_config"
          ));
        }

        RobotRoute route;
        route.name = gen_route(route_node["name"].as<std::string>());
        route.start = parse_pose2(route_node["start"]);

        const auto& waypoints_node = route_node["waypoints"];
        if (waypoints_node.size() == 0) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("robot route '{}' must define at least one waypoint", route.name),
            "parse_map_config"
          ));
        }

        NameGenerator gen_waypoint("waypoint", "waypoint");
        for (size_t waypoint_idx = 0; waypoint_idx < waypoints_node.size(); ++waypoint_idx) {
          const auto& waypoint_node = waypoints_node[waypoint_idx];
          if (!waypoint_node["name"] || !waypoint_node["pos"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("robot route '{}' waypoint[{}] must define 'name' and 'pos'", route.name, waypoint_idx),
              "parse_map_config"
            ));
          }

          RobotWaypoint waypoint;
          waypoint.name = gen_waypoint(waypoint_node["name"].as<std::string>());
          const auto& pos_node = waypoint_node["pos"];
          if (!pos_node.IsSequence() || pos_node.size() != 2) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("robot route '{}' waypoint '{}' pos must be [x, y]", route.name, waypoint.name),
              "parse_map_config"
            ));
          }
          waypoint.pos = parse_point2(pos_node);
          waypoint.tolerance = waypoint_node["tolerance"].as<double>(1.0);
          if (waypoint.tolerance <= 0.0) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("robot route '{}' waypoint '{}' tolerance must be > 0", route.name, waypoint.name),
              "parse_map_config"
            ));
          }
          route.waypoints.push_back(std::move(waypoint));
        }

        cfg.robot_routes.push_back(std::move(route));
      }
      if (cfg.robot_routes.empty()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG, "At least one 'robot_routes' entry must be defined", "parse_map_config"
        ));
      }

      DDRL_LOG_DEBUG(
        g_logger,
        "Loaded map '{}': {} static objects, {} robot routes",
        cfg.name,
        cfg.static_instances.size(),
        cfg.robot_routes.size()
      );

      return cfg;

    } catch (const YAML::Exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_map_config"));
    } catch (const std::exception& e) {
      return std::unexpected(make_error(ErrorCode::CONFIG, e.what(), "parse_map_config"));
    }
  }

private:
  // ============================================================================
  // World Config Helpers
  // ============================================================================

  template <typename Profile>
  std::optional<Error> parse_profile_library(
    const YAML::Node& node, core::Result<Profile> (Impl::*loader)(std::string_view),
    std::unordered_map<std::string, Profile>& dest, std::string_view lib_name,
    NameGenerator& name_gen, const std::filesystem::path& root_path
  )
  {
    if (!node) {
      return std::nullopt;
    }

    auto load_from_path = [&](const std::string& path) -> std::optional<Error> {
      auto res = (this->*loader)(path);
      if (!res) {
        return res.error();
      }

      auto        profile      = std::move(*res);
      const auto& profile_name = profile.name;

      if (profile_name.empty()) {
        return make_error(
          ErrorCode::CONFIG,
          std::format("'{}' profile missing name", lib_name),
          "parse_profile_library"
        );
      }

      auto unique_name = name_gen(profile_name);
      profile.name     = unique_name;

      auto insert_res = dest.emplace(unique_name, std::move(profile));
      if (!insert_res.second) {
        return make_error(
          ErrorCode::CONFIG,
          std::format(
            "'{}' duplicate profile name '{}' even after automatic renaming", lib_name, unique_name
          ),
          "parse_profile_library"
        );
      }
      return std::nullopt;
    };

    if (node.IsSequence()) {
      for (size_t i = 0; i < node.size(); ++i) {
        const auto& entry = node[i];

        if (entry.IsScalar()) {
          fs::path path_or_pattern = entry.as<std::string>();
          fs::path res_path;
          if (path_or_pattern.string().contains('*')) {
            res_path = fs::canonical(root_path / path_or_pattern.parent_path()).string();
            res_path /= path_or_pattern.filename().string();
          } else {
            res_path = fs::canonical(root_path / path_or_pattern).string();
          }
          // Expand glob pattern if it contains wildcards
          auto expanded_paths = expand_glob_pattern(res_path.string());

          if (expanded_paths.empty()) {
            DDRL_LOG_WARN(
              g_logger, "'{}': pattern '{}' did not match any files", lib_name, res_path.string()
            );
            continue;
          }

          // Load each expanded path
          for (const auto& path : expanded_paths) {
            if (auto res = load_from_path(path); res) {
              return res;
            }
          }
          continue;
        }

        return make_error(
          ErrorCode::CONFIG,
          std::format("'{}'[{}] must be a scalar path with 'file'/'path'", lib_name, i),
          "parse_profile_library"
        );
      }
      return std::nullopt;
    }

    return make_error(
      ErrorCode::CONFIG, std::format("'{}' must be a sequence", lib_name), "parse_profile_library"
    );
  }

  static core::Result<void> parse_collision_categories(
    const YAML::Node& cats_node, WorldConfig& cfg, NameGenerator& gen_category
  )
  {
    if (!cats_node) {
      return {};
    }

    if (!cats_node.IsSequence()) {
      return std::unexpected(make_error(
        ErrorCode::CONFIG, "'collision_categories' must be a sequence", "parse_collision_categories"
      ));
    }

    for (size_t i = 0; i < cats_node.size(); ++i) {
      const auto& cat_node = cats_node[i];
      if (!cat_node["category"]) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          std::format("collision_categories[{}] missing 'category'", i),
          "parse_collision_categories"
        ));
      }

      CollisionFilter filter;
      filter.category = gen_category(cat_node["category"].as<std::string>());

      if (const auto& collides_node = cat_node["collides_with"]; collides_node) {
        if (!collides_node.IsSequence()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            std::format("collision_categories[{}].collides_with must be a sequence", i),
            "parse_collision_categories"
          ));
        }
        for (const auto& name : collides_node) {
          filter.collides_with.emplace_back(name.as<std::string>());
        }
      }

      cfg.collision_categories.emplace_back(std::move(filter));
    }

    return {};
  }

  static core::Result<void> parse_zones(
    const YAML::Node& zones_node, WorldConfig& cfg, NameGenerator& gen_zone,
    std::unordered_map<std::string, std::string>& zone_aliases
  )
  {
    // Zones are now loaded from map files, not world config
    // This function is kept for backward compatibility but does nothing
    (void)zones_node;
    (void)cfg;
    (void)gen_zone;
    (void)zone_aliases;
    return {};
  }

  static core::Result<void> parse_robot_config(
    const YAML::Node& robot_node, WorldConfig& cfg, NameGenerator& gen_robot,
    [[maybe_unused]] const std::unordered_map<std::string, std::string>& zone_aliases
  )
  {
    if (!robot_node) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "World config missing 'robot' section", "parse_robot_config")
      );
    }
    if (!robot_node.IsMap()) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "'robot' must be a mapping", "parse_robot_config")
      );
    }
    if (!robot_node["robot_profile_id"]) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "'robot.robot_profile_id' is required", "parse_robot_config")
      );
    }

    cfg.robot.name             = gen_robot(robot_node["name"].as<std::string>("robot"));
    cfg.robot.robot_profile_id = robot_node["robot_profile_id"].as<std::string>();
    cfg.robot.action_delay_s   = robot_node["action_delay_s"].as<double>(0.0);

    // Zones are now defined in MapConfig, not in RobotInstanceCfg
    return {};
  }

  static core::Result<void> parse_world_section(
    const YAML::Node& world_node, WorldConfig& cfg, NameGenerator& gen_instance,
    NameGenerator& gen_group
  )
  {
    if (!world_node) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "World config missing 'world' section", "parse_world_section")
      );
    }
    if (!world_node.IsMap()) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "'world' must be a mapping", "parse_world_section")
      );
    }
    if (!world_node["type"]) {
      return std::unexpected(
        make_error(ErrorCode::CONFIG, "'world.type' is required", "parse_world_section")
      );
    }

    const auto world_type = world_node["type"].as<std::string>();

    if (world_type == "deterministic") {
      DeterministicWorldCfg det_cfg;

      if (world_node["robot_pose_world"]) {
        det_cfg.robot_pose_world = parse_pose2(world_node["robot_pose_world"]);
      }

      if (const auto& dynamic_node = world_node["dynamic_instances"]; dynamic_node) {
        if (!dynamic_node.IsSequence()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG, "world.dynamic_instances must be a sequence", "parse_world_section"
          ));
        }

        for (size_t i = 0; i < dynamic_node.size(); ++i) {
          const auto&        di_node = dynamic_node[i];
          DynamicInstanceCfg inst;

          if (di_node["name"]) {
            inst.name = di_node["name"].as<std::string>();
          } else if (di_node["id"]) {
            inst.name = di_node["id"].as<std::string>();
          } else {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("dynamic_instances[{}] missing 'name'", i),
              "parse_world_section"
            ));
          }

          inst.name = gen_instance(inst.name);

          if (!di_node["model_id"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("dynamic instance '{}' missing 'model_id'", inst.name),
              "parse_world_section"
            ));
          }
          if (!di_node["dynamic_profile_id"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("dynamic instance '{}' missing 'dynamic_profile_id'", inst.name),
              "parse_world_section"
            ));
          }
          if (!di_node["pose_world"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("dynamic instance '{}' missing 'pose_world'", inst.name),
              "parse_world_section"
            ));
          }

          inst.model_id           = di_node["model_id"].as<std::string>();
          inst.dynamic_profile_id = di_node["dynamic_profile_id"].as<std::string>();
          inst.pose_world         = parse_pose2(di_node["pose_world"]);

          det_cfg.dynamic_instances.emplace_back(std::move(inst));
        }
      }

      cfg.world = std::move(det_cfg);

    } else if (world_type == "random") {
      RandomWorldCfg random_cfg;

      if (const auto& groups = world_node["random_dynamic_instances"]; groups) {
        if (!groups.IsSequence()) {
          return std::unexpected(make_error(
            ErrorCode::CONFIG,
            "world.random_dynamic_instances must be a sequence",
            "parse_world_section"
          ));
        }

        for (size_t i = 0; i < groups.size(); ++i) {
          const auto&               group_node = groups[i];
          RandomDynamicInstancesCfg group;

          if (!group_node["name"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("random_dynamic_instances[{}] missing 'name'", i),
              "parse_world_section"
            ));
          }
          group.name = gen_group(group_node["name"].as<std::string>());

          if (!group_node["model_id"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("random_dynamic_instances[{}] missing 'model_id'", i),
              "parse_world_section"
            ));
          }
          if (!group_node["dynamic_profile_id"]) {
            return std::unexpected(make_error(
              ErrorCode::CONFIG,
              std::format("random_dynamic_instances[{}] missing 'dynamic_profile_id'", i),
              "parse_world_section"
            ));
          }

          group.model_id           = group_node["model_id"].as<std::string>();
          group.dynamic_profile_id = group_node["dynamic_profile_id"].as<std::string>();
          group.count              = group_node["count"].as<size_t>(0);

          random_cfg.random_dynamic_instances.emplace_back(std::move(group));
        }
      }

      cfg.world = std::move(random_cfg);

    } else {
      return std::unexpected(make_error(
        ErrorCode::CONFIG, std::format("Unknown world type '{}'", world_type), "parse_world_section"
      ));
    }

    return {};
  }

  // ============================================================================
  // Model Part Parsing
  // ============================================================================

  static core::Result<ModelPart> parse_model_part(const YAML::Node& part_node)
  {
    ModelPart part;
    part.name = part_node["name"].as<std::string>("unnamed_part");

    const auto& geom_node = part_node["geom"];
    if (!geom_node || !geom_node.IsMap()) {
      return std::unexpected(make_error(
        ErrorCode::CONFIG,
        std::format("Model part '{}' missing 'geom' section", part.name),
        "parse_model_part"
      ));
    }

    if (!geom_node["type"]) {
      return std::unexpected(make_error(
        ErrorCode::CONFIG,
        std::format("Model part '{}' missing 'geom.type'", part.name),
        "parse_model_part"
      ));
    }

    core::geom::Shape2p5 geom;
    geom.z_min = geom_node["z_min"].as<double>(0.0);
    geom.z_max = geom_node["z_max"].as<double>(0.0);

    const auto geom_type = geom_node["type"].as<std::string>();

    if (geom_type == "circle") {
      if (!geom_node["radius"]) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          std::format("Model part '{}' circle missing 'radius'", part.name),
          "parse_model_part"
        ));
      }
      geom.base = core::geom::Circle(geom_node["radius"].as<double>());

    } else if (geom_type == "box") {
      if (!geom_node["hx"] || !geom_node["hy"]) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          std::format("Model part '{}' box requires 'hx' and 'hy'", part.name),
          "parse_model_part"
        ));
      }
      geom.base = core::geom::Box2(geom_node["hx"].as<double>(), geom_node["hy"].as<double>());

    } else if (geom_type == "polygon") {
      const auto& vertices_node = geom_node["vertices"];
      if (!vertices_node || !vertices_node.IsSequence()) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          std::format("Model part '{}' polygon geom missing 'vertices' list", part.name),
          "parse_model_part"
        ));
      }
      if (vertices_node.size() < 3) {
        return std::unexpected(make_error(
          ErrorCode::CONFIG,
          std::format("Model part '{}' polygon requires at least 3 vertices", part.name),
          "parse_model_part"
        ));
      }
      geom.base = parse_polygon2(vertices_node);

    } else {
      return std::unexpected(make_error(
        ErrorCode::CONFIG,
        std::format("Unsupported geom type '{}' in part '{}'", geom_type, part.name),
        "parse_model_part"
      ));
    }

    part.geom = std::move(geom);

    // Relative pose (optional)
    if (part_node["rel_pose"]) {
      part.rel_pose = parse_pose2(part_node["rel_pose"]);
    }

    // Material (optional, defaults to unit white)
    part.material.rgba = {1.0f, 1.0f, 1.0f, 1.0f};
    if (const auto& material_node = part_node["material"]) {
      part.material.name = material_node["name"].as<std::string>(part.name);
      if (const auto& rgba_node = material_node["rgba"];
          rgba_node && rgba_node.IsSequence() && rgba_node.size() == 4) {
        for (size_t i = 0; i < 4; ++i) {
          part.material.rgba[i] = static_cast<float>(rgba_node[i].as<double>());
        }
      }
    } else {
      part.material.name = part.name;
    }

    return part;
  }

  // ============================================================================
  // Logging Helpers
  // ============================================================================

  static void log_simulator_config(const SimConfig& cfg, std::string_view path)
  {
    DDRL_LOG_DEBUG(g_logger, "Parsed simulator config from '{}':", path);
    DDRL_LOG_DEBUG(g_logger, "  Physics:");
    DDRL_LOG_DEBUG(g_logger, "    fixed_dt: {}", cfg.physics.fixed_dt);
    DDRL_LOG_DEBUG(g_logger, "    max_substeps_per_step: {}", cfg.physics.max_substeps_per_step);
    DDRL_LOG_DEBUG(g_logger, "  Decision:");
    DDRL_LOG_DEBUG(g_logger, "    decision_dt: {}", cfg.decision.decision_dt);
    DDRL_LOG_DEBUG(g_logger, "    action_delay_s: {}", cfg.decision.action_delay_s);
    DDRL_LOG_DEBUG(g_logger, "    first_order_tau_s: {}", cfg.decision.first_order_tau_s);
    DDRL_LOG_DEBUG(g_logger, "  Episode:");
    DDRL_LOG_DEBUG(g_logger, "    max_decision_steps: {}", cfg.episode.max_decision_steps);
    DDRL_LOG_DEBUG(g_logger, "    max_sim_seconds: {}", cfg.episode.max_sim_seconds);
    DDRL_LOG_DEBUG(g_logger, "    collision_timeout_s: {}", cfg.episode.collision_timeout_s);
    DDRL_LOG_DEBUG(g_logger, "  SpawnManager:");
    DDRL_LOG_DEBUG(
      g_logger,
      "    spawn_distance: [{:.2f}, {:.2f}]",
      cfg.spawn_manager.min_spawn_distance,
      cfg.spawn_manager.max_spawn_distance
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    active_objects: [{}, {}]",
      cfg.spawn_manager.min_active_objects,
      cfg.spawn_manager.max_active_objects
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    fov_angle_deg: {:.2f}, fov_distance: {:.2f}, max_sensor_range: {:.2f}",
      cfg.spawn_manager.fov_angle_deg,
      cfg.spawn_manager.fov_distance,
      cfg.spawn_manager.max_sensor_range
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    max_spawns_per_frame: {}, max_spawn_attempts_per_frame: {}",
      cfg.spawn_manager.max_spawns_per_frame,
      cfg.spawn_manager.max_spawn_attempts_per_frame
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    respect_map_geometry: {}, avoid_fov: {}",
      cfg.spawn_manager.respect_map_geometry,
      cfg.spawn_manager.avoid_fov
    );
    DDRL_LOG_DEBUG(g_logger, "  rng_seed: {}", cfg.rng_seed);
  }

  static void log_world_config(const WorldConfig& cfg, std::string_view path)
  {
    DDRL_LOG_DEBUG(g_logger, "Parsed world config from '{}':", path);
    DDRL_LOG_DEBUG(g_logger, "  version: {}", cfg.version);
    DDRL_LOG_DEBUG(g_logger, "  collision_categories: {}", cfg.collision_categories.size());
    for (const auto& cat : cfg.collision_categories) {
      DDRL_LOG_DEBUG(
        g_logger, "    - {} -> [{}]", cat.category, core::join(cat.collides_with, ",")
      );
    }

    DDRL_LOG_DEBUG(g_logger, "  model_profiles: {}", cfg.model_profiles.size());
    for (const auto& entry : cfg.model_profiles) {
      DDRL_LOG_DEBUG(g_logger, "    - {}", entry.first);
    }

    DDRL_LOG_DEBUG(g_logger, "  dynamic_profiles: {}", cfg.dynamic_profiles.size());
    for (const auto& entry : cfg.dynamic_profiles) {
      DDRL_LOG_DEBUG(g_logger, "    - {}", entry.first);
    }

    DDRL_LOG_DEBUG(g_logger, "  robot_profiles: {}", cfg.robot_profiles.size());
    for (const auto& [name, profile] : cfg.robot_profiles) {
      DDRL_LOG_DEBUG(g_logger, "    - {} (model: {})", name, profile.model_id);
    }

    // static_instances and zones are now loaded from map files
    // DDRL_LOG_DEBUG(g_logger, "  static_instances: {}", cfg.static_instances.size());
    // for (const auto& inst : cfg.static_instances) {
    //   DDRL_LOG_DEBUG(
    //     g_logger,
    //     "    - {} (model: {}, pose: [{:.2f}, {:.2f}, {:.2f}])",
    //     inst.name,
    //     inst.model_id,
    //     inst.pose_world.pos.x,
    //     inst.pose_world.pos.y,
    //     inst.pose_world.yaw
    //   );
    // }

    DDRL_LOG_DEBUG(
      g_logger,
      "  robot: {} (profile: {}, delay: {:.2f}s)",
      cfg.robot.name,
      cfg.robot.robot_profile_id,
      cfg.robot.action_delay_s
    );
    // Robot zones are now defined in the active map, not in robot config

    if (std::holds_alternative<DeterministicWorldCfg>(cfg.world)) {
      const auto& det = std::get<DeterministicWorldCfg>(cfg.world);
      DDRL_LOG_DEBUG(g_logger, "  world type: deterministic");
      DDRL_LOG_DEBUG(g_logger, "    dynamic_instances: {}", det.dynamic_instances.size());
      for (const auto& di : det.dynamic_instances) {
        DDRL_LOG_DEBUG(
          g_logger,
          "      - {} (model: {}, profile: {})",
          di.name,
          di.model_id,
          di.dynamic_profile_id
        );
      }
    } else {
      const auto& rnd = std::get<RandomWorldCfg>(cfg.world);
      DDRL_LOG_DEBUG(g_logger, "  world type: random");
      DDRL_LOG_DEBUG(
        g_logger, "    random_dynamic_instances: {}", rnd.random_dynamic_instances.size()
      );
      for (const auto& group : rnd.random_dynamic_instances) {
        DDRL_LOG_DEBUG(
          g_logger,
          "      - {} (model: {}, profile: {}, count: {})",
          group.name,
          group.model_id,
          group.dynamic_profile_id,
          group.count
        );
      }
    }

    DDRL_LOG_DEBUG(g_logger, "Finished parsing world config.");
  }

  static void log_model_profile(const ModelProfile& profile, std::string_view path)
  {
    DDRL_LOG_DEBUG(
      g_logger,
      "Parsed model profile from '{}': {} (category: {}, parts: {})",
      path,
      profile.name,
      profile.category,
      profile.parts.size()
    );
    for (const auto& part : profile.parts) {
      DDRL_LOG_DEBUG(
        g_logger, "  - part '{}': z[{:.2f}, {:.2f}]", part.name, part.geom.z_min, part.geom.z_max
      );
    }
  }

  static void log_robot_profile(const RobotProfile& profile, std::string_view path)
  {
    DDRL_LOG_DEBUG(
      g_logger,
      "Parsed robot profile from '{}': {} (model: {})",
      path,
      profile.name,
      profile.model_id
    );
    DDRL_LOG_DEBUG(g_logger, "  kinematics:");
    DDRL_LOG_DEBUG(
      g_logger,
      "    forward caps: v={:.2f} a={:.2f} decel={:.2f} cmd_rate={:.2f}Hz delay={:.4f}s",
      profile.kinematics.caps_fwd.max_speed,
      profile.kinematics.caps_fwd.max_accel,
      profile.kinematics.caps_fwd.max_decel,
      profile.kinematics.caps_fwd.cmd_rate_hz,
      profile.kinematics.caps_fwd.cmd_delay_s
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    reverse caps: v={:.2f} a={:.2f} decel={:.2f} cmd_rate={:.2f}Hz delay={:.4f}s",
      profile.kinematics.caps_rev.max_speed,
      profile.kinematics.caps_rev.max_accel,
      profile.kinematics.caps_rev.max_decel,
      profile.kinematics.caps_rev.cmd_rate_hz,
      profile.kinematics.caps_rev.cmd_delay_s
    );
    DDRL_LOG_DEBUG(g_logger, "    wheel_base: {:.2f} m", profile.kinematics.wheel_base);
    DDRL_LOG_DEBUG(g_logger, "    max_steer_angle: {:.2f} rad", profile.kinematics.max_steer_angle);
    DDRL_LOG_DEBUG(g_logger, "    max_steer_rate: {:.2f} rad/s", profile.kinematics.max_steer_rate);
    DDRL_LOG_DEBUG(g_logger, "    min_turn_radius: {:.2f} m", profile.kinematics.min_turn_radius);
    DDRL_LOG_DEBUG(
      g_logger, "    gear_hold_time: {:.2f} s", profile.kinematics.gear_hold_time_s
    );
  }

  static void log_visualizer_config(const VizConfig& cfg, std::string_view path)
  {
    DDRL_LOG_DEBUG(g_logger, "Parsed visualizer config from '{}':", path);
    DDRL_LOG_DEBUG(g_logger, "  Backend: VTK");
    DDRL_LOG_DEBUG(
      g_logger, "  Window: {}x{} - {}", cfg.window_width, cfg.window_height, cfg.window_title
    );
    DDRL_LOG_DEBUG(g_logger, "  Performance:");
    DDRL_LOG_DEBUG(g_logger, "    target_fps: {}", cfg.target_fps);
    DDRL_LOG_DEBUG(g_logger, "  Display:");
    DDRL_LOG_DEBUG(g_logger, "  Interactive Style:");
    DDRL_LOG_DEBUG(
      g_logger,
      "    zoom: max_in={:.2f}, max_out={:.2f}",
      cfg.interactor_style.zoom.max_zoom_in,
      cfg.interactor_style.zoom.max_zoom_out
    );
    DDRL_LOG_DEBUG(g_logger, "    pan: max={:.2f}", cfg.interactor_style.pan.max_pan);
    DDRL_LOG_DEBUG(
      g_logger,
      "    rotate: max_pitch={:.2f}, min_pitch={:.2f}",
      cfg.interactor_style.rotate.max_pitch,
      cfg.interactor_style.rotate.min_pitch
    );
    DDRL_LOG_DEBUG(g_logger, "  Camera Follow Settings:");
    DDRL_LOG_DEBUG(g_logger, "    smooth_alpha: {:.2f}", cfg.camera_follow.smooth_alpha);
    DDRL_LOG_DEBUG(
      g_logger,
      "    chase: back={:.2f}m, up={:.2f}m, focal_height={:.2f}m",
      cfg.camera_follow.chase.back_distance,
      cfg.camera_follow.chase.up_distance,
      cfg.camera_follow.chase.focal_height
    );
    DDRL_LOG_DEBUG(g_logger, "    top_down: height={:.2f}m", cfg.camera_follow.top_down.height);
    DDRL_LOG_DEBUG(
      g_logger,
      "    orbit: radius={:.2f}m, height={:.2f}m, speed={:.2f}rad/s, focal_height={:.2f}m",
      cfg.camera_follow.orbit.radius,
      cfg.camera_follow.orbit.height,
      cfg.camera_follow.orbit.speed,
      cfg.camera_follow.orbit.focal_height
    );
    DDRL_LOG_DEBUG(
      g_logger,
      "    first_person: eye_height={:.2f}m, forward_offset={:.2f}m, focal_distance={:.2f}m",
      cfg.camera_follow.first_person.eye_height,
      cfg.camera_follow.first_person.forward_offset,
      cfg.camera_follow.first_person.focal_distance
    );
  }
};

// ============================================================================
// ConfigParser Public Interface Implementation
// ============================================================================

ConfigParser::ConfigParser() : impl_(std::make_unique<Impl>())
{
}

ConfigParser::~ConfigParser() = default;

ConfigParser::ConfigParser(ConfigParser&&) noexcept            = default;
ConfigParser& ConfigParser::operator=(ConfigParser&&) noexcept = default;

core::Result<SimConfig> ConfigParser::parse_simulator_config(std::string_view path)
{
  return impl_->parse_simulator_config(path);
}

core::Result<RLConfig> ConfigParser::parse_rl_config(std::string_view path)
{
  return impl_->parse_rl_config(path);
}

core::Result<RLPolicyConfig::Architecture> ConfigParser::parse_policy_arch_config(
  std::string_view path
)
{
  return impl_->parse_policy_arch_config(path);
}

core::Result<RLRewardConfig> ConfigParser::parse_reward_config(std::string_view path)
{
  return impl_->parse_reward_config(path);
}

core::Result<WorldConfig> ConfigParser::parse_world_config(std::string_view path)
{
  return impl_->parse_world_config(path);
}

core::Result<ModelProfile> ConfigParser::parse_model_profile(std::string_view path)
{
  return impl_->parse_model_profile(path);
}

core::Result<DynamicProfile> ConfigParser::parse_dynamic_profile(std::string_view path)
{
  return impl_->parse_dynamic_profile(path);
}

core::Result<RobotProfile> ConfigParser::parse_robot_profile(std::string_view path)
{
  return impl_->parse_robot_profile(path);
}

core::Result<VizConfig> ConfigParser::parse_visualizer_config(std::string_view path)
{
  return impl_->parse_visualizer_config(path);
}

core::Result<MapConfig> ConfigParser::parse_map_config(std::string_view path)
{
  return impl_->parse_map_config(path);
}

} // namespace ddrl::config
