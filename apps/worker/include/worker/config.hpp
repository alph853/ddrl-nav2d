#pragma once

#include <CLI/CLI.hpp>
#include <core/base/result.hpp>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

#include "logging/logging.hpp"
#include "config/rl.hpp"

namespace ddrl::worker {

/**
 * @brief Worker configuration loaded from YAML
 */
struct WorkerConfig {
  uint16_t num_sim_instances{0};   ///< 0 = auto-detect
  uint16_t num_raycast_threads{0}; ///< 0 = auto-detect
  bool     enable_visualization{true};
  double   real_time_factor{1.0}; ///< Real-time factor for simulation speed (1.0 = real-time)

  std::string sim_config_path;
  std::string world_config_path;
  std::string visualizer_config_path;
  bool        standalone_run_enable{false};

  std::string map_directory{"config/map"}; ///< Directory containing map preset files
  size_t      unroll_length{20};           ///< Steps per rollout/unroll
  size_t      max_batch_rollouts{32};      ///< Rollouts per streamed RolloutBatch
  std::string policy_onnx_path;            ///< Optional ONNX policy path for offline inference
  bool        deterministic_policy_eval{false}; ///< Use tanh(mean) instead of stochastic sampling.

  struct EventBusCfg {
    size_t buffer_size{1000};
  };

  struct CoordinatorCfg {
    std::string target{"127.0.0.1:5700"};
    std::string worker_identity;
    std::string hostname;
    std::string version;

    std::vector<std::pair<std::string, std::string>> capabilities;
    std::map<std::string, std::string>               labels;
  };

  config::RLPolicyConfig::Type policy_type{config::RLPolicyConfig::Type::RANDOM};
  std::string                  policy_arch_config_path;
  std::string                  reward_config_path;
  std::string                  rl_config_path;
  EventBusCfg    event_bus;
  CoordinatorCfg coordinator;

  struct DemoCfg {
    enum class Mode : uint8_t { OFF, RECORD_ONLY, STREAM_AND_RECORD };
    Mode        mode{Mode::OFF};
    std::string output_dir{"data/demos"};
    uint32_t    max_files{20000};
  };

  DemoCfg demos{};
};

/**
 * @brief Optional overrides injected via CLI or environment.
 */
struct WorkerCLIConfig {
  std::string config_path;
  std::string logging_config_path;

  std::optional<uint16_t>    num_sim_instances;
  std::optional<uint16_t>    num_raycast_threads;
  std::optional<bool>        enable_visualization;
  std::optional<double>      real_time_factor;
  std::optional<int64_t>     seed;
  std::optional<std::string> coordinator_target;
  std::optional<std::string> worker_identity;
  std::optional<bool>        standalone_run_enable;
  std::optional<bool>        deterministic_policy_eval;
  std::optional<std::string> policy_onnx_path;
};

core::Result<WorkerConfig>           parse_worker_config(std::string_view config_path);
core::Result<logging::LoggingConfig> parse_logging_config(std::string_view config_path);

std::filesystem::path get_executable_dir(const char* argv0);

std::string resolve_default_path(
  std::initializer_list<std::filesystem::path> candidates, std::string_view fallback
);

WorkerCLIConfig parse_cli_config(CLI::App& app, std::span<std::string_view> argv);

void log_cli_config(
  const WorkerCLIConfig& cli_config, const std::shared_ptr<logging::Logger>& logger
);

void apply_cli_config_overrides(WorkerConfig& config, const WorkerCLIConfig& cli_config);

} // namespace ddrl::worker
