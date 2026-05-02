#include "worker/config.hpp"

#include <CLI/CLI.hpp>
#include <core/base/result.hpp>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <yaml-cpp/yaml.h>

namespace ddrl::worker {

namespace fs = std::filesystem;

namespace {

core::Result<config::RLPolicyConfig::Type> parse_policy_type(std::string_view value)
{
  if (value == "random") {
    return config::RLPolicyConfig::Type::RANDOM;
  }
  if (value == "neural_network") {
    return config::RLPolicyConfig::Type::NEURAL_NETWORK;
  }
  if (value == "expert") {
    return config::RLPolicyConfig::Type::EXPERT;
  }
  return std::unexpected(core::make_error(
    core::ErrorCode::CONFIG,
    "Unknown policy_type: " + std::string(value),
    core::make_context("parse_worker_config")
  ));
}

} // namespace

std::filesystem::path get_executable_dir(std::string_view argv0)
{
  try {
    return fs::canonical(fs::path(argv0)).parent_path();
  } catch (const fs::filesystem_error&) {
    return fs::current_path();
  }
}

std::string resolve_default_path(std::initializer_list<std::filesystem::path> candidates)
{
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (!candidate.empty() && fs::exists(candidate, ec)) {
      return fs::weakly_canonical(candidate, ec).string();
    }
  }
  std::string paths_str;
  for (const auto& p : candidates) {
    paths_str += p.string() + '\n';
  }
  throw std::runtime_error("No candidate paths resolved.\n" + paths_str);
}

WorkerCLIConfig parse_cli_config(CLI::App& app, std::span<std::string_view> argv)
{
  const auto      exe_dir = get_executable_dir(argv[0]);
  WorkerCLIConfig cli_config;

  cli_config.config_path = resolve_default_path({
    "config/worker.yaml",
    exe_dir / "../share/ddrl/config/worker.yaml",
  });

  cli_config.logging_config_path = resolve_default_path({
    "config/logging.yaml",
    exe_dir / "../share/ddrl/config/logging.yaml",
  });

  app.add_option("-c,--config", cli_config.config_path, "Path to worker configuration YAML")
    ->check(CLI::ExistingFile)
    ->type_name("FILE");

  app
    .add_option(
      "-l,--log-config", cli_config.logging_config_path, "Path to logging configuration YAML"
    )
    ->check(CLI::ExistingFile)
    ->type_name("FILE");

  constexpr auto kMaxUnsignedU16 = static_cast<unsigned>(std::numeric_limits<uint16_t>::max());
  app
    .add_option(
      "--num-sim-instances",
      cli_config.num_sim_instances,
      "Override number of simulator instances (0 keeps auto-detect)"
    )
    ->check(CLI::Range(0u, kMaxUnsignedU16))
    ->type_name("UINT16")
    ->envname("DDRL_WORKER_NUM_SIM_INSTANCES");

  app
    .add_option(
      "--num-raycast-threads",
      cli_config.num_raycast_threads,
      "Override number of raycast threads (0 keeps auto-detect)"
    )
    ->check(CLI::Range(0u, kMaxUnsignedU16))
    ->type_name("UINT16")
    ->envname("DDRL_WORKER_NUM_RAYCAST_THREADS");

  app
    .add_option(
      "--enable-visualization",
      cli_config.enable_visualization,
      "Force visualization on/off (true/false)"
    )
    ->type_name("BOOL")
    ->envname("DDRL_WORKER_ENABLE_VISUALIZATION");

  app
    .add_option(
      "--real-time-factor",
      cli_config.real_time_factor,
      "Override real-time factor (simulation speed multiplier)"
    )
    ->type_name("FLOAT")
    ->envname("DDRL_WORKER_REAL_TIME_FACTOR");

  app
    .add_option(
      "--seed",
      cli_config.seed,
      "Override simulator seed (-1 uses a random seed, otherwise deterministic)"
    )
    ->type_name("INT64")
    ->envname("DDRL_WORKER_SEED");

  app
    .add_option(
      "--coordinator-target", cli_config.coordinator_target, "Coordinator gRPC endpoint (host:port)"
    )
    ->type_name("STRING")
    ->envname("DDRL_COORDINATOR_TARGET");

  app
    .add_option(
      "--worker-identity",
      cli_config.worker_identity,
      "Explicit worker identity (set per instance when running multiple workers on the same host)"
    )
    ->type_name("STRING")
    ->envname("DDRL_WORKER_IDENTITY");

  app
    .add_option(
      "--standalone-run-enable",
      cli_config.standalone_run_enable,
      "Run worker without coordinator/Kafka (true/false)"
    )
    ->type_name("BOOL")
    ->envname("DDRL_WORKER_STANDALONE_RUN_ENABLE");

  app
    .add_option(
      "--deterministic-policy-eval",
      cli_config.deterministic_policy_eval,
      "Evaluate neural policy deterministically with tanh(mean) instead of sampling"
    )
    ->type_name("BOOL")
    ->envname("DDRL_WORKER_DETERMINISTIC_POLICY_EVAL");

  app
    .add_option(
      "--policy-onnx",
      cli_config.policy_onnx_path,
      "Optional ONNX policy file for offline inference"
    )
    ->check(CLI::ExistingFile)
    ->type_name("FILE")
    ->envname("DDRL_WORKER_POLICY_ONNX");

  return cli_config;
}

void log_cli_config(
  const worker::WorkerCLIConfig& cli_config, const std::shared_ptr<logging::Logger>& logger
)
{
  DDRL_LOG_INFO(logger, "Runtime cli_config:");
  DDRL_LOG_INFO(logger, "  config_path = {}", cli_config.config_path);
  DDRL_LOG_INFO(logger, "  logging_config_path = {}", cli_config.logging_config_path);

  if (cli_config.num_sim_instances) {
    DDRL_LOG_INFO(logger, "  num_sim_instances = {}", *cli_config.num_sim_instances);
  }
  if (cli_config.num_raycast_threads) {
    DDRL_LOG_INFO(logger, "  num_raycast_threads = {}", *cli_config.num_raycast_threads);
  }
  if (cli_config.enable_visualization) {
    DDRL_LOG_INFO(
      logger, "  enable_visualization = {}", (*cli_config.enable_visualization ? "true" : "false")
    );
  }
  if (cli_config.real_time_factor) {
    DDRL_LOG_INFO(logger, "  real_time_factor = {}", *cli_config.real_time_factor);
  }
  if (cli_config.seed) {
    DDRL_LOG_INFO(logger, "  seed = {}", *cli_config.seed);
  }
  if (cli_config.coordinator_target) {
    DDRL_LOG_INFO(logger, "  coordinator_target = {}", *cli_config.coordinator_target);
  }
  if (cli_config.standalone_run_enable) {
    DDRL_LOG_INFO(
      logger, "  standalone_run_enable = {}", (*cli_config.standalone_run_enable ? "true" : "false")
    );
  }
  if (cli_config.deterministic_policy_eval) {
    DDRL_LOG_INFO(
      logger,
      "  deterministic_policy_eval = {}",
      (*cli_config.deterministic_policy_eval ? "true" : "false")
    );
  }
  if (cli_config.policy_onnx_path) {
    DDRL_LOG_INFO(logger, "  policy_onnx_path = {}", *cli_config.policy_onnx_path);
  }
}

std::string expand_env(const std::string& input)
{
  std::string result;
  result.reserve(input.size());
  for (std::size_t idx = 0; idx < input.size();) {
    if (input[idx] == '$' && idx + 1 < input.size() && input[idx + 1] == '{') {
      const auto end = input.find('}', idx + 2);
      if (end != std::string::npos) {
        const auto        key         = input.substr(idx + 2, end - (idx + 2));
        const char*       value       = key.empty() ? nullptr : std::getenv(key.c_str());
        const std::string placeholder = input.substr(idx, end - idx + 1);
        if (value != nullptr) {
          result.append(value);
        } else {
          result.append(placeholder);
        }
        idx = end + 1;
        continue;
      }
    }
    result.push_back(input[idx]);
    ++idx;
  }
  return result;
}

core::Result<WorkerConfig> parse_worker_config(std::string_view config_path)
{
  fs::path config_dir = fs::canonical(fs::absolute(config_path)).parent_path();

  try {
    YAML::Node yaml = YAML::LoadFile(std::string(config_path));

    WorkerConfig cfg;

    cfg.num_sim_instances    = yaml["num_sim_instances"].as<uint16_t>(0);
    cfg.num_raycast_threads  = yaml["num_raycast_threads"].as<uint16_t>(0);
    cfg.enable_visualization = yaml["enable_visualization"].as<bool>(true);
    cfg.real_time_factor     = yaml["real_time_factor"].as<double>(1.0);

    fs::path config_p            = yaml["sim_config_path"].as<std::string>();
    fs::path world_config_p      = yaml["world_config_path"].as<std::string>();
    fs::path visualizer_config_p = yaml["visualizer_config_path"].as<std::string>();
    fs::path map_directory_p     = yaml["map_directory"].as<std::string>();

    cfg.sim_config_path           = fs::canonical(config_dir / config_p);
    cfg.world_config_path         = fs::canonical(config_dir / world_config_p);
    cfg.visualizer_config_path    = fs::canonical(config_dir / visualizer_config_p);
    cfg.map_directory             = fs::canonical(config_dir / map_directory_p);
    cfg.unroll_length             = yaml["unroll_length"].as<size_t>(20);
    cfg.max_batch_rollouts        = yaml["max_batch_rollouts"].as<size_t>(32);
    cfg.deterministic_policy_eval = yaml["deterministic_policy_eval"].as<bool>(false);
    if (yaml["policy_onnx_path"]) {
      const auto rel = expand_env(yaml["policy_onnx_path"].as<std::string>());
      if (!rel.empty()) {
        cfg.policy_onnx_path = fs::canonical(config_dir / rel).string();
      }
    }

    const auto policy_type = parse_policy_type(yaml["policy_type"].as<std::string>("random"));
    if (!policy_type) {
      return std::unexpected(policy_type.error());
    }
    cfg.policy_type = *policy_type;

    if (yaml["rl_config_path"]) {
      const auto rl_rel  = expand_env(yaml["rl_config_path"].as<std::string>());
      cfg.rl_config_path = fs::canonical(config_dir / rl_rel).string();
    }
    if (yaml["policy_arch_config_path"] || cfg.rl_config_path.empty()) {
      const auto policy_arch_rel =
        expand_env(yaml["policy_arch_config_path"].as<std::string>("policy_arch.yaml"));
      cfg.policy_arch_config_path = fs::canonical(config_dir / policy_arch_rel).string();
    }
    if (yaml["reward_config_path"] || cfg.rl_config_path.empty()) {
      const auto reward_rel =
        expand_env(yaml["reward_config_path"].as<std::string>("rl_reward.yaml"));
      cfg.reward_config_path = fs::canonical(config_dir / reward_rel).string();
    }

    if (yaml["demos"]) {
      const auto& demos = yaml["demos"];
      const auto  mode  = demos["mode"].as<std::string>("off");
      if (mode == "off") {
        cfg.demos.mode = WorkerConfig::DemoCfg::Mode::OFF;
      } else if (mode == "record_only") {
        cfg.demos.mode = WorkerConfig::DemoCfg::Mode::RECORD_ONLY;
      } else if (mode == "stream_and_record") {
        cfg.demos.mode = WorkerConfig::DemoCfg::Mode::STREAM_AND_RECORD;
      } else {
        return std::unexpected(core::make_error(
          core::ErrorCode::CONFIG,
          "Unknown demos.mode: " + mode,
          core::make_context("parse_worker_config")
        ));
      }
      if (demos["output_dir"]) {
        const auto rel = expand_env(demos["output_dir"].as<std::string>(cfg.demos.output_dir));
        cfg.demos.output_dir = fs::weakly_canonical(config_dir / rel).string();
      } else {
        cfg.demos.output_dir = fs::weakly_canonical(config_dir / cfg.demos.output_dir).string();
      }
      cfg.demos.max_files = demos["max_files"].as<uint32_t>(cfg.demos.max_files);
    } else {
      cfg.demos.output_dir = fs::weakly_canonical(config_dir / cfg.demos.output_dir).string();
    }

    // Event bus
    if (yaml["event_bus"]) {
      cfg.event_bus.buffer_size = yaml["event_bus"]["buffer_size"].as<size_t>(1000);
    }

    if (yaml["coordinator"]) {
      const auto& coord = yaml["coordinator"];
      if (coord["target"]) {
        cfg.coordinator.target =
          expand_env(coord["target"].as<std::string>(cfg.coordinator.target));
      }
      if (coord["worker_identity"]) {
        cfg.coordinator.worker_identity =
          expand_env(coord["worker_identity"].as<std::string>(cfg.coordinator.worker_identity));
      }
      if (coord["hostname"]) {
        cfg.coordinator.hostname =
          expand_env(coord["hostname"].as<std::string>(cfg.coordinator.hostname));
      }
      if (coord["version"]) {
        cfg.coordinator.version =
          expand_env(coord["version"].as<std::string>(cfg.coordinator.version));
      }
      if (coord["capabilities"] && coord["capabilities"].IsMap()) {
        for (const auto& capability : coord["capabilities"]) {
          cfg.coordinator.capabilities.emplace_back(
            capability.first.as<std::string>(), expand_env(capability.second.as<std::string>())
          );
        }
      }
      if (coord["labels"] && coord["labels"].IsMap()) {
        for (const auto& label : coord["labels"]) {
          cfg.coordinator.labels[label.first.as<std::string>()] =
            expand_env(label.second.as<std::string>());
        }
      }
    }

    if (yaml["standalone_run_enable"]) {
      cfg.standalone_run_enable = yaml["standalone_run_enable"].as<bool>(false);
    }

    return cfg;

  } catch (const YAML::Exception& e) {
    return std::unexpected(
      make_error(core::ErrorCode::CONFIG, e.what(), "worker::parse_worker_config")
    );
  } catch (const std::exception& e) {
    return std::unexpected(
      make_error(core::ErrorCode::CONFIG, e.what(), "worker::parse_worker_config")
    );
  }
}

core::Result<logging::LoggingConfig> parse_logging_config(std::string_view config_path)
{
  try {
    YAML::Node yaml = YAML::LoadFile(std::string(config_path));

    logging::LoggingConfig cfg;

    // Parse default log level
    if (yaml["default_level"]) {
      auto level_str    = yaml["default_level"].as<std::string>();
      cfg.default_level = logging::log_level_from_string(level_str);
    }

    // Parse output options
    cfg.log_to_console = yaml["log_to_console"].as<bool>(true);
    cfg.log_to_file    = yaml["log_to_file"].as<bool>(true);
    cfg.log_file_path  = yaml["log_file_path"].as<std::string>("ddrl.log");

    // Parse async options
    cfg.async_logging    = yaml["async_logging"].as<bool>(false);
    cfg.async_queue_size = yaml["async_queue_size"].as<size_t>(8192);

    // Parse patterns
    cfg.console_pattern = yaml["console_pattern"].as<std::string>("[%^%l%$] [%n] [%g:%#] %v");
    cfg.file_pattern =
      yaml["file_pattern"].as<std::string>("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%g:%#] %v");

    // Parse flush and rotation options
    cfg.flush_on_error   = yaml["flush_on_error"].as<bool>(true);
    cfg.max_file_size_mb = yaml["max_file_size_mb"].as<size_t>(50);
    cfg.max_files        = yaml["max_files"].as<size_t>(3);

    // Parse backtrace options
    cfg.enable_backtrace = yaml["enable_backtrace"].as<bool>(false);
    cfg.backtrace_size   = yaml["backtrace_size"].as<size_t>(32);

    return cfg;

  } catch (const YAML::Exception& e) {
    return std::unexpected(
      make_error(core::ErrorCode::CONFIG, e.what(), "worker::parse_logging_config")
    );
  }
}

void apply_cli_config_overrides(WorkerConfig& config, const WorkerCLIConfig& cli_config)
{
  if (cli_config.num_sim_instances) {
    config.num_sim_instances = *cli_config.num_sim_instances;
  }
  if (cli_config.num_raycast_threads) {
    config.num_raycast_threads = *cli_config.num_raycast_threads;
  }
  if (cli_config.enable_visualization) {
    config.enable_visualization = *cli_config.enable_visualization;
  }
  if (cli_config.real_time_factor) {
    config.real_time_factor = *cli_config.real_time_factor;
  }
  if (cli_config.seed) {
    if (*cli_config.seed < -1) {
      throw std::runtime_error("Worker seed override must be -1 or a non-negative integer");
    }
  }
  if (cli_config.coordinator_target) {
    config.coordinator.target = *cli_config.coordinator_target;
  }
  if (cli_config.worker_identity) {
    config.coordinator.worker_identity = *cli_config.worker_identity;
  }
  if (cli_config.standalone_run_enable) {
    config.standalone_run_enable = *cli_config.standalone_run_enable;
  }
  if (cli_config.deterministic_policy_eval) {
    config.deterministic_policy_eval = *cli_config.deterministic_policy_eval;
  }
  if (cli_config.policy_onnx_path) {
    config.policy_onnx_path = *cli_config.policy_onnx_path;
  }
}

} // namespace ddrl::worker
