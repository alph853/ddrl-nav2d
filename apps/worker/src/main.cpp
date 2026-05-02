#include <CLI/CLI.hpp>
#include <csignal>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "logging/logging.hpp"
#include "worker/config.hpp"
#include "worker/worker.hpp"

using namespace ddrl;

// Global worker instance for signal handling
static std::unique_ptr<worker::Worker> g_worker = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    if (g_worker != nullptr) {
      g_worker->request_stop();
    }
  }
}

int main(int argc, char* argv[])
{
  CLI::App app{"DDRL Worker Agent"};
  app.footer("Example: ./bin --config config/worker.yaml --log-config config/logging.yaml");

  const std::string version = "1.0.0";
  app.set_version_flag("--version", version);

  std::vector<std::string_view> args;
  args.reserve(static_cast<size_t>(argc));
  for (int idx = 0; idx < argc; ++idx) {
    args.emplace_back(argv[idx]);
  }

  auto cli_config = worker::parse_cli_config(app, std::span(args));

  try {
    CLI11_PARSE(app, argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  std::shared_ptr<logging::Logger> logger;

  try {
    // Parse logging configuration
    auto log_config_result = worker::parse_logging_config(cli_config.logging_config_path);
    if (!log_config_result) {
      std::cerr << "Failed to parse logging config: " << to_string(log_config_result.error())
                << '\n';
      return EXIT_FAILURE;
    }

    logging::init_logging(log_config_result.value());
    logger = logging::get_logger("worker");

    DDRL_LOG_INFO(logger, "========================================");
    DDRL_LOG_INFO(logger, "DDRL Worker Agent v{}", version);
    DDRL_LOG_INFO(logger, "========================================");
    DDRL_LOG_INFO(logger, "");
    log_cli_config(cli_config, logger);

    // Create worker
    worker::Worker::Params params;
    params.config_path = cli_config.config_path;
    params.overrides   = cli_config;
    params.version     = version;

    g_worker = std::make_unique<worker::Worker>(std::move(params));

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (auto init_result = g_worker->init(); !init_result) {
      DDRL_LOG_ERROR(logger, "Failed to initialize worker: {}", to_string(init_result.error()));
      return EXIT_FAILURE;
    }

    auto run_result  = g_worker->run(); // Blocks until calling request_stop()
    auto stop_result = g_worker->stop();
    g_worker.reset();

    if (!run_result) {
      DDRL_LOG_ERROR(logger, "Worker runtime error: {}", to_string(run_result.error()));
      logging::shutdown_logging();
      return EXIT_FAILURE;
    }

    if (!stop_result) {
      DDRL_LOG_ERROR(logger, "Worker shutdown error: {}", to_string(stop_result.error()));
      logging::shutdown_logging();
      return EXIT_FAILURE;
    }

    logging::shutdown_logging();
    DDRL_LOG_INFO(logger, "Worker shutdown complete");
    return EXIT_SUCCESS;

  } catch (const std::exception& e) {
    if (logger) {
      DDRL_LOG_ERROR(logger, "Fatal error: {}", e.what());
      logging::shutdown_logging();
    } else {
      std::cerr << "Fatal error before logging initialized: " << e.what() << '\n';
    }
    return EXIT_FAILURE;
  }
}
