#include "logging/logging.hpp"

#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace ddrl::logging {
namespace detail {

/**
 * Concrete implementation of LoggerImpl using spdlog
 */
class LoggerImpl
{
private:
  std::shared_ptr<spdlog::logger> logger_;

  // Throttling state
  struct ThrottleState {
    std::chrono::steady_clock::time_point last_log_time;
    size_t                                suppressed_count = 0;
    std::mutex                            mutex;
  };

  // Global throttle state map
  static std::unordered_map<std::string, std::unique_ptr<ThrottleState>> g_throttle_states;
  static std::mutex                                                      g_throttle_mutex;

  [[nodiscard]] static spdlog::level::level_enum convert_to_spdlog(LogLevel level)
  {
    switch (level) {
      case LogLevel::TRACE:
        return spdlog::level::trace;
      case LogLevel::DEBUG:
        return spdlog::level::debug;
      case LogLevel::INFO:
        return spdlog::level::info;
      case LogLevel::WARN:
        return spdlog::level::warn;
      case LogLevel::ERROR:
        return spdlog::level::err;
      case LogLevel::CRITICAL:
        return spdlog::level::critical;
      case LogLevel::OFF:
        return spdlog::level::off;
      default:
        return spdlog::level::info;
    }
  }

  [[nodiscard]] static LogLevel convert_from_spdlog(spdlog::level::level_enum level)
  {
    switch (level) {
      case spdlog::level::trace:
        return LogLevel::TRACE;
      case spdlog::level::debug:
        return LogLevel::DEBUG;
      case spdlog::level::info:
        return LogLevel::INFO;
      case spdlog::level::warn:
        return LogLevel::WARN;
      case spdlog::level::err:
        return LogLevel::ERROR;
      case spdlog::level::critical:
        return LogLevel::CRITICAL;
      case spdlog::level::off:
        return LogLevel::OFF;
      default:
        return LogLevel::INFO;
    }
  }

public:
  explicit LoggerImpl(const std::string& name)
  {
    if (auto existing_logger = spdlog::get(name)) {
      logger_ = existing_logger;
    } else {
      // Inherit sinks from default logger
      if (spdlog::default_logger()) {
        auto sinks = spdlog::default_logger()->sinks();
        logger_    = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
      } else {
        // Fallback: create with console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        logger_           = std::make_shared<spdlog::logger>(name, console_sink);
      }
      spdlog::register_logger(logger_);
    }
  }

  [[nodiscard]] const std::string& name() const { return logger_->name(); }

  [[nodiscard]] bool should_log(LogLevel level) const
  {
    return logger_->should_log(convert_to_spdlog(level));
  }

  void set_level(LogLevel level) { logger_->set_level(convert_to_spdlog(level)); }

  [[nodiscard]] LogLevel get_level() const { return convert_from_spdlog(logger_->level()); }

  void flush() { logger_->flush(); }

  void log_trace(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::trace,
        "{}",
        message);
  }

  void log_debug(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::debug,
        "{}",
        message);
  }

  void log_info(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::info,
        "{}",
        message);
  }

  void log_warn(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::warn,
        "{}",
        message);
  }

  void log_error(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::err,
        "{}",
        message);
  }

  void log_critical(std::string_view message, const std::source_location& loc)
  {
    logger_->log(
        spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
        spdlog::level::critical,
        "{}",
        message);
  }

  static bool should_log_throttled(const std::string& key, std::chrono::milliseconds period)
  {
    std::lock_guard<std::mutex> lock(g_throttle_mutex);

    auto& state = g_throttle_states[key];
    if (!state) {
      state = std::make_unique<ThrottleState>();
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state->last_log_time);

    if (elapsed >= period) {
      state->last_log_time = now;
      return true;
    }
    state->suppressed_count++;
    return false;
  }

  void log_suppressed_count(const std::string& key, LogLevel /* level */)
  {
    std::lock_guard<std::mutex> lock(g_throttle_mutex);

    auto it = g_throttle_states.find(key);
    if (it != g_throttle_states.end() && it->second->suppressed_count > 0) {
      auto count                   = it->second->suppressed_count;
      it->second->suppressed_count = 0;

      // Log suppressed count at debug level to avoid spam
      if (logger_->should_log(spdlog::level::debug)) {
        logger_->debug("(suppressed {} similar messages)", count);
      }
    }
  }

  void log_trace_throttle(std::chrono::milliseconds   period,
                          const std::string&          throttle_id,
                          std::string_view            message,
                          const std::source_location& loc)
  {
    auto key = std::string(logger_->name()) + ":trace:" + throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::TRACE)) {
        log_trace(message, loc);
      }
      log_suppressed_count(key, LogLevel::TRACE);
    }
  }

  void log_debug_throttle(std::chrono::milliseconds   period,
                          const std::string&          throttle_id,
                          std::string_view            message,
                          const std::source_location& loc)
  {
    auto key = std::string(logger_->name()) + ":debug:" + throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::DEBUG)) {
        log_debug(message, loc);
      }
      log_suppressed_count(key, LogLevel::DEBUG);
    }
  }

  void log_info_throttle(std::chrono::milliseconds   period,
                         const std::string&          throttle_id,
                         std::string_view            message,
                         const std::source_location& loc)
  {
    auto key = std::string(logger_->name());
    key += ":info:";
    key += throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::INFO)) {
        log_info(message, loc);
      }
      log_suppressed_count(key, LogLevel::INFO);
    }
  }

  void log_warn_throttle(std::chrono::milliseconds   period,
                         const std::string&          throttle_id,
                         std::string_view            message,
                         const std::source_location& loc)
  {
    auto key = std::string(logger_->name());
    key += ":warn:";
    key += throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::WARN)) {
        log_warn(message, loc);
      }
      log_suppressed_count(key, LogLevel::WARN);
    }
  }

  void log_error_throttle(std::chrono::milliseconds   period,
                          const std::string&          throttle_id,
                          std::string_view            message,
                          const std::source_location& loc)
  {
    auto key = std::string(logger_->name());
    key += ":error:";
    key += throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::ERROR)) {
        log_error(message, loc);
      }
      log_suppressed_count(key, LogLevel::ERROR);
    }
  }

  void log_critical_throttle(std::chrono::milliseconds   period,
                             const std::string&          throttle_id,
                             std::string_view            message,
                             const std::source_location& loc)
  {
    auto key = std::string(logger_->name());
    key += ":critical:";
    key += throttle_id;
    if (should_log_throttled(key, period)) {
      if (should_log(LogLevel::CRITICAL)) {
        log_critical(message, loc);
      }
      log_suppressed_count(key, LogLevel::CRITICAL);
    }
  }
};

// Static member definitions
std::unordered_map<std::string, std::unique_ptr<LoggerImpl::ThrottleState>>
           LoggerImpl::g_throttle_states;
std::mutex LoggerImpl::g_throttle_mutex;

} // namespace detail

// Global state for logger management
static std::unordered_map<std::string, std::weak_ptr<Logger>> g_logger_cache;
static std::mutex                                             g_logger_cache_mutex;
static bool                                                   g_logging_initialized = false;

void init_logging(const LoggingConfig& config)
{
  std::vector<spdlog::sink_ptr> sinks;

  if (config.log_to_console) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern(config.console_pattern);
    sinks.push_back(console_sink);
  }

  if (config.log_to_file) {
    std::shared_ptr<spdlog::sinks::base_sink<std::mutex>> file_sink;

    if (config.max_file_size_mb > 0) {
      // Use rotating file sink
      file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(config.log_file_path,
                                                                         config.max_file_size_mb *
                                                                             1024 * 1024,
                                                                         config.max_files);
    } else {
      // Use basic file sink
      file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true);
    }

    file_sink->set_pattern(config.file_pattern);
    sinks.push_back(file_sink);
  }

  if (sinks.empty()) {
    // Fallback: at least have console output
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern(config.console_pattern);
    sinks.push_back(console_sink);
  }

  if (config.async_logging) {
    spdlog::init_thread_pool(config.async_queue_size, 1);
    auto logger = std::make_shared<spdlog::async_logger>("ddrl",
                                                         sinks.begin(),
                                                         sinks.end(),
                                                         spdlog::thread_pool());
    spdlog::set_default_logger(logger);
  } else {
    auto logger = std::make_shared<spdlog::logger>("ddrl", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
  }

  // Configure global settings
  spdlog::set_level(static_cast<spdlog::level::level_enum>(static_cast<int>(config.default_level)));

  if (config.flush_on_error) {
    spdlog::flush_on(spdlog::level::err);
  }

  if (config.enable_backtrace) {
    spdlog::enable_backtrace(config.backtrace_size);
  }

  g_logging_initialized = true;
}

void shutdown_logging()
{
  flush_all();
  spdlog::shutdown();
  g_logging_initialized = false;
}

void set_global_log_level(LogLevel level)
{
  spdlog::set_level(static_cast<spdlog::level::level_enum>(static_cast<int>(level)));
}

// Logger implementation

Logger::Logger(std::string_view name)
    : impl_(std::make_unique<detail::LoggerImpl>(std::string{name}))
{
  if (name.empty()) {
    throw std::invalid_argument("Logger name cannot be empty");
  }
}

Logger::Logger(Logger&&) noexcept            = default;
Logger& Logger::operator=(Logger&&) noexcept = default;
Logger::~Logger()                            = default;

std::string_view Logger::name() const
{
  return impl_->name();
}

bool Logger::should_log(LogLevel level) const
{
  return impl_->should_log(level);
}

void Logger::set_level(LogLevel level)
{
  impl_->set_level(level);
}

LogLevel Logger::get_level() const
{
  return impl_->get_level();
}

void Logger::flush()
{
  impl_->flush();
}

// Standard logging methods
void Logger::trace(std::string_view message, const std::source_location& loc) const
{
  impl_->log_trace(message, loc);
}

void Logger::debug(std::string_view message, const std::source_location& loc) const
{
  impl_->log_debug(message, loc);
}

void Logger::info(std::string_view message, const std::source_location& loc) const
{
  impl_->log_info(message, loc);
}

void Logger::warn(std::string_view message, const std::source_location& loc) const
{
  impl_->log_warn(message, loc);
}

void Logger::error(std::string_view message, const std::source_location& loc) const
{
  impl_->log_error(message, loc);
}

void Logger::critical(std::string_view message, const std::source_location& loc) const
{
  impl_->log_critical(message, loc);
}

// Throttled logging methods
void Logger::trace_throttle(std::chrono::milliseconds   period,
                            std::string_view            throttle_id,
                            std::string_view            message,
                            const std::source_location& loc) const
{
  impl_->log_trace_throttle(period, std::string{throttle_id}, message, loc);
}

void Logger::debug_throttle(std::chrono::milliseconds   period,
                            std::string_view            throttle_id,
                            std::string_view            message,
                            const std::source_location& loc) const
{
  impl_->log_debug_throttle(period, std::string{throttle_id}, message, loc);
}

void Logger::info_throttle(std::chrono::milliseconds   period,
                           std::string_view            throttle_id,
                           std::string_view            message,
                           const std::source_location& loc) const
{
  impl_->log_info_throttle(period, std::string{throttle_id}, message, loc);
}

void Logger::warn_throttle(std::chrono::milliseconds   period,
                           std::string_view            throttle_id,
                           std::string_view            message,
                           const std::source_location& loc) const
{
  impl_->log_warn_throttle(period, std::string{throttle_id}, message, loc);
}

void Logger::error_throttle(std::chrono::milliseconds   period,
                            std::string_view            throttle_id,
                            std::string_view            message,
                            const std::source_location& loc) const
{
  impl_->log_error_throttle(period, std::string{throttle_id}, message, loc);
}

void Logger::critical_throttle(std::chrono::milliseconds   period,
                               std::string_view            throttle_id,
                               std::string_view            message,
                               const std::source_location& loc) const
{
  impl_->log_critical_throttle(period, std::string{throttle_id}, message, loc);
}

// Global functions

std::shared_ptr<Logger> get_logger(std::string_view name)
{
  if (name.empty()) {
    throw std::invalid_argument("Logger name cannot be empty");
  }

  if (!g_logging_initialized) {
    // Auto-initialize with default config
    init_logging();
  }

  std::lock_guard<std::mutex> lock(g_logger_cache_mutex);

  // Check if we have a cached logger
  auto key = std::string{name};
  auto it  = g_logger_cache.find(key);
  if (it != g_logger_cache.end()) {
    if (auto logger = it->second.lock()) {
      return logger;
    }
    // Remove expired weak_ptr
    g_logger_cache.erase(it);
  }

  // Create new logger
  auto logger          = std::make_shared<Logger>(name);
  g_logger_cache[key]  = logger;
  return logger;
}

std::vector<std::string> list_logger_names()
{
  std::lock_guard<std::mutex> lock(g_logger_cache_mutex);
  std::vector<std::string>    names;

  for (auto it = g_logger_cache.begin(); it != g_logger_cache.end();) {
    if (it->second.expired()) {
      it = g_logger_cache.erase(it);
    } else {
      names.push_back(it->first);
      ++it;
    }
  }

  return names;
}

void drop_logger(std::string_view name)
{
  std::lock_guard<std::mutex> lock(g_logger_cache_mutex);
  auto                        key = std::string{name};
  g_logger_cache.erase(key);
  spdlog::drop(key);
}

void flush_all()
{
  spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& logger) { logger->flush(); });
}

ddrl::logging::LogLevel log_level_from_string(std::string_view level_str)
{
  if (level_str == "TRACE") {
    return ddrl::logging::LogLevel::TRACE;
  }
  if (level_str == "DEBUG") {
    return ddrl::logging::LogLevel::DEBUG;
  }
  if (level_str == "INFO") {
    return ddrl::logging::LogLevel::INFO;
  }
  if (level_str == "WARN") {
    return ddrl::logging::LogLevel::WARN;
  }
  if (level_str == "ERROR") {
    return ddrl::logging::LogLevel::ERROR;
  }
  if (level_str == "CRITICAL") {
    return ddrl::logging::LogLevel::CRITICAL;
  }
  if (level_str == "OFF") {
    return ddrl::logging::LogLevel::OFF;
  }
  throw std::invalid_argument(std::format("Invalid log level string: '{}'", level_str));
}

} // namespace ddrl::logging
