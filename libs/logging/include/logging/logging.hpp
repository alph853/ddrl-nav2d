#pragma once

#include <chrono>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

#ifndef DDRL_ENABLE_LOGGING
#  define DDRL_ENABLE_LOGGING 1
#endif

namespace ddrl::logging {

/**
 * Log levels in order of severity
 */
enum class LogLevel : uint8_t {
  TRACE    = 0,
  DEBUG    = 1,
  INFO     = 2,
  WARN     = 3,
  ERROR    = 4,
  CRITICAL = 5,
  OFF      = 6
};

/**
 * Configuration for logging system initialization
 */
struct LoggingConfig {
  LogLevel    default_level    = LogLevel::INFO;
  bool        log_to_console   = true;
  bool        log_to_file      = true;
  std::string log_file_path    = "ddrl.log";
  bool        async_logging    = false;
  size_t      async_queue_size = 8192;
  std::string console_pattern  = "[%^%l%$] [%n] [%g:%#] %v";
  std::string file_pattern     = "[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%g:%#] %v";
  bool        flush_on_error   = true;
  size_t      max_file_size_mb = 50;
  size_t      max_files        = 3;
  bool        enable_backtrace = false;
  size_t      backtrace_size   = 32;
};

/**
 * Initialize the logging system with configuration.
 * Must be called once at application startup before creating any loggers.
 */
void init_logging(const LoggingConfig& config = LoggingConfig{});

/**
 * Shutdown the logging system and flush all pending logs.
 * Should be called before application exit.
 */
void shutdown_logging();

/**
 * Set global log level for all existing and future loggers.
 */
void set_global_log_level(LogLevel level);

// Forward declarations to hide implementation details
namespace detail {
class LoggerImpl;
}

/**
 * Professional logging class with PIMPL pattern to hide implementation details.
 * Thread-safe and supports throttled logging similar to ROS2's RCLCPP_*_THROTTLE.
 * Uses string_view for efficient string handling without template complexity.
 */
class Logger
{
private:
  std::unique_ptr<detail::LoggerImpl> impl_;

public:
  /**
   * Create a logger with the specified name.
   * @param name Logger name - must be unique and non-empty
   */
  explicit Logger(std::string_view name);
  ~Logger();

  // Move-only semantics to ensure proper resource management
  Logger(const Logger&)            = delete;
  Logger& operator=(const Logger&) = delete;
  Logger(Logger&&) noexcept;
  Logger& operator=(Logger&&) noexcept;

  /**
   * Get the logger name
   */
  [[nodiscard]] std::string_view name() const;

  /**
   * Check if a log level should be logged (for performance-sensitive code)
   */
  [[nodiscard]] bool should_log(LogLevel level) const;

  /**
   * Set the log level for this specific logger
   */
  void set_level(LogLevel level);

  /**
   * Get the current log level for this logger
   */
  [[nodiscard]] LogLevel get_level() const;

  /**
   * Standard logging methods using string_view for efficiency
   */
  void trace(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;
  void debug(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;
  void info(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;
  void warn(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;
  void error(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;
  void critical(
    std::string_view message, const std::source_location& loc = std::source_location::current()
  ) const;

  /**
   * Throttled logging methods - similar to ROS2's RCLCPP_*_THROTTLE
   * These methods will only log once per specified period and track suppressed messages.
   * The throttle_id is used to group related throttled messages together.
   */
  void trace_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;
  void debug_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;
  void info_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;
  void warn_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;
  void error_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;
  void critical_throttle(
    std::chrono::milliseconds period, std::string_view throttle_id, std::string_view message,
    const std::source_location& loc = std::source_location::current()
  ) const;

  /**
   * Force flush all pending log messages for this logger
   */
  void flush();
};

/**
 * Get or create a logger by name.
 * Loggers are cached and reused across calls with the same name.
 * Thread-safe.
 *
 * @param name Logger name - must be non-empty
 * @return Shared pointer to logger instance
 */
std::shared_ptr<Logger> get_logger(std::string_view name);

/**
 * List all currently active logger names
 */
std::vector<std::string> list_logger_names();

/**
 * Remove a logger from the cache (does not affect existing instances)
 */
void drop_logger(std::string_view name);

/**
 * Flush all active loggers
 */
void flush_all();

ddrl::logging::LogLevel log_level_from_string(std::string_view level_str);

} // namespace ddrl::logging

#if DDRL_ENABLE_LOGGING

// Convenience macros for cleaner code - these use std::format for formatting
#define DDRL_LOG_TRACE(logger, fmt, ...)                                                           \
  (logger)->trace(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())
#define DDRL_LOG_DEBUG(logger, fmt, ...)                                                           \
  (logger)->debug(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())
#define DDRL_LOG_INFO(logger, fmt, ...)                                                            \
  (logger)->info(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())
#define DDRL_LOG_WARN(logger, fmt, ...)                                                            \
  (logger)->warn(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())
#define DDRL_LOG_ERROR(logger, fmt, ...)                                                           \
  (logger)->error(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())
#define DDRL_LOG_CRITICAL(logger, fmt, ...)                                                        \
  (logger)->critical(std::format(fmt __VA_OPT__(, ) __VA_ARGS__), std::source_location::current())

// Helper macro to generate unique throttle ID from source location and format string
#define DDRL_THROTTLE_ID(fmt) (__FILE__ ":" + std::to_string(__LINE__) + ":" + std::string(fmt))

#define DDRL_LOG_TRACE_THROTTLE(logger, period_ms, fmt, ...)                                       \
  (logger)->trace_throttle(                                                                        \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )
#define DDRL_LOG_DEBUG_THROTTLE(logger, period_ms, fmt, ...)                                       \
  (logger)->debug_throttle(                                                                        \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )
#define DDRL_LOG_INFO_THROTTLE(logger, period_ms, fmt, ...)                                        \
  (logger)->info_throttle(                                                                         \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )
#define DDRL_LOG_WARN_THROTTLE(logger, period_ms, fmt, ...)                                        \
  (logger)->warn_throttle(                                                                         \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )
#define DDRL_LOG_ERROR_THROTTLE(logger, period_ms, fmt, ...)                                       \
  (logger)->error_throttle(                                                                        \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )
#define DDRL_LOG_CRITICAL_THROTTLE(logger, period_ms, fmt, ...)                                    \
  (logger)->critical_throttle(                                                                     \
    std::chrono::milliseconds(period_ms),                                                          \
    DDRL_THROTTLE_ID(fmt),                                                                         \
    std::format(fmt __VA_OPT__(, ) __VA_ARGS__),                                                   \
    std::source_location::current()                                                                \
  )

#else

#  define DDRL_THROTTLE_ID(fmt) ""

#  define DDRL_LOG_TRACE(logger, fmt, ...) do { } while (0)
#  define DDRL_LOG_DEBUG(logger, fmt, ...) do { } while (0)
#  define DDRL_LOG_INFO(logger, fmt, ...) do { } while (0)
#  define DDRL_LOG_WARN(logger, fmt, ...) do { } while (0)
#  define DDRL_LOG_ERROR(logger, fmt, ...) do { } while (0)
#  define DDRL_LOG_CRITICAL(logger, fmt, ...) do { } while (0)

#  define DDRL_LOG_TRACE_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)
#  define DDRL_LOG_DEBUG_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)
#  define DDRL_LOG_INFO_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)
#  define DDRL_LOG_WARN_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)
#  define DDRL_LOG_ERROR_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)
#  define DDRL_LOG_CRITICAL_THROTTLE(logger, period_ms, fmt, ...) do { } while (0)

#endif
