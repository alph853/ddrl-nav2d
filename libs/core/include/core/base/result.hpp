#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ddrl::core {

enum class ErrorCode : uint8_t {
  CONFIG,
  FILE_IO,
  SIM,
  SENSOR,
  IPC,
  RPC,
  DOMAIN,
  OVERFLOW,
  TIMEOUT,
  CANCELLED,
  TRANSFORM,
  WORKER,
  WORLD,
  EVENT_BUS,
  EVENT_BUS_REGISTERED,
  OUT_OF_MEMORY,
  UNKNOWN
};

struct Error {
  ErrorCode                code;
  std::string              msg;
  std::vector<std::string> ctx_stack;

  Error() = default;
  Error(ErrorCode c_, std::string_view msg_, std::string_view ctx) : code(c_), msg(msg_)
  {
    this->ctx_stack.emplace_back(ctx);
  }
  template <typename... Args>
  Error(ErrorCode                   c,
        std::format_string<Args...> fmt_msg,
        Args&&... args_msg,
        std::format_string<Args...> fmt_ctx,
        Args&&... args_ctx)
      : code(c), msg(std::format(fmt_msg, std::forward<Args>(args_msg)...))
  {
    this->ctx_stack.emplace_back(std::format(fmt_ctx, std::forward<Args>(args_ctx)...));
  }
  template <typename... Args>
  void push_context(std::format_string<Args...> fmt_ctx, Args&&... args_ctx)
  {
    ctx_stack.emplace_back(std::format(fmt_ctx, std::forward<Args>(args_ctx)...));
  }   

  [[nodiscard]] const char* code_string() const noexcept
  {
    switch (code) {
      case ErrorCode::CONFIG:
        return "CONFIG";
      case ErrorCode::FILE_IO:
        return "FILE_IO";
      case ErrorCode::SIM:
        return "SIM";
      case ErrorCode::SENSOR:
        return "SENSOR";
      case ErrorCode::IPC:
        return "IPC";
      case ErrorCode::RPC:
        return "RPC";
      case ErrorCode::DOMAIN:
        return "DOMAIN";
      case ErrorCode::OVERFLOW:
        return "OVERFLOW";   
      case ErrorCode::TIMEOUT:
        return "TIMEOUT";
      case ErrorCode::CANCELLED:
        return "CANCELLED";
      case ErrorCode::UNKNOWN:
        return "UNKNOWN";
      default:
        return "INVALID";
    }
  }
  bool operator==(const Error& other) const noexcept
  {
    return code == other.code && msg == other.msg;
  }
};

template <typename T>
using Result = std::expected<T, Error>;

inline std::string to_string(const Error& e)
{
  std::stringstream ss;
  ss << "[" << e.code_string() << "] " << e.msg << ".\nContext:\n";
  for (const auto& rit : std::ranges::reverse_view(e.ctx_stack)) {
    ss << "-> " << rit << "\n";
  }
  return ss.str();
}

[[nodiscard]] inline Error make_error(ErrorCode c, std::string_view msg, std::string_view ctx)
{
  return Error{c, msg, ctx};
}

template <class... Args>
[[nodiscard]] inline std::string make_context(std::string_view func, Args&&... args)
{
  static_assert(sizeof...(Args) % 2 == 0, "core::make_context requires name/value pairs: name, value, ...");

  std::string out;
  out.reserve(128);
  std::format_to(std::back_inserter(out), "{}(", func);
  auto tup = std::forward_as_tuple(std::forward<Args>(args)...);

  constexpr std::size_t kNpairs = sizeof...(Args) / 2;

  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (
        [&] {
          std::format_to(std::back_inserter(out),
                         "{}={}",
                         std::get<2 * I>(tup),
                         std::get<2 * I + 1>(tup));
          if constexpr (I + 1 < kNpairs) {
            out.append(", ");
          }
        }(),
        ...);
  }(std::make_index_sequence<kNpairs>{});

  out.push_back(')');
  return out;
}

[[nodiscard]] inline Error make_error_from(Error err, ErrorCode new_code, std::string_view new_ctx)
{
  err.push_context("Error: {}\n-> {}", err.code_string(), new_ctx);
  err.code = new_code;
  return err; // NRVO/move elision
}

} // namespace ddrl::core
