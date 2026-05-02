#pragma once

#include "core/math/primitives.hpp"
#include <cmath>
#include <format>

namespace ddrl::core::math
{

struct Twist2 {
  Vec2 linear{};       // in local frame
  double angular{0.0}; // yaw rate (rad/s)

  constexpr Twist2() = default;
  constexpr Twist2(const Vec2 &lin, double ang) noexcept
      : linear(lin), angular(ang) {}
};


} // namespace ddrl::core::math

inline std::string to_string(const ddrl::core::math::Twist2 &twist)
{
  return std::format("(linear=({}, {}), angular={})",
                     twist.linear.x,
                     twist.linear.y,
                     twist.angular);
}
