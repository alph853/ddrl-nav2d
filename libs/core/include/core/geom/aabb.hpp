// ddrl::core::aabb.hpp
#pragma once

#include "core/math/pose.hpp"
#include "core/math/primitives.hpp"
#include <algorithm>
#include <array>
#include <ostream>

namespace ddrl::core::geom {

using math::Point2;
using math::Point3;
using math::Pose2;
using math::Vec2;
using math::Vec3;

struct AABB2 {
  Point2 min, max;

  [[nodiscard]] constexpr Point2 center() const noexcept
  {
    return {(min.x + max.x) * 0.5, (min.y + max.y) * 0.5};
  }
  [[nodiscard]] constexpr Vec2 size() const noexcept { return {max.x - min.x, max.y - min.y}; }
  [[nodiscard]] constexpr bool contains(const Point2& p) const noexcept
  {
    return (p.x >= min.x && p.x <= max.x) && (p.y >= min.y && p.y <= max.y);
  }
  [[nodiscard]] constexpr bool contains(const AABB2& inner) const noexcept
  {
    return (inner.min.x >= min.x &&
          inner.min.y >= min.y &&
          inner.max.x <= max.x &&
          inner.max.y <= max.y);
  }
  [[nodiscard]] constexpr bool fit_in(const AABB2& outer) const noexcept
  {
    return (min.x >= outer.min.x &&
            min.y >= outer.min.y &&
            max.x <= outer.max.x &&
            max.y <= outer.max.y);
  }
  [[nodiscard]] constexpr bool intersects(const AABB2& o) const noexcept
  {
    return (min.x <= o.max.x && max.x >= o.min.x) && (min.y <= o.max.y && max.y >= o.min.y);
  }
  [[nodiscard]] constexpr AABB2 merge(const AABB2& o) const noexcept
  {
    return {
      {std::min(min.x, o.min.x), std::min(min.y, o.min.y)},
      {std::max(max.x, o.max.x), std::max(max.y, o.max.y)}
    };
  }
  constexpr void merge_inplace(const AABB2& o) noexcept
  {
    min.x = std::min(min.x, o.min.x);
    min.y = std::min(min.y, o.min.y);
    max.x = std::max(max.x, o.max.x);
    max.y = std::max(max.y, o.max.y);
  }
  [[nodiscard]] constexpr size_t longest_axis() const noexcept
  {
    double sx = max.x - min.x, sy = max.y - min.y;
    return (sx > sy) ? 0 : 1;
  }
  constexpr void expand_to_include(const Point2& p) noexcept
  {
    min.x = std::min(p.x, min.x);
    min.y = std::min(p.y, min.y);
    max.x = std::max(p.x, max.x);
    max.y = std::max(p.y, max.y);
  }
  [[nodiscard]] constexpr AABB2 transform(const Pose2& pose) const noexcept
  {
    const double c = std::cos(pose.yaw);
    const double s = std::sin(pose.yaw);

    const Point2 c00{min.x, min.y};
    const Point2 c10{max.x, min.y};
    const Point2 c01{min.x, max.y};
    const Point2 c11{max.x, max.y};

    AABB2 out;
    out.min = out.max = pose.transform_point(c00, c, s);

    out.expand_to_include(pose.transform_point(c10, c, s));
    out.expand_to_include(pose.transform_point(c01, c, s));
    out.expand_to_include(pose.transform_point(c11, c, s));

    return out;
  }
  constexpr void transform_inplace(const Pose2& pose) noexcept { *this = transform(pose); }
};

struct AABB3 {
  Point3                         min, max;
  [[nodiscard]] constexpr Point3 center() const noexcept
  {
    return {(min.x + max.x) * 0.5, (min.y + max.y) * 0.5, (min.z + max.z) * 0.5};
  }
  [[nodiscard]] constexpr Vec3 size() const noexcept
  {
    return {max.x - min.x, max.y - min.y, max.z - min.z};
  }
  [[nodiscard]] constexpr bool contains(const Point3& p) const noexcept
  {
    return (p.x >= min.x && p.x <= max.x) && (p.y >= min.y && p.y <= max.y) &&
           (p.z >= min.z && p.z <= max.z);
  }
  [[nodiscard]] constexpr bool intersects(const AABB3& o) const noexcept
  {
    return (min.x <= o.max.x && max.x >= o.min.x) && (min.y <= o.max.y && max.y >= o.min.y) &&
           (min.z <= o.max.z && max.z >= o.min.z);
  }
};

} // namespace ddrl::core::geom

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::geom::AABB2& aabb)
{
  return os << "AABB2{" << aabb.min << ", " << aabb.max << "}";
}

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::geom::AABB3& aabb)
{
  return os << "AABB3{" << aabb.min << ", " << aabb.max << "}";
}
