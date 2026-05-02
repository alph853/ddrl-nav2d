#pragma once

#include "core/geom/aabb.hpp"
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <utility>

namespace ddrl::core::geom {

struct Shape2;

// 2D circle (immutable radius; bounds is derived)
struct Circle {
  double radius{0.0};

  Circle() = default;
  explicit Circle(double r);

  [[nodiscard]] const AABB2& bounds() const noexcept;
private:
  mutable std::optional<AABB2> bounds_cache_;
};

// Bounding box with half extents hx, hy (bounds is trivial)
struct Box2 {
  double hx{0.0}, hy{0.0};

  Box2() = default;
  Box2(double hx_, double hy_);

  [[nodiscard]] const AABB2& bounds() const noexcept;
private:
  mutable std::optional<AABB2> bounds_cache_;
};

// 2D polygon CCW winding vertices
struct Polygon2 {
  std::vector<math::Point2> v;

  Polygon2() = default;
  explicit Polygon2(std::vector<math::Point2> verts);

  // Lazily computed/cached bounds (no explicit dirty flag needed)
  [[nodiscard]] const AABB2& bounds() const;

  // Check if a point is inside the polygon using ray casting algorithm
  [[nodiscard]] bool contains_point(const math::Point2& point) const;

  // Check if two line segments intersect
  [[nodiscard]] static bool segments_intersect(
    const math::Point2& p1, const math::Point2& p2,
    const math::Point2& p3, const math::Point2& p4);

  // Check if this polygon intersects with another polygon
  [[nodiscard]] bool check_intersects(const Polygon2& other) const;

  [[nodiscard]] bool contains_shape(const Shape2& shape, const Pose2& shape_pose) const;

private:
  mutable std::optional<AABB2> bounds_cache_;
};

struct Shape2 {
  using Variant = std::variant<Circle, Box2, Polygon2>;
  Variant value;

  Shape2() = default;

  // Convenience constructors
  Shape2(const Circle& c)   : value(c) {}
  Shape2(const Box2& b)     : value(b) {}
  Shape2(const Polygon2& p) : value(p) {}

  Shape2(Circle&& c)   : value(std::move(c)) {}
  Shape2(Box2&& b)     : value(std::move(b)) {}
  Shape2(Polygon2&& p) : value(std::move(p)) {}

  template <class Visitor>
  decltype(auto) visit(Visitor&& vis) {
    return std::visit(std::forward<Visitor>(vis), value);
  }
  template <class Visitor>
  decltype(auto) visit(Visitor&& vis) const {
    return std::visit(std::forward<Visitor>(vis), value);
  }
};

// Dispatch that uses the concrete bounds() methods above.
AABB2 get_bounds(const Shape2& shape);

struct Shape2p5 {
  Shape2 base; // 2D base shape
  double z_min{0.0};
  double z_max{0.0};

  [[nodiscard]] double height() const { return z_max - z_min; }
};

// normalized direction preferred
struct Ray2 {
  math::Point2 origin;
  math::Vec2   direction;

  [[nodiscard]] constexpr math::Point2 at(double t) const noexcept
  {
    return origin + direction * t;
  }
};

// normalized direction preferred
struct Ray3 {
  Point3 origin;
  Vec3   direction;

  [[nodiscard]] constexpr Point3 at(double t) const noexcept { return origin + direction * t; }
};

} // namespace ddrl::core::geom

std::string to_string(const ddrl::core::geom::Polygon2& poly);
