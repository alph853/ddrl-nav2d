#include "core/base/variant_overload.hpp"
#include "core/math/numeric.hpp" // Added for moved helpers
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#include "collision/narrow_phase.hpp"

using ddrl::core::geom::Shape2p5;
using ddrl::core::math::centroid;
using ddrl::core::math::closest_point_on_polygon;
using ddrl::core::math::edge_normal;
using ddrl::core::math::kEps;
using ddrl::core::math::Point2;
using ddrl::core::math::Pose2;
using ddrl::core::math::Vec2;
using ddrl::core::math::z_ranges_overlap;

namespace ddrl::collision {

namespace detail {

struct Projection {
  double min;
  double max;
};

static Projection project_polygon(std::span<const Point2> poly, const Vec2& axis)
{
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  for (const auto& p : poly) {
    const double val = axis.dot({.x = p.x, .y = p.y});
    min              = std::min(min, val);
    max              = std::max(max, val);
  }
  return {.min = min, .max = max};
}

static Projection project_circle(const Point2& center, double radius, const Vec2& axis)
{
  const double c = axis.dot({.x = center.x, .y = center.y});
  return {.min = c - radius, .max = c + radius};
}

static bool overlap_1d(const Projection& a, const Projection& b, double& penetration)
{
  const double min_max = std::min(a.max, b.max);
  const double max_min = std::max(a.min, b.min);
  penetration          = min_max - max_min;
  return penetration > 0.0;
}

// Forward declarations for shape conversion utilities
static std::vector<Point2>        to_polygon(const core::geom::Box2& box);
static const std::vector<Point2>& to_polygon(const core::geom::Polygon2& poly);
static std::vector<Point2> transform_polygon(std::span<const Point2> verts, const Pose2& pose);

static CollisionResult sat_polygon_polygon(std::span<const Point2> a, std::span<const Point2> b)
{
  CollisionResult res;
  if (a.empty() || b.empty()) {
    return res;
  }

  double       min_pen = std::numeric_limits<double>::infinity();
  Vec2         best_axis{};
  const Point2 ca = centroid(a);
  const Point2 cb = centroid(b);

  auto test_axes = [&](std::span<const Point2> poly) -> bool {
    for (std::size_t i = 0; i < poly.size(); ++i) {
      const Vec2 axis = edge_normal(poly[i], poly[(i + 1) % poly.size()]);
      if (axis.length_sq() <= kEps) {
        continue;
      }
      const auto proj_a = project_polygon(a, axis);
      const auto proj_b = project_polygon(b, axis);
      double     pen    = 0.0;
      if (!overlap_1d(proj_a, proj_b, pen)) {
        return false;
      }
      if (pen < min_pen) {
        min_pen            = pen;
        const Vec2 centers = Vec2{cb.x, cb.y} - Vec2{ca.x, ca.y};
        best_axis          = (centers.dot(axis) < 0.0) ? -axis : axis;
      }
    }
    return true;
  };

  if (!test_axes(a) || !test_axes(b)) {
    return res;
  }

  if (best_axis.length_sq() <= kEps) {
    best_axis = {1.0, 0.0};
  }
  res.intersects        = true;
  res.penetration_depth = min_pen;
  res.normal            = best_axis.normalized();
  return res;
}

static CollisionResult
sat_circle_circle(const Point2& a_center, double a_radius, const Point2& b_center, double b_radius)
{
  CollisionResult res;
  const Vec2      delta   = b_center - a_center;
  const double    dist_sq = delta.length_sq();
  const double    r_sum   = a_radius + b_radius;
  if (dist_sq >= r_sum * r_sum) {
    return res;
  }
  const double dist     = std::sqrt(std::max(dist_sq, 0.0));
  res.intersects        = true;
  res.penetration_depth = r_sum - dist;
  res.normal            = dist > kEps ? (delta / dist) : Vec2{1.0, 0.0};
  return res;
}

static CollisionResult
sat_polygon_circle(std::span<const Point2> poly, const Point2& center, double radius)
{
  CollisionResult res;
  if (poly.empty()) {
    return res;
  }

  double       min_pen = std::numeric_limits<double>::infinity();
  Vec2         best_axis{};
  const Point2 poly_centroid = centroid(poly);

  for (std::size_t i = 0; i < poly.size(); ++i) {
    const Vec2 axis = edge_normal(poly[i], poly[(i + 1) % poly.size()]);
    if (axis.length_sq() <= kEps) {
      continue;
    }
    const auto proj_poly   = project_polygon(poly, axis);
    const auto proj_circle = project_circle(center, radius, axis);
    double     pen         = 0.0;
    if (!overlap_1d(proj_poly, proj_circle, pen)) {
      return res;
    }
    if (pen < min_pen) {
      min_pen            = pen;
      const Vec2 centers = Vec2{center.x, center.y} - Vec2{poly_centroid.x, poly_centroid.y};
      best_axis          = (centers.dot(axis) < 0.0) ? -axis : axis;
    }
  }

  const Point2 closest  = closest_point_on_polygon(poly, center);
  Vec2         axis     = Vec2{center.x - closest.x, center.y - closest.y};
  double       axis_len = axis.length();
  if (axis_len <= kEps) {
    axis     = best_axis.length_sq() > kEps ? best_axis : Vec2{1.0, 0.0};
    axis_len = axis.length();
  }

  if (axis_len > kEps) {
    axis /= axis_len;
    const auto proj_poly   = project_polygon(poly, axis);
    const auto proj_circle = project_circle(center, radius, axis);
    double     pen         = 0.0;
    if (!overlap_1d(proj_poly, proj_circle, pen)) {
      return res;
    }
    if (pen < min_pen) {
      min_pen = pen;
      best_axis =
        (Vec2{center.x, center.y} - Vec2{poly_centroid.x, poly_centroid.y}).dot(axis) < 0.0 ? -axis
                                                                                            : axis;
    }
  }

  if (best_axis.length_sq() <= kEps) {
    best_axis = {1.0, 0.0};
  }
  res.intersects        = true;
  res.penetration_depth = min_pen;
  res.normal            = best_axis.normalized();
  return res;
}

// --- Shape conversion helpers (local to SAT implementation) ---
static std::vector<Point2> to_polygon(const core::geom::Box2& box)
{
  return {
    {Point2{-box.hx, -box.hy},
     Point2{box.hx, -box.hy},
     Point2{box.hx, box.hy},
     Point2{-box.hx, box.hy}}
  };
}
static const std::vector<Point2>& to_polygon(const core::geom::Polygon2& poly)
{
  return poly.v;
}
static std::vector<Point2> transform_polygon(std::span<const Point2> verts, const Pose2& pose)
{
  std::vector<Point2> out;
  out.reserve(verts.size());
  for (const auto& p : verts) {
    out.push_back(pose.transform_point(p));
  }
  return out;
}

} // namespace detail

CollisionResult
collide_shapes(const Shape2p5& a, const Pose2& a_world, const Shape2p5& b, const Pose2& b_world)
{
  CollisionResult res;

  if (!z_ranges_overlap(a.z_min, a.z_max, b.z_min, b.z_max)) {
    return res;
  }

  auto shape_base_a = a.base.value;
  auto shape_base_b = b.base.value;

  return std::visit(
    core::Overloaded{
      [&](const core::geom::Circle& circle_a, const core::geom::Circle& circle_b)
        -> CollisionResult {
        const Point2 ca = a_world.transform_point({0.0, 0.0});
        const Point2 cb = b_world.transform_point({0.0, 0.0});
        return detail::sat_circle_circle(ca, circle_a.radius, cb, circle_b.radius);
      },
      [&](const core::geom::Circle& circle_a, const auto& shape_b) -> CollisionResult {
        const Point2 ca     = a_world.transform_point({0.0, 0.0});
        const auto   poly_b = detail::transform_polygon(detail::to_polygon(shape_b), b_world);
        auto         result = detail::sat_polygon_circle(poly_b, ca, circle_a.radius);
        if (result.intersects) {
          result.normal = -result.normal;
          if (result.normal.length_sq() > kEps) {
            result.normal = result.normal.normalized();
          } else {
            result.normal = {1.0, 0.0};
          }
        }
        return result;
      },
      [&](const auto& shape_a, const core::geom::Circle& circle_b) -> CollisionResult {
        const Point2 cb     = b_world.transform_point({0.0, 0.0});
        const auto   poly_a = detail::transform_polygon(detail::to_polygon(shape_a), a_world);
        return detail::sat_polygon_circle(poly_a, cb, circle_b.radius);
      },
      [&](const auto& shape_a, const auto& shape_b) -> CollisionResult {
        const auto poly_a = detail::transform_polygon(detail::to_polygon(shape_a), a_world);
        const auto poly_b = detail::transform_polygon(detail::to_polygon(shape_b), b_world);
        return detail::sat_polygon_polygon(poly_a, poly_b);
      },
    },
    shape_base_a,
    shape_base_b
  );
}

} // namespace ddrl::collision
