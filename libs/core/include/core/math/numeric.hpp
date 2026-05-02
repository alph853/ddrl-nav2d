#pragma once
#include "core/math/primitives.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace ddrl::core::math {

template <typename T, typename U>
constexpr auto lerp(const T& a, const T& b, U t) -> decltype(a + (b - a) * t)
{
  return a + (b - a) * t;
}

template <typename T>
constexpr int sign(T v)
{
  return (T(0) < v) - (v < T(0));
}

constexpr double deg_to_rad(double d)
{
  return d * M_PI / 180.0;
}
constexpr double rad_to_deg(double r)
{
  return r * 180.0 / M_PI;
}

inline bool approx_equal(double a, double b, double eps = kEps)
{
  return std::fabs(a - b) <= eps;
}

inline double safe_divide(double num, double den, double eps = kEps)
{
  return std::fabs(den) < eps ? 0.0 : num / den;
}

inline double normalize(double value, double min_val, double max_val)
{
  if (max_val <= min_val) {
    return 0.0;
  }
  return (value - min_val) / (max_val - min_val);
}

inline double
map_range(double value, double from_min, double from_max, double to_min, double to_max)
{
  double n = normalize(value, from_min, from_max);
  return to_min + n * (to_max - to_min);
}

[[nodiscard]] inline double distance_sq(const Point3& a, const Point3& b) noexcept
{
  const Vec3 d = a - b;
  return d.length_sq();
}
[[nodiscard]] inline double distance(const Point3& a, const Point3& b) noexcept
{
  return std::sqrt(distance_sq(a, b));
}

inline double cross_product(const Vec2& a, const Vec2& b) noexcept
{
  return a.x * b.y - a.y * b.x;
}

// Point on segment (inclusive endpoints)
inline bool point_on_segment(const Segment2& seg, const Point2& p, double eps = kEps) noexcept
{
  // Collinearity via cross product
  const double cross = cross_product(seg.p2 - seg.p1, p - seg.p1);
  if (std::abs(cross) > eps) {
    return false;
  }

  // Within bounding box (with epsilon)
  const double minx = std::min(seg.p1.x, seg.p2.x) - eps, maxx = std::max(seg.p1.x, seg.p2.x) + eps;
  const double miny = std::min(seg.p1.y, seg.p2.y) - eps, maxy = std::max(seg.p1.y, seg.p2.y) + eps;
  return (p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy);
}

// Added generic 2D geometry helpers migrated from collision SAT implementation.

inline bool z_ranges_overlap(double amin, double amax, double bmin, double bmax)
{
  if (amin > amax) {
    std::swap(amin, amax);
  }
  if (bmin > bmax) {
    std::swap(bmin, bmax);
  }
  return (amin <= bmax) && (amax >= bmin);
}

inline Vec2 edge_normal(const Point2& a, const Point2& b, double eps = kEps)
{
  const Vec2   edge = b - a;
  const double lsq  = edge.length_sq();
  if (lsq <= eps) {
    return {0.0, 0.0};
  }
  Vec2         n{-edge.y, edge.x};
  const double len = n.length();
  if (len <= eps) {
    return {0.0, 0.0};
  }
  return n / len;
}

inline Point2 centroid(std::span<const Point2> poly)
{
  Vec2 sum{0.0, 0.0};
  for (const auto& p : poly) {
    sum += Vec2{p.x, p.y};
  }
  if (!poly.empty()) {
    sum /= static_cast<double>(poly.size());
  }
  return {sum.x, sum.y};
}

inline Point2
closest_point_on_segment(const Point2& a, const Point2& b, const Point2& p, double eps = kEps)
{
  const Vec2   ab        = b - a;
  const double ab_len_sq = ab.length_sq();
  if (ab_len_sq <= eps) {
    return a;
  }
  const double t = std::clamp((p - a).dot(ab) / ab_len_sq, 0.0, 1.0);
  return {a.x + ab.x * t, a.y + ab.y * t};
}

inline Point2 closest_point_on_polygon(std::span<const Point2> poly, const Point2& p)
{
  if (poly.empty()) {
    return p;
  }
  Point2 closest = poly.front();
  auto   dist_sq = [](const Point2& u, const Point2& v) {
    const double dx = u.x - v.x;
    const double dy = u.y - v.y;
    return dx * dx + dy * dy;
  };
  double best = dist_sq(closest, p);
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const auto&  a    = poly[i];
    const auto&  b    = poly[(i + 1) % poly.size()];
    const Point2 cand = closest_point_on_segment(a, b, p);
    const double d    = dist_sq(cand, p);
    if (d < best) {
      best    = d;
      closest = cand;
    }
  }
  return closest;
}

inline bool point_in_polygon(std::span<const Point2> poly, const Point2& p, double eps = kEps)
{
  if (poly.empty()) {
    return false;
  }
  bool inside = false;
  for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
    const auto& pi        = poly[i];
    const auto& pj        = poly[j];
    const bool  intersect = ((pi.y > p.y) != (pj.y > p.y)) &&
                           (p.x < (pj.x - pi.x) * (p.y - pi.y) / ((pj.y - pi.y) + eps) + pi.x);
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

inline double normalize_angle(double angle)
{
  while (angle > std::numbers::pi) {
    angle -= 2.0 * std::numbers::pi;
  }
  while (angle < -std::numbers::pi) {
    angle += 2.0 * std::numbers::pi;
  }
  return angle;
}

inline bool segment_intersects_circle(
  const Point2& start, const Point2& end, const Point2& center, double radius
)
{
  if (radius <= 0.0) {
    return false;
  }

  const double dx       = end.x - start.x;
  const double dy       = end.y - start.y;
  const double seg_len2 = dx * dx + dy * dy;
  const double eps      = 1e-6;
  if (seg_len2 <= eps) {
    return false;
  }

  const double t = ((center.x - start.x) * dx + (center.y - start.y) * dy) / seg_len2;
  if (t <= 0.0 || t >= 1.0) {
    return false;
  }

  const double closest_x = start.x + t * dx;
  const double closest_y = start.y + t * dy;
  const double dist_x    = center.x - closest_x;
  const double dist_y    = center.y - closest_y;
  const double dist2     = dist_x * dist_x + dist_y * dist_y;

  if (dist2 > radius * radius) {
    return false;
  }

  const double center_dist2 =
    (center.x - start.x) * (center.x - start.x) + (center.y - start.y) * (center.y - start.y);
  const double candidate_dist2 =
    (end.x - start.x) * (end.x - start.x) + (end.y - start.y) * (end.y - start.y);
  return center_dist2 < candidate_dist2;
}

} // namespace ddrl::core::math
