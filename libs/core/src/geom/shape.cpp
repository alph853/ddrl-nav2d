#include "core/geom/shape.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

namespace ddrl::core::geom {

// Circle implementation
Circle::Circle(double r) : radius(r)
{
  assert(r > 0 && "Circle radius must be positive");
}

const AABB2& Circle::bounds() const noexcept
{
  bounds_cache_ = AABB2{
    .min = {-radius, -radius},
    .max = {radius, radius}
  };
  return *bounds_cache_;
}

// Box2 implementation
Box2::Box2(double hx_, double hy_) : hx(hx_), hy(hy_)
{
  assert(hx > 0 && hy > 0 && "OBB half extents must be positive");
}

const AABB2& Box2::bounds() const noexcept
{
  bounds_cache_ = AABB2{
    .min = {-hx, -hy},
    .max = {hx, hy}
  };
  return *bounds_cache_;
}

// Polygon2 implementation
Polygon2::Polygon2(std::vector<math::Point2> verts)
{
  assert(verts.size() > 2 && "Polygon must have at least 3 vertices");
  v           = std::move(verts);
  double minx = v[0].x, maxx = v[0].x, miny = v[0].y, maxy = v[0].y;
  for (const auto& p : v) {
    minx = std::min(minx, p.x);
    maxx = std::max(maxx, p.x);
    miny = std::min(miny, p.y);
    maxy = std::max(maxy, p.y);
  }
}

const AABB2& Polygon2::bounds() const
{
  if (!bounds_cache_) {
    double minx = v[0].x, maxx = v[0].x, miny = v[0].y, maxy = v[0].y;
    for (const auto& p : v) {
      minx = std::min(minx, p.x);
      maxx = std::max(maxx, p.x);
      miny = std::min(miny, p.y);
      maxy = std::max(maxy, p.y);
    }
    bounds_cache_ = AABB2{
      .min = {minx, miny},
      .max = {maxx, maxy}
    };
  }
  return *bounds_cache_;
}

bool Polygon2::contains_point(const math::Point2& point) const
{
  if (v.size() < 3) {
    return false;
  }

  bool inside = false;
  const size_t n = v.size();

  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const auto& vi = v[i];
    const auto& vj = v[j];

    // Ray casting: check if horizontal ray from point intersects edge
    if (((vi.y > point.y) != (vj.y > point.y)) &&
        (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)) {
      inside = !inside;
    }
  }

  return inside;
}

bool Polygon2::segments_intersect(
  const math::Point2& p1, const math::Point2& p2,
  const math::Point2& p3, const math::Point2& p4)
{
  const double dx1 = p2.x - p1.x;
  const double dy1 = p2.y - p1.y;
  const double dx2 = p4.x - p3.x;
  const double dy2 = p4.y - p3.y;
  const double dx3 = p1.x - p3.x;
  const double dy3 = p1.y - p3.y;

  const double denom = dy2 * dx1 - dx2 * dy1;

  // Parallel or collinear
  if (std::abs(denom) < 1e-10) {
    return false;
  }

  const double ua = (dx2 * dy3 - dy2 * dx3) / denom;
  const double ub = (dx1 * dy3 - dy1 * dx3) / denom;

  // Check if intersection point is within both segments
  return (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0);
}

bool Polygon2::check_intersects(const Polygon2& other) const
{
  // Early exit: check AABB intersection first
  if (!bounds().intersects(other.bounds())) {
    return false;
  }

  // Check if any vertex of this polygon is inside the other
  for (const auto& point : v) {
    if (other.contains_point(point)) {
      return true;
    }
  }

  // Check if any vertex of the other polygon is inside this one
  for (const auto& point : other.v) {
    if (contains_point(point)) {
      return true;
    }
  }

  // Check if any edges intersect
  const size_t n1 = v.size();
  const size_t n2 = other.v.size();

  for (size_t i = 0; i < n1; ++i) {
    const auto& p1 = v[i];
    const auto& p2 = v[(i + 1) % n1];

    for (size_t j = 0; j < n2; ++j) {
      const auto& p3 = other.v[j];
      const auto& p4 = other.v[(j + 1) % n2];

      if (segments_intersect(p1, p2, p3, p4)) {
        return true;
      }
    }
  }

  return false;
}

inline Point2 rotate_point(const Point2& p, double yaw) noexcept {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Point2{
    c * p.x - s * p.y,
    s * p.x + c * p.y
  };
}

// transform a point from local shape frame -> world frame using pose
inline Point2 transform_point(const Point2& p_local, const Pose2& pose) noexcept {
  const Point2 r = rotate_point(p_local, pose.yaw);
  return Point2{
    r.x + pose.pos.x,
    r.y + pose.pos.y
  };
}

// Return true if point p is inside or on boundary of convex polygon poly (CCW).
// We'll use a half-space test for every edge.
inline bool point_in_convex_polygon(const std::vector<Point2>& poly, const Point2& p) noexcept {
  const size_t n = poly.size();
  if (n < 3) {
    return false; // degenerate polygon
  }

  // For CCW polygon, for each directed edge (a -> b),
  // the interior is to the LEFT (cross >= 0).
  for (size_t i = 0; i < n; ++i) {
    const Point2& a = poly[i];
    const Point2& b = poly[(i + 1) % n];

    // edge vector
    const double ex = b.x - a.x;
    const double ey = b.y - a.y;
    // vector from a to point p
    const double px = p.x - a.x;
    const double py = p.y - a.y;

    // 2D cross product z-value: ex*py - ey*px
    const double cross = ex * py - ey * px;

    if (cross < 0.0) {
      // p is to the RIGHT of this edge -> outside
      return false;
    }
  }
  return true;
}

// For a circle fully inside a convex polygon, we need: the circle center is
// at least `radius` away from every edge outward.
// That's stricter than just "center in polygon".
//
// We'll compute signed distance from center to each edge line.
// For CCW polygon, inside means distance >= radius (>= 0 for the point).
inline bool circle_fully_inside_convex_polygon(
    const std::vector<Point2>& poly,
    const Point2& center,
    double radius) noexcept
{
  const size_t n = poly.size();
  if (n < 3) { return false;
}

  for (size_t i = 0; i < n; ++i) {
    const Point2& a = poly[i];
    const Point2& b = poly[(i + 1) % n];

    // outward? inward? For CCW polygon, inward normal points LEFT of edge.
    const double ex = b.x - a.x;
    const double ey = b.y - a.y;

    // inward normal (nx, ny) = normalized left normal of edge (ex,ey)
    // left normal of (ex,ey) is (-ey, ex)
    const double nx = -ey;
    const double ny =  ex;
    const double len = std::sqrt(nx*nx + ny*ny);
    if (len == 0.0) { return false; // degenerate edge
}
    const double inv_len = 1.0 / len;

    // signed distance from center to line through a,b
    // dist = dot( (center - a), n_hat ), where n_hat is inward normal
    const double dx = center.x - a.x;
    const double dy = center.y - a.y;
    const double signed_dist = (dx * nx + dy * ny) * inv_len;

    // If signed_dist < radius, then some part of the circle pokes outside
    if (signed_dist < radius) {
      return false;
    }
  }
  return true;
}

// Compute world-space corners of an oriented Box2 at pose
inline void box_world_corners(const Box2& box,
                              const Pose2& pose,
                              std::vector<Point2>& out) {
  out.clear();
  out.reserve(4);

  // local corners (hx,hy)
  // CCW order doesn't strictly matter for containment check (we test each corner),
  // but let's keep it CCW-ish in local frame.
  const std::array<Point2, 4> local_corners = {
    Point2{ -box.hx, -box.hy },
    Point2{  box.hx, -box.hy },
    Point2{  box.hx,  box.hy },
    Point2{ -box.hx,  box.hy },
  };

  for (const auto& c : local_corners) {
    out.push_back(transform_point(c, pose));
  }
}

// Compute world-space vertices of a Polygon2 at pose
inline void poly_world_vertices(const Polygon2& poly,
                                const Pose2&   pose,
                                std::vector<Point2>& out) {
  out.clear();
  out.reserve(poly.v.size());
  for (const auto& pv : poly.v) {
    out.push_back(transform_point(pv, pose));
  }
}

bool Polygon2::contains_shape(const Shape2& shape_T, const Pose2& shape_pose) const
{
  // Fast reject using bounding boxes in world coordinates.
  // We'll build the shape's world AABB first.
auto shape = shape_T.value;

  AABB2 shape_world_bb{
    .min = { std::numeric_limits<double>::infinity(),  std::numeric_limits<double>::infinity() },
    .max = { -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity() }
  };

  auto accumulate_world_points = [&](const auto& pts_world) {
    for (const auto& p : pts_world) {
      shape_world_bb.expand_to_include(p);
    }
  };

  Point2 circle_center_ws{};
  double circle_radius = 0.0;
  std::vector<Point2> box_corners_ws;
  std::vector<Point2> poly_verts_ws;

  // Visit the variant to compute world geometry
  std::visit(
    [&](auto const& obj) {
      using T = std::decay_t<decltype(obj)>;

      if constexpr (std::is_same_v<T, Circle>) {
        // Circle center is pose.pos (yaw doesn't affect circle)
        circle_center_ws = shape_pose.pos;
        circle_radius    = obj.radius;

        // AABB of a circle in world is just center ± radius
        Point2 p1{circle_center_ws.x - circle_radius,
                  circle_center_ws.y - circle_radius};
        Point2 p2{circle_center_ws.x + circle_radius,
                  circle_center_ws.y + circle_radius};

        shape_world_bb.expand_to_include(p1);
        shape_world_bb.expand_to_include(p2);
      }
      else if constexpr (std::is_same_v<T, Box2>) {
        box_world_corners(obj, shape_pose, box_corners_ws);
        accumulate_world_points(box_corners_ws);
      }
      else if constexpr (std::is_same_v<T, Polygon2>) {
        poly_world_vertices(obj, shape_pose, poly_verts_ws);
        accumulate_world_points(poly_verts_ws);
      }
      else {
        // static_assert false for unknown type
        static_assert(!sizeof(T*), "Unhandled shape type in contains_shape");
      }
    },
    shape
  );

  // Quick reject: if the shape's world AABB leaks outside this polygon's AABB,
  // then it cannot be fully contained.
  if (!shape_world_bb.fit_in(this->bounds())) {
    return false;
  }

  // Now do precise test per shape type.
  return std::visit(
    [&](auto const& obj) -> bool {
      using T = std::decay_t<decltype(obj)>;

      if constexpr (std::is_same_v<T, Circle>) {
        // strict check: circle fully inside polygon
        return circle_fully_inside_convex_polygon(this->v,
                                                  circle_center_ws,
                                                  circle_radius);
      }
      else if constexpr (std::is_same_v<T, Box2>) {
        // check every box corner is inside polygon
        return std::ranges::all_of(
          box_corners_ws,
          [&](const Point2& c) {
            return point_in_convex_polygon(this->v, c);
          }
        );
      }
      else if constexpr (std::is_same_v<T, Polygon2>) {
        // check every vertex of the other polygon is inside this polygon
        return std::ranges::all_of(
          poly_verts_ws,
          [&](const Point2& p) {
            return point_in_convex_polygon(this->v, p);
          }
        );
      }
      else {
        static_assert(!sizeof(T*), "Unhandled shape type in contains_shape dispatch");
      }
    },
    shape
  );
}

AABB2 get_bounds(const Shape2& shape)
{
  return std::visit([](const auto& s) { return s.bounds(); }, shape.value);
}

} // namespace ddrl::core::geom

std::string to_string(const ddrl::core::geom::Polygon2& poly)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < poly.v.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << "(" << poly.v[i].x << ", " << poly.v[i].y << ")";
  }
  oss << "]";
  return oss.str();
}

std::string to_string(const ddrl::core::geom::Box2& box)
{
  std::ostringstream oss;
  oss << "(hx=" << box.hx << ", hy=" << box.hy << ")";
  return oss.str();
}
std::string to_string(const ddrl::core::geom::Circle& circle)
{
  std::ostringstream oss;
  oss << "(r=" << circle.radius << ")";
  return oss.str();
}
