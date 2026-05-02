#include "control/pure_pursuit.hpp"

#include <cmath>

namespace ddrl::control {

using core::math::Point2;
using core::math::Pose2;

PurePursuitController::ClosestProj PurePursuitController::closest_point_on_path(
  const std::vector<Point2>& path, double x, double y)
{
  ClosestProj best;
  if (path.size() < 2) {
    return best;
  }

  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const auto&  p0 = path[i];
    const auto&  p1 = path[i + 1];
    const double vx = p1.x - p0.x;
    const double vy = p1.y - p0.y;
    const double wx = x - p0.x;
    const double wy = y - p0.y;
    const double vv = (vx * vx) + (vy * vy);
    double       t  = 0.0;
    if (vv > 1e-12) {
      t = (vx * wx + vy * wy) / vv;
      t = std::clamp(t, 0.0, 1.0);
    }
    const double projx = p0.x + (t * vx);
    const double projy = p0.y + (t * vy);
    const double dx    = x - projx;
    const double dy    = y - projy;
    const double d2    = (dx * dx) + (dy * dy);
    if (d2 < best.dist2) {
      best.seg_idx = i;
      best.t       = t;
      best.x       = projx;
      best.y       = projy;
      best.dist2   = d2;
    }
  }
  return best;
}

bool PurePursuitController::advance_along_path(const std::vector<Point2>& path, std::size_t seg_idx,
  double t_on_seg, double distance, Point2& out_pt, std::size_t& out_used_seg_idx)
{
  if (path.size() < 2) {
    return false;
  }

  double      remaining = distance;
  std::size_t i         = seg_idx;
  double      t         = t_on_seg;

  // Start from interpolation point on current segment
  double curx = path[i].x + ((path[i + 1].x - path[i].x) * t);
  double cury = path[i].y + ((path[i + 1].y - path[i].y) * t);

  while (true) {
    const double nx      = path[i + 1].x;
    const double ny      = path[i + 1].y;
    const double seg_dx  = nx - curx;
    const double seg_dy  = ny - cury;
    const double seg_len = std::hypot(seg_dx, seg_dy);

    if (seg_len >= remaining && seg_len > 1e-9) {
      const double r   = remaining / seg_len;
      out_pt.x         = curx + seg_dx * r;
      out_pt.y         = cury + seg_dy * r;
      out_used_seg_idx = i;
      return true;
    }

    // Move to next segment
    remaining -= seg_len;
    ++i;
    if (i + 1 >= path.size()) {
      // Clamp to last point if distance exceeds path end
      out_pt           = path.back();
      out_used_seg_idx = path.size() - 2;
      return true;
    }
    curx = path[i].x;
    cury = path[i].y;
  }
}

PurePursuitOutput PurePursuitController::track(const Pose2& pose, const std::vector<Point2>& path,
  double current_speed, double nominal_speed) const
{
  PurePursuitOutput out{};
  out.target_speed = nominal_speed;

  if (path.size() < 2) {
    out.valid     = false;
    out.curvature = 0.0;
    return out;
  }

  // 1) Closest projection on path
  const auto proj = closest_point_on_path(path, pose.pos.x, pose.pos.y);

  // 2) Lookahead distance (clamped per config)
  const double ld = lookahead(current_speed);

  // 3) Find lookahead point along path
  Point2      lookahead_pt{};
  std::size_t used_seg_idx = proj.seg_idx;
  const bool  ok = advance_along_path(path, proj.seg_idx, proj.t, ld, lookahead_pt, used_seg_idx);
  if (!ok) {
    out.valid     = false;
    out.curvature = 0.0;
    return out;
  }

  // 4) Transform to vehicle frame
  double xb = 0.0, yb = 0.0;
  world_to_body(lookahead_pt.x, lookahead_pt.y, pose, xb, yb);

  // Ensure lookahead distance in body frame (numerical guard)
  const double ld_body = std::max(1e-6, std::hypot(xb, yb));

  // 5) Pure pursuit curvature
  const double kappa = curvature_from_lookahead(yb, ld_body);

  // 6) Optional speed coupling to curvature
  double v = nominal_speed;
  if (config_.speed_coupled_to_curvature) {
    const double kv = std::max(0.0, config_.kv);
    v               = nominal_speed / (1.0 + kv * std::abs(kappa));
  }

  out.valid        = true;
  out.curvature    = kappa;
  out.target_speed = v;
  out.path_index   = used_seg_idx;
  // steering left as NaN; user can convert with wheelbase via helper
  return out;
}

} // namespace ddrl::control
