#pragma once
#include "core/control/pure_pursuit.hpp"
#include "core/math/pose.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ddrl::control {

struct PurePursuitOutput {
  // Geometric curvature to follow (1/m). Positive => left turn.
  double curvature{0.0};
  // If user converts curvature to Ackermann steering using wheelbase.
  // This field is NaN when not computed by the controller (use helper).
  double steering{std::numeric_limits<double>::quiet_NaN()};
  // Recommended speed given curvature (if enabled in config).
  double target_speed{0.0};
  // Index of the path vertex used as the segment start for the lookahead search.
  std::size_t path_index{0};
  // Whether a valid lookahead point was found.
  bool valid{false};
};

class PurePursuitController
{
public:
  // ...existing code...
  PurePursuitController(const core::control::PurePursuitCaps& cfg) noexcept : config_(cfg) {}

  // Compute control using pure pursuit.
  // - path: polyline waypoints (in world frame)
  // - pose: vehicle pose (world frame, yaw in radians)
  // - current_speed: current vehicle forward speed (m/s)
  // - nominal_speed: desired cruising speed before curvature-based slowdown
  [[nodiscard]] PurePursuitOutput track(const core::math::Pose2& pose,
    const std::vector<core::math::Point2>& path, double current_speed, double nominal_speed) const;

  // Convert curvature to Ackermann steering angle using wheelbase (m).
  static double curvature_to_steering(double curvature, double wheelbase_m)
  {
    return std::atan(wheelbase_m * curvature);
  }

  // Compute lookahead distance after clamping by config.
  [[nodiscard]] double lookahead(double /*current_speed*/) const
  {
    // Constant lookahead with clamping, per config schema.
    return std::clamp(config_.lookahead_m, config_.min_lookahead_m, config_.max_lookahead_m);
  }

private:
  static double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

  struct ClosestProj {
    std::size_t seg_idx{0};     // segment start vertex index
    double      t{0.0};         // param along segment [0,1]
    double      x{0.0}, y{0.0}; // projected point in world frame
    double      dist2{std::numeric_limits<double>::infinity()};
  };

  static ClosestProj closest_point_on_path(
    const std::vector<core::math::Point2>& path, double x, double y);

  static bool advance_along_path(const std::vector<core::math::Point2>& path, std::size_t seg_idx,
    double t_on_seg, double distance, core::math::Point2& out_pt, std::size_t& out_used_seg_idx);

  static void world_to_body(
    double xw, double yw, const core::math::Pose2& pose, double& xb, double& yb)
  {
    const double c  = std::cos(pose.yaw);
    const double s  = std::sin(pose.yaw);
    const double dx = xw - pose.pos.x;
    const double dy = yw - pose.pos.y;
    xb              = c * dx + s * dy;
    yb              = -s * dx + c * dy;
  }

  static double curvature_from_lookahead(double yb, double Ld)
  {
    if (Ld <= core::math::kEps) {
      return 0.0;
    }
    return 2.0 * yb / (Ld * Ld);
  }

  core::control::PurePursuitCaps config_;
};
} // namespace ddrl::control
