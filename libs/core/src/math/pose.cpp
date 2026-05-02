#include "core/math/pose.hpp"

#include <cmath>

namespace ddrl::core::math {

// ==================== Pose2 Implementations ====================

Vec2 Pose2::forward() const noexcept
{
  return {std::cos(yaw), std::sin(yaw)};
}

Vec2 Pose2::right() const noexcept
{
  return {-std::sin(yaw), std::cos(yaw)};
}

Point2 Pose2::transform_point(const Point2& p) const noexcept
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double xr = c * p.x - s * p.y;
  const double yr = s * p.x + c * p.y;
  return {pos.x + xr, pos.y + yr};
}

Point2 Pose2::transform_point(const Point2& p, double cos_yaw, double sin_yaw) const noexcept
{
  const double xr = cos_yaw * p.x - sin_yaw * p.y;
  const double yr = sin_yaw * p.x + cos_yaw * p.y;
  return {pos.x + xr, pos.y + yr};
}

Vec2 Pose2::transform_vector(const Vec2& v) const noexcept
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  return {c * v.x - s * v.y, s * v.x + c * v.y};
}

Point2 Pose2::inverse_transform_point(const Point2& p) const noexcept
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double dx = p.x - pos.x, dy = p.y - pos.y;
  return {c * dx + s * dy, -s * dx + c * dy};
}

Vec2 Pose2::inverse_transform_vector(const Vec2& v) const noexcept
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  return {c * v.x + s * v.y, -s * v.x + c * v.y};
}

Pose3 Pose2::to_pose3() const noexcept
{
  // Convert 2D yaw to quaternion around Z-axis
  const double half = yaw * 0.5;
  return {Point3{pos.x, pos.y, 0.0}, Quaternion{std::cos(half), 0, 0, std::sin(half)}};
}

// ==================== Pose3 Implementations ====================

Vec3 Pose3::rotate(const Vec3& v_local) const
{
  return q.rotate(v_local);
}

Point3 Pose3::apply(const Point3& p_local) const
{
  const Vec3 rp = rotate(to_vec3(p_local));
  return {rp.x + t.x, rp.y + t.y, rp.z + t.z};
}

Vec3 Pose3::apply_to_vector(const Vec3& v_local) const
{
  return rotate(v_local);
}

Pose3 Pose3::compose(const Pose3& rhs) const
{
  Pose3 out;
  out.q             = q.multiply(rhs.q); // R = R_this * R_rhs
  out.q             = out.q.normalize();
  const Vec3 rt_rhs = rotate(to_vec3(rhs.t)); // R_this * t_rhs
  out.t             = {t.x + rt_rhs.x, t.y + rt_rhs.y, t.z + rt_rhs.z};
  return out;
}

Pose3 Pose3::inverse() const
{
  const Quaternion qi = q.inverse(); // R^T
  const Vec3       minus_t{-t.x, -t.y, -t.z};
  const Vec3       ti_v = qi.rotate(minus_t); // -R^T t
  return {to_point3(ti_v), qi};
}

Pose2 Pose3::to_pose2() const
{
  // Extract yaw from quaternion
  const double yaw = get_yaw();
  return {Point2{t.x, t.y}, yaw};
}

double Pose3::get_yaw() const
{
  // Extract yaw (rotation around Z) from quaternion
  // Formula: atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

Pose3 Pose3::lerp(const Pose3& a, const Pose3& b, double u)
{
  auto lerp1 = [&](double A, double B) { return A + (B - A) * u; };

  Pose3 o;
  o.t = {lerp1(a.t.x, b.t.x), lerp1(a.t.y, b.t.y), lerp1(a.t.z, b.t.z)};

  double     dot = a.q.w * b.q.w + a.q.x * b.q.x + a.q.y * b.q.y + a.q.z * b.q.z;
  Quaternion bb  = b.q;
  if (dot < 0.0) {
    dot = -dot;
    bb  = {-bb.w, -bb.x, -bb.y, -bb.z};
  }

  if (1.0 - dot < kEps) {
    // nearly identical: lerp then normalize
    o.q = Quaternion{lerp1(a.q.w, bb.w), lerp1(a.q.x, bb.x), lerp1(a.q.y, bb.y), lerp1(a.q.z, bb.z)}
              .normalize();
  } else {
    const double th = std::acos(dot), s = std::sin(th);
    const double w1 = std::sin((1.0 - u) * th) / s;
    const double w2 = std::sin(u * th) / s;
    o.q             = Quaternion{a.q.w * w1 + bb.w * w2,
                     a.q.x * w1 + bb.x * w2,
                     a.q.y * w1 + bb.y * w2,
                     a.q.z * w1 + bb.z * w2}
              .normalize();
  }
  return o;
}

Pose3 Pose3::identity()
{
  return {{0, 0, 0}, {1, 0, 0, 0}};
}

} // namespace ddrl::core::math
