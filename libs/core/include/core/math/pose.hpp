#pragma once

#include "core/math/quaternion.hpp"
#include <ostream>
#include <format>

namespace ddrl::core::math {

struct Pose3;

struct Pose2 {
  Point2 pos;
  double yaw{0.0}; // rad

  [[nodiscard]] Vec2 forward() const noexcept;
  [[nodiscard]] Vec2 right() const noexcept;

  [[nodiscard]] Pose2 operator*(const Pose2& o) const noexcept
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double xr = c * o.pos.x - s * o.pos.y;
    const double yr = s * o.pos.x + c * o.pos.y;
    return {Point2{pos.x + xr, pos.y + yr}, yaw + o.yaw};
  }

  [[nodiscard]] Point2 transform_point(const Point2& p) const noexcept;
  [[nodiscard]] Point2
  transform_point(const Point2& p, double cos_yaw, double sin_yaw) const noexcept;
  [[nodiscard]] Vec2   transform_vector(const Vec2& v) const noexcept;
  [[nodiscard]] Point2 inverse_transform_point(const Point2& p) const noexcept;
  [[nodiscard]] Vec2   inverse_transform_vector(const Vec2& v) const noexcept;
  [[nodiscard]] Pose3  to_pose3() const noexcept;
};

struct Pose3 {
  Point3     t; // translation (world_from_local origin)
  Quaternion q; // rotation   (world_from_local)

  // --- Helpers to convert between Point3 and Vec3 coordinates ---
  static Vec3   to_vec3(const Point3& p) { return {p.x, p.y, p.z}; }
  static Point3 to_point3(const Vec3& v) { return {v.x, v.y, v.z}; }

  // --- Vector rotation (no translation) ---
  [[nodiscard]] Vec3 rotate(const Vec3& v_local) const;

  // --- Apply rigid transform to a point: p_world = R * p_local + t ---
  [[nodiscard]] Point3 apply(const Point3& p_local) const;

  // --- Apply only rotation to a vector (e.g., angular vel, ray dir) ---
  [[nodiscard]] Vec3 apply_to_vector(const Vec3& v_local) const;

  // --- Compose: this * rhs  (apply rhs, then this) ---
  [[nodiscard]] Pose3 compose(const Pose3& rhs) const;

  // --- Inverse: (R,t)^-1 = (R^T, -R^T t) ---
  [[nodiscard]] Pose3 inverse() const;

  [[nodiscard]] Pose2  to_pose2() const;
  [[nodiscard]] double get_yaw() const;

  // --- Interpolate (linear t + slerp q) ---
  static Pose3 lerp(const Pose3& a, const Pose3& b, double u);

  // --- Convenience: identity pose ---
  static Pose3 identity();
};

} // namespace ddrl::core::math

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::math::Pose2& pose)
{
  return os << "{pos: " << pose.pos << ", yaw: " << pose.yaw << "}";
}

inline std::string to_string(const ddrl::core::math::Pose2& pose)
{
  return std::format("(pos=({}, {}), yaw={})", pose.pos.x, pose.pos.y, pose.yaw);
}

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::math::Pose3& pose)
{
  return os << "{t: " << pose.t << ", q: " << pose.q << "}";
}
