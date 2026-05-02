// (Minor comment additions; functions unchanged)
#pragma once
#include "core/math/primitives.hpp"
#include <cmath>
#include <ostream>

namespace ddrl::core::math {

struct Quaternion {
  double w{1}, x{0}, y{0}, z{0};

  // Normalize this quaternion
  [[nodiscard]] Quaternion normalize() const;

  // Multiply this quaternion by another
  [[nodiscard]] Quaternion multiply(const Quaternion& other) const;

  // Get the inverse (conjugate for unit quaternions)
  [[nodiscard]] Quaternion inverse() const;

  // Rotate a vector by this quaternion (assuming normalized)
  [[nodiscard]] Vec3 rotate(const Vec3& v) const;
};

} // namespace ddrl::core::math

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::math::Quaternion& q)
{
  return os << "Quaternion{" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << "}";
}
