#include "core/math/quaternion.hpp"
#include <cmath>

namespace ddrl::core::math {

Quaternion Quaternion::normalize() const
{
  double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n == 0) {
    return {1, 0, 0, 0};
  }
  return {w / n, x / n, y / n, z / n};
}

Quaternion Quaternion::multiply(const Quaternion& other) const
{
  return {w * other.w - x * other.x - y * other.y - z * other.z,
          w * other.x + x * other.w + y * other.z - z * other.y,
          w * other.y - x * other.z + y * other.w + z * other.x,
          w * other.z + x * other.y - y * other.x + z * other.w};
}

Quaternion Quaternion::inverse() const
{
  return {w, -x, -y, -z};
}

Vec3 Quaternion::rotate(const Vec3& v) const
{
  const double xx = x * x, yy = y * y, zz = z * z;
  const double xy = x * y, xz = x * z, yz = y * z;
  const double wx = w * x, wy = w * y, wz = w * z;
  return {(1 - 2 * (yy + zz)) * v.x + 2 * (xy - wz) * v.y + 2 * (xz + wy) * v.z,
          2 * (xy + wz) * v.x + (1 - 2 * (xx + zz)) * v.y + 2 * (yz - wx) * v.z,
          2 * (xz - wy) * v.x + 2 * (yz + wx) * v.y + (1 - 2 * (xx + yy)) * v.z};
}

} // namespace ddrl::core::math
