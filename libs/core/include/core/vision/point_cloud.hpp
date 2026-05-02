#pragma once

#include "core/math/primitives.hpp"
#include <vector>

namespace ddrl::core::vision {

struct PointCloud {
  std::vector<core::math::Point3> points;

  void                      clear() { points.clear(); }
  void                      reserve(std::size_t n) { points.reserve(n); }
  void                      push_back(const core::math::Point3& p) { points.push_back(p); }
  [[nodiscard]] std::size_t size() const noexcept { return points.size(); }
  [[nodiscard]] bool        empty() const noexcept { return points.empty(); }
};

} // namespace ddrl::core::vision
