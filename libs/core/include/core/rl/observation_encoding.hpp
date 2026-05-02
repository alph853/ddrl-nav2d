#pragma once

#include "core/math/primitives.hpp"
#include "core/rl/rl.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace ddrl::core::rl {

struct ScanGridEncodingConfig {
  uint32_t azimuth_bins{32};
  std::vector<std::pair<double, double>> z_bands;
  double range_max_m{35.0};
};

struct ScanGridEncodingResult {
  std::vector<float>             normalized_ranges;
  std::vector<core::math::Point3> representative_points_world;
};

[[nodiscard]] ScanGridEncodingResult
encode_scan_grid(const Observation& obs, const ScanGridEncodingConfig& cfg);

} // namespace ddrl::core::rl
