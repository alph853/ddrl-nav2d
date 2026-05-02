#include "core/rl/observation_encoding.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ddrl::core::rl {

namespace {

struct CellSample {
  float             normalized_range{1.0F};
  core::math::Point3 point_world{};
};

} // namespace

ScanGridEncodingResult encode_scan_grid(const Observation& obs, const ScanGridEncodingConfig& cfg)
{
  const std::size_t azimuth_bins = std::max<std::size_t>(1, cfg.azimuth_bins);
  const std::size_t z_bands      = std::max<std::size_t>(1, cfg.z_bands.size());
  const std::size_t grid_dim     = azimuth_bins * z_bands;
  const float       range_max_m  = static_cast<float>(std::max(cfg.range_max_m, 1e-3));

  std::vector<std::vector<CellSample>> cell_samples(grid_dim);

  const double cos_yaw = std::cos(-obs.est_pose.yaw);
  const double sin_yaw = std::sin(-obs.est_pose.yaw);
  const double rx      = obs.est_pose.pos.x;
  const double ry      = obs.est_pose.pos.y;

  for (const auto& sensor_obs : obs.sensor_data) {
    const auto cloud = sensor_obs.point_cloud;
    if (cloud == nullptr) {
      continue;
    }
    for (const auto& pt : cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const double dx = pt.x - rx;
      const double dy = pt.y - ry;
      const double ex = cos_yaw * dx - sin_yaw * dy;
      const double ey = sin_yaw * dx + cos_yaw * dy;
      const double planar_range = std::hypot(ex, ey);
      if (!std::isfinite(planar_range) || planar_range <= 1e-6) {
        continue;
      }

      std::size_t z_idx = z_bands;
      for (std::size_t idx = 0; idx < cfg.z_bands.size(); ++idx) {
        const auto& [z_min, z_max] = cfg.z_bands[idx];
        if (pt.z >= z_min && pt.z < z_max) {
          z_idx = idx;
          break;
        }
      }
      if (z_idx >= z_bands) {
        continue;
      }

      const double azimuth = std::atan2(ey, ex);
      if (!std::isfinite(azimuth)) {
        continue;
      }
      const double wrapped = (azimuth + std::numbers::pi) / (2.0 * std::numbers::pi);
      const auto azimuth_idx = static_cast<std::size_t>(std::clamp(
        static_cast<long>(std::floor(wrapped * static_cast<double>(azimuth_bins))),
        0L,
        static_cast<long>(azimuth_bins - 1)
      ));

      const std::size_t cell_idx = z_idx * azimuth_bins + azimuth_idx;
      cell_samples[cell_idx].push_back(CellSample{
        .normalized_range =
          std::min(static_cast<float>(planar_range), range_max_m) / range_max_m,
        .point_world = pt,
      });
    }
  }

  ScanGridEncodingResult result;
  result.normalized_ranges.reserve(grid_dim);
  result.representative_points_world.reserve(grid_dim);

  for (auto& cell : cell_samples) {
    if (cell.empty()) {
      result.normalized_ranges.push_back(1.0F);
      continue;
    }

    const std::size_t idx = static_cast<std::size_t>(
      std::floor(0.05 * static_cast<double>(cell.size() - 1))
    );
    std::nth_element(
      cell.begin(),
      cell.begin() + static_cast<std::ptrdiff_t>(idx),
      cell.end(),
      [](const CellSample& lhs, const CellSample& rhs) {
        return lhs.normalized_range < rhs.normalized_range;
      }
    );

    result.normalized_ranges.push_back(cell[idx].normalized_range);
    result.representative_points_world.push_back(cell[idx].point_world);
  }

  return result;
}

} // namespace ddrl::core::rl
