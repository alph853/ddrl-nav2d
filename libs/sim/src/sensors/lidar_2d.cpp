#include <cmath>
#include <random>
#include <span>

#include "sim/sensors.hpp"

namespace ddrl::sim {

using core::ErrorCode;
using core::make_error;
using core::Result;

//=============================================================================
// Lidar2DSensor Implementation
//=============================================================================

class Lidar2DSensor::DerivedImpl
{
public:
  explicit DerivedImpl(const SensorBase::Params& params);

  void process_hits(
    std::span<const collision::RayHit> hits, const core::math::Pose3& sensor_pose_world,
    core::vision::PointCloud& out_cloud, SensorBase& base
  );

  [[nodiscard]] const std::shared_ptr<const core::sensors::Lidar2D>& config_lidar() const
  {
    return config_lidar_;
  }
  [[nodiscard]] std::mt19937& rng() { return rng_; }
  [[nodiscard]] std::vector<collision::RayRequest>& ray_buffer() { return ray_buffer_; }
  [[nodiscard]] std::vector<collision::RayHit>&     hit_buffer() { return hit_buffer_; }

private:
  std::shared_ptr<const core::sensors::Lidar2D> config_lidar_;
  std::mt19937                                  rng_;
  std::vector<collision::RayRequest>            ray_buffer_;
  std::vector<collision::RayHit>                hit_buffer_;
};

Lidar2DSensor::DerivedImpl::DerivedImpl(const SensorBase::Params& params)
{
  if (!std::holds_alternative<core::sensors::Lidar2D>(*params.config)) {
    throw std::runtime_error("Expected Lidar2D config in Lidar2DSensor constructor");
  }

  const auto& lidar2d_variant = std::get<core::sensors::Lidar2D>(*params.config);
  config_lidar_               = std::make_shared<const core::sensors::Lidar2D>(lidar2d_variant);

  std::random_device rd;
  rng_.seed(rd());
}

void Lidar2DSensor::DerivedImpl::process_hits(
  std::span<const collision::RayHit> hits, const core::math::Pose3& sensor_pose_world,
  core::vision::PointCloud& out_cloud, SensorBase& base
)
{
  (void)sensor_pose_world;

  out_cloud.clear();
  out_cloud.reserve(hits.size());

  const auto& base_cfg = base.base_config();
  const double min_range = static_cast<double>(base_cfg.min_range);
  const double max_range = static_cast<double>(base_cfg.max_range);
  const bool   has_noise = base_cfg.noise_std > 0.0f;
  const bool   has_dropout = base_cfg.dropout_prob > 0.0f;

  if (!has_noise && !has_dropout) {
    for (const auto& hit : hits) {
      if (!hit.hit) {
        continue;
      }

      const double range = hit.distance;
      if (range >= min_range && range <= max_range) {
        out_cloud.push_back(hit.position);
      }
    }
    return;
  }

  std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
  std::normal_distribution<double>      noise_dist(0.0, static_cast<double>(base_cfg.noise_std));

  for (const auto& hit : hits) {
    if (!hit.hit) {
      continue;
    }

    if (has_dropout && dropout_dist(rng()) < base_cfg.dropout_prob) {
      continue;
    }

    double range = hit.distance;
    if (has_noise) {
      range += noise_dist(rng());
    }

    if (range >= min_range && range <= max_range) {
      out_cloud.push_back(hit.position);
    }
  }
}

Lidar2DSensor::Lidar2DSensor(const Params& params)
    : SensorBase(params), derived_impl_(std::make_unique<DerivedImpl>(params))
{
}

Lidar2DSensor::~Lidar2DSensor() = default;

Lidar2DSensor::Lidar2DSensor(Lidar2DSensor&&) noexcept            = default;
Lidar2DSensor& Lidar2DSensor::operator=(Lidar2DSensor&&) noexcept = default;

std::shared_ptr<const core::vision::PointCloud> Lidar2DSensor::get_point_cloud() const
{
  return get_cached_cloud();
}

Result<void> Lidar2DSensor::update(double dt, double current_time)
{
  if (!should_update(dt, current_time)) {
    return {};
  }

  auto scene_trees = get_scene_trees();
  auto thread_pool = get_thread_pool();
  auto tf_buffer   = get_tf_buffer();

  if (!scene_trees || !thread_pool || !tf_buffer) {
    return std::unexpected(make_error(
      ErrorCode::SIM,
      "Scene trees or thread pool or tf buffer not initialized",
      "Lidar2DSensor::update"
    ));
  }

  // Get sensor pose in world frame using TF buffer
  const auto& sensor_frame = base_config().frame_id;
  auto        tf_result    = tf_buffer->lookup("world", sensor_frame, current_time);
  if (!tf_result.has_value()) {
    return std::unexpected(tf_result.error());
  }
  const auto sensor_pose_world = tf_result.value();

  // Generate rays using collision library
  auto& rays = derived_impl_->ray_buffer();
  collision::generate_lidar_2d_rays(sensor_pose_world, *derived_impl_->config_lidar(), rays);

  // Perform raycasting
  auto& hits = derived_impl_->hit_buffer();
  hits.resize(rays.size());
  collision::trace_rays(rays, hits, *scene_trees, *thread_pool);

  // Process hits
  auto pending_cloud = get_mutable_pending_cloud();
  derived_impl_->process_hits(hits, sensor_pose_world, *pending_cloud, *this);

  // Schedule data to be available after latency
  set_pending_data(current_time + static_cast<double>(base_config().latency_s));

  return {};
}

} // namespace ddrl::sim
