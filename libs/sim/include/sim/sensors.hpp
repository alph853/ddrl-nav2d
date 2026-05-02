#pragma once

#include <collision/ray_casting.hpp>
#include <core/base/result.hpp>
#include <core/math/pose.hpp>
#include <core/math/primitives.hpp>
#include <core/sensors/sensor_models.hpp>
#include <core/transform/tf.hpp>
#include <core/vision/point_cloud.hpp>
#include <memory>
#include <random>
#include <threading/thread_pool.hpp>

namespace ddrl::sim {

/// Base class for all sensors with cooperative scheduling
class SensorBase
{
public:
  virtual ~SensorBase();

  /// Constructor parameters for sensors
  struct Params {
    std::shared_ptr<const core::sensors::SensorConfig> config;
    std::shared_ptr<collision::TLAS2Pair>              scene_trees;
    std::shared_ptr<threading::ThreadPool>             thread_pool;
    std::shared_ptr<core::tf::TfBuffer>                tf_buffer;
    std::string                                        robot_base_frame;
  };

  /// Check if sensor should update based on timing model
  [[nodiscard]] bool should_update(double dt, double current_time);

  /// Update sensor - performs raycasting if needed
  virtual core::Result<void> update(double dt, double current_time) = 0;

  /// Get cached sensor data (zero-copy via shared_ptr)
  [[nodiscard]] virtual std::shared_ptr<const core::vision::PointCloud> get_point_cloud() const = 0;

  /// Get sensor ID
  [[nodiscard]] std::string id() const;

  /// Get sensor frame ID
  [[nodiscard]] std::string frame_id() const;

  /// Check if sensor has valid data
  [[nodiscard]] bool has_data() const;

  SensorBase(const SensorBase&) = delete;
  SensorBase& operator=(const SensorBase&) = delete;
  SensorBase(SensorBase&&) noexcept;
  SensorBase& operator=(SensorBase&&) noexcept;

  /// Apply noise model to range measurement
  [[nodiscard]] double apply_noise(double range, std::mt19937& rng) const;

  /// Check if ray should be dropped out
  [[nodiscard]] bool should_dropout(std::mt19937& rng) const;

  /// Get base config from variant
  [[nodiscard]] const core::sensors::SensorBase& base_config() const;

protected:
  SensorBase(const Params& params);

  // Protected accessors for derived classes
  [[nodiscard]] std::shared_ptr<collision::TLAS2Pair> get_scene_trees() const;
  [[nodiscard]] std::shared_ptr<threading::ThreadPool> get_thread_pool() const;
  [[nodiscard]] std::shared_ptr<core::tf::TfBuffer> get_tf_buffer() const;
  [[nodiscard]] std::shared_ptr<const core::vision::PointCloud> get_cached_cloud() const;
  [[nodiscard]] std::shared_ptr<core::vision::PointCloud> get_mutable_pending_cloud();
  void set_pending_data(double time);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Depth camera sensor
class DepthCameraSensor : public SensorBase
{
public:
  explicit DepthCameraSensor(const Params& params);
  ~DepthCameraSensor() override;

  DepthCameraSensor(const DepthCameraSensor&) = delete;
  DepthCameraSensor& operator=(const DepthCameraSensor&) = delete;
  DepthCameraSensor(DepthCameraSensor&&) noexcept;
  DepthCameraSensor& operator=(DepthCameraSensor&&) noexcept;

  core::Result<void>                                              update(double dt, double current_time) override;
  [[nodiscard]] std::shared_ptr<const core::vision::PointCloud> get_point_cloud() const override;

private:
  class DerivedImpl;
  std::unique_ptr<DerivedImpl> derived_impl_;
};

/// 2D planar lidar sensor
class Lidar2DSensor : public SensorBase
{
public:
  explicit Lidar2DSensor(const Params& params);
  ~Lidar2DSensor() override;

  Lidar2DSensor(const Lidar2DSensor&) = delete;
  Lidar2DSensor& operator=(const Lidar2DSensor&) = delete;
  Lidar2DSensor(Lidar2DSensor&&) noexcept;
  Lidar2DSensor& operator=(Lidar2DSensor&&) noexcept;

  core::Result<void>                                              update(double dt, double current_time) override;
  [[nodiscard]] std::shared_ptr<const core::vision::PointCloud> get_point_cloud() const override;

private:
  class DerivedImpl;
  std::unique_ptr<DerivedImpl> derived_impl_;
};

/// 3D multi-ring lidar sensor
class Lidar3DSensor : public SensorBase
{
public:
  explicit Lidar3DSensor(const Params& params);
  ~Lidar3DSensor() override;

  Lidar3DSensor(const Lidar3DSensor&) = delete;
  Lidar3DSensor& operator=(const Lidar3DSensor&) = delete;
  Lidar3DSensor(Lidar3DSensor&&) noexcept;
  Lidar3DSensor& operator=(Lidar3DSensor&&) noexcept;

  core::Result<void>                                              update(double dt, double current_time) override;
  [[nodiscard]] std::shared_ptr<const core::vision::PointCloud> get_point_cloud() const override;

private:
  class DerivedImpl;
  std::unique_ptr<DerivedImpl> derived_impl_;
};

/// Type-erased sensor pointer
using SensorType = std::unique_ptr<SensorBase>;

/// Factory function to create sensors from config
[[nodiscard]] core::Result<SensorType> create_sensor(const core::sensors::SensorConfig& sensor_cfg,
                                                     const SensorBase::Params&          params);

} // namespace ddrl::sim
