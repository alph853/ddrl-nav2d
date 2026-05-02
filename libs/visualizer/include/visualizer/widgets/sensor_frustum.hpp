/**
 * @file sensor_frustum.hpp
 * @brief Sensor frustum visualization widget for cameras and lidars
 */

#pragma once

#include "core/math/pose.hpp"
#include "core/sensors/sensor_models.hpp"
#include <array>
#include <memory>
#include <vector>
#include <vtkSmartPointer.h>

// Forward declarations
class vtkActor;
class vtkPolyData;
class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Visualizes sensor field of view as a 3D frustum
 *
 * Supports multiple sensor types:
 * - DepthCamera: Pyramid frustum based on camera intrinsics
 * - Lidar2D: Fan/wedge frustum in horizontal plane
 * - Lidar3D: Conical frustum with vertical FOV
 */
class SensorFrustumActor
{
public:
  /**
   * @brief Create a sensor frustum visualization
   * @param sensor_config Sensor configuration (variant type)
   * @param color RGB color [0,1] for the frustum wireframe
   * @param opacity Opacity [0,1] for the frustum faces
   */
  explicit SensorFrustumActor(
    const core::sensors::SensorConfig& sensor_config,
    const std::array<double, 3>&       color   = {0.0, 1.0, 1.0}, // Cyan
    double                             opacity = 0.08
  );

  ~SensorFrustumActor() = default;

  // Non-copyable but movable
  SensorFrustumActor(const SensorFrustumActor&)            = delete;
  SensorFrustumActor& operator=(const SensorFrustumActor&) = delete;
  SensorFrustumActor(SensorFrustumActor&&)                 = default;
  SensorFrustumActor& operator=(SensorFrustumActor&&)      = default;

  /**
   * @brief Update the frustum pose (relative to robot base)
   * @param robot_pose World pose of the robot base
   */
  void update_pose(const core::math::Pose2& robot_pose);

  /**
   * @brief Add frustum actor to a VTK renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove frustum actor from a VTK renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Set visibility of the frustum
   */
  void set_visible(bool visible);

  /**
   * @brief Get the underlying VTK actor
   */
  [[nodiscard]] vtkActor* get_actor() const { return actors_.empty() ? nullptr : actors_.front().Get(); }

private:
  enum class MountMarkerType : uint8_t {
    Camera,
    Lidar,
  };

  void create_mount_marker();
  void add_actor(vtkSmartPointer<vtkActor> actor);
  static vtkSmartPointer<vtkActor> make_polydata_actor(
    vtkSmartPointer<vtkPolyData> polydata, const std::array<double, 3>& color, double opacity,
    bool edge_visibility, double line_width
  );
  void create_depth_camera_frustum(const core::sensors::DepthCamera& config);
  void create_lidar2d_frustum(const core::sensors::Lidar2D& config);
  void create_lidar3d_frustum(const core::sensors::Lidar3D& config);

  std::vector<vtkSmartPointer<vtkActor>> actors_;
  core::math::Pose3                      sensor_pose_rel_base_; // Sensor pose relative to robot base
  std::array<double, 3>                  color_;
  double                                 opacity_;
  MountMarkerType                        mount_marker_type_{MountMarkerType::Lidar};
};

} // namespace ddrl::visualizer::widgets
