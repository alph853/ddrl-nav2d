/**
 * @file point_cloud_actor.hpp
 * @brief Widget for rendering point clouds (sensor data)
 *
 * This widget encapsulates the VTK pipeline for rendering point cloud data.
 * Optimized for frequent point updates.
 *
 * @note This widget depends on VTK (Visualization Toolkit).
 *       Requires: vtkActor, vtkSmartPointer, vtkPolyData, vtkPolyDataMapper,
 *                 vtkPoints, vtkVertexGlyphFilter
 */

#pragma once

#include <array>
#include <core/math/pose.hpp>
#include <memory>
#include <span>

class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Widget for rendering point clouds (sensor data)
 * Optimized for frequent point updates, making it suitable for real-time
 * sensor data visualization (LiDAR, depth cameras, etc.).
 *
 * @note Depends on VTK library
 */
class PointCloudActor
{
public:
  /**
   * @brief Construct point cloud actor
   */
  PointCloudActor(
    std::span<const core::math::Point3> points, const std::array<double, 3>& color,
    double point_size = 2.0
  );

  ~PointCloudActor();

  // Non-copyable, movable
  PointCloudActor(const PointCloudActor&)            = delete;
  PointCloudActor& operator=(const PointCloudActor&) = delete;
  PointCloudActor(PointCloudActor&&) noexcept;
  PointCloudActor& operator=(PointCloudActor&&) noexcept;

  /**
   * @brief Register widget geometry with a renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove widget geometry from a renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Update point cloud data
   * @param points New point cloud
   */
  void update_points(std::span<const core::math::Point3> points);

  /**
   * @brief Update color
   * @param color RGB color [0-1]
   */
  void set_color(const std::array<double, 3>& color);

  /**
   * @brief Update point size
   * @param size Point size in pixels
   */
  void set_point_size(double size);

  /**
   * @brief Set visibility
   * @param visible Visibility flag
   */
  void set_visible(bool visible);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
