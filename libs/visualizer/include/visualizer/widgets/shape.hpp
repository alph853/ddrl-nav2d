/**
 * @file shape_actor.hpp
 * @brief Widget for rendering 2.5D shapes (robot, dynamic objects)
 *
 * This widget encapsulates the full VTK pipeline for rendering extruded 2D
 * shapes.
 *
 * @note This widget depends on VTK (Visualization Toolkit).
 *       Requires: vtkActor, vtkSmartPointer, vtkPolyData, vtkPolyDataMapper,
 *                 vtkPoints, vtkCellArray
 */

#pragma once

#include <array>
#include <core/geom/shape.hpp>
#include <core/math/pose.hpp>
#include <memory>

class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Widget for rendering 2.5D shapes (robot, dynamic objects)
 * @note Depends on VTK library
 */
class ShapeActor
{
public:
  /**
   * @brief Construct shape actor from 2.5D shape
   */
  ShapeActor(const core::geom::Shape2p5& shape,
    const std::array<double, 3>& color, double opacity = 1.0);

  ~ShapeActor();

  // Non-copyable, movable
  ShapeActor(const ShapeActor&)            = delete;
  ShapeActor& operator=(const ShapeActor&) = delete;
  ShapeActor(ShapeActor&&) noexcept;
  ShapeActor& operator=(ShapeActor&&) noexcept;

  /**
   * @brief Register widget geometry with a renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove widget geometry from a renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Update world transform
   * @param pose New pose (position + orientation)
   */
  void set_transform(const core::math::Pose2& pose);

  /**
   * @brief Update color
   */
  void set_color(const std::array<double, 3>& color);

  /**
   * @brief Update opacity
   */
  void set_opacity(double opacity);

  /**
   * @brief Set visibility
   */
  void set_visible(bool visible);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
