/**
 * @file grid_widget.hpp
 * @brief Widget for creating ground grid visualization
 *
 * This widget creates a grid on the ground plane for spatial reference.
 *
 * @note This widget depends on VTK (Visualization Toolkit).
 *       Requires: vtkActor, vtkSmartPointer, vtkPolyData, vtkPolyDataMapper
 */

#pragma once

#include <memory>

class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Widget for rendering ground grid
 * Creates a grid visualization on the ground plane (XY plane) for spatial
 * @note Depends on VTK library
 */
class GridWidget
{
public:
  /**
   * @brief Construct ground grid widget
   * @param size Grid size in meters (total extent from center)
   * @param spacing Grid line spacing in meters
   */
  GridWidget(double size = 100.0, double spacing = 1.0);

  ~GridWidget();

  // Non-copyable, movable
  GridWidget(const GridWidget&)            = delete;
  GridWidget& operator=(const GridWidget&) = delete;
  GridWidget(GridWidget&&) noexcept;
  GridWidget& operator=(GridWidget&&) noexcept;

  /**
   * @brief Register widget geometry with a renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove widget geometry from a renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Update grid size
   */
  void set_size(double size);

  /**
   * @brief Update grid spacing
   */
  void set_spacing(double spacing);

  /**
   * @brief Set grid visibility
   */
  void set_visible(bool visible);

  /**
   * @brief Update grid color
   * @note r, g, b are in range [0-1]
   */
  void set_color(double r, double g, double b);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
