/**
 * @file axes_widget.hpp
 * @brief Widget for creating coordinate axes visualization
 * This widget creates 3D coordinate axes for orientation reference.
 */

#pragma once

#include <memory>

class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Widget for rendering coordinate axes
 * Creates 3D coordinate axes (X, Y, Z) for orientation reference.
 * - X axis: Red
 * - Y axis: Green
 * - Z axis: Blue
 */
class AxesWidget
{
public:
  /**
   * @brief Construct coordinate axes widget
   * @param length Axis length in meters
   *
   * Example:
   * @code
  * AxesWidget axes(10.0); // 10m long axes
  * axes.add_to_renderer(*renderer);
   * @endcode
   */
  explicit AxesWidget(double length = 10.0);

  ~AxesWidget();

  // Non-copyable, movable
  AxesWidget(const AxesWidget&)            = delete;
  AxesWidget& operator=(const AxesWidget&) = delete;
  AxesWidget(AxesWidget&&) noexcept;
  AxesWidget& operator=(AxesWidget&&) noexcept;

  /**
   * @brief Register widget geometry with a renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove widget geometry from a renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Update axes length
   * @param length New axis length in meters
   */
  void set_length(double length);

  /**
   * @brief Set axes visibility
   * @param visible Visibility flag
   */
  void set_visible(bool visible);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
