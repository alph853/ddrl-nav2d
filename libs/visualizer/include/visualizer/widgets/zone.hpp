/**
 * @file zone.hpp
 * @brief Widget for visualizing Zone2 areas
 *
 * This widget creates a semi-transparent polygon on the ground plane
 * to visualize zones (start, goal, no-go, etc.) in the simulation.
 *
 * @note This widget depends on VTK (Visualization Toolkit).
 *       Requires: vtkActor, vtkSmartPointer, vtkPolyData, vtkPolyDataMapper
 */

#pragma once

#include "core/geom/shape.hpp"
#include <array>
#include <memory>
#include <string>

class vtkRenderer;

namespace ddrl::visualizer::widgets {

/**
 * @brief Zone type for determining visualization color
 */
enum class ZoneType : uint8_t
{
  START,  ///< Start zone - green
  GOAL,   ///< Goal zone - blue/cyan
  NO_GO,  ///< No-go zone - red
  OTHER   ///< Other/generic zone - gray
};

/**
 * @brief Widget for rendering Zone2 polygons on the ground plane
 * Creates a semi-transparent filled polygon for spatial zone visualization
 * @note Depends on VTK library
 */
class ZoneActor
{
public:
  /**
   * @brief Construct zone actor
   * @param name Zone name
   * @param polygon 2D polygon defining the zone boundary
   * @param zone_type Type of zone (determines color)
   * @param height Height of the zone visualization (z_max)
   * @param opacity Transparency (0.0 = fully transparent, 1.0 = opaque)
   */
  ZoneActor(const std::string&            name,
            const core::geom::Polygon2&   polygon,
            ZoneType                      zone_type = ZoneType::OTHER,
            double                        height    = 0.05,
            double                        opacity   = 0.3);

  ~ZoneActor();

  // Non-copyable, movable
  ZoneActor(const ZoneActor&)            = delete;
  ZoneActor& operator=(const ZoneActor&) = delete;
  ZoneActor(ZoneActor&&) noexcept;
  ZoneActor& operator=(ZoneActor&&) noexcept;

  /**
   * @brief Register widget geometry with a renderer
   */
  void add_to_renderer(vtkRenderer& renderer);

  /**
   * @brief Remove widget geometry from a renderer
   */
  void remove_from_renderer(vtkRenderer& renderer);

  /**
   * @brief Set zone visibility
   */
  void set_visible(bool visible);

  /**
   * @brief Update zone color
   * @note r, g, b are in range [0-1]
   */
  void set_color(double r, double g, double b);

  /**
   * @brief Update zone opacity
   * @param opacity Transparency (0.0 = fully transparent, 1.0 = opaque)
   */
  void set_opacity(double opacity);

  /**
   * @brief Get zone name
   */
  [[nodiscard]] const std::string& get_name() const;

  /**
   * @brief Get default color for a zone type
   * @return RGB color array
   */
  static std::array<double, 3> get_color_for_type(ZoneType type);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
