/**
 * @file map_geometry_manager.hpp
 * @brief Manager for pre-building and caching map geometry templates
 */

#pragma once

#include "core/geom/aabb.hpp"
#include "core/math/pose.hpp"
#include "core/math/primitives.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/map.hpp"
#include "config/world.hpp"
#include "logging/logging.hpp"

namespace ddrl::visualizer {

/**
 * @brief Manages pre-built geometry templates for maps
 *
 * MapGeometryManager analyzes map configurations and builds lightweight templates
 * that can be used by each viewport to instantiate their own VTK actors.
 * This follows the same pattern as ProfileManager but for map geometry.
 *
 * Key design principles:
 * - Stores templates/metadata, NOT VTK actors (actors can't be shared between viewports)
 * - Each viewport creates its own actors from these templates
 * - Static object geometry is instantiated via ProfileManager
 * - No fallback behavior - errors are reported clearly
 */
class MapGeometryManager
{
public:
  /**
   * @brief Template for a static object instance in a map
   *
   * Contains reference to model profile and world pose.
   * Actual VTK geometry is created via ProfileManager.
   */
  struct StaticObjectTemplate {
    std::string       model_id;   ///< Reference to ProfileManager model
    core::math::Pose2 world_pose; ///< Absolute world position
    std::string       name;       ///< For debugging
  };

  struct RouteWaypointTemplate {
    std::string          name;
    core::math::Point2   position;
    double               tolerance{1.0};
    bool                 is_final{false};
  };

  struct RouteTemplate {
    std::string                       name;
    core::math::Pose2                 start_pose;
    std::vector<RouteWaypointTemplate> waypoints;
  };

  /**
   * @brief Complete geometry template for a map
   *
   * This is a lightweight template that viewports use to instantiate
   * their own actors. No VTK objects are stored here.
   */
  struct MapGeometry {
    std::string                       map_name;
    std::vector<StaticObjectTemplate> static_objects; ///< Template for static objects
    std::vector<RouteTemplate>        routes;         ///< Template for robot routes
    core::geom::AABB2                 bounds;         ///< Optional map bounds
  };

  /**
   * @brief Initialize and build geometry templates for all provided maps
   *
   * @param maps Vector of map configurations to pre-build
   * @param world_config World configuration (for zone type resolution)
   *
   * @note This analyzes all maps and builds templates. Actual VTK actors
   *       are created per-viewport on demand.
   */
  explicit MapGeometryManager(
    const std::vector<std::shared_ptr<const config::MapConfig>>& maps,
    const std::shared_ptr<const config::WorldConfig>&            world_config
  );

  /**
   * @brief Get geometry template for a map
   *
   * @param map_name Name of the map to retrieve
   * @return Pointer to geometry template, or nullptr if not found
   *
   * @note Returns nullptr on error - caller must handle this
   */
  [[nodiscard]] const MapGeometry* get_map_geometry(const std::string& map_name) const;

  /**
   * @brief Get list of all cached map names
   *
   * @return Vector of map names available in the cache
   */
  [[nodiscard]] std::vector<std::string> get_available_map_names() const;

private:
  /**
   * @brief Build geometry template for a single map
   */
  void build_map_geometry(const config::MapConfig& map, const config::WorldConfig& world_config);

  std::unordered_map<std::string, MapGeometry> map_cache_;
  std::shared_ptr<logging::Logger>             logger_;
};

} // namespace ddrl::visualizer
