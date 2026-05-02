/**
 * @file map_geometry_manager.cpp
 * @brief Implementation of MapGeometryManager
 */

#include "visualizer/map_geometry_manager.hpp"

#include "logging/logging.hpp"

namespace ddrl::visualizer {

MapGeometryManager::MapGeometryManager(
  const std::vector<std::shared_ptr<const config::MapConfig>>& maps,
  const std::shared_ptr<const config::WorldConfig>&            world_config
)
    : logger_(logging::get_logger("MapGeometryManager"))
{
  DDRL_LOG_DEBUG(logger_, "Initializing MapGeometryManager with {} maps", maps.size());

  for (const auto& map : maps) {
    if (!map) {
      DDRL_LOG_WARN(logger_, "Skipping null map configuration");
      continue;
    }

    build_map_geometry(*map, *world_config);
  }

  DDRL_LOG_DEBUG(logger_, "MapGeometryManager initialized with {} map templates", map_cache_.size());
}

void MapGeometryManager::build_map_geometry(
  const config::MapConfig& map, const config::WorldConfig& world_config
)
{
  (void)world_config;
  DDRL_LOG_DEBUG(
    logger_,
    "Building geometry template for map '{}' with {} static objects",
    map.name,
    map.static_instances.size()
  );

  MapGeometry map_geom;
  map_geom.map_name = map.name;
  map_geom.bounds   = map.bounds;

  // Build static object templates
  map_geom.static_objects.reserve(map.static_instances.size());
  for (const auto& static_obj : map.static_instances) {
    StaticObjectTemplate obj_template;
    obj_template.model_id   = static_obj.model_id;
    obj_template.world_pose = static_obj.pose_world;
    obj_template.name       = static_obj.name;

    map_geom.static_objects.push_back(obj_template);

    DDRL_LOG_DEBUG(
      logger_,
      "  Added static object '{}' (model_id: '{}') at ({:.2f}, {:.2f}, {:.2f}°)",
      obj_template.name,
      obj_template.model_id,
      obj_template.world_pose.pos.x,
      obj_template.world_pose.pos.y,
      obj_template.world_pose.yaw * 180.0 / M_PI
    );
  }

  map_geom.routes.reserve(map.robot_routes.size());
  for (const auto& route : map.robot_routes) {
    RouteTemplate route_template;
    route_template.name       = route.name;
    route_template.start_pose = route.start;
    route_template.waypoints.reserve(route.waypoints.size());

    for (std::size_t idx = 0; idx < route.waypoints.size(); ++idx) {
      const auto& waypoint = route.waypoints[idx];
      route_template.waypoints.push_back(RouteWaypointTemplate{
        .name      = waypoint.name,
        .position  = waypoint.pos,
        .tolerance = waypoint.tolerance,
        .is_final  = (idx + 1 == route.waypoints.size()),
      });
    }

    map_geom.routes.push_back(std::move(route_template));
  }

  // Store in cache
  map_cache_[map.name] = std::move(map_geom);

  DDRL_LOG_DEBUG(logger_, "Successfully built geometry template for map '{}'", map.name);
}

const MapGeometryManager::MapGeometry*
MapGeometryManager::get_map_geometry(const std::string& map_name) const
{
  auto it = map_cache_.find(map_name);
  if (it == map_cache_.end()) {
    DDRL_LOG_ERROR(
      logger_,
      "Map '{}' not found in geometry cache. Available maps: [{}]",
      map_name,
      [this]() {
        std::string names;
        for (const auto& [name, _] : map_cache_) {
          if (!names.empty()) {
            names += ", ";
          }
          names += "'" + name + "'";
        }
        return names;
      }()
    );
    return nullptr;
  }

  return &it->second;
}

std::vector<std::string> MapGeometryManager::get_available_map_names() const
{
  std::vector<std::string> map_names;
  map_names.reserve(map_cache_.size());

  for (const auto& [name, _] : map_cache_) {
    map_names.push_back(name);
  }

  return map_names;
}

} // namespace ddrl::visualizer
