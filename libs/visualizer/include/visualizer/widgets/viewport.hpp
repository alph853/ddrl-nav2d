/**
 * @file viewport.hpp
 * @brief Viewport widget interface (Pimpl) for a single simulation view
 */

#pragma once

#include "core/sim/world_state.hpp"
#include "core/vision/point_cloud.hpp"
#include <array>
#include <memory>

#include "config/world.hpp"

class vtkRenderer;

namespace ddrl::visualizer {

// Forward declarations
class ProfileManager;
class MapGeometryManager;

/**
 * @brief Single viewport managing one sim instance
 */
class Viewport
{
public:
  struct Params {
    uint32_t                                    sim_id{0};
    std::array<double, 4>                       viewport_rect{0.0, 0.0, 1.0, 1.0};
    bool                                        show_grid{true};
    bool                                        show_pointcloud{true};
    bool                                        show_sensor_frustums{true};
    std::shared_ptr<const ProfileManager>       profile_manager;
    std::shared_ptr<const MapGeometryManager>   map_geometry_manager;
    std::shared_ptr<const config::WorldConfig>  world_config;
    std::shared_ptr<const config::RobotProfile> robot_profile;
  };
  explicit Viewport(Params params);
  ~Viewport();

  // Non-copyable, movable
  // Viewport(const Viewport&)            = delete;
  // Viewport& operator=(const Viewport&) = delete;
  // Viewport(Viewport&&) noexcept;
  // Viewport& operator=(Viewport&&) noexcept;

  void update(const core::sim::WorldStateSnapshot& state);
  void set_viewport_rect(double xmin, double ymin, double xmax, double ymax);
  void set_show_pointcloud(bool enable);
  void set_show_sensor_frustums(bool enable);
  void toggle_route_visualization_mode();
  void set_observation_debug_cloud(std::shared_ptr<const core::vision::PointCloud> cloud);
  [[nodiscard]] bool show_pointcloud_enabled() const;
  [[nodiscard]] bool show_sensor_frustums_enabled() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  friend class Visualizer;

  // Note: to keep API VTK-free, we expose the renderer only to Visualizer
  [[nodiscard]] ::vtkRenderer* renderer_for_friend() const;
};

} // namespace ddrl::visualizer
