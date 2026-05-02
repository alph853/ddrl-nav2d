/**
 * @file visualizer.hpp
 * @brief Core visualizer for 2.5D simulation visualization
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "config/map.hpp"
#include "config/visualizer.hpp"
#include "config/world.hpp"
#include "core/rl/rl_metadata.hpp"
#include "core/vision/point_cloud.hpp"
#include "interface/base_provider.hpp"

// Forward declare select VTK types to avoid pulling VTK headers here
class vtkRenderer;

namespace ddrl::visualizer {

/**
 * @brief Main visualizer managing multiple viewports
 *
 * Handles rendering of simulation world states with support for
 * multiple view modes (focused, grid) and interactive controls.
 */
class Visualizer
{
public:
  struct Params {
    std::shared_ptr<IVisualizerProvider>        provider; ///< Data provider for fetching states
    std::shared_ptr<const config::VizConfig>    config;   ///< Visualizer configuration
    std::vector<uint32_t>                       sim_ids;  ///< Simulator instance IDs to visualize
    std::shared_ptr<const config::WorldConfig>  world_config;
    std::shared_ptr<const config::RobotProfile> robot_profile;
    std::vector<std::shared_ptr<const config::MapConfig>>
      maps; ///< All maps for pre-building geometry
    std::function<void()> on_shutdown; ///< Invoked when the user requests the viz to close
  };

  explicit Visualizer(Params params);
  ~Visualizer();

  // Non-copyable, non-movable (due to VTK resource management)
  Visualizer(const Visualizer&)            = delete;
  Visualizer& operator=(const Visualizer&) = delete;
  Visualizer(Visualizer&&)                 = delete;
  Visualizer& operator=(Visualizer&&)      = delete;

  /**
   * @brief Initialize VTK rendering context
   * Must be called on the same thread that will call start_interactive()
   * This is separated from the constructor to support multi-threaded usage
   */
  void init();

  /**
   * @brief Start interactive event loop (blocking)
   * Uses VTK timer for automatic updates at target FPS
   * Must call init() first on the same thread
   */
  void start_interactive();

  /**
   * @brief Stop interactive event loop
   * Signals the VTK interactor to terminate gracefully.
   * Causes start_interactive() to return.
   */
  void stop_interactive();

  /**
   * @brief Update RL information display
   */
  void update_rl_info(uint32_t sim_id, const core::rl::RLMetadata& metadata);

  /**
   * @brief Update worker-side observation debug points for a simulator viewport.
   */
  void update_observation_debug_cloud(
    uint32_t sim_id, std::shared_ptr<const core::vision::PointCloud> cloud
  );

private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  friend class Viewport; // Viewport needs access to internal renderer
};

} // namespace ddrl::visualizer
