/**
 * @file world_state_provider.hpp
 * @brief Abstract interface for providing world state data to visualizer
 *
 * This abstraction decouples the Viewer from specific data sources,
 * enabling easy testing with mocks and flexibility to add other data sources
 * (file replay, network streaming, etc.) in the future.
 */

#pragma once

#include "core/base/result.hpp"
#include "core/sim/world_state.hpp"
#include <cstdint>

namespace ddrl::visualizer {

/**
 * @brief Abstract interface for world state data provider
 *
 * Implementations provide world state snapshots for visualization,
 * abstracting away the underlying data source mechanism.
 */
class IVisualizerProvider
{
public:
  virtual ~IVisualizerProvider() = default;

  /**
   * @brief Update the provider (e.g., fetch new data) using pull-observer
   * pattern
   */
  virtual core::Result<void> update() { return {}; }

  /**
   * @brief Poll for new world state for a specific simulation
   */
  virtual core::Result<core::sim::WorldStateSnapshot> get_world_state(uint32_t sim_id) = 0;

  /**
   * @brief Check if provider is still active/valid
   */
  [[nodiscard]] virtual bool is_active() const { return true; }
};

} // namespace ddrl::visualizer
