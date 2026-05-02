/**
 * @file event_bus_provider.hpp
 * @brief EventBus-based world state provider declaration
 */

#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>

#include "logging/logging.hpp"
#include "visualizer/interface/base_provider.hpp"

namespace ddrl::events {
class EventBus;
}

namespace ddrl::visualizer {

/**
 * @brief EventBus-based world state provider
 *
 * Polls world state events from the event bus for visualization.
 */
class EventBusStateProvider : public IVisualizerProvider
{
public:
  /**
   * @brief Construct provider from event bus
   * @param event_bus Shared event bus instance
   */
  explicit EventBusStateProvider(std::shared_ptr<events::EventBus> event_bus);

  /**
   * @brief Update provider - fetches latest data from event bus
   */
  core::Result<void> update() override;

  /**
   * @brief Get cached world state for a sim
   */
  core::Result<core::sim::WorldStateSnapshot> get_world_state(uint32_t sim_id) override;

  [[nodiscard]] bool is_active() const override;

private:
  struct CachedState {
    core::sim::WorldStateSnapshot                              snapshot;
    std::chrono::steady_clock::time_point                      cached_at;
    double                                                     sim_timestamp{0.0};
  };

  static constexpr double kStalenessThresholdS = 1.0; ///< Max age for cached data (seconds)

  std::shared_ptr<logging::Logger>  logger_ = logging::get_logger("visualizer.event_bus_provider");
  std::shared_ptr<events::EventBus> event_bus_;
  std::unordered_map<uint32_t, CachedState> cached_states_;
};

} // namespace ddrl::visualizer
