/**
 * @file mock_provider.hpp
 * @brief Mock world state provider for testing
 */

#pragma once

#include "visualizer/interface/base_provider.hpp"
#include <cstdint>
#include <functional>

namespace ddrl::visualizer {

/**
 * @brief Mock world state provider for testing
 *
 * Generates simple mock world state snapshots without requiring EventBus.
 * Useful for unit tests and offline visualization development.
 */
class MockStateProvider : public IVisualizerProvider
{
public:
  MockStateProvider();
  ~MockStateProvider() override;

  /**
   * @brief Set a callback to generate states on-demand
   * @param sim_id Simulation instance ID
   * @param generator Function that generates world states
   */
  void set_generator(uint32_t sim_id,
                     std::function<core::sim::WorldStateSnapshot()> generator);

  /**
   * @brief Update provider - generates new states from generators
   */
  core::Result<void> update() override;

  /**
   * @brief Get cached world state for a sim
   */
  core::Result<core::sim::WorldStateSnapshot> get_world_state(uint32_t sim_id) override;

  /**
   * @brief Check if there are any active generators
   */
  [[nodiscard]] bool is_active() const override;

  /**
   * @brief Clear all generators and cached states
   */
  void clear();

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace ddrl::visualizer
