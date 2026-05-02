/**
 * @file sim_models.hpp
 * @brief Simulation-related event models
 * @version 0.1
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <core/sim/world_state.hpp>

#include "base_event.hpp"

namespace ddrl::events {

// ============================================================================
// World State Events (for visualization)
// ============================================================================

struct WorldStateEvent {
  EventMetadata                                        meta;
  std::shared_ptr<const core::sim::WorldStateSnapshot> state; ///< Shared world state

  WorldStateEvent() = default;
  WorldStateEvent(uint32_t sim_id, double timestamp,
    std::shared_ptr<const core::sim::WorldStateSnapshot> snapshot)
      : meta{sim_id, timestamp, 0}, state(std::move(snapshot))
  {
  }
};

} // namespace ddrl::events
