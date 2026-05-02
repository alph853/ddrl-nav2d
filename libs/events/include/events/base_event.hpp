#pragma once

#include <cstdint>

namespace ddrl::events {

// ============================================================================
// Event Metadata
// ============================================================================

/// Common metadata for all events
struct EventMetadata {
  uint32_t sim_id{0};      ///< Source simulator instance ID
  double   timestamp{0.0}; ///< Simulation time (seconds)
  uint64_t sequence{0};    ///< Monotonic sequence number for ordering
};


} // namespace ddrl::events
