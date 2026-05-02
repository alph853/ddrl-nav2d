/**
 * @file event_bus_provider.cpp
 * @brief EventBus-based world state provider implementation
 */

#include "visualizer/provider/event_bus_provider.hpp"
#include "events/event_bus.hpp"
#include "events/sim_models.hpp"

namespace ddrl::visualizer {

EventBusStateProvider::EventBusStateProvider(std::shared_ptr<events::EventBus> event_bus)
    : event_bus_(std::move(event_bus))
{
}

core::Result<void> EventBusStateProvider::update()
{
  if (!event_bus_) {
    return {};
  }

  return {};
}

core::Result<core::sim::WorldStateSnapshot> EventBusStateProvider::get_world_state(uint32_t sim_id)
{
  if (!event_bus_) {
    return {};
  }
  if (!event_bus_->is_registered<events::WorldStateEvent>(sim_id)) {
    return std::unexpected(
      core::make_error(
        core::ErrorCode::EVENT_BUS,
        std::format("No WorldStateEvent registered for sim_id {} in event bus", sim_id),
        core::make_context("EventBusStateProvider::get_world_state", "sim_id", sim_id)
      )
    );
  }

  // Try to fetch from event bus
  events::WorldStateEvent event;
  if (event_bus_->try_dequeue<events::WorldStateEvent>(sim_id, event)) {
    if (event.state) {
      // Cache the state with current timestamp
      CachedState cached;
      cached.snapshot       = *event.state;
      cached.cached_at      = std::chrono::steady_clock::now();
      cached.sim_timestamp  = event.meta.timestamp;
      cached_states_[sim_id] = std::move(cached);
      return *event.state;
    }
  } else {
    DDRL_LOG_DEBUG(
      logger_,
      "No WorldStateEvent available for sim_id {} in event bus",
      sim_id
    );
  }

  // Check if we have cached state
  auto it = cached_states_.find(sim_id);
  if (it != cached_states_.end()) {
    const auto& cached = it->second;

    // Check if cached data is stale (wall clock time)
    const auto now         = std::chrono::steady_clock::now();
    const auto age_seconds = std::chrono::duration<double>(now - cached.cached_at).count();

    if (age_seconds > kStalenessThresholdS) {
      // Data is stale, return error instead
      DDRL_LOG_WARN_THROTTLE(
        logger_, 5000,
        "Cached WorldStateEvent for sim_id {} is stale ({:.3f}s old, threshold={:.3f}s)",
        sim_id,
        age_seconds,
        kStalenessThresholdS
      );
      return std::unexpected(
        core::make_error(
          core::ErrorCode::EVENT_BUS,
          std::format(
            "Cached world state is stale ({:.3f}s old, threshold={:.3f}s)",
            age_seconds,
            kStalenessThresholdS
          ),
          core::make_context("EventBusStateProvider::get_world_state", "sim_id", sim_id)
        )
      );
    }

    return cached.snapshot;
  }

  return {};
}

bool EventBusStateProvider::is_active() const
{
  return event_bus_ != nullptr;
}

} // namespace ddrl::visualizer
