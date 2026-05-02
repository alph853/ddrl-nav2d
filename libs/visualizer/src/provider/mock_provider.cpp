/**
 * @file mock_provider.cpp
 * @brief Mock world state provider implementation
 */

#include "visualizer/provider/mock_provider.hpp"
#include <unordered_map>

namespace ddrl::visualizer {

class MockStateProvider::Impl
{
public:
  std::unordered_map<uint32_t, std::function<core::sim::WorldStateSnapshot()>> generators_;
  std::unordered_map<uint32_t, core::sim::WorldStateSnapshot>                  cached_states_;
};

MockStateProvider::MockStateProvider()
    : pimpl_(std::make_unique<Impl>())
{
}

MockStateProvider::~MockStateProvider() = default;

void MockStateProvider::set_generator(
    uint32_t                                        sim_id,
    std::function<core::sim::WorldStateSnapshot()> generator)
{
  pimpl_->generators_[sim_id] = std::move(generator);
}

core::Result<void> MockStateProvider::update()
{
  // Generate new states from all generators and cache them
  for (const auto& [sim_id, generator] : pimpl_->generators_) {
    if (generator) {
      pimpl_->cached_states_[sim_id] = generator();
    }
  }
  return {};
}

core::Result<core::sim::WorldStateSnapshot> MockStateProvider::get_world_state(uint32_t sim_id)
{
  // Return cached state if available
  auto it = pimpl_->cached_states_.find(sim_id);
  if (it != pimpl_->cached_states_.end()) {
    return it->second;
  }
  return {};
}

bool MockStateProvider::is_active() const
{
  return !pimpl_->generators_.empty();
}

void MockStateProvider::clear()
{
  pimpl_->generators_.clear();
  pimpl_->cached_states_.clear();
}

} // namespace ddrl::visualizer
