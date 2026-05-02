#pragma once
#include "core/base/result.hpp"
#include "core/rl/rl.hpp"
#include "core/rl/rl_metadata.hpp"
#include <events/event_bus.hpp>
#include <memory>
#include <optional>

#include "config/rl.hpp"
#include "config/simulator.hpp"
#include "config/world.hpp"
#include "threading/thread_pool.hpp"

namespace ddrl::sim {

class MapManager;

class Simulator
{
public:
  struct Params {
    std::shared_ptr<const config::SimConfig>   config;
    std::shared_ptr<const config::WorldConfig> world_config;
    std::shared_ptr<const config::RLConfig>    rl_config;
    std::shared_ptr<threading::ThreadPool>     thread_pool;
    std::shared_ptr<events::EventBus>          event_bus; ///< Optional event bus for pub/sub
    std::shared_ptr<const config::MapConfig>   initial_map;
    std::shared_ptr<MapManager>                map_manager;
    double                                     visualization_rate_hz{0.0};
    uint32_t                                   sim_id{0}; ///< Simulator instance ID
  };

  struct StepRequest {
    /// When std::nullopt, the simulator continues with the previously applied command.
    std::optional<core::rl::Action> action;
  };

  struct StepResult {
    core::rl::Observation observation;
    double                sim_time{0.0};
    bool                  episode_done{false};
    bool                  time_limit_reached{false};
    core::rl::RLMetadata  rl_metadata{};
  };

  struct ResetParams {
  };

  struct ResetResult {
    /// Observation snapshot emitted immediately after reset to bootstrap the next action.
    StepResult  initial_observation;
    std::size_t episode_index{0};
  };

  explicit Simulator(Params params);
  ~Simulator();

  Simulator(const Simulator&)            = delete;
  Simulator& operator=(const Simulator&) = delete;
  Simulator(Simulator&&) noexcept;
  Simulator& operator=(Simulator&&) noexcept;

  core::Result<ResetResult> reset(ResetParams params = {});
  core::Result<StepResult>  step(StepRequest request);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim
