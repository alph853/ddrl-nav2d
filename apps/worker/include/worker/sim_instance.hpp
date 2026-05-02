#pragma once

#include <atomic>
#include <core/base/result.hpp>
#include <core/rl/rl.hpp>
#include <events/event_bus.hpp>
#include <memory>
#include <thread>

#include "policy_model.hpp"
#include "trajectory_buffer.hpp"

namespace ddrl::sim {
class Simulator;
}

namespace ddrl::worker {

/**
 * @brief Lifecycle state for a single simulator thread.
 */
struct SimInstance {
  struct ExpertState {
    bool reverse_recovery_active{false};
  };

  uint32_t                              sim_id{0};
  std::shared_ptr<ddrl::sim::Simulator> simulator;
  std::jthread                          thread;
  TrajectoryBuffer trajectory_buffer; ///< Unroll length configured at construction
  uint64_t         rollout_sequence{0};

  PolicyModel::PolicyContext policy;
  ExpertState                expert;

  explicit SimInstance(size_t unroll_length = 20) : trajectory_buffer(unroll_length) {}

  // std::thread and std::atomic are not copyable
  SimInstance(const SimInstance&)            = delete;
  SimInstance& operator=(const SimInstance&) = delete;

  // std::atomic cannot be moved, must copy value
  SimInstance(SimInstance&& other) noexcept
      : sim_id(other.sim_id),
        simulator(std::move(other.simulator)),
        thread(std::move(other.thread)),
        trajectory_buffer(std::move(other.trajectory_buffer)),
        rollout_sequence(other.rollout_sequence),
        policy(std::move(other.policy)),
        expert(other.expert)
  {
  }

  // std::atomic cannot be moved, must copy value
  SimInstance& operator=(SimInstance&& other) noexcept
  {
    if (this != &other) {
      if (thread.joinable()) {
        thread.join();
      }
      sim_id    = other.sim_id;
      simulator = std::move(other.simulator);
      thread    = std::move(other.thread);
      trajectory_buffer = std::move(other.trajectory_buffer);
      rollout_sequence  = other.rollout_sequence;
      policy            = std::move(other.policy);
      expert            = other.expert;
    }
    return *this;
  }
};
} // namespace ddrl::worker
