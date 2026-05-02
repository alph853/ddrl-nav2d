#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "logging/logging.hpp"
#include "worker/trajectory_buffer.hpp"

namespace ddrl::worker {

/**
 * @brief Serialized experience batch emitted by a simulator instance.
 */
struct Rollout {
  uint32_t sim_id{0};
  uint64_t rollout_id{0};
  uint64_t policy_version{0};
  bool     is_expert{false};
  double   episode_return{0.0};
  bool     is_terminal{false};

  std::chrono::system_clock::time_point timestamp;
  std::vector<worker::TrajectoryStep>   steps;
  std::optional<worker::RecurrentStateSnapshot> initial_recurrent_state;
};

/**
 * @brief Asynchronously publishes rollouts to external sinks.
 */
class RolloutPublisher
{
public:
  struct Params {
    bool                             enable_streaming{false};
    size_t                           max_queue_depth{4096};
    size_t                           max_batch_rollouts{8};
    std::chrono::milliseconds        flush_interval{5};
    std::shared_ptr<logging::Logger> logger;
  };

  using PublishHandler = std::function<void(const std::vector<Rollout>&)>;

  RolloutPublisher(Params params, PublishHandler handler);
  ~RolloutPublisher();
  RolloutPublisher(RolloutPublisher&&)            = delete;
  RolloutPublisher& operator=(RolloutPublisher&&) = delete;

  void start();
  void stop();

  bool               enqueue(Rollout rollout);
  [[nodiscard]] bool running() const;

private:
  void thread_main();
  void publish_batch(const std::vector<Rollout>& batch);

  Params                params_;
  PublishHandler        handler_;
  std::atomic<bool>     running_{false};
  std::atomic<bool>     should_stop_{false};

  std::mutex              mutex_;
  std::condition_variable cv_;
  std::deque<Rollout>     queue_;

  std::thread                        worker_thread_;
};

} // namespace ddrl::worker
