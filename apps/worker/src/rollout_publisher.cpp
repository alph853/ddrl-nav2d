#include "worker/rollout_publisher.hpp"

#include <utility>

namespace ddrl::worker {

namespace {
} // namespace

RolloutPublisher::RolloutPublisher(Params params, PublishHandler handler)
    : params_(std::move(params)),
      handler_(std::move(handler))
{
}

RolloutPublisher::~RolloutPublisher()
{
  stop();
}

void RolloutPublisher::start()
{
  if (!params_.enable_streaming) {
    return;
  }
  if (running_.load(std::memory_order_relaxed)) {
    return;
  }

  should_stop_.store(false);
  worker_thread_ = std::thread([this]() { thread_main(); });
  running_.store(true, std::memory_order_relaxed);
}

void RolloutPublisher::stop()
{
  should_stop_.store(true);
  cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  running_.store(false, std::memory_order_relaxed);
}

bool RolloutPublisher::enqueue(Rollout rollout)
{
  if (!params_.enable_streaming) {
    return false;
  }

  std::lock_guard lock(mutex_);
  if (queue_.size() >= params_.max_queue_depth) {
    if (params_.logger) {
      DDRL_LOG_WARN(
        params_.logger,
          "RolloutPublisher queue full ({}). Dropping rollout from sim {} batch {}",
        queue_.size(),
        rollout.sim_id,
        rollout.rollout_id
      );
    }
    return false;
  }
  queue_.push_back(std::move(rollout));
  cv_.notify_one();
  return true;
}

bool RolloutPublisher::running() const
{
  return running_.load(std::memory_order_relaxed);
}

void RolloutPublisher::thread_main()
{
  if (!params_.enable_streaming) {
    return;
  }

  std::unique_lock lock(mutex_);
  while (!should_stop_.load(std::memory_order_relaxed)) {
    cv_.wait_for(lock, params_.flush_interval, [this]() {
      return should_stop_.load(std::memory_order_relaxed) || !queue_.empty();
    });

    if (queue_.empty()) {
      continue;
    }

    std::vector<Rollout> batch;
    batch.reserve(params_.max_batch_rollouts);
    while (!queue_.empty() && batch.size() < params_.max_batch_rollouts) {
      batch.push_back(std::move(queue_.front()));
      queue_.pop_front();
    }

    lock.unlock();
    publish_batch(batch);
    lock.lock();
  }
}

void RolloutPublisher::publish_batch(const std::vector<Rollout>& batch)
{
  if (batch.empty()) {
    return;
  }

  if (handler_) {
    handler_(batch);
  }
}

} // namespace ddrl::worker
