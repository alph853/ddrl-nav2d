/**
 * @file thread_pool.hpp
 * @brief Thread pool implementation for parallel task execution
 */

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace ddrl::threading {

/**
 * @class ThreadPool
 * @brief A thread pool for executing tasks asynchronously
 *
 * ThreadPool manages a pool of worker threads that execute submitted tasks.
 * It supports both individual task submission and parallel-for operations.
 *
 * Thread Safety:
 * - Multiple threads can safely call submit() and parallel_for() simultaneously
 * - If a shared_ptr<ThreadPool> is shared among multiple owners, they can ALL
 *   execute tasks simultaneously. The ThreadPool is internally synchronized.
 * - resize() and stop() should not be called concurrently with task submissions
 *
 * @example
 * @code
 * auto pool = std::make_shared<ThreadPool>(4, "worker_pool");
 *
 * // Submit a task
 * auto future = pool->submit([] { return 42; });
 * int result = future.get();
 *
 * // Parallel for loop
 * pool->parallel_for(0, 100, [](size_t i) {
 *   // Process item i
 * });
 * @endcode
 */
class ThreadPool
{
public:
  /**
   * @brief Construct a thread pool with a specified number of threads
   *
   * @param nthreads Number of worker threads to create
   * @param name Optional name for the thread pool (useful for debugging)
   *
   * @throws std::system_error if threads cannot be created
   */
  explicit ThreadPool(size_t nthreads, std::string name = "pool") : name_(std::move(name))
  {
    resize(nthreads);
  }

  /**
   * @brief Destructor - stops all threads and waits for them to finish
   */
  ~ThreadPool() { stop(); }

  // Non-copyable, non-movable (contains mutex and threads)
  ThreadPool(const ThreadPool&)            = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&)                 = delete;
  ThreadPool& operator=(ThreadPool&&)      = delete;

  /**
   * @brief Resize the thread pool
   *
   * Stops all current threads, discards pending tasks, and creates
   * a new set of worker threads.
   *
   * @param n New number of worker threads
   *
   * @warning This operation is NOT thread-safe with respect to task submission.
   *          Do not call resize() while other threads are submitting tasks.
   */
  void resize(size_t n)
  {
    stop();
    running_ = true;
    for (size_t i = 0; i < n; ++i) {
      workers_.emplace_back([this, i] { this->worker_loop(i); });
    }
  }

  /**
   * @brief Submit a task for asynchronous execution
   *
   * @tparam F Callable type (function, lambda, functor)
   * @param fn Task to execute
   * @return std::future containing the result of the task
   *
   * @note Multiple threads can safely call submit() simultaneously.
   *
   * @example
   * @code
   * auto future = pool.submit([] { return 42; });
   * int result = future.get(); // Blocks until task completes
   * @endcode
   */
  template <class F>
  auto submit(F&& fn) -> std::future<decltype(fn())>
  {
    using R   = decltype(fn());
    auto task = std::packaged_task<R()>(std::forward<F>(fn));
    auto fut  = task.get_future();
    {
      std::scoped_lock lk(lock_);
      task_q_.emplace([t = std::move(task)]() mutable { t(); });
    }
    condvar_.notify_one();
    return fut;
  }

  /**
   * @brief Execute a function in parallel over a range of indices
   *
   * Divides the range [begin, end) into chunks and executes fn(i) for each
   * index i in parallel across worker threads. Blocks until all iterations complete.
   *
   * @tparam F Callable type accepting a single std::size_t parameter
   * @param begin Start index (inclusive)
   * @param end End index (exclusive)
   * @param fn Function to call for each index
   * @param grain Minimum chunk size (default: 1)
   *
   * @note If the range is small or grain size is large, the function may
   *       execute serially on the calling thread.
   * @note This function blocks until all parallel work is complete.
   * @note Multiple threads can safely call parallel_for() simultaneously.
   *
   * @example
   * @code
   * std::vector<int> data(1000);
   * pool.parallel_for(0, data.size(), [&](size_t i) {
   *   data[i] = i * i;
   * });
   * @endcode
   */
  template <class F>
  void parallel_for(std::size_t begin, std::size_t end, F&& fn, std::size_t grain = 1)
  {
    if (begin >= end) {
      return;
    }
    const std::size_t count = end - begin;
    if (workers_.empty() || count <= grain) {
      for (std::size_t i = begin; i < end; ++i) {
        fn(i);
      }
      return;
    }

    const std::size_t threads = std::max<std::size_t>(1, workers_.size());
    const std::size_t chunk   = std::max<std::size_t>(grain, (count + threads - 1) / threads);
    std::vector<std::future<void>> futures;
    futures.reserve((count + chunk - 1) / chunk);

    auto func = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));

    for (std::size_t start = begin; start < end; start += chunk) {
      const std::size_t stop = std::min(start + chunk, end);
      futures.emplace_back(submit([start, stop, func]() {
        for (std::size_t idx = start; idx < stop; ++idx) {
          (*func)(idx);
        }
      }));
    }

    for (auto& f : futures) {
      f.get();
    }
  }

  /**
   * @brief Stop all worker threads and discard pending tasks
   * Waits for currently executing tasks to complete, but discards
   * any tasks still in the queue.
   *
   * @note Can be called multiple times safely (subsequent calls are no-ops).
   * @warning Not thread-safe with respect to task submission.
   */
  void stop()
  {
    if (!running_) {
      return;
    }
    {
      std::scoped_lock lk(lock_);
      running_ = false;
    }
    condvar_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
    workers_.clear();
    // Drain task queue
    std::queue<Task> empty;
    std::swap(task_q_, empty);
  }

  /**
   * @brief Get the number of worker threads
   * @return Current number of worker threads in the pool
   */
  [[nodiscard]] size_t size() const { return workers_.size(); }
  /**
   * @brief Get the number of tasks waiting in the queue
   * @return Pending task count
   */
  size_t pending_tasks()
  {
    std::scoped_lock lk(lock_);
    return task_q_.size();
  }

private:
  /**
   * @brief Worker thread main loop
   * Each thread waits for tasks on the queue and executes them.
   * @param idx Worker thread index (for debugging)
   */
  void worker_loop([[maybe_unused]] size_t idx)
  {
    for (;;) {
      Task job{[] {}};
      {
        std::unique_lock lk(lock_);
        condvar_.wait(lk, [&] { return !running_ || !task_q_.empty(); });
        if (!running_ && task_q_.empty()) {
          return;
        }
        job = std::move(task_q_.front());
        task_q_.pop();
      }
      job();
    }
  }

  // Move-only function wrapper for type-erased callables
  struct Task {
    struct Concept {
      virtual ~Concept()  = default;
      virtual void call() = 0;
    };

    template <typename F>
    struct Model : Concept {
      F func_;
      explicit Model(F&& f) : func_(std::forward<F>(f)) {}
      void call() override { func_(); }
    };

    std::unique_ptr<Concept> impl_;

    template <typename F>
    explicit Task(F&& f) : impl_(std::make_unique<Model<std::decay_t<F>>>(std::forward<F>(f)))
    {
    }

    Task(Task&&)            = default;
    Task& operator=(Task&&) = default;

    void operator()() { impl_->call(); }
  };

  std::string name_; ///< Pool name for debugging
  std::mutex  lock_; ///< Protects task queue and running flag

  std::condition_variable  condvar_; ///< Notifies workers of new tasks
  std::queue<Task>         task_q_;  ///< Queue of pending tasks (move-only)
  std::vector<std::thread> workers_; ///< Worker threads

  bool running_{false}; ///< Flag indicating pool is running
};

} // namespace ddrl::threading
