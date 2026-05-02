#pragma once

#include <algorithm>
#include <chrono>
#include <thread>

namespace ddrl::core {

/**
 * @brief Utility that enforces a real-time pacing constraint on simulator loops.
 */
class RealTimePacer {
public:
  using Clock = std::chrono::steady_clock;

  explicit RealTimePacer(double real_time_factor)
    : real_time_factor_(real_time_factor), enabled_(real_time_factor > 0.0)
  {
  }

  /**
   * @brief Reset pacing state with a new simulation timestamp.
   */
  void reset(double initial_sim_time)
  {
    last_sim_time_            = initial_sim_time;
    expected_wall_elapsed_s_  = 0.0;
    wall_origin_              = Clock::now();
  }

  /**
   * @brief Sleep as needed so that wall clock time stays ahead of simulation time.
   */
  void pace(double sim_time)
  {
    if (!enabled_) {
      last_sim_time_ = sim_time;
      return;
    }

    const double sim_dt = std::max(sim_time - last_sim_time_, 0.0);
    expected_wall_elapsed_s_ += sim_dt / real_time_factor_;
    const double wall_elapsed_s =
      std::chrono::duration<double>(Clock::now() - wall_origin_).count();
    const double sleep_s = expected_wall_elapsed_s_ - wall_elapsed_s;
    if (sleep_s > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
    last_sim_time_ = sim_time;
  }

  [[nodiscard]] bool enabled() const { return enabled_; }

private:
  double            real_time_factor_{0.0};
  bool              enabled_{false};
  double            last_sim_time_{0.0};
  double            expected_wall_elapsed_s_{0.0};
  Clock::time_point wall_origin_{Clock::now()};
};

} // namespace ddrl::core
