#pragma once

#include "core/rl/rl.hpp"
#include "core/rl/reward.hpp"
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace ddrl::worker {

/**
 * @brief Single step in a trajectory for IMPALA-style learning.
 */
struct TrajectoryStep {
  core::rl::Observation state;       ///< S_t
  std::vector<float>    encoded_observation_features; ///< Cached learner features for S_t
  core::rl::Action      action;      ///< A_t
  core::rl::RewardTermValues reward_terms; ///< Reward term breakdown for R_{t+1}

  double reward{0.0};              ///< R_{t+1}
  bool   is_terminal{false};       ///< Episode ended after this step
  double behaviour_log_prob{0.0};

  TrajectoryStep() = default;
  TrajectoryStep(core::rl::Observation s, std::vector<float> encoded_features, core::rl::Action a, double log_prob)
      : state(std::move(s)),
        encoded_observation_features(std::move(encoded_features)),
        action(a),
        behaviour_log_prob(log_prob)
  {
  }
};

struct RecurrentStateSnapshot {
  std::vector<float> hidden;
  std::vector<float> cell;
  std::vector<int32_t> shape;
};

/**
 * @brief Trajectory buffer for collecting experience before sending to the learner.
 * Implements IMPALA-style trajectory collection: (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, ...).
 */
class TrajectoryBuffer
{
public:
  explicit TrajectoryBuffer(size_t unroll_length);
  ~TrajectoryBuffer();
  TrajectoryBuffer(TrajectoryBuffer&&) noexcept;
  TrajectoryBuffer& operator=(TrajectoryBuffer&&) noexcept;
  TrajectoryBuffer(const TrajectoryBuffer&)            = delete;
  TrajectoryBuffer& operator=(const TrajectoryBuffer&) = delete;

  void push_step(TrajectoryStep step);
  void set_pending_step(
    core::rl::Observation&& state, std::vector<float> encoded_features, core::rl::Action action, double log_prob
  );
  TrajectoryStep complete_pending_step(
    double reward, bool is_terminal, core::rl::RewardTermValues reward_terms = {}
  );

  /// True when the buffer has reached its configured unroll length.
  [[nodiscard]] bool          is_full() const;
  /// True when no completed steps are stored.
  [[nodiscard]] bool          empty() const;
  [[nodiscard]] size_t        size() const;
  /// True when a pending (S_t, A_t) is waiting for its reward.
  [[nodiscard]] bool          has_pending() const;
  /// Move out all stored steps and clear the buffer.
  std::vector<TrajectoryStep> extract();
  void                        set_initial_recurrent_state(RecurrentStateSnapshot state);
  std::optional<RecurrentStateSnapshot> extract_initial_recurrent_state();
  void                        clear();
  [[nodiscard]] size_t        unroll_length() const;

private:
  size_t                     unroll_length_;
  std::deque<TrajectoryStep> buffer_;

  TrajectoryStep   pending_step_;
  bool             has_pending_{false};
  std::optional<RecurrentStateSnapshot> initial_recurrent_state_;
};

} // namespace ddrl::worker
