#include "worker/trajectory_buffer.hpp"

#include <stdexcept>
#include <utility>

namespace ddrl::worker {

TrajectoryBuffer::TrajectoryBuffer(size_t unroll_length)
    : unroll_length_(unroll_length)
{
}

TrajectoryBuffer::~TrajectoryBuffer()                                      = default;
TrajectoryBuffer::TrajectoryBuffer(TrajectoryBuffer&&) noexcept            = default;
TrajectoryBuffer& TrajectoryBuffer::operator=(TrajectoryBuffer&&) noexcept = default;

void TrajectoryBuffer::push_step(TrajectoryStep step)
{
  buffer_.push_back(std::move(step));
}

void TrajectoryBuffer::set_pending_step(
  core::rl::Observation&& state, std::vector<float> encoded_features, core::rl::Action action,
  double log_prob
)
{
  pending_step_                 =
    TrajectoryStep(std::move(state), std::move(encoded_features), action, log_prob);
  has_pending_                  = true;
}

TrajectoryStep TrajectoryBuffer::complete_pending_step(
  double reward, bool is_terminal, core::rl::RewardTermValues reward_terms
)
{
  if (!has_pending_) {
    throw std::logic_error("complete_pending_step called without pending step");
  }
  pending_step_.reward      = reward;
  pending_step_.is_terminal = is_terminal;
  pending_step_.reward_terms = std::move(reward_terms);

  has_pending_     = false;
  return std::move(pending_step_);
}

bool TrajectoryBuffer::is_full() const
{
  return buffer_.size() >= unroll_length_;
}

bool TrajectoryBuffer::empty() const
{
  return buffer_.empty();
}

size_t TrajectoryBuffer::size() const
{
  return buffer_.size();
}

bool TrajectoryBuffer::has_pending() const
{
  return has_pending_;
}

std::vector<TrajectoryStep> TrajectoryBuffer::extract()
{
  std::vector<TrajectoryStep> out(
    std::move_iterator(buffer_.begin()), std::move_iterator(buffer_.end())
  );
  buffer_.clear();
  return out;
}

void TrajectoryBuffer::set_initial_recurrent_state(RecurrentStateSnapshot state)
{
  if (!initial_recurrent_state_) {
    initial_recurrent_state_ = std::move(state);
  }
}

std::optional<RecurrentStateSnapshot> TrajectoryBuffer::extract_initial_recurrent_state()
{
  auto out = std::move(initial_recurrent_state_);
  initial_recurrent_state_.reset();
  return out;
}

void TrajectoryBuffer::clear()
{
  buffer_.clear();
  has_pending_     = false;
  pending_step_    = {};
  initial_recurrent_state_.reset();
}

size_t TrajectoryBuffer::unroll_length() const
{
  return unroll_length_;
}

} // namespace ddrl::worker
