#include <cmath>

#include "sim/sensors.hpp"

namespace ddrl::sim {

using core::ErrorCode;
using core::make_error;
using core::math::Point3;
using core::math::Pose3;
using core::math::Vec3;
using core::vision::PointCloud;

//=============================================================================
// SensorBase - Cooperative Scheduler Implementation
//=============================================================================

class SensorBase::Impl
{
public:
  explicit Impl(const SensorBase::Params& params);

  [[nodiscard]] const core::sensors::SensorBase& base_config() const
  {
    return std::visit(
      [](const auto& cfg) -> const core::sensors::SensorBase& { return cfg.base; }, *config_
    );
  }

  // Accessors
  [[nodiscard]] const std::shared_ptr<const core::sensors::SensorConfig>& config() const
  {
    return config_;
  }
  [[nodiscard]] std::shared_ptr<collision::TLAS2Pair>  scene_trees() const { return scene_trees_; }
  [[nodiscard]] std::shared_ptr<threading::ThreadPool> thread_pool() const { return thread_pool_; }
  [[nodiscard]] std::shared_ptr<core::tf::TfBuffer>    tf_buffer() const { return tf_buffer_; }
  [[nodiscard]] const std::string& robot_base_frame() const { return robot_base_frame_; }

  // Timing state accessors and mutators
  [[nodiscard]] double& time_since_last_update() { return time_since_last_update_; }
  [[nodiscard]] double  update_period() const { return update_period_; }
  [[nodiscard]] double& pending_data_time() { return pending_data_time_; }

  [[nodiscard]] bool has_data() const { return has_data_; }
  void               set_has_data(bool value) { has_data_ = value; }

  [[nodiscard]] bool has_pending_data() const { return has_pending_data_; }
  void               set_has_pending_data(bool value) { has_pending_data_ = value; }

  // Cloud accessors
  [[nodiscard]] std::shared_ptr<const core::vision::PointCloud> cached_cloud() const
  {
    return cached_cloud_;
  }
  void set_cached_cloud(std::shared_ptr<const core::vision::PointCloud> cloud)
  {
    cached_cloud_ = std::move(cloud);
  }

  [[nodiscard]] std::shared_ptr<core::vision::PointCloud> pending_cloud() const
  {
    return pending_cloud_;
  }
  void set_pending_cloud(std::shared_ptr<core::vision::PointCloud> cloud)
  {
    pending_cloud_ = std::move(cloud);
  }

private:
  std::shared_ptr<const core::sensors::SensorConfig> config_;
  std::shared_ptr<collision::TLAS2Pair>              scene_trees_;
  std::shared_ptr<threading::ThreadPool>             thread_pool_{nullptr};
  std::shared_ptr<core::tf::TfBuffer>                tf_buffer_{nullptr};
  std::string                                        robot_base_frame_;

  // Timing state
  double time_since_last_update_{0.0};
  double update_period_{0.0};      // 1.0 / update_hz
  double pending_data_time_{-1.0}; // time when pending data becomes available

  bool has_data_{false};
  bool has_pending_data_{false};

  // Cached data ready to be consumed (shared ownership for zero-copy)
  std::shared_ptr<const core::vision::PointCloud> cached_cloud_;

  // Pending data being prepared (simulating latency) - mutable for sensor updates
  std::shared_ptr<core::vision::PointCloud> pending_cloud_;
};

SensorBase::Impl::Impl(const SensorBase::Params& params)
    : config_(params.config),
      scene_trees_(params.scene_trees),
      thread_pool_(params.thread_pool),
      tf_buffer_(params.tf_buffer),
      robot_base_frame_(params.robot_base_frame)
{
  update_period_          = 1.0 / static_cast<double>(base_config().update_hz);
  time_since_last_update_ = 0.0;
}

SensorBase::SensorBase(const Params& params) : impl_(std::make_unique<Impl>(params))
{
}

SensorBase::~SensorBase() = default;

SensorBase::SensorBase(SensorBase&&) noexcept            = default;
SensorBase& SensorBase::operator=(SensorBase&&) noexcept = default;

bool SensorBase::should_update(double dt, double current_time)
{
  impl_->time_since_last_update() += dt;

  // Check if pending data is ready to be published (latency elapsed)
  if (impl_->has_pending_data() && current_time >= impl_->pending_data_time()) {
    // Publish pending data (zero-copy via shared_ptr)
    impl_->set_cached_cloud(impl_->pending_cloud());
    impl_->set_has_data(true);
    impl_->set_has_pending_data(false);
    impl_->set_pending_cloud(nullptr);
  }

  // Check if it's time to trigger a new sensor update
  if (impl_->time_since_last_update() >= impl_->update_period()) {
    impl_->time_since_last_update() = 0.0;
    return true;
  }

  return false;
}

double SensorBase::apply_noise(double range, std::mt19937& rng) const
{
  const auto& base = base_config();
  if (base.noise_std <= 0.0f) {
    return range;
  }

  std::normal_distribution<double> dist(0.0, static_cast<double>(base.noise_std));
  return range + dist(rng);
}

bool SensorBase::should_dropout(std::mt19937& rng) const
{
  const auto& base = base_config();
  if (base.dropout_prob <= 0.0f) {
    return false;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng) < base.dropout_prob;
}

std::string SensorBase::id() const
{
  return std::visit([](const auto& cfg) { return cfg.base.id; }, *impl_->config());
}

std::string SensorBase::frame_id() const
{
  return std::visit([](const auto& cfg) { return cfg.base.frame_id; }, *impl_->config());
}

bool SensorBase::has_data() const
{
  return impl_->has_data();
}

const core::sensors::SensorBase& SensorBase::base_config() const
{
  return impl_->base_config();
}

std::shared_ptr<collision::TLAS2Pair> SensorBase::get_scene_trees() const
{
  return impl_->scene_trees();
}

std::shared_ptr<threading::ThreadPool> SensorBase::get_thread_pool() const
{
  return impl_->thread_pool();
}

std::shared_ptr<core::tf::TfBuffer> SensorBase::get_tf_buffer() const
{
  return impl_->tf_buffer();
}

std::shared_ptr<const core::vision::PointCloud> SensorBase::get_cached_cloud() const
{
  return impl_->cached_cloud();
}

std::shared_ptr<core::vision::PointCloud> SensorBase::get_mutable_pending_cloud()
{
  // Create a new pending cloud if it doesn't exist
  if (!impl_->pending_cloud()) {
    impl_->set_pending_cloud(std::make_shared<core::vision::PointCloud>());
  }
  return impl_->pending_cloud();
}

void SensorBase::set_pending_data(double time)
{
  impl_->pending_data_time() = time;
  impl_->set_has_pending_data(true);
}

} // namespace ddrl::sim
