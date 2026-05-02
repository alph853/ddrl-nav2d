/**
 * @file vtk_camera.cpp
 * @brief Implementation of VTK camera management
 */

#include "visualizer/vtk_components/vtk_camera.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <cmath>

namespace ddrl::visualizer {

using core::math::Point3;
using core::math::Pose2;
using core::math::Vec3;

VtkCamera::VtkCamera(
  vtkRenderer* renderer, const config::CameraFollowSettings& config, double target_fps
)
    : renderer_(renderer),
      camera_(renderer != nullptr ? renderer->GetActiveCamera() : nullptr),
      config_(config),
      dt_(1.0 / target_fps),
      max_interp_distance_sq_(compute_snap_distance_sq())
{
}

void VtkCamera::update_follow(const Pose2& target_pose, CameraFollowMode mode)
{
  if (renderer_ == nullptr || camera_ == nullptr) {
    return;
  }

  switch (mode) {
    case CameraFollowMode::CHASE:
      update_chase_mode(target_pose);
      break;
    case CameraFollowMode::TOPDOWN:
      update_topdown_mode(target_pose);
      break;
    case CameraFollowMode::ORBIT:
      update_orbit_mode(target_pose);
      break;
    case CameraFollowMode::FIRSTPERSON:
      update_firstperson_mode(target_pose);
      break;
    default:
      return;
  }

  renderer_->ResetCameraClippingRange();
}

void VtkCamera::update_manual()
{
  if (renderer_ == nullptr || camera_ == nullptr) {
    return;
  }

  if (!state_.initialized) {
    state_.initialized = true;
  }

  // Reset zoom factor by applying inverse dolly
  camera_->Dolly(1.0 / state_.zoom_factor);
  state_.zoom_factor = 1.0;

  renderer_->ResetCameraClippingRange();
}

void VtkCamera::adjust_zoom(
  bool zoom_in, double sensitivity, double max_zoom_in, double max_zoom_out
)
{
  const double adjusted_factor = zoom_in ? (1.0 - 0.1 * sensitivity) : (1.0 + 0.1 * sensitivity);
  state_.zoom_factor *= adjusted_factor;

  // Clamp to configured range
  state_.zoom_factor = std::max(max_zoom_in, std::min(max_zoom_out, state_.zoom_factor));
}

void VtkCamera::reset_view()
{
  if (renderer_ == nullptr || camera_ == nullptr) {
    return;
  }

  camera_->SetFocalPoint(0, 0, 0);
  camera_->SetPosition(20, -30, 25);
  camera_->SetViewUp(0, 0, 1);
  renderer_->ResetCamera();
  renderer_->ResetCameraClippingRange();

  state_.zoom_factor = 1.0;
}

void VtkCamera::force_snap()
{
  force_snap_next_ = true;
}

void VtkCamera::update_chase_mode(const Pose2& target_pose)
{
  // Target forward vector from yaw
  const double c   = std::cos(target_pose.yaw);
  const double s   = std::sin(target_pose.yaw);
  Vec3         fwd = {c, s, 0.0};

  // Apply zoom to distances
  const double chase_back = config_.chase.back_distance * state_.zoom_factor;
  const double chase_up   = config_.chase.up_distance * state_.zoom_factor;

  // Camera behind and above the target
  Point3 desired_pos;
  desired_pos.x = target_pose.pos.x - chase_back * fwd.x;
  desired_pos.y = target_pose.pos.y - chase_back * fwd.y;
  desired_pos.z = chase_up;

  Point3 desired_focal;
  desired_focal.x = target_pose.pos.x;
  desired_focal.y = target_pose.pos.y;
  desired_focal.z = config_.chase.focal_height;

  update_smoothed_targets(desired_pos, desired_focal, config_.smooth_alpha);

  apply_camera_transform({0.0, 0.0, 1.0}); // Z-up
}

void VtkCamera::update_topdown_mode(const Pose2& target_pose)
{
  const double top_down_height = config_.top_down.height * state_.zoom_factor;

  Point3 desired_pos;
  desired_pos.x = target_pose.pos.x;
  desired_pos.y = target_pose.pos.y;
  desired_pos.z = top_down_height;

  Point3 desired_focal;
  desired_focal.x = target_pose.pos.x;
  desired_focal.y = target_pose.pos.y;
  desired_focal.z = 0.0;

  update_smoothed_targets(desired_pos, desired_focal, config_.smooth_alpha);

  apply_camera_transform({0.0, 1.0, 0.0}); // Y-up for top-down
}

void VtkCamera::update_orbit_mode(const Pose2& target_pose)
{
  const double orbit_radius = config_.orbit.radius * state_.zoom_factor;
  const double orbit_height = config_.orbit.height * state_.zoom_factor;

  // Update orbit angle
  state_.orbit_angle += config_.orbit.speed * dt_;

  Point3 desired_pos;
  desired_pos.x = target_pose.pos.x + orbit_radius * std::cos(state_.orbit_angle);
  desired_pos.y = target_pose.pos.y + orbit_radius * std::sin(state_.orbit_angle);
  desired_pos.z = orbit_height;

  Point3 desired_focal;
  desired_focal.x = target_pose.pos.x;
  desired_focal.y = target_pose.pos.y;
  desired_focal.z = config_.orbit.focal_height;

  update_smoothed_targets(desired_pos, desired_focal, config_.smooth_alpha);

  apply_camera_transform({0.0, 0.0, 1.0}); // Z-up
}

void VtkCamera::update_firstperson_mode(const Pose2& target_pose)
{
  // Target forward vector from yaw
  const double c   = std::cos(target_pose.yaw);
  const double s   = std::sin(target_pose.yaw);
  Vec3         fwd = {c, s, 0.0};

  // Camera at robot's "eye" position with slight forward offset
  Point3 desired_pos;
  desired_pos.x = target_pose.pos.x + config_.first_person.forward_offset * fwd.x;
  desired_pos.y = target_pose.pos.y + config_.first_person.forward_offset * fwd.y;
  desired_pos.z = config_.first_person.eye_height;

  // Look ahead in the direction robot is facing
  Point3 desired_focal;
  desired_focal.x = target_pose.pos.x + config_.first_person.focal_distance * fwd.x;
  desired_focal.y = target_pose.pos.y + config_.first_person.focal_distance * fwd.y;
  desired_focal.z = config_.first_person.eye_height; // Look straight ahead

  const double fp_smooth_alpha = config_.smooth_alpha * 0.5; // dampened smoothing for first-person
  update_smoothed_targets(desired_pos, desired_focal, fp_smooth_alpha);

  apply_camera_transform({0.0, 0.0, 1.0}); // Z-up
}

void VtkCamera::apply_camera_transform(const Vec3& view_up)
{
  if (camera_ == nullptr) {
    return;
  }

  camera_->SetPosition(state_.sm_pos.x, state_.sm_pos.y, state_.sm_pos.z);
  camera_->SetFocalPoint(state_.sm_focal.x, state_.sm_focal.y, state_.sm_focal.z);
  camera_->SetViewUp(view_up.x, view_up.y, view_up.z);
}

void VtkCamera::update_smoothed_targets(
  const Point3& desired_pos, const Point3& desired_focal, double alpha
)
{
  alpha = std::clamp(alpha, 0.0, 1.0);

  const bool exceeds_threshold = state_.initialized && max_interp_distance_sq_ > 0.0
                                 && core::math::distance_sq(state_.sm_pos, desired_pos)
                                      >= max_interp_distance_sq_;

  if (!state_.initialized || force_snap_next_ || exceeds_threshold) {
    state_.sm_pos      = desired_pos;
    state_.sm_focal    = desired_focal;
    state_.initialized = true;
    force_snap_next_   = false;
    return;
  }

  state_.sm_pos   = core::math::lerp(state_.sm_pos, desired_pos, alpha);
  state_.sm_focal = core::math::lerp(state_.sm_focal, desired_focal, alpha);
}

double VtkCamera::compute_snap_distance_sq() const
{
  constexpr double kMinSnapDistance = 50.0;
  const double chase_hw             = std::hypot(config_.chase.back_distance, config_.chase.up_distance);
  const double orbit_hw             = std::hypot(config_.orbit.radius, config_.orbit.height);
  const double top_hw               = config_.top_down.height;
  const double fp_hw                = config_.first_person.focal_distance;

  const double baseline  = std::max({chase_hw, orbit_hw, top_hw, fp_hw, kMinSnapDistance});
  const double snap_dist = std::max(baseline * 2.5, kMinSnapDistance);
  return snap_dist * snap_dist;
}

} // namespace ddrl::visualizer
