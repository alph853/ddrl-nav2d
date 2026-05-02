/**
 * @file vtk_camera.hpp
 * @brief VTK camera management for visualizer viewports
 *
 * @internal This is an internal header used by Visualizer implementation.
 * It is not part of the public API and should not be included directly by
 * users.
 */

#pragma once

#include "core/math/pose.hpp"
#include "core/math/primitives.hpp"
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include "config/visualizer.hpp"
#include "visualizer/models.hpp"

namespace ddrl::visualizer {

/**
 * @brief Manages camera state and updates for a single viewport
 *
 * Encapsulates all camera-related logic including:
 * - Follow modes (chase, top-down, orbit, first-person)
 * - Manual camera control
 * - Zoom adjustments
 * - Smooth camera transitions
 */
class VtkCamera
{
public:
  /**
   * @brief Camera state for follow modes
   */
  struct FollowState {
    bool               initialized{false};
    core::math::Point3 sm_pos;           ///< Smoothed camera position
    core::math::Point3 sm_focal;         ///< Smoothed focal point
    double             orbit_angle{0.0}; ///< Current orbit angle (radians)
    double             zoom_factor{1.0}; ///< Current zoom factor
  };

  /**
   * @brief Construct camera manager for a viewport
   * @param renderer VTK renderer for the viewport
   * @param config Camera follow configuration
   * @param target_fps Target frame rate for time-based updates
   */
  VtkCamera(vtkRenderer* renderer, const config::CameraFollowSettings& config, double target_fps);

  /**
   * @brief Update camera for follow mode
   * @param target_pose Pose to follow
   * @param mode Follow mode to use
   */
  void update_follow(const core::math::Pose2& target_pose, CameraFollowMode mode);

  /**
   * @brief Update camera for manual mode
   */
  void update_manual();

  /**
   * @brief Adjust zoom level
   * @param zoom_in True to zoom in, false to zoom out
   * @param sensitivity Zoom sensitivity multiplier
   * @param max_zoom_in Maximum zoom in factor
   * @param max_zoom_out Maximum zoom out factor
   */
  void adjust_zoom(bool zoom_in, double sensitivity, double max_zoom_in, double max_zoom_out);

  /**
   * @brief Reset camera to default view
   */
  void reset_view();

  /**
   * @brief Force the next update to snap directly to the desired pose
   */
  void force_snap();

  /**
   * @brief Get current follow state
   */
  [[nodiscard]] const FollowState& get_state() const { return state_; }

  /**
   * @brief Get current zoom factor
   */
  [[nodiscard]] double get_zoom_factor() const { return state_.zoom_factor; }

private:
  vtkRenderer*                       renderer_;
  vtkCamera*                         camera_;
  const config::CameraFollowSettings config_;
  double                             dt_; ///< Delta time per frame
  FollowState                        state_;
  bool                               force_snap_next_{false};
  double                             max_interp_distance_sq_{0.0};

  /**
   * @brief Update camera for chase mode
   */
  void update_chase_mode(const core::math::Pose2& target_pose);

  /**
   * @brief Update camera for top-down mode
   */
  void update_topdown_mode(const core::math::Pose2& target_pose);

  /**
   * @brief Update camera for orbit mode
   */
  void update_orbit_mode(const core::math::Pose2& target_pose);

  /**
   * @brief Update camera for first-person mode
   */
  void update_firstperson_mode(const core::math::Pose2& target_pose);

  /**
   * @brief Apply smoothed position and focal point to VTK camera
   */
  void apply_camera_transform(const core::math::Vec3& view_up);

  /**
   * @brief Update smoothed targets, snapping when jumps exceed the threshold
   */
  void update_smoothed_targets(
    const core::math::Point3& desired_pos,
    const core::math::Point3& desired_focal,
    double                    alpha
  );

  [[nodiscard]] double compute_snap_distance_sq() const;
};

} // namespace ddrl::visualizer
