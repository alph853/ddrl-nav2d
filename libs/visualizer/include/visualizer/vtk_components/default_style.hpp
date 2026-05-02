/**
 * @file interactive_style.hpp
 * @brief Custom VTK interactor style for multi-viewport interaction
 *
 * @internal This is an internal header used by VtkRenderer implementation.
 * It is not part of the public API and should not be included directly by
 * users.
 */

#pragma once

#include <functional>
#include <unordered_map>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include "config/visualizer.hpp"
#include "visualizer/models.hpp"

namespace ddrl::visualizer {

using ViewAllCallbackType       = std::function<void()>;
using ViewportFocusCallbackType = std::function<void()>;
using HUDToggleCallbackType     = std::function<void()>;
using RewardTermsToggleCallbackType = std::function<void()>;
// true == zoom in, false == zoom out
using ZoomAdjustCallbackType          = std::function<void(bool)>;
using CamResetViewCallbackType        = std::function<void()>;
using ToggleRouteVisualizationCallbackType = std::function<void()>;
using TogglePointCloudCallbackType    = std::function<void()>;
using ToggleSensorFrustumCallbackType = std::function<void()>;

/**
 * @brief Custom interactor style with two view modes
 */
class InteractiveStyle : public vtkInteractorStyleTrackballCamera
{
public:
  static InteractiveStyle* New();
  vtkTypeMacro(InteractiveStyle, vtkInteractorStyleTrackballCamera)

    void set_view_all_callback(ViewAllCallbackType callback)
  {
    view_all_cb_ = std::move(callback);
  }

  void set_viewport_focus_callback(ViewportFocusCallbackType callback)
  {
    viewport_focus_cb_ = std::move(callback);
  }

  void set_hud_toggle_callback(HUDToggleCallbackType callback)
  {
    hud_toggle_cb_ = std::move(callback);
  }

  void set_reward_terms_toggle_callback(RewardTermsToggleCallbackType callback)
  {
    reward_terms_toggle_cb_ = std::move(callback);
  }

  void set_zoom_adjust_callback(ZoomAdjustCallbackType callback)
  {
    zoom_adjust_cb_ = std::move(callback);
  }

  void set_cam_reset_view_callback(CamResetViewCallbackType callback)
  {
    cam_reset_view_cb_ = std::move(callback);
  }

  void set_pointcloud_toggle_callback(TogglePointCloudCallbackType callback)
  {
    pointcloud_toggle_cb_ = std::move(callback);
  }

  void set_route_visualization_toggle_callback(ToggleRouteVisualizationCallbackType callback)
  {
    route_visualization_toggle_cb_ = std::move(callback);
  }

  void set_sensor_frustum_toggle_callback(ToggleSensorFrustumCallbackType callback)
  {
    sensor_frustum_toggle_cb_ = std::move(callback);
  }

  void set_style_config(const config::InteractorStyleCfg& cfg) { style_config_ = cfg; }

  void register_viewport(vtkRenderer* renderer, uint32_t sim_id);
  void clear_viewports() { viewports_.clear(); }
  void set_focused_viewport(uint32_t sim_id) { focused_sim_id_ = sim_id; }
  [[nodiscard]] uint32_t get_focused_viewport() const { return focused_sim_id_; }

  /**
   * @brief Get current view mode
   */
  [[nodiscard]] ViewMode get_view_mode() const { return view_mode_; }

  /**
   * @brief Get current camera follow mode
   */
  [[nodiscard]] CameraFollowMode get_camera_follow_mode() const { return cam_follow_mode_; }

  /**
   * @brief Set view mode
   */
  void set_view_mode(ViewMode mode) { view_mode_ = mode; }

  /**
   * @brief Handle left mouse button press
   */
  void OnLeftButtonDown() override;

  /**
   * @brief Handle left mouse button release
   */
  void OnLeftButtonUp() override;

  /**
   * @brief Handle right mouse button press
   */
  void OnRightButtonDown() override;

  /**
   * @brief Handle right mouse button release
   */
  void OnRightButtonUp() override;

  /**
   * @brief Handle middle mouse button press
   */
  void OnMiddleButtonDown() override;

  /**
   * @brief Handle middle mouse button release
   */
  void OnMiddleButtonUp() override;

  /**
   * @brief Handle mouse move
   */
  void OnMouseMove() override;

  /**
   * @brief Handle mouse wheel forward (zoom in)
   */
  void OnMouseWheelForward() override;

  /**
   * @brief Handle mouse wheel backward (zoom out)
   */
  void OnMouseWheelBackward() override;

  /**
   * @brief Handle key press events
   */
  void OnKeyPress() override;

  /**
   * @brief Reset camera to default view
   */
  void handle_cam_reset_view();

protected:
  InteractiveStyle()           = default;
  ~InteractiveStyle() override = default;

private:
  std::unordered_map<uint32_t, vtkRenderer*> viewports_;
  std::vector<uint32_t>                      viewport_sim_ids_; // Ordered list for navigation

  ViewAllCallbackType       view_all_cb_;
  CamResetViewCallbackType  cam_reset_view_cb_;
  ViewportFocusCallbackType viewport_focus_cb_;
  HUDToggleCallbackType     hud_toggle_cb_;
  RewardTermsToggleCallbackType reward_terms_toggle_cb_;
  ZoomAdjustCallbackType    zoom_adjust_cb_;
  ToggleRouteVisualizationCallbackType route_visualization_toggle_cb_;
  TogglePointCloudCallbackType    pointcloud_toggle_cb_;
  ToggleSensorFrustumCallbackType sensor_frustum_toggle_cb_;

  uint32_t                   focused_sim_id_{0};
  ViewMode                   view_mode_{ViewMode::FOCUSED};
  CameraFollowMode           cam_follow_mode_{CameraFollowMode::MANUAL};
  config::InteractorStyleCfg style_config_; // Interactive style configuration

  /**
   * @brief Find which viewport was clicked
   * @param x Mouse x coordinate
   * @param y Mouse y coordinate
   * @return Renderer that was clicked, or nullptr
   */
  vtkRenderer* find_clicked_renderer(int x, int y);

  /**
   * @brief Check if renderer is the focused one
   */
  bool is_focused_renderer(vtkRenderer* renderer);
};

} // namespace ddrl::visualizer
