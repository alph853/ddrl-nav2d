/**
 * @file interactive_style.cpp
 * @brief Implementation of custom VTK interactor style
 */

#include "visualizer/vtk_components/default_style.hpp"

#include "core/base/enum_helper.hpp"
#include <algorithm>
#include <cctype>
#include <vtkCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRendererCollection.h>

namespace ddrl::visualizer {

vtkStandardNewMacro(InteractiveStyle);

void InteractiveStyle::register_viewport(vtkRenderer* renderer, uint32_t sim_id)
{
  viewports_[sim_id] = renderer;

  if (std::ranges::find(viewport_sim_ids_, sim_id) == viewport_sim_ids_.end()) {
    viewport_sim_ids_.push_back(sim_id);
    std::ranges::sort(viewport_sim_ids_);
  }
}

void InteractiveStyle::OnLeftButtonDown()
{
  if (view_mode_ == ViewMode::ALL_VIEWPORTS) {
    // In ALL_VIEWPORTS mode, clicking switches to FOCUSED mode with that viewport
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    vtkRenderer* clicked_renderer = find_clicked_renderer(x, y);
    if (clicked_renderer != nullptr) {
      // Find which sim_id was clicked
      for (const auto& [sim_id, renderer] : viewports_) {
        if (renderer == clicked_renderer) {
          // Switch to FOCUSED mode with this viewport
          view_mode_      = ViewMode::FOCUSED;
          focused_sim_id_ = sim_id;

          // Notify callbacks
          if (view_all_cb_) {
            view_all_cb_();
          }
          if (viewport_focus_cb_) {
            viewport_focus_cb_();
          }
          this->Interactor->Render();
          break;
        }
      }
    }
    return;
  }

  // FOCUSED mode: allow pan interaction
  if (view_mode_ == ViewMode::FOCUSED) {
    if (cam_follow_mode_ != CameraFollowMode::MANUAL) {
      return;
    }

    vtkRenderer* focused_renderer = nullptr;
    auto         it               = viewports_.find(focused_sim_id_);
    if (it != viewports_.end()) {
      focused_renderer = it->second;
    }

    if (focused_renderer != nullptr) {
      this->SetCurrentRenderer(focused_renderer);
      this->StartPan();
    }
  }
}

void InteractiveStyle::OnLeftButtonUp()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    if (cam_follow_mode_ != CameraFollowMode::MANUAL) {
      return;
    }
    this->EndPan();
    if (this->Interactor != nullptr) {
      this->Interactor->Render();
    }
  }
}

void InteractiveStyle::OnRightButtonDown()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    if (cam_follow_mode_ != CameraFollowMode::MANUAL) {
      return;
    }
    vtkRenderer* focused_renderer = nullptr;
    auto         it               = viewports_.find(focused_sim_id_);
    if (it != viewports_.end()) {
      focused_renderer = it->second;
    }

    if (focused_renderer != nullptr) {
      this->SetCurrentRenderer(focused_renderer);
      this->StartRotate();
    }
  }
}

void InteractiveStyle::OnRightButtonUp()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    if (cam_follow_mode_ != CameraFollowMode::MANUAL) {
      return;
    }
    this->EndRotate();
    if (this->Interactor != nullptr) {
      this->Interactor->Render();
    }
  }
}

void InteractiveStyle::OnMiddleButtonDown()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    vtkRenderer* focused_renderer = nullptr;
    auto         it               = viewports_.find(focused_sim_id_);
    if (it != viewports_.end()) {
      focused_renderer = it->second;
    }

    if (focused_renderer != nullptr) {
      this->SetCurrentRenderer(focused_renderer);
      this->StartDolly();
    }
  }
}

void InteractiveStyle::OnMiddleButtonUp()
{
  if (view_mode_ != ViewMode::FOCUSED) {
    return;
  }

  this->EndDolly();
  if (this->Interactor != nullptr) {
    this->Interactor->Render();
  }
}

void InteractiveStyle::OnMouseMove()
{
  // Only allow camera control in FOCUSED mode
  if (view_mode_ != ViewMode::FOCUSED) {
    return;
  }

  vtkInteractorStyleTrackballCamera::OnMouseMove();
}

void InteractiveStyle::OnMouseWheelForward()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    vtkRenderer* focused_renderer = nullptr;
    auto         it               = viewports_.find(focused_sim_id_);
    if (it != viewports_.end()) {
      focused_renderer = it->second;
    }

    if (focused_renderer != nullptr) {
      this->SetCurrentRenderer(focused_renderer);

      if (zoom_adjust_cb_) {
        zoom_adjust_cb_(true);
        if (this->Interactor != nullptr) {
          this->Interactor->Render();
        }
      } else {
        vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
      }
    }
  }
}

void InteractiveStyle::OnMouseWheelBackward()
{
  if (view_mode_ == ViewMode::FOCUSED) {
    vtkRenderer* focused_renderer = nullptr;
    auto         it               = viewports_.find(focused_sim_id_);
    if (it != viewports_.end()) {
      focused_renderer = it->second;
    }

    if (focused_renderer != nullptr) {
      this->SetCurrentRenderer(focused_renderer);

      if (zoom_adjust_cb_) {
        zoom_adjust_cb_(false);
        if (this->Interactor != nullptr) {
          this->Interactor->Render();
        }
      } else {
        vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
      }
    }
  }
}

void InteractiveStyle::OnKeyPress()
{
  if (this->Interactor == nullptr) {
    return;
  }

  std::string key   = this->Interactor->GetKeySym();
  bool        ctrl  = this->Interactor->GetControlKey() != 0;
  bool        shift = this->Interactor->GetShiftKey() != 0;
  bool        alt   = this->Interactor->GetAltKey() != 0;

  // H key - Toggle HUD
  if (!ctrl && !shift && !alt && (key == "h" || key == "H")) {
    if (hud_toggle_cb_) {
      hud_toggle_cb_();
    }
    this->Interactor->Render();
    return;
  }

  if (!ctrl && !shift && !alt && (key == "t" || key == "T")) {
    if (reward_terms_toggle_cb_) {
      reward_terms_toggle_cb_();
    }
    this->Interactor->Render();
    return;
  }

  if (!ctrl && !shift && !alt && (key == "p" || key == "P")) {
    if (route_visualization_toggle_cb_) {
      route_visualization_toggle_cb_();
    }
    if (this->Interactor != nullptr) {
      this->Interactor->Render();
    }
    return;
  }

  if (!ctrl && !shift && !alt && (key == "c" || key == "C")) {
    if (pointcloud_toggle_cb_) {
      pointcloud_toggle_cb_();
    }
    if (this->Interactor != nullptr) {
      this->Interactor->Render();
    }
    return;
  }

  if (!ctrl && !shift && !alt && (key == "s" || key == "S")) {
    if (sensor_frustum_toggle_cb_) {
      sensor_frustum_toggle_cb_();
    }
    if (this->Interactor != nullptr) {
      this->Interactor->Render();
    }
    return;
  }

  if (ctrl && !shift && !alt && (key == "f" || key == "F")) {
    cam_follow_mode_ = core::next_enum(cam_follow_mode_, CameraFollowMode::COUNT);
    if (cam_follow_mode_ == CameraFollowMode::MANUAL && view_mode_ == ViewMode::ALL_VIEWPORTS) {
      cam_follow_mode_ = CameraFollowMode::CHASE;
    }
    this->Interactor->Render();
    return;
  }

  // FOCUSED mode only shortcuts
  if (view_mode_ == ViewMode::FOCUSED) {
    // Ctrl+R - Reset camera view
    if (ctrl && !shift && !alt && (key == "r" || key == "R")) {
      cam_follow_mode_ = CameraFollowMode::MANUAL;
      if (cam_reset_view_cb_) {
        cam_reset_view_cb_();
      }
      this->Interactor->Render();
      return;
    }

    // Ctrl+A - View all viewports
    if (ctrl && !shift && !alt && (key == "a" || key == "A")) {
      if (viewport_sim_ids_.size() <= 1) {
        return;
      }
      view_mode_       = ViewMode::ALL_VIEWPORTS;
      cam_follow_mode_ = CameraFollowMode::CHASE;

      if (view_all_cb_) {
        view_all_cb_();
      }

      this->Interactor->Render();
      return;
    }

    // Ctrl+Tab - Next viewport
    if (ctrl && !shift && !alt && key == "Tab") {
      if (!viewport_sim_ids_.empty() && viewport_focus_cb_) {
        // Find current focused viewport index
        auto it = std::ranges::find(viewport_sim_ids_, focused_sim_id_);
        if (it != viewport_sim_ids_.end()) {
          // Move to next viewport (wrap around)
          auto current_idx = std::distance(viewport_sim_ids_.begin(), it);
          auto next_idx    = (static_cast<size_t>(current_idx) + 1) % viewport_sim_ids_.size();
          focused_sim_id_  = viewport_sim_ids_[next_idx];
          viewport_focus_cb_();
        }
      }
      this->Interactor->Render();
      return;
    }

    // Ctrl+Shift+Tab - Previous viewport
    if (ctrl && shift && !alt && key == "Tab") {
      if (!viewport_sim_ids_.empty() && viewport_focus_cb_) {
        // Find current focused viewport index
        auto it = std::ranges::find(viewport_sim_ids_, focused_sim_id_);
        if (it != viewport_sim_ids_.end()) {
          // Move to previous viewport (wrap around)
          auto current_idx = std::distance(viewport_sim_ids_.begin(), it);
          auto prev_idx    = (current_idx == 0) ? (viewport_sim_ids_.size() - 1)
                                                : (static_cast<size_t>(current_idx - 1));
          focused_sim_id_  = viewport_sim_ids_[prev_idx];
          viewport_focus_cb_();
        }
      }
      this->Interactor->Render();
      return;
    }
  }

  // Forward other keys to base class
  vtkInteractorStyleTrackballCamera::OnKeyPress();
}

void InteractiveStyle::handle_cam_reset_view()
{
  vtkRenderer* focused_renderer = nullptr;

  auto it = viewports_.find(focused_sim_id_);
  if (it != viewports_.end()) {
    focused_renderer = it->second;
  }

  if (focused_renderer != nullptr) {
    auto* camera = focused_renderer->GetActiveCamera();
    camera->SetFocalPoint(0, 0, 0);
    camera->SetPosition(20, -30, 25);
    camera->SetViewUp(0, 0, 1);
    focused_renderer->ResetCamera();
    focused_renderer->ResetCameraClippingRange();
  }
}

vtkRenderer* InteractiveStyle::find_clicked_renderer(int x, int y)
{
  if (this->Interactor == nullptr) {
    return nullptr;
  }

  vtkRenderWindow* render_window = this->Interactor->GetRenderWindow();
  if (render_window == nullptr) {
    return nullptr;
  }

  // Get window size
  int* window_size = render_window->GetSize();
  int  width       = window_size[0];
  int  height      = window_size[1];

  // Normalize coordinates (0-1)
  double norm_x = static_cast<double>(x) / width;
  double norm_y = static_cast<double>(y) / height;

  // Check each registered viewport
  for (const auto& [_, ren] : viewports_) {
    double* vp   = ren->GetViewport();
    double  xmin = vp[0];
    double  ymin = vp[1];
    double  xmax = vp[2];
    double  ymax = vp[3];

    if (norm_x >= xmin && norm_x <= xmax && norm_y >= ymin && norm_y <= ymax) {
      return ren;
    }
  }

  return nullptr;
}

bool InteractiveStyle::is_focused_renderer(vtkRenderer* renderer)
{
  if (renderer == nullptr) {
    return false;
  }

  return std::ranges::any_of(viewports_, [&](const auto& pair) {
    return pair.second == renderer && pair.first == focused_sim_id_;
  });

  return false;
}

} // namespace ddrl::visualizer
