/**
 * @file visualizer.cpp
 * @brief Implementation of visualizer
 */

#include "visualizer/visualizer.hpp"

#include "core/math/numeric.hpp"
#include "core/math/primitives.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <ranges>
#include <unordered_set>
#include <utility>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "logging/logging.hpp"
#include "visualizer/map_geometry_manager.hpp"
#include "visualizer/profile_manager.hpp"
#include "visualizer/vtk_components/default_style.hpp"
#include "visualizer/vtk_components/vtk_camera.hpp"
#include "visualizer/widgets/hud.hpp"
#include "visualizer/widgets/viewport.hpp"

namespace ddrl::visualizer {

using core::math::Point3;
using core::math::Pose2;
using core::math::Vec3;
using core::sim::WorldStateSnapshot;
using widgets::HUDWidget;

// ============================================================================
// Visualizer::Impl definition
// ============================================================================

class Visualizer::Impl
{
public:
  explicit Impl(Visualizer::Params&& params);
  void init();
  void setup_render_window();
  void configure_viewports();
  void setup_initial_ui();

  void init_profile_manager();

  struct ViewportLayout {
    uint32_t              sim_id{0};
    std::array<double, 4> viewport_rect{0.0, 0.0, 1.0, 1.0};
  };

  [[nodiscard]] std::vector<ViewportLayout> compute_viewport_layout() const;
  void                                      handle_view_all();
  void                                      handle_viewport_focus();
  void                                      update_viewport_layout();

  void setup_interactive_style();
  void update_interactive_style_viewports();
  void setup_interactor();

  void setup_window_resize_handler();
  void handle_window_resize_event(vtkObject* caller);

  void setup_viz_loop();
  void handle_viz_loop(vtkObject* caller);

  void handle_cam_reset_view();
  void toggle_route_visualization();
  void handle_zoom_adjust(bool zoom_in);
  void toggle_pointclouds();
  void toggle_sensor_frustums();

  [[nodiscard]] double compute_camera_snap_distance_sq() const;
  bool                 should_snap_camera(uint32_t sim_id, const Pose2& pose);

  void start_interactive();
  void stop_interactive();
  void notify_shutdown_requested();

  void update_rl_info(uint32_t sim_id, const core::rl::RLMetadata& metadata);
  void update_observation_debug_cloud(
    uint32_t sim_id, std::shared_ptr<const core::vision::PointCloud> cloud
  );

  // Static helper for viewport access
  static ::vtkRenderer* get_viewport_renderer(Viewport& vp) { return vp.renderer_for_friend(); }

private:
  std::shared_ptr<logging::Logger>                      logger_ = logging::get_logger("visualizer");
  std::shared_ptr<const config::VizConfig>              config_;
  std::shared_ptr<IVisualizerProvider>                  provider_;
  std::vector<uint32_t>                                 sim_ids_;
  std::shared_ptr<const config::WorldConfig>            world_config_;
  std::shared_ptr<const config::RobotProfile>           robot_profile_;
  std::vector<std::shared_ptr<const config::MapConfig>> maps_;

  std::shared_ptr<const ProfileManager>     profile_manager_;
  std::shared_ptr<const MapGeometryManager> map_geometry_manager_;

  std::unordered_map<uint32_t, std::unique_ptr<Viewport>>  viewports_;
  std::unordered_map<uint32_t, std::unique_ptr<VtkCamera>> cameras_;
  std::unordered_map<uint32_t, Point3>                     last_robot_positions_;
  std::unordered_map<uint32_t, std::shared_ptr<const core::vision::PointCloud>>
    observation_debug_clouds_;
  std::mutex observation_debug_clouds_mutex_;

  vtkSmartPointer<vtkRenderWindow>           render_window_;
  vtkSmartPointer<vtkRenderWindowInteractor> interactor_;
  vtkSmartPointer<InteractiveStyle>          interactive_style_;

  std::unique_ptr<HUDWidget>   hud_widget_;
  vtkSmartPointer<vtkRenderer> hud_renderer_;
  vtkSmartPointer<vtkRenderer> bg_renderer_ = vtkSmartPointer<vtkRenderer>::New();

  std::chrono::steady_clock::time_point start_time_;
  std::function<void()>                 on_shutdown_;

  int    viz_timer_id_;
  bool   show_grid_{true};
  bool   show_pointcloud_{true};
  bool   show_sensor_frustums_{true};
  double camera_snap_distance_sq_{0.0};
};

// ============================================================================
// Visualizer::Impl Implementation
// ============================================================================

Visualizer::Impl::Impl(Visualizer::Params&& params)
    : config_(std::move(params.config)),
      provider_(std::move(params.provider)),
      sim_ids_(std::move(params.sim_ids)),
      world_config_(std::move(params.world_config)),
      robot_profile_(std::move(params.robot_profile)),
      maps_(std::move(params.maps)),
      profile_manager_(std::make_shared<ProfileManager>(*world_config_)),
      map_geometry_manager_(std::make_shared<MapGeometryManager>(maps_, world_config_)),
      start_time_(std::chrono::steady_clock::now()),
      on_shutdown_(std::move(params.on_shutdown)),
      camera_snap_distance_sq_(compute_camera_snap_distance_sq())
{
}

void Visualizer::Impl::init()
{
  // Initialize VTK components - must be called on the rendering thread
  setup_render_window();
  setup_interactor();
  setup_interactive_style();

  configure_viewports();
  setup_initial_ui();
}

void Visualizer::Impl::setup_render_window()
{
  render_window_ = vtkSmartPointer<vtkRenderWindow>::New();

  render_window_->SetNumberOfLayers(2); // base viewports (0) + HUD overlay (1)
  render_window_->SetOffScreenRendering(0);
  render_window_->SetWindowName(config_->window_title.c_str());
  render_window_->SetSize(
    static_cast<int>(config_->window_width), static_cast<int>(config_->window_height)
  );
  if (render_window_ != nullptr) {
    render_window_->Render();

    auto close_cb = vtkSmartPointer<vtkCallbackCommand>::New();
    close_cb->SetClientData(this);
    close_cb->SetCallback([](vtkObject*, unsigned long, void* client_data, void*) {
      auto* self = static_cast<Visualizer::Impl*>(client_data);
      if (self != nullptr) {
        self->notify_shutdown_requested();
      }
    });
    render_window_->AddObserver(vtkCommand::ExitEvent, close_cb);
  }
}

void Visualizer::Impl::setup_interactor()
{
  interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor_->SetRenderWindow(render_window_);
  interactor_->Initialize();

  auto exit_cb = vtkSmartPointer<vtkCallbackCommand>::New();
  exit_cb->SetClientData(this);
  exit_cb->SetCallback([](vtkObject*, unsigned long, void* client_data, void*) {
    auto* self = static_cast<Visualizer::Impl*>(client_data);
    if (self != nullptr) {
      self->notify_shutdown_requested();
    }
  });
  interactor_->AddObserver(vtkCommand::ExitEvent, exit_cb);
}

void Visualizer::Impl::setup_interactive_style()
{
  // Create or update interactive style
  if (interactive_style_ == nullptr) {
    interactive_style_ = vtkSmartPointer<InteractiveStyle>::New();
    interactor_->SetInteractorStyle(interactive_style_);

    if (config_) {
      interactive_style_->set_style_config(config_->interactor_style);
      interactive_style_->SetMotionFactor(config_->interactor_style.sensitivity);
    }
    interactive_style_->set_view_all_callback([this]() { handle_view_all(); });
    interactive_style_->set_hud_toggle_callback([this]() {
      if (hud_widget_) {
        hud_widget_->toggle_visibility();
      }
    });
    interactive_style_->set_reward_terms_toggle_callback([this]() {
      if (hud_widget_) {
        hud_widget_->toggle_reward_terms_visibility();
      }
    });
    interactive_style_->set_viewport_focus_callback([this]() { handle_viewport_focus(); });
    interactive_style_->set_zoom_adjust_callback([this](bool zoom_in) {
      handle_zoom_adjust(zoom_in);
    });
    interactive_style_->set_cam_reset_view_callback([this]() { this->handle_cam_reset_view(); });
    interactive_style_->set_route_visualization_toggle_callback(
      [this]() { toggle_route_visualization(); }
    );
    interactive_style_->set_pointcloud_toggle_callback([this]() { toggle_pointclouds(); });
    interactive_style_->set_sensor_frustum_toggle_callback([this]() { toggle_sensor_frustums(); });

    interactive_style_->set_focused_viewport(sim_ids_[0]);
  }

  update_interactive_style_viewports();
}

void Visualizer::Impl::update_interactive_style_viewports()
{
  interactive_style_->clear_viewports();

  for (const auto& [sim_id, viewport] : viewports_) {
    interactive_style_->register_viewport(Impl::get_viewport_renderer(*viewport), sim_id);
  }
  uint32_t current_focused = interactive_style_->get_focused_viewport();
  if (std::ranges::find(sim_ids_, current_focused) == sim_ids_.end()) {
    current_focused = sim_ids_[0];
  }
  interactive_style_->set_focused_viewport(current_focused);

  auto it = viewports_.find(current_focused);
  if (it != viewports_.end()) {
    interactive_style_->SetDefaultRenderer(Impl::get_viewport_renderer(*it->second));
  }
}

void Visualizer::Impl::configure_viewports()
{
  // Add background renderer first (layer 0)
  bg_renderer_->SetLayer(0);
  render_window_->AddRenderer(bg_renderer_);

  // Create and add all viewport renderers to the window once
  for (uint32_t sim_id : sim_ids_) {
    Viewport::Params params{};
    params.sim_id               = sim_id;
    params.viewport_rect        = {0.0, 0.0, 1.0, 1.0};
    params.show_grid            = show_grid_;
    params.show_pointcloud      = show_pointcloud_;
    params.show_sensor_frustums = show_sensor_frustums_;
    params.profile_manager      = profile_manager_;
    params.map_geometry_manager = map_geometry_manager_;
    params.world_config         = world_config_;
    params.robot_profile        = robot_profile_;

    auto viewport = std::make_unique<Viewport>(std::move(params));

    // Add this viewport's renderer to the render window once
    auto* renderer = Impl::get_viewport_renderer(*viewport);
    renderer->SetLayer(0);
    render_window_->AddRenderer(renderer);

    viewports_[sim_id] = std::move(viewport);

    // Create camera manager for this viewport
    cameras_[sim_id] =
      std::make_unique<VtkCamera>(renderer, config_->camera_follow, config_->target_fps);
  }

  update_viewport_layout();
}

void Visualizer::Impl::setup_initial_ui()
{
  hud_widget_ = std::make_unique<HUDWidget>(sim_ids_.size());
  hud_widget_->update_sim_id(sim_ids_[0]);
  hud_widget_->update_view_mode(ViewMode::FOCUSED);

  if (hud_renderer_ == nullptr) {
    hud_renderer_ = vtkSmartPointer<vtkRenderer>::New();
    hud_renderer_->InteractiveOff();
    hud_renderer_->SetLayer(1);
    hud_renderer_->SetViewport(0.0, 0.0, 1.0, 1.0);
    hud_renderer_->SetBackground(0.0, 0.0, 0.0);
    hud_renderer_->SetBackgroundAlpha(0.0);
    render_window_->AddRenderer(hud_renderer_);
  }
  hud_widget_->add_to_renderer(hud_renderer_);

  bg_renderer_->SetViewport(0.0, 0.0, 1.0, 1.0);
  bg_renderer_->EraseOn();
  bg_renderer_->SetBackground(0.1, 0.1, 0.15); // or your UI bg color
  bg_renderer_->SetBackgroundAlpha(1.0);       // opaque so it actually clears
}

void Visualizer::Impl::setup_viz_loop()
{
  if (interactor_ == nullptr || !provider_) {
    return;
  }

  auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
  cb->SetClientData(this);
  cb->SetCallback(
    [](vtkObject* caller, unsigned long event_id, void* client_data, void* call_data) {
      (void)call_data;
      (void)event_id;
      auto* self = static_cast<Visualizer::Impl*>(client_data);
      self->handle_viz_loop(caller);
    }
  );

  interactor_->AddObserver(vtkCommand::TimerEvent, cb);

  const double fps            = std::max(1.0, config_->target_fps);
  const int timer_interval_ms = static_cast<int>(std::round(1000.0 / std::clamp(fps, 1.0, 240.0)));
  DDRL_LOG_INFO(
    logger_,
    "Starting visualization loop with target FPS={} (interval={} ms)",
    config_->target_fps,
    timer_interval_ms
  );
  viz_timer_id_ = interactor_->CreateRepeatingTimer(static_cast<unsigned long>(timer_interval_ms));
}

void Visualizer::Impl::handle_viz_loop(vtkObject* caller)
{
  for (auto& [sim_id, viewport_ptr] : viewports_) {
    if (!viewport_ptr) {
      continue;
    }
    auto state_result = provider_->get_world_state(sim_id);
    if (!state_result) {
      DDRL_LOG_WARN_THROTTLE(
        logger_,
        5000,
        "No world state available for sim_id {}: {}",
        sim_id,
        to_string(state_result.error())
      );
      continue;
    }
    auto state = std::move(*state_result);
    viewport_ptr->update(state);
    {
      std::lock_guard lock(observation_debug_clouds_mutex_);
      auto cloud_it = observation_debug_clouds_.find(sim_id);
      if (cloud_it != observation_debug_clouds_.end()) {
        viewport_ptr->set_observation_debug_cloud(cloud_it->second);
      } else {
        viewport_ptr->set_observation_debug_cloud(nullptr);
      }
    }

    auto camera_mode = interactive_style_->get_camera_follow_mode();
    auto camera_it   = cameras_.find(sim_id);
    if (camera_it != cameras_.end() && camera_it->second) {
      const bool force_snap = should_snap_camera(sim_id, state.robot.pose);
      if (force_snap) {
        camera_it->second->force_snap();
      }
      if (camera_mode == CameraFollowMode::MANUAL) {
        camera_it->second->update_manual();
      } else {
        camera_it->second->update_follow(state.robot.pose, camera_mode);
      }
    }
  }

  if (hud_widget_) {
    auto                          now     = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - start_time_;
    hud_widget_->update(elapsed.count());
  }

  if (auto* iren = vtkRenderWindowInteractor::SafeDownCast(caller)) {
    iren->GetRenderWindow()->Render();
  }
}

void Visualizer::Impl::setup_window_resize_handler()
{
  auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
  cb->SetClientData(this);
  cb->SetCallback([](vtkObject* caller, unsigned long event_id, void* client_data, void*) {
    (void)event_id;
    auto* self = static_cast<Visualizer::Impl*>(client_data);
    if (self && self->hud_widget_) {
      self->handle_window_resize_event(caller);
    }
  });
  render_window_->AddObserver(vtkCommand::WindowResizeEvent, cb);
}

void Visualizer::Impl::handle_window_resize_event(vtkObject* caller)
{
  auto* render_win = vtkRenderWindow::SafeDownCast(caller);
  if (render_win == nullptr) {
    return;
  }

  int* size = render_win->GetSize();
  hud_widget_->update_window_size(size[0], size[1]);
}

void Visualizer::Impl::update_rl_info(uint32_t sim_id, const core::rl::RLMetadata& metadata)
{
  if (hud_widget_) {
    hud_widget_->update_rl_info(sim_id, metadata);
  }
}

void Visualizer::Impl::update_observation_debug_cloud(
  uint32_t sim_id, std::shared_ptr<const core::vision::PointCloud> cloud
)
{
  std::lock_guard lock(observation_debug_clouds_mutex_);
  if (cloud != nullptr && !cloud->empty()) {
    observation_debug_clouds_[sim_id] = std::move(cloud);
  } else {
    observation_debug_clouds_.erase(sim_id);
  }
}

void Visualizer::Impl::handle_zoom_adjust(bool zoom_in)
{
  auto sim_id    = interactive_style_->get_focused_viewport();
  auto camera_it = cameras_.find(sim_id);
  if (camera_it == cameras_.end() || !camera_it->second) {
    return;
  }

  const auto&  zoom_settings = config_->interactor_style.zoom;
  const double sensitivity   = config_->interactor_style.sensitivity;
  const double min_zoom      = config_ ? zoom_settings.max_zoom_in : 0.8;
  const double max_zoom      = config_ ? zoom_settings.max_zoom_out : 5.0;

  camera_it->second->adjust_zoom(zoom_in, sensitivity, min_zoom, max_zoom);
}

void Visualizer::Impl::toggle_pointclouds()
{
  show_pointcloud_ = !show_pointcloud_;
  for (auto& [_, viewport] : viewports_) {
    if (viewport) {
      viewport->set_show_pointcloud(show_pointcloud_);
    }
  }

  DDRL_LOG_INFO(
    logger_,
    "Cloud visualization mode: {}",
    show_pointcloud_ ? "raw sensor clouds" : "processed observation points"
  );

  if (render_window_ != nullptr) {
    render_window_->Render();
  }
}

void Visualizer::Impl::toggle_route_visualization()
{
  for (auto& [_, viewport] : viewports_) {
    if (viewport) {
      viewport->toggle_route_visualization_mode();
    }
  }

  DDRL_LOG_INFO(logger_, "Route visualization mode toggled");

  if (render_window_ != nullptr) {
    render_window_->Render();
  }
}

void Visualizer::Impl::toggle_sensor_frustums()
{
  show_sensor_frustums_ = !show_sensor_frustums_;
  for (auto& [_, viewport] : viewports_) {
    if (viewport) {
      viewport->set_show_sensor_frustums(show_sensor_frustums_);
    }
  }

  DDRL_LOG_INFO(
    logger_, "Sensor frustum visualization {}", show_sensor_frustums_ ? "enabled" : "disabled"
  );

  if (render_window_ != nullptr) {
    render_window_->Render();
  }
}

double Visualizer::Impl::compute_camera_snap_distance_sq() const
{
  constexpr double kMinSnapDistance = 50.0; // meters; avoid false positives during normal motion
  if (!config_) {
    return kMinSnapDistance * kMinSnapDistance;
  }

  const auto&  cam_cfg  = config_->camera_follow;
  const double chase_hw = std::hypot(cam_cfg.chase.back_distance, cam_cfg.chase.up_distance);
  const double orbit_hw = std::hypot(cam_cfg.orbit.radius, cam_cfg.orbit.height);
  const double top_hw   = cam_cfg.top_down.height;
  const double fp_hw    = cam_cfg.first_person.focal_distance;

  const double baseline  = std::max({chase_hw, orbit_hw, top_hw, fp_hw, kMinSnapDistance});
  const double snap_dist = std::max(baseline * 2.5, kMinSnapDistance);
  return snap_dist * snap_dist;
}

bool Visualizer::Impl::should_snap_camera(uint32_t sim_id, const Pose2& pose)
{
  const Point3 current{pose.pos.x, pose.pos.y, 0.0};
  auto         it = last_robot_positions_.find(sim_id);
  if (it == last_robot_positions_.end()) {
    last_robot_positions_.emplace(sim_id, current);
    return false;
  }

  const double dist_sq = core::math::distance_sq(it->second, current);
  it->second           = current;

  if (camera_snap_distance_sq_ <= 0.0) {
    return false;
  }
  return dist_sq >= camera_snap_distance_sq_;
}

void Visualizer::Impl::handle_cam_reset_view()
{
  auto sim_id    = interactive_style_->get_focused_viewport();
  auto camera_it = cameras_.find(sim_id);
  if (camera_it != cameras_.end() && camera_it->second) {
    camera_it->second->reset_view();
  }
  interactive_style_->handle_cam_reset_view();
}

void Visualizer::Impl::start_interactive()
{
  DDRL_LOG_INFO(logger_, "Visualizer::start_interactive >> Starting interactively in Visualizer");
  setup_viz_loop();
  setup_window_resize_handler();
  handle_cam_reset_view();
  interactor_->Start();
}

void Visualizer::Impl::stop_interactive()
{
  if (interactor_ == nullptr) {
    return;
  }
  if (viz_timer_id_ >= 0) {
    interactor_->DestroyTimer(viz_timer_id_);
    viz_timer_id_ = -1;
  }
  interactor_->SetDone(true); // just set the loop flag
  interactor_->InvokeEvent(vtkCommand::AbortCheckEvent);
  DDRL_LOG_INFO(logger_, "Visualizer::stop_interactive >> Visualization stopped");
}

void Visualizer::Impl::notify_shutdown_requested()
{
  DDRL_LOG_INFO(logger_, "Visualizer requested shutdown");
  if (on_shutdown_) {
    on_shutdown_();
  }
  stop_interactive();
}

void Visualizer::Impl::update_viewport_layout()
{
  auto viewport_layouts = compute_viewport_layout();

  // Create a set of visible viewport sim_ids for fast lookup
  std::unordered_set<uint32_t> visible_sim_ids;
  visible_sim_ids.reserve(viewport_layouts.size());
  for (const auto& layout : viewport_layouts) {
    visible_sim_ids.insert(layout.sim_id);
  }

  // Toggle visibility for all viewports based on current layout
  for (const auto& [sim_id, viewport_ptr] : viewports_) {
    auto* renderer = Impl::get_viewport_renderer(*viewport_ptr);
    if (renderer == nullptr) {
      continue;
    }

    bool should_be_visible = visible_sim_ids.contains(sim_id);

    if (should_be_visible) {
      // Find the layout for this viewport
      auto layout_it = std::ranges::find_if(viewport_layouts, [sim_id](const auto& layout) {
        return layout.sim_id == sim_id;
      });

      if (layout_it != viewport_layouts.end()) {
        const auto& rect = layout_it->viewport_rect;

        // Update viewport rectangle
        viewport_ptr->set_viewport_rect(rect[0], rect[1], rect[2], rect[3]);

        // Enable rendering for this viewport
        renderer->DrawOn();
      }
    } else {
      // Disable rendering for hidden viewports
      renderer->DrawOff();
    }
  }

  update_interactive_style_viewports();
  render_window_->Render();
}

std::vector<Visualizer::Impl::ViewportLayout> Visualizer::Impl::compute_viewport_layout() const
{
  const size_t num_sims = sim_ids_.size();

  std::vector<ViewportLayout> viewports;
  viewports.reserve(num_sims);

  auto current_view_mode = interactive_style_->get_view_mode();

  if (current_view_mode == ViewMode::FOCUSED) {
    ViewportLayout focused_vp;
    focused_vp.sim_id        = interactive_style_->get_focused_viewport();
    focused_vp.viewport_rect = {0.0, 0.0, 1.0, 1.0};
    viewports.push_back(focused_vp);
  } else {
    auto cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_sims))));
    auto rows = static_cast<int>(std::ceil(static_cast<double>(num_sims) / cols));

    // Spacing between viewports (in normalized coordinates)
    constexpr double kGap = 0.005; // 0.5% spacing between viewports

    for (size_t i = 0; i < num_sims; ++i) {
      const int row = static_cast<int>(i) / cols;
      const int col = static_cast<int>(i) % cols;

      ViewportLayout vp;
      vp.sim_id = sim_ids_[i];

      // Compute normalized viewport coordinates with spacing
      const double col_width  = 1.0 / cols;
      const double row_height = 1.0 / rows;

      const double x_min = col * col_width + kGap;
      const double x_max = (col + 1) * col_width - kGap;
      const double y_min = 1.0 - (row + 1) * row_height + kGap; // y bottom-up
      const double y_max = 1.0 - row * row_height - kGap;

      vp.viewport_rect = {x_min, y_min, x_max, y_max};

      viewports.push_back(vp);
    }
  }

  return viewports;
}

void Visualizer::Impl::handle_view_all()
{
  // for (auto& [sim_id, state] : camera_follow_states_) {
  //   state.zoom_factor = 10.0;
  // }
  if (hud_widget_) {
    hud_widget_->update_view_mode(ViewMode::ALL_VIEWPORTS);
  }

  update_viewport_layout();
}

void Visualizer::Impl::handle_viewport_focus()
{
  if (hud_widget_) {
    hud_widget_->update_view_mode(ViewMode::FOCUSED);
    hud_widget_->update_sim_id(interactive_style_->get_focused_viewport());
  }
  update_viewport_layout();
}

// =================================================================
//                    Visualizer public interface
// =================================================================

Visualizer::Visualizer(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

Visualizer::~Visualizer() = default;

void Visualizer::init()
{
  impl_->init();
}

void Visualizer::start_interactive()
{
  impl_->start_interactive();
}

void Visualizer::stop_interactive()
{
  impl_->stop_interactive();
}

void Visualizer::update_rl_info(uint32_t sim_id, const core::rl::RLMetadata& metadata)
{
  impl_->update_rl_info(sim_id, metadata);
}

void Visualizer::update_observation_debug_cloud(
  uint32_t sim_id, std::shared_ptr<const core::vision::PointCloud> cloud
)
{
  impl_->update_observation_debug_cloud(sim_id, std::move(cloud));
}

} // namespace ddrl::visualizer
