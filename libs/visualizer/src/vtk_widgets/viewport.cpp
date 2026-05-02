/**
 * @file viewport.cpp
 * @brief Implementation of Viewport widget (Pimpl)
 */

#include "visualizer/widgets/viewport.hpp"

#include "core/geom/aabb.hpp"
#include <array>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCompositeDataDisplayAttributes.h>
#include <vtkCompositePolyDataMapper2.h>
#include <vtkConeSource.h>
#include <vtkDiskSource.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkProperty.h>
#include <vtkRegularPolygonSource.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTubeFilter.h>

#include "logging/logging.hpp"
#include "visualizer/map_geometry_manager.hpp"
#include "visualizer/profile_manager.hpp"
#include "visualizer/widgets/grid.hpp"
#include "visualizer/widgets/point_cloud.hpp"
#include "visualizer/widgets/sensor_frustum.hpp"
#include "visualizer/widgets/shape.hpp"

namespace ddrl::visualizer {

using core::sim::WorldStateSnapshot;
using widgets::GridWidget;
using widgets::PointCloudActor;
using widgets::SensorFrustumActor;
using widgets::ShapeActor;
// ============================================================================
// Viewport Implementation (Pimpl)
// ============================================================================

/**
 * @brief Instance of a model backed by a single composite actor
 */
struct ActorInstance {
  vtkSmartPointer<vtkActor>            actor;     ///< Combined actor for the model
  vtkSmartPointer<vtkTransform>        transform; ///< Shared transform for fast updates
  std::string                          model_id;  ///< Model identifier
  const ProfileManager::ModelGeometry* model_geometry{nullptr};   ///< Cached geometry reference
  std::array<double, 3>                base_color{0.8, 0.4, 0.2}; ///< Default color (first block)
  core::math::Pose2                    last_pose;                 ///< Last pose for optimization
  vtkSmartPointer<vtkActor>            ring_actor;      ///< Bounding ring actor (for robot)
  vtkSmartPointer<vtkActor>            indicator_actor; ///< Overhead indicator actor (for robot)
  vtkSmartPointer<vtkTransform>        ring_transform;  ///< Reusable ring transform
  vtkSmartPointer<vtkTransform>        indicator_transform; ///< Reusable indicator transform
};

struct RouteActorSet {
  struct WaypointActor {
    vtkSmartPointer<vtkActor>      actor;
    vtkSmartPointer<vtkDiskSource> disk_source;
  };

  std::string                            route_name;
  std::vector<vtkSmartPointer<vtkActor>> path_actors;
  vtkSmartPointer<vtkActor>              start_actor;
  std::vector<WaypointActor>             waypoint_actors;
};

class Viewport::Impl
{
public:
  enum class RouteVisualizationMode : uint8_t {
    FullPath,
    ActiveGoalOnly,
  };

  explicit Impl(Params&& params);
  void switch_map(const std::string& new_map_name);
  void update_dynamic_geometry(const WorldStateSnapshot& state);
  void update_robot(const core::sim::RobotState& robot_state);
  void update_point_clouds(const WorldStateSnapshot& state);
  void update_observation_debug_cloud();
  void update_route_highlight(const WorldStateSnapshot& state);
  void remember_state(const WorldStateSnapshot& state);
  void set_observation_debug_cloud(std::shared_ptr<const core::vision::PointCloud> cloud);
  void toggle_route_visualization_mode();

  void                   set_show_pointcloud(bool show_raw_sensor_clouds);
  void                   set_show_sensor_frustums(bool enable);
  [[nodiscard]] bool     show_pointcloud() const { return show_pointcloud_; }
  [[nodiscard]] bool     show_sensor_frustums() const { return show_sensor_frustums_; }
  [[nodiscard]] uint32_t sim_id() const { return sim_id_; }

  [[nodiscard]] vtkRenderer* renderer() const { return renderer_; }
  void                       set_viewport_rect(double xmin, double ymin, double xmax, double ymax);
  void update_grid_from_bounds(const core::geom::AABB2& bounds, const std::string& map_name);

private:
  static vtkSmartPointer<vtkActor> create_ring_actor(double radius);
  static RouteActorSet::WaypointActor create_waypoint_actor(double radius);
  static vtkSmartPointer<vtkActor> create_overhead_indicator(double height);
  static vtkSmartPointer<vtkActor> create_route_path_actor(
    const core::math::Point2& start, const core::math::Point2& end,
    const std::array<double, 3>& color
  );
  void apply_pose_to_transform(const core::math::Pose2& pose, vtkTransform& transform) const;
  void tint_actor_blocks(
    vtkActor& actor, const ProfileManager::ModelGeometry& geometry,
    const std::array<double, 3>& color
  ) const;
  void restore_actor_blocks(vtkActor& actor, const ProfileManager::ModelGeometry& geometry) const;

  std::shared_ptr<logging::Logger> logger_ = logging::get_logger("Viewport");

  uint32_t              sim_id_{0};
  std::array<double, 4> viewport_rect_{0.0, 0.0, 1.0, 1.0};

  bool show_grid_{true};
  bool show_pointcloud_{true};
  bool show_sensor_frustums_{true};

  std::shared_ptr<const ProfileManager>       profile_manager_;
  std::shared_ptr<const MapGeometryManager>   map_geometry_manager_;
  std::shared_ptr<const config::WorldConfig>  world_config_;
  std::shared_ptr<const config::RobotProfile> robot_profile_;

  vtkSmartPointer<vtkRenderer> renderer_;

  // Current map state
  std::string                            current_map_name_;
  std::vector<vtkSmartPointer<vtkActor>> current_static_actors_;

  // Instance storage (uses ProfileManager for geometry)
  std::unordered_map<std::uint32_t, ActorInstance> dynamic_instances_;
  ActorInstance                                    robot_instance_;
  bool                                             robot_initialized_ = false;

  // Widgets for non-profiled geometry
  std::vector<std::unique_ptr<PointCloudActor>>    point_cloud_actors_;
  std::unique_ptr<PointCloudActor>                 observation_debug_cloud_actor_;
  std::vector<std::unique_ptr<SensorFrustumActor>> sensor_frustum_actors_;
  std::unique_ptr<GridWidget>                      grid_widget_;

  // Route path visualization (current map only)
  std::vector<RouteActorSet> current_route_actors_;
  double                     grid_spacing_{5.0};
  RouteVisualizationMode     route_visualization_mode_{RouteVisualizationMode::FullPath};
  std::optional<WorldStateSnapshot> last_state_;
  std::shared_ptr<const core::vision::PointCloud> observation_debug_cloud_;
};

Viewport::Impl::Impl(Params&& params)
    : sim_id_(params.sim_id),
      viewport_rect_(params.viewport_rect),
      show_grid_(params.show_grid),
      show_pointcloud_(params.show_pointcloud),
      show_sensor_frustums_(params.show_sensor_frustums),
      profile_manager_(std::move(params.profile_manager)),
      map_geometry_manager_(std::move(params.map_geometry_manager)),
      world_config_(std::move(params.world_config)),
      robot_profile_(std::move(params.robot_profile))
{
  renderer_ = vtkSmartPointer<vtkRenderer>::New();
  renderer_->SetBackground(0.1, 0.1, 0.15); // Dark blue background
  renderer_->SetViewport(
    viewport_rect_[0], viewport_rect_[1], viewport_rect_[2], viewport_rect_[3]
  );

  // Note: Map-specific geometry (zones, static objects) will be loaded
  // on first update() call when WorldStateSnapshot provides current_map

  // Set default camera position
  auto* camera = renderer_->GetActiveCamera();
  camera->SetPosition(25, 25, 30);
  camera->SetFocalPoint(25, 25, 0);
  camera->SetViewUp(0, 0, 1);
  renderer_->ResetCamera();
}

vtkSmartPointer<vtkActor> Viewport::Impl::create_ring_actor(double radius)
{
  // Create a ring using a disk with inner radius
  auto ring_source = vtkSmartPointer<vtkDiskSource>::New();
  ring_source->SetInnerRadius(radius);
  ring_source->SetOuterRadius(radius + 0.15); // Ring thickness of 0.15 units
  ring_source->SetRadialResolution(1);
  ring_source->SetCircumferentialResolution(64); // Smooth circle

  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(ring_source->GetOutputPort());

  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  // Bright color to stand out
  actor->GetProperty()->SetColor(1.0, 0.843, 0.0); // Gold color
  actor->GetProperty()->SetOpacity(0.7);
  actor->GetProperty()->SetLineWidth(3.0);

  return actor;
}

RouteActorSet::WaypointActor Viewport::Impl::create_waypoint_actor(double radius)
{
  RouteActorSet::WaypointActor waypoint_actor;
  waypoint_actor.disk_source = vtkSmartPointer<vtkDiskSource>::New();
  waypoint_actor.disk_source->SetInnerRadius(radius);
  waypoint_actor.disk_source->SetOuterRadius(radius + 0.15);
  waypoint_actor.disk_source->SetRadialResolution(1);
  waypoint_actor.disk_source->SetCircumferentialResolution(64);

  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(waypoint_actor.disk_source->GetOutputPort());

  waypoint_actor.actor = vtkSmartPointer<vtkActor>::New();
  waypoint_actor.actor->SetMapper(mapper);
  waypoint_actor.actor->GetProperty()->SetOpacity(0.7);
  waypoint_actor.actor->GetProperty()->SetLineWidth(3.0);
  return waypoint_actor;
}

vtkSmartPointer<vtkActor> Viewport::Impl::create_overhead_indicator(double /* height */)
{
  // Create a downward-pointing cone as an indicator
  auto cone_source = vtkSmartPointer<vtkConeSource>::New();
  cone_source->SetHeight(0.8);
  cone_source->SetRadius(0.4);
  cone_source->SetResolution(32);
  cone_source->SetDirection(0, 0, -1); // Point downward

  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(cone_source->GetOutputPort());

  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  // Bright color to stand out
  actor->GetProperty()->SetColor(0.0, 1.0, 0.5); // Bright cyan-green
  actor->GetProperty()->SetOpacity(0.85);

  return actor;
}

vtkSmartPointer<vtkActor> Viewport::Impl::create_route_path_actor(
  const core::math::Point2& start, const core::math::Point2& end,
  const std::array<double, 3>& color
)
{
  auto vtk_points = vtkSmartPointer<vtkPoints>::New();
  auto poly_line  = vtkSmartPointer<vtkPolyLine>::New();
  poly_line->GetPointIds()->SetNumberOfIds(2);
  vtk_points->InsertNextPoint(start.x, start.y, 0.05);
  vtk_points->InsertNextPoint(end.x, end.y, 0.05);
  poly_line->GetPointIds()->SetId(0, 0);
  poly_line->GetPointIds()->SetId(1, 1);

  auto cells = vtkSmartPointer<vtkCellArray>::New();
  cells->InsertNextCell(poly_line);

  auto poly_data = vtkSmartPointer<vtkPolyData>::New();
  poly_data->SetPoints(vtk_points);
  poly_data->SetLines(cells);

  auto tube = vtkSmartPointer<vtkTubeFilter>::New();
  tube->SetInputData(poly_data);
  tube->SetRadius(0.12);
  tube->SetNumberOfSides(16);
  tube->CappingOn();

  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(tube->GetOutputPort());

  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(color[0], color[1], color[2]);
  actor->GetProperty()->SetOpacity(0.8);
  return actor;
}

void Viewport::Impl::apply_pose_to_transform(
  const core::math::Pose2& pose, vtkTransform& transform
) const
{
  transform.Identity();
  transform.Translate(pose.pos.x, pose.pos.y, 0.0);
  transform.RotateZ(pose.yaw * 180.0 / M_PI);
}

void Viewport::Impl::tint_actor_blocks(
  vtkActor& actor, const ProfileManager::ModelGeometry& geometry, const std::array<double, 3>& color
) const
{
  if (auto* mapper = vtkCompositePolyDataMapper2::SafeDownCast(actor.GetMapper())) {
    if (auto* attributes = mapper->GetCompositeDataDisplayAttributes()) {
      for (unsigned int block_index = 0; block_index < geometry.block_colors.size();
           ++block_index) {
        if (auto* block = geometry.multiblock->GetBlock(block_index)) {
          attributes->SetBlockColor(block, color.data());
        }
      }
      return;
    }
  }
  actor.GetProperty()->SetColor(color[0], color[1], color[2]);
}

void Viewport::Impl::restore_actor_blocks(
  vtkActor& actor, const ProfileManager::ModelGeometry& geometry
) const
{
  if (auto* mapper = vtkCompositePolyDataMapper2::SafeDownCast(actor.GetMapper())) {
    if (auto* attributes = mapper->GetCompositeDataDisplayAttributes()) {
      for (unsigned int block_index = 0; block_index < geometry.block_colors.size();
           ++block_index) {
        if (auto* block = geometry.multiblock->GetBlock(block_index)) {
          const auto& block_color = geometry.block_colors[block_index];
          attributes->SetBlockColor(block, block_color.data());
        }
      }
      return;
    }
  }
  actor.GetProperty()->SetColor(
    geometry.block_colors[0][0], geometry.block_colors[0][1], geometry.block_colors[0][2]
  );
}

void Viewport::Impl::switch_map(const std::string& new_map_name)
{
  // Don't reload if already on this map
  if (current_map_name_ == new_map_name) {
    return;
  }

  // Get map geometry template
  const auto* map_geom = map_geometry_manager_->get_map_geometry(new_map_name);
  if (map_geom == nullptr) {
    DDRL_LOG_ERROR(
      logger_,
      "Cannot switch to map '{}': not found in geometry cache. Keeping current map '{}'.",
      new_map_name,
      current_map_name_
    );
    return;
  }

  update_grid_from_bounds(map_geom->bounds, new_map_name);

  // Remove old static actors
  for (auto& actor : current_static_actors_) {
    renderer_->RemoveActor(actor);
  }
  current_static_actors_.clear();

  for (auto& route_actors : current_route_actors_) {
    for (auto& path_actor : route_actors.path_actors) {
      renderer_->RemoveActor(path_actor);
    }
    if (route_actors.start_actor != nullptr) {
      renderer_->RemoveActor(route_actors.start_actor);
    }
    for (auto& waypoint_actor : route_actors.waypoint_actors) {
      renderer_->RemoveActor(waypoint_actor.actor);
    }
  }
  current_route_actors_.clear();

  // Create static object actors from templates
  for (const auto& obj_template : map_geom->static_objects) {
    auto actor = profile_manager_->create_instance_actor(obj_template.model_id);
    if (actor == nullptr) {
      DDRL_LOG_WARN(
        logger_,
        "Failed to create actor for static object '{}' with model_id '{}' in map '{}'",
        obj_template.name,
        obj_template.model_id,
        new_map_name
      );
      continue;
    }

    auto transform = vtkSmartPointer<vtkTransform>::New();
    actor->SetUserTransform(transform);
    apply_pose_to_transform(obj_template.world_pose, *transform);

    renderer_->AddActor(actor);
    current_static_actors_.push_back(actor);
  }

  static constexpr std::array<std::array<double, 3>, 4> kRouteColors{{
    {0.00, 0.70, 1.00},
    {1.00, 0.75, 0.20},
    {0.30, 0.90, 0.45},
    {1.00, 0.40, 0.60},
  }};
  for (std::size_t route_idx = 0; route_idx < map_geom->routes.size(); ++route_idx) {
    const auto& route = map_geom->routes[route_idx];
    const auto& color = kRouteColors[route_idx % kRouteColors.size()];
    RouteActorSet route_actors;
    route_actors.route_name = route.name;

    core::math::Point2 segment_start = route.start_pose.pos;
    for (const auto& waypoint : route.waypoints) {
      auto path_actor = create_route_path_actor(segment_start, waypoint.position, color);
      renderer_->AddActor(path_actor);
      route_actors.path_actors.push_back(path_actor);
      segment_start = waypoint.position;
    }

    auto start_marker = vtkSmartPointer<vtkActor>::New();
    start_marker = create_ring_actor(0.45);
    start_marker->SetPosition(route.start_pose.pos.x, route.start_pose.pos.y, 0.02);
    start_marker->GetProperty()->SetColor(1.0, 1.0, 1.0);
    start_marker->GetProperty()->SetOpacity(0.9);
    renderer_->AddActor(start_marker);
    route_actors.start_actor = start_marker;

    for (const auto& waypoint : route.waypoints) {
      auto waypoint_actor = create_waypoint_actor(std::max(0.2, waypoint.tolerance));
      waypoint_actor.actor->SetPosition(waypoint.position.x, waypoint.position.y, 0.02);
      if (waypoint.is_final) {
        waypoint_actor.actor->GetProperty()->SetColor(1.0, 0.25, 0.25);
      } else {
        waypoint_actor.actor->GetProperty()->SetColor(color[0], color[1], color[2]);
      }
      renderer_->AddActor(waypoint_actor.actor);
      route_actors.waypoint_actors.push_back(std::move(waypoint_actor));
    }

    current_route_actors_.push_back(std::move(route_actors));
  }

  current_map_name_ = new_map_name;
}

void Viewport::Impl::update_route_highlight(const WorldStateSnapshot& state)
{
  for (auto& route_actors : current_route_actors_) {
    const bool is_active = !state.active_route_name.empty() && route_actors.route_name == state.active_route_name;
    const bool show_full_path = route_visualization_mode_ == RouteVisualizationMode::FullPath;

    for (std::size_t segment_idx = 0; segment_idx < route_actors.path_actors.size(); ++segment_idx) {
      auto& path_actor = route_actors.path_actors[segment_idx];
      if (path_actor == nullptr) {
        continue;
      }
      const bool is_completed_segment =
        is_active && segment_idx < static_cast<std::size_t>(state.active_waypoint_index);
      path_actor->SetVisibility(show_full_path ? 1 : 0);
      path_actor->GetProperty()->SetOpacity(show_full_path ? (is_active ? (is_completed_segment ? 0.20 : 1.0) : 0.18) : 0.0);
      path_actor->GetProperty()->SetLineWidth(is_active ? 4.0 : 2.0);
    }
    if (route_actors.start_actor != nullptr) {
      const bool has_departed_start =
        is_active && static_cast<std::size_t>(state.active_waypoint_index) > 0U;
      route_actors.start_actor->SetVisibility(show_full_path ? 1 : 0);
      route_actors.start_actor->GetProperty()->SetOpacity(
        show_full_path ? (is_active ? (has_departed_start ? 0.35 : 1.0) : 0.25) : 0.0
      );
      route_actors.start_actor->SetScale(1.0, 1.0, 1.0);
    }

    for (std::size_t idx = 0; idx < route_actors.waypoint_actors.size(); ++idx) {
      auto& waypoint_actor = route_actors.waypoint_actors[idx];
      const bool is_current_waypoint =
        is_active && idx == static_cast<std::size_t>(state.active_waypoint_index);
      const bool is_completed_waypoint =
        is_active && idx < static_cast<std::size_t>(state.active_waypoint_index);
      const bool is_visible =
        show_full_path ? true : is_current_waypoint;
      vtkActor* actor      = waypoint_actor.actor.GetPointer();
      vtkDiskSource* disk  = waypoint_actor.disk_source.GetPointer();
      if (actor == nullptr || disk == nullptr) {
        continue;
      }

      const double outer_radius = disk->GetOuterRadius();
      const double ring_radius  = std::max(0.0, outer_radius - 0.15);
      disk->SetInnerRadius(is_current_waypoint ? 0.0 : ring_radius);
      disk->Modified();

      actor->SetVisibility(is_visible ? 1 : 0);
      actor->GetProperty()->SetAmbient(0.0);
      actor->GetProperty()->SetDiffuse(1.0);
      actor->GetProperty()->SetSpecular(0.0);
      actor->GetProperty()->SetSpecularPower(1.0);
      actor->GetProperty()->SetOpacity(
        is_visible ? (is_current_waypoint ? 1.0 : (is_active ? (is_completed_waypoint ? 0.28 : 0.95) : 0.22)) : 0.0
      );
      actor->SetScale(1.0, 1.0, 1.0);
      actor->GetProperty()->SetLineWidth(is_current_waypoint ? 5.0 : 3.0);
      actor->GetProperty()->SetEdgeVisibility(0);
    }
  }
}

void Viewport::Impl::toggle_route_visualization_mode()
{
  route_visualization_mode_ =
    route_visualization_mode_ == RouteVisualizationMode::FullPath
      ? RouteVisualizationMode::ActiveGoalOnly
      : RouteVisualizationMode::FullPath;

  if (last_state_.has_value()) {
    update_route_highlight(*last_state_);
  }
}

void Viewport::Impl::remember_state(const WorldStateSnapshot& state)
{
  last_state_ = state;
}

void Viewport::Impl::set_observation_debug_cloud(
  std::shared_ptr<const core::vision::PointCloud> cloud
)
{
  observation_debug_cloud_ = std::move(cloud);
}

void Viewport::Impl::update_dynamic_geometry(const WorldStateSnapshot& state)
{
  std::unordered_set<std::uint32_t> active_ids;
  active_ids.reserve(state.dynamic_objects.size());

  for (const auto& obj : state.dynamic_objects) {
    if (obj.state_name == "INACTIVE") {
      auto it = dynamic_instances_.find(obj.id);
      if (it != dynamic_instances_.end()) {
        if (it->second.actor != nullptr) {
          renderer_->RemoveActor(it->second.actor);
        }
        dynamic_instances_.erase(it);
      }
      continue;
    }

    active_ids.insert(obj.id);
    auto it = dynamic_instances_.find(obj.id);

    if (it == dynamic_instances_.end()) {
      ActorInstance instance;
      instance.model_id       = obj.model_id;
      instance.last_pose      = obj.pose;
      instance.model_geometry = profile_manager_->get_model_geometry(obj.model_id);

      if (instance.model_geometry != nullptr && !instance.model_geometry->block_colors.empty()) {
        instance.base_color = instance.model_geometry->block_colors.front();
      }

      instance.actor = profile_manager_->create_instance_actor(obj.model_id);
      if (instance.actor == nullptr) {
        DDRL_LOG_WARN(
          logger_,
          "Failed to establish actor instance of object '{}' (id={}) with model_id '{}'",
          obj.name,
          obj.id,
          obj.model_id
        );
        continue;
      }

      instance.transform = vtkSmartPointer<vtkTransform>::New();
      instance.actor->SetUserTransform(instance.transform);

      renderer_->AddActor(instance.actor);
      apply_pose_to_transform(obj.pose, *instance.transform);

      dynamic_instances_[obj.id] = std::move(instance);
    } else {
      auto& instance = it->second;
      if (instance.transform != nullptr) {
        apply_pose_to_transform(obj.pose, *instance.transform);
      }
      instance.last_pose = obj.pose;
    }
  }

  for (auto it = dynamic_instances_.begin(); it != dynamic_instances_.end();) {
    if (active_ids.find(it->first) == active_ids.end()) {
      if (it->second.actor != nullptr) {
        renderer_->RemoveActor(it->second.actor);
      }
      it = dynamic_instances_.erase(it);
    } else {
      ++it;
    }
  }
}

void Viewport::Impl::update_robot(const core::sim::RobotState& robot_state)
{
  if (!robot_initialized_) {
    robot_instance_.model_id       = robot_state.model_id;
    robot_instance_.last_pose      = robot_state.pose;
    robot_instance_.model_geometry = profile_manager_->get_model_geometry(robot_state.model_id);

    if (robot_instance_.model_geometry != nullptr &&
        !robot_instance_.model_geometry->block_colors.empty()) {
      robot_instance_.base_color = robot_instance_.model_geometry->block_colors.front();
    } else {
      robot_instance_.base_color = {0.2, 0.6, 0.9};
    }

    robot_instance_.actor = profile_manager_->create_instance_actor(robot_state.model_id);
    if (robot_instance_.actor == nullptr) {
      DDRL_LOG_WARN(logger_, "Unable to create robot actor for model '{}'", robot_state.model_id);
      return;
    }

    robot_instance_.transform = vtkSmartPointer<vtkTransform>::New();
    robot_instance_.actor->SetUserTransform(robot_instance_.transform);
    renderer_->AddActor(robot_instance_.actor);

    robot_instance_.ring_actor          = create_ring_actor(1.2);
    robot_instance_.indicator_actor     = create_overhead_indicator(3.0);
    robot_instance_.ring_transform      = vtkSmartPointer<vtkTransform>::New();
    robot_instance_.indicator_transform = vtkSmartPointer<vtkTransform>::New();

    if (robot_instance_.ring_actor != nullptr) {
      robot_instance_.ring_actor->SetUserTransform(robot_instance_.ring_transform);
      renderer_->AddActor(robot_instance_.ring_actor);
    }
    if (robot_instance_.indicator_actor != nullptr) {
      robot_instance_.indicator_actor->SetUserTransform(robot_instance_.indicator_transform);
      renderer_->AddActor(robot_instance_.indicator_actor);
    }

    if (robot_profile_ != nullptr) {
      for (const auto& sensor_config : robot_profile_->sensors) {
        auto frustum = std::make_unique<SensorFrustumActor>(sensor_config);
        frustum->add_to_renderer(*renderer_);
        frustum->set_visible(show_sensor_frustums_);
        frustum->update_pose(robot_state.pose);
        sensor_frustum_actors_.push_back(std::move(frustum));
      }
      DDRL_LOG_DEBUG(
        logger_, "Created {} sensor frustum actors for robot", sensor_frustum_actors_.size()
      );
    }

    robot_initialized_ = true;
  }

  if (robot_instance_.transform != nullptr) {
    apply_pose_to_transform(robot_state.pose, *robot_instance_.transform);
  }

  if (robot_instance_.model_geometry != nullptr) {
    if (robot_state.is_collided) {
      tint_actor_blocks(*robot_instance_.actor, *robot_instance_.model_geometry, {1.0, 0.0, 0.0});
    } else {
      restore_actor_blocks(*robot_instance_.actor, *robot_instance_.model_geometry);
    }
  }

  if (robot_instance_.ring_transform != nullptr) {
    robot_instance_.ring_transform->Identity();
    robot_instance_.ring_transform->Translate(robot_state.pose.pos.x, robot_state.pose.pos.y, 0.05);
    robot_instance_.ring_transform->RotateZ(robot_state.pose.yaw * 180.0 / M_PI);
  }

  if (robot_instance_.ring_actor != nullptr) {
    if (robot_state.is_collided) {
      robot_instance_.ring_actor->GetProperty()->SetColor(1.0, 0.2, 0.1);
    } else {
      robot_instance_.ring_actor->GetProperty()->SetColor(1.0, 0.843, 0.0);
    }
  }

  if (robot_instance_.indicator_transform != nullptr) {
    robot_instance_.indicator_transform->Identity();
    robot_instance_.indicator_transform->Translate(
      robot_state.pose.pos.x, robot_state.pose.pos.y, 3.0
    );
    robot_instance_.indicator_transform->RotateZ(robot_state.pose.yaw * 180.0 / M_PI);
  }

  if (robot_instance_.indicator_actor != nullptr) {
    if (robot_state.is_collided) {
      robot_instance_.indicator_actor->GetProperty()->SetColor(1.0, 0.2, 0.1);
    } else {
      robot_instance_.indicator_actor->GetProperty()->SetColor(0.0, 1.0, 0.5);
    }
  }

  if (show_sensor_frustums_) {
    for (auto& frustum : sensor_frustum_actors_) {
      frustum->update_pose(robot_state.pose);
    }
  }

  robot_instance_.last_pose = robot_state.pose;
}

void Viewport::Impl::update_point_clouds(const WorldStateSnapshot& state)
{
  // Count valid (non-empty) clouds
  size_t valid_cloud_count = 0;
  for (const auto& cloud_ptr : state.sensor_clouds) {
    if (cloud_ptr && !cloud_ptr->empty()) {
      ++valid_cloud_count;
    }
  }

  // Adjust actor pool size to match valid cloud count
  if (point_cloud_actors_.size() > valid_cloud_count) {
    // Remove excess actors
    for (size_t i = valid_cloud_count; i < point_cloud_actors_.size(); ++i) {
      point_cloud_actors_[i]->remove_from_renderer(*renderer_);
    }
    point_cloud_actors_.resize(valid_cloud_count);
  } else if (point_cloud_actors_.size() < valid_cloud_count) {
    // Add new actors for additional clouds
    std::array<double, 3> color         = {0.0, 1.0, 0.0}; // Green
    size_t                actors_to_add = valid_cloud_count - point_cloud_actors_.size();

    for (size_t i = 0; i < actors_to_add; ++i) {
      // Create with empty initial data (will be updated immediately below)
      std::vector<core::math::Point3> empty_points;
      auto widget = std::make_unique<PointCloudActor>(empty_points, color, 2.0);
      widget->add_to_renderer(*renderer_);
      point_cloud_actors_.push_back(std::move(widget));
    }
  }

  // Update existing actors with new point data (zero-copy via shared_ptr)
  size_t actor_index = 0;
  for (const auto& cloud_ptr : state.sensor_clouds) {
    if (cloud_ptr && !cloud_ptr->empty()) {
      // Use update_points() to reuse existing VTK pipeline (efficient!)
      // Note: update_points takes non-const ref to span (API quirk), so create variable
      auto points_span = std::span<const core::math::Point3>(cloud_ptr->points);
      point_cloud_actors_[actor_index]->update_points(points_span);
      ++actor_index;
    }
  }
}

void Viewport::Impl::update_observation_debug_cloud()
{
  const bool has_points =
    observation_debug_cloud_ != nullptr && !observation_debug_cloud_->empty();

  if (!has_points) {
    if (observation_debug_cloud_actor_ != nullptr) {
      observation_debug_cloud_actor_->set_visible(false);
    }
    return;
  }

  if (observation_debug_cloud_actor_ == nullptr) {
    std::vector<core::math::Point3> empty_points;
    observation_debug_cloud_actor_ =
      std::make_unique<PointCloudActor>(empty_points, std::array<double, 3>{1.0, 0.2, 1.0}, 5.0);
    observation_debug_cloud_actor_->add_to_renderer(*renderer_);
  }

  const auto points_span =
    std::span<const core::math::Point3>(observation_debug_cloud_->points);
  observation_debug_cloud_actor_->update_points(points_span);
  observation_debug_cloud_actor_->set_visible(!show_pointcloud_);
}

void Viewport::Impl::set_show_pointcloud(bool show_raw_sensor_clouds)
{
  if (show_pointcloud_ == show_raw_sensor_clouds) {
    return;
  }
  show_pointcloud_ = show_raw_sensor_clouds;

  for (auto& actor : point_cloud_actors_) {
    actor->set_visible(show_raw_sensor_clouds);
  }
  if (observation_debug_cloud_actor_ != nullptr) {
    observation_debug_cloud_actor_->set_visible(!show_raw_sensor_clouds);
  }
}

void Viewport::Impl::set_show_sensor_frustums(bool enable)
{
  if (show_sensor_frustums_ == enable) {
    return;
  }
  show_sensor_frustums_ = enable;

  for (auto& frustum : sensor_frustum_actors_) {
    frustum->set_visible(enable);
    if (enable) {
      frustum->update_pose(robot_instance_.last_pose);
    }
  }
}

void Viewport::Impl::update_grid_from_bounds(
  const core::geom::AABB2& bounds, const std::string& map_name
)
{
  if (!show_grid_) {
    return;
  }

  const double fallback_size = 100.0;
  const double width         = bounds.max.x - bounds.min.x;
  const double height        = bounds.max.y - bounds.min.y;

  double world_size = fallback_size;
  if (std::isfinite(width) && std::isfinite(height) && width > 0.0 && height > 0.0) {
    world_size = std::max(width, height);
  } else {
    DDRL_LOG_WARN(
      logger_,
      "Map '{}' has invalid bounds (min: [{:.2f}, {:.2f}], max: [{:.2f}, {:.2f}]); using fallback "
      "grid size {}",
      map_name,
      bounds.min.x,
      bounds.min.y,
      bounds.max.x,
      bounds.max.y,
      fallback_size
    );
  }

  // Ensure grid has a minimum usable size
  world_size = std::max(world_size, 10.0);

  if (!grid_widget_) {
    grid_widget_ = std::make_unique<GridWidget>(world_size, grid_spacing_);
    grid_widget_->add_to_renderer(*renderer_);
  } else {
    grid_widget_->set_size(world_size);
  }

  grid_widget_->set_visible(true);
}

void Viewport::Impl::set_viewport_rect(double xmin, double ymin, double xmax, double ymax)
{
  viewport_rect_[0] = xmin;
  viewport_rect_[1] = ymin;
  viewport_rect_[2] = xmax;
  viewport_rect_[3] = ymax;

  // Update the renderer's viewport
  renderer_->SetViewport(xmin, ymin, xmax, ymax);
}

// ==============================================================
// Viewport public interface
// ==============================================================

Viewport::Viewport(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

Viewport::~Viewport() = default;

void Viewport::update(const WorldStateSnapshot& state)
{
  impl_->remember_state(state);

  // Check if map changed and switch if needed
  if (!state.current_map.empty()) {
    impl_->switch_map(state.current_map);
  }

  impl_->update_route_highlight(state);

  // Update dynamic geometry
  impl_->update_dynamic_geometry(state);

  // Update robot
  impl_->update_robot(state.robot);

  // Update sensor point clouds
  if (impl_->show_pointcloud()) {
    impl_->update_point_clouds(state);
  }
  impl_->update_observation_debug_cloud();
}

void Viewport::set_viewport_rect(double xmin, double ymin, double xmax, double ymax)
{
  impl_->set_viewport_rect(xmin, ymin, xmax, ymax);
}

void Viewport::set_show_pointcloud(bool enable)
{
  impl_->set_show_pointcloud(enable);
}

void Viewport::set_show_sensor_frustums(bool enable)
{
  impl_->set_show_sensor_frustums(enable);
}

void Viewport::toggle_route_visualization_mode()
{
  impl_->toggle_route_visualization_mode();
}

void Viewport::set_observation_debug_cloud(std::shared_ptr<const core::vision::PointCloud> cloud)
{
  impl_->set_observation_debug_cloud(std::move(cloud));
  impl_->update_observation_debug_cloud();
}

bool Viewport::show_pointcloud_enabled() const
{
  return impl_->show_pointcloud();
}

bool Viewport::show_sensor_frustums_enabled() const
{
  return impl_->show_sensor_frustums();
}

::vtkRenderer* Viewport::renderer_for_friend() const
{
  return impl_->renderer();
}

} // namespace ddrl::visualizer
