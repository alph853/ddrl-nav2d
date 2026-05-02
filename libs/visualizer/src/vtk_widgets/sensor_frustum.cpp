/**
 * @file sensor_frustum.cpp
 * @brief Implementation of sensor frustum visualization
 */

#include "visualizer/widgets/sensor_frustum.hpp"

#include <cmath>
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkConeSource.h>
#include <vtkCylinderSource.h>
#include <vtkMatrix4x4.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace ddrl::visualizer::widgets {

namespace {

vtkSmartPointer<vtkPolyData> build_depth_camera_polydata(
  double range, double hfov_half, double vfov_half, bool include_faces, bool include_rays
)
{
  const double z_range  = range;
  const double x_left   = -range * hfov_half;
  const double x_right  = range * hfov_half;
  const double y_bottom = range * vfov_half;
  const double y_top    = -range * vfov_half;

  auto points = vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(0.0, 0.0, 0.0);
  points->InsertNextPoint(x_left, y_bottom, z_range);
  points->InsertNextPoint(x_right, y_bottom, z_range);
  points->InsertNextPoint(x_right, y_top, z_range);
  points->InsertNextPoint(x_left, y_top, z_range);

  auto lines = vtkSmartPointer<vtkCellArray>::New();
  if (include_rays) {
    for (vtkIdType i = 1; i <= 4; ++i) {
      lines->InsertNextCell(2);
      lines->InsertCellPoint(0);
      lines->InsertCellPoint(i);
    }
  }

  constexpr std::array<vtkIdType, 5> rect_indices = {1, 2, 3, 4, 1};
  for (size_t i = 0; i < 4; ++i) {
    lines->InsertNextCell(2);
    lines->InsertCellPoint(rect_indices[i]);
    lines->InsertCellPoint(rect_indices[i + 1]);
  }

  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->SetLines(lines);

  if (include_faces) {
    auto triangles = vtkSmartPointer<vtkCellArray>::New();
    constexpr std::array<std::array<vtkIdType, 3>, 4> pyramid_faces = {{
      {0, 1, 2},
      {0, 2, 3},
      {0, 3, 4},
      {0, 4, 1}
    }};

    for (const auto& face : pyramid_faces) {
      triangles->InsertNextCell(3, face.data());
    }

    constexpr std::array<std::array<vtkIdType, 3>, 2> far_tris = {{{1, 2, 3}, {1, 3, 4}}};
    for (const auto& tri : far_tris) {
      triangles->InsertNextCell(3, tri.data());
    }
    polydata->SetPolys(triangles);
  }

  return polydata;
}

vtkSmartPointer<vtkPolyData> build_lidar2d_polydata(
  double range, double angle_min, double angle_max, int num_rays, bool include_faces,
  bool include_rays
)
{
  auto points = vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(0.0, 0.0, 0.0);

  for (int i = 0; i <= num_rays; ++i) {
    const double angle = angle_min + (angle_max - angle_min) * static_cast<double>(i) / num_rays;
    points->InsertNextPoint(range * std::cos(angle), range * std::sin(angle), 0.0);
  }

  auto lines = vtkSmartPointer<vtkCellArray>::New();
  if (include_rays) {
    lines->InsertNextCell(2);
    lines->InsertCellPoint(0);
    lines->InsertCellPoint(1);

    lines->InsertNextCell(2);
    lines->InsertCellPoint(0);
    lines->InsertCellPoint(num_rays + 1);
  }

  for (int i = 1; i <= num_rays; ++i) {
    lines->InsertNextCell(2);
    lines->InsertCellPoint(i);
    lines->InsertCellPoint(i + 1);
  }

  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->SetLines(lines);

  if (include_faces) {
    auto triangles = vtkSmartPointer<vtkCellArray>::New();
    for (int i = 1; i <= num_rays; ++i) {
      const std::array<vtkIdType, 3> tri = {
        0, static_cast<vtkIdType>(i), static_cast<vtkIdType>(i + 1)
      };
      triangles->InsertNextCell(3, tri.data());
    }
    polydata->SetPolys(triangles);
  }

  return polydata;
}

vtkSmartPointer<vtkPolyData> build_lidar3d_polydata(
  double range, double vfov_half, int num_azimuth, int num_vertical, bool include_faces
)
{
  auto points = vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(0.0, 0.0, 0.0);

  for (int v = 0; v <= num_vertical; ++v) {
    const double elev         = -vfov_half + 2.0 * vfov_half * static_cast<double>(v) / num_vertical;
    const double r_horizontal = range * std::cos(elev);
    const double z            = range * std::sin(elev);

    for (int h = 0; h <= num_azimuth; ++h) {
      const double azimuth = 2.0 * M_PI * static_cast<double>(h) / num_azimuth;
      points->InsertNextPoint(r_horizontal * std::cos(azimuth), r_horizontal * std::sin(azimuth), z);
    }
  }

  auto lines = vtkSmartPointer<vtkCellArray>::New();
  for (int h = 0; h <= num_azimuth; h += 4) {
    for (int v = 0; v < num_vertical; ++v) {
      const vtkIdType p1 = 1 + v * (num_azimuth + 1) + h;
      const vtkIdType p2 = 1 + (v + 1) * (num_azimuth + 1) + h;
      lines->InsertNextCell(2);
      lines->InsertCellPoint(p1);
      lines->InsertCellPoint(p2);
    }
  }

  for (int v = 0; v <= num_vertical; v += 2) {
    for (int h = 0; h < num_azimuth; ++h) {
      const vtkIdType p1 = 1 + v * (num_azimuth + 1) + h;
      const vtkIdType p2 = 1 + v * (num_azimuth + 1) + h + 1;
      lines->InsertNextCell(2);
      lines->InsertCellPoint(p1);
      lines->InsertCellPoint(p2);
    }
  }

  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->SetLines(lines);

  if (include_faces) {
    auto triangles = vtkSmartPointer<vtkCellArray>::New();
    for (int v = 0; v < num_vertical; ++v) {
      for (int h = 0; h < num_azimuth; ++h) {
        const vtkIdType p1 = 1 + v * (num_azimuth + 1) + h;
        const vtkIdType p2 = 1 + v * (num_azimuth + 1) + h + 1;
        const vtkIdType p3 = 1 + (v + 1) * (num_azimuth + 1) + h;
        const vtkIdType p4 = 1 + (v + 1) * (num_azimuth + 1) + h + 1;

        const std::array<vtkIdType, 3> tri1 = {p1, p2, p3};
        const std::array<vtkIdType, 3> tri2 = {p2, p4, p3};
        triangles->InsertNextCell(3, tri1.data());
        triangles->InsertNextCell(3, tri2.data());
      }
    }
    polydata->SetPolys(triangles);
  }

  return polydata;
}

} // namespace

SensorFrustumActor::SensorFrustumActor(
  const core::sensors::SensorConfig& sensor_config, const std::array<double, 3>& color,
  double opacity
)
    : color_(color), opacity_(opacity)
{
  std::visit(
    [this](const auto& config) {
      using T               = std::decay_t<decltype(config)>;
      sensor_pose_rel_base_ = config.base.pose_rel_base;

      if constexpr (std::is_same_v<T, core::sensors::DepthCamera>) {
        mount_marker_type_ = MountMarkerType::Camera;
        create_depth_camera_frustum(config);
      } else if constexpr (std::is_same_v<T, core::sensors::Lidar2D>) {
        mount_marker_type_ = MountMarkerType::Lidar;
        create_lidar2d_frustum(config);
      } else if constexpr (std::is_same_v<T, core::sensors::Lidar3D>) {
        mount_marker_type_ = MountMarkerType::Lidar;
        create_lidar3d_frustum(config);
      }
    },
    sensor_config
  );

  create_mount_marker();
}

void SensorFrustumActor::create_mount_marker()
{
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  auto actor = vtkSmartPointer<vtkActor>::New();
  if (mount_marker_type_ == MountMarkerType::Camera) {
    auto cone = vtkSmartPointer<vtkConeSource>::New();
    cone->SetRadius(0.16);
    cone->SetHeight(0.42);
    cone->SetResolution(24);
    cone->SetDirection(0.0, 0.0, 1.0);
    mapper->SetInputConnection(cone->GetOutputPort());
  } else {
    auto cylinder = vtkSmartPointer<vtkCylinderSource>::New();
    cylinder->SetRadius(0.12);
    cylinder->SetHeight(0.32);
    cylinder->SetResolution(24);
    cylinder->CappingOn();

    auto rotate = vtkSmartPointer<vtkTransform>::New();
    rotate->Identity();
    rotate->RotateX(90.0);

    auto transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(rotate);
    transform_filter->SetInputConnection(cylinder->GetOutputPort());
    mapper->SetInputConnection(transform_filter->GetOutputPort());
  }

  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(color_[0], color_[1], color_[2]);
  actor->GetProperty()->SetOpacity(0.95);
  add_actor(actor);
}

void SensorFrustumActor::add_actor(vtkSmartPointer<vtkActor> actor)
{
  if (actor != nullptr) {
    actors_.push_back(std::move(actor));
  }
}

vtkSmartPointer<vtkActor> SensorFrustumActor::make_polydata_actor(
  vtkSmartPointer<vtkPolyData> polydata, const std::array<double, 3>& color, double opacity,
  bool edge_visibility, double line_width
)
{
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(polydata);

  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(color[0], color[1], color[2]);
  actor->GetProperty()->SetOpacity(opacity);
  actor->GetProperty()->SetEdgeVisibility(edge_visibility ? 1 : 0);
  actor->GetProperty()->SetLineWidth(line_width);
  return actor;
}

void SensorFrustumActor::create_depth_camera_frustum(const core::sensors::DepthCamera& config)
{
  const double min_range = static_cast<double>(config.base.min_range);
  const double max_range = static_cast<double>(config.base.max_range);
  const double width     = static_cast<double>(config.width);
  const double height    = static_cast<double>(config.height);
  const double hfov_half = width / 2.0 / static_cast<double>(config.fx);
  const double vfov_half = height / 2.0 / static_cast<double>(config.fy);

  if (min_range > 1e-6) {
    add_actor(make_polydata_actor(
      build_depth_camera_polydata(min_range, hfov_half, vfov_half, false, false),
      color_,
      std::min(0.95, opacity_ + 0.35),
      true,
      2.5
    ));
  }

  add_actor(make_polydata_actor(
    build_depth_camera_polydata(max_range, hfov_half, vfov_half, true, true),
    color_,
    opacity_,
    true,
    2.0
  ));
}

void SensorFrustumActor::create_lidar2d_frustum(const core::sensors::Lidar2D& config)
{
  const double min_range = static_cast<double>(config.base.min_range);
  const double max_range = static_cast<double>(config.base.max_range);
  const double angle_min = -config.fov / 2.0;
  const double angle_max = config.fov / 2.0;
  const int    num_rays  = 32;

  if (min_range > 1e-6) {
    add_actor(make_polydata_actor(
      build_lidar2d_polydata(min_range, angle_min, angle_max, num_rays, false, false),
      color_,
      std::min(0.95, opacity_ + 0.35),
      true,
      2.5
    ));
  }

  add_actor(make_polydata_actor(
    build_lidar2d_polydata(max_range, angle_min, angle_max, num_rays, true, true),
    color_,
    opacity_,
    true,
    2.0
  ));
}

void SensorFrustumActor::create_lidar3d_frustum(const core::sensors::Lidar3D& config)
{
  const double min_range = static_cast<double>(config.base.min_range);
  const double max_range = static_cast<double>(config.base.max_range);
  const double vfov_half = config.vfov / 2.0;
  const int    num_azimuth = 32;
  const int    num_vertical = 8;

  if (min_range > 1e-6) {
    add_actor(make_polydata_actor(
      build_lidar3d_polydata(min_range, vfov_half, num_azimuth, num_vertical, false),
      color_,
      std::min(0.95, opacity_ + 0.35),
      true,
      2.0
    ));
  }

  add_actor(make_polydata_actor(
    build_lidar3d_polydata(max_range, vfov_half, num_azimuth, num_vertical, true),
    color_,
    opacity_,
    true,
    1.5
  ));
}

void SensorFrustumActor::update_pose(const core::math::Pose2& robot_pose)
{
  if (actors_.empty()) {
    return;
  }

  auto transform = vtkSmartPointer<vtkTransform>::New();
  transform->Identity();
  transform->Translate(robot_pose.pos.x, robot_pose.pos.y, 0.0);
  transform->RotateZ(robot_pose.yaw * 180.0 / M_PI);
  transform->Translate(
    sensor_pose_rel_base_.t.x, sensor_pose_rel_base_.t.y, sensor_pose_rel_base_.t.z
  );

  const auto& q = sensor_pose_rel_base_.q;
  const auto  x_axis = q.rotate(core::math::Vec3{1, 0, 0});
  const auto  y_axis = q.rotate(core::math::Vec3{0, 1, 0});
  const auto  z_axis = q.rotate(core::math::Vec3{0, 0, 1});

  auto mat = vtkSmartPointer<vtkMatrix4x4>::New();
  mat->Identity();
  mat->SetElement(0, 0, x_axis.x);
  mat->SetElement(0, 1, y_axis.x);
  mat->SetElement(0, 2, z_axis.x);
  mat->SetElement(1, 0, x_axis.y);
  mat->SetElement(1, 1, y_axis.y);
  mat->SetElement(1, 2, z_axis.y);
  mat->SetElement(2, 0, x_axis.z);
  mat->SetElement(2, 1, y_axis.z);
  mat->SetElement(2, 2, z_axis.z);
  transform->Concatenate(mat);

  for (auto& actor : actors_) {
    if (actor != nullptr) {
      actor->SetUserTransform(transform);
    }
  }
}

void SensorFrustumActor::add_to_renderer(vtkRenderer& renderer)
{
  for (auto& actor : actors_) {
    if (actor != nullptr) {
      renderer.AddActor(actor);
    }
  }
}

void SensorFrustumActor::remove_from_renderer(vtkRenderer& renderer)
{
  for (auto& actor : actors_) {
    if (actor != nullptr) {
      renderer.RemoveActor(actor);
    }
  }
}

void SensorFrustumActor::set_visible(bool visible)
{
  for (auto& actor : actors_) {
    if (actor != nullptr) {
      actor->SetVisibility(visible ? 1 : 0);
    }
  }
}

} // namespace ddrl::visualizer::widgets
