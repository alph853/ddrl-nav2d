/**
 * @file vtk_widgets.cpp
 * @brief Implementation of VTK widget classes
 */

#include "visualizer/widgets/point_cloud.hpp"

#include "core/math/primitives.hpp"
#include <cmath>
#include <vtkActor.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkVertexGlyphFilter.h>

namespace ddrl::visualizer::widgets {

using core::math::Point3;

class PointCloudActor::Impl
{
public:
  Impl(std::span<const Point3> points, const std::array<double, 3>& color, double point_size);

  void                    update_points(std::span<const Point3> points);
  void                    set_color(const std::array<double, 3>& color);
  void                    set_point_size(double size);
  void                    set_visible(bool visible);
  [[nodiscard]] vtkActor* actor() const { return actor_; }

private:
  vtkSmartPointer<vtkPoints>            points_;
  vtkSmartPointer<vtkPolyData>          polydata_;
  vtkSmartPointer<vtkVertexGlyphFilter> vertex_filter_;
  vtkSmartPointer<vtkPolyDataMapper>    mapper_;
  vtkSmartPointer<vtkActor>             actor_;
};

PointCloudActor::Impl::Impl(
  std::span<const Point3> points, const std::array<double, 3>& color, double point_size
)
{
  // Points as float32 + one-time allocation + in-place fill
  points_ = vtkSmartPointer<vtkPoints>::New();
  points_->SetDataTypeToFloat();
  points_->SetNumberOfPoints(static_cast<vtkIdType>(points.size()));

  {
    auto*  arr = vtkFloatArray::SafeDownCast(points_->GetData());
    float* xyz = arr->WritePointer(0, 3 * static_cast<vtkIdType>(points.size()));
    for (vtkIdType i = 0; i < static_cast<vtkIdType>(points.size()); ++i) {
      const auto&     p = points[size_t(i)];
      const vtkIdType b = 3 * i;
      xyz[b + 0]        = static_cast<float>(p.x);
      xyz[b + 1]        = static_cast<float>(p.y);
      xyz[b + 2]        = static_cast<float>(p.z);
    }
    // Mark only the array as modified
    arr->Modified();
  }

  polydata_ = vtkSmartPointer<vtkPolyData>::New();
  polydata_->SetPoints(points_);

  // Vertex glyph filter (kept, but no per-frame Update() calls)
  vertex_filter_ = vtkSmartPointer<vtkVertexGlyphFilter>::New();
  vertex_filter_->SetInputData(polydata_);
  vertex_filter_->ReleaseDataFlagOn(); // don’t keep old outputs around
  vertex_filter_->Update();            // one-time to prime the pipeline

  mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper_->SetInputConnection(vertex_filter_->GetOutputPort());

  actor_ = vtkSmartPointer<vtkActor>::New();
  actor_->SetMapper(mapper_);
  actor_->GetProperty()->SetColor(color[0], color[1], color[2]);
  actor_->GetProperty()->SetPointSize(static_cast<float>(point_size));
#if VTK_MAJOR_VERSION >= 9
  actor_->GetProperty()->SetRenderPointsAsSpheres(true); // optional, still GL points
#endif
}

void PointCloudActor::Impl::update_points(std::span<const Point3> points)
{
  const auto num_pts = static_cast<vtkIdType>(points.size());
  points_->SetNumberOfPoints(num_pts);

  auto*  arr = vtkFloatArray::SafeDownCast(points_->GetData());
  float* xyz = arr->WritePointer(0, 3 * num_pts); // ensures capacity for 3*num_pts values

  for (vtkIdType i = 0; i < num_pts; ++i) {
    const auto&     p = points[size_t(i)];
    const vtkIdType b = 3 * i;
    xyz[b + 0]        = static_cast<float>(p.x);
    xyz[b + 1]        = static_cast<float>(p.y);
    xyz[b + 2]        = static_cast<float>(p.z);
  }

  arr->Modified();
}

void PointCloudActor::Impl::set_color(const std::array<double, 3>& color)
{
  actor_->GetProperty()->SetColor(color[0], color[1], color[2]);
}

void PointCloudActor::Impl::set_point_size(double size)
{
  actor_->GetProperty()->SetPointSize(static_cast<float>(size));
}

void PointCloudActor::Impl::set_visible(bool visible)
{
  actor_->SetVisibility(visible ? 1 : 0);
}

PointCloudActor::PointCloudActor(
  std::span<const Point3> points, const std::array<double, 3>& color, double point_size
)
    : impl_(std::make_unique<Impl>(points, color, point_size))
{
}

PointCloudActor::~PointCloudActor() = default;

PointCloudActor::PointCloudActor(PointCloudActor&&) noexcept            = default;
PointCloudActor& PointCloudActor::operator=(PointCloudActor&&) noexcept = default;

void PointCloudActor::update_points(std::span<const Point3> points)
{
  impl_->update_points(points);
}

void PointCloudActor::set_color(const std::array<double, 3>& color)
{
  impl_->set_color(color);
}

void PointCloudActor::set_point_size(double size)
{
  impl_->set_point_size(size);
}

void PointCloudActor::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

void PointCloudActor::add_to_renderer(vtkRenderer& renderer)
{
  renderer.AddActor(impl_->actor());
}

void PointCloudActor::remove_from_renderer(vtkRenderer& renderer)
{
  renderer.RemoveActor(impl_->actor());
}

} // namespace ddrl::visualizer::widgets
