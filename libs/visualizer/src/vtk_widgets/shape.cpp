#include "visualizer/widgets/shape.hpp"

#include <cmath>
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkVertexGlyphFilter.h>
#include "visualizer/widgets/utils.hpp"

namespace ddrl::visualizer::widgets {

using core::geom::Box2;
using core::geom::Circle;
using core::geom::Polygon2;
using core::geom::Shape2p5;
using core::math::Point2;
using core::math::Pose2;

class ShapeActor::Impl
{
public:
  Impl(const Shape2p5& shape,
    const std::array<double, 3>& color, double opacity);

  void                    set_transform(const Pose2& pose);
  void                    set_color(const std::array<double, 3>& color);
  void                    set_opacity(double opacity);
  void                    set_visible(bool visible);
  [[nodiscard]] vtkActor* actor() const { return actor_; }

private:
  vtkSmartPointer<vtkPolyData>       polydata_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_;
  vtkSmartPointer<vtkActor>          actor_;
};

ShapeActor::Impl::Impl(const Shape2p5& shape,
  const std::array<double, 3>& color, double opacity)
{
  // Create geometry
  polydata_ = utils::create_shape_polydata(shape);

  // Create mapper
  mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper_->SetInputData(polydata_);

  // Create actor
  actor_ = vtkSmartPointer<vtkActor>::New();
  actor_->SetMapper(mapper_);
  actor_->GetProperty()->SetColor(color[0], color[1], color[2]);
  actor_->GetProperty()->SetOpacity(opacity);

  // // Apply transform
  // auto transform = vtkSmartPointer<vtkTransform>::New();
  // transform->Identity();
  // transform->Translate(pose.pos.x, pose.pos.y, 0.0);
  // transform->RotateZ(pose.yaw * 180.0 / std::numbers::pi); // Convert to degrees
  // actor_->SetUserTransform(transform);
}

void ShapeActor::Impl::set_transform(const Pose2& pose)
{
  auto transform = vtkSmartPointer<vtkTransform>::New();
  transform->Identity();
  transform->Translate(pose.pos.x, pose.pos.y, 0.0);
  transform->RotateZ(pose.yaw * 180.0 / std::numbers::pi);
  actor_->SetUserTransform(transform);
}

void ShapeActor::Impl::set_color(const std::array<double, 3>& color)
{
  actor_->GetProperty()->SetColor(color[0], color[1], color[2]);
}

void ShapeActor::Impl::set_opacity(double opacity)
{
  actor_->GetProperty()->SetOpacity(opacity);
}

void ShapeActor::Impl::set_visible(bool visible)
{
  actor_->SetVisibility(visible ? 1 : 0);
}

// ============================================================================
// ShapeActor Public Interface
// ============================================================================

ShapeActor::ShapeActor(const Shape2p5& shape,
  const std::array<double, 3>& color, double opacity)
    : impl_(std::make_unique<Impl>(shape, color, opacity))
{
}

ShapeActor::~ShapeActor() = default;

ShapeActor::ShapeActor(ShapeActor&&) noexcept            = default;
ShapeActor& ShapeActor::operator=(ShapeActor&&) noexcept = default;

void ShapeActor::set_transform(const Pose2& pose)
{
  impl_->set_transform(pose);
}

void ShapeActor::set_color(const std::array<double, 3>& color)
{
  impl_->set_color(color);
}

void ShapeActor::set_opacity(double opacity)
{
  impl_->set_opacity(opacity);
}

void ShapeActor::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

void ShapeActor::add_to_renderer(vtkRenderer& renderer)
{
  renderer.AddActor(impl_->actor());
}

void ShapeActor::remove_from_renderer(vtkRenderer& renderer)
{
  renderer.RemoveActor(impl_->actor());
}

} // namespace ddrl::visualizer::widgets
