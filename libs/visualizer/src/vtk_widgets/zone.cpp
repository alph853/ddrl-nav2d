/**
 * @file zone.cpp
 * @brief Implementation of ZoneActor widget
 */

#include "visualizer/widgets/zone.hpp"

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

namespace ddrl::visualizer::widgets {

// ============================================================================
// ZoneActor Implementation
// ============================================================================

class ZoneActor::Impl
{
public:
  Impl(const std::string&          name,
       const core::geom::Polygon2& polygon,
       ZoneType                    zone_type,
       double                      height,
       double                      opacity);

  [[nodiscard]] vtkActor*          actor() const { return actor_; }
  [[nodiscard]] const std::string& get_name() const { return name_; }

  void set_visible(bool visible);
  void set_color(double r, double g, double b);
  void set_opacity(double opacity);

private:
  void build_zone_geometry(const core::geom::Polygon2& polygon, double height);

  std::string name_;

  vtkSmartPointer<vtkActor>          actor_;
  vtkSmartPointer<vtkPolyData>       polydata_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_;
};

ZoneActor::Impl::Impl(const std::string&          name,
                      const core::geom::Polygon2& polygon,
                      ZoneType                    zone_type,
                      double                      height,
                      double                      opacity)
    : name_(name)
{
  polydata_ = vtkSmartPointer<vtkPolyData>::New();
  mapper_   = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_    = vtkSmartPointer<vtkActor>::New();

  // Build zone geometry
  build_zone_geometry(polygon, height);

  // Setup mapper and actor
  mapper_->SetInputData(polydata_);
  actor_->SetMapper(mapper_);

  // Set color based on zone type
  auto color = ZoneActor::get_color_for_type(zone_type);
  actor_->GetProperty()->SetColor(color[0], color[1], color[2]);

  // Set visual properties
  actor_->GetProperty()->SetOpacity(opacity);
  actor_->GetProperty()->SetAmbient(0.3);
  actor_->GetProperty()->SetDiffuse(0.7);
  actor_->GetProperty()->SetSpecular(0.1);

  // Enable transparency rendering
  actor_->GetProperty()->SetInterpolationToFlat();
}

void ZoneActor::Impl::build_zone_geometry(const core::geom::Polygon2& polygon, double height)
{
  const auto& vertices = polygon.v;
  const auto  n        = vertices.size();

  if (n < 3) {
    return; // Invalid polygon
  }

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto polys  = vtkSmartPointer<vtkCellArray>::New();

  // Create bottom and top vertices
  for (const auto& v : vertices) {
    points->InsertNextPoint(v.x, v.y, 0.0); // Bottom at z=0
  }
  for (const auto& v : vertices) {
    points->InsertNextPoint(v.x, v.y, height); // Top at z=height
  }

  // Bottom face (reversed winding for correct normal pointing down)
  std::vector<vtkIdType> bottom_face(n);
  for (size_t i = 0; i < n; ++i) {
    bottom_face[n - 1 - i] = static_cast<vtkIdType>(i);
  }
  polys->InsertNextCell(static_cast<vtkIdType>(n), bottom_face.data());

  // Top face (normal pointing up)
  std::vector<vtkIdType> top_face(n);
  for (size_t i = 0; i < n; ++i) {
    top_face[i] = static_cast<vtkIdType>(n) + static_cast<vtkIdType>(i);
  }
  polys->InsertNextCell(static_cast<vtkIdType>(n), top_face.data());

  // Side faces (quads)
  for (size_t i = 0; i < n; ++i) {
    const size_t             next_i = (i + 1) % n;
    std::array<vtkIdType, 4> quad   = {
      static_cast<vtkIdType>(i),          // Bottom current
      static_cast<vtkIdType>(next_i),     // Bottom next
      static_cast<vtkIdType>(n + next_i), // Top next
      static_cast<vtkIdType>(n + i)       // Top current
    };
    polys->InsertNextCell(4, quad.data());
  }

  polydata_->SetPoints(points);
  polydata_->SetPolys(polys);
}

void ZoneActor::Impl::set_visible(bool visible)
{
  actor_->SetVisibility(visible ? 1 : 0);
}

void ZoneActor::Impl::set_color(double r, double g, double b)
{
  actor_->GetProperty()->SetColor(r, g, b);
}

void ZoneActor::Impl::set_opacity(double opacity)
{
  actor_->GetProperty()->SetOpacity(opacity);
}

// ============================================================================
// ZoneActor public interface
// ============================================================================

ZoneActor::ZoneActor(const std::string&          name,
                     const core::geom::Polygon2& polygon,
                     ZoneType                    zone_type,
                     double                      height,
                     double                      opacity)
    : impl_(std::make_unique<Impl>(name, polygon, zone_type, height, opacity))
{
}

ZoneActor::~ZoneActor() = default;

ZoneActor::ZoneActor(ZoneActor&&) noexcept            = default;
ZoneActor& ZoneActor::operator=(ZoneActor&&) noexcept = default;

void ZoneActor::add_to_renderer(vtkRenderer& renderer)
{
  renderer.AddActor(impl_->actor());
}

void ZoneActor::remove_from_renderer(vtkRenderer& renderer)
{
  renderer.RemoveActor(impl_->actor());
}

void ZoneActor::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

void ZoneActor::set_color(double r, double g, double b)
{
  impl_->set_color(r, g, b);
}

void ZoneActor::set_opacity(double opacity)
{
  impl_->set_opacity(opacity);
}

const std::string& ZoneActor::get_name() const
{
  return impl_->get_name();
}

std::array<double, 3> ZoneActor::get_color_for_type(ZoneType type)
{
  switch (type) {
    case ZoneType::START:
      return {0.2, 0.8, 0.2}; // Green - safe to start
    case ZoneType::GOAL:
      return {0.2, 0.6, 0.9}; // Blue/Cyan - target destination
    case ZoneType::NO_GO:
      return {0.9, 0.2, 0.2}; // Red - danger/forbidden
    case ZoneType::OTHER:
    default:
      return {0.5, 0.5, 0.5}; // Gray - neutral
  }
}

} // namespace ddrl::visualizer::widgets
