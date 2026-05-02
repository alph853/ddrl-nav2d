#include "core/base/variant_overload.hpp"
#include "core/geom/shape.hpp"
#include "core/math/pose.hpp"
#include <array>
#include <cmath>
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkVertexGlyphFilter.h>

namespace ddrl::visualizer::widgets::utils {

using core::geom::Box2;
using core::geom::Circle;
using core::geom::Polygon2;
using core::geom::Shape2p5;
using core::math::Point2;
using core::math::Pose2;

inline vtkSmartPointer<vtkPolyData> create_extruded_mesh(
  const std::vector<Point2>& base_vertices, double z_min, double z_max)
{
  const auto n = base_vertices.size();
  if (n < 3) {
    return vtkSmartPointer<vtkPolyData>::New(); // Empty mesh
  }

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto polys  = vtkSmartPointer<vtkCellArray>::New();

  // Create bottom and top vertices
  for (const auto& v : base_vertices) {
    points->InsertNextPoint(v.x, v.y, z_min); // Bottom
  }
  for (const auto& v : base_vertices) {
    points->InsertNextPoint(v.x, v.y, z_max); // Top
  }

  // Bottom face (reversed winding for correct normal)
  std::vector<vtkIdType> bottom_face(n);
  for (size_t i = 0; i < n; ++i) {
    bottom_face[n - 1 - i] = static_cast<vtkIdType>(i);
  }
  polys->InsertNextCell(static_cast<vtkIdType>(n), bottom_face.data());

  // Top face
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

  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->SetPolys(polys);

  return polydata;
}

// Helper: Create polydata from shape
inline vtkSmartPointer<vtkPolyData> create_shape_polydata(const Shape2p5& shape)
{
  std::vector<Point2> vertices = std::visit(
    core::Overloaded{[](const Polygon2& poly) { return poly.v; },
      [](const Circle& circle) {
        std::vector<Point2> out;
        constexpr int       kResolution = 32;
        out.reserve(kResolution);

        for (int i = 0; i < kResolution; ++i) {
          const double angle = 2.0 * std::numbers::pi * i / kResolution;
          out.emplace_back(
            circle.radius * std::cos(angle), circle.radius * std::sin(angle));
        }
        return out;
      },
      [](const Box2& box) {
        std::vector<Point2> out;

        out = {
          {-box.hx, -box.hy},
          {box.hx, -box.hy},
          {box.hx, box.hy},
          {-box.hx, box.hy},
        };
        return out;
      }},
    shape.base.value);

  return create_extruded_mesh(vertices, shape.z_min, shape.z_max);
}

} // namespace ddrl::visualizer::widgets::utils
