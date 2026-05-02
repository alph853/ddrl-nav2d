/**
 * @file grid.cpp
 * @brief Implementation of GridWidget
 */

#include "visualizer/widgets/grid.hpp"

#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

namespace ddrl::visualizer::widgets {

// ============================================================================
// GridWidget Implementation
// ============================================================================

class GridWidget::Impl
{
public:
  Impl(double size, double spacing);
  [[nodiscard]] vtkActor* actor() const { return actor_; }

  void set_size(double size);
  void set_spacing(double spacing);
  void set_visible(bool visible);
  void set_color(double r, double g, double b);

private:
  void rebuild_grid();

  double size_;
  double spacing_;

  vtkSmartPointer<vtkActor>          actor_;
  vtkSmartPointer<vtkAppendPolyData> append_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_;
};

GridWidget::Impl::Impl(double size, double spacing)
    : size_(size), spacing_(spacing)
{
  append_ = vtkSmartPointer<vtkAppendPolyData>::New();
  mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_  = vtkSmartPointer<vtkActor>::New();

  // Build initial grid
  rebuild_grid();

  // Setup mapper and actor
  mapper_->SetInputConnection(append_->GetOutputPort());
  actor_->SetMapper(mapper_);
  actor_->GetProperty()->SetColor(0.3, 0.3, 0.3);
  actor_->GetProperty()->SetLineWidth(1.0);
}

void GridWidget::Impl::rebuild_grid()
{
  append_->RemoveAllInputs();

  const int num_lines = static_cast<int>(size_ / spacing_);

  // Lines along X
  for (int i = -num_lines / 2; i <= num_lines / 2; ++i) {
    const double y = i * spacing_;

    auto line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(-size_ / 2, y, 0);
    line->SetPoint2(size_ / 2, y, 0);
    line->Update();
    append_->AddInputData(line->GetOutput());
  }

  // Lines along Y
  for (int i = -num_lines / 2; i <= num_lines / 2; ++i) {
    const double x = i * spacing_;

    auto line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(x, -size_ / 2, 0);
    line->SetPoint2(x, size_ / 2, 0);
    line->Update();
    append_->AddInputData(line->GetOutput());
  }

  append_->Update();
}

void GridWidget::Impl::set_size(double size)
{
  size_ = size;
  rebuild_grid();
}

void GridWidget::Impl::set_spacing(double spacing)
{
  spacing_ = spacing;
  rebuild_grid();
}

void GridWidget::Impl::set_visible(bool visible)
{
  actor_->SetVisibility(visible ? 1 : 0);
}

void GridWidget::Impl::set_color(double r, double g, double b)
{
  actor_->GetProperty()->SetColor(r, g, b);
}

GridWidget::GridWidget(double size, double spacing)
    : impl_(std::make_unique<Impl>(size, spacing))
{
}

GridWidget::~GridWidget() = default;

GridWidget::GridWidget(GridWidget&&) noexcept            = default;
GridWidget& GridWidget::operator=(GridWidget&&) noexcept = default;

void GridWidget::add_to_renderer(vtkRenderer& renderer)
{
  renderer.AddActor(impl_->actor());
}

void GridWidget::remove_from_renderer(vtkRenderer& renderer)
{
  renderer.RemoveActor(impl_->actor());
}

void GridWidget::set_size(double size)
{
  impl_->set_size(size);
}

void GridWidget::set_spacing(double spacing)
{
  impl_->set_spacing(spacing);
}

void GridWidget::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

void GridWidget::set_color(double r, double g, double b)
{
  impl_->set_color(r, g, b);
}

} // namespace ddrl::visualizer::widgets
