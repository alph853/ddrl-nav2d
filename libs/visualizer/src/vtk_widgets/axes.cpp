/**
 * @file axes.cpp
 * @brief Implementation of AxesWidget
 */

#include "visualizer/widgets/axes.hpp"

#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCellData.h>
#include <vtkLineSource.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

namespace ddrl::visualizer::widgets {

// ============================================================================
// AxesWidget Implementation
// ============================================================================

class AxesWidget::Impl
{
public:
  explicit Impl(double length);

  // Accessor for the underlying actor
  [[nodiscard]] vtkActor* actor() const { return actor_; }

  // Mutators used by the outer class
  void set_visible(bool visible) { actor_->SetVisibility(visible ? 1 : 0); }
  void set_length(double length)
  {
    length_ = length;
    rebuild_axes();
  }

private:
  void rebuild_axes();

  double length_;

  vtkSmartPointer<vtkAppendPolyData> append_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_;
  vtkSmartPointer<vtkActor>          actor_;

  vtkSmartPointer<vtkLineSource> x_line_;
  vtkSmartPointer<vtkLineSource> y_line_;
  vtkSmartPointer<vtkLineSource> z_line_;
};

AxesWidget::Impl::Impl(double length) : length_(length)
{
  append_ = vtkSmartPointer<vtkAppendPolyData>::New();
  mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_  = vtkSmartPointer<vtkActor>::New();

  x_line_ = vtkSmartPointer<vtkLineSource>::New();
  y_line_ = vtkSmartPointer<vtkLineSource>::New();
  z_line_ = vtkSmartPointer<vtkLineSource>::New();

  // Build initial axes
  rebuild_axes();

  // Setup mapper and actor
  mapper_->SetInputConnection(append_->GetOutputPort());
  actor_->SetMapper(mapper_);
  actor_->GetProperty()->SetLineWidth(3.0);
}

void AxesWidget::Impl::rebuild_axes()
{
  // Clear existing geometry
  append_->RemoveAllInputs();

  // X axis (red)
  x_line_->SetPoint1(0, 0, 0);
  x_line_->SetPoint2(length_, 0, 0);
  x_line_->Update();
  append_->AddInputData(x_line_->GetOutput());

  // Y axis (green)
  y_line_->SetPoint1(0, 0, 0);
  y_line_->SetPoint2(0, length_, 0);
  y_line_->Update();
  append_->AddInputData(y_line_->GetOutput());

  // Z axis (blue)
  z_line_->SetPoint1(0, 0, 0);
  z_line_->SetPoint2(0, 0, length_);
  z_line_->Update();
  append_->AddInputData(z_line_->GetOutput());

  append_->Update();
}

AxesWidget::AxesWidget(double length) : impl_(std::make_unique<Impl>(length))
{
}

AxesWidget::~AxesWidget() = default;

AxesWidget::AxesWidget(AxesWidget&&) noexcept            = default;
AxesWidget& AxesWidget::operator=(AxesWidget&&) noexcept = default;

void AxesWidget::add_to_renderer(vtkRenderer& renderer)
{
  renderer.AddActor(impl_->actor());
}

void AxesWidget::remove_from_renderer(vtkRenderer& renderer)
{
  renderer.RemoveActor(impl_->actor());
}

void AxesWidget::set_length(double length)
{
  impl_->set_length(length);
}

void AxesWidget::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

} // namespace ddrl::visualizer::widgets
