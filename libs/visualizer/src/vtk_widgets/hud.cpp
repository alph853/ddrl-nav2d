/**
 * @file hud.cpp
 * @brief Implementation of HUD widget
 */

#include "visualizer/widgets/hud.hpp"

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include "logging/logging.hpp"

namespace ddrl::visualizer::widgets {

using core::rl::RLMetadata;

// ============================================================================
// HUDWidget::Impl
// ============================================================================

class HUDWidget::Impl
{
public:
  Impl(size_t num_sims);

  void               add_to_renderer(vtkRenderer* renderer);
  void               remove_from_renderer(vtkRenderer* renderer);
  void               toggle_visibility();
  void               set_visible(bool visible);
  [[nodiscard]] bool is_visible() const { return visible_; }
  void               toggle_reward_terms_visibility();
  void               set_reward_terms_visible(bool visible);
  [[nodiscard]] bool is_reward_terms_visible() const { return reward_terms_visible_; }

  void update(double seconds);
  void update_view_mode(ViewMode mode);
  void update_sim_id(uint32_t sim_id);
  void update_window_size(int width, int height);
  void update_rl_info(uint32_t sim_id, const RLMetadata& metadata);

private:
  void create_text_actors();
  void update_shortcut_guides();
  void update_top_left_info();

  static std::string format_elapsed_time(double seconds);

  // Text actors for different HUD elements
  vtkSmartPointer<vtkTextActor> shortcuts_actor_;     // Left bottom
  vtkSmartPointer<vtkTextActor> elapsed_time_actor_;  // Right top
  vtkSmartPointer<vtkTextActor> top_left_info_actor_; // Left top
  vtkSmartPointer<vtkTextActor> reward_terms_actor_;  // Right bottom

  // State
  bool     visible_{true};
  bool     reward_terms_visible_{true};
  ViewMode current_mode_{ViewMode::FOCUSED};
  double   elapsed_time_{0.0};
  uint32_t current_sim_id_{0};
  size_t   total_sims_{1};

  // Episode info (bottom right)
  std::vector<RLMetadata> rl_metadata_; // Support up to 20 sims
  std::vector<std::string> reward_term_display_order_;
  std::mutex rl_metadata_mutex_;
};

HUDWidget::Impl::Impl(size_t num_sims)
{
  total_sims_ = num_sims;
  rl_metadata_.resize(num_sims);

  create_text_actors();
  update_shortcut_guides();
  update_top_left_info();
}

void HUDWidget::Impl::create_text_actors()
{
  // Create shortcut guides actor (left bottom)
  shortcuts_actor_ = vtkSmartPointer<vtkTextActor>::New();
  shortcuts_actor_->GetTextProperty()->SetFontSize(16);
  shortcuts_actor_->GetTextProperty()->SetColor(0.9, 0.9, 0.9);
  shortcuts_actor_->GetTextProperty()->SetFontFamilyToCourier();
  shortcuts_actor_->GetTextProperty()->SetBold(0);
  shortcuts_actor_->GetTextProperty()->SetShadow(1);
  shortcuts_actor_->GetTextProperty()->SetBackgroundColor(0.0, 0.0, 0.0);
  shortcuts_actor_->GetTextProperty()->SetBackgroundOpacity(0.0);

  // Create elapsed time actor (right top)
  elapsed_time_actor_ = vtkSmartPointer<vtkTextActor>::New();
  elapsed_time_actor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  elapsed_time_actor_->GetTextProperty()->SetFontSize(16);
  elapsed_time_actor_->GetTextProperty()->SetColor(0.2, 1.0, 0.2);
  elapsed_time_actor_->GetTextProperty()->SetFontFamilyToCourier();
  elapsed_time_actor_->GetTextProperty()->SetBold(1);
  elapsed_time_actor_->GetTextProperty()->SetShadow(1);
  elapsed_time_actor_->GetTextProperty()->SetJustificationToRight();
  elapsed_time_actor_->GetTextProperty()->SetVerticalJustificationToTop();

  // Create top-left info actor (left top)
  top_left_info_actor_ = vtkSmartPointer<vtkTextActor>::New();
  top_left_info_actor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  top_left_info_actor_->GetTextProperty()->SetFontSize(18);
  top_left_info_actor_->GetTextProperty()->SetColor(0.3, 0.7, 1.0);
  top_left_info_actor_->GetTextProperty()->SetFontFamilyToCourier();
  top_left_info_actor_->GetTextProperty()->SetBold(1);
  top_left_info_actor_->GetTextProperty()->SetShadow(1);
  top_left_info_actor_->GetTextProperty()->SetJustificationToLeft();
  top_left_info_actor_->GetTextProperty()->SetVerticalJustificationToTop();

  // Create reward terms actor (right bottom)
  reward_terms_actor_ = vtkSmartPointer<vtkTextActor>::New();
  reward_terms_actor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  reward_terms_actor_->GetTextProperty()->SetFontSize(16);
  reward_terms_actor_->GetTextProperty()->SetColor(1.0, 0.9, 0.4);
  reward_terms_actor_->GetTextProperty()->SetFontFamilyToCourier();
  reward_terms_actor_->GetTextProperty()->SetBold(0);
  reward_terms_actor_->GetTextProperty()->SetShadow(1);
  reward_terms_actor_->GetTextProperty()->SetJustificationToRight();
  reward_terms_actor_->GetTextProperty()->SetVerticalJustificationToBottom();
}

void HUDWidget::Impl::update_shortcut_guides()
{
  std::ostringstream oss;

  if (current_mode_ == ViewMode::FOCUSED) {
    oss << std::string(20, '=') << "\n"
        << "=== FOCUSED MODE ===\n"
        << std::string(20, '=') << "\n"
        << "View Controls:\n"
        << "  Ctrl+R          Reset camera view\n"
        << "  Ctrl+F          Toggle camera following\n"
        << "\n"
        << "Navigation:\n"
        << "  Ctrl+Tab        Next viewport\n"
        << "  Ctrl+Shift+Tab  Previous viewport\n"
        << "\n"
        << "Mode Switch:\n"
        << "  Ctrl+A          View ALL INSTANCES\n"
        << "\n"
        << "Display:\n"
        << "  H               Toggle HUD\n"
        << "  T               Toggle reward terms\n"
        << "  P               Toggle route view\n"
        << "  C               Switch cloud mode\n"
        << "  S               Toggle sensor frustums";
  } else if (current_mode_ == ViewMode::ALL_VIEWPORTS) {
    oss << std::string(21, '=') << "\n"
        << "=== VIEW ALL MODE ===\n"
        << std::string(21, '=') << "\n"
        << "Camera:\n"
        << "  Auto-following all robots\n"
        << "\n"
        << "Interaction:\n"
        << "  Left Click  Jump to viewport\n"
        << "\n"
        << "Display:\n"
        << "  H           Toggle HUD\n"
        << "  T           Toggle reward terms\n"
        << "  P           Toggle route view\n"
        << "  C           Switch cloud mode\n"
        << "  S           Toggle sensor frustums";
  } else {
    oss << "Error: Unknown view mode!";
  }

  shortcuts_actor_->SetInput(oss.str().c_str());
}

void HUDWidget::Impl::update_top_left_info()
{
  std::ostringstream oss;

  if (current_mode_ == ViewMode::FOCUSED) {
    RLMetadata               metadata;
    std::vector<std::string> reward_term_display_order;
    {
      std::lock_guard lock(rl_metadata_mutex_);
      if (current_sim_id_ < rl_metadata_.size()) {
        metadata = rl_metadata_[current_sim_id_];
      }
      reward_term_display_order = reward_term_display_order_;
    }
    oss << "Sim ID: " << current_sim_id_ << '\n'
        << "  Episode: " << metadata.episode_index << '\n'
        << "  Steps: " << metadata.episode_step_count << '\n'
        << "  Step Reward: " << std::fixed << std::setprecision(2)
        << metadata.step_reward << '\n'
        << "  Acc. Reward: " << std::fixed << std::setprecision(2)
        << metadata.episode_return;

    std::ostringstream reward_oss;
    reward_oss << "Reward Terms\n";
    reward_oss << std::string(12, '=') << '\n';
    if (reward_term_display_order.empty() && metadata.reward_terms.empty()) {
      reward_oss << "  (waiting for terms)";
    } else {
      for (std::size_t i = 0; i < reward_term_display_order.size(); ++i) {
        const auto& name = reward_term_display_order[i];
        core::rl::RewardTermValue term{};
        if (const auto it = metadata.reward_terms.find(name); it != metadata.reward_terms.end()) {
          term = it->second;
        }
        reward_oss << std::left << std::setw(22) << name << " ";
        reward_oss << "raw=" << std::right << std::setw(8) << std::fixed << std::setprecision(3)
                   << term.raw << " ";
        reward_oss << "w=" << std::setw(8) << std::fixed << std::setprecision(3)
                   << term.weighted;
        if (i + 1 < reward_term_display_order.size()) {
          reward_oss << '\n';
        }
      }
    }
    reward_terms_actor_->SetInput(reward_oss.str().c_str());
  } else {
    oss << "Total Sims: " << total_sims_;
    reward_terms_actor_->SetInput("");
  }

  top_left_info_actor_->SetInput(oss.str().c_str());
}

std::string HUDWidget::Impl::format_elapsed_time(double seconds)
{
  int hours   = static_cast<int>(seconds) / 3600;
  int minutes = (static_cast<int>(seconds) % 3600) / 60;
  int secs    = static_cast<int>(seconds) % 60;
  int millis  = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);

  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << hours << ":" << std::setw(2) << minutes << ":"
      << std::setw(2) << secs << "." << std::setw(3) << millis;

  return oss.str();
}

void HUDWidget::Impl::add_to_renderer(vtkRenderer* renderer)
{
  if (renderer == nullptr) {
    return;
  }

  auto* size = renderer->GetRenderWindow()->GetSize();
  this->update_window_size(size[0], size[1]);

  renderer->AddActor2D(shortcuts_actor_);
  renderer->AddActor2D(elapsed_time_actor_);
  renderer->AddActor2D(top_left_info_actor_);
  renderer->AddActor2D(reward_terms_actor_);
}

void HUDWidget::Impl::update_window_size(int width, int height)
{
  DDRL_LOG_DEBUG(
    logging::get_logger("HUD"),
    "HUDWidget::Impl::update_window_size: width={}, height={}",
    width,
    height
  );
  shortcuts_actor_->SetDisplayPosition(15, 15);
  elapsed_time_actor_->SetDisplayPosition(width - 30, height - 30);
  top_left_info_actor_->SetDisplayPosition(15, height - 30);
  reward_terms_actor_->SetDisplayPosition(width - 30, 20);
}

void HUDWidget::Impl::update_rl_info(uint32_t sim_id, const RLMetadata& metadata)
{
  std::lock_guard lock(rl_metadata_mutex_);
  if (sim_id >= rl_metadata_.size()) {
    DDRL_LOG_WARN(
      logging::get_logger("HUD"),
      "Sim ID {} out of bounds for RL metadata (size {})",
      sim_id,
      rl_metadata_.size()
    );
    return;
  }
  rl_metadata_[sim_id] = metadata;
  for (const auto& [name, _] : metadata.reward_terms) {
    const bool seen =
      std::find(reward_term_display_order_.begin(), reward_term_display_order_.end(), name) !=
      reward_term_display_order_.end();
    if (!seen) {
      reward_term_display_order_.push_back(name);
    }
  }
}

void HUDWidget::Impl::remove_from_renderer(vtkRenderer* renderer)
{
  if (renderer == nullptr) {
    return;
  }

  renderer->RemoveActor(shortcuts_actor_);
  renderer->RemoveActor(elapsed_time_actor_);
  renderer->RemoveActor(top_left_info_actor_);
  renderer->RemoveActor(reward_terms_actor_);
}

void HUDWidget::Impl::toggle_visibility()
{
  visible_ = !visible_;
  set_visible(visible_);
}

void HUDWidget::Impl::set_visible(bool visible)
{
  visible_ = visible;
  shortcuts_actor_->SetVisibility(visible ? 1 : 0);
  elapsed_time_actor_->SetVisibility(visible ? 1 : 0);
  top_left_info_actor_->SetVisibility(visible ? 1 : 0);
  reward_terms_actor_->SetVisibility((visible && reward_terms_visible_) ? 1 : 0);
}

void HUDWidget::Impl::toggle_reward_terms_visibility()
{
  set_reward_terms_visible(!reward_terms_visible_);
}

void HUDWidget::Impl::set_reward_terms_visible(bool visible)
{
  reward_terms_visible_ = visible;
  reward_terms_actor_->SetVisibility((visible_ && reward_terms_visible_) ? 1 : 0);
}

void HUDWidget::Impl::update(double seconds)
{
  elapsed_time_ = seconds;
  elapsed_time_actor_->SetInput(format_elapsed_time(seconds).c_str());

  update_top_left_info();
}

void HUDWidget::Impl::update_view_mode(ViewMode mode)
{
  if (current_mode_ != mode) {
    current_mode_ = mode;
    update_shortcut_guides();
    update_top_left_info();
  }
}

void HUDWidget::Impl::update_sim_id(uint32_t sim_id)
{
  if (current_sim_id_ != sim_id) {
    current_sim_id_ = sim_id;
    update_top_left_info();
  }
}

// ============================================================================
// HUDWidget public interface
// ============================================================================

HUDWidget::HUDWidget(size_t num_sims) : impl_(std::make_unique<Impl>(num_sims))
{
}

HUDWidget::~HUDWidget() = default;

HUDWidget::HUDWidget(HUDWidget&&) noexcept = default;

HUDWidget& HUDWidget::operator=(HUDWidget&&) noexcept = default;

void HUDWidget::add_to_renderer(vtkRenderer* renderer)
{
  impl_->add_to_renderer(renderer);
}

void HUDWidget::remove_from_renderer(vtkRenderer* renderer)
{
  impl_->remove_from_renderer(renderer);
}

void HUDWidget::toggle_visibility()
{
  impl_->toggle_visibility();
}

void HUDWidget::set_visible(bool visible)
{
  impl_->set_visible(visible);
}

bool HUDWidget::is_visible() const
{
  return impl_->is_visible();
}

void HUDWidget::toggle_reward_terms_visibility()
{
  impl_->toggle_reward_terms_visibility();
}

void HUDWidget::set_reward_terms_visible(bool visible)
{
  impl_->set_reward_terms_visible(visible);
}

bool HUDWidget::is_reward_terms_visible() const
{
  return impl_->is_reward_terms_visible();
}

void HUDWidget::update(double seconds)
{
  impl_->update(seconds);
}

void HUDWidget::update_view_mode(ViewMode mode)
{
  impl_->update_view_mode(mode);
}

void HUDWidget::update_sim_id(uint32_t sim_id)
{
  impl_->update_sim_id(sim_id);
}

void HUDWidget::update_window_size(int width, int height)
{
  impl_->update_window_size(width, height);
}

void HUDWidget::update_rl_info(uint32_t sim_id, const RLMetadata& metadata)
{
  impl_->update_rl_info(sim_id, metadata);
}

} // namespace ddrl::visualizer::widgets
