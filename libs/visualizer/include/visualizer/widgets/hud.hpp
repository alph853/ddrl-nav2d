/**
 * @file hud.hpp
 * @brief Widget for creating heads-up display overlay
 *
 * This widget creates a HUD overlay with information and shortcuts.
 *
 * @note This widget depends on VTK (Visualization Toolkit).
 *       Requires: vtkTextActor, vtkTextProperty, vtkSmartPointer
 */

#pragma once

#include <memory>

#include "core/rl/rl_metadata.hpp"
#include "visualizer/models.hpp"

// Forward declarations
class vtkRenderer;
class vtkTextActor;

namespace ddrl::visualizer::widgets {

/**
 * @brief Widget for rendering heads-up display (HUD) overlay
 */
class HUDWidget
{
public:
  /**
   * @brief Construct HUD widget
   */
  HUDWidget(size_t num_sims);

  ~HUDWidget();

  // Non-copyable, movable
  HUDWidget(const HUDWidget&)            = delete;
  HUDWidget& operator=(const HUDWidget&) = delete;
  HUDWidget(HUDWidget&&) noexcept;
  HUDWidget& operator=(HUDWidget&&) noexcept;

  /**
   * @brief Add HUD actors to renderer
   * @param renderer VTK renderer to add actors to
   */
  void add_to_renderer(vtkRenderer* renderer);

  /**
   * @brief Remove HUD actors from renderer
   * @param renderer VTK renderer to remove actors from
   */
  void remove_from_renderer(vtkRenderer* renderer);

  /**
   * @brief Toggle HUD visibility
   */
  void toggle_visibility();

  /**
   * @brief Set HUD visibility
   */
  void set_visible(bool visible);

  /**
   * @brief Check if HUD is visible
   */
  [[nodiscard]] bool is_visible() const;

  /**
   * @brief Toggle reward breakdown panel visibility
   */
  void toggle_reward_terms_visibility();

  /**
   * @brief Set reward breakdown panel visibility
   */
  void set_reward_terms_visible(bool visible);

  /**
   * @brief Check if reward breakdown panel is visible
   */
  [[nodiscard]] bool is_reward_terms_visible() const;

  /**
   * @brief Update elapsed time display
   */
  void update(double seconds);

  /**
   * @brief Update current view mode
   */
  void update_view_mode(ViewMode mode);

  /**
   * @brief Update focused simulation ID
   */
  void update_sim_id(uint32_t sim_id);

  /**
   * @brief Update HUD layout based on window size
   */
  void update_window_size(int width, int height);

  /**
   * @brief Set the HUD RL info display
   */
  void update_rl_info(uint32_t sim_id, const core::rl::RLMetadata& metadata);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::visualizer::widgets
