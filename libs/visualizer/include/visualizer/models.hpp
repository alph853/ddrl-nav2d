/**
 * @file view_mode.hpp
 * @brief Core view mode enumeration for visualizer module
 *
 * This header defines the centralized ViewMode enum used across all
 * visualizer components (Viewer, HUDWidget, InteractiveStyle, etc.)
 */

#pragma once

#include <cstdint>

namespace ddrl::visualizer {

/**
 * @brief View mode for the visualizer
 */
enum class ViewMode : uint8_t {
  FOCUSED,      ///< Focused mode: single viewport, manual camera control
  ALL_VIEWPORTS ///< All viewports mode: grid layout, automatic camera following
};

enum class CameraFollowMode : uint8_t {
  MANUAL,      ///< Manual camera control by user
  CHASE,       ///< Camera view following robot
  TOPDOWN,     ///< Top-down fixed camera view
  ORBIT,       ///< Orbiting camera view around robot
  FIRSTPERSON, ///< First-person view from robot's perspective
  COUNT,
};

} // namespace ddrl::visualizer
