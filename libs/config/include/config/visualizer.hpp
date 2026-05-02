#pragma once

#include <cstdint>
#include <string>

namespace ddrl::config {

enum class VisualizerBackend : uint8_t {
  VTK,
  // Future: OpenGL, Vulkan, etc.
};

struct InteractorStyleCfg {
  struct Zoom {
    double max_zoom_in  = 0.8;
    double max_zoom_out = 5.0;
  };
  struct Pan {
    double max_pan = 1.5; // Max pan relative to world size
  };
  struct Rotate {
    double max_pitch = 80.0;  // degrees
    double min_pitch = -80.0; // degrees
  };
  double sensitivity = 1.0;
  Zoom   zoom;
  Pan    pan;
  Rotate rotate;
};

struct CameraFollowSettings {
  struct ChaseMode {
    double back_distance{6.0}; ///< Meters behind target
    double up_distance{3.0};   ///< Meters above target
    double focal_height{0.5};  ///< Height of focal point above ground
  };

  struct TopDownMode {
    double height{20.0}; ///< Camera height above target
  };

  struct OrbitMode {
    double radius{10.0};      ///< Orbit radius in meters
    double height{6.0};       ///< Orbit height above target
    double speed{0.6};        ///< Orbit speed in rad/s
    double focal_height{0.5}; ///< Height of focal point above ground
  };

  struct FirstPersonMode {
    double eye_height{1.5};      ///< Height of camera above robot base
    double forward_offset{0.3};  ///< Forward offset from robot center (meters)
    double focal_distance{10.0}; ///< How far ahead the camera looks (meters)
  };

  double smooth_alpha{0.15}; ///< Exponential smoothing factor (0-1)

  ChaseMode       chase;
  TopDownMode     top_down;
  OrbitMode       orbit;
  FirstPersonMode first_person;
};

/// Visualizer configuration
struct VizConfig {
  VisualizerBackend backend;

  uint32_t    window_width;
  uint32_t    window_height;
  std::string window_title;
  double      target_fps{30.0};

  InteractorStyleCfg   interactor_style;
  CameraFollowSettings camera_follow;
};

} // namespace ddrl::config
