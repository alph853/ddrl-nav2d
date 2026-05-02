#pragma once

#include <core/math/pose.hpp>
#include <cstdint>
#include <string>
#include <variant>

namespace ddrl::core::sensors {

/// Base configuration shared by all sensor types
struct SensorBase {
  std::string id;            // Unique sensor identifier
  std::string frame_id;      // Sensor frame identifier for TF tree
  math::Pose3 pose_rel_base; // Transform from robot base to sensor
  float       min_range;     // Minimum detection range (m)
  float       max_range;     // Maximum detection range (m)
  float       noise_std;     // Range noise standard deviation (m)
  float       dropout_prob;  // Per-ray dropout probability [0,1]
  float       update_hz;     // Sensor update frequency (Hz)
  float       latency_s;     // Processing latency (s)
};

/// Depth camera configuration with pinhole camera model
struct DepthCamera {
  SensorBase base;

  // Intrinsic parameters
  float    fx, fy; // Focal lengths (pixels)
  float    cx, cy; // Principal point (pixels)
  uint16_t width;  // Image width (pixels)
  uint16_t height; // Image height (pixels)

  double near_z; // Near clipping plane (m)
  double far_z;  // Far clipping plane (m)
};

/// 2D planar lidar configuration
struct Lidar2D {
  SensorBase base;

  double fov;         // Field of view (rad)
  double angular_res; // Angular resolution (rad/step)
};

/// 3D multi-ring lidar configuration
struct Lidar3D {
  SensorBase base;

  double hfov;        // Horizontal FOV (rad, typically 2π)
  double angular_res; // Azimuth angular resolution (rad/step)
  double vfov;        // Vertical FOV (rad)
  size_t num_rings;   // Number of vertical channels/rings
};

/// Variant type holding any sensor configuration
using SensorConfig = std::variant<DepthCamera, Lidar2D, Lidar3D>;

} // namespace ddrl::core::sensors
