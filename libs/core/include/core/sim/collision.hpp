#pragma once

#include "core/math/pose.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ddrl::core::sim {

struct CollisionContact {
  std::string       other_name;
  core::math::Pose2 other_pose{};
  bool              other_is_dynamic{false};
};

struct CollisionSeverity {
  enum class Level : uint8_t { NONE, MINOR, MODERATE, SEVERE, CRITICAL };

  Level  severity_level{Level::NONE};
  double score{0.0};             // Normalized 0-1 continuous severity measure
  double penetration_depth{0.0}; // Maximum penetration depth (meters)
  double impact_velocity{0.0};   // Velocity component along collision normal (m/s)
  double robot_speed{0.0};       // Overall robot speed magnitude (m/s)
  size_t contact_count{0};
  bool   has_dynamic_contact{false};

  [[nodiscard]] double get_penalty() const
  {
    switch (severity_level) {
      case Level::NONE:
        return 0.0;
      case Level::MINOR:
        return -4.0;
      case Level::MODERATE:
        return -8.0;
      case Level::SEVERE:
        return -12.0;
      case Level::CRITICAL:
        return -20.0;
    }
    return 0.0;
  }

  /// Check if any collision is active (not NONE)
  [[nodiscard]] bool is_active() const { return severity_level != Level::NONE; }
};

/// Complete collision report for a physics timestep
/// Combines severity assessment with contact details
struct CollisionReport {
  double                        timestamp{0.0}; ///< Simulation time of collision
  CollisionSeverity             severity{};     ///< Assessed collision severity
  std::vector<CollisionContact> contacts;       ///< All contacts detected

  /// Check if this report represents an active collision
  [[nodiscard]] bool is_active() const { return severity.is_active(); }
};

} // namespace ddrl::core::sim

inline std::string to_string(ddrl::core::sim::CollisionSeverity::Level level)
{
  switch (level) {
    case ddrl::core::sim::CollisionSeverity::Level::NONE:
      return "NONE";
    case ddrl::core::sim::CollisionSeverity::Level::MINOR:
      return "MINOR";
    case ddrl::core::sim::CollisionSeverity::Level::MODERATE:
      return "MODERATE";
    case ddrl::core::sim::CollisionSeverity::Level::SEVERE:
      return "SEVERE";
    case ddrl::core::sim::CollisionSeverity::Level::CRITICAL:
      return "CRITICAL";
  }
  return "UNKNOWN";
}

inline std::string to_string(const ddrl::core::sim::CollisionContact& contact)
{
  return "CollisionContact{other_name='" + contact.other_name +
         "', other_is_dynamic=" + (contact.other_is_dynamic ? "true" : "false") + "}";
}

inline std::string to_string(std::span<const ddrl::core::sim::CollisionContact> contacts)
{
  std::string res = "CollisionContacts[";
  for (size_t i = 0; i < contacts.size(); ++i) {
    res += to_string(contacts[i]);
    if (i + 1 < contacts.size()) {
      res += ", ";
    }
  }
  res += "]";
  return res;
}

inline std::string to_string(const ddrl::core::sim::CollisionSeverity& severity)
{
  return "CollisionSeverity{level=" + to_string(severity.severity_level) +
         ", score=" + std::to_string(severity.score) +
         ", penetration_depth=" + std::to_string(severity.penetration_depth) +
         ", impact_velocity=" + std::to_string(severity.impact_velocity) +
         ", robot_speed=" + std::to_string(severity.robot_speed) +
         ", contact_count=" + std::to_string(severity.contact_count) +
         ", has_dynamic_contact=" + (severity.has_dynamic_contact ? "true" : "false") + "}";
}
