#pragma once

#include <core/kinematics/ackermann.hpp>
#include <core/math/pose.hpp>
#include <core/sensors/sensor_models.hpp>
#include <string>
#include <vector>

namespace ddrl::config {

struct RobotProfile {
  std::string name;
  std::string model_id; // reference to ModelProfile

  core::kinematics::AckermannCaps          kinematics;
  std::vector<core::sensors::SensorConfig> sensors;
};

} // namespace ddrl::config
