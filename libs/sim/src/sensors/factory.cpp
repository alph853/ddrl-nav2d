#include "sim/sensors.hpp"

namespace ddrl::sim {

using core::ErrorCode;
using core::make_error;
using core::Result;

//=============================================================================
// Factory Function
//=============================================================================

Result<SensorType> create_sensor(const core::sensors::SensorConfig& sensor_cfg,
                                 const SensorBase::Params&          params)
{
  return std::visit(
      [&params](const auto& cfg) -> Result<SensorType> {
        using T = std::decay_t<decltype(cfg)>;

        if constexpr (std::is_same_v<T, core::sensors::DepthCamera>) {
          return std::make_unique<DepthCameraSensor>(params);
        } else if constexpr (std::is_same_v<T, core::sensors::Lidar2D>) {
          return std::make_unique<Lidar2DSensor>(params);
        } else if constexpr (std::is_same_v<T, core::sensors::Lidar3D>) {
          return std::make_unique<Lidar3DSensor>(params);
        } else {
          return std::unexpected(
              make_error(ErrorCode::SIM, "Unknown sensor type", "create_sensor"));
        }
      },
      sensor_cfg);
}

} // namespace ddrl::sim
