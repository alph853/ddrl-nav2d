/**
 * @file replay_provider.hpp
 * @brief File replay provider for recorded simulations
 */

#pragma once

#include "visualizer/interface/base_provider.hpp"
#include <cstdint>
#include <string>

namespace ddrl::visualizer {

/**
 * @brief File replay provider (future implementation)
 *
 * Could replay recorded simulation sessions from disk.
 * Left as placeholder for future development.
 */
class FileReplayProvider : public IVisualizerProvider
{
public:
  /**
   * @brief Construct replay provider from file path
   * @param file_path Path to recorded simulation file
   */
  explicit FileReplayProvider(std::string file_path);
  ~FileReplayProvider() override;

  core::Result<core::sim::WorldStateSnapshot> get_world_state(uint32_t sim_id) override;

  [[nodiscard]] bool is_active() const override;

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace ddrl::visualizer
