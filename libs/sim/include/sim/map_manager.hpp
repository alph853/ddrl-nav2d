#pragma once

#include "core/base/result.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "config/map.hpp"
#include "config/parsing.hpp"
#include "config/world.hpp"

namespace ddrl::sim {

/**
 * @brief Map manager responsible for selecting world maps based on configured policy.
 * Thread-safe for concurrent load_map() calls across simulator threads.
 */
class MapManager
{
public:
  struct Params {
    std::string                           map_directory;
    std::shared_ptr<config::ConfigParser> parser;
    std::shared_ptr<config::WorldConfig>  world_config;
  };

  explicit MapManager(Params params);
  ~MapManager();

  MapManager(const MapManager&)            = delete;
  MapManager& operator=(const MapManager&) = delete;
  MapManager(MapManager&&) noexcept;
  MapManager& operator=(MapManager&&) noexcept;

  /// Load a map using the configured selection strategy
  std::shared_ptr<const config::MapConfig> get_map_by_name(std::string_view name);
  std::shared_ptr<const config::MapConfig> get_map_random();

  [[nodiscard]] std::span<const std::string> get_available_map_names() const;
  [[nodiscard]] std::vector<std::shared_ptr<const config::MapConfig>> get_available_maps() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::sim
