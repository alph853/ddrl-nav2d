#include "sim/map_manager.hpp"

#include "core/geom/shape.hpp"
#include "core/math/pose.hpp"
#include <filesystem>
#include <format>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "logging/logging.hpp"

namespace ddrl::sim {

using config::MapConfig;
using config::ModelProfile;
using config::StaticInstanceCfg;
using core::Error;
using core::ErrorCode;
using core::make_error;
using core::Result;

namespace fs = std::filesystem;

class MapManager::Impl
{
private:
  mutable std::mutex mutex_;

  fs::path                              map_directory_;
  std::shared_ptr<config::ConfigParser> parser_;
  std::shared_ptr<config::WorldConfig>  world_config_;

  std::shared_ptr<logging::Logger> logger_;

  std::vector<std::string>                                    available_map_names_;
  std::unordered_map<std::string, std::shared_ptr<MapConfig>> directory_maps_;

public:
  Impl(Params&& params)
      : map_directory_(std::move(params.map_directory)),
        parser_(std::move(params.parser)),
        world_config_(std::move(params.world_config)),
        logger_(logging::get_logger("sim.map_manager"))
  {
    if (!parser_) {
      throw std::runtime_error("MapManager requires a valid ConfigParser");
    }
    if (!world_config_) {
      throw std::runtime_error("MapManager requires a valid WorldConfig");
    }

    if (!map_directory_.empty()) {
      if (!fs::exists(map_directory_) || !fs::is_directory(map_directory_)) {
        DDRL_LOG_WARN(
          logger_, "Map directory '{}' is invalid or not a directory", map_directory_.string()
        );
        return;
      }

      for (const auto& entry : fs::directory_iterator(map_directory_)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".yaml") {
          continue;
        }

        auto result = parser_->parse_map_config(entry.path().string());
        if (!result) {
          DDRL_LOG_WARN(
            logger_,
            "Failed to parse map file '{}': {}",
            entry.path().string(),
            core::to_string(result.error())
          );
          continue;
        }

        auto map = std::make_shared<MapConfig>(std::move(result.value()));
        enrich_map_bounds(*map, entry.path().stem().string());

        available_map_names_.push_back(map->name);
        directory_maps_.emplace(map->name, std::move(map));
      }
      if (available_map_names_.empty()) {
        throw std::runtime_error(
          std::format("No valid map files found in directory '{}'", map_directory_.string())
        );
      }
    }
  }

  std::shared_ptr<const config::MapConfig> get_map_by_name(std::string_view name)
  {
    if (!name.empty()) {
      auto it = directory_maps_.find(std::string(name));
      if (it != directory_maps_.end()) {
        DDRL_LOG_DEBUG(logger_, "Selected fixed preset map '{}'", name);
        return it->second;
      }
    }

    DDRL_LOG_WARN(
      logger_, "Requested map '{}' not found in available maps", name.empty() ? "<empty>" : name
    );
    return {};
  }

  std::shared_ptr<const config::MapConfig> get_map_random()
  {
    if (!directory_maps_.empty()) {
      static thread_local std::mt19937      rng{std::random_device{}()};
      std::uniform_int_distribution<size_t> dist(0, available_map_names_.size() - 1);
      const auto&                           name = available_map_names_.at(dist(rng));
      return directory_maps_.at(name);
    }

    if (!directory_maps_.empty()) {
      static thread_local std::mt19937 rng{std::random_device{}()};
      std::vector<std::string>         keys;
      keys.reserve(directory_maps_.size());
      for (const auto& [key, _] : directory_maps_) {
        keys.push_back(key);
      }
      std::uniform_int_distribution<size_t> dist(0, keys.size() - 1);
      const auto&                           chosen = keys.at(dist(rng));
      return directory_maps_.at(chosen);
    }

    DDRL_LOG_WARN(logger_, "No maps available for RANDOM selection");
    return {};
  }

  [[nodiscard]] std::span<const std::string> get_available_map_names() const
  {
    return available_map_names_;
  }

  [[nodiscard]] std::vector<std::shared_ptr<const config::MapConfig>> get_available_maps() const
  {
    std::vector<std::shared_ptr<const MapConfig>> maps;
    maps.reserve(directory_maps_.size());
    for (const auto& [_, map] : directory_maps_) {
      maps.push_back(map);
    }
    return maps;
  }

private:
  void enrich_map_bounds(MapConfig& map, const std::string& map_name)
  {
    const auto& b = map.bounds;

    const double cx             = (b.min.x + b.max.x) / 2.0;
    const double cy             = (b.min.y + b.max.y) / 2.0;
    const double hx             = (b.max.x - b.min.x) / 2.0;
    const double hy             = (b.max.y - b.min.y) / 2.0;
    const double wall_thickness = 2.0;
    const double ht             = wall_thickness / 2.0;

    const auto   model_id = std::format("{}_bounds_model", map_name);
    ModelProfile model_profile{
      .name     = model_id,
      .category = "structure",
      .parts    = {
        {.name = std::format("{}_bounds_left", map_name),
            .geom = {.base = core::geom::Box2{ht, hy + wall_thickness}, .z_min = 0.0, .z_max = 10.0},
            .rel_pose = core::math::Pose2{{b.min.x - ht, cy}, 0.0},
            .material = {.name = "default_wall", .rgba = {0.1f, 0.1f, 0.15f, 1.0f}}},
        {.name = std::format("{}_bounds_right", map_name),
            .geom = {.base = core::geom::Box2{ht, hy + wall_thickness}, .z_min = 0.0, .z_max = 10.0},
            .rel_pose = core::math::Pose2{{b.max.x + ht, cy}, 0.0},
            .material = {.name = "default_wall", .rgba = {0.1f, 0.1f, 0.15f, 1.0f}}},
        {.name = std::format("{}_bounds_bottom", map_name),
            .geom = {.base = core::geom::Box2{hx + wall_thickness, ht}, .z_min = 0.0, .z_max = 10.0},
            .rel_pose = core::math::Pose2{{cx, b.min.y - ht}, 0.0},
            .material = {.name = "default_wall", .rgba = {0.1f, 0.1f, 0.15f, 1.0f}}},
        {.name = std::format("{}_bounds_top", map_name),
            .geom = {.base = core::geom::Box2{hx + wall_thickness, ht}, .z_min = 0.0, .z_max = 10.0},
            .rel_pose = core::math::Pose2{{cx, b.max.y + ht}, 0.0},
            .material = {.name = "default_wall", .rgba = {0.1f, 0.1f, 0.15f, 1.0f}}}
      }
    };

    world_config_->model_profiles.emplace(model_id, model_profile);
    map.static_instances.push_back(
      StaticInstanceCfg{
        .name       = std::format("{}_bounds", map_name),
        .model_id   = model_id,
        .pose_world = core::math::Pose2{{0.0, 0.0}, 0.0}
      }
    );
  }
};

MapManager::MapManager(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

MapManager::~MapManager() = default;

MapManager::MapManager(MapManager&&) noexcept            = default;
MapManager& MapManager::operator=(MapManager&&) noexcept = default;

std::shared_ptr<const config::MapConfig> MapManager::get_map_by_name(std::string_view name)
{
  return impl_->get_map_by_name(name);
}

std::shared_ptr<const config::MapConfig> MapManager::get_map_random()
{
  return impl_->get_map_random();
}

std::span<const std::string> MapManager::get_available_map_names() const
{
  return impl_->get_available_map_names();
}

std::vector<std::shared_ptr<const config::MapConfig>> MapManager::get_available_maps() const
{
  return impl_->get_available_maps();
}

} // namespace ddrl::sim
