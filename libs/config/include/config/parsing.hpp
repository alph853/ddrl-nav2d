/**
 * @file parsing.hpp
 * @brief Configuration parsing utilities for YAML config files
 */

#pragma once

#include <core/base/result.hpp>
#include <memory>
#include <string_view>

#include "dynamic_profile.hpp"
#include "model_profile.hpp"
#include "rl.hpp"
#include "robot_profile.hpp"
#include "simulator.hpp"
#include "visualizer.hpp"
#include "world.hpp"

namespace ddrl::config {

/**
 * @brief Modern configuration parser with clean encapsulation
 *
 * This class provides methods for parsing all configuration file types.
 * It uses the PIMPL pattern to hide implementation details and YAML dependencies.
 */
class ConfigParser {
public:
  ConfigParser();
  ~ConfigParser();

  // Move-only semantics
  ConfigParser(ConfigParser&&) noexcept;
  ConfigParser& operator=(ConfigParser&&) noexcept;

  ConfigParser(const ConfigParser&) = delete;
  ConfigParser& operator=(const ConfigParser&) = delete;

  /**
   * @brief Parse simulator configuration from YAML file
   */
  [[nodiscard]] core::Result<SimConfig> parse_simulator_config(std::string_view path);

  /**
   * @brief Parse reinforcement-learning configuration from YAML file.
   */
  [[nodiscard]] core::Result<RLConfig> parse_rl_config(std::string_view path);

  /**
   * @brief Parse shared policy architecture configuration from YAML file.
   */
  [[nodiscard]] core::Result<RLPolicyConfig::Architecture> parse_policy_arch_config(
    std::string_view path
  );

  /**
   * @brief Parse reward configuration from YAML file.
   */
  [[nodiscard]] core::Result<RLRewardConfig> parse_reward_config(std::string_view path);

  /**
   * @brief Parse world configuration from YAML file
   */
  [[nodiscard]] core::Result<WorldConfig> parse_world_config(std::string_view path);

  /**
   * @brief Parse model profile from YAML file
   */
  [[nodiscard]] core::Result<ModelProfile> parse_model_profile(std::string_view path);

  /**
   * @brief Parse dynamic profile from YAML file
   */
  [[nodiscard]] core::Result<DynamicProfile> parse_dynamic_profile(std::string_view path);

  /**
   * @brief Parse robot profile from YAML file
   */
  [[nodiscard]] core::Result<RobotProfile> parse_robot_profile(std::string_view path);

  /**
   * @brief Parse visualizer configuration from YAML file
   */
  [[nodiscard]] core::Result<VizConfig> parse_visualizer_config(std::string_view path);

  /**
   * @brief Parse map configuration from YAML file
   */
  [[nodiscard]] core::Result<MapConfig> parse_map_config(std::string_view path);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::config
