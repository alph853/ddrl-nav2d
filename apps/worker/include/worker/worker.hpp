#pragma once

#include <core/base/result.hpp>
#include <core/rl/rl.hpp>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "worker/config.hpp"
#include "worker/coordinator_client.hpp"
#include "worker/observation_encoder.hpp"
#include "worker/policy_subscriber.hpp"
#include "worker/rollout_publisher.hpp"
#include "worker/rollout_recorder.hpp"
#include "worker/rollout_stream_client.hpp"
#include "worker/sim_instance.hpp"
#include "worker/viz_instance.hpp"

namespace ddrl::config {
class ConfigParser;
class RLConfig;
class RobotProfile;
class SimConfig;
class VizConfig;
class WorldConfig;
} // namespace ddrl::config

namespace ddrl::events {
class EventBus;
} // namespace ddrl::events

namespace ddrl::sim {
class MapManager;
} // namespace ddrl::sim

namespace ddrl::threading {
class ThreadPool;
} // namespace ddrl::threading

namespace ddrl::logging {
class Logger;
} // namespace ddrl::logging

namespace ddrl::worker {

class PolicyModel;

/**
 * @brief Worker for distributed RL training
 *
 * Spawns multiple Simulator instances, each running on a separate thread.
 * Uses a shared thread pool for raycasting operations across all simulators.
 */
class Worker
{
public:
  struct Params {
    std::string     config_path; ///< Path to worker.yaml
    WorkerCLIConfig overrides;   ///< Optional runtime overrides
    std::string     version;
  };

  explicit Worker(Params params);
  ~Worker();

  // Non-copyable, movable
  Worker(const Worker&)            = delete;
  Worker& operator=(const Worker&) = delete;
  Worker(Worker&&)            = delete;
  Worker& operator=(Worker&&) = delete;

  /**
   * @brief Start all simulator threads and worker loop
   */
  core::Result<void> init();

  /**
   * @brief Stop all simulator threads and worker loop
   */
  core::Result<void> stop();

  /**
   * @brief Run the worker (blocks until stopped)
   */
  core::Result<void> run();

  /**
   * @brief Check if worker is running
   */
  [[nodiscard]] bool is_running() const;

  /**
   * @brief Asynchronously request the worker to stop (signal-safe)
   */
  void request_stop();

private:
  struct PolicyEval {
    core::rl::Action action;
    double           log_prob{0.0};
  };

  void sim_thread_main(const std::stop_token& stoken, uint32_t sim_index);
  [[nodiscard]] PolicyEval
  generate_action(
    uint32_t sim_id, const core::rl::Observation& obs, const EncodedObservation& encoded_obs,
    double sim_time
  );
  [[nodiscard]] PolicyEval run_random_policy(double sim_time) const;
  PolicyEval run_neural_policy(
    uint32_t sim_id, std::span<const float> encoded_features,
    double sim_time
  );
  [[nodiscard]] PolicyEval
  run_expert_policy(uint32_t sim_id, const core::rl::Observation& obs, double sim_time);
  void reset_policy_context(SimInstance& instance, uint64_t version) const;
  void handle_rollout_ready(
    uint32_t sim_id, std::vector<TrajectoryStep>&& steps, double episode_return, bool is_terminal
  );
  void publish_rollout_batch(const std::vector<Rollout>& batch);
  core::Result<void> setup_control_plane(uint32_t num_sim_instances);
  [[nodiscard]] HeartbeatSnapshot  build_heartbeat_snapshot() const;
  void               handle_control_directive(const ddrl::comm::ControlDirective& directive);
  static std::string make_default_worker_identity();
  [[nodiscard]] std::shared_ptr<const config::RobotProfile> get_robot_profile() const;
  core::Result<void> load_offline_policy();
  [[nodiscard]] uint64_t current_policy_version() const;

  std::shared_ptr<logging::Logger> logger_;

  std::string                   config_path_;
  std::string                   worker_version_;
  WorkerCLIConfig               cli_overrides_;
  std::shared_ptr<WorkerConfig> config_;

  std::shared_ptr<config::WorldConfig>     world_config_;
  std::shared_ptr<config::SimConfig>       sim_config_;
  std::shared_ptr<const config::RLConfig>  rl_config_;
  std::shared_ptr<const config::VizConfig> viz_config_;
  std::shared_ptr<sim::MapManager>         map_manager_;

  std::shared_ptr<events::EventBus>            event_bus_;
  std::shared_ptr<ddrl::threading::ThreadPool> raycast_pool_;
  std::shared_ptr<config::ConfigParser>        config_parser_;

  std::vector<SimInstance>            sim_instances_;
  std::unique_ptr<RolloutPublisher>   rollout_publisher_;
  std::unique_ptr<RolloutStreamClient> rollout_stream_client_;
  std::unique_ptr<RolloutRecorder>    rollout_recorder_;
  uint64_t                            rollout_batch_sequence_{0};
  std::unique_ptr<CoordinatorClient>  coordinator_client_;
  std::unique_ptr<PolicySubscriber>   policy_subscriber_;
  std::shared_ptr<const PolicyModel>  offline_policy_model_;
  VizInstance                         viz_;

  bool              initialized_{false};
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
};

} // namespace ddrl::worker
