#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "config/rl.hpp"
#include "logging/logging.hpp"
#include "worker/rollout_publisher.hpp"

namespace ddrl::worker {

class RolloutRecorder
{
public:
  struct Params {
    std::filesystem::path       output_dir;
    uint32_t                    max_files{20000};
    std::shared_ptr<logging::Logger> logger;
  };

  explicit RolloutRecorder(Params params);

  void record_batch(
    const std::vector<Rollout>& rollouts, uint32_t worker_id, uint64_t batch_sequence,
    const config::RLPolicyConfig::Architecture& architecture
  );

private:
  std::filesystem::path         output_dir_;
  uint32_t                      max_files_{20000};
  std::shared_ptr<logging::Logger> logger_;
  uint64_t                      written_{0};
};

} // namespace ddrl::worker
