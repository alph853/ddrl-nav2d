#pragma once

#include <memory>
#include <thread>

#include "visualizer/visualizer.hpp"

namespace ddrl::worker {

class VizInstance {
public:
  void set(std::unique_ptr<visualizer::Visualizer> viz) { viz_ = std::move(viz); }

  void launch()
  {
    if (viz_ && !viz_thread_.joinable()) {
      viz_thread_ = std::thread([this]() {
        viz_->init();
        viz_->start_interactive(); // Blocks until visualization is stopped
      });
    }
  }

  void stop()
  {
    if (!viz_) {
      return;
    }
    viz_->stop_interactive();
    if (viz_thread_.joinable()) {
      viz_thread_.join();
    }
  }

  [[nodiscard]] bool enabled() const { return static_cast<bool>(viz_); }
  [[nodiscard]] visualizer::Visualizer* get() { return viz_.get(); }

private:
  std::unique_ptr<visualizer::Visualizer> viz_;
  std::thread                             viz_thread_;
};

} // namespace ddrl::worker
