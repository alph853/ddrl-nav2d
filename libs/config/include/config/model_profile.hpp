#pragma once
#include "core/geom/shape.hpp"
#include "core/math/pose.hpp"
#include <array>
#include <string>

namespace ddrl::config {

struct MaterialRef {
  std::string          name;
  std::array<float, 4> rgba;
};

struct ModelPart {
  std::string          name;
  core::geom::Shape2p5 geom;
  core::math::Pose2    rel_pose{}; // relative to model origin

  MaterialRef material{}; // visual
};

struct ModelProfile {
  std::string name;
  std::string category;

  std::vector<ModelPart> parts;
};

} // namespace ddrl::config
