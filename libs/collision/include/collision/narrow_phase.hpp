#pragma once

#include "core/geom/shape.hpp"
#include "core/math/pose.hpp"
#include "core/math/primitives.hpp"


namespace ddrl::collision {

struct CollisionResult {
  bool intersects{false};
  double penetration_depth{0.0};
  core::math::Vec2 normal{}; // points from shape A to B when penetrating
};

[[nodiscard]] CollisionResult collide_shapes(const core::geom::Shape2p5 &a,
                                            const core::math::Pose2 &a_world,
                                            const core::geom::Shape2p5 &b,
                                            const core::math::Pose2 &b_world);

} // namespace ddrl::collision

