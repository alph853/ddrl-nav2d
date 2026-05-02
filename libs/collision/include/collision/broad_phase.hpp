#pragma once

#include "core/base/result.hpp"
#include "core/geom/aabb.hpp"
#include "core/math/pose.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "collision/bvh.hpp"
#include "config/world.hpp"

namespace ddrl::collision {

/**
 * @brief Broadphase collision detection manager
 */
class BroadPhase
{
public:
  struct Instance {
    std::string                                 name;
    core::math::Pose2                           pose_world;
    std::shared_ptr<const config::ModelProfile> model_profile;
    bool                                        is_static{true};
    std::uint32_t                               entity_id;
  };

  struct Params {
    std::vector<Instance>                instances;
    std::vector<config::CollisionFilter> collision_masks;
  };

  BroadPhase(Params params);
  ~BroadPhase();

  // Movable but not copyable
  BroadPhase(BroadPhase&&) noexcept;
  BroadPhase& operator=(BroadPhase&&) noexcept;
  BroadPhase(const BroadPhase&)            = delete;
  BroadPhase& operator=(const BroadPhase&) = delete;

  /**
   * @brief Add an object instance to the broadphase
   */
  core::Result<void> add_instance(const Instance& instance);

  /**
   * @brief Build or rebuild all acceleration structures
   * Constructs BVH trees for both static and dynamic objects.
   */
  void build();

  /**
   * @brief Rebuild only the static BVH
   */
  void rebuild_statics();

  /**
   * @brief Rebuild only the dynamic BVH
   */
  void rebuild_dynamics();

  /**
   * @brief Update the pose of a dynamic instance
   */
  void update_dynamics(std::uint32_t entity_id, const core::math::Pose2& new_pose);

  /**
   * @brief Refit dynamic BVH after batch updates
   */
  void refit_dynamics();

  /**
   * @brief Query objects overlapping an AABB
   *
   * @param query Query AABB in world space
   * @param include_static Include static objects in results
   * @param include_dynamic Include dynamic objects in results
   * @return std::vector<std::uint32_t> List of overlapping entity IDs
   */
  [[nodiscard]] std::vector<std::uint32_t> query_entity_ids(
    const core::geom::AABB2& query, bool include_static = true, bool include_dynamic = true
  ) const;

  [[nodiscard]] std::optional<std::uint32_t> entity_id(const std::string& name) const;

  /**
   * @brief Get the underlying TLAS pair (for ray casting)
   * @return Reference to TLAS2Pair
   */
  [[nodiscard]] const TLAS2Pair& trees() const noexcept;

  /**
   * @brief Get all static instances
   * @return Span of static instances
   */
  [[nodiscard]] std::span<const TLAS2::Instance> static_instances() const noexcept;

  /**
   * @brief Get all dynamic instances
   * @return Span of dynamic instances
   */
  [[nodiscard]] std::span<const TLAS2::Instance> dynamic_instances() const noexcept;

private:
  class Impl; ///< Private implementation
  std::unique_ptr<Impl> impl_;
};

} // namespace ddrl::collision
