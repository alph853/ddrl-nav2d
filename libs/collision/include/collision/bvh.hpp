#pragma once

#include "core/geom/aabb.hpp"
#include "core/math/pose.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include "config/model_profile.hpp"

namespace ddrl::collision {

using core::geom::AABB2;
using core::math::Point2;
using core::math::Pose2;

/**
 * @brief Axis-aligned BVH node used by both BLAS and TLAS trees
 * Each node is either an internal node (with left/right children)
 * or a leaf node (referencing geometry).
 */
struct BVHNode2 {
  static constexpr std::uint32_t kInvalid = std::numeric_limits<std::uint32_t>::max();

  AABB2         bounds{};         ///< Axis-aligned bounding box
  std::uint32_t left{kInvalid};   ///< Left child index (kInvalid if none)
  std::uint32_t right{kInvalid};  ///< Right child index (kInvalid if none)
  std::uint32_t parent{kInvalid}; ///< Parent node index (kInvalid if root)
  std::uint32_t leaf{kInvalid};   ///< Leaf data index (kInvalid if internal)

  [[nodiscard]] constexpr bool is_leaf() const noexcept { return leaf != kInvalid; }
};

/**
 * @brief Bottom-Level Acceleration Structure (BLAS) for model-local geometry
 * BLAS2 builds a BVH over the parts of a single ModelProfile in local space.
 * Multiple instances can share the same BLAS with different world transforms.
 */
class BLAS2
{
public:
  /**
   * @brief Leaf data for BLAS containing part geometry bounds
   */
  struct Leaf {
    AABB2         bounds{};      ///< Local-space AABB of the part
    double        z_min{0.0};    ///< Minimum Z coordinate
    double        z_max{0.0};    ///< Maximum Z coordinate
    std::uint32_t part_index{0}; ///< Index into ModelProfile::parts array
  };

  /**
   * @brief Build a BLAS from a model's geometry
   */
  [[nodiscard]] static BLAS2 build(const config::ModelProfile& model);

  BLAS2();
  ~BLAS2();

  // Movable but not copyable
  BLAS2(BLAS2&&) noexcept;
  BLAS2& operator=(BLAS2&&) noexcept;
  BLAS2(const BLAS2&)            = delete;
  BLAS2& operator=(const BLAS2&) = delete;

  /**
   * @brief Check if BLAS is empty (no geometry)
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Get root node index
   */
  [[nodiscard]] std::uint32_t root() const noexcept;

  /**
   * @brief Get all BVH nodes
   */
  [[nodiscard]] std::span<const BVHNode2> nodes() const noexcept;

  /**
   * @brief Get all leaf data
   */
  [[nodiscard]] std::span<const Leaf> leaves() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Top-Level Acceleration Structure (TLAS) for world-space instances
 * TLAS2 builds a BVH over multiple object instances, each with a BLAS.
 * Supports efficient refitting when objects move (dynamic objects).
 */
class TLAS2
{
public:
  /**
   * @brief Instance descriptor for adding to TLAS
   */
  struct Instance {
    std::shared_ptr<const config::ModelProfile> model; ///< Geometric model
    std::shared_ptr<const BLAS2>    blas;  ///< Pre-built BLAS

    Pose2         world_from_model{}; ///< World transform
    AABB2         bounds{};           ///< World-space AABB
    double        z_min{0.0};         ///< World-space Z min
    double        z_max{0.0};         ///< World-space Z max
    std::uint32_t entity_id{0};       ///< Unique entity identifier
  };

  /**
   * @brief Leaf data in TLAS (same as Instance)
   */
  struct Leaf {
    std::shared_ptr<const config::ModelProfile> model;
    std::shared_ptr<const BLAS2>    blas;

    Pose2         world_from_model{};
    AABB2         bounds{};
    double        z_min{0.0};
    double        z_max{0.0};
    std::uint32_t entity_id{0};
  };

  /**
   * @brief Compact state for updating a single leaf
   */
  struct LeafState {
    Pose2  world_from_model{};
    AABB2  bounds{};
    double z_min{0.0};
    double z_max{0.0};
  };

  /**
   * @brief Build a TLAS from instances
   *
   * @param instances Object instances to add
   * @return TLAS2 Built acceleration structure
   */
  [[nodiscard]] static TLAS2 build(std::span<const Instance> instances);

  TLAS2();
  ~TLAS2();

  // Movable but not copyable
  TLAS2(TLAS2&&) noexcept;
  TLAS2& operator=(TLAS2&&) noexcept;
  TLAS2(const TLAS2&)            = delete;
  TLAS2& operator=(const TLAS2&) = delete;

  /**
   * @brief Rebuild TLAS from new instance set
   *
   * @param instances New set of instances
   */
  void rebuild(std::span<const Instance> instances);

  /**
   * @brief Refit all leaf bounds (after updating leaf data)
   *
   * @param states New states for all leaves (must match leaf count)
   */
  void refit(std::span<const LeafState> states);

  /**
   * @brief Update and refit a single leaf
   *
   * @param idx Leaf index
   * @param state New state for the leaf
   */
  void update_leaf(std::size_t idx, const LeafState& state);

  /**
   * @brief Refit all nodes from current leaf data
   *
   * Call after manually updating leaf transforms.
   */
  void refit_all();

  /**
   * @brief Check if TLAS is empty
   * @return true if no leaves
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Get root node index
   * @return Root node index (BVHNode2::kInvalid if empty)
   */
  [[nodiscard]] std::uint32_t root() const noexcept;

  /**
   * @brief Get all BVH nodes
   * @return Span of BVH nodes
   */
  [[nodiscard]] std::span<const BVHNode2> nodes() const noexcept;

  /**
   * @brief Get all leaf data
   * @return Span of leaf instances
   */
  [[nodiscard]] std::span<const Leaf> leaves() const noexcept;

  /**
   * @brief Get leaf-to-node mapping
   * @return Span of node indices (one per leaf)
   */
  [[nodiscard]] std::span<const std::uint32_t> leaf_nodes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Pair of TLAS trees for static and dynamic objects
 *
 * Separating static and dynamic objects improves performance:
 * - Static tree is built once and never refitted
 * - Dynamic tree is refitted every frame as objects move
 */
struct TLAS2Pair {
  TLAS2 static_tree;  ///< Tree for static (non-moving) objects
  TLAS2 dynamic_tree; ///< Tree for dynamic (moving) objects

  /**
   * @brief Build both static and dynamic trees
   *
   * @param statics Static object instances
   * @param dynamics Dynamic object instances
   * @return TLAS2Pair Built pair of trees
   */
  [[nodiscard]] static TLAS2Pair build(std::span<const TLAS2::Instance> statics,
                                       std::span<const TLAS2::Instance> dynamics);

  /**
   * @brief Rebuild the static tree
   * @param instances New static instances
   */
  void rebuild_statics(std::span<const TLAS2::Instance> instances);

  /**
   * @brief Rebuild the dynamic tree
   * @param instances New dynamic instances
   */
  void rebuild_dynamics(std::span<const TLAS2::Instance> instances);

  /**
   * @brief Refit static tree bounds
   * @param states New states for all static leaves
   */
  void refit_statics(std::span<const TLAS2::LeafState> states);

  /**
   * @brief Refit dynamic tree bounds
   * @param states New states for all dynamic leaves
   */
  void refit_dynamics(std::span<const TLAS2::LeafState> states);

  /**
   * @brief Update a single static leaf
   * @param idx Leaf index in static tree
   * @param state New state
   */
  void update_static_leaf(std::size_t idx, const TLAS2::LeafState& state);

  /**
   * @brief Update a single dynamic leaf
   * @param idx Leaf index in dynamic tree
   * @param state New state
   */
  void update_dynamic_leaf(std::size_t idx, const TLAS2::LeafState& state);

  /**
   * @brief Refit both trees from current leaf data
   */
  void refit_all();
};

} // namespace ddrl::collision
