#include "collision/bvh.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <ranges>
#include <vector>
#include <logging/logging.hpp>

namespace ddrl::collision {

using core::geom::get_bounds;
using core::geom::Shape2p5;

//==============================================================================
// Helper functions
//==============================================================================

namespace detail {

inline AABB2 transform_bounds(const AABB2& local, const Pose2& pose)
{
  const std::array<Point2, 4> corners{{
      {local.min.x, local.min.y},
      {local.min.x, local.max.y},
      {local.max.x, local.min.y},
      {local.max.x, local.max.y},
  }};
  const auto                  first = pose.transform_point(corners.front());
  AABB2                       out{.min = first, .max = first};
  for (std::size_t i = 1; i < corners.size(); ++i) {
    const auto p = pose.transform_point(corners[i]);
    out.min.x    = std::min(p.x, out.min.x);
    out.min.y    = std::min(p.y, out.min.y);
    out.max.x    = std::max(p.x, out.max.x);
    out.max.y    = std::max(p.y, out.max.y);
  }
  return out;
}

inline AABB2 part_bounds(const Shape2p5& part, const Pose2& pose)
{
  const auto base_bb = get_bounds(part.base);
  return transform_bounds(base_bb, pose);
}

} // namespace detail

//==============================================================================
// BLAS2::Impl - Private Implementation
//==============================================================================

struct BLAS2::Impl {
  std::vector<BVHNode2>    nodes;
  std::vector<BLAS2::Leaf> leaves;
  std::uint32_t            root{BVHNode2::kInvalid};

  std::shared_ptr<logging::Logger> logger = logging::get_logger("collision.blas2");

  void build_from(const config::ModelProfile& model)
  {
    nodes.clear();
    leaves.clear();
    root = BVHNode2::kInvalid;

    if (model.parts.empty()) {
      return;
    }

    leaves.reserve(model.parts.size());
    for (std::uint32_t idx = 0; idx < model.parts.size(); ++idx) {
      const auto& part     = model.parts[idx];
      const auto& rel_pose = part.rel_pose;
      BLAS2::Leaf leaf{};
      leaf.bounds     = detail::part_bounds(part.geom, rel_pose);
      leaf.z_min      = part.geom.z_min;
      leaf.z_max      = part.geom.z_max;
      leaf.part_index = idx;
      leaves.push_back(leaf);
    }

    std::vector<std::uint32_t> indices(leaves.size());
    std::ranges::iota(indices, 0);

    const auto build_recursive =
        [this](auto&& self, std::span<std::uint32_t> range, std::uint32_t parent) -> std::uint32_t {
      const auto node_index = static_cast<std::uint32_t>(nodes.size());
      nodes.push_back({});
      auto& node  = nodes.back();
      node.parent = parent;

      if (range.size() == 1) {
        const auto leaf_idx = range.front();
        node.bounds         = leaves[leaf_idx].bounds;
        node.leaf           = leaf_idx;
        return node_index;
      }

      AABB2 bounds = leaves[range.front()].bounds;
      for (std::size_t i = 1; i < range.size(); ++i) {
        bounds = bounds.merge(leaves[range[i]].bounds);
      }
      const auto axis = bounds.longest_axis();
      const auto mid  = range.size() / 2;
      const auto cmp  = [axis, this](std::uint32_t lhs, std::uint32_t rhs) {
        const auto lc = leaves[lhs].bounds.center();
        const auto rc = leaves[rhs].bounds.center();
        return axis == 0 ? lc.x < rc.x : lc.y < rc.y;
      };
      auto mid_it = range.begin() + static_cast<std::ptrdiff_t>(mid);
      std::ranges::nth_element(range, mid_it, cmp);

      const auto left_span  = range.first(mid);
      const auto right_span = range.subspan(mid);

      node.left   = self(self, left_span, node_index);
      node.right  = self(self, right_span, node_index);
      node.bounds = nodes[node.left].bounds.merge(nodes[node.right].bounds);
      return node_index;
    };

    nodes.reserve(leaves.size() * 2);
    root = build_recursive(build_recursive, std::span(indices), BVHNode2::kInvalid);
  }
};

//==============================================================================
// BLAS2 Implementation
//==============================================================================

BLAS2 BLAS2::build(const config::ModelProfile& model)
{
  BLAS2 blas;
  blas.impl_->build_from(model);
  return blas;
}

BLAS2::BLAS2() : impl_(std::make_unique<Impl>())
{
}

BLAS2::~BLAS2() = default;

BLAS2::BLAS2(BLAS2&&) noexcept = default;

BLAS2& BLAS2::operator=(BLAS2&&) noexcept = default;

bool BLAS2::empty() const noexcept
{
  return impl_->leaves.empty();
}

std::uint32_t BLAS2::root() const noexcept
{
  return impl_->root;
}

std::span<const BVHNode2> BLAS2::nodes() const noexcept
{
  return impl_->nodes;
}

std::span<const BLAS2::Leaf> BLAS2::leaves() const noexcept
{
  return impl_->leaves;
}

//==============================================================================
// TLAS2::Impl - Private Implementation
//==============================================================================

struct TLAS2::Impl {
  std::vector<BVHNode2>      nodes;
  std::vector<TLAS2::Leaf>   leaves;
  std::vector<std::uint32_t> leaf_nodes; // Maps leaf index to node index
  std::uint32_t              root{BVHNode2::kInvalid};

  std::shared_ptr<logging::Logger> logger = logging::get_logger("collision.tlas2");

  void rebuild_tree(std::span<const TLAS2::Instance> instances)
  {
    DDRL_LOG_DEBUG(logger, "TLAS2::Impl::rebuild_tree called with {} instances", instances.size());

    nodes.clear();
    leaf_nodes.clear();
    leaves.clear();
    root = BVHNode2::kInvalid;

    if (instances.empty()) {
      DDRL_LOG_WARN(logger, "TLAS2::rebuild_tree: No instances provided, creating empty tree");
      return;
    }

    // Validate instances before building
    for (size_t i = 0; i < instances.size(); ++i) {
      const auto& inst = instances[i];
      if (!inst.model) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Instance {} has null model pointer", i);
        return;
      }
      if (!inst.blas) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Instance {} has null BLAS pointer", i);
        return;
      }
      // Validate AABB
      if (inst.bounds.min.x > inst.bounds.max.x || inst.bounds.min.y > inst.bounds.max.y) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Instance {} has invalid AABB: min=({}, {}), max=({}, {})",
                       i, inst.bounds.min.x, inst.bounds.min.y, inst.bounds.max.x, inst.bounds.max.y);
        return;
      }
      if (inst.z_min > inst.z_max) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Instance {} has invalid Z range: z_min={}, z_max={}",
                       i, inst.z_min, inst.z_max);
        return;
      }
    }

    DDRL_LOG_DEBUG(logger, "TLAS2::rebuild_tree: Creating {} leaves", instances.size());
    leaves.reserve(instances.size());
    for (const auto& inst : instances) {
      leaves.push_back({inst.model,
                        inst.blas,
                        inst.world_from_model,
                        inst.bounds,
                        inst.z_min,
                        inst.z_max,
                        inst.entity_id});
    }

    leaf_nodes.resize(leaves.size(), BVHNode2::kInvalid);
    std::vector<std::uint32_t> indices(leaves.size());
    std::ranges::iota(indices, 0);

    DDRL_LOG_DEBUG(logger, "TLAS2::rebuild_tree: Starting recursive BVH build");

    const auto build_recursive =
        [this](auto&& self, std::span<std::uint32_t> range, std::uint32_t parent) -> std::uint32_t {
      const auto node_index = static_cast<std::uint32_t>(nodes.size());

      if (nodes.size() >= nodes.capacity()) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Node vector capacity exceeded at index {}", node_index);
        return BVHNode2::kInvalid;
      }

      nodes.push_back({});
      auto& node  = nodes.back();
      node.parent = parent;

      if (range.size() == 1) {
        const auto leaf_idx  = range.front();
        if (leaf_idx >= leaves.size()) {
          DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Invalid leaf index {} (leaves.size()={})",
                         leaf_idx, leaves.size());
          return BVHNode2::kInvalid;
        }
        node.bounds          = leaves[leaf_idx].bounds;
        node.leaf            = leaf_idx;
        leaf_nodes[leaf_idx] = node_index;
        return node_index;
      }

      // Compute bounding box for this node
      AABB2 bounds = leaves[range.front()].bounds;
      for (std::size_t i = 1; i < range.size(); ++i) {
        const auto idx = range[i];
        if (idx >= leaves.size()) {
          DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Invalid range index {} (leaves.size()={})",
                         idx, leaves.size());
          return BVHNode2::kInvalid;
        }
        bounds = bounds.merge(leaves[idx].bounds);
      }

      const auto axis = bounds.longest_axis();
      const auto mid  = range.size() / 2;
      const auto cmp  = [axis, this](std::uint32_t lhs, std::uint32_t rhs) {
        const auto lc = leaves[lhs].bounds.center();
        const auto rc = leaves[rhs].bounds.center();
        return axis == 0 ? lc.x < rc.x : lc.y < rc.y;
      };
      auto mid_it = range.begin() + static_cast<std::ptrdiff_t>(mid);
      std::ranges::nth_element(range, mid_it, cmp);

      const auto left_span  = range.first(mid);
      const auto right_span = range.subspan(mid);

      node.left  = self(self, left_span, node_index);
      node.right = self(self, right_span, node_index);

      if (node.left == BVHNode2::kInvalid || node.right == BVHNode2::kInvalid) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Failed to build child nodes");
        return BVHNode2::kInvalid;
      }

      if (node.left >= nodes.size() || node.right >= nodes.size()) {
        DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Invalid child indices left={}, right={}, nodes.size()={}",
                       node.left, node.right, nodes.size());
        return BVHNode2::kInvalid;
      }

      node.bounds = nodes[node.left].bounds.merge(nodes[node.right].bounds);
      return node_index;
    };

    nodes.reserve(leaves.size() * 2);
    root = build_recursive(build_recursive, std::span(indices), BVHNode2::kInvalid);

    if (root == BVHNode2::kInvalid) {
      DDRL_LOG_ERROR(logger, "TLAS2::rebuild_tree: Failed to build tree root");
    } else {
      DDRL_LOG_DEBUG(logger, "TLAS2::rebuild_tree: Successfully built tree with {} nodes, {} leaves, root={}",
                    nodes.size(), leaves.size(), root);
    }
  }

  void refit_internal()
  {
    if (root == BVHNode2::kInvalid) {
      return;
    }

    for (std::size_t i = 0; i < leaves.size(); ++i) {
      const auto node_idx = leaf_nodes[i];
      if (node_idx != BVHNode2::kInvalid) {
        nodes[node_idx].bounds = leaves[i].bounds;
      }
    }

    std::vector<char>          visited(nodes.size(), 0);
    std::vector<std::uint32_t> to_update;
    to_update.reserve(nodes.size());

    for (const auto node_idx : leaf_nodes) {
      auto parent = node_idx == BVHNode2::kInvalid ? BVHNode2::kInvalid : nodes[node_idx].parent;
      while (parent != BVHNode2::kInvalid && (visited[parent] == 0)) {
        visited[parent] = 1;
        to_update.push_back(parent);
        parent = nodes[parent].parent;
      }
    }

    for (unsigned int& it : std::ranges::reverse_view(to_update)) {
      auto& node  = nodes[it];
      node.bounds = nodes[node.left].bounds.merge(nodes[node.right].bounds);
    }
  }

  void refit_from_leaf(std::size_t leaf_idx)
  {
    const auto node_idx = leaf_nodes[leaf_idx];
    if (node_idx == BVHNode2::kInvalid) {
      return;
    }

    nodes[node_idx].bounds = leaves[leaf_idx].bounds;
    auto parent            = nodes[node_idx].parent;
    while (parent != BVHNode2::kInvalid) {
      auto& node  = nodes[parent];
      node.bounds = nodes[node.left].bounds.merge(nodes[node.right].bounds);
      parent      = node.parent;
    }
  }
};

//==============================================================================
// TLAS2 Implementation
//==============================================================================

TLAS2 TLAS2::build(std::span<const Instance> instances)
{
  TLAS2 tlas;
  tlas.impl_->rebuild_tree(instances);
  return tlas;
}

TLAS2::TLAS2() : impl_(std::make_unique<Impl>())
{
}

TLAS2::~TLAS2() = default;

TLAS2::TLAS2(TLAS2&&) noexcept = default;

TLAS2& TLAS2::operator=(TLAS2&&) noexcept = default;

void TLAS2::rebuild(std::span<const Instance> instances)
{
  impl_->rebuild_tree(instances);
}

void TLAS2::refit(std::span<const LeafState> states)
{
  if (states.size() != impl_->leaves.size()) {
    return;
  }
  for (std::size_t i = 0; i < impl_->leaves.size(); ++i) {
    impl_->leaves[i].world_from_model = states[i].world_from_model;
    impl_->leaves[i].bounds           = states[i].bounds;
    impl_->leaves[i].z_min            = states[i].z_min;
    impl_->leaves[i].z_max            = states[i].z_max;
  }
  impl_->refit_internal();
}

void TLAS2::update_leaf(std::size_t idx, const LeafState& state)
{
  if (idx >= impl_->leaves.size()) {
    return;
  }
  auto& leaf            = impl_->leaves[idx];
  leaf.world_from_model = state.world_from_model;
  leaf.bounds           = state.bounds;
  leaf.z_min            = state.z_min;
  leaf.z_max            = state.z_max;
  impl_->refit_from_leaf(idx);
}

void TLAS2::refit_all()
{
  impl_->refit_internal();
}

bool TLAS2::empty() const noexcept
{
  return impl_->leaves.empty();
}

std::uint32_t TLAS2::root() const noexcept
{
  return impl_->root;
}

std::span<const BVHNode2> TLAS2::nodes() const noexcept
{
  return impl_->nodes;
}

std::span<const TLAS2::Leaf> TLAS2::leaves() const noexcept
{
  return impl_->leaves;
}

std::span<const std::uint32_t> TLAS2::leaf_nodes() const noexcept
{
  return impl_->leaf_nodes;
}

//==============================================================================
// TLAS2Pair Implementation
//==============================================================================

TLAS2Pair TLAS2Pair::build(std::span<const TLAS2::Instance> statics,
                           std::span<const TLAS2::Instance> dynamics)
{
  TLAS2Pair pair;
  pair.static_tree.rebuild(statics);
  pair.dynamic_tree.rebuild(dynamics);
  return pair;
}

void TLAS2Pair::rebuild_statics(std::span<const TLAS2::Instance> instances)
{
  static_tree.rebuild(instances);
}

void TLAS2Pair::rebuild_dynamics(std::span<const TLAS2::Instance> instances)
{
  dynamic_tree.rebuild(instances);
}

void TLAS2Pair::refit_statics(std::span<const TLAS2::LeafState> states)
{
  static_tree.refit(states);
}

void TLAS2Pair::refit_dynamics(std::span<const TLAS2::LeafState> states)
{
  dynamic_tree.refit(states);
}

void TLAS2Pair::update_static_leaf(std::size_t idx, const TLAS2::LeafState& state)
{
  static_tree.update_leaf(idx, state);
}

void TLAS2Pair::update_dynamic_leaf(std::size_t idx, const TLAS2::LeafState& state)
{
  dynamic_tree.update_leaf(idx, state);
}

void TLAS2Pair::refit_all()
{
  static_tree.refit_all();
  dynamic_tree.refit_all();
}

} // namespace ddrl::collision
