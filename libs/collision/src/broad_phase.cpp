#include "collision/broad_phase.hpp"

#include "core/geom/shape.hpp"
#include <algorithm>
#include <expected>
#include <span>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddrl::collision {

using config::ModelProfile;
using core::Result;
using core::geom::AABB2;
using core::geom::get_bounds;
using core::geom::Shape2p5;
using core::math::Pose2;

struct CollisionMask {
  uint32_t category_bits{0xFFFFFFFF};
  uint32_t mask_bits{0xFFFFFFFF};
};

//==============================================================================
// BroadPhase::Impl - Private Implementation
//==============================================================================

class BroadPhase::Impl
{
private:
  std::vector<CollisionMask> collision_masks_;

  TLAS2Pair                     pair_{};
  std::vector<TLAS2::Instance>  static_instances_;
  std::vector<TLAS2::Instance>  dynamic_instances_;
  std::vector<TLAS2::LeafState> dynamic_states_;

  std::unordered_map<std::string, std::size_t>   static_lookup_;
  std::unordered_map<std::string, std::size_t>   dynamic_lookup_;
  std::unordered_map<std::uint32_t, std::size_t> entity_to_dynamic_idx_;

  bool          dynamic_dirty_{true};
  std::uint32_t next_entity_id_{0};

public:
  explicit Impl(Params&& params);

  Result<void> add_instance(const Instance& instance);
  Result<void> add_instance(Instance&& instance);

  void build();
  void rebuild_statics();
  void rebuild_dynamics();
  void update_dynamics(std::uint32_t entity_id, const Pose2& new_pose);
  void refit_dynamics();

  [[nodiscard]] std::vector<std::uint32_t>
  query_entity_ids(const AABB2& query, bool include_static, bool include_dynamic) const;

  [[nodiscard]] const TLAS2Pair&                 trees() const noexcept;
  [[nodiscard]] std::span<const TLAS2::Instance> static_instances() const noexcept;
  [[nodiscard]] std::span<const TLAS2::Instance> dynamic_instances() const noexcept;
  [[nodiscard]] std::optional<std::uint32_t>     entity_id(const std::string& name) const;

private:
  static TLAS2::LeafState make_leaf_state(const TLAS2::Instance& instance);
  void                    synchronize_dynamic_states();
  Result<void>            init_instance_info(const Instance& instance);
  static void             collect_overlaps(
                const TLAS2& tree, std::span<const TLAS2::Leaf> leaves, std::uint32_t root, const AABB2& query,
                std::vector<std::uint32_t>& out
              );
  static std::vector<CollisionMask>
  build_collision_masks(std::span<const config::CollisionFilter> filters);

  struct BoundsResult {
    AABB2  bounds;
    double z_min;
    double z_max;
  };

  [[nodiscard]] static BoundsResult
  compute_bounds(const ModelProfile& model, const Pose2& world_from_model);
};

BroadPhase::Impl::Impl(Params&& params)
{
  for (auto& instance : params.instances) {
    auto result = add_instance(instance);
    if (!result) {
      throw std::runtime_error("Failed to add BroadPhase instance: " + to_string(result.error()));
    }
  }
  collision_masks_ = build_collision_masks(params.collision_masks);
}

Result<void> BroadPhase::Impl::add_instance(const Instance& instance)
{
  return init_instance_info(instance);
}

void BroadPhase::Impl::build()
{
  pair_          = TLAS2Pair::build(static_instances_, dynamic_instances_);
  dynamic_dirty_ = false;
  synchronize_dynamic_states();
}

void BroadPhase::Impl::rebuild_statics()
{
  if (static_instances_.empty()) {
    pair_.static_tree = TLAS2{};
  } else {
    pair_.rebuild_statics(static_instances_);
  }
}

void BroadPhase::Impl::rebuild_dynamics()
{
  if (dynamic_instances_.empty()) {
    pair_.dynamic_tree = TLAS2{};
    dynamic_states_.clear();
  } else {
    pair_.rebuild_dynamics(dynamic_instances_);
    synchronize_dynamic_states();
  }
  dynamic_dirty_ = false;
}

void BroadPhase::Impl::update_dynamics(std::uint32_t entity_id, const Pose2& new_pose)
{
  const auto it = entity_to_dynamic_idx_.find(entity_id);
  if (it == entity_to_dynamic_idx_.end()) {
    return;
  }

  const std::size_t index = it->second;
  if (index >= dynamic_instances_.size()) {
    return;
  }

  auto& instance            = dynamic_instances_[index];
  instance.world_from_model = new_pose;

  const auto& model = instance.model;
  if (model == nullptr) {
    return;
  }

  const auto bounds = compute_bounds(*model, new_pose);
  instance.bounds   = bounds.bounds;
  instance.z_min    = bounds.z_min;
  instance.z_max    = bounds.z_max;

  TLAS2::LeafState state{};
  state.world_from_model = new_pose;
  state.bounds           = bounds.bounds;
  state.z_min            = bounds.z_min;
  state.z_max            = bounds.z_max;

  dynamic_states_[index] = state;

  if (!dynamic_dirty_) {
    pair_.update_dynamic_leaf(index, state);
  }
}

void BroadPhase::Impl::refit_dynamics()
{
  if (dynamic_dirty_ || dynamic_states_.empty()) {
    return;
  }
  pair_.refit_dynamics(dynamic_states_);
}

std::vector<std::uint32_t> BroadPhase::Impl::query_entity_ids(
  const AABB2& query, bool include_static, bool include_dynamic
) const
{
  std::vector<std::uint32_t> hits;
  if (include_static && !static_instances_.empty()) {
    collect_overlaps(
      pair_.static_tree, pair_.static_tree.leaves(), pair_.static_tree.root(), query, hits
    );
  }
  if (include_dynamic && !dynamic_instances_.empty()) {
    collect_overlaps(
      pair_.dynamic_tree, pair_.dynamic_tree.leaves(), pair_.dynamic_tree.root(), query, hits
    );
  }
  return hits;
}

const TLAS2Pair& BroadPhase::Impl::trees() const noexcept
{
  return pair_;
}

std::span<const TLAS2::Instance> BroadPhase::Impl::static_instances() const noexcept
{
  return static_instances_;
}

std::span<const TLAS2::Instance> BroadPhase::Impl::dynamic_instances() const noexcept
{
  return dynamic_instances_;
}

std::optional<std::uint32_t> BroadPhase::Impl::entity_id(const std::string& name) const
{
  if (const auto static_it = static_lookup_.find(name); static_it != static_lookup_.end()) {
    const auto idx = static_it->second;
    if (idx < static_instances_.size()) {
      return static_instances_[idx].entity_id;
    }
  }
  if (const auto dynamic_it = dynamic_lookup_.find(name); dynamic_it != dynamic_lookup_.end()) {
    const auto idx = dynamic_it->second;
    if (idx < dynamic_instances_.size()) {
      return dynamic_instances_[idx].entity_id;
    }
  }
  return std::nullopt;
}

TLAS2::LeafState BroadPhase::Impl::make_leaf_state(const TLAS2::Instance& instance)
{
  TLAS2::LeafState state{};
  state.world_from_model = instance.world_from_model;
  state.bounds           = instance.bounds;
  state.z_min            = instance.z_min;
  state.z_max            = instance.z_max;
  return state;
}

void BroadPhase::Impl::synchronize_dynamic_states()
{
  dynamic_states_.resize(dynamic_instances_.size());
  for (std::size_t i = 0; i < dynamic_instances_.size(); ++i) {
    dynamic_states_[i] = make_leaf_state(dynamic_instances_[i]);
  }
}

Result<void> BroadPhase::Impl::init_instance_info(const Instance& instance)
{
  const std::uint32_t entity_id = instance.entity_id;
  next_entity_id_               = std::max(next_entity_id_, entity_id + 1);

  const std::string& name  = instance.name;
  const auto         model = instance.model_profile;
  if (model == nullptr) {
    const std::string msg = "BroadPhase instance '" + name + "' missing model_profile";
    return std::unexpected(
      core::make_error(
        core::ErrorCode::CONFIG,
        msg,
        core::make_context("BroadPhase::Impl::init_instance_info", "name", name)
      )
    );
  }

  auto blas = std::make_shared<BLAS2>(BLAS2::build(*model));

  TLAS2::Instance tlas_instance{};
  tlas_instance.model            = model;
  tlas_instance.blas             = blas;
  tlas_instance.world_from_model = instance.pose_world;
  tlas_instance.entity_id        = entity_id;

  const auto bounds    = compute_bounds(*model, instance.pose_world);
  tlas_instance.bounds = bounds.bounds;
  tlas_instance.z_min  = bounds.z_min;
  tlas_instance.z_max  = bounds.z_max;

  if (instance.is_static) {
    const std::size_t index = static_instances_.size();
    static_instances_.push_back(std::move(tlas_instance));
    static_lookup_[name] = index;
  } else {
    const std::size_t index = dynamic_instances_.size();
    dynamic_instances_.push_back(std::move(tlas_instance));
    if (dynamic_states_.size() < dynamic_instances_.size()) {
      dynamic_states_.push_back(make_leaf_state(dynamic_instances_.back()));
    } else {
      dynamic_states_[index] = make_leaf_state(dynamic_instances_.back());
    }
    dynamic_lookup_[name]             = index;
    entity_to_dynamic_idx_[entity_id] = index;
    dynamic_dirty_                    = true;
  }

  return {};
}

void BroadPhase::Impl::collect_overlaps(
  const TLAS2& tree, std::span<const TLAS2::Leaf> leaves, std::uint32_t root, const AABB2& query,
  std::vector<std::uint32_t>& out
)
{
  if (root == BVHNode2::kInvalid) {
    return;
  }

  const auto                nodes = tree.nodes();
  std::stack<std::uint32_t> stack;
  stack.push(root);

  while (!stack.empty()) {
    const auto node_idx = stack.top();
    stack.pop();

    const auto& node = nodes[node_idx];
    if (!node.bounds.intersects(query)) {
      continue;
    }
    if (node.is_leaf()) {
      out.push_back(leaves[node.leaf].entity_id);
      continue;
    }
    if (node.left != BVHNode2::kInvalid) {
      stack.push(node.left);
    }
    if (node.right != BVHNode2::kInvalid) {
      stack.push(node.right);
    }
  }
}

std::vector<CollisionMask>
BroadPhase::Impl::build_collision_masks(std::span<const config::CollisionFilter> filters)
{
  std::vector<CollisionMask> masks;
  masks.reserve(filters.size());

  std::unordered_map<std::string, uint32_t> bit_index;
  uint32_t                                  next_bit = 0;

  for (const auto& f : filters) {
    if (!bit_index.contains(f.category)) {
      if (next_bit >= 32) {
        throw std::runtime_error("Too many collision categories (>32)");
      }
      bit_index[f.category] = next_bit++;
    }
    for (const auto& cw : f.collides_with) {
      if (!bit_index.contains(cw)) {
        if (next_bit >= 32) {
          throw std::runtime_error("Too many collision categories (>32)");
        }
        bit_index[cw] = next_bit++;
      }
    }
  }

  for (const auto& f : filters) {
    CollisionMask m;
    m.category_bits = 1u << bit_index.at(f.category);

    uint32_t mask = 0;
    for (const auto& cw : f.collides_with) {
      mask |= 1u << bit_index.at(cw);
    }

    m.mask_bits = mask;
    masks.push_back(m);
  }
  return masks;
}

BroadPhase::Impl::BoundsResult
BroadPhase::Impl::compute_bounds(const ModelProfile& model, const Pose2& world_from_model)
{
  BoundsResult result{};

  if (model.parts.empty()) {
    result.bounds.min = world_from_model.pos;
    result.bounds.max = world_from_model.pos;
    return result;
  }

  auto  it           = model.parts.begin();
  Pose2 part_pose    = world_from_model * it->rel_pose;
  auto  local_bounds = get_bounds(it->geom.base);
  auto  world_bounds = local_bounds.transform(part_pose);

  result.bounds = world_bounds;
  result.z_min  = it->geom.z_min;
  result.z_max  = it->geom.z_max;
  ++it;

  for (; it != model.parts.end(); ++it) {
    part_pose    = world_from_model * it->rel_pose;
    local_bounds = get_bounds(it->geom.base);
    world_bounds = local_bounds.transform(part_pose);
    result.bounds.merge_inplace(world_bounds);
    result.z_min = std::min(result.z_min, it->geom.z_min);
    result.z_max = std::max(result.z_max, it->geom.z_max);
  }

  return result;
}

//==============================================================================
// BroadPhase Implementation
//==============================================================================

BroadPhase::BroadPhase(Params params) : impl_(std::make_unique<Impl>(std::move(params)))
{
}

BroadPhase::~BroadPhase() = default;

BroadPhase::BroadPhase(BroadPhase&&) noexcept = default;

BroadPhase& BroadPhase::operator=(BroadPhase&&) noexcept = default;

Result<void> BroadPhase::add_instance(const Instance& instance)
{
  return impl_->add_instance(instance);
}

void BroadPhase::build()
{
  impl_->build();
}

void BroadPhase::rebuild_statics()
{
  impl_->rebuild_statics();
}

void BroadPhase::rebuild_dynamics()
{
  impl_->rebuild_dynamics();
}

void BroadPhase::update_dynamics(std::uint32_t entity_id, const core::math::Pose2& new_pose)
{
  impl_->update_dynamics(entity_id, new_pose);
}

void BroadPhase::refit_dynamics()
{
  impl_->refit_dynamics();
}

std::vector<std::uint32_t>
BroadPhase::query_entity_ids(const AABB2& query, bool include_static, bool include_dynamic) const
{
  return impl_->query_entity_ids(query, include_static, include_dynamic);
}

const TLAS2Pair& BroadPhase::trees() const noexcept
{
  return impl_->trees();
}

std::span<const TLAS2::Instance> BroadPhase::static_instances() const noexcept
{
  return impl_->static_instances();
}

std::span<const TLAS2::Instance> BroadPhase::dynamic_instances() const noexcept
{
  return impl_->dynamic_instances();
}

std::optional<std::uint32_t> BroadPhase::entity_id(const std::string& name) const
{
  return impl_->entity_id(name);
}

} // namespace ddrl::collision
