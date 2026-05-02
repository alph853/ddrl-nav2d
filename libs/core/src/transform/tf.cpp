#include "core/transform/tf.hpp"

#include <algorithm>
#include <cmath>

namespace ddrl::core::tf {

void TfBuffer::register_frame(std::string_view parent, std::string_view child)
{
  auto [cit, _]      = nodes_.try_emplace(std::string(child), TfNode{});
  cit->second.parent = std::string(parent); // store persistent copy
  nodes_.try_emplace(std::string(parent), TfNode{});
}

void TfBuffer::set_static_transform(std::string_view parent,
                                    std::string_view child,
                                    const Pose3&     t_parent_child)
{
  auto& n                 = nodes_[std::string(child)];
  n.parent                = parent;
  n.static_t_parent_child = t_parent_child;
  n.has_static            = true;
  nodes_.try_emplace(std::string(parent), TfNode{});
}

core::Result<void> TfBuffer::push_dynamic(std::string_view parent,
                                          std::string_view child,
                                          double           stamp,
                                          const Pose3&     t_parent_child)
{
  if (stamp < 0.0) {
    return std::unexpected(core::make_error(core::ErrorCode::TRANSFORM,
                                            "Negative timestamp not allowed",
                                            "TfBuffer::push_dynamic"));
  }

  auto& n  = nodes_[std::string(child)];
  n.parent = parent;
  auto& h  = n.history;
  if (!h.empty() && stamp < h.back().stamp) {
    auto it = std::ranges::upper_bound(h, stamp, std::less<>{}, &TfNode::TimedPose::stamp);

    h.insert(it, {stamp, t_parent_child});
  } else {
    h.push_back({stamp, t_parent_child});
  }
  while (h.size() > n.max_hist) {
    h.pop_front();
  }

  return {};
}

core::Result<Pose3>
TfBuffer::lookup(std::string_view target, std::string_view source, double stamp) const
{
  if (target == source) {
    return Pose3::identity();
  }

  auto up_target_res = collect_chain(target);
  if (!up_target_res) {
    return std::unexpected(core::make_error_from(
        up_target_res.error(),
        core::ErrorCode::TRANSFORM,
        std::format("TfBuffer::lookup >Error collecting target chain {}", target)));
  }
  auto up_source_res = collect_chain(source);
  if (!up_source_res) {
    return std::unexpected(core::make_error_from(
        up_source_res.error(),
        core::ErrorCode::TRANSFORM,
        std::format("TfBuffer::lookup >Error collecting source chain {}", source)));
  }
  auto up_target = std::move(up_target_res.value());
  auto up_source = std::move(up_source_res.value());

  std::ranges::reverse(up_target);
  std::ranges::reverse(up_source);
  size_t i = 0, max_i = std::min(up_target.size(), up_source.size());
  while (i < max_i && up_target[i].first == up_source[i].first) {
    ++i;
  }
  if (i == 0) {
    return std::unexpected(core::make_error(core::ErrorCode::TRANSFORM,
                                            "Frames are in different transform trees",
                                            "TfBuffer::lookup"));
  }
  const size_t split = i - 1;

  Pose3 t_lca_target = Pose3::identity();
  for (size_t k = split + 1; k < up_target.size(); ++k) {
    const TfNode* n     = up_target[k].second;
    auto          trans = resolve_edge(*n, stamp);
    t_lca_target        = t_lca_target.compose(trans);
  }
  Pose3 t_lca_source = Pose3::identity();
  for (size_t k = split + 1; k < up_source.size(); ++k) {
    const TfNode* n     = up_source[k].second;
    auto          trans = resolve_edge(*n, stamp);
    t_lca_source        = t_lca_source.compose(trans);
  }

  Pose3 t_target_lca = t_lca_target.inverse();
  return t_target_lca.compose(t_lca_source);
}

core::Result<std::vector<std::pair<std::string, const TfNode*>>>
TfBuffer::collect_chain(std::string_view frame) const
{
  std::vector<std::pair<std::string, const TfNode*>> out;

  auto it = nodes_.find(std::string(frame));
  if (it == nodes_.end()) {
    return std::unexpected(core::make_error(core::ErrorCode::TRANSFORM,
                                            std::format("Frame not found: {}", frame),
                                            "TfBuffer::collect_chain"));
  }
  const TfNode* n = &it->second;
  out.emplace_back(frame, n);
  std::string cur_parent = n->parent;
  while (!cur_parent.empty()) {
    auto itp = nodes_.find(cur_parent);
    if (itp == nodes_.end()) {
      return std::unexpected(core::make_error(core::ErrorCode::TRANSFORM,
                                              "Frame not found: " + cur_parent,
                                              "TfBuffer::collect_chain"));
    }
    n = &itp->second;
    out.emplace_back(cur_parent, n);
    cur_parent = n->parent;
  }
  return out;
}

Pose3 TfBuffer::resolve_edge(const TfNode& n, double stamp)
{
  if (!n.history.empty()) {
    const auto& h = n.history;
    if (stamp <= h.front().stamp) {
      return h.front().t_parent_child;
    }
    if (stamp >= h.back().stamp) {
      return h.back().t_parent_child;
    }
    auto        it = std::ranges::upper_bound(h, stamp, std::less<>{}, &TfNode::TimedPose::stamp);
    const auto& b  = *it;
    const auto& a  = *(it - 1);
    double      u  = (stamp - a.stamp) / (b.stamp - a.stamp);
    return Pose3::lerp(a.t_parent_child, b.t_parent_child, u);
  }
  if (n.has_static) {
    return n.static_t_parent_child;
  }
  return Pose3::identity();
}

core::Result<math::Point3> TfBuffer::transform_point(const math::Point3& point,
                                                            std::string_view    target,
                                                            std::string_view    source,
                                                            double              stamp) const
{
  auto tf_result = lookup(target, source, stamp);
  if (!tf_result.has_value()) {
    return std::unexpected(tf_result.error());
  }
  return tf_result.value().apply(point);
}

core::Result<math::Vec3> TfBuffer::transform_vector(const math::Vec3& vec,
                                                           std::string_view  target,
                                                           std::string_view  source,
                                                           double            stamp) const
{
  auto tf_result = lookup(target, source, stamp);
  if (!tf_result.has_value()) {
    return std::unexpected(tf_result.error());
  }
  return tf_result.value().apply_to_vector(vec);
}

} // namespace ddrl::core::tf
