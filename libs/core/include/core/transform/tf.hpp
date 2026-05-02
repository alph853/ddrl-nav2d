/**
 * @file tf.hpp
 * @brief Transform tree management for coordinate frame transformations
 */
#pragma once

#include "core/base/result.hpp"
#include "core/math/pose.hpp"
#include <deque>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ddrl::core::tf {

using math::Pose3;

/**
 * @brief Transform tree node representing one coordinate frame
 */
struct TfNode {
  std::string parent;                ///< Parent frame name (empty if root)
  Pose3       static_t_parent_child; ///< Static transform from parent to child
  bool        has_static{false};     ///< Whether static transform is set

  /**
   * @brief Timestamped pose sample for dynamic transforms
   */
  struct TimedPose {
    double stamp;          ///< Timestamp
    Pose3  t_parent_child; ///< Transform from parent to child
  };

  std::deque<TimedPose> history;       ///< Dynamic pose history (time-sorted)
  size_t                max_hist{256}; ///< Maximum history length
};

/**
 * @brief Transform buffer managing coordinate frame relationships
 *
 * Maintains a tree of coordinate frames with both static and dynamic
 * transforms, supporting interpolation for dynamic transforms.
 */
class TfBuffer
{
public:
  /**
   * @brief Register frame parentage without setting transform
   * @param parent Parent frame name
   * @param child Child frame name
   */
  void register_frame(std::string_view parent, std::string_view child);

  /**
   * @brief Set static transform between frames
   * @param parent Parent frame name
   * @param child Child frame name
   * @param t_parent_child Transform from parent to child
   */
  void set_static_transform(std::string_view parent,
                            std::string_view child,
                            const Pose3&       t_parent_child);

  /**
   * @brief Add dynamic transform sample
   * @param parent Parent frame name
   * @param child Child frame name
   * @param stamp Timestamp
   * @param t_parent_child Transform from parent to child
   * @return Result indicating success or error
   */
  core::Result<void> push_dynamic(std::string_view parent,
                                  std::string_view child,
                                  double             stamp,
                                  const Pose3&       t_parent_child);

  /**
   * @brief Lookup transform between frames at given time
   * @param target Target frame name
   * @param source Source frame name
   * @param stamp Query timestamp
   * @return Transform from source to target, or error if not found
   */
  core::Result<Pose3>
  lookup(std::string_view target, std::string_view source, double stamp) const;

  /**
   * @brief Transform a point from source frame to target frame
   * @param point Point in source frame
   * @param target Target frame name
   * @param source Source frame name
   * @param stamp Query timestamp
   * @return Point in target frame, or error if transform not found
   */
  core::Result<math::Point3> transform_point(const math::Point3& point,
                                             std::string_view  target,
                                             std::string_view  source,
                                             double              stamp) const;

  /**
   * @brief Transform a vector from source frame to target frame
   * @param vec Vector in source frame
   * @param target Target frame name
   * @param source Source frame name
   * @param stamp Query timestamp
   * @return Vector in target frame, or error if transform not found
   */
  core::Result<math::Vec3> transform_vector(const math::Vec3&  vec,
                                            std::string_view target,
                                            std::string_view source,
                                            double             stamp) const;

private:
  /**
   * @brief Collect the chain of frames from a frame up to the tree root.
   *
   * The output vector receives pairs of (frame name, node pointer) in the order:
   *   frame (child), parent, ..., root.
   *
   * @param frame Starting frame name.
   * @return Output vector to which the chain entries are appended.
   */
  core::Result<std::vector<std::pair<std::string, const TfNode*>>>
  collect_chain(std::string_view frame) const;

  /**
   * @brief Resolve (find) the parent -> child transform for a node at a given time.
   *
   * @param n Node whose edge (parent→child) is to be resolved.
   * @param stamp Query timestamp (seconds).
   * @return Pose3 transform from parent to child at @p stamp.
   */
  static Pose3 resolve_edge(const TfNode& n, double stamp);

  std::unordered_map<std::string, TfNode> nodes_;
};

} // namespace ddrl::core::tf

inline std::ostream& operator<<(std::ostream& os, const ddrl::core::tf::TfNode::TimedPose& tp)
{
  return os << "TimedPose{" << tp.stamp << ", " << tp.t_parent_child << "}";
}
