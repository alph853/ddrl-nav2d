#include "collision/ray_casting.hpp"

#include "core/math/numeric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ddrl::collision {

using ddrl::core::geom::AABB2;
using ddrl::core::geom::Ray3;
using ddrl::core::geom::Shape2p5;
using ddrl::core::math::kEps;
using ddrl::core::math::Point2;
using ddrl::core::math::Point3;
using ddrl::core::math::Pose2;
using ddrl::core::math::Vec2;
using ddrl::core::math::Vec3;

namespace {
constexpr std::size_t kMinRaysPerTask          = 1024;
constexpr std::size_t kMaxQueuedTasksPerWorker = 4;

struct IntervalHit {
  bool   valid{false};
  double enter{0.0};
  double exit{0.0};
  Vec2   normal{}; // outward, normalized
};

[[nodiscard]] IntervalHit
intersect_polygon_xy(const Ray3& ray, std::span<const Point2> verts, double t_min, double t_max)
{
  IntervalHit hit;
  if (verts.empty()) {
    return hit;
  }

  const Vec2   origin{ray.origin.x, ray.origin.y};
  const Vec2   dir{ray.direction.x, ray.direction.y};
  const double dir_len_sq = dir.length_sq();

  if (dir_len_sq <= kEps) {
    if (!core::math::point_in_polygon(verts, {origin.x, origin.y})) {
      return hit;
    }
    hit.valid  = true;
    hit.enter  = t_min;
    hit.exit   = t_max;
    hit.normal = {0.0, 0.0};
    return hit;
  }

  double t_enter = t_min;
  double t_exit  = t_max;
  Vec2   best_normal{0.0, 0.0};

  for (std::size_t i = 0; i < verts.size(); ++i) {
    const Point2& vi = verts[i];
    const Point2& vj = verts[(i + 1) % verts.size()];
    const Vec2    edge{vj.x - vi.x, vj.y - vi.y};
    if (edge.length_sq() <= kEps) {
      continue;
    }
    const Vec2   normal{edge.y, -edge.x}; // outward normal
    const double denom = normal.dot(dir);
    const Vec2   w     = origin - Vec2{vi.x, vi.y};
    const double numer = normal.dot(w);

    if (std::abs(denom) <= kEps) {
      if (numer > 0.0) {
        return IntervalHit{}; // outside and parallel
      }
      continue;
    }

    const double t = -numer / denom;
    if (denom < 0.0) {
      if (t > t_enter) {
        t_enter     = t;
        best_normal = normal;
      }
    } else {
      t_exit = std::min(t_exit, t);
    }
    if (t_enter > t_exit) {
      return IntervalHit{};
    }
  }

  hit.valid = true;
  hit.enter = t_enter;
  hit.exit  = t_exit;
  if (best_normal.length_sq() > kEps) {
    hit.normal = best_normal.normalized();
  }
  return hit;
}

[[nodiscard]] IntervalHit
intersect_circle_xy(const Ray3& ray, double radius, double t_min, double t_max)
{
  IntervalHit  hit;
  const Vec2   origin{ray.origin.x, ray.origin.y};
  const Vec2   dir{ray.direction.x, ray.direction.y};
  const double dir_len_sq = dir.length_sq();
  const double radius_sq  = radius * radius;

  if (dir_len_sq <= kEps) {
    if (origin.length_sq() > radius_sq) {
      return hit;
    }
    hit.valid  = true;
    hit.enter  = t_min;
    hit.exit   = t_max;
    hit.normal = {0.0, 0.0};
    return hit;
  }

  const double a    = dir_len_sq;
  const double b    = 2.0 * origin.dot(dir);
  const double c    = origin.dot(origin) - radius_sq;
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) {
    return hit;
  }
  const double sqrt_disc = std::sqrt(disc);
  double       t0        = (-b - sqrt_disc) / (2.0 * a);
  double       t1        = (-b + sqrt_disc) / (2.0 * a);
  if (t0 > t1) {
    std::swap(t0, t1);
  }

  if (t1 < t_min || t0 > t_max) {
    return hit;
  }

  hit.valid              = true;
  hit.enter              = std::max(t0, t_min);
  hit.exit               = std::min(t1, t_max);
  const Vec2   hit_point = origin + dir * hit.enter;
  const double len       = hit_point.length();
  if (len > kEps) {
    hit.normal = (hit_point / len);
  }
  return hit;
}

struct LocalIntersection {
  bool   hit{false};
  double t{0.0};
  Vec2   normal_xy{};      // local-space
  int    normal_z_sign{0}; // -1 bottom, +1 top, 0 for lateral
};

[[nodiscard]] LocalIntersection intersect_local(
  const Ray3& ray, const Shape2p5& shape, double t_min, double t_max, double eps = kEps
)
{
  LocalIntersection result;

  IntervalHit xy_hit;
  auto        shape_base = shape.base.value;
  if (std::holds_alternative<core::geom::Circle>(shape_base)) {
    const auto& circle = std::get<core::geom::Circle>(shape_base);
    xy_hit             = intersect_circle_xy(ray, circle.radius, t_min, t_max);
  } else if (std::holds_alternative<core::geom::Box2>(shape_base)) {
    const auto&                 box = std::get<core::geom::Box2>(shape_base);
    const std::array<Point2, 4> verts{
      {{-box.hx, -box.hy}, {box.hx, -box.hy}, {box.hx, box.hy}, {-box.hx, box.hy}}
    };
    xy_hit = intersect_polygon_xy(ray, verts, t_min, t_max);
  } else {
    const auto& poly = std::get<core::geom::Polygon2>(shape_base);
    xy_hit           = intersect_polygon_xy(ray, poly.v, t_min, t_max);
  }

  if (!xy_hit.valid) {
    return result;
  }

  const double z_min = std::min(shape.z_min, shape.z_max);
  const double z_max = std::max(shape.z_min, shape.z_max);
  const double dz    = ray.direction.z;
  const double oz    = ray.origin.z;

  double tz_enter = t_min;
  double tz_exit  = t_max;
  int    tz_face  = 0;

  if (std::abs(dz) <= eps) {
    if (oz < z_min || oz > z_max) {
      return result;
    }
  } else {
    const double inv_dz = 1.0 / dz;
    double       t0     = (z_min - oz) * inv_dz;
    double       t1     = (z_max - oz) * inv_dz;
    tz_enter            = std::min(t0, t1);
    tz_exit             = std::max(t0, t1);
    tz_face             = (t0 < t1) ? -1 : 1;
  }

  const double t_enter = std::max({xy_hit.enter, tz_enter, t_min});
  const double t_exit  = std::min({xy_hit.exit, tz_exit, t_max});

  if (t_enter > t_exit || t_exit < t_min) {
    return result;
  }

  result.hit = true;
  result.t   = t_enter;

  const double t_xy_enter = std::max(xy_hit.enter, t_min);
  const double t_z_enter  = std::max(tz_enter, t_min);

  if (xy_hit.normal.length_sq() > kEps && std::abs(t_enter - t_xy_enter) <= eps) {
    result.normal_xy     = xy_hit.normal;
    result.normal_z_sign = 0;
  } else if (tz_face != 0 && std::abs(t_enter - t_z_enter) <= eps) {
    result.normal_xy     = {0.0, 0.0};
    result.normal_z_sign = tz_face;
  } else if (xy_hit.normal.length_sq() > kEps) {
    result.normal_xy     = xy_hit.normal;
    result.normal_z_sign = 0;
  } else {
    result.normal_xy     = {0.0, 0.0};
    result.normal_z_sign = (dz > 0.0) ? -1 : (dz < 0.0 ? 1 : 0);
  }

  return result;
}

[[nodiscard]] Ray3 to_local(const Ray3& ray, const Pose2& world_from_local)
{
  Ray3         out;
  const Point2 origin_local =
    world_from_local.inverse_transform_point({ray.origin.x, ray.origin.y});
  const Vec2 dir_local =
    world_from_local.inverse_transform_vector({ray.direction.x, ray.direction.y});
  out.origin    = {origin_local.x, origin_local.y, ray.origin.z};
  out.direction = {dir_local.x, dir_local.y, ray.direction.z};
  return out;
}

[[nodiscard]] bool intersect_aabb2(const Ray3& ray, const AABB2& bounds, double t_min, double t_max)
{
  double       t0 = t_min;
  double       t1 = t_max;
  const double ox = ray.origin.x;
  const double oy = ray.origin.y;
  const double dx = ray.direction.x;
  const double dy = ray.direction.y;

  if (std::abs(dx) <= kEps) {
    if (ox < bounds.min.x || ox > bounds.max.x) {
      return false;
    }
  } else {
    double inv       = 1.0 / dx;
    double tmin_axis = (bounds.min.x - ox) * inv;
    double tmax_axis = (bounds.max.x - ox) * inv;
    if (tmin_axis > tmax_axis) {
      std::swap(tmin_axis, tmax_axis);
    }
    t0 = std::max(t0, tmin_axis);
    t1 = std::min(t1, tmax_axis);
    if (t0 > t1) {
      return false;
    }
  }

  if (std::abs(dy) <= kEps) {
    if (oy < bounds.min.y || oy > bounds.max.y) {
      return false;
    }
  } else {
    double inv       = 1.0 / dy;
    double tmin_axis = (bounds.min.y - oy) * inv;
    double tmax_axis = (bounds.max.y - oy) * inv;
    if (tmin_axis > tmax_axis) {
      std::swap(tmin_axis, tmax_axis);
    }
    t0 = std::max(t0, tmin_axis);
    t1 = std::min(t1, tmax_axis);
    if (t0 > t1) {
      return false;
    }
  }

  return t1 >= t_min && t0 <= t_max;
}

struct InstanceHit {
  bool          hit{false};
  double        distance{std::numeric_limits<double>::infinity()};
  Vec3          normal;
  std::uint32_t entity_id{0};
  std::uint32_t part_index{0};
};

[[nodiscard]] InstanceHit
trace_instance(const RayRequest& req, const TLAS2::Leaf& leaf, double current_best)
{
  InstanceHit best;
  if (!leaf.blas || !leaf.model) {
    return best;
  }

  const auto& model     = *leaf.model;
  const auto& blas      = *leaf.blas;
  const auto  ray_model = to_local(req.ray, leaf.world_from_model);

  double      best_so_far = std::min(current_best, req.t_max);
  const auto& blas_leaves = blas.leaves();

  for (const auto& blas_leaf : blas_leaves) {
    if (!intersect_aabb2(ray_model, blas_leaf.bounds, req.t_min, best_so_far)) {
      continue;
    }

    const std::size_t part_index = blas_leaf.part_index;
    if (part_index >= model.parts.size()) {
      continue;
    }

    const Pose2& part_pose_local = model.parts[part_index].rel_pose;
    const Pose2  part_world      = leaf.world_from_model * part_pose_local;

    const Ray3              ray_part = to_local(req.ray, part_world);
    const auto&             shape    = model.parts[part_index];
    const LocalIntersection local_hit =
      intersect_local(ray_part, shape.geom, req.t_min, best_so_far);
    if (!local_hit.hit) {
      continue;
    }

    const double t = local_hit.t;
    if (t >= best_so_far || t < req.t_min) {
      continue;
    }

    best.hit        = true;
    best.distance   = t;
    best_so_far     = t;
    best.entity_id  = leaf.entity_id;
    best.part_index = static_cast<std::uint32_t>(part_index);

    if (local_hit.normal_z_sign != 0) {
      best.normal = {0.0, 0.0, static_cast<double>(local_hit.normal_z_sign)};
    } else if (local_hit.normal_xy.length_sq() > kEps) {
      const Vec2   world_xy = part_world.transform_vector(local_hit.normal_xy);
      Vec3         n{world_xy.x, world_xy.y, 0.0};
      const double len = n.length();
      best.normal      = (len > kEps) ? (n / len) : Vec3{0.0, 0.0, 0.0};
    } else {
      best.normal = {0.0, 0.0, 0.0};
    }
  }

  return best;
}

[[nodiscard]] RayHit trace_tree(const RayRequest& req, const TLAS2& tree, double current_best)
{
  RayHit best_hit;
  best_hit.distance = current_best;
  if (tree.empty()) {
    return best_hit;
  }

  const auto  nodes  = tree.nodes();
  const auto& leaves = tree.leaves();
  if (tree.root() == BVHNode2::kInvalid) {
    return best_hit;
  }

  std::vector<std::uint32_t> stack;
  stack.reserve(nodes.size());
  stack.push_back(tree.root());

  while (!stack.empty()) {
    const auto node_idx = stack.back();
    stack.pop_back();
    const auto& node = nodes[node_idx];
    if (!intersect_aabb2(req.ray, node.bounds, req.t_min, best_hit.distance)) {
      continue;
    }
    if (node.is_leaf()) {
      const auto&       leaf     = leaves[node.leaf];
      const InstanceHit inst_hit = trace_instance(req, leaf, best_hit.distance);
      if (inst_hit.hit && inst_hit.distance < best_hit.distance) {
        best_hit.hit        = true;
        best_hit.distance   = inst_hit.distance;
        best_hit.entity_id  = inst_hit.entity_id;
        best_hit.part_index = inst_hit.part_index;
        best_hit.position   = req.ray.at(inst_hit.distance);
        best_hit.normal     = inst_hit.normal;
      }
      continue;
    }
    if (node.left != BVHNode2::kInvalid) {
      stack.push_back(node.left);
    }
    if (node.right != BVHNode2::kInvalid) {
      stack.push_back(node.right);
    }
  }

  return best_hit;
}

} // namespace

RayHit trace_ray(const RayRequest& request, const TLAS2Pair& trees)
{
  RayHit best;
  best.distance = request.t_max;

  best = trace_tree(request, trees.static_tree, best.distance);
  if (!best.hit) {
    best = trace_tree(request, trees.dynamic_tree, best.distance);
  } else {
    const RayHit dynamic_hit = trace_tree(request, trees.dynamic_tree, best.distance);
    if (dynamic_hit.hit && dynamic_hit.distance < best.distance) {
      best = dynamic_hit;
    }
  }

  return best;
}

void trace_rays(
  std::span<const RayRequest> rays, std::span<RayHit> results, const TLAS2Pair& trees,
  threading::ThreadPool& pool
)
{
  if (rays.size() != results.size()) {
    throw std::runtime_error("trace_rays size mismatch");
  }
  if (rays.empty()) {
    return;
  }

  const std::size_t pool_threads   = std::max<std::size_t>(1, pool.size());
  const std::size_t usable_workers =
    std::min(pool_threads, rays.size() / kMinRaysPerTask);
  const std::size_t backlog_limit = pool_threads * kMaxQueuedTasksPerWorker;
  const bool        pool_overloaded =
    (backlog_limit > 0) && (pool.pending_tasks() >= backlog_limit);

  if (usable_workers <= 1 || pool_overloaded) {
    for (std::size_t i = 0; i < rays.size(); ++i) {
      results[i] = trace_ray(rays[i], trees);
    }
    return;
  }

  const std::size_t thread_count = usable_workers;
  const std::size_t chunk =
    std::max<std::size_t>(1, (rays.size() + thread_count - 1) / thread_count);
  std::vector<std::future<void>> futures;
  futures.reserve(thread_count);

  for (std::size_t begin = 0; begin < rays.size(); begin += chunk) {
    const std::size_t end = std::min(begin + chunk, rays.size());
    futures.emplace_back(pool.submit([&trees, rays, results, begin, end]() {
      for (std::size_t i = begin; i < end; ++i) {
        results[i] = trace_ray(rays[i], trees);
      }
    }));
  }

  for (auto& f : futures) {
    f.get();
  }
}

//=============================================================================
// Ray Generation Utilities
//=============================================================================

void generate_depth_camera_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::DepthCamera& config,
  std::vector<RayRequest>& out_rays
)
{
  const auto& cam      = config;
  const auto  num_rays = static_cast<size_t>(cam.width * cam.height);
  out_rays.clear();
  out_rays.reserve(num_rays);

  const auto fx = cam.fx;
  const auto fy = cam.fy;
  const auto cx = cam.cx;
  const auto cy = cam.cy;

  // Generate rays for each pixel
  for (uint16_t v = 0; v < cam.height; ++v) {
    for (uint16_t u = 0; u < cam.width; ++u) {
      // Pixel center in camera frame
      const double x_cam =
        (static_cast<double>(u) - static_cast<double>(cx)) / static_cast<double>(fx);
      const double y_cam =
        (static_cast<double>(v) - static_cast<double>(cy)) / static_cast<double>(fy);
      const double z_cam = 1.0;

      // Ray direction in camera frame (camera looks along +Z)
      Vec3         dir_cam{x_cam, y_cam, z_cam};
      const double len = dir_cam.length();
      if (len > 1e-9) {
        dir_cam = dir_cam / len;
      }

      // Transform ray direction to world frame using quaternion
      const Vec3 dir_world = sensor_pose_world.apply_to_vector(dir_cam);

      RayRequest req;
      req.ray.origin    = sensor_pose_world.t;
      req.ray.direction = dir_world;
      req.t_min         = static_cast<double>(cam.base.min_range);
      req.t_max         = static_cast<double>(cam.base.max_range);

      out_rays.push_back(req);
    }
  }
}

void generate_lidar_2d_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::Lidar2D& config,
  std::vector<RayRequest>& out_rays
)
{
  const auto& lidar    = config;
  const auto  num_rays = static_cast<size_t>(std::ceil(lidar.fov / lidar.angular_res));
  out_rays.clear();
  out_rays.reserve(num_rays);

  const double start_angle = -lidar.fov / 2.0;

  for (size_t i = 0; i < num_rays; ++i) {
    const double angle_rel = start_angle + static_cast<double>(i) * lidar.angular_res;

    // Ray direction in sensor frame (2D scan in XY plane)
    Vec3 dir_sensor;
    dir_sensor.x = std::cos(angle_rel);
    dir_sensor.y = std::sin(angle_rel);
    dir_sensor.z = 0.0; // Planar scan

    // Transform to world frame
    const Vec3 dir_world = sensor_pose_world.apply_to_vector(dir_sensor);

    RayRequest req;
    req.ray.origin    = sensor_pose_world.t;
    req.ray.direction = dir_world;
    req.t_min         = static_cast<double>(lidar.base.min_range);
    req.t_max         = static_cast<double>(lidar.base.max_range);

    out_rays.push_back(req);
  }
}

void generate_lidar_3d_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::Lidar3D& config,
  std::vector<RayRequest>& out_rays
)
{
  const auto& lidar       = config;
  const auto  num_azimuth = static_cast<size_t>(std::ceil(lidar.hfov / lidar.angular_res));
  const auto  num_rings   = lidar.num_rings;

  out_rays.clear();
  out_rays.reserve(static_cast<size_t>(num_azimuth * num_rings));

  const double start_azimuth   = -lidar.hfov / 2.0;
  const double start_elevation = -lidar.vfov / 2.0;
  const double elevation_step =
    (num_rings > 1) ? (lidar.vfov / (static_cast<double>(num_rings) - 1)) : 0.0;

  for (size_t ring = 0; ring < num_rings; ++ring) {
    const double elevation = start_elevation + static_cast<double>(ring) * elevation_step;

    for (size_t i = 0; i < num_azimuth; ++i) {
      const double azimuth_rel = start_azimuth + static_cast<double>(i) * lidar.angular_res;

      // Spherical to Cartesian in sensor frame
      const double cos_elev = std::cos(elevation);
      Vec3         dir_sensor;
      dir_sensor.x = cos_elev * std::cos(azimuth_rel);
      dir_sensor.y = cos_elev * std::sin(azimuth_rel);
      dir_sensor.z = std::sin(elevation);

      // Transform to world frame
      const Vec3 dir_world = sensor_pose_world.apply_to_vector(dir_sensor);

      RayRequest req;
      req.ray.origin    = sensor_pose_world.t;
      req.ray.direction = dir_world;
      req.t_min         = static_cast<double>(lidar.base.min_range);
      req.t_max         = static_cast<double>(lidar.base.max_range);

      out_rays.push_back(req);
    }
  }
}

} // namespace ddrl::collision
