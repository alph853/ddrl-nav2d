#pragma once

#include "core/geom/shape.hpp"
#include "core/math/primitives.hpp"
#include "core/sensors/sensor_models.hpp"
#include <limits>
#include <span>

#include "bvh.hpp"
#include "threading/thread_pool.hpp"

namespace ddrl::collision {

/**
 * @brief Ray query descriptor for intersection tests
 *
 * Describes a ray with optional range constraints for intersection testing
 * against the scene geometry.
 */
struct RayRequest {
  core::geom::Ray3 ray{};      ///< Ray origin and direction (direction should be normalized)
  double           t_min{0.0}; ///< Minimum distance along ray to consider (m)
  double t_max{std::numeric_limits<double>::infinity()}; ///< Maximum distance along ray (m)
};

/**
 * @brief Result of a ray-scene intersection test
 *
 * Contains information about the closest intersection point along the ray,
 * including geometric and object identification data.
 */
struct RayHit {
  bool   hit{false}; ///< True if ray intersected any geometry
  double distance{std::numeric_limits<double>::infinity()}; ///< Distance to hit point (m)
  core::math::Point3 position;                              ///< World-space hit position (m)
  core::math::Vec3   normal;        ///< World-space surface normal at hit point (normalized)
  std::uint32_t      entity_id{0};  ///< Unique entity identifier of the hit object
  std::uint32_t      part_index{0}; ///< Index of the specific part within the object's model
};

/**
 * @brief Trace a single ray through the scene
 *
 * Performs ray-scene intersection against both static and dynamic geometry
 * using the hierarchical BVH structures. Returns the closest hit along the ray.
 *
 * @param request Ray query parameters (origin, direction, range)
 * @param trees TLAS pair containing static and dynamic scene geometry
 * @return RayHit Closest intersection result (hit=false if no intersection)
 *
 * @note Ray direction in request.ray should be normalized for accurate distance results
 * @note Searches both static and dynamic trees, returning the nearest hit
 *
 * Example:
 * @code
 * RayRequest req;
 * req.ray.origin = {0.0, 0.0, 1.0};
 * req.ray.direction = {1.0, 0.0, 0.0}; // normalized
 * req.t_min = 0.1;  // ignore hits closer than 10cm
 * req.t_max = 100.0; // maximum range 100m
 *
 * RayHit hit = trace_ray(req, scene_trees);
 * if (hit.hit) {
 *   std::cout << "Hit at distance: " << hit.distance << "m\n";
 *   std::cout << "Entity ID: " << hit.entity_id << "\n";
 * }
 * @endcode
 */
[[nodiscard]] RayHit trace_ray(const RayRequest& request, const TLAS2Pair& trees);

/**
 * @brief Trace multiple rays through the scene with automatic parallelization
 *
 * Efficiently traces a batch of rays using thread pool parallelization when
 * beneficial. Automatically falls back to sequential execution for small batches
 * or single-threaded pools.
 *
 * @param rays Input ray queries (must have rays.size() == results.size())
 * @param results Output intersection results (pre-allocated, same size as rays)
 * @param trees TLAS pair containing static and dynamic scene geometry
 * @param pool Thread pool for parallel execution
 * @param batch_threshold Minimum number of rays to enable parallelization (default: 32)
 *
 * @throws std::runtime_error if rays.size() != results.size()
 *
 * @note Results are written directly to the provided span
 * @note Parallelization is automatically disabled if pool.size() <= 1 or rays.size() <
 * batch_threshold
 * @note Each thread processes a chunk of rays to minimize overhead
 */
void trace_rays(
  std::span<const RayRequest> rays, std::span<RayHit> results, const TLAS2Pair& trees,
  threading::ThreadPool& pool
);

//=============================================================================
// Ray Generation Utilities
//=============================================================================

/**
 * @brief Generate rays for a pinhole depth camera
 *
 * Creates rays for each pixel in a depth camera image using the pinhole camera
 * model. Rays originate at the sensor position and point through each pixel in
 * the image plane according to the camera's intrinsic parameters.
 *
 * @param sensor_pose_world Camera pose in world frame (position and orientation)
 * @param config Depth camera configuration (intrinsics, resolution, range)
 * @param out_rays Output ray buffer (cleared and populated with width×height rays)
 *
 * @note Rays are generated in row-major order: pixel (u,v) → ray index (v*width + u)
 * @note Ray directions are transformed from camera frame to world frame using sensor_pose_world
 * @note Each ray's t_min and t_max are set from config.base.min_range and max_range
 *
 * Example:
 * @code
 * core::sensors::DepthCamera cam_config;
 * cam_config.width = 640;
 * cam_config.height = 480;
 * cam_config.fx = 500.0;
 * cam_config.fy = 500.0;
 * cam_config.cx = 320.0;
 * cam_config.cy = 240.0;
 * cam_config.base.min_range = 0.1;
 * cam_config.base.max_range = 10.0;
 *
 * core::math::Pose3 camera_pose = ...;
 * std::vector<RayRequest> rays;
 * generate_depth_camera_rays(camera_pose, cam_config, rays);
 * // rays.size() == 640 * 480
 * @endcode
 */
void generate_depth_camera_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::DepthCamera& config,
  std::vector<RayRequest>& out_rays
);

/**
 * @brief Generate rays for a 2D planar lidar scan
 *
 * Creates rays for a 2D lidar scan in a horizontal plane. Rays are uniformly
 * distributed across the field of view with the specified angular resolution.
 *
 * @param sensor_pose_world Lidar pose in world frame
 * @param config 2D lidar configuration (FOV, angular resolution, range)
 * @param out_rays Output ray buffer (cleared and populated)
 *
 * @note Number of rays = ceil(FOV / angular_res)
 * @note Rays fan out in the XY plane (z component = 0 in sensor frame)
 * @note First ray points at -FOV/2 relative to sensor forward direction
 *
 * Example:
 * @code
 * core::sensors::Lidar2D lidar_config;
 * lidar_config.fov = M_PI;  // 180 degrees
 * lidar_config.angular_res = M_PI / 180.0;  // 1 degree resolution
 * lidar_config.base.min_range = 0.1;
 * lidar_config.base.max_range = 30.0;
 *
 * core::math::Pose3 lidar_pose = ...;
 * std::vector<RayRequest> rays;
 * generate_lidar_2d_rays(lidar_pose, lidar_config, rays);
 * // rays.size() == 180 (180 degree FOV / 1 degree resolution)
 * @endcode
 */
void generate_lidar_2d_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::Lidar2D& config,
  std::vector<RayRequest>& out_rays
);

/**
 * @brief Generate rays for a 3D multi-ring lidar scan
 *
 * Creates rays for a 3D lidar with multiple vertical rings. Rays are distributed
 * uniformly in azimuth (horizontal) and across vertical rings to create a
 * spherical scan pattern.
 *
 * @param sensor_pose_world Lidar pose in world frame
 * @param config 3D lidar configuration (horizontal/vertical FOV, rings, range)
 * @param out_rays Output ray buffer (cleared and populated)
 *
 * @note Total rays = ceil(hfov / angular_res) × num_rings
 * @note Vertical rings are evenly distributed across vfov
 * @note Azimuth rays start at -hfov/2 and increment by angular_res
 * @note Elevation ranges from -vfov/2 to +vfov/2
 *
 * Example:
 * @code
 * core::sensors::Lidar3D lidar_config;
 * lidar_config.hfov = 2.0 * M_PI;  // 360 degrees horizontal
 * lidar_config.vfov = M_PI / 6.0;  // 30 degrees vertical
 * lidar_config.angular_res = M_PI / 180.0;  // 1 degree azimuth resolution
 * lidar_config.num_rings = 16;  // 16 vertical channels
 * lidar_config.base.min_range = 0.5;
 * lidar_config.base.max_range = 100.0;
 *
 * core::math::Pose3 lidar_pose = ...;
 * std::vector<RayRequest> rays;
 * generate_lidar_3d_rays(lidar_pose, lidar_config, rays);
 * // rays.size() == 360 * 16 = 5760
 * @endcode
 */
void generate_lidar_3d_rays(
  const core::math::Pose3& sensor_pose_world, const core::sensors::Lidar3D& config,
  std::vector<RayRequest>& out_rays
);

} // namespace ddrl::collision
