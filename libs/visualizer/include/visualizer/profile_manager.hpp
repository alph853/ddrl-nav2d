#pragma once

#include "core/math/pose.hpp"
#include <array>
#include <string>
#include <unordered_map>
#include <vector>
#include <vtkSmartPointer.h>

#include "config/world.hpp"

// Forward declarations
class vtkPolyData;
class vtkActor;
class vtkMultiBlockDataSet;
class vtkCompositeDataDisplayAttributes;

namespace ddrl::visualizer {

/**
 * @brief Manages reusable VTK geometry templates for model profiles
 *
 * ProfileManager builds VTK geometry once per model_id from WorldConfig,
 * then provides lightweight actor instances that share the same geometry.
 * This significantly reduces memory usage and initialization overhead.
 */
class ProfileManager
{
public:
  /**
   * @brief Geometry part for a single shape in a model
   */
  struct GeometryPart {
    vtkSmartPointer<vtkPolyData> polydata;   ///< Pre-built VTK geometry
    core::math::Pose2            rel_pose;   ///< Relative pose from model origin
    std::array<double, 3>        base_color; ///< Base RGB color
  };

  /**
   * @brief Complete geometry template for a model profile
   */
  struct ModelGeometry {
    std::vector<GeometryPart>                         parts;         ///< All geometry parts
    vtkSmartPointer<vtkMultiBlockDataSet>             multiblock;    ///< Combined geometry
    std::vector<std::array<double, 3>>                block_colors;  ///< Cached colors per block
    std::string                                       category;      ///< Model category (for debug)
  };

  /**
   * @brief Initialize and build all geometry templates from world config
   * @param world_config Configuration containing model profiles
   */
  explicit ProfileManager(const config::WorldConfig& world_config);

  /**
   * @brief Get geometry template for a model
   * @param model_id Model identifier from config
   * @return Pointer to geometry template, or nullptr if not found
   */
  [[nodiscard]] const ModelGeometry* get_model_geometry(const std::string& model_id) const;

  /**
   * @brief Create actor instances from a model template
   * @param model_id Model identifier
   * @return Vector of actors (one per part), or empty if model not found
   * @note Each actor shares the polydata from the template (memory efficient)
   */
  [[nodiscard]] vtkSmartPointer<vtkActor>
  create_instance_actor(const std::string& model_id) const;

  [[nodiscard]] std::string get_all_profile_ids() const;

private:
  std::unordered_map<std::string, ModelGeometry> geometry_cache_;
};

} // namespace ddrl::visualizer
