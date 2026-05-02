/**
 * @file profile_manager.cpp
 * @brief Implementation of ProfileManager for geometry caching
 */

#include "visualizer/profile_manager.hpp"

#include <cmath>
#include <vtkActor.h>
#include <vtkCompositePolyDataMapper2.h>
#include <vtkCompositeDataDisplayAttributes.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include "logging/logging.hpp"
#include "visualizer/widgets/utils.hpp"

namespace ddrl::visualizer {

ProfileManager::ProfileManager(const config::WorldConfig& world_config)
{
  auto logger = logging::get_logger("ProfileManager");

  // Build geometry templates for all model profiles
  for (const auto& [model_id, model_profile] : world_config.model_profiles) {
    ModelGeometry model_geom;
    model_geom.category = model_profile.category;

    model_geom.multiblock = vtkSmartPointer<vtkMultiBlockDataSet>::New();

    for (size_t part_index = 0; part_index < model_profile.parts.size(); ++part_index) {
      const auto& part      = model_profile.parts[part_index];
      const auto& shape2p5 = part.geom;

      auto polydata = widgets::utils::create_shape_polydata(shape2p5);

      std::array<double, 3> base_color = {
        static_cast<double>(part.material.rgba[0]),
        static_cast<double>(part.material.rgba[1]),
        static_cast<double>(part.material.rgba[2])
      };

      GeometryPart geom_part;
      geom_part.polydata   = polydata;
      geom_part.rel_pose   = part.rel_pose;
      geom_part.base_color = base_color;

      model_geom.parts.push_back(geom_part);

      // Build transformed polydata for combined multiblock
      auto transformed_poly = vtkSmartPointer<vtkPolyData>::New();
      if (std::abs(part.rel_pose.yaw) > 1e-6 || std::abs(part.rel_pose.pos.x) > 1e-6 ||
          std::abs(part.rel_pose.pos.y) > 1e-6) {
        auto transform = vtkSmartPointer<vtkTransform>::New();
        transform->Identity();
        // Translate in the part's local frame (after rotation), not world frame
        transform->Translate(part.rel_pose.pos.x, part.rel_pose.pos.y, 0.0);
        transform->RotateZ(part.rel_pose.yaw * 180.0 / M_PI);

        auto transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
        transform_filter->SetTransform(transform);
        transform_filter->SetInputData(polydata);
        transform_filter->Update();
        transformed_poly->DeepCopy(transform_filter->GetOutput());
      } else {
        transformed_poly->DeepCopy(polydata);
      }

      model_geom.multiblock->SetBlock(static_cast<unsigned int>(part_index), transformed_poly);
      model_geom.block_colors.push_back(base_color);
    }

    // Cache the complete model geometry
    geometry_cache_[model_id] = std::move(model_geom);

    DDRL_LOG_DEBUG(logger,
                  "ProfileManager: Built geometry for model '{}' with {} parts",
                  model_id,
                  geometry_cache_[model_id].parts.size());
  }

  DDRL_LOG_DEBUG(logger, "ProfileManager: Initialized with {} model profiles", geometry_cache_.size());
}

const ProfileManager::ModelGeometry*
ProfileManager::get_model_geometry(const std::string& model_id) const
{
  auto it = geometry_cache_.find(model_id);
  if (it != geometry_cache_.end()) {
    return &it->second;
  }
  return nullptr;
}

vtkSmartPointer<vtkActor> ProfileManager::create_instance_actor(const std::string& model_id) const
{
  const auto* model_geom = get_model_geometry(model_id);
  if (model_geom == nullptr) {
    auto logger = logging::get_logger("ProfileManager");
    DDRL_LOG_WARN(logger, "Model '{}' not found in geometry cache", model_id);
    DDRL_LOG_WARN(logger, "All profiles ids: {}", get_all_profile_ids());
    return nullptr;
  }

  if (model_geom->multiblock == nullptr) {
    auto logger = logging::get_logger("ProfileManager");
    DDRL_LOG_WARN(logger, "Model '{}' has no cached multiblock geometry", model_id);
    return nullptr;
  }

  auto mapper = vtkSmartPointer<vtkCompositePolyDataMapper2>::New();
  mapper->SetInputDataObject(model_geom->multiblock);
  mapper->SetScalarVisibility(0);

  auto attributes = vtkSmartPointer<vtkCompositeDataDisplayAttributes>::New();
  for (unsigned int block_index = 0;
       block_index < model_geom->block_colors.size();
       ++block_index) {
    const auto& color = model_geom->block_colors[block_index];
    if (auto* block = model_geom->multiblock->GetBlock(block_index)) {
      attributes->SetBlockColor(block, color.data());
    }
  }
  mapper->SetCompositeDataDisplayAttributes(attributes);

  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  return actor;
}

std::string ProfileManager::get_all_profile_ids() const
{
  std::string ids;
  for (const auto& [model_id, _] : geometry_cache_) {
    if (!ids.empty()) {
      ids += ", ";
    }
    ids += model_id;
  }
  return ids;
}

} // namespace ddrl::visualizer
