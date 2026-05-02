"""
Model profile models matching C++ schemas from libs/config/include/config/model_profile.hpp
"""
from dataclasses import dataclass, field
from typing import List
from .geometry import Shape2p5, Pose2
from .material import MaterialRef


@dataclass
class ModelPart:
    """Single part of a model with geometry, material, and relative pose"""
    name: str = "part"
    geom: Shape2p5 = field(default_factory=Shape2p5)
    material: MaterialRef = field(default_factory=MaterialRef)
    rel_pose: Pose2 = field(default_factory=Pose2)

    def to_dict(self):
        return {
            "name": self.name,
            "geom": self.geom.to_dict(),
            "material": self.material.to_dict(),
            "rel_pose": self.rel_pose.to_dict()
        }


@dataclass
class ModelProfile:
    """Complete model composed of multiple parts"""
    name: str = "unnamed_model"
    category: str = "structure"  # vehicle, pedestrian, structure, etc.
    parts: List[ModelPart] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "parts": [part.to_dict() for part in self.parts]
        }
