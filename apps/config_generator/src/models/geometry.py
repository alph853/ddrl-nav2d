"""
Geometry models matching C++ schemas from libs/core/include/core/geom/shape.hpp
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import numpy as np


@dataclass
class Circle:
    """2D circle with radius"""
    radius: float = 1.0

    def to_dict(self):
        return {"type": "circle", "radius": self.radius}


@dataclass
class Box2:
    """2D box with half-extents"""
    hx: float = 1.0  # half width
    hy: float = 1.0  # half height

    def to_dict(self):
        return {"type": "box", "hx": self.hx, "hy": self.hy}


@dataclass
class Polygon2:
    """2D polygon with CCW winding vertices"""
    v: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self):
        return {
            "type": "polygon",
            "v": [[v[0], v[1]] for v in self.v]
        }


# Shape2 is a union of Circle, Box2, or Polygon2
Shape2 = Union[Circle, Box2, Polygon2]


@dataclass
class Shape2p5:
    """2.5D shape: 2D base shape extruded in Z"""
    base: Shape2 = field(default_factory=lambda: Box2())
    z_min: float = 0.0
    z_max: float = 1.0

    def height(self) -> float:
        return self.z_max - self.z_min

    def to_dict(self):
        result = self.base.to_dict()
        result["z_min"] = self.z_min
        result["z_max"] = self.z_max
        return result


@dataclass
class Pose2:
    """2D pose: position (x, y) and orientation (yaw)"""
    pos: Tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0  # radians

    def to_dict(self):
        return {
            "pos": [self.pos[0], self.pos[1]],
            "yaw": self.yaw
        }
