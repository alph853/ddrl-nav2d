"""
Material models matching C++ MaterialRef schema
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MaterialRef:
    """Material with name and RGBA color"""
    name: str = "default_material"
    rgba: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)

    def to_dict(self):
        return {
            "name": self.name,
            "rgba": list(self.rgba)
        }
