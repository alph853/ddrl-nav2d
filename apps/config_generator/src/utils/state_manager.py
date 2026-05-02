"""
Global state manager for all libraries and data
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.models.model_profile import ModelProfile


@dataclass
class AppState:
    """Global application state"""

    # Model profiles library (loaded from config/model_profile/*.yaml)
    model_profiles: Dict[str, ModelProfile] = field(default_factory=dict)

    # Current map being built (Tab 5)
    current_map_instances: List[Dict] = field(default_factory=list)  # List of placed instances
    current_map_zones: List[Dict] = field(default_factory=list)  # List of zones
    current_map_routes: List[Dict] = field(default_factory=list)  # List of robot routes
    current_map_name: str = "untitled_map"
    current_map_description: str = ""
    current_map_grid_size: float = 500.0

    # Config root path (where model_profile/ folder is located)
    config_root_path: Optional[str] = None

    def get_model_profile(self, name: str) -> Optional[ModelProfile]:
        """Get model profile by name"""
        return self.model_profiles.get(name)

    def get_model_profile_names(self) -> List[str]:
        """Get list of all model profile names"""
        return sorted(list(self.model_profiles.keys()))

    def add_model_profile(self, name: str, profile: ModelProfile):
        """Add or update a model profile"""
        self.model_profiles[name] = profile

    def clear_map(self):
        """Clear current map data"""
        self.current_map_instances.clear()
        self.current_map_zones.clear()
        self.current_map_routes.clear()
        self.current_map_name = "untitled_map"
        self.current_map_description = ""


# Global state instance
_state = AppState()


def get_state() -> AppState:
    return _state
