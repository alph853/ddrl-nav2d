"""
Project manager for saving and loading project state.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.map_editor.documents import (
    MapDocument,
    MapInstanceDoc,
    RobotRouteDoc,
    RobotWaypointDoc,
    ZoneDoc,
    generate_instance_id,
    generate_route_id,
    generate_zone_id,
)
from src.models.geometry import Box2, Circle, Polygon2, Pose2, Shape2p5
from src.models.material import MaterialRef
from src.utils.state_manager import AppState
from src.models.model_profile import ModelPart, ModelProfile


class ProjectManager:
    """Manages project save/load operations"""

    def __init__(self, state_manager: AppState):
        self.state = state_manager

    def load_config_schemas(self, config_root: Path) -> bool:
        """
        Load all config schemas from the config folder
        Currently only loads model profiles from config/model_profile/*.yaml

        Args:
            config_root: Path to config/ folder

        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.config_root_path = str(config_root)

            # Load model profiles
            model_profile_dir = config_root / "model_profile"
            if model_profile_dir.exists():
                self._load_model_profiles(model_profile_dir)
                print(f"Loaded {len(self.state.model_profiles)} model profiles")
                return True
            else:
                print(f"Warning: model_profile directory not found at {model_profile_dir}")
                return False

        except Exception as e:
            print(f"Error loading config schemas: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model_profiles(self, model_profile_dir: Path):
        """Load all model profiles from YAML files"""
        for yaml_file in model_profile_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                # Parse model profile
                profile = self._parse_model_profile(data)
                self.state.add_model_profile(profile.name, profile)

            except Exception as e:
                print(f"Error loading {yaml_file.name}: {e}")
                import traceback
                traceback.print_exc()

    def _parse_model_profile(self, data: Dict[str, Any]) -> ModelProfile:
        """Parse model profile from YAML data"""
        name = data.get("name", "unnamed")
        category = data.get("category", "structure")
        parts = []

        for part_data in data.get("parts", []):
            part = self._parse_model_part(part_data)
            parts.append(part)

        return ModelProfile(name=name, category=category, parts=parts)

    def _parse_model_part(self, data: Dict[str, Any]) -> ModelPart:
        """Parse a single model part from YAML data"""
        name = data.get("name", "part")

        # Parse geometry
        geom_data = data.get("geom", {})
        geom = self._parse_geometry(geom_data)

        # Parse material
        material_data = data.get("material", {})
        material = MaterialRef(
            name=material_data.get("name", "default"),
            rgba=tuple(material_data.get("rgba", [0.5, 0.5, 0.5, 1.0]))
        )

        # Parse relative pose
        pose_data = data.get("rel_pose", {})
        rel_pose = Pose2(
            pos=pose_data.get("pos", [0.0, 0.0]),
            yaw=pose_data.get("yaw", 0.0)
        )

        return ModelPart(name=name, geom=geom, material=material, rel_pose=rel_pose)

    def _parse_geometry(self, data: Dict[str, Any]) -> Shape2p5:
        """Parse geometry from YAML data"""
        geom_type = data.get("type", "box")
        z_min = data.get("z_min", 0.0)
        z_max = data.get("z_max", 1.0)

        if geom_type == "box":
            base = Box2(
                hx=data.get("hx", 1.0),
                hy=data.get("hy", 1.0)
            )
        elif geom_type == "circle":
            base = Circle(
                radius=data.get("radius", 1.0)
            )
        elif geom_type == "polygon":
            base = Polygon2(
                v=data.get("vertices", [[0, 0], [1, 0], [1, 1], [0, 1]])
            )
        else:
            # Default to box
            base = Box2(hx=1.0, hy=1.0)

        return Shape2p5(base=base, z_min=z_min, z_max=z_max)

    # ------------------------------------------------------------------ map io
    def load_map_document(self, file_path: Path) -> MapDocument:
        """
        Load a map YAML file and return a MapDocument.
        """
        with open(file_path, "r", encoding="utf-8") as handle:
            map_data = yaml.safe_load(handle) or {}

        document = MapDocument(
            name=map_data.get("name", "untitled_map"),
            description=map_data.get("description", ""),
        )

        for inst_data in map_data.get("static_instances", []):
            pose_info = inst_data.get("pose_world", {})
            pose = Pose2(
                pos=pose_info.get("pos", [0.0, 0.0]),
                yaw=pose_info.get("yaw", 0.0),
            )
            instance = MapInstanceDoc(
                instance_id=generate_instance_id(),
                name=inst_data.get("name", "instance"),
                model_id=inst_data.get("model_id", ""),
                pose=pose,
            )
            document.instances.append(instance)

        for route_data in map_data.get("robot_routes", []):
            route = RobotRouteDoc(
                route_id=generate_route_id(),
                name=route_data.get("name", "route"),
                start=Pose2(
                    pos=route_data.get("start", {}).get("pos", [0.0, 0.0]),
                    yaw=route_data.get("start", {}).get("yaw", 0.0),
                ),
                waypoints=[
                    RobotWaypointDoc(
                        name=waypoint.get("name", f"wp_{idx + 1}"),
                        pos=(
                            float(waypoint.get("pos", [0.0, 0.0])[0]),
                            float(waypoint.get("pos", [0.0, 0.0])[1]),
                        ),
                        tolerance=float(waypoint.get("tolerance", 1.0)),
                    )
                    for idx, waypoint in enumerate(route_data.get("waypoints", []))
                ],
            )
            document.robot_routes.append(route)

        return document

    def save_map_document(
        self,
        document: MapDocument,
        file_path: Optional[Path] = None,
    ) -> Path:
        """
        Persist a MapDocument to disk. When file_path is None the document is
        stored under config_root/map/<name>.yaml.
        """
        if file_path is None:
            if not self.state.config_root_path:
                raise ValueError("Config root path not configured.")
            map_dir = Path(self.state.config_root_path) / "map"
            map_dir.mkdir(parents=True, exist_ok=True)
            safe_name = document.name or "untitled_map"
            file_path = map_dir / f"{safe_name}.yaml"
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        half_size = self.state.current_map_grid_size / 2.0
        map_payload = {
            "version": "1.0",
            "name": document.name or "untitled_map",
            "description": document.description,
            "static_instances": [instance.to_dict() for instance in document.instances],
            "robot_routes": [route.to_dict() for route in document.robot_routes],
            "bounds": {
                "min": [-half_size, -half_size],
                "max": [half_size, half_size],
            },
        }

        with open(file_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(map_payload, handle, default_flow_style=False, sort_keys=False)

        return file_path
