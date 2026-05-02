"""
Map configuration models matching YAML schema
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .geometry import Pose2


@dataclass
class MapInstance:
    """A placed instance of a model in the world"""
    name: str  # Unique instance name (e.g., "wall_north")
    model_id: str  # Reference to model profile (e.g., "wall")
    pose_world: Pose2 = field(default_factory=Pose2)  # World pose

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "pose_world": self.pose_world.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MapInstance':
        return MapInstance(
            name=data["name"],
            model_id=data["model_id"],
            pose_world=Pose2(
                pos=data["pose_world"]["pos"],
                yaw=data["pose_world"]["yaw"]
            )
        )


@dataclass
class Zone:
    """A named zone with polygon shape and allowed categories"""
    name: str
    vertices: List[List[float]]  # List of [x, y] vertices
    allowed_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": {
                "vertices": self.vertices
            },
            "allowed_categories": self.allowed_categories
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Zone':
        return Zone(
            name=data["name"],
            vertices=data["shape"]["vertices"],
            allowed_categories=data.get("allowed_categories", [])
        )


@dataclass
class RobotWaypoint:
    """A point waypoint in a robot route"""
    name: str
    pos: List[float]
    tolerance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "tolerance": float(self.tolerance),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RobotWaypoint':
        return RobotWaypoint(
            name=data["name"],
            pos=data["pos"],
            tolerance=float(data.get("tolerance", 1.0)),
        )


@dataclass
class RobotRoute:
    """An ordered robot route"""
    name: str
    start: Pose2 = field(default_factory=Pose2)
    waypoints: List[RobotWaypoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start.to_dict(),
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RobotRoute':
        return RobotRoute(
            name=data["name"],
            start=Pose2(
                pos=data.get("start", {}).get("pos", [0.0, 0.0]),
                yaw=data.get("start", {}).get("yaw", 0.0),
            ),
            waypoints=[RobotWaypoint.from_dict(wp) for wp in data.get("waypoints", [])],
        )


@dataclass
class MapConfig:
    """Complete map configuration"""
    version: str = "1.0"
    name: str = "untitled_map"
    description: str = ""

    static_instances: List[MapInstance] = field(default_factory=list)
    robot_routes: List[RobotRoute] = field(default_factory=list)

    bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        "min": [-100.0, -100.0],
        "max": [100.0, 100.0]
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "static_instances": [inst.to_dict() for inst in self.static_instances],
            "robot_routes": [route.to_dict() for route in self.robot_routes],
            "bounds": self.bounds
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MapConfig':
        return MapConfig(
            version=data.get("version", "1.0"),
            name=data.get("name", "untitled_map"),
            description=data.get("description", ""),
            static_instances=[MapInstance.from_dict(inst) for inst in data.get("static_instances", [])],
            robot_routes=[RobotRoute.from_dict(route) for route in data.get("robot_routes", [])],
            bounds=data.get("bounds", {"min": [-100.0, -100.0], "max": [100.0, 100.0]})
        )
