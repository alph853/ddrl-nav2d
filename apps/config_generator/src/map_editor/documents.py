"""
Data structures that describe the logical map document used by the editor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
from uuid import uuid4

from src.models.geometry import Pose2


def generate_instance_id() -> str:
    """Return a short unique identifier for map instances."""
    return f"inst_{uuid4().hex[:8]}"


def generate_zone_id() -> str:
    """Return a short unique identifier for zones."""
    return f"zone_{uuid4().hex[:8]}"


def generate_route_id() -> str:
    """Return a short unique identifier for robot routes."""
    return f"route_{uuid4().hex[:8]}"


@dataclass(slots=True)
class MapInstanceDoc:
    """
    Lightweight immutable description of a placed instance in the map.

    Attributes:
        instance_id: Unique identifier used for referencing the instance.
        name: Human readable label displayed in the UI.
        model_id: Name of the model profile used for rendering.
        pose: Pose of the instance in world coordinates.
    """

    instance_id: str
    name: str
    model_id: str
    pose: Pose2

    def to_dict(self) -> dict:
        """Serialize the instance to the legacy dictionary format."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "pose_world": {
                "pos": list(self.pose.pos),
                "yaw": float(self.pose.yaw),
            },
        }


@dataclass(slots=True)
class ZoneDoc:
    """
    Description of a polygon zone.

    Attributes:
        zone_id: Unique identifier for UI selection.
        name: Display name of the zone.
        zone_type: Semantic type (start, goal, danger, ...).
        vertices: Sequence of vertices in world coordinates.
        allowed_categories: List of permitted model categories.
    """

    zone_id: str
    name: str
    zone_type: str
    vertices: List[Tuple[float, float]]
    allowed_categories: List[str]

    def to_dict(self) -> dict:
        """Serialize to the legacy dictionary format."""
        return {
            "name": self.name,
            "shape": {
                "vertices": [[float(x), float(y)] for x, y in self.vertices],
            },
            "allowed_categories": list(self.allowed_categories),
        }


@dataclass(slots=True)
class RobotWaypointDoc:
    """A point waypoint in an ordered robot route."""

    name: str
    pos: Tuple[float, float]
    tolerance: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "tolerance": float(self.tolerance),
        }


@dataclass(slots=True)
class RobotRouteDoc:
    """An ordered sequence of robot waypoint points."""

    route_id: str
    name: str
    start: Pose2 = field(default_factory=Pose2)
    waypoints: List[RobotWaypointDoc] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": {
                "pos": list(self.start.pos),
                "yaw": float(self.start.yaw),
            },
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
        }


@dataclass(slots=True)
class MapDocument:
    """
    Complete map description used by the controller and persistence layer.
    """

    name: str = "untitled_map"
    description: str = ""
    instances: List[MapInstanceDoc] = field(default_factory=list)
    zones: List[ZoneDoc] = field(default_factory=list)
    robot_routes: List[RobotRouteDoc] = field(default_factory=list)

    def clear(self) -> None:
        """Remove all instances and zones."""
        self.instances.clear()
        self.zones.clear()
        self.robot_routes.clear()

    def add_instance(self, instance: MapInstanceDoc) -> None:
        """Append an instance to the document."""
        self.instances.append(instance)

    def remove_instance(self, instance_id: str) -> MapInstanceDoc | None:
        """Remove an instance by id and return it."""
        for idx, instance in enumerate(self.instances):
            if instance.instance_id == instance_id:
                return self.instances.pop(idx)
        return None

    def add_zone(self, zone: ZoneDoc) -> None:
        """Append a zone to the document."""
        self.zones.append(zone)

    def add_route(self, route: RobotRouteDoc) -> None:
        """Append a robot route to the document."""
        self.robot_routes.append(route)

    def remove_zone(self, zone_id: str) -> ZoneDoc | None:
        """Remove a zone by id and return it."""
        for idx, zone in enumerate(self.zones):
            if zone.zone_id == zone_id:
                return self.zones.pop(idx)
        return None

    def to_instance_state(self) -> List[dict]:
        """Legacy helper used by the global state manager."""
        return [instance.to_dict() for instance in self.instances]

    def to_zone_state(self) -> List[dict]:
        """
        Convert zones to dictionaries carrying `zone_type` for rendering.
        """
        zone_state: List[dict] = []
        for zone in self.zones:
            zone_state.append(
                {
                    "name": zone.name,
                    "vertices": [[float(x), float(y)] for x, y in zone.vertices],
                    "allowed_categories": list(zone.allowed_categories),
                    "zone_type": zone.zone_type,
                }
            )
        return zone_state

    def to_route_state(self) -> List[dict]:
        """Convert robot routes to dictionaries stored in AppState."""
        return [route.to_dict() for route in self.robot_routes]

    def update_from_state(
        self,
        instances_state: Sequence[dict],
        zones_state: Sequence[dict],
        routes_state: Sequence[dict] = (),
    ) -> None:
        """
        Populate the document using dictionaries stored in AppState.
        """
        self.clear()

        for inst in instances_state:
            pose_dict = inst.get("pose_world", {})
            pose = Pose2(
                pos=pose_dict.get("pos", [0.0, 0.0]),
                yaw=pose_dict.get("yaw", 0.0),
            )
            instance = MapInstanceDoc(
                instance_id=generate_instance_id(),
                name=inst.get("name", "unnamed"),
                model_id=inst.get("model_id", ""),
                pose=pose,
            )
            self.instances.append(instance)

        for zone in zones_state:
            vertices = zone.get("vertices", [])
            zone_doc = ZoneDoc(
                zone_id=generate_zone_id(),
                name=zone.get("name", "zone"),
                zone_type=zone.get("zone_type", "robot_start_zones"),
                vertices=[(float(x), float(y)) for x, y in vertices],
                allowed_categories=zone.get("allowed_categories", []),
            )
            self.zones.append(zone_doc)

        for route in routes_state:
            route_doc = RobotRouteDoc(
                route_id=generate_route_id(),
                name=route.get("name", "route"),
                start=Pose2(
                    pos=route.get("start", {}).get("pos", [0.0, 0.0]),
                    yaw=route.get("start", {}).get("yaw", 0.0),
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
                    for idx, waypoint in enumerate(route.get("waypoints", []))
                ],
            )
            self.robot_routes.append(route_doc)
