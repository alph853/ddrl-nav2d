"""
Controller that coordinates UI widgets, the scene store, and persistence.
"""
from __future__ import annotations

from math import atan2, hypot
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

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
from src.map_editor.scene_store import SceneStore
from src.models.geometry import Pose2
from src.ui.widgets.viewport_3d import MapBuildingMode, Viewport3D
from src.utils.state_manager import AppState
from src.io.project_manager import ProjectManager

StatusCallback = Callable[[str], None]
InstancesCallback = Callable[[Sequence[MapInstanceDoc]], None]
ZonesCallback = Callable[[Sequence[ZoneDoc]], None]
RoutesCallback = Callable[[Sequence[RobotRouteDoc]], None]
ZoneNameProvider = Callable[[str, str], Optional[str]]


class MapEditorController:
    """
    High-level coordinator that owns the document, scene store, and mode state.
    """

    def __init__(
        self,
        *,
        state: AppState,
        viewport: Viewport3D,
        project_manager: ProjectManager,
        on_instances_changed: InstancesCallback,
        on_zones_changed: ZonesCallback,
        on_routes_changed: RoutesCallback,
        on_status_changed: StatusCallback,
        request_zone_name: ZoneNameProvider,
    ):
        self._state = state
        self._viewport = viewport
        self._project_manager = project_manager
        self._on_instances_changed = on_instances_changed
        self._on_zones_changed = on_zones_changed
        self._on_routes_changed = on_routes_changed
        self._on_status_changed = on_status_changed
        self._request_zone_name = request_zone_name

        self.document = MapDocument()
        self._scene_store = SceneStore(viewport, state)
        self._current_zone_type = "route"
        self._pending_zone_vertices: List[Tuple[float, float]] = []
        self._active_route: Optional[RobotRouteDoc] = None
        self._active_route_tolerance = 1.5
        self._active_route_start_set = False
        self._instance_counter = 1
        self._highlighted_instance_id: Optional[str] = None

        viewport.point_clicked.connect(self._handle_point_clicked)
        viewport.model_placed.connect(self._handle_model_placed)
        viewport.object_moved.connect(self._handle_object_moved)

    # ------------------------------------------------------------------ public api
    def refresh_from_state(self) -> None:
        """
        Build the document from legacy AppState caches and sync the viewport.
        """
        self.document.update_from_state(
            self._state.current_map_instances,
            [],
            self._state.current_map_routes,
        )
        self._scene_store.sync_document(self.document)
        self._recompute_instance_counter()
        self._emit_instances()
        self._emit_routes()

    def begin_placement(self, model_id: str) -> None:
        """Enter placement mode for the provided model."""
        self._viewport.enter_placement_mode(model_id)
        self._notify_status(
            f"Placing {model_id} | R rotate | Left click place | Right click to cancel"
        )

    def highlight_instance(self, instance_id: str) -> None:
        """Highlight selection in the viewport."""
        self._highlighted_instance_id = instance_id
        self._scene_store.highlight_instance(instance_id)

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an instance and update the scene."""
        removed = self.document.remove_instance(instance_id)
        if removed is None:
            return False

        self._scene_store.remove_instance(instance_id)
        self._sync_state_cache()
        self._emit_instances()
        self._notify_status(f"Deleted {removed.name}")

        if self._highlighted_instance_id == instance_id:
            self._highlighted_instance_id = None

        return True

    def clear_all(self) -> None:
        """Remove every instance and zone."""
        self.document.clear()
        self._scene_store.clear()
        self._sync_state_cache()
        self._emit_instances()
        self._emit_zones()
        self._emit_routes()
        self._notify_status("Cleared map")

    def enter_list_edit_mode(self, instance_id: str) -> bool:
        """Enter edit mode for the provided instance from the list."""
        entry = self._scene_store.get_entry(instance_id)
        if entry is None:
            self._notify_status("Unable to edit the selected instance.")
            return False

        self._viewport.clear_selection_highlight()
        self._viewport.enter_edit_mode(
            instance_id,
            entry.instance.model_id,
            entry.instance.pose,
            entry.actors,
        )
        self._notify_status(
            f"Editing {entry.instance.name} | R rotate | Left click confirm | Right click cancel"
        )
        return True

    def toggle_select_mode(self) -> None:
        """Toggle between select and idle modes for in-viewport editing."""
        if self._viewport.mode == MapBuildingMode.SELECT:
            self._viewport.exit_mode()
            self._notify_status("Selection cancelled")
        else:
            self._viewport.clear_selection_highlight()
            self._viewport.enter_select_mode()
            self._notify_status("Click an object to edit it (Esc to cancel)")

    def start_zone(self, zone_type: str) -> None:
        """Begin drawing a new zone of the given type."""
        self._current_zone_type = zone_type
        self._pending_zone_vertices.clear()
        self._viewport.set_mode(MapBuildingMode.ZONING)
        self._notify_status(
            f"Creating {zone_type} - click vertices, Enter=complete, Esc=cancel"
        )

    def start_route(self, route_name: str, waypoint_tolerance: float = 1.5) -> None:
        """Begin collecting ordered route waypoint points."""
        self._active_route_tolerance = max(0.01, float(waypoint_tolerance))
        self._active_route = RobotRouteDoc(
            route_id=generate_route_id(),
            name=route_name or f"route_{len(self.document.robot_routes) + 1}",
            start=Pose2(),
            waypoints=[],
        )
        self._active_route_start_set = False
        self._viewport.set_mode(MapBuildingMode.ROUTE)
        self._notify_status("Creating route - click start point, then ordered waypoints")

    def finish_route(self) -> bool:
        """Store the active route if it has at least one waypoint."""
        if self._active_route is None:
            self._notify_status("No active route.")
            return False
        if not self._active_route.waypoints:
            self._notify_status("A route requires a start point and at least one waypoint.")
            return False

        self.document.add_route(self._active_route)
        self._active_route = None
        self._active_route_start_set = False
        self._viewport.set_mode(MapBuildingMode.IDLE)
        self._sync_state_cache()
        self._emit_routes()
        self._notify_status("Created route")
        return True

    def cancel_route(self) -> None:
        """Abort active route creation."""
        self._active_route = None
        self._active_route_start_set = False
        if self._viewport.mode == MapBuildingMode.ROUTE:
            self._viewport.set_mode(MapBuildingMode.IDLE)
        self._emit_routes()
        self._notify_status("Route creation cancelled")

    def cancel_zone(self) -> None:
        """Abort the active zone creation."""
        had_vertices = bool(self._pending_zone_vertices)
        self._pending_zone_vertices.clear()
        self._viewport.clear_polygon_visuals()
        if self._viewport.mode == MapBuildingMode.ZONING:
            self._viewport.set_mode(MapBuildingMode.IDLE)
        if had_vertices:
            self._notify_status("Zone creation cancelled")

    def complete_zone(self) -> None:
        """Trigger polygon completion."""
        self._viewport.complete_polygon()

    def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone by id."""
        removed = self.document.remove_zone(zone_id)
        if removed is None:
            return False

        self._scene_store.update_zones(self.document.zones)
        self._sync_state_cache()
        self._emit_zones()
        self._notify_status(f"Deleted zone {removed.name}")
        return True

    def delete_route(self, route_id: str) -> bool:
        for idx, route in enumerate(self.document.robot_routes):
            if route.route_id == route_id:
                removed = self.document.robot_routes.pop(idx)
                self._sync_state_cache()
                self._emit_routes()
                self._notify_status(f"Deleted route {removed.name}")
                return True
        return False

    def load_map(self, file_path: Path) -> None:
        """Load a map document from disk."""
        document = self._project_manager.load_map_document(file_path)
        self.document = document
        self._scene_store.sync_document(self.document)
        self._recompute_instance_counter()
        self._sync_state_cache()
        self._emit_instances()
        self._emit_zones()
        self._emit_routes()
        self._notify_status(f"Loaded map {self.document.name}")

    def save_map(self, name: str) -> Path:
        """Persist the document to disk and return the written path."""
        self.document.name = name or "untitled_map"
        save_path = self._project_manager.save_map_document(self.document)
        self._sync_state_cache()
        self._notify_status(f"Saved map to {save_path}")
        return save_path

    def set_map_description(self, description: str) -> None:
        """Update the document description field."""
        self.document.description = description
        self._sync_state_cache()

    def set_zone_type(self, zone_type: str) -> None:
        """Update the currently selected zone type without starting creation."""
        self._current_zone_type = zone_type

    # ------------------------------------------------------------------ signal handlers
    def _handle_model_placed(self, model_id: str, x: float, y: float, rotation: float) -> None:
        if not model_id:
            self._notify_status("Placement cancelled - invalid model.")
            return

        pose = Pose2(pos=(x, y), yaw=rotation)
        instance_name = self._generate_instance_name(model_id)
        instance = MapInstanceDoc(
            instance_id=generate_instance_id(),
            name=instance_name,
            model_id=model_id,
            pose=pose,
        )

        self.document.add_instance(instance)
        self._scene_store.add_instance(instance)
        self._sync_state_cache()
        self._emit_instances()
        self._scene_store.highlight_instance(instance.instance_id)
        self._highlighted_instance_id = instance.instance_id
        self._notify_status(f"Placed {instance_name} at ({x:.1f}, {y:.1f})")

    def _handle_point_clicked(self, x: float, y: float) -> None:
        mode = self._viewport.mode
        if mode == MapBuildingMode.SELECT:
            self._select_instance_near(x, y)
        elif mode == MapBuildingMode.ZONING:
            self._pending_zone_vertices.append((x, y))
            self._viewport.add_polygon_vertex(x, y)
            vertex_count = len(self._pending_zone_vertices)
            self._notify_status(
                f"Creating {self._current_zone_type} - {vertex_count} vertices"
            )
        elif mode == MapBuildingMode.ROUTE and self._active_route is not None:
            if not self._active_route_start_set:
                self._active_route.start = Pose2(pos=(float(x), float(y)), yaw=0.0)
                self._active_route_start_set = True
                self._emit_routes()
                self._notify_status(f"Creating {self._active_route.name} - start point set")
                return

            waypoint = RobotWaypointDoc(
                name=f"wp_{len(self._active_route.waypoints) + 1}",
                pos=(float(x), float(y)),
                tolerance=self._active_route_tolerance,
            )
            if not self._active_route.waypoints:
                sx, sy = self._active_route.start.pos
                self._active_route.start.yaw = atan2(float(y) - sy, float(x) - sx)
            self._active_route.waypoints.append(waypoint)
            self._emit_routes()
            self._notify_status(
                f"Creating {self._active_route.name} - {len(self._active_route.waypoints)} waypoints"
            )

    def _handle_polygon_completed(self, vertices: List[List[float]]) -> None:
        if len(vertices) < 3:
            self._notify_status("A zone requires at least 3 vertices.")
            self.cancel_zone()
            return

        default_name = f"{self._current_zone_type}_{len(self.document.zones)+1}"
        zone_name = self._request_zone_name(self._current_zone_type, default_name)

        if not zone_name:
            self.cancel_zone()
            return

        zone = ZoneDoc(
            zone_id=generate_zone_id(),
            name=zone_name,
            zone_type=self._current_zone_type,
            vertices=[(float(v[0]), float(v[1])) for v in vertices],
            allowed_categories=self._allowed_categories(self._current_zone_type),
        )

        self.document.add_zone(zone)
        self._scene_store.update_zones(self.document.zones)
        self._sync_state_cache()
        self._emit_zones()
        self._notify_status(f"Created zone {zone_name}")
        self._pending_zone_vertices.clear()

    def _handle_object_moved(
        self,
        instance_id: str,
        x: float,
        y: float,
        rotation: float,
    ) -> None:
        updated_pose = Pose2(pos=(x, y), yaw=rotation)
        updated = self._scene_store.update_instance_pose(instance_id, updated_pose)
        if updated is None:
            return

        self._replace_instance(updated)
        self._sync_state_cache()
        self._emit_instances()
        self._scene_store.highlight_instance(instance_id)
        self._highlighted_instance_id = instance_id
        self._notify_status(f"Moved {updated.name} to ({x:.1f}, {y:.1f})")

    # ------------------------------------------------------------------ helpers
    def _select_instance_near(self, x: float, y: float) -> None:
        threshold = self._viewport.grid_spacing * 2.0
        closest: Optional[MapInstanceDoc] = None
        closest_dist = threshold

        for instance in self.document.instances:
            inst_x, inst_y = instance.pose.pos
            dist = hypot(inst_x - x, inst_y - y)
            if dist < closest_dist:
                closest = instance
                closest_dist = dist

        if closest is None:
            self._notify_status("No object found at that location.")
            return

        self.enter_list_edit_mode(closest.instance_id)

    def _generate_instance_name(self, model_id: str) -> str:
        name = f"{model_id}_{self._instance_counter}"
        self._instance_counter += 1
        return name

    def _recompute_instance_counter(self) -> None:
        highest = 0
        for instance in self.document.instances:
            parts = instance.name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                highest = max(highest, int(parts[1]))
        self._instance_counter = highest + 1 if highest else 1

    def _replace_instance(self, updated: MapInstanceDoc) -> None:
        for idx, instance in enumerate(self.document.instances):
            if instance.instance_id == updated.instance_id:
                self.document.instances[idx] = updated
                return

    def _sync_state_cache(self) -> None:
        self._state.current_map_instances = self.document.to_instance_state()
        self._state.current_map_zones = []
        self._state.current_map_routes = self.document.to_route_state()
        self._state.current_map_name = self.document.name
        self._state.current_map_description = self.document.description

    def _emit_instances(self) -> None:
        self._on_instances_changed(self.document.instances)

    def _emit_zones(self) -> None:
        self._on_zones_changed(self.document.zones)

    def _emit_routes(self) -> None:
        routes = list(self.document.robot_routes)
        if self._active_route is not None:
            routes.append(self._active_route)
        self._on_routes_changed(routes)

    def _notify_status(self, message: str) -> None:
        self._on_status_changed(message)

    @staticmethod
    def _allowed_categories(zone_type: str) -> List[str]:
        return []
