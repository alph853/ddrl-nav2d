"""
Runtime scene store that acts as the single source of truth for rendered actors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pyvista as pv

from src.map_editor.documents import MapDocument, MapInstanceDoc, ZoneDoc
from src.models.geometry import Pose2
from src.ui.widgets.viewport_3d import Viewport3D
from src.utils.state_manager import AppState


@dataclass(slots=True)
class InstanceSceneEntry:
    """Bundle that keeps the logical instance and the rendered actors."""

    instance: MapInstanceDoc
    actors: List[pv.Actor]


class SceneStore:
    """
    Stores rendered actors and exposes high level helpers for the controller.
    """

    def __init__(self, viewport: Viewport3D, state: AppState):
        self._viewport = viewport
        self._state = state
        self._instances: Dict[str, InstanceSceneEntry] = {}
        self._zones: Dict[str, ZoneDoc] = {}
        self._highlighted_instance: Optional[str] = None

    def clear(self) -> None:
        """Remove every rendered object from the viewport."""
        for entry in self._instances.values():
            for actor in entry.actors:
                self._viewport.plotter.remove_actor(actor)
        self._instances.clear()
        self._zones.clear()
        self._viewport.clear_selection_highlight()
        self._viewport.clear_zones()

    def sync_document(self, document: MapDocument) -> None:
        """Rebuild the scene from the provided document."""
        self.clear()

        for instance in document.instances:
            self.add_instance(instance)

        self.update_zones(document.zones)

    def add_instance(self, instance: MapInstanceDoc) -> None:
        """Render and track a new instance."""
        actors = self._render_instance(instance)
        self._instances[instance.instance_id] = InstanceSceneEntry(
            instance=instance,
            actors=actors,
        )

    def remove_instance(self, instance_id: str) -> None:
        """Remove the instance with the given id."""
        entry = self._instances.pop(instance_id, None)
        if not entry:
            return

        for actor in entry.actors:
            self._viewport.plotter.remove_actor(actor)

        if self._highlighted_instance == instance_id:
            self._viewport.clear_selection_highlight()
            self._highlighted_instance = None

    def update_instance_pose(self, instance_id: str, pose: Pose2) -> Optional[MapInstanceDoc]:
        """
        Update the pose by removing old actors and re-rendering.
        """
        entry = self._instances.get(instance_id)
        if not entry:
            return None

        for actor in entry.actors:
            self._viewport.plotter.remove_actor(actor)

        updated_instance = MapInstanceDoc(
            instance_id=entry.instance.instance_id,
            name=entry.instance.name,
            model_id=entry.instance.model_id,
            pose=pose,
        )

        new_actors = self._render_instance(updated_instance)
        entry.instance = updated_instance
        entry.actors = new_actors

        if self._highlighted_instance == instance_id:
            self.highlight_instance(instance_id)

        return updated_instance

    def highlight_instance(self, instance_id: str) -> None:
        """Highlight the selected instance."""
        entry = self._instances.get(instance_id)
        if not entry:
            self._viewport.clear_selection_highlight()
            self._highlighted_instance = None
            return

        self._viewport.highlight_instance(entry.instance.pose, entry.instance.model_id)
        self._highlighted_instance = instance_id

    def get_entry(self, instance_id: str) -> Optional[InstanceSceneEntry]:
        """Return the stored scene entry for an instance."""
        return self._instances.get(instance_id)

    def update_zones(self, zones: Iterable[ZoneDoc]) -> None:
        """Render the provided zones."""
        self._zones = {zone.zone_id: zone for zone in zones}
        zone_state = []
        for zone in zones:
            zone_state.append(
                {
                    "name": zone.name,
                    "vertices": [[float(x), float(y)] for x, y in zone.vertices],
                    "allowed_categories": list(zone.allowed_categories),
                    "zone_type": zone.zone_type,
                }
            )

        self._viewport.render_zones(zone_state)

    def _render_instance(self, instance: MapInstanceDoc) -> List[pv.Actor]:
        """
        Render a model profile instance at the provided pose.
        """
        profile = self._state.get_model_profile(instance.model_id)
        if profile is None:
            raise ValueError(f"Model profile '{instance.model_id}' is not loaded.")

        multiblock, materials = self._viewport.get_cached_model_mesh(profile.name)
        if multiblock is None or materials is None:
            raise RuntimeError(f"Model '{profile.name}' is missing from the cache.")

        actors: List[object] = []

        for mesh, (color, opacity) in zip(multiblock, materials):
            mesh.points = mesh._original_points.copy()

            yaw = float(instance.pose.yaw)
            if not np.isclose(yaw, 0.0):
                mesh.rotate_z(np.degrees(yaw), inplace=True)

            pos_x, pos_y = instance.pose.pos
            mesh.translate([pos_x, pos_y, 0.0], inplace=True)

            actor = self._viewport.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=True,
                edge_color="black",
            )
            actors.append(actor)

        return actors
