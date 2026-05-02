"""
3D Viewport widget using PyVista.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from src.models.geometry import Pose2


class MapBuildingMode(Enum):
    """Enum for different map building modes"""
    IDLE = auto()           # Normal viewport state (no active operation)
    PLACEMENT = auto()      # Placing new models
    ZONING = auto()         # Creating polygon zones
    ROUTE = auto()          # Creating ordered robot route waypoints
    SELECT = auto()         # Selecting object to edit (after E key)
    EDIT = auto()           # Editing existing objects (moving/rotating)


@dataclass(slots=True)
class ModelCacheEntry:
    """Cached data per model for rendering/preview/highlight."""

    multiblock: pv.MultiBlock
    materials: List[Tuple[object, float]]
    preview_mesh: pv.PolyData
    preview_actor: Optional[pv.Actor] = None
    bounds_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_lengths: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class Viewport3D(QWidget):
    """3D viewport widget with PyVista rendering"""

    point_clicked = pyqtSignal(float, float)  # x, y (for polygon placement)
    model_placed = pyqtSignal(str, float, float, float)  # model_name, x, y, rotation (for click placement)
    polygon_completed = pyqtSignal(list)  # vertices: List of [x, y] (convex hull computed)
    zone_cancelled = pyqtSignal()  # Emitted when zoning is cancelled via ESC
    object_selected = pyqtSignal(int)  # index in placed_instances (for edit mode)
    object_moved = pyqtSignal(str, float, float, float)  # instance id, x, y, rotation (for edit mode)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.plotter: Optional[QtInteractor] = None
        self.grid_actor: Optional[pv.Actor] = None
        self.axes_actors: List[pv.Actor] = []
        self.name_label_actor: Optional[pv.Actor] = None
        self.grid_visible: bool = True
        self.grid_spacing: float = 1.0
        self.current_z_offset: float = 0.0  # For Ctrl+drag Z movement

        # Mouse tracking for Ctrl+drag
        self.ctrl_drag_active: bool = False
        self.last_mouse_y: int = 0

        # Unified mode system
        self.mode: MapBuildingMode = MapBuildingMode.IDLE

        # Polygon mode state
        self.vertex_actors: List[pv.Actor] = []
        self.edge_actors: List[pv.Actor] = []
        self.polygon_vertices: List[Tuple[float, float]] = []

        # Placement/Edit mode state (unified)
        self.current_model_name: Optional[str] = None
        self.current_pose: Pose2 = Pose2(pos=(0.0, 0.0), yaw=0.0)
        self.preview_actor: Optional[pv.Actor] = None

        # Edit mode specific state
        self.selected_instance_id: Optional[str] = None  # Instance id being edited
        self.selected_instance_actors: Optional[List[pv.Actor]] = None  # Original actors

        # Model mesh cache
        self.model_cache: Dict[str, ModelCacheEntry] = {}

        # Zone visualization
        self.zone_actors: List[pv.Actor] = []
        self.route_actors: List[pv.Actor] = []

        # Selection highlight
        self.selection_highlight_actor: Optional[pv.Actor] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup PyVista plotter widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create PyVista plotter with Qt integration
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Enable focus for keyboard events
        self.plotter.interactor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Install event filter for mouse events and keyboard shortcuts
        self.plotter.interactor.installEventFilter(self)

        # Setup initial scene
        self._setup_scene()

    def _setup_scene(self) -> None:
        """Setup initial 3D scene with grid and axes"""
        self.plotter.set_background("#2a2a2a")

        self.show_grid(True)
        self.show_axes(True)

        # Set initial camera position (looking down at 45 degrees)
        self.plotter.camera_position = [
            (15, 15, 15),  # camera position
            (0, 0, 0),     # focal point
            (0, 0, 1)      # view up
        ]
        self.plotter.enable_anti_aliasing()

    def show_grid(
        self,
        visible: bool = True,
        size: float = 500.0,
        spacing: Optional[float] = None,
    ) -> None:
        """
        Show/hide grid floor

        Args:
            visible: Whether grid should be visible
            size: Size of grid
            spacing: Grid line spacing (uses self.grid_spacing if None)
        """
        if spacing is not None:
            self.grid_spacing = spacing

        # Remove existing grid
        if self.grid_actor is not None:
            self.plotter.remove_actor(self.grid_actor)
            self.grid_actor = None

        if visible:
            # Create grid mesh
            from src.rendering.grid import create_grid
            grid_mesh = create_grid(size=size, spacing=self.grid_spacing)

            # Add to scene with better visibility
            self.grid_actor = self.plotter.add_mesh(
                grid_mesh,
                color="gray",
                line_width=2,  # Thicker lines
                opacity=0.8    # More opaque
            )

        self.grid_visible = visible

    def show_axes(self, visible: bool = True, length: float = 5.0) -> None:
        """
        Show/hide coordinate axes

        Args:
            visible: Whether axes should be visible
            length: Length of axes
        """
        # Remove existing axes
        for actor in self.axes_actors:
            self.plotter.remove_actor(actor)
        self.axes_actors.clear()

        if visible:
            from src.rendering.grid import create_axes
            x_axis, y_axis, z_axis = create_axes(length=length)

            # Add axes with colors (thicker lines)
            x_actor = self.plotter.add_mesh(x_axis, color="red", line_width=5)
            y_actor = self.plotter.add_mesh(y_axis, color="green", line_width=5)
            z_actor = self.plotter.add_mesh(z_axis, color="blue", line_width=5)

            self.axes_actors = [x_actor, y_actor, z_actor]

            # Add axis labels at 3D positions
            # Create point at each axis endpoint for label
            x_label_point = pv.PolyData([length * 1.1, 0, 0])
            y_label_point = pv.PolyData([0, length * 1.1, 0])
            z_label_point = pv.PolyData([0, 0, length * 1.1])

            self.plotter.add_point_labels(x_label_point, ["X"], font_size=20, text_color="red",
                                         point_size=0, render_points_as_spheres=False)
            self.plotter.add_point_labels(y_label_point, ["Y"], font_size=20, text_color="green",
                                         point_size=0, render_points_as_spheres=False)
            self.plotter.add_point_labels(z_label_point, ["Z"], font_size=20, text_color="blue",
                                         point_size=0, render_points_as_spheres=False)

    def clear_scene(self) -> None:
        """Clear all objects from scene (except grid and axes)"""
        self.plotter.clear()
        self.current_mesh = None
        self.current_mesh_actor = None
        self._setup_scene()

    def reset_camera(self) -> None:
        """Reset camera to default position"""
        self.plotter.reset_camera()

    def set_camera_top_view(self) -> None:
        """Set camera to top-down view"""
        self.plotter.view_xy()

    def set_camera_side_view(self) -> None:
        """Set camera to side view"""
        self.plotter.view_xz()

    def set_camera_front_view(self) -> None:
        """Set camera to front view"""
        self.plotter.view_yz()

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        """
        Event filter to handle placement mode, Ctrl+Left drag for Z-axis movement, and click events.
        """

        # Handle placement/edit mode events first
        if self.mode == MapBuildingMode.PLACEMENT or self.mode == MapBuildingMode.EDIT:
            # Set focus when mouse enters viewport during placement
            if event.type() == QEvent.Type.Enter:
                self.plotter.interactor.setFocus()

            if event.type() == QEvent.Type.MouseMove:
                # Update preview position as mouse moves
                mouse_pos = event.pos()
                world_point = self._screen_to_world_z0(mouse_pos.x(), mouse_pos.y())
                if world_point is not None:
                    # Snap to grid
                    snapped_x = round(world_point[0] / self.grid_spacing) * self.grid_spacing
                    snapped_y = round(world_point[1] / self.grid_spacing) * self.grid_spacing
                    self._update_preview(snapped_x, snapped_y)
                return False

            elif event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Place or move the model
                    mouse_pos = event.pos()
                    world_point = self._screen_to_world_z0(mouse_pos.x(), mouse_pos.y())
                    if world_point is not None:
                        # Snap to grid
                        snapped_x = round(world_point[0] / self.grid_spacing) * self.grid_spacing
                        snapped_y = round(world_point[1] / self.grid_spacing) * self.grid_spacing

                        if self.mode == MapBuildingMode.PLACEMENT:
                            print(f"[Viewport3D] Placing model '{self.current_model_name}' at ({snapped_x:.2f}, {snapped_y:.2f}) rotation={np.degrees(self.current_pose.yaw):.1f}°")
                            self.model_placed.emit(
                                self.current_model_name,
                                snapped_x,
                                snapped_y,
                                self.current_pose.yaw
                            )
                            self._update_preview(snapped_x, snapped_y)

                        elif self.mode == MapBuildingMode.EDIT:
                            if self.selected_instance_id is not None:
                                print(f"[Viewport3D] Moving object {self.selected_instance_id} to ({snapped_x:.2f}, {snapped_y:.2f}) rotation={np.degrees(self.current_pose.yaw):.1f}°")
                                self.object_moved.emit(
                                    self.selected_instance_id,
                                    snapped_x,
                                    snapped_y,
                                    self.current_pose.yaw
                                )
                            # Exit edit mode after confirming move
                            self.exit_mode()
                    return True

                elif event.button() == Qt.MouseButton.RightButton:
                    # Cancel placement/edit
                    print(f"[Viewport3D] {'Placement' if self.mode == MapBuildingMode.PLACEMENT else 'Edit'} cancelled by right click")
                    self.exit_mode()
                    return True

            elif event.type() == QEvent.Type.KeyPress:
                from PyQt6.QtCore import Qt as QtKey

                if event.key() == QtKey.Key.Key_R and (event.modifiers() & QtKey.KeyboardModifier.ShiftModifier):
                    self.current_pose.yaw -= np.radians(15)
                    if self.current_pose.yaw < 0:
                        self.current_pose.yaw += 2 * np.pi
                    print(f"[Viewport3D] Rotated preview to {np.degrees(self.current_pose.yaw):.1f}°")
                    self._update_preview(self.current_pose.pos[0], self.current_pose.pos[1])
                    return True

                if event.key() == QtKey.Key.Key_R:
                    self.current_pose.yaw += np.radians(15)
                    if self.current_pose.yaw >= 2 * np.pi:
                        self.current_pose.yaw -= 2 * np.pi
                    print(f"[Viewport3D] Rotated preview to {np.degrees(self.current_pose.yaw):.1f}°")
                    self._update_preview(self.current_pose.pos[0], self.current_pose.pos[1])
                    return True

                # ESC to cancel placement/edit
                elif event.key() == QtKey.Key.Key_Escape:
                    print(f"[Viewport3D] {'Placement' if self.mode == MapBuildingMode.PLACEMENT else 'Edit'} cancelled by Escape")
                    self.exit_mode()
                    return True

        # Handle polygon placement mode key events
        if self.mode == MapBuildingMode.ZONING:
            if event.type() == QEvent.Type.KeyPress:
                from PyQt6.QtCore import Qt as QtKey

                # Ctrl+N to start a new zone (check this first before other keys)
                if event.key() == QtKey.Key.Key_N and (event.modifiers() & QtKey.KeyboardModifier.ControlModifier):
                    print("[Viewport3D] Starting new zone (Ctrl+N pressed)")
                    # Complete current polygon first if there are vertices
                    if len(self.polygon_vertices) >= 3:
                        self.complete_polygon()
                    else:
                        # Just clear current vertices and start fresh
                        self.clear_polygon_visuals()
                        self.polygon_vertices.clear()
                    return True

                elif event.key() in (QtKey.Key.Key_Return, QtKey.Key.Key_Enter):
                    print("[Viewport3D] Completing polygon (Enter pressed)")
                    self.complete_polygon()
                    return True

                # ESC to cancel polygon
                elif event.key() == QtKey.Key.Key_Escape:
                    print("[Viewport3D] Zone placement cancelled (Escape pressed)")
                    self.set_mode(MapBuildingMode.IDLE)
                    self.zone_cancelled.emit()
                    return True

            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Get mouse position and compute ray-plane intersection with Z=0
                    mouse_pos = event.pos()
                    x_screen = mouse_pos.x()
                    y_screen = mouse_pos.y()

                    # Pick on Z=0 plane using PyVista's coordinate conversion
                    # Get the render window interactor
                    iren = self.plotter.iren
                    if iren:
                        world_point = self._screen_to_world_z0(x_screen, y_screen)
                        if world_point is not None:
                            self.point_clicked.emit(world_point[0], world_point[1])
                    return True

        # Handle route waypoint click mode.
        if self.mode == MapBuildingMode.ROUTE:
            if event.type() == QEvent.Type.KeyPress:
                from PyQt6.QtCore import Qt as QtKey
                if event.key() == QtKey.Key.Key_Escape:
                    print("[Viewport3D] Route placement cancelled (Escape pressed)")
                    self.set_mode(MapBuildingMode.IDLE)
                    return True

            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    mouse_pos = event.pos()
                    world_point = self._screen_to_world_z0(mouse_pos.x(), mouse_pos.y())
                    if world_point is not None:
                        snapped_x = round(world_point[0] / self.grid_spacing) * self.grid_spacing
                        snapped_y = round(world_point[1] / self.grid_spacing) * self.grid_spacing
                        self.point_clicked.emit(snapped_x, snapped_y)
                    return True

        # Handle SELECT mode events (waiting for object selection)
        if self.mode == MapBuildingMode.SELECT:
            if event.type() == QEvent.Type.KeyPress:
                from PyQt6.QtCore import Qt as QtKey

                # ESC or E to cancel selection
                if event.key() in (QtKey.Key.Key_Escape, QtKey.Key.Key_E):
                    print("[Viewport3D] Select mode cancelled")
                    self.exit_mode()
                    return True

        # Handle clicks when in SELECT mode (waiting for object selection)
        if self.mode == MapBuildingMode.SELECT:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Get mouse position and compute ray-plane intersection with Z=0
                    mouse_pos = event.pos()
                    x_screen = mouse_pos.x()
                    y_screen = mouse_pos.y()

                    # Pick on Z=0 plane using PyVista's coordinate conversion
                    iren = self.plotter.iren
                    if iren:
                        world_point = self._screen_to_world_z0(x_screen, y_screen)
                        if world_point is not None:
                            self.point_clicked.emit(world_point[0], world_point[1])
                    return True

        return super().eventFilter(obj, event)

    def _screen_to_world_z0(self, x_screen: int, y_screen: int) -> Optional[np.ndarray]:
        """
        Convert screen coordinates to world coordinates on Z=0 plane

        Args:
            x_screen: Screen X coordinate (Qt widget coordinates)
            y_screen: Screen Y coordinate (Qt widget coordinates)

        Returns:
            (x, y, z) world coordinates on Z=0 plane, or None if no intersection
        """
        renderer = self.plotter.renderer

        # Get the interactor to access display coordinates
        render_window = self.plotter.render_window

        # VTK uses bottom-left origin, Qt uses top-left
        # Need to flip Y coordinate
        window_height = render_window.GetSize()[1]
        vtk_y = window_height - y_screen

        # Alternative: Use world point picker with ray-plane intersection
        # Get world point at near and far clipping planes
        renderer.SetDisplayPoint(x_screen, vtk_y, 0)  # Near plane (z=0 in display)
        renderer.DisplayToWorld()
        world_near = np.array(renderer.GetWorldPoint()[:3])

        renderer.SetDisplayPoint(x_screen, vtk_y, 1)  # Far plane (z=1 in display)
        renderer.DisplayToWorld()
        world_far = np.array(renderer.GetWorldPoint()[:3])

        # Ray from near to far
        ray_origin = world_near
        ray_dir = world_far - world_near
        ray_dir_norm = np.linalg.norm(ray_dir)

        if ray_dir_norm < 1e-10:
            return None

        ray_dir = ray_dir / ray_dir_norm

        # Intersect with Z=0 plane
        # Plane: Z = 0
        # Ray: P = ray_origin + t * ray_dir
        # Solve: ray_origin[2] + t * ray_dir[2] = 0

        if abs(ray_dir[2]) < 1e-6:
            # Ray parallel to plane
            return None

        t = -ray_origin[2] / ray_dir[2]

        if t < 0:
            # Intersection behind camera
            return None

        # Compute intersection point
        intersection = ray_origin + t * ray_dir

        return intersection

    def set_mode(self, mode: MapBuildingMode) -> None:
        """
        Set the current map building mode

        Args:
            mode: The mode to set
        """
        # Exit current mode first
        if self.mode != MapBuildingMode.IDLE:
            self.exit_mode()

        self.mode = mode

        if mode == MapBuildingMode.ZONING:
            # Enable mouse tracking for zoning mode
            self.plotter.interactor.setFocus()
        elif mode == MapBuildingMode.IDLE:
            # Clear visual feedback
            self.clear_polygon_visuals()
            self.polygon_vertices.clear()

    def add_polygon_vertex(self, x: float, y: float) -> None:
        """
        Add a vertex to the current polygon

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.polygon_vertices.append((x, y))
        self.add_polygon_vertex_visual(x, y)
        self.update_polygon_edge_visuals(self.polygon_vertices)

    def complete_polygon(self) -> None:
        """
        Complete the current polygon and compute convex hull
        Emits polygon_completed signal with convex hull vertices
        """
        if len(self.polygon_vertices) < 3:
            print(f"[Viewport3D] Need at least 3 vertices to complete polygon, got {len(self.polygon_vertices)}")
            return

        # Compute convex hull
        hull_vertices = self._compute_convex_hull(self.polygon_vertices)

        print(f"[Viewport3D] Polygon completed with {len(self.polygon_vertices)} vertices, convex hull has {len(hull_vertices)} vertices")

        # Emit signal with convex hull
        self.polygon_completed.emit([[v[0], v[1]] for v in hull_vertices])

        # Clear current polygon
        self.clear_polygon_visuals()
        self.polygon_vertices.clear()

    def _compute_convex_hull(
        self,
        vertices: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Compute convex hull of given vertices using Graham scan

        Args:
            vertices: List of (x, y) tuples

        Returns:
            List of (x, y) tuples representing convex hull in CCW order
        """
        from scipy.spatial import ConvexHull

        if len(vertices) < 3:
            return vertices

        # Convert to numpy array
        points = np.array(vertices)

        try:
            # Compute convex hull
            hull = ConvexHull(points)

            # Get vertices in CCW order
            hull_vertices = [tuple(points[idx]) for idx in hull.vertices]

            return hull_vertices
        except Exception as e:
            print(f"[Viewport3D] Error computing convex hull: {e}")
            # Return original vertices if convex hull fails
            return vertices

    def clear_polygon_visuals(self) -> None:
        """Clear vertex and edge visual feedback"""
        for actor in self.vertex_actors:
            self.plotter.remove_actor(actor)
        for actor in self.edge_actors:
            self.plotter.remove_actor(actor)
        self.vertex_actors.clear()
        self.edge_actors.clear()

    def add_polygon_vertex_visual(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Add visual feedback for a polygon vertex

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate (default 0.0 for grid plane)
        """
        # Create sphere at vertex position
        sphere = pv.Sphere(radius=0.3, center=(x, y, z))
        actor = self.plotter.add_mesh(
            sphere,
            color="yellow",
            opacity=0.9,
            show_edges=False
        )
        self.vertex_actors.append(actor)

    def update_polygon_edge_visuals(self, vertices: List[Tuple[float, float]]) -> None:
        """
        Update edge lines connecting polygon vertices

        Args:
            vertices: List of (x, y) vertex tuples
        """
        # Clear existing edges
        for actor in self.edge_actors:
            self.plotter.remove_actor(actor)
        self.edge_actors.clear()

        if len(vertices) < 2:
            return

        # Create lines between consecutive vertices
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices) if len(vertices) > 2 else i + 1
            if next_i >= len(vertices):
                break

            # Create line
            points = np.array([
                [vertices[i][0], vertices[i][1], 0.0],
                [vertices[next_i][0], vertices[next_i][1], 0.0]
            ])
            line = pv.Line(points[0], points[1])
            actor = self.plotter.add_mesh(
                line,
                color="yellow",
                opacity=0.9,
                line_width=3
            )
            self.edge_actors.append(actor)

    def enter_placement_mode(self, model_name: str) -> None:
        """
        Enter model placement mode with preview
        """
        print(f"[Viewport3D] Entering placement mode for model: {model_name}")
        self.mode = MapBuildingMode.PLACEMENT
        self.current_model_name = model_name
        self.current_pose = Pose2(pos=(0.0, 0.0), yaw=0.0)

        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocus()

        self._activate_preview(model_name)

    def enter_edit_mode(
        self,
        instance_id: str,
        model_name: str,
        current_pose: Pose2,
        instance_actors: List[pv.Actor],
    ) -> None:
        """
        Enter edit mode for an existing instance

        Args:
            instance_id: Identifier of instance being edited
            model_name: Model name of the instance
            current_pose: Current pose of the instance
            instance_actors: List of actors for the instance
        """
        print(f"[Viewport3D] Entering edit mode for instance {instance_id}: {model_name}")
        self.mode = MapBuildingMode.EDIT
        self.selected_instance_id = instance_id
        self.current_model_name = model_name
        self.current_pose = Pose2(pos=current_pose.pos, yaw=current_pose.yaw)
        self.selected_instance_actors = instance_actors

        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocus()

        # Hide the original instance actors
        for actor in instance_actors:
            actor.SetVisibility(False)

        # Get or create cached model mesh
        self._activate_preview(model_name)
        self._update_preview(current_pose.pos[0], current_pose.pos[1])

    def enter_select_mode(self) -> None:
        """Enter select mode - waiting for user to click an object to edit"""
        print("[Viewport3D] Entering select mode")
        self.mode = MapBuildingMode.SELECT
        self.plotter.interactor.setFocus()

    def exit_mode(self) -> None:
        """Exit current mode and clean up"""
        print(f"[Viewport3D] Exiting {self.mode.name} mode")

        if self.mode == MapBuildingMode.EDIT:
            # Restore original instance visibility
            if self.selected_instance_actors:
                for actor in self.selected_instance_actors:
                    actor.SetVisibility(True)
            self.selected_instance_id = None
            self.selected_instance_actors = None

        # Disable mouse tracking
        self.plotter.interactor.setMouseTracking(False)

        # Clear preview
        self._clear_preview()

        # Reset state
        self.mode = MapBuildingMode.IDLE
        self.current_model_name = None
        self.current_pose = Pose2(pos=(0.0, 0.0), yaw=0.0)

    def _clear_preview(self) -> None:
        """Remove preview actor from scene"""
        if self.preview_actor is not None:
            self.preview_actor.SetVisibility(False)
        self.preview_actor = None

    def _ensure_model_cached(self, model_name: str) -> None:
        """
        Ensure model is in cache, build and store if not present

        Args:
            model_name: Name of the model profile
        """
        if model_name in self.model_cache:
            return  # Already cached

        from src.utils.state_manager import _state
        from src.rendering.geometry_builder import create_combined_model_mesh

        model_profile = _state.get_model_profile(model_name)
        if model_profile is None:
            print(f"[Viewport3D] Warning: Model profile '{model_name}' not found")
            return

        print(f"[Viewport3D] Building and caching model: {model_name}")

        # Create combined MultiBlock mesh with materials
        combined_multiblock, materials = create_combined_model_mesh(model_profile)

        # Store original points for each block in the MultiBlock
        for mesh in combined_multiblock:
            mesh._original_points = mesh.points.copy()

        preview_mesh = self._build_preview_mesh(combined_multiblock)
        bounds_center, bounds_lengths = self._compute_bounds_metadata(combined_multiblock)

        entry = ModelCacheEntry(
            multiblock=combined_multiblock,
            materials=materials,
            preview_mesh=preview_mesh,
            bounds_center=bounds_center,
            bounds_lengths=bounds_lengths,
        )
        self.model_cache[model_name] = entry

        total_points = sum(mesh.n_points for mesh in combined_multiblock)
        print(
            f"[Viewport3D] Cached model '{model_name}' with "
            f"{len(combined_multiblock)} parts, {total_points} vertices total"
        )

    def _build_preview_mesh(self, multiblock: pv.MultiBlock) -> pv.PolyData:
        """Combine a multiblock into a single mesh for previews."""
        try:
            combined = multiblock.combine()
            return combined
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[Viewport3D] Preview combine failed: {exc}. Using first block.")
            first = multiblock[0].copy(deep=True)
            return first

    def _compute_bounds_metadata(
        self,
        multiblock: pv.MultiBlock,
        padding: float = 0.5,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute center/length data for highlight transforms."""
        points: List[np.ndarray] = []
        for mesh in multiblock:
            points.append(mesh._original_points)

        if not points:
            return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

        combined = np.vstack(points)
        min_vals = combined.min(axis=0) - padding
        max_vals = combined.max(axis=0) + padding
        center = tuple(((min_vals + max_vals) / 2.0).tolist())
        lengths = tuple((max_vals - min_vals).tolist())
        return center, lengths


    def _activate_preview(self, model_name: str) -> None:
        """Ensure preview actor exists for the model and make it visible."""
        entry = self.model_cache.get(model_name)
        if entry is None:
            self._ensure_model_cached(model_name)
            entry = self.model_cache.get(model_name)

        if entry is None:
            print(f"[Viewport3D] Unable to activate preview for '{model_name}'")
            return

        actor = self._ensure_preview_actor(entry)
        if actor is None:
            return

        if self.preview_actor and self.preview_actor is not actor:
            self.preview_actor.SetVisibility(False)

        self.preview_actor = actor

        # Update styling per mode
        property_ = actor.GetProperty()
        if self.mode == MapBuildingMode.PLACEMENT:
            property_.SetColor(1.0, 1.0, 1.0)
            property_.SetEdgeColor(0.0, 1.0, 1.0)
            property_.SetOpacity(0.45)
        else:
            property_.SetColor(1.0, 1.0, 0.0)
            property_.SetEdgeColor(1.0, 0.5, 0.0)
            property_.SetOpacity(0.6)

        actor.SetVisibility(True)
        self._update_preview(self.current_pose.pos[0], self.current_pose.pos[1])

    def _ensure_preview_actor(self, entry: ModelCacheEntry) -> Optional[pv.Actor]:
        """Create the preview actor if it does not exist."""
        if entry.preview_actor is not None:
            return entry.preview_actor

        actor = self.plotter.add_mesh(
            entry.preview_mesh,
            color="white",
            opacity=0.45,
            show_edges=True,
            edge_color="cyan",
            line_width=2,
            pickable=False,
        )
        actor.SetVisibility(False)
        entry.preview_actor = actor
        return actor


    def _update_preview(self, x: float, y: float) -> None:
        """
        Update preview position by transforming existing mesh

        Args:
            x: World X coordinate
            y: World Y coordinate
        """
        # Store current position
        self.current_pose.pos = (x, y)

        self._transform_preview(x, y, self.current_pose.yaw)

    def _transform_preview(self, x: float, y: float, rotation: float) -> None:
        """
        Transform preview mesh to new position and rotation

        Args:
            x: World X coordinate
            y: World Y coordinate
            rotation: Rotation in radians
        """
        if self.preview_actor is None:
            return

        yaw_degrees = np.degrees(rotation)
        self.preview_actor.SetOrientation(0.0, 0.0, yaw_degrees)
        self.preview_actor.SetPosition(float(x), float(y), 0.0)

    def precache_models(self, model_names: List[str]) -> None:
        """
        Pre-cache multiple models at startup to avoid lag during placement

        Args:
            model_names: List of model profile names to cache
        """
        print(f"[Viewport3D] Pre-caching {len(model_names)} models...")
        for model_name in model_names:
            if model_name not in self.model_cache:
                self._ensure_model_cached(model_name)
        print(f"[Viewport3D] Pre-caching complete. {len(self.model_cache)} models in cache.")

    def get_cached_model_mesh(
        self,
        model_name: str,
    ) -> Tuple[Optional[pv.MultiBlock], Optional[List[Tuple[object, float]]]]:
        """
        Get a copy of the cached model mesh for placement

        Args:
            model_name: Name of the model profile

        Returns:
            Tuple of (MultiBlock copy, materials list) or (None, None)
        """
        if model_name not in self.model_cache:
            self._ensure_model_cached(model_name)

        entry = self.model_cache.get(model_name)
        if entry is None:
            return None, None

        multiblock_copy = pv.MultiBlock()
        for mesh in entry.multiblock:
            mesh_copy = mesh.copy(deep=True)
            mesh_copy._original_points = mesh._original_points.copy()
            multiblock_copy.append(mesh_copy)

        return multiblock_copy, entry.materials

    def render_zones(self, zones: List[dict]) -> None:
        """
        Render zones on the viewport with color coding

        Args:
            zones: List of Zone objects or zone dicts
        """
        # Clear existing zone actors
        self.clear_zones()

        # Zone type to color mapping
        zone_colors = {
            'robot_start_zones': (0.0, 1.0, 0.0),  # Green
            'robot_danger_zones': (1.0, 0.0, 0.0), # Red
            'general_zones': (0.8, 0.8, 0.0)       # Yellow
        }

        for zone in zones:
            # Extract zone data (handle both Zone objects and dicts)
            if hasattr(zone, 'vertices'):
                vertices = zone.vertices
                zone_name = zone.name
                # Infer zone type from allowed_categories
                if hasattr(zone, 'allowed_categories'):
                    allowed = zone.allowed_categories
                    if allowed == ['robot']:
                        zone_type = 'robot_start_zones'  # Or goal, can't distinguish
                    elif len(allowed) == 0:
                        zone_type = 'robot_danger_zones'
                    else:
                        zone_type = 'general_zones'
                else:
                    zone_type = 'general_zones'
            else:
                vertices = zone.get('vertices', [])
                zone_name = zone.get('name', 'unknown')
                zone_type = zone.get('zone_type', 'general_zones')

            if len(vertices) < 3:
                continue

            # Get color for this zone type
            color = zone_colors.get(zone_type, (0.8, 0.8, 0.0))

            # Create polygon mesh at z=0.01 (slightly above grid)
            points_3d = [[v[0], v[1], 0.01] for v in vertices]

            # Create face (list of vertex indices)
            n_verts = len(vertices)
            face = [n_verts] + list(range(n_verts))

            # Create polygon mesh
            points_array = np.array(points_3d)
            poly = pv.PolyData(points_array, face)

            # Render with semi-transparent color
            actor = self.plotter.add_mesh(
                poly,
                color=color,
                opacity=0.3,
                show_edges=True,
                edge_color=color,
                line_width=3
            )
            self.zone_actors.append(actor)

            # Add zone name label at centroid
            centroid = np.mean(points_array, axis=0)
            label_point = pv.PolyData([centroid])
            self.plotter.add_point_labels(
                label_point,
                [zone_name],
                font_size=12,
                text_color=color,
                point_size=0,
                render_points_as_spheres=False,
                always_visible=True
            )

    def clear_zones(self) -> None:
        """Clear all rendered zones from viewport"""
        for actor in self.zone_actors:
            self.plotter.remove_actor(actor)
        self.zone_actors.clear()

    def render_routes(self, routes: List[object]) -> None:
        """Render route starts, waypoints, and connecting lines."""
        self.clear_routes()
        for route_idx, route in enumerate(routes):
            start = getattr(route, "start", None)
            waypoints = getattr(route, "waypoints", [])
            if start is None:
                continue

            start_pos = getattr(start, "pos", (0.0, 0.0))
            points = [(float(start_pos[0]), float(start_pos[1]), 0.12)]
            for waypoint in waypoints:
                pos = getattr(waypoint, "pos", (0.0, 0.0))
                points.append((float(pos[0]), float(pos[1]), 0.12))

            color = "#f0c419" if route_idx % 2 == 0 else "#3ecf8e"
            start_mesh = pv.Sphere(radius=0.45, center=points[0])
            self.route_actors.append(self.plotter.add_mesh(start_mesh, color="#ffffff"))

            for idx, point in enumerate(points[1:], start=1):
                radius = 0.35 if idx < len(points) - 1 else 0.5
                mesh = pv.Sphere(radius=radius, center=point)
                self.route_actors.append(self.plotter.add_mesh(mesh, color=color))

            if len(points) >= 2:
                line = pv.PolyData(np.asarray(points))
                line.lines = np.hstack([[len(points)], np.arange(len(points))])
                self.route_actors.append(self.plotter.add_mesh(line, color=color, line_width=3))

    def clear_routes(self) -> None:
        """Clear rendered route overlays."""
        for actor in self.route_actors:
            self.plotter.remove_actor(actor)
        self.route_actors.clear()

    def highlight_instance(self, instance_pose: Pose2, model_name: str) -> None:
        """
        Highlight a selected instance with a reusable bounding box actor.
        """
        entry = self.model_cache.get(model_name)
        if entry is None:
            self._ensure_model_cached(model_name)
            entry = self.model_cache.get(model_name)

        if entry is None:
            return

        actor = self._ensure_highlight_actor()
        if actor is None:
            return

        lengths = entry.bounds_lengths
        center = entry.bounds_center
        yaw = float(instance_pose.yaw)
        yaw_degrees = np.degrees(yaw)

        actor.SetScale(lengths[0], lengths[1], lengths[2])
        actor.SetOrientation(0.0, 0.0, yaw_degrees)

        rotated_center = self._rotate_point(center, yaw)
        pos_x, pos_y = instance_pose.pos
        actor.SetPosition(
            float(pos_x + rotated_center[0]),
            float(pos_y + rotated_center[1]),
            float(rotated_center[2]),
        )
        actor.SetVisibility(True)

    def _ensure_highlight_actor(self) -> Optional[pv.Actor]:
        """Create the reusable highlight actor if missing."""
        if self.selection_highlight_actor is not None:
            return self.selection_highlight_actor

        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)
        actor = self.plotter.add_mesh(
            cube,
            style="wireframe",
            color="cyan",
            line_width=4,
            opacity=1.0,
            pickable=False,
        )
        actor.SetVisibility(False)
        self.selection_highlight_actor = actor
        return actor

    def _rotate_point(self, point: Tuple[float, float, float], yaw: float) -> Tuple[float, float, float]:
        """Rotate a local point around Z by yaw radians."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        x_local, y_local, z_local = point
        x_rot = x_local * cos_yaw - y_local * sin_yaw
        y_rot = x_local * sin_yaw + y_local * cos_yaw
        return x_rot, y_rot, z_local

    def clear_selection_highlight(self) -> None:
        """Hide the selection highlight bounding box."""
        if self.selection_highlight_actor is not None:
            self.selection_highlight_actor.SetVisibility(False)
