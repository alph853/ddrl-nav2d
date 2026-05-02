"""
Tab 5: Map Building

Modernized map editor that delegates rendering to the SceneStore/Viewport trio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from src.io.project_manager import ProjectManager
from src.map_editor.controller import MapEditorController
from src.map_editor.documents import MapInstanceDoc, RobotRouteDoc
from src.ui.widgets.thumbnail_list_widget import ThumbnailListWidget
from src.ui.widgets.viewport_3d import Viewport3D
from src.utils.state_manager import get_state


class MapBuildingTab(QWidget):
    """
    Two-pane layout that exposes placement and zone tools driven by the new controller.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.state = get_state()
        self.project_manager = ProjectManager(self.state)
        self.viewport = Viewport3D()

        self._setup_ui()
        self._setup_shortcuts()
        self._sync_grid_size_state()

        self.controller = MapEditorController(
            state=self.state,
            viewport=self.viewport,
            project_manager=self.project_manager,
            on_instances_changed=self._refresh_instance_list,
            on_zones_changed=lambda _: None,
            on_routes_changed=self._refresh_route_list,
            on_status_changed=self._update_status,
            request_zone_name=self._request_zone_name,
        )
        self.controller.refresh_from_state()
        self.refresh_model_profiles()

    # ------------------------------------------------------------------ ui setup
    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        self.library_panel = ThumbnailListWidget(title="Model Profiles")
        self.library_panel.setFixedWidth(220)
        self.library_panel.add_clicked.connect(self.refresh_model_profiles)
        self.library_panel.item_selected.connect(self._on_model_profile_selected)
        self.library_panel.btn_delete.setEnabled(False)
        self.library_panel.btn_delete.setVisible(False)
        content_layout.addWidget(self.library_panel)

        self.viewport.set_camera_top_view()
        content_layout.addWidget(self.viewport, stretch=1)

        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        instance_panel = self._create_instance_panel()
        right_layout.addWidget(instance_panel, stretch=1)

        route_panel = self._create_route_panel()
        right_layout.addWidget(route_panel, stretch=1)

        content_layout.addWidget(right_panel)

    def _setup_shortcuts(self) -> None:
        delete_shortcut = QShortcut(QKeySequence("Delete"), self)
        delete_shortcut.activated.connect(self._on_delete_selected_instance)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._on_save_map)

        edit_shortcut = QShortcut(QKeySequence("E"), self)
        edit_shortcut.activated.connect(self._on_edit_mode_shortcut)

    # ------------------------------------------------------------------ toolbars/panels
    def _create_toolbar(self) -> QToolBar:
        toolbar = QToolBar()

        self.btn_placement_mode = QPushButton("📦 Placement Mode")
        self.btn_placement_mode.setCheckable(True)
        self.btn_placement_mode.setChecked(True)
        self.btn_placement_mode.clicked.connect(self._on_placement_mode)
        toolbar.addWidget(self.btn_placement_mode)

        self.btn_route_mode = QPushButton("Route Mode")
        self.btn_route_mode.setCheckable(True)
        self.btn_route_mode.clicked.connect(self._on_route_mode)
        toolbar.addWidget(self.btn_route_mode)

        toolbar.addSeparator()

        self.status_label = QLabel("Select a model and click in the viewport to place.")
        self.status_label.setMinimumWidth(400)
        toolbar.addWidget(self.status_label)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Grid Snap:"))
        self.combo_grid_snap = QComboBox()
        self.combo_grid_snap.addItems(["0.5 m", "1.0 m", "5.0 m", "10.0 m"])
        self.combo_grid_snap.setCurrentIndex(1)
        self.combo_grid_snap.currentTextChanged.connect(self._on_grid_snap_changed)
        toolbar.addWidget(self.combo_grid_snap)

        toolbar.addWidget(QLabel("Grid Size:"))
        self.combo_grid_size = QComboBox()
        self.combo_grid_size.addItems(["50 m", "100 m", "200 m", "500 m", "1000 m"])
        self.combo_grid_size.setCurrentIndex(3)
        self.combo_grid_size.currentTextChanged.connect(self._on_grid_size_changed)
        toolbar.addWidget(self.combo_grid_size)

        toolbar.addSeparator()

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_all)
        toolbar.addWidget(btn_clear)

        btn_load = QPushButton("Load Map YAML")
        btn_load.clicked.connect(self._on_load_map)
        toolbar.addWidget(btn_load)

        btn_save = QPushButton("Export Map YAML")
        btn_save.clicked.connect(self._on_save_map)
        toolbar.addWidget(btn_save)

        toolbar.addSeparator()

        btn_top = QPushButton("Top View")
        btn_top.clicked.connect(self.viewport.set_camera_top_view)
        toolbar.addWidget(btn_top)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self.viewport.reset_camera)
        toolbar.addWidget(btn_reset)

        return toolbar

    def _create_instance_panel(self) -> QGroupBox:
        panel = QGroupBox("Placed Instances")
        layout = QVBoxLayout(panel)

        self.instance_list = QListWidget()
        self.instance_list.itemSelectionChanged.connect(self._on_instance_selected)
        layout.addWidget(self.instance_list, stretch=1)

        button_row = QHBoxLayout()
        btn_edit = QPushButton("Edit Selected (E)")
        btn_edit.clicked.connect(self._on_edit_selected_instance)
        button_row.addWidget(btn_edit)

        btn_delete = QPushButton("Delete")
        btn_delete.clicked.connect(self._on_delete_selected_instance)
        button_row.addWidget(btn_delete)

        layout.addLayout(button_row)
        return panel

    def _create_route_panel(self) -> QGroupBox:
        panel = QGroupBox("Robot Routes")
        layout = QVBoxLayout(panel)

        self.btn_start_route = QPushButton("Start New Route")
        self.btn_start_route.clicked.connect(self._on_start_new_route)
        self.btn_start_route.setEnabled(False)
        layout.addWidget(self.btn_start_route)

        self.btn_finish_route = QPushButton("Finish Route")
        self.btn_finish_route.clicked.connect(self._on_finish_route)
        self.btn_finish_route.setEnabled(False)
        layout.addWidget(self.btn_finish_route)

        self.btn_cancel_route = QPushButton("Cancel Route")
        self.btn_cancel_route.clicked.connect(self._on_cancel_route)
        self.btn_cancel_route.setEnabled(False)
        layout.addWidget(self.btn_cancel_route)

        self.route_list = QListWidget()
        layout.addWidget(self.route_list, stretch=1)

        btn_delete_route = QPushButton("Delete Selected Route")
        btn_delete_route.clicked.connect(self._on_delete_route)
        layout.addWidget(btn_delete_route)

        return panel

    # ------------------------------------------------------------------ callbacks
    def refresh_model_profiles(self) -> None:
        """Refresh thumbnail list grouped by category."""
        self.library_panel.clear()
        profiles_by_category: dict[str, list[str]] = {}
        for profile_name in self.state.get_model_profile_names():
            profile = self.state.get_model_profile(profile_name)
            category = getattr(profile, "category", "other")
            profiles_by_category.setdefault(category, []).append(profile_name)

        for category in sorted(profiles_by_category):
            for profile_name in sorted(profiles_by_category[category]):
                profile = self.state.get_model_profile(profile_name)
                if profile is None:
                    continue
                self.library_panel.add_item(profile_name, profile)

        model_names = self.state.get_model_profile_names()
        if model_names:
            self.viewport.precache_models(model_names)

    def _refresh_instance_list(self, instances: Sequence[MapInstanceDoc]) -> None:
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        for instance in instances:
            item = QListWidgetItem(
                f"{instance.name} - {instance.model_id} @ ({instance.pose.pos[0]:.1f}, {instance.pose.pos[1]:.1f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, instance.instance_id)
            self.instance_list.addItem(item)
        self.instance_list.blockSignals(False)

    def _refresh_route_list(self, routes: Sequence[RobotRouteDoc]) -> None:
        self.viewport.render_routes(list(routes))
        self.route_list.clear()
        for route in routes:
            item = QListWidgetItem(
                f"{route.name} start=({route.start.pos[0]:.1f}, {route.start.pos[1]:.1f}) "
                f"({len(route.waypoints)} waypoints)"
            )
            item.setData(Qt.ItemDataRole.UserRole, route.route_id)
            self.route_list.addItem(item)

    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _request_zone_name(self, zone_type: str, suggested: str) -> str | None:
        name, ok = QInputDialog.getText(
            self,
            "Zone Name",
            f"Enter name for {zone_type}:",
            text=suggested,
        )
        return name if ok and name else None

    # ------------------------------------------------------------------ instance actions
    def _on_model_profile_selected(self, model_name: str) -> None:
        profile = self.state.get_model_profile(model_name)
        if profile is None:
            QMessageBox.warning(self, "Missing Profile", f"{model_name} is not loaded.")
            return
        self.controller.begin_placement(model_name)

    def _on_instance_selected(self) -> None:
        current_item = self.instance_list.currentItem()
        if not current_item:
            self.viewport.clear_selection_highlight()
            return
        instance_id = current_item.data(Qt.ItemDataRole.UserRole)
        if instance_id:
            self.controller.highlight_instance(instance_id)

    def _on_edit_selected_instance(self) -> None:
        current_item = self.instance_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Select an instance first.")
            return
        instance_id = current_item.data(Qt.ItemDataRole.UserRole)
        if instance_id:
            self.controller.enter_list_edit_mode(instance_id)

    def _on_delete_selected_instance(self) -> None:
        current_item = self.instance_list.currentItem()
        if not current_item:
            return
        instance_id = current_item.data(Qt.ItemDataRole.UserRole)
        if not instance_id:
            return

        reply = QMessageBox.question(
            self,
            "Delete Instance",
            "Delete selected instance?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.delete_instance(instance_id)

    def _on_clear_all(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear Map",
            "Remove all instances and zones?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.clear_all()
            self.instance_list.clear()
            self.route_list.clear()

    def _on_placement_mode(self) -> None:
        self.btn_placement_mode.setChecked(True)
        self.btn_placement_mode.setEnabled(False)
        self.btn_route_mode.setChecked(False)
        self.btn_route_mode.setEnabled(True)
        self.library_panel.setVisible(True)
        self.btn_start_route.setEnabled(False)
        self.btn_finish_route.setEnabled(False)
        self.btn_cancel_route.setEnabled(False)
        self.controller.cancel_route()
        self.viewport.exit_mode()
        self._update_status("Placement Mode: select a model to place.")

    def _on_route_mode(self) -> None:
        self.btn_route_mode.setChecked(True)
        self.btn_route_mode.setEnabled(False)
        self.btn_placement_mode.setChecked(False)
        self.btn_placement_mode.setEnabled(True)
        self.library_panel.setVisible(False)
        self.btn_start_route.setEnabled(True)
        self._update_status("Route Mode: click Start New Route to begin.")

    def _on_start_new_route(self) -> None:
        suggested = f"route_{len(self.state.current_map_routes) + 1}"
        name, ok = QInputDialog.getText(self, "Route Name", "Enter route name:", text=suggested)
        if not ok or not name:
            return
        tolerance, ok = QInputDialog.getDouble(
            self,
            "Waypoint Tolerance",
            "Reach tolerance for route waypoints (m):",
            1.5,
            0.1,
            100.0,
            2,
        )
        if not ok:
            return
        self.controller.start_route(name, tolerance)
        self.btn_start_route.setEnabled(False)
        self.btn_finish_route.setEnabled(True)
        self.btn_cancel_route.setEnabled(True)

    def _on_finish_route(self) -> None:
        if self.controller.finish_route():
            self.btn_start_route.setEnabled(True)
            self.btn_finish_route.setEnabled(False)
            self.btn_cancel_route.setEnabled(False)

    def _on_cancel_route(self) -> None:
        self.controller.cancel_route()
        self.btn_start_route.setEnabled(True)
        self.btn_finish_route.setEnabled(False)
        self.btn_cancel_route.setEnabled(False)

    def _on_delete_route(self) -> None:
        current_item = self.route_list.currentItem()
        if not current_item:
            return
        route_id = current_item.data(Qt.ItemDataRole.UserRole)
        if route_id:
            self.controller.delete_route(route_id)

    # ------------------------------------------------------------------ persistence
    def _on_load_map(self) -> None:
        if not self.state.config_root_path:
            QMessageBox.warning(self, "Load Failed", "Load config schemas first.")
            return

        map_dir = Path(self.state.config_root_path) / "map"
        map_dir.mkdir(exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Map YAML",
            str(map_dir),
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return

        try:
            self.controller.load_map(Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))

    def _on_save_map(self) -> None:
        default_name = self.controller.document.name
        name, ok = QInputDialog.getText(
            self,
            "Export Map",
            "Enter map name:",
            text=default_name,
        )
        if not ok or not name:
            return

        try:
            save_path = self.controller.save_map(name)
            QMessageBox.information(self, "Export Success", f"Saved to:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    # ------------------------------------------------------------------ misc ui
    def _on_grid_snap_changed(self, snap_text: str) -> None:
        spacing = float(snap_text.split()[0])
        self.viewport.show_grid(True, spacing=spacing)

    def _on_grid_size_changed(self, size_text: str) -> None:
        size = float(size_text.split()[0])
        spacing = float(self.combo_grid_snap.currentText().split()[0])
        self.viewport.show_grid(True, size=size, spacing=spacing)
        self.state.current_map_grid_size = size

    def _sync_grid_size_state(self) -> None:
        size = float(self.combo_grid_size.currentText().split()[0])
        self.state.current_map_grid_size = size

    def _on_edit_mode_shortcut(self) -> None:
        self.controller.toggle_select_mode()
