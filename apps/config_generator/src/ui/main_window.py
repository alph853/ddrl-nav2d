"""
Main application window with 6-tab structure
"""
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from pathlib import Path
from src.utils.state_manager import AppState
from src.io.project_manager import ProjectManager


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DDRL Config Generator")
        self._set_initial_geometry()

        # Get shared state manager (singleton)
        from src.utils.state_manager import get_state
        self.state = get_state()
        self.project_manager = ProjectManager(self.state)
        self.current_project_file = None

        self._setup_statusbar()
        self._setup_ui()
        self._setup_menu()

    def _set_initial_geometry(self):
        """Choose an initial size that fits the active screen."""
        screen = self.screen()
        if screen is None:
            self.resize(1280, 800)
            return

        available = screen.availableGeometry()
        width = min(1600, max(900, available.width() - 100))
        height = min(900, max(600, available.height() - 100))
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(x, y, width, height)

    def _setup_ui(self):
        """Setup main UI layout with tabs"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        from src.ui.tabs.map_building_tab import MapBuildingTab

        self.tab_map_building = MapBuildingTab()

        self.tabs.addTab(self.tab_map_building, "Map Building")

        # Load config schemas (model profiles, etc.)
        self._load_config_schemas()

    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        # new_action.setShortcut("Ctrl+N")  # Removed to avoid conflict with tab shortcuts
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        # save_action.setShortcut("Ctrl+S")  # Removed to avoid conflict with tab shortcuts
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addAction(redo_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        grid_action = QAction("Toggle &Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        view_menu.addAction(grid_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def set_tab(self, index: int, widget: QWidget, title: str):
        """Replace a placeholder tab with actual implementation"""
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, widget, title)

    def _on_new_project(self):
        """Create a new project (clear all libraries)"""
        # Clear current map data
        self.state.clear_map()

        self.current_project_file = None
        self.setWindowTitle("DDRL Config Generator - New Project")
        self.statusbar.showMessage("New project created")

    def _on_open_project(self):
        """Open an existing project file"""
        # Calculate config path relative to this file
        config_path = Path(__file__).parent.parent.parent.parent.parent / "config"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(config_path),
            "Project Files (*.cgproj);;All Files (*)"
        )

        if not file_path:
            return

        pass

    def _on_save_project(self):
        """Save current project"""
        pass

    def _on_save_project_as(self):
        """Save project with new filename"""
        # Calculate config path relative to this file
        config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "project.cgproj"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(config_path),
            "Project Files (*.cgproj);;All Files (*)"
        )

        if not file_path:
            return

        # Add extension if not present
        if not file_path.endswith('.cgproj'):
            file_path += '.cgproj'

        pass

    def _load_config_schemas(self):
        """Load config schemas (model profiles) from config/ folder"""
        from pathlib import Path

        # Calculate config root path relative to this file
        # Path structure: apps/config_generator/src/ui/main_window.py -> ../../../../config
        config_root = Path(__file__).parent.parent.parent.parent.parent / "config"

        if config_root.exists():
            success = self.project_manager.load_config_schemas(config_root)
            if success:
                # Refresh map building tab's model profile library
                if hasattr(self.tab_map_building, 'refresh_model_profiles'):
                    self.tab_map_building.refresh_model_profiles()

                self.statusbar.showMessage(
                    f"Loaded {len(self.state.model_profiles)} model profiles from {config_root}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Load Warning",
                    f"Failed to load config schemas from:\n{config_root}"
                )
        else:
            QMessageBox.warning(
                self,
                "Config Not Found",
                f"Config folder not found at:\n{config_root}\n\n"
                "Please ensure the config folder exists."
            )
