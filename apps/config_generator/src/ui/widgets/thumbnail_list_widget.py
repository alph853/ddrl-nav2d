"""
List widget with thumbnail previews next to block names
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QIcon
from src.models.model_profile import ModelPart, ModelProfile
import pyvista as pv


class ThumbnailListWidget(QWidget):
    """List widget showing blocks with thumbnail icons"""

    item_selected = pyqtSignal(str)  # Emits block name
    item_deleted = pyqtSignal(str)   # Emits block name when deleted
    add_clicked = pyqtSignal()       # Emitted when Add button clicked

    def __init__(self, title: str = "Library", parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI layout"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(f"<b>{self.title}</b>")
        layout.addWidget(title_label)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(64, 64))  # Icon size for thumbnails
        self.list_widget.setSpacing(2)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemClicked.connect(self._on_item_clicked)

        layout.addWidget(self.list_widget)        # Buttons
        button_layout = QHBoxLayout()

        self.btn_add = QPushButton("+ New")
        self.btn_add.clicked.connect(self.add_clicked.emit)
        button_layout.addWidget(self.btn_add)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_delete.setEnabled(False)
        button_layout.addWidget(self.btn_delete)

        layout.addLayout(button_layout)

    def add_item(self, name: str, model_data=None):
        """
        Add an item to the list with thumbnail

        Args:
            name: Block name
            model_data: ModelPart or ModelProfile object (optional, for generating thumbnail)
        """
        item = QListWidgetItem(name)

        if model_data:
            # Generate thumbnail
            thumbnail = self._generate_thumbnail(model_data)
            icon = QIcon(thumbnail)
            item.setIcon(icon)

        self.list_widget.addItem(item)

    def remove_item(self, name: str):
        """Remove an item from the list"""
        items = self.list_widget.findItems(name, Qt.MatchFlag.MatchExactly)
        for item in items:
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

    def clear(self):
        """Clear all items"""
        self.list_widget.clear()

    def get_items(self):
        """Get list of all item names"""
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

    def get_selected_item(self):
        """Get currently selected item name"""
        selected = self.list_widget.selectedItems()
        if selected:
            return selected[0].text()
        return None

    def refresh(self):
        """Refresh the list (does nothing - items managed by add/remove)"""
        pass

    def _on_selection_changed(self):
        """Handle selection change"""
        selected = self.get_selected_item()
        self.btn_delete.setEnabled(selected is not None)

    def _on_item_clicked(self, item):
        """Handle single-click on item - activate placement mode"""
        self.item_selected.emit(item.text())

    def _on_delete_clicked(self):
        """Handle delete button click"""
        selected = self.get_selected_item()
        if selected:
            self.item_deleted.emit(selected)
            self.remove_item(selected)

    def _generate_thumbnail(self, model_data) -> QPixmap:
        """
        Generate a thumbnail image of a block using off-screen rendering

        Args:
            model_data: ModelPart or ModelProfile object

        Returns:
            QPixmap containing the rendered thumbnail
        """
        try:
            # Create off-screen plotter
            plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
            plotter.set_background('#2a2a2a')

            # Handle both ModelPart and ModelProfile
            if isinstance(model_data, ModelProfile):
                # Render all parts of the model profile
                for part in model_data.parts:
                    self._add_part_to_plotter(plotter, part)
            else:
                # Assume it's a ModelPart
                self._add_part_to_plotter(plotter, model_data)

            # Set camera to isometric view
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.3)

            # Render to image
            image = plotter.screenshot(return_img=True)
            plotter.close()

            # Convert numpy array to QPixmap
            height, width, channels = image.shape
            bytes_per_line = channels * width

            # Convert numpy array to bytes for QImage
            image_bytes = image.tobytes()
            q_image = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.copy())  # Copy to ensure data persists

            return pixmap

        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            import traceback
            traceback.print_exc()

            # Return a placeholder image
            pixmap = QPixmap(64, 64)
            pixmap.fill(QColor(60, 60, 60))

            painter = QPainter(pixmap)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "?")
            painter.end()

            return pixmap

    def _add_part_to_plotter(self, plotter, model_part: ModelPart):
        """Add a single model part to the plotter"""
        from src.rendering.geometry_builder import create_mesh_at_origin

        # Create mesh using shared geometry builder
        mesh = create_mesh_at_origin(model_part)

        # Extract material
        material = model_part.material
        color = material.rgba[:3]
        opacity = material.rgba[3]

        # Add mesh to plotter
        if mesh is not None:
            plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=True, edge_color='black')

