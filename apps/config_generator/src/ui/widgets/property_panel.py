"""
Property panel widget for displaying and editing properties
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFormLayout,
    QLineEdit, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import pyqtSignal


class PropertyPanel(QWidget):
    """Panel for displaying and editing properties"""

    property_changed = pyqtSignal(str, object)  # property_name, value

    def __init__(self, title: str = "Properties", parent=None):
        super().__init__(parent)
        self.title = title
        self.property_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI layout"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(f"<b>{self.title}</b>")
        layout.addWidget(title_label)

        # Scrollable area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        # Container widget
        self.container = QWidget()
        self.form_layout = QFormLayout(self.container)
        scroll.setWidget(self.container)

        layout.addWidget(scroll)

    def clear(self):
        """Clear all properties"""
        # Remove all widgets from form layout
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.property_widgets.clear()

    def add_property(self, name: str, label: str, widget_type: str, value=None, **kwargs):
        """
        Add a property to the panel

        Args:
            name: Property name (key)
            label: Display label
            widget_type: Type of widget ('text', 'float', 'int')
            value: Initial value
            kwargs: Additional widget parameters (e.g., min, max, decimals)
        """
        if widget_type == 'text':
            widget = QLineEdit()
            if value is not None:
                widget.setText(str(value))
            widget.textChanged.connect(lambda v: self.property_changed.emit(name, v))

        elif widget_type == 'float':
            widget = QDoubleSpinBox()
            widget.setMinimum(kwargs.get('min', -999999.0))
            widget.setMaximum(kwargs.get('max', 999999.0))
            widget.setDecimals(kwargs.get('decimals', 2))
            widget.setSingleStep(kwargs.get('step', 0.1))
            if value is not None:
                widget.setValue(float(value))
            widget.valueChanged.connect(lambda v: self.property_changed.emit(name, v))

        elif widget_type == 'int':
            widget = QSpinBox()
            widget.setMinimum(kwargs.get('min', -999999))
            widget.setMaximum(kwargs.get('max', 999999))
            if value is not None:
                widget.setValue(int(value))
            widget.valueChanged.connect(lambda v: self.property_changed.emit(name, v))

        else:
            widget = QLabel(f"Unknown widget type: {widget_type}")

        self.property_widgets[name] = widget
        self.form_layout.addRow(label, widget)

    def set_property_value(self, name: str, value):
        """Set the value of a property"""
        if name in self.property_widgets:
            widget = self.property_widgets[name]
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)

    def get_property_value(self, name: str):
        """Get the value of a property"""
        if name in self.property_widgets:
            widget = self.property_widgets[name]
            if isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                return widget.value()
        return None
