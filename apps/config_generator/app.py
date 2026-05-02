#!/usr/bin/env python3
"""
DDRL Config Generator - Main Entry Point

A 3D interactive tool for generating YAML configuration files
for the 2.5D Distributed Deep Reinforcement Learning simulator.
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor, QFont
from src.ui.main_window import MainWindow


def setup_dark_theme(app: QApplication):
    """Setup dark theme for the application"""
    # Create dark palette
    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))

    # Base colors
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))

    # Text colors
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(127, 127, 127))

    # Button colors
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))

    # Highlight colors
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    # Link colors
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

    # Tooltip colors
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))

    app.setPalette(palette)

    # Set stylesheet for additional styling
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2a82da;
            border: 1px solid white;
        }
        QMenuBar {
            background-color: #353535;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #2a82da;
        }
        QMenu {
            background-color: #353535;
            color: #ffffff;
        }
        QMenu::item:selected {
            background-color: #2a82da;
        }
        QToolBar {
            background-color: #353535;
            border: 1px solid #555555;
            padding: 3px;
        }
        QPushButton {
            background-color: #454545;
            border: 1px solid #5a5a5a;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QPushButton:pressed {
            background-color: #2a82da;
        }
        QPushButton:disabled {
            background-color: #22313f;  /* soft desaturated navy */
            color: #666c75;             /* muted gray text */
        }
    """)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("DDRL Config Generator")
    app.setOrganizationName("DDRL")

    setup_dark_theme(app)

    font = QFont()
    font.setPointSize(12)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
