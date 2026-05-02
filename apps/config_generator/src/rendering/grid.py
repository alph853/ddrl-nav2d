"""
Grid rendering utilities
"""
import numpy as np
import pyvista as pv


def create_grid(size: float = 50.0, spacing: float = 1.0, color: str = "gray") -> pv.PolyData:
    """
    Create a grid mesh for the floor plane

    Args:
        size: Total size of the grid (meters)
        spacing: Spacing between grid lines (meters)
        color: Grid color

    Returns:
        PyVista PolyData mesh for the grid
    """
    # Create grid lines
    lines = []
    points = []

    # Create lines parallel to X axis
    num_lines = int(size / spacing) + 1
    for i in range(num_lines):
        y = -size / 2 + i * spacing
        points.extend([[-size / 2, y, 0], [size / 2, y, 0]])

    # Create lines parallel to Y axis
    for i in range(num_lines):
        x = -size / 2 + i * spacing
        points.extend([[x, -size / 2, 0], [x, size / 2, 0]])

    points = np.array(points)

    # Create line connectivity
    for i in range(0, len(points), 2):
        lines.extend([2, i, i + 1])

    grid = pv.PolyData(points, lines=lines)
    return grid


def create_axes(length: float = 5.0) -> pv.PolyData:
    """
    Create coordinate axes (X=red, Y=green, Z=blue)

    Args:
        length: Length of each axis

    Returns:
        PyVista PolyData mesh for axes
    """
    # X axis (red)
    x_axis = pv.Line((0, 0, 0), (length, 0, 0))

    # Y axis (green)
    y_axis = pv.Line((0, 0, 0), (0, length, 0))

    # Z axis (blue)
    z_axis = pv.Line((0, 0, 0), (0, 0, length))

    return x_axis, y_axis, z_axis
