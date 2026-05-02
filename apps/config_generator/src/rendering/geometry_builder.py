"""
Shared geometry builder for creating PyVista meshes from model parts
"""
import numpy as np
import pyvista as pv
from src.models.geometry import Box2, Circle, Polygon2
from src.models.model_profile import ModelPart


def create_mesh_from_part(part: ModelPart, world_x: float = 0.0, world_y: float = 0.0, world_yaw: float = 0.0) -> pv.PolyData:
    """
    Create a PyVista mesh from a ModelPart with world pose transformation

    Args:
        part: ModelPart containing geometry, material, and relative pose
        world_x: World X position
        world_y: World Y position
        world_yaw: World yaw rotation in radians

    Returns:
        PyVista PolyData mesh
    """
    geom = part.geom

    # Calculate part's world position considering rel_pose
    cos_yaw = np.cos(world_yaw)
    sin_yaw = np.sin(world_yaw)

    # Apply rotation to rel_pose offset
    rel_x = part.rel_pose.pos[0]
    rel_y = part.rel_pose.pos[1]
    rotated_rel_x = rel_x * cos_yaw - rel_y * sin_yaw
    rotated_rel_y = rel_x * sin_yaw + rel_y * cos_yaw

    part_x = world_x + rotated_rel_x
    part_y = world_y + rotated_rel_y

    # Also consider part's own rotation
    part_yaw = world_yaw + part.rel_pose.yaw

    # Create mesh based on geometry type
    height = geom.z_max - geom.z_min
    center_z = (geom.z_min + geom.z_max) / 2

    mesh = None

    if isinstance(geom.base, Box2):
        # Create box at origin
        mesh = pv.Box(bounds=[
            -geom.base.hx, geom.base.hx,
            -geom.base.hy, geom.base.hy,
            -height/2, height/2
        ])
        # Apply part's own rotation if needed
        if part_yaw != 0:
            mesh.rotate_z(np.degrees(part_yaw), inplace=True)
        # Translate to world position
        mesh.translate([part_x, part_y, center_z], inplace=True)

    elif isinstance(geom.base, Circle):
        # Cylinder (rotation doesn't affect circles in 2D)
        mesh = pv.Cylinder(
            center=(part_x, part_y, center_z),
            direction=(0, 0, 1),
            radius=geom.base.radius,
            height=height
        )

    elif isinstance(geom.base, Polygon2):
        # Create extruded polygon
        vertices = geom.base.v

        # Apply rotation to polygon vertices
        bottom_verts = []
        top_verts = []

        for v in vertices:
            # Rotate vertex by part_yaw
            if part_yaw != 0:
                vx = v[0] * np.cos(part_yaw) - v[1] * np.sin(part_yaw)
                vy = v[0] * np.sin(part_yaw) + v[1] * np.cos(part_yaw)
            else:
                vx, vy = v[0], v[1]

            # Translate to world position
            bottom_verts.append([part_x + vx, part_y + vy, geom.z_min])
            top_verts.append([part_x + vx, part_y + vy, geom.z_max])

        all_verts = np.vstack([bottom_verts, top_verts])

        # Create faces
        faces = []
        n = len(vertices)
        # Bottom face (counter-clockwise from below)
        faces.extend([n] + list(range(n-1, -1, -1)))
        # Top face (counter-clockwise from above)
        faces.extend([n] + list(range(n, 2*n)))
        # Side faces
        for i in range(n):
            next_i = (i + 1) % n
            faces.extend([4, i, next_i, n + next_i, n + i])

        mesh = pv.PolyData(all_verts, faces)

    else:
        # Fallback: simple cube
        mesh = pv.Cube()
        if part_yaw != 0:
            mesh.rotate_z(np.degrees(part_yaw), inplace=True)
        mesh.translate([part_x, part_y, center_z], inplace=True)

    return mesh


def create_mesh_at_origin(part: ModelPart) -> pv.PolyData:
    """
    Create a PyVista mesh from a ModelPart at origin (for preview/thumbnails)

    Args:
        part: ModelPart containing geometry

    Returns:
        PyVista PolyData mesh at origin
    """
    return create_mesh_from_part(part, 0.0, 0.0, 0.0)


def create_combined_model_mesh(model_profile) -> pv.PolyData:
    """
    Create a single combined PyVista mesh from all parts of a model profile.
    All parts are transformed relative to the model origin and merged into one mesh.
    Returns a MultiBlock to preserve per-part materials.

    Args:
        model_profile: ModelProfile containing multiple parts

    Returns:
        Tuple of (PyVista MultiBlock, list of (color, opacity) tuples for each part)
    """
    meshes = []
    materials = []  # Store (color, opacity) for each part

    for part in model_profile.parts:
        # Create mesh for this part at its relative position
        geom = part.geom
        height = geom.z_max - geom.z_min
        center_z = (geom.z_min + geom.z_max) / 2

        # Get relative pose
        rel_x = part.rel_pose.pos[0]
        rel_y = part.rel_pose.pos[1]
        rel_yaw = part.rel_pose.yaw

        mesh = None

        if isinstance(geom.base, Box2):
            # Create box at origin
            mesh = pv.Box(bounds=[
                -geom.base.hx, geom.base.hx,
                -geom.base.hy, geom.base.hy,
                -height/2, height/2
            ])
            # Apply part's own rotation
            if rel_yaw != 0:
                mesh.rotate_z(np.degrees(rel_yaw), inplace=True)
            # Translate to relative position
            mesh.translate([rel_x, rel_y, center_z], inplace=True)

        elif isinstance(geom.base, Circle):
            # Cylinder
            mesh = pv.Cylinder(
                center=(rel_x, rel_y, center_z),
                direction=(0, 0, 1),
                radius=geom.base.radius,
                height=height
            )

        elif isinstance(geom.base, Polygon2):
            # Create extruded polygon
            vertices = geom.base.v

            # Apply rotation to polygon vertices
            bottom_verts = []
            top_verts = []

            for v in vertices:
                # Rotate vertex by rel_yaw
                if rel_yaw != 0:
                    vx = v[0] * np.cos(rel_yaw) - v[1] * np.sin(rel_yaw)
                    vy = v[0] * np.sin(rel_yaw) + v[1] * np.cos(rel_yaw)
                else:
                    vx, vy = v[0], v[1]

                # Translate to relative position
                bottom_verts.append([rel_x + vx, rel_y + vy, geom.z_min])
                top_verts.append([rel_x + vx, rel_y + vy, geom.z_max])

            all_verts = np.vstack([bottom_verts, top_verts])

            # Create faces
            faces = []
            n = len(vertices)
            # Bottom face
            faces.extend([n] + list(range(n-1, -1, -1)))
            # Top face
            faces.extend([n] + list(range(n, 2*n)))
            # Side faces
            for i in range(n):
                next_i = (i + 1) % n
                faces.extend([4, i, next_i, n + next_i, n + i])

            mesh = pv.PolyData(all_verts, faces)

        else:
            # Fallback: simple cube
            mesh = pv.Cube()
            if rel_yaw != 0:
                mesh.rotate_z(np.degrees(rel_yaw), inplace=True)
            mesh.translate([rel_x, rel_y, center_z], inplace=True)

        if mesh is not None:
            meshes.append(mesh)
            # Extract material
            material = part.material
            color = material.rgba[:3]
            opacity = material.rgba[3]
            materials.append((color, opacity))

    # Create MultiBlock to preserve per-part materials
    if len(meshes) == 0:
        # Fallback
        fallback = pv.Cube()
        multi = pv.MultiBlock([fallback])
        return multi, [(0.5, 0.5, 0.5, 1.0)]
    else:
        multi = pv.MultiBlock(meshes)
        return multi, materials
