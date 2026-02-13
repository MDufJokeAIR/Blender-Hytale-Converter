# coding: utf-8
"""
Hytale Blocky Model Exporter v18
New approach:
  1. Start with one block covering the full AABB of the mesh
  2. Subdivide into max_blocks_number smaller blocks following mesh proportions
  3. Trim each block to fit only the mesh volume (down to 0 = plane)
  4. Optionally rotate each block to align with local surface (adjust_orientation)
  5. Repeat for each mesh object
"""

bl_info = {
    "name": "Export Hytale Blocky Model",
    "author": "Claude",
    "version": (18, 0, 0),
    "blender": (2, 80, 0),
    "location": "File > Export > Hytale Blocky Model (.blockymodel)",
    "description": "Export meshes to Hytale .blockymodel format",
    "category": "Import-Export",
}

import bpy
import bmesh
import json
import math
import time
import numpy as np

from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
from mathutils import Vector, Matrix, Quaternion


# ============================================================================
# JSON node helpers
# ============================================================================

def make_box_node(node_id, name, position, size, orientation):
    face_layout = {}
    for face in ("back", "right", "front", "left", "top", "bottom"):
        face_layout[face] = {
            "offset": {"x": 0, "y": 0},
            "mirror": {"x": False, "y": False},
            "angle":  0
        }
    return {
        "id": str(node_id),
        "name": name,
        "position": {
            "x": round(position.x, 3),
            "y": round(position.y, 3),
            "z": round(position.z, 3),
        },
        "orientation": {
            "x": round(orientation.x, 6),
            "y": round(orientation.y, 6),
            "z": round(orientation.z, 6),
            "w": round(orientation.w, 6),
        },
        "shape": {
            "type": "box",
            "offset":  {"x": 0, "y": 0, "z": 0},
            "stretch": {"x": 1, "y": 1, "z": 1},
            "settings": {
                "isPiece": False,
                "size": {
                    "x": round(size.x, 3),
                    "y": round(size.y, 3),
                    "z": round(size.z, 3),
                },
                "isStaticBox": True,
            },
            "textureLayout": face_layout,
            "unwrapMode": "custom",
            "visible":     True,
            "doubleSided": False,
            "shadingMode": "flat",
        },
    }


def make_group_node(node_id, name, position):
    return {
        "id": str(node_id),
        "name": name,
        "position": {
            "x": round(position.x, 3),
            "y": round(position.y, 3),
            "z": round(position.z, 3),
        },
        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
        "shape": {
            "type": "none",
            "offset":  {"x": 0, "y": 0, "z": 0},
            "stretch": {"x": 1, "y": 1, "z": 1},
            "settings": {"isPiece": False},
            "textureLayout": {},
            "unwrapMode": "custom",
            "visible":     True,
            "doubleSided": False,
            "shadingMode": "flat",
        },
        "children": [],
    }


# ============================================================================
# Mesh data helpers  (all copied before bm.free())
# ============================================================================

def get_mesh_data(obj):
    """Return (vertices, faces, normals) with plain Python/mathutils types."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.normal_update()

    vertices = [v.co.copy()                for v in bm.verts]
    faces    = [[v.index for v in f.verts] for f in bm.faces]
    normals  = [f.normal.copy()            for f in bm.faces]

    bm.free()
    return vertices, faces, normals


def mesh_aabb(vertices):
    mn = Vector((min(v.x for v in vertices),
                 min(v.y for v in vertices),
                 min(v.z for v in vertices)))
    mx = Vector((max(v.x for v in vertices),
                 max(v.y for v in vertices),
                 max(v.z for v in vertices)))
    return mn, mx


def sample_face_points(vertices, faces):
    """
    Pre-compute sample points on every mesh face:
    vertices + edge midpoints + face center + sub-samples.
    These are used to test if a face actually passes through a cell.
    """
    samples = []
    for face in faces:
        verts = [vertices[i] for i in face]
        n = len(verts)
        if n == 0:
            continue

        # Vertices
        for v in verts:
            samples.append(v)

        # Edge midpoints
        for k in range(n):
            mid = (verts[k] + verts[(k + 1) % n]) * 0.5
            samples.append(mid)

        # Face center
        center = sum(verts, Vector((0, 0, 0))) * (1.0 / n)
        samples.append(center)

        # For larger faces, add barycentric sub-samples to avoid missing thin cells
        if n >= 3:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            for u in (0.25, 0.5, 0.75):
                for v in (0.25, 0.5, 0.75):
                    if u + v <= 1.0:
                        samples.append(v0 * (1 - u - v) + v1 * u + v2 * v)

    return samples


# ============================================================================
# Core subdivision algorithm
# ============================================================================

def subdivide_mesh_into_blocks(obj, max_blocks, adjust_orientation, plane_threshold, merge_threshold=0.95):
    """
    1. Compute AABB of the mesh.
    2. Compute proportional grid (nx x ny x nz) so total <= max_blocks,
       with more cuts along the longest axis.
    3. For each cell:
       - Check if any mesh FACE actually passes through the cell
         (using pre-sampled face points - no margin bleeding).
       - Compute tight AABB of those face points clamped to the cell.
       - Apply plane_threshold (thin dim -> 0).
       - Optionally run PCA to find a better-fitting oriented box.
    4. Return list of (center, size, quaternion).
    """
    vertices, faces, normals = get_mesh_data(obj)
    if not vertices:
        return []

    mn, mx = mesh_aabb(vertices)
    extent = mx - mn

    total_ext = extent.x + extent.y + extent.z
    if total_ext < 1e-6:
        return []

    # Pre-compute face sample points (replaces the margin-based vertex approach)
    face_samples = sample_face_points(vertices, faces)
    print(f"  Face samples: {len(face_samples)}")

    # --- Proportional grid ---
    wx = extent.x / total_ext
    wy = extent.y / total_ext
    wz = extent.z / total_ext

    cbrt = max_blocks ** (1.0 / 3.0)
    nx = max(1, round(cbrt * wx * 3))
    ny = max(1, round(cbrt * wy * 3))
    nz = max(1, round(cbrt * wz * 3))

    # Clamp to max_blocks by reducing along smallest axis
    while nx * ny * nz > max_blocks:
        if   nx <= ny and nx <= nz and nx > 1:  nx -= 1
        elif ny <= nx and ny <= nz and ny > 1:  ny -= 1
        elif nz > 1:                             nz -= 1
        else:                                    break

    print(f"  Grid: {nx} x {ny} x {nz}  ({nx*ny*nz} cells)")

    step   = Vector((extent.x / nx, extent.y / ny, extent.z / nz))
    blocks = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cell_min = Vector((
                    mn.x + ix * step.x,
                    mn.y + iy * step.y,
                    mn.z + iz * step.z,
                ))
                cell_max = cell_min + step

                # Only keep samples that land STRICTLY inside this cell
                # (no margin -> no bleeding into neighbouring empty cells)
                cell_samples = [
                    p for p in face_samples
                    if (cell_min.x <= p.x <= cell_max.x and
                        cell_min.y <= p.y <= cell_max.y and
                        cell_min.z <= p.z <= cell_max.z)
                ]

                if not cell_samples:
                    continue   # No mesh face passes through this cell -> skip

                # Tight AABB of samples clamped to cell
                tight_min = Vector((
                    max(cell_min.x, min(p.x for p in cell_samples)),
                    max(cell_min.y, min(p.y for p in cell_samples)),
                    max(cell_min.z, min(p.z for p in cell_samples)),
                ))
                tight_max = Vector((
                    min(cell_max.x, max(p.x for p in cell_samples)),
                    min(cell_max.y, max(p.y for p in cell_samples)),
                    min(cell_max.z, max(p.z for p in cell_samples)),
                ))

                size   = tight_max - tight_min
                center = (tight_min + tight_max) * 0.5

                # Apply plane threshold
                if size.x < plane_threshold: size.x = 0.0
                if size.y < plane_threshold: size.y = 0.0
                if size.z < plane_threshold: size.z = 0.0

                orientation = Quaternion((1, 0, 0, 0))

                # Optional: find best-fitting oriented box via PCA
                if adjust_orientation:
                    orientation, size = pca_oriented_box(
                        cell_samples, size, plane_threshold
                    )

                blocks.append((center, size, orientation))

    print(f"  Kept {len(blocks)} non-empty blocks")
    blocks = merge_planar_blocks(blocks)
    blocks = merge_adjacent_and_contained_blocks(blocks, merge_threshold)
    return blocks


# ============================================================================
# Merge coplanar adjacent plane-blocks
# ============================================================================

def merge_planar_blocks(blocks, gap_tolerance=0.1):
    """
    Merge small coplanar adjacent plane blocks (one dim == 0) into fewer large ones.

    Algorithm:
    1. Separate planar blocks (one dim == 0) from volumetric ones.
    2. Group planar blocks by: which axis is flat + rounded quaternion + position on flat axis.
    3. Within each group, build adjacency (2-D projections touch or are within gap_tolerance).
    4. Union-Find to find connected components, then merge each component into one block
       (bounding rectangle of all members).
    """
    PLANE_EPS = 1e-4

    planar     = []  # (center, size, quat, flat_axis)
    volumetric = []  # (center, size, quat)

    for center, size, quat in blocks:
        zero = [size.x < PLANE_EPS, size.y < PLANE_EPS, size.z < PLANE_EPS]
        if any(zero):
            flat_axis = zero.index(True)
            planar.append((center, size, quat, flat_axis))
        else:
            volumetric.append((center, size, quat))

    if not planar:
        return blocks

    # --- Group ---
    groups = {}
    for center, size, quat, flat_axis in planar:
        qkey = (round(quat.w, 3), round(quat.x, 3),
                round(quat.y, 3), round(quat.z, 3))
        flat_pos = (center.x, center.y, center.z)[flat_axis]
        # Snap flat position to nearest gap_tolerance bucket
        flat_bucket = round(flat_pos / max(gap_tolerance, 1e-6))
        key = (flat_axis, qkey, flat_bucket)
        groups.setdefault(key, []).append((center, size, quat, flat_axis))

    # --- Merge each group ---
    merged_planar = []

    for key, group in groups.items():
        flat_axis = group[0][3]
        ax1, ax2  = [a for a in (0, 1, 2) if a != flat_axis]

        def get_rect(b):
            c = (b[0].x, b[0].y, b[0].z)
            s = (b[1].x, b[1].y, b[1].z)
            return (c[ax1] - s[ax1] * 0.5, c[ax1] + s[ax1] * 0.5,
                    c[ax2] - s[ax2] * 0.5, c[ax2] + s[ax2] * 0.5)

        rects = [get_rect(b) for b in group]
        n     = len(group)

        # Union-Find
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        tol = gap_tolerance
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = rects[i], rects[j]
                if (r1[0] <= r2[1] + tol and r2[0] <= r1[1] + tol and
                    r1[2] <= r2[3] + tol and r2[2] <= r1[3] + tol):
                    union(i, j)

        # Collect components and merge
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)

        for indices in comps.values():
            if len(indices) == 1:
                center, size, quat, _ = group[indices[0]]
                merged_planar.append((center, size, quat))
                continue

            u_min = min(rects[i][0] for i in indices)
            u_max = max(rects[i][1] for i in indices)
            v_min = min(rects[i][2] for i in indices)
            v_max = max(rects[i][3] for i in indices)

            flat_pos_avg = sum((group[i][0].x, group[i][0].y, group[i][0].z)[flat_axis]
                               for i in indices) / len(indices)

            new_center = Vector((0.0, 0.0, 0.0))
            new_center[flat_axis] = flat_pos_avg
            new_center[ax1]       = (u_min + u_max) / 2
            new_center[ax2]       = (v_min + v_max) / 2

            new_size = Vector((0.0, 0.0, 0.0))
            new_size[flat_axis] = 0.0
            new_size[ax1]       = u_max - u_min
            new_size[ax2]       = v_max - v_min

            merged_planar.append((new_center, new_size, group[indices[0]][2]))

    before = len(planar)
    after  = len(merged_planar)
    if before != after:
        print(f"  Plane merge: {before} planes -> {after}  (saved {before - after})")

    return volumetric + merged_planar


# ============================================================================
# PCA orientation fitting
# ============================================================================

def pca_oriented_box(cell_verts, size_aabb, plane_threshold, min_rotation_deg=5.0):
    """
    Run PCA on cell_verts to find a rotation that minimises the bounding box volume.
    Returns (quaternion, size).  Falls back to identity if PCA gives no gain.
    
    Args:
        cell_verts: list of Vector points
        size_aabb: Vector of AABB size
        plane_threshold: threshold for flattening dimensions to 0
        min_rotation_deg: minimum rotation angle in degrees required to apply PCA (default 5.0)
    """
    if len(cell_verts) < 3:
        return Quaternion((1, 0, 0, 0)), size_aabb

    pts = np.array([[v.x, v.y, v.z] for v in cell_verts])
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return Quaternion((1, 0, 0, 0)), size_aabb

    # Largest eigenvalue first
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Bounding box in oriented space
    oriented = centered @ eigenvectors
    obb_size_np = oriented.max(axis=0) - oriented.min(axis=0)
    obb_size = Vector(obb_size_np)

    # Apply plane threshold
    if obb_size.x < plane_threshold: obb_size.x = 0.0
    if obb_size.y < plane_threshold: obb_size.y = 0.0
    if obb_size.z < plane_threshold: obb_size.z = 0.0

    # Only use if meaningfully smaller AND rotation is significant enough
    vol_aabb = size_aabb.x * size_aabb.y * size_aabb.z
    vol_obb  = obb_size.x  * obb_size.y  * obb_size.z

    if vol_obb < vol_aabb * 0.95:
        rot_mat = Matrix.Identity(3)
        for i in range(3):
            rot_mat.col[i] = Vector(eigenvectors[:, i])
        pca_quat = rot_mat.to_quaternion()
        
        # Calculate rotation angle from identity
        # Angle = 2 * acos(|w|) where w is the w component
        angle_rad = 2.0 * math.acos(min(1.0, abs(pca_quat.w)))
        angle_deg = math.degrees(angle_rad)
        
        # Only apply if rotation is significant enough
        if angle_deg >= min_rotation_deg:
            return pca_quat, obb_size

    return Quaternion((1, 0, 0, 0)), size_aabb


# ============================================================================
# Coordinate space conversion  (Blender Z-up -> Hytale Y-up)
# ============================================================================

# Change-of-basis: Blender (X right, Y fwd, Z up) -> Hytale (X right, Y up, Z fwd)
#   Hytale X = Blender X
#   Hytale Y = Blender Z
#   Hytale Z = Blender Y
_COORD_CHANGE = Matrix(((1, 0, 0),
                        (0, 0, 1),
                        (0, 1, 0)))


def blender_quat_to_hytale(quat):
    """
    Properly converts a rotation from Blender Z-up to Hytale Y-up space
    by conjugating with the change-of-basis matrix.

    Wrong: Quaternion((w, x, z, y))  <- just swaps components, INCORRECT
    Right: C @ R_blender @ C         <- actual basis change (C == C^-1 here)
    """
    R_h = _COORD_CHANGE @ quat.to_matrix() @ _COORD_CHANGE
    return R_h.to_quaternion()


# ============================================================================
# Per-mesh export wrapper
# ============================================================================

def apply_mirroring(blocks, mirror_axis, merge_threshold=0.95):
    """
    Mirror blocks across the specified axis.
    Splits generation area in half on the given axis, then mirrors blocks.
    
    Args:
        blocks: list of (center, size, quaternion) tuples
        mirror_axis: 'X', 'Y', or 'Z' - which axis to mirror along
        merge_threshold: minimum volume overlap ratio to merge (default 0.95)
    
    Returns:
        Extended list with both original and mirrored blocks
    """
    if not blocks:
        return blocks
    
    # Map axis names to indices
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[mirror_axis]
    
    mirrored_blocks = []
    
    for center, size, quat in blocks:
        # Keep original block
        mirrored_blocks.append((center, size, quat))
        
        # Create mirrored block
        mirror_center = Vector(center)
        mirror_center[axis_idx] = -mirror_center[axis_idx]
        
        # Mirror the quaternion rotation across the axis
        # For a rotation matrix, mirroring means flipping the signs of rotations around the other axes
        quat_arr = [quat.w, quat.x, quat.y, quat.z]
        mirror_quat = Quaternion(quat_arr)
        
        # Flip the appropriate rotation components for the mirrored side
        # We need to adjust the quaternion for the mirror
        rot_matrix = quat.to_matrix()
        mirror_mat = Matrix.Identity(3)
        mirror_mat[axis_idx][axis_idx] = -1
        
        # Apply mirror: R_mirror = M @ R @ M (where M is the mirror matrix)
        mirrored_rot_matrix = mirror_mat @ rot_matrix @ mirror_mat
        mirror_quat = mirrored_rot_matrix.to_quaternion()
        
        mirrored_blocks.append((mirror_center, size, mirror_quat))
    
    print(f"  Mirrored along {mirror_axis}: {len(mirrored_blocks)} blocks before merge")
    
    # Merge blocks with complete face contacts and contained volumes
    merged = merge_adjacent_and_contained_blocks(mirrored_blocks, merge_threshold)
    return merged


def merge_adjacent_and_contained_blocks(blocks, volume_overlap_threshold=0.95):
    """
    Merge blocks in three ways:
    1. Blocks with complete face-to-face contact on any axis (same orientation required)
    2. Blocks whose volume is completely contained within another block
    3. Blocks that have significant volume overlap (>= threshold, default 95%)
    
    Args:
        blocks: list of (center, size, quaternion) tuples
        volume_overlap_threshold: minimum volume overlap ratio to merge (default 0.95)
    
    Returns:
        Deduplicated list with merged blocks
    """
    if not blocks:
        return blocks
    
    merged_blocks = []
    processed = set()
    
    def quaternions_equal(q1, q2, tol=0.1):
        """Check if two quaternions represent the same rotation (within tolerance)."""
        dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
        return dot > (1.0 - tol)
    
    def get_extents(center, size):
        """Get min and max for each axis."""
        return (
            (center.x - size.x / 2, center.x + size.x / 2),
            (center.y - size.y / 2, center.y + size.y / 2),
            (center.z - size.z / 2, center.z + size.z / 2),
        )
    
    def is_contained(ext_a, ext_b, tol=0.1):
        """Check if extent A is completely contained within extent B."""
        return all(ext_a[i][0] >= ext_b[i][0] - tol and 
                   ext_a[i][1] <= ext_b[i][1] + tol 
                   for i in range(3))
    
    def calculate_overlap_volume(ext_a, ext_b):
        """Calculate the volume of intersection between two extents."""
        overlap = 1.0
        for i in range(3):
            overlap_min = max(ext_a[i][0], ext_b[i][0])
            overlap_max = min(ext_a[i][1], ext_b[i][1])
            overlap_size = max(0.0, overlap_max - overlap_min)
            overlap *= overlap_size
        return overlap
    
    def has_significant_overlap(ext_a, ext_b, size_a, size_b, threshold):
        """Check if blocks have significant volume overlap."""
        vol_a = size_a.x * size_a.y * size_a.z
        vol_b = size_b.x * size_b.y * size_b.z
        
        if vol_a < 1e-6 or vol_b < 1e-6:
            return False
        
        overlap_vol = calculate_overlap_volume(ext_a, ext_b)
        
        # Check if overlap is >= threshold of either block's volume
        overlap_ratio_a = overlap_vol / vol_a
        overlap_ratio_b = overlap_vol / vol_b
        
        return overlap_ratio_a >= threshold or overlap_ratio_b >= threshold
    
    def faces_fully_touching(ext_a, ext_b, axis, tol=0.01):
        """Check if blocks touch completely on a given axis."""
        other_axes = [i for i in range(3) if i != axis]
        
        # Check if they touch on the given axis
        max_a = ext_a[axis][1]
        min_b = ext_b[axis][0]
        max_b = ext_b[axis][1]
        min_a = ext_a[axis][0]
        
        touching = False
        face_a = None  # Will be 'max' or 'min'
        face_b = None
        
        # Case 1: max of A touches min of B
        if abs(max_a - min_b) < 1e-6:
            touching = True
            face_a = 'max'
            face_b = 'min'
        # Case 2: max of B touches min of A
        elif abs(max_b - min_a) < 1e-6:
            touching = True
            face_a = 'min'
            face_b = 'max'
        
        if not touching:
            return False
        
        # Check if the faces fully overlap on the other two axes
        for oa in other_axes:
            # One block's extent must fully contain the other on this axis
            # OR they must match exactly
            a_min, a_max = ext_a[oa]
            b_min, b_max = ext_b[oa]
            
            a_size = a_max - a_min
            b_size = b_max - b_min
            
            # Check if either fully covers the other
            a_covers_b = (a_min <= b_min + tol and a_max >= b_max - tol)
            b_covers_a = (b_min <= a_min + tol and b_max >= a_max - tol)
            
            if not (a_covers_b or b_covers_a):
                return False
        
        return True
    
    for i, (center_a, size_a, quat_a) in enumerate(blocks):
        if i in processed:
            continue
        
        matched = False
        ext_a = get_extents(center_a, size_a)
        vol_a = size_a.x * size_a.y * size_a.z
        
        # Try to find another block to merge with
        for j in range(i + 1, len(blocks)):
            if j in processed:
                continue
            
            center_b, size_b, quat_b = blocks[j]
            ext_b = get_extents(center_b, size_b)
            vol_b = size_b.x * size_b.y * size_b.z
            
            # Check 1: Volume containment (A inside B or B inside A)
            if is_contained(ext_a, ext_b, tol=0.1):
                # A is inside B, keep B
                processed.add(i)
                matched = True
                break
            elif is_contained(ext_b, ext_a, tol=0.1):
                # B is inside A, keep A (skip B later)
                processed.add(j)
                continue
            
            # Check 2: Significant volume overlap (orientation-independent)
            if has_significant_overlap(ext_a, ext_b, size_a, size_b, volume_overlap_threshold):
                # Merge with mean position and mean rotation
                new_center = (center_a + center_b) * 0.5
                
                # Average the quaternions using SLERP (ensure shortest path)
                # If dot product is negative, they're on opposite sides - negate one
                dot = quat_a.x * quat_b.x + quat_a.y * quat_b.y + quat_a.z * quat_b.z + quat_a.w * quat_b.w
                if dot < 0:
                    quat_b_adj = Quaternion((-quat_b.w, -quat_b.x, -quat_b.y, -quat_b.z))
                else:
                    quat_b_adj = quat_b
                new_quat = quat_a.slerp(quat_b_adj, 0.5)
                
                # Compute merged size: transform both blocks into mean rotation frame
                # to find the AABB that covers both in the merged quaternion's space
                inv_rot = new_quat.inverted().to_matrix()
                
                # Get all 8 corners of both blocks in original frame, transform to mean frame
                def get_corners(center, size):
                    return [
                        Vector((center.x + dx * size.x / 2, center.y + dy * size.y / 2, center.z + dz * size.z / 2))
                        for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
                    ]
                
                corners_a = [inv_rot @ (c - new_center) for c in get_corners(center_a, size_a)]
                corners_b = [inv_rot @ (c - new_center) for c in get_corners(center_b, size_b)]
                
                all_corners = corners_a + corners_b
                
                # AABB in mean frame
                if all_corners:
                    min_corner = Vector((
                        min(c.x for c in all_corners),
                        min(c.y for c in all_corners),
                        min(c.z for c in all_corners),
                    ))
                    max_corner = Vector((
                        max(c.x for c in all_corners),
                        max(c.y for c in all_corners),
                        max(c.z for c in all_corners),
                    ))
                    # Get theoretical AABB proportions
                    theoretical_aabb = max_corner - min_corner
                    vol_theoretical = theoretical_aabb.x * theoretical_aabb.y * theoretical_aabb.z
                    
                    # Calculate combined volume
                    vol_a = size_a.x * size_a.y * size_a.z
                    vol_b = size_b.x * size_b.y * size_b.z
                    combined_volume = vol_a + vol_b
                    
                    # Scale theoretical AABB to match combined volume while preserving proportions
                    if vol_theoretical > 1e-6:
                        scale_factor = (combined_volume / vol_theoretical) ** (1.0 / 3.0)
                        new_size = theoretical_aabb * scale_factor
                    else:
                        new_size = theoretical_aabb
                else:
                    new_size = size_a
                
                merged_blocks.append((new_center, new_size, new_quat))
                processed.add(i)
                processed.add(j)
                matched = True
                break
            
            # Check 3: Complete face touching (requires same orientation)
            if not quaternions_equal(quat_a, quat_b):
                continue
            
            # Check if they have complete face contact on any axis
            merged = False
            for axis in range(3):
                if faces_fully_touching(ext_a, ext_b, axis):
                    # Merge them
                    new_center = Vector(center_a)
                    new_size = Vector(size_a)
                    
                    # Expand to cover both blocks
                    for k in range(3):
                        new_min = min(ext_a[k][0], ext_b[k][0])
                        new_max = max(ext_a[k][1], ext_b[k][1])
                        new_center[k] = (new_min + new_max) / 2
                        new_size[k] = new_max - new_min
                    
                    merged_blocks.append((new_center, new_size, quat_a))
                    processed.add(i)
                    processed.add(j)
                    matched = True
                    merged = True
                    break
            
            if merged:
                break
        
        if not matched:
            merged_blocks.append((center_a, size_a, quat_a))
            processed.add(i)
    
    if len(merged_blocks) < len(blocks):
        print(f"  Merged adjacent/contained blocks: {len(blocks)} -> {len(merged_blocks)}")
    
    return merged_blocks


def export_mesh(obj, node_id, max_blocks, adjust_orientation, plane_threshold, enable_mirror=False, mirror_axis='X', merge_threshold=0.95):
    """Build group node + child box nodes for one mesh object."""
    world_pos = obj.matrix_world.translation

    # Position: Blender Z-up -> Hytale Y-up
    hytale_pos = Vector((world_pos.x, world_pos.z, world_pos.y))
    group = make_group_node(node_id, obj.name, hytale_pos)

    blocks = subdivide_mesh_into_blocks(
        obj, max_blocks, adjust_orientation, plane_threshold, merge_threshold
    )

    # Apply mirroring if enabled
    if enable_mirror:
        blocks = apply_mirroring(blocks, mirror_axis, merge_threshold)
    
    # Final merge pass: catch any blocks that became adjacent after previous operations
    print(f"  Final merge pass...")
    blocks = merge_adjacent_and_contained_blocks(blocks, merge_threshold)

    for i, (center, size, quat) in enumerate(blocks):
        # Relative position
        rel = center - world_pos
        rel_h = Vector((rel.x, rel.z, rel.y))

        # Size: swap Y and Z axes
        size_h = Vector((size.x, size.z, size.y))

        # Rotation: proper coordinate-space conversion
        quat_h = blender_quat_to_hytale(quat)

        box = make_box_node(
            node_id=f"{node_id}_{i}",
            name=f"{obj.name}_block_{i}",
            position=rel_h,
            size=size_h,
            orientation=quat_h,
        )
        group["children"].append(box)

    return group


# ============================================================================
# Blender Operator
# ============================================================================

class ExportBlockyModel(Operator, ExportHelper):
    """Export meshes to Hytale Blocky Model format"""
    bl_idname  = "export_scene.blockymodel"
    bl_label   = "Export Blocky Model"
    bl_options = {'PRESET'}

    filename_ext = ".blockymodel"
    filter_glob: StringProperty(default="*.blockymodel", options={'HIDDEN'})

    max_blocks: IntProperty(
        name="Max Blocks per Mesh",
        description="Maximum number of blocks generated per mesh object",
        default=8,
        min=1,
        max=1024,
    )

    plane_threshold: FloatProperty(
        name="Plane Threshold",
        description="Block dimension smaller than this becomes 0 (flat plane)",
        default=0.5,
        min=0.0,
        max=5.0,
        precision=2,
    )

    adjust_orientation: BoolProperty(
        name="Adjust Orientation",
        description=(
            "Use PCA to rotate each block for a tighter fit to local geometry. "
            "Blocks may overlap but external surface is preserved."
        ),
        default=False,
    )

    export_selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected mesh objects",
        default=False,
    )

    enable_mirror: BoolProperty(
        name="Enable Mirroring",
        description="Split mesh in half and mirror blocks to double coverage",
        default=False,
    )

    mirror_axis: bpy.props.EnumProperty(
        name="Mirror Axis",
        description="Axis to split and mirror along",
        items=[
            ('X', "X Axis", "Mirror along X axis"),
            ('Y', "Y Axis", "Mirror along Y axis"),
            ('Z', "Z Axis", "Mirror along Z axis"),
        ],
        default='X',
    )

    merge_threshold: FloatProperty(
        name="Merge Threshold",
        description="Minimum volume overlap ratio to merge blocks (0.0-1.0)",
        default=0.95,
        min=0.0,
        max=1.0,
        precision=2,
    )

    def execute(self, context):
        t0 = time.time()

        if self.export_selected_only:
            objects = [o for o in context.selected_objects if o.type == 'MESH']
        else:
            objects = [o for o in context.scene.objects   if o.type == 'MESH']

        if not objects:
            self.report({'ERROR'}, "No mesh objects to export")
            return {'CANCELLED'}

        print("\n" + "=" * 60)
        print("HYTALE BLOCKY MODEL EXPORT v18")
        print(f"  Objects       : {len(objects)}")
        print(f"  Max blocks    : {self.max_blocks}")
        print(f"  Plane thresh  : {self.plane_threshold}")
        print(f"  Adjust orient : {self.adjust_orientation}")
        print(f"  Merge thresh  : {self.merge_threshold * 100:.0f}%")
        if self.enable_mirror:
            print(f"  Mirror axis   : {self.mirror_axis}")
        print("=" * 60)

        top_nodes = []
        for idx, obj in enumerate(objects):
            print(f"\n[{idx+1}/{len(objects)}] {obj.name}")
            group = export_mesh(
                obj,
                node_id=idx + 1,
                max_blocks=self.max_blocks,
                adjust_orientation=self.adjust_orientation,
                plane_threshold=self.plane_threshold,
                enable_mirror=self.enable_mirror,
                mirror_axis=self.mirror_axis,
                merge_threshold=self.merge_threshold,
            )
            top_nodes.append(group)

        output = {"nodes": top_nodes, "format": "prop", "lod": "auto"}

        filepath = bpy.path.abspath(self.filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        elapsed      = time.time() - t0
        total_blocks = sum(len(g["children"]) for g in top_nodes)
        print(f"\n✓ {total_blocks} blocks in {elapsed:.1f}s  ->  {filepath}")
        self.report({'INFO'}, f"Exported {total_blocks} blocks")
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "export_selected_only")
        layout.separator()

        box = layout.box()
        box.label(text="Block Settings:", icon='MESH_CUBE')
        box.prop(self, "max_blocks")
        box.prop(self, "plane_threshold")

        layout.separator()

        box2 = layout.box()
        box2.label(text="Orientation:", icon='ORIENTATION_GLOBAL')
        box2.prop(self, "adjust_orientation")
        if self.adjust_orientation:
            box2.label(text="PCA fit — may be slower on dense meshes", icon='INFO')

        layout.separator()

        box3 = layout.box()
        box3.label(text="Mirroring:", icon='MOD_MIRROR')
        box3.prop(self, "enable_mirror")
        if self.enable_mirror:
            box3.prop(self, "mirror_axis")
            box3.label(text="Splits mesh in half, generates half blocks", icon='INFO')

        layout.separator()

        box4 = layout.box()
        box4.label(text="Merging:", icon='AUTOMERGE_ON')
        box4.prop(self, "merge_threshold")
        box4.label(text="Blocks with this much volume overlap will merge", icon='INFO')

        layout.separator()
        tip = layout.box()
        tip.label(text="💡 Tips:", icon='SETTINGS')
        tip.label(text="• Max blocks: start low (4-16), increase for detail")
        tip.label(text="• Plane threshold 0 = never flatten to plane")
        tip.label(text="• Adjust orientation: best for curved/diagonal surfaces")
        tip.label(text="• Mirroring: useful for symmetric models to save blocks")


# ============================================================================
# Registration
# ============================================================================

def menu_func_export(self, context):
    self.layout.operator(ExportBlockyModel.bl_idname,
                         text="Hytale Blocky Model (.blockymodel)")


def register():
    bpy.utils.register_class(ExportBlockyModel)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportBlockyModel)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()