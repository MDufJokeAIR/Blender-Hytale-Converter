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
    "version": (18, 10, 1),  # Fixed projection bounds, better placement order (origin-first)
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
import os

from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
from mathutils import Vector, Matrix, Quaternion

# ============================================================================
# Proportional Texture System
# ============================================================================

try:
    from PIL import Image, ImageDraw
    TEXTURE_AVAILABLE = True
except ImportError:
    TEXTURE_AVAILABLE = False

def calculate_blockymodel_bounds(blocks):
    """Calculate total bounds of all blocks."""
    if not blocks:
        return (0, 0, 0, 0, 0, 0)
    
    first_center, first_size, _ = blocks[0]
    min_x = first_center.x - first_size.x / 2
    max_x = first_center.x + first_size.x / 2
    min_y = first_center.y - first_size.y / 2
    max_y = first_center.y + first_size.y / 2
    min_z = first_center.z - first_size.z / 2
    max_z = first_center.z + first_size.z / 2
    
    for center, size, _ in blocks[1:]:
        min_x = min(min_x, center.x - size.x / 2)
        max_x = max(max_x, center.x + size.x / 2)
        min_y = min(min_y, center.y - size.y / 2)
        max_y = max(max_y, center.y + size.y / 2)
        min_z = min(min_z, center.z - size.z / 2)
        max_z = max(max_z, center.z + size.z / 2)
    
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def create_compact_projections(blocks):
    """
    Create 6 compact orthographic projections (one per axis).
    Each projection is a tight 2D view with scale=1 (1 pixel = 1 unit).
    """
    if not blocks:
        return {}, (0,0,0,0,0,0), 10, 10
    
    scale = 1  # 1 pixel = 1 unit
    bounds = calculate_blockymodel_bounds(blocks)
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    
    # Calculate size of each projection
    projections = {
        'nord': {
            'width': int((max_x - min_x) * scale),
            'height': int((max_z - min_z) * scale),
            'offset_x': 0,
        },
        'sud': {
            'width': int((max_x - min_x) * scale),
            'height': int((max_z - min_z) * scale),
            'offset_x': 0,
        },
        'est': {
            'width': int((max_y - min_y) * scale),
            'height': int((max_z - min_z) * scale),
            'offset_x': 0,
        },
        'ouest': {
            'width': int((max_y - min_y) * scale),
            'height': int((max_z - min_z) * scale),
            'offset_x': 0,
        },
        'haut': {
            'width': int((max_x - min_x) * scale),
            'height': int((max_y - min_y) * scale),
            'offset_x': 0,
        },
        'bas': {
            'width': int((max_x - min_x) * scale),
            'height': int((max_y - min_y) * scale),
            'offset_x': 0,
        },
    }
    
    # Layout projections horizontally
    current_x = 0
    max_height = 0
    
    for proj_name in ['nord', 'sud', 'est', 'ouest', 'haut', 'bas']:
        projections[proj_name]['offset_x'] = current_x
        current_x += projections[proj_name]['width']
        max_height = max(max_height, projections[proj_name]['height'])
    
    return projections, bounds, current_x, max_height

def calculate_texture_size(blocks):
    """
    Calculate texture size using compact projections.
    Now returns projections instead of sections.
    """
    projections, bounds, total_width, max_height = create_compact_projections(blocks)
    return total_width, max_height, projections, bounds


def calculate_mesh_projections(blocks):
    """
    Create 6 orthogonal projections stacked vertically with generous spacing.
    Each projection gets 2x the mesh size for local overflow handling.
    
    Layout (vertical):
    [top]
    [front]
    [back]
    [left]
    [right]
    [bottom]
    """
    if not blocks:
        return (10, 10, [], {}, (0, 0, 0, 0, 0, 0))
    
    # Calculate bounds of ALL blocks together
    min_x, max_x, min_y, max_y, min_z, max_z = calculate_blockymodel_bounds(blocks)
    
    # Dimensions for each projection (1 pixel = 1 unit)
    scale = 1
    
    # Base dimensions with padding
    size_x = int((max_x - min_x) * scale) + 2
    size_y = int((max_y - min_y) * scale) + 2
    size_z = int((max_z - min_z) * scale) + 2
    
    # Multiply by 2 for local overflow space
    size_x_2x = size_x * 2
    size_y_2x = size_y * 2
    size_z_2x = size_z * 2
    
    # Vertical stacking
    current_y = 0
    
    projections = {
        'top': {
            'width': size_x_2x,
            'height': size_z_2x,
            'offset_x': 0,
            'offset_y': current_y,
        },
    }
    current_y += size_z_2x
    
    projections['front'] = {
        'width': size_x_2x,
        'height': size_y_2x,
        'offset_x': 0,
        'offset_y': current_y,
    }
    current_y += size_y_2x
    
    projections['back'] = {
        'width': size_x_2x,
        'height': size_y_2x,
        'offset_x': 0,
        'offset_y': current_y,
    }
    current_y += size_y_2x
    
    projections['left'] = {
        'width': size_z_2x,
        'height': size_y_2x,
        'offset_x': 0,
        'offset_y': current_y,
    }
    current_y += size_y_2x
    
    projections['right'] = {
        'width': size_z_2x,
        'height': size_y_2x,
        'offset_x': 0,
        'offset_y': current_y,
    }
    current_y += size_y_2x
    
    projections['bottom'] = {
        'width': size_x_2x,
        'height': size_z_2x,
        'offset_x': 0,
        'offset_y': current_y,
    }
    current_y += size_z_2x
    
    # Total texture size
    total_width = max(size_x_2x, size_z_2x)  # Widest projection
    total_height = current_y
    
    bounds = (min_x, max_x, min_y, max_y, min_z, max_z)
    
    print(f"  Mesh bounds: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}] Z[{min_z:.1f}, {max_z:.1f}]")
    print(f"  Base sizes: X={size_x}, Y={size_y}, Z={size_z}")
    print(f"  Layout: Vertical stack, {total_width}×{total_height}px (2x space per zone)")
    
    return total_width, total_height, projections, bounds, scale


def detect_uv_overlap(face1, face2, spacing=1):
    """Check if two UV rectangles overlap (with small spacing padding)."""
    u1, v1, w1, h1 = face1
    u2, v2, w2, h2 = face2
    
    # Add minimal spacing padding
    u1_padded = u1 - spacing
    v1_padded = v1 - spacing
    w1_padded = w1 + spacing * 2
    h1_padded = h1 + spacing * 2
    
    u2_padded = u2 - spacing
    v2_padded = v2 - spacing
    w2_padded = w2 + spacing * 2
    h2_padded = h2 + spacing * 2
    
    # No overlap if one is completely to the side/above/below the other
    return not (
        u1_padded + w1_padded <= u2_padded or  # face1 left of face2
        u1_padded >= u2_padded + w2_padded or  # face1 right of face2
        v1_padded + h1_padded <= v2_padded or  # face1 above face2
        v1_padded >= v2_padded + h2_padded     # face1 below face2
    )


def calculate_face_distance_from_origin(block_center, face_normal):
    """
    Calculate distance of a face from origin.
    Faces further from origin are more visible.
    """
    # Distance of block center from origin
    distance = math.sqrt(block_center.x**2 + block_center.y**2 + block_center.z**2)
    
    # Add component in the direction of the face normal
    # This prioritizes faces pointing away from origin
    if face_normal == 'front':  # +Z
        distance += block_center.z
    elif face_normal == 'back':  # -Z
        distance -= block_center.z
    elif face_normal == 'right':  # +X
        distance += block_center.x
    elif face_normal == 'left':  # -X
        distance -= block_center.x
    elif face_normal == 'top':  # +Y
        distance += block_center.y
    elif face_normal == 'bottom':  # -Y
        distance -= block_center.y
    
    return distance


def calculate_projection_based_uvs(blocks, enable_mirror=False, mirror_axis='X'):
    """
    Orthogonal projection-based UV placement:
    - Each direction gets an orthogonal 2D projection of the mesh
    - Faces placed at their actual projected 2D positions
    - Blocks sorted by distance from origin for better placement order
    - Collision detection ensures no overlap
    - Maintains spatial relationships between blocks
    """
    if not blocks:
        return (10, 10, [], {})
    
    from mathutils import Vector
    import math
    
    min_x, max_x, min_y, max_y, min_z, max_z = calculate_blockymodel_bounds(blocks)
    scale = 1
    
    # Color mapping per face type
    face_colors = {
        'front': (255, 100, 100, 255),   # Red
        'back': (100, 255, 100, 255),    # Green
        'right': (100, 100, 255, 255),   # Blue
        'left': (255, 255, 100, 255),    # Yellow
        'top': (255, 100, 255, 255),     # Magenta
        'bottom': (100, 255, 255, 255),  # Cyan
    }
    
    # Global direction vectors
    global_directions = {
        'front': Vector((0, 0, 1)),
        'back': Vector((0, 0, -1)),
        'right': Vector((1, 0, 0)),
        'left': Vector((-1, 0, 0)),
        'top': Vector((0, 1, 0)),
        'bottom': Vector((0, -1, 0)),
    }
    
    local_face_normals = {
        'front': Vector((0, 0, 1)),
        'back': Vector((0, 0, -1)),
        'right': Vector((1, 0, 0)),
        'left': Vector((-1, 0, 0)),
        'top': Vector((0, 1, 0)),
        'bottom': Vector((0, -1, 0)),
    }
    
    def find_closest_global_direction(rotated_normal):
        best_match = 'front'
        best_dot = -2
        for direction, global_vec in global_directions.items():
            dot = rotated_normal.dot(global_vec)
            if dot > best_dot:
                best_dot = dot
                best_match = direction
        return best_match
    
    # Sort blocks by distance from origin (closest first)
    # This gives better spatial organization
    blocks_with_distance = []
    for idx, (center, size, quat) in enumerate(blocks):
        distance = math.sqrt(center.x**2 + center.y**2 + center.z**2)
        blocks_with_distance.append((idx, center, size, quat, distance))
    
    blocks_with_distance.sort(key=lambda x: x[4])  # Sort by distance
    
    # Collect faces with their 3D positions and directions
    faces_by_direction = {
        'front': [], 'back': [], 'left': [], 'right': [], 'top': [], 'bottom': []
    }
    
    num_reoriented = 0
    
    for original_idx, block_center, block_size, quat, distance in blocks_with_distance:
        is_rotated = not (abs(quat.w - 1.0) < 0.001 and 
                         abs(quat.x) < 0.001 and 
                         abs(quat.y) < 0.001 and 
                         abs(quat.z) < 0.001)
        
        face_data = {
            'front': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'back': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'right': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'left': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'top': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
            'bottom': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
        }
        
        face_direction_mapping = {}
        
        if is_rotated:
            rotation_matrix = quat.to_matrix()
            for local_face, local_normal in local_face_normals.items():
                rotated_normal = rotation_matrix @ local_normal
                global_dir = find_closest_global_direction(rotated_normal)
                face_direction_mapping[local_face] = global_dir
                if local_face != global_dir:
                    num_reoriented += 1
        else:
            for face in face_data.keys():
                face_direction_mapping[face] = face
        
        # Add faces with their 3D block center position
        for local_face, data in face_data.items():
            target_direction = face_direction_mapping[local_face]
            
            faces_by_direction[target_direction].append({
                'block_idx': original_idx,  # Use original index for JSON
                'local_face': local_face,
                'position_3d': block_center.copy(),
                'size_3d': block_size.copy(),
                'width': data['width'],
                'height': data['height'],
                'color': face_colors[target_direction],
                'distance': distance  # For sorting
            })
    
    if num_reoriented > 0:
        print(f"  Reoriented {num_reoriented} faces based on rotation")
    
    # Class to track occupied pixels
    class OccupancyGrid:
        def __init__(self):
            self.occupied = set()
        
        def is_available(self, x, y, width, height):
            for py in range(int(y), int(y + height)):
                for px in range(int(x), int(x + width)):
                    if (px, py) in self.occupied:
                        return False
            return True
        
        def mark_occupied(self, x, y, width, height):
            for py in range(int(y), int(y + height)):
                for px in range(int(x), int(x + width)):
                    self.occupied.add((px, py))
        
        def get_bounds(self):
            if not self.occupied:
                return 0, 0, 0, 0
            xs = [p[0] for p in self.occupied]
            ys = [p[1] for p in self.occupied]
            return min(xs), max(xs), min(ys), max(ys)
    
    # Create projections for each direction
    projections = {}
    current_y = 0
    texture_width = 0
    
    all_block_uvs = [{} for _ in blocks]
    
    for direction in ['top', 'front', 'back', 'left', 'right', 'bottom']:
        faces = faces_by_direction[direction]
        if not faces:
            projections[direction] = {'offset_y': current_y, 'height': 0, 'width': 0}
            continue
        
        # Sort faces by distance from origin (closest first)
        faces.sort(key=lambda f: f['distance'])
        
        print(f"  {direction}: Projecting {len(faces)} faces...")
        
        # Calculate 2D projection coordinates
        face_projections = []
        
        for face in faces:
            pos = face['position_3d']
            
            # Project to 2D based on direction
            if direction in ['top', 'bottom']:
                proj_x = (pos.x - min_x) * scale
                proj_y = (pos.z - min_z) * scale
            elif direction in ['front', 'back']:
                proj_x = (pos.x - min_x) * scale
                proj_y = (pos.y - min_y) * scale
            elif direction in ['left', 'right']:
                proj_x = (pos.z - min_z) * scale
                proj_y = (pos.y - min_y) * scale
            
            face_projections.append({
                'face': face,
                'proj_x': int(proj_x),
                'proj_y': int(proj_y)
            })
        
        # First pass: place faces and track actual bounds
        padding = 10
        occupancy = OccupancyGrid()
        temp_placements = []  # Store temporary placements
        placed = 0
        collisions = 0
        
        for fp in face_projections:
            face = fp['face']
            target_x = fp['proj_x']
            target_y = fp['proj_y']
            w = face['width']
            h = face['height']
            
            # Try to place at projected position (in local projection space)
            placed_x, placed_y = target_x, target_y
            
            if not occupancy.is_available(target_x, target_y, w, h):
                # Collision! Try offsets: right, left, down, up
                collisions += 1
                found = False
                
                for offset_dist in range(1, 30):
                    # Try right
                    for dx in range(0, offset_dist + 1):
                        test_x = target_x + dx
                        test_y = target_y
                        if occupancy.is_available(test_x, test_y, w, h):
                            placed_x, placed_y = test_x, test_y
                            found = True
                            break
                    if found:
                        break
                    
                    # Try left
                    for dx in range(1, offset_dist + 1):
                        test_x = target_x - dx
                        test_y = target_y
                        if occupancy.is_available(test_x, test_y, w, h):
                            placed_x, placed_y = test_x, test_y
                            found = True
                            break
                    if found:
                        break
                    
                    # Try down
                    for dy in range(1, offset_dist + 1):
                        test_x = target_x
                        test_y = target_y + dy
                        if occupancy.is_available(test_x, test_y, w, h):
                            placed_x, placed_y = test_x, test_y
                            found = True
                            break
                    if found:
                        break
                    
                    # Try up
                    for dy in range(1, offset_dist + 1):
                        test_x = target_x
                        test_y = target_y - dy
                        if occupancy.is_available(test_x, test_y, w, h):
                            placed_x, placed_y = test_x, test_y
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    placed_x, placed_y = target_x, target_y
            
            # Mark as occupied (in local space)
            occupancy.mark_occupied(placed_x, placed_y, w, h)
            
            # Store temporary placement
            temp_placements.append({
                'block_idx': face['block_idx'],
                'local_face': face['local_face'],
                'local_x': placed_x,
                'local_y': placed_y,
                'width': w,
                'height': h,
                'color': face['color']
            })
            placed += 1
        
        # Calculate actual bounds used in local space
        min_x_used, max_x_used, min_y_used, max_y_used = occupancy.get_bounds()
        
        if min_x_used < max_x_used:
            # Normalize all placements to start at (padding, padding)
            offset_x = -min_x_used + padding
            offset_y = -min_y_used + padding
            
            proj_width = (max_x_used - min_x_used) + padding * 2
            proj_height = (max_y_used - min_y_used) + padding * 2
        else:
            offset_x = padding
            offset_y = padding
            proj_width = 20
            proj_height = 20
        
        # Second pass: apply normalization and store in global coordinates
        for placement in temp_placements:
            block_idx = placement['block_idx']
            local_face = placement['local_face']
            
            # Normalize to projection space with padding
            final_u = placement['local_x'] + offset_x
            final_v = placement['local_y'] + offset_y
            
            # Add global offset for this projection
            global_u = final_u
            global_v = current_y + final_v
            
            all_block_uvs[block_idx][local_face] = {
                'u': global_u,
                'v': global_v,
                'width': placement['width'],
                'height': placement['height'],
                'color': placement['color'],
                'mirror': {'x': False, 'y': False}
            }
        
        print(f"    Placed {placed} faces, {collisions} collisions resolved")
        print(f"    Zone size: {proj_width}×{proj_height}px")
        
        projections[direction] = {
            'offset_y': current_y,
            'width': proj_width,
            'height': proj_height
        }
        
        texture_width = max(texture_width, proj_width)
        current_y += proj_height
    
    total_height = current_y
    
    print(f"  Projection layout: {texture_width}×{total_height}px")
    print(f"  Total faces: {sum(len(faces_by_direction[d]) for d in faces_by_direction)}")
    
    return texture_width, total_height, all_block_uvs, projections
    """
    Orthogonal projection-based UV placement:
    - Each direction gets an orthogonal 2D projection of the mesh
    - Faces placed at their actual projected 2D positions
    - Collision detection ensures no overlap
    - Maintains spatial relationships between blocks
    """
    if not blocks:
        return (10, 10, [], {})
    
    from mathutils import Vector
    import math
    
    min_x, max_x, min_y, max_y, min_z, max_z = calculate_blockymodel_bounds(blocks)
    scale = 1
    
    # Color mapping per face type
    face_colors = {
        'front': (255, 100, 100, 255),   # Red
        'back': (100, 255, 100, 255),    # Green
        'right': (100, 100, 255, 255),   # Blue
        'left': (255, 255, 100, 255),    # Yellow
        'top': (255, 100, 255, 255),     # Magenta
        'bottom': (100, 255, 255, 255),  # Cyan
    }
    
    # Global direction vectors
    global_directions = {
        'front': Vector((0, 0, 1)),
        'back': Vector((0, 0, -1)),
        'right': Vector((1, 0, 0)),
        'left': Vector((-1, 0, 0)),
        'top': Vector((0, 1, 0)),
        'bottom': Vector((0, -1, 0)),
    }
    
    local_face_normals = {
        'front': Vector((0, 0, 1)),
        'back': Vector((0, 0, -1)),
        'right': Vector((1, 0, 0)),
        'left': Vector((-1, 0, 0)),
        'top': Vector((0, 1, 0)),
        'bottom': Vector((0, -1, 0)),
    }
    
    def find_closest_global_direction(rotated_normal):
        best_match = 'front'
        best_dot = -2
        for direction, global_vec in global_directions.items():
            dot = rotated_normal.dot(global_vec)
            if dot > best_dot:
                best_dot = dot
                best_match = direction
        return best_match
    
    # Collect faces with their 3D positions and directions
    faces_by_direction = {
        'front': [], 'back': [], 'left': [], 'right': [], 'top': [], 'bottom': []
    }
    
    num_reoriented = 0
    
    for block_idx, (block_center, block_size, quat) in enumerate(blocks):
        is_rotated = not (abs(quat.w - 1.0) < 0.001 and 
                         abs(quat.x) < 0.001 and 
                         abs(quat.y) < 0.001 and 
                         abs(quat.z) < 0.001)
        
        face_data = {
            'front': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'back': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'right': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'left': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'top': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
            'bottom': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
        }
        
        face_direction_mapping = {}
        
        if is_rotated:
            rotation_matrix = quat.to_matrix()
            for local_face, local_normal in local_face_normals.items():
                rotated_normal = rotation_matrix @ local_normal
                global_dir = find_closest_global_direction(rotated_normal)
                face_direction_mapping[local_face] = global_dir
                if local_face != global_dir:
                    num_reoriented += 1
        else:
            for face in face_data.keys():
                face_direction_mapping[face] = face
        
        # Add faces with their 3D block center position
        for local_face, data in face_data.items():
            target_direction = face_direction_mapping[local_face]
            
            faces_by_direction[target_direction].append({
                'block_idx': block_idx,
                'local_face': local_face,
                'position_3d': block_center.copy(),  # 3D position of block
                'size_3d': block_size.copy(),
                'width': data['width'],
                'height': data['height'],
                'color': face_colors[target_direction]
            })
    
    if num_reoriented > 0:
        print(f"  Reoriented {num_reoriented} faces based on rotation")
    
    # Class to track occupied pixels
    class OccupancyGrid:
        def __init__(self):
            self.occupied = set()  # Set of (x, y) tuples
        
        def is_available(self, x, y, width, height):
            """Check if rectangle is free."""
            for py in range(int(y), int(y + height)):
                for px in range(int(x), int(x + width)):
                    if (px, py) in self.occupied:
                        return False
            return True
        
        def mark_occupied(self, x, y, width, height):
            """Mark rectangle as occupied."""
            for py in range(int(y), int(y + height)):
                for px in range(int(x), int(x + width)):
                    self.occupied.add((px, py))
    
    # Create projections for each direction
    projections = {}
    current_y = 0
    texture_width = 0
    
    all_block_uvs = [{} for _ in blocks]
    
    for direction in ['top', 'front', 'back', 'left', 'right', 'bottom']:
        faces = faces_by_direction[direction]
        if not faces:
            projections[direction] = {'offset_y': current_y, 'height': 0}
            continue
        
        print(f"  {direction}: Projecting {len(faces)} faces...")
        
        # Calculate 2D projection coordinates for this direction
        # Top/Bottom: project onto XZ plane
        # Front/Back: project onto XY plane  
        # Left/Right: project onto ZY plane
        
        face_projections = []
        
        for face in faces:
            pos = face['position_3d']
            size = face['size_3d']
            
            # Project to 2D based on direction
            if direction in ['top', 'bottom']:
                # XZ plane projection
                proj_x = (pos.x - min_x) * scale
                proj_y = (pos.z - min_z) * scale
            elif direction in ['front', 'back']:
                # XY plane projection
                proj_x = (pos.x - min_x) * scale
                proj_y = (pos.y - min_y) * scale
            elif direction in ['left', 'right']:
                # ZY plane projection
                proj_x = (pos.z - min_z) * scale
                proj_y = (pos.y - min_y) * scale
            
            face_projections.append({
                'face': face,
                'proj_x': int(proj_x),
                'proj_y': int(proj_y)
            })
        
        # Calculate projection bounds
        min_proj_x = min(fp['proj_x'] for fp in face_projections)
        max_proj_x = max(fp['proj_x'] + fp['face']['width'] for fp in face_projections)
        min_proj_y = min(fp['proj_y'] for fp in face_projections)
        max_proj_y = max(fp['proj_y'] + fp['face']['height'] for fp in face_projections)
        
        proj_width = max_proj_x - min_proj_x + 4  # +4 for padding
        proj_height = max_proj_y - min_proj_y + 4
        
        # Normalize projections to start at (2, 2) with padding
        for fp in face_projections:
            fp['proj_x'] = fp['proj_x'] - min_proj_x + 2
            fp['proj_y'] = fp['proj_y'] - min_proj_y + 2
        
        # Place faces with collision detection
        occupancy = OccupancyGrid()
        placed = 0
        collisions = 0
        
        for fp in face_projections:
            face = fp['face']
            target_x = fp['proj_x']
            target_y = current_y + fp['proj_y']
            w = face['width']
            h = face['height']
            
            # Try to place at projected position
            placed_x, placed_y = target_x, target_y
            
            if not occupancy.is_available(target_x, fp['proj_y'], w, h):
                # Collision! Try offsets
                collisions += 1
                found = False
                
                # Try small offsets in a spiral pattern
                for offset_dist in range(1, 20):
                    for dx in range(-offset_dist, offset_dist + 1):
                        for dy in range(-offset_dist, offset_dist + 1):
                            test_x = target_x + dx
                            test_y = fp['proj_y'] + dy
                            
                            if test_x >= 0 and test_y >= 0:
                                if occupancy.is_available(test_x, test_y, w, h):
                                    placed_x = test_x
                                    placed_y = current_y + test_y
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    # Couldn't find spot, place anyway and hope for the best
                    placed_x = target_x
                    placed_y = target_y
            
            # Mark as occupied (in local projection space)
            occupancy.mark_occupied(placed_x if placed_x >= target_x else target_x, 
                                   placed_y - current_y if placed_y >= target_y else fp['proj_y'], 
                                   w, h)
            
            # Store UV coordinates
            block_idx = face['block_idx']
            local_face = face['local_face']
            
            all_block_uvs[block_idx][local_face] = {
                'u': placed_x,
                'v': placed_y,
                'width': w,
                'height': h,
                'color': face['color'],
                'mirror': {'x': False, 'y': False}
            }
            placed += 1
        
        print(f"    Placed {placed} faces, {collisions} collisions resolved")
        
        projections[direction] = {
            'offset_y': current_y,
            'width': proj_width,
            'height': proj_height
        }
        
        texture_width = max(texture_width, proj_width)
        current_y += proj_height
    
    total_height = current_y
    
    print(f"  Projection layout: {texture_width}×{total_height}px")
    print(f"  Total faces: {sum(len(faces_by_direction[d]) for d in faces_by_direction)}")
    
    return texture_width, total_height, all_block_uvs, projections
    """
    Simple grid-based UV placement:
    - One grid per face direction (front, back, left, right, top, bottom)
    - Grid cells = size of largest face in that direction + 2px padding
    - No overlap possible - each face gets its own cell
    - Faces placed with 1px offset inside cells
    - For rotated blocks: faces assigned to grids based on actual world direction
    """
    if not blocks:
        return (10, 10, [], {})
    
    min_x, max_x, min_y, max_y, min_z, max_z = calculate_blockymodel_bounds(blocks)
    scale = 1
    
    # Color mapping per face type
    face_colors = {
        'front': (255, 100, 100, 255),   # Red
        'back': (100, 255, 100, 255),    # Green
        'right': (100, 100, 255, 255),   # Blue
        'left': (255, 255, 100, 255),    # Yellow
        'top': (255, 100, 255, 255),     # Magenta
        'bottom': (100, 255, 255, 255),  # Cyan
    }
    
    # Global direction vectors
    global_directions = {
        'front': Vector((0, 0, 1)),   # +Z
        'back': Vector((0, 0, -1)),   # -Z
        'right': Vector((1, 0, 0)),   # +X
        'left': Vector((-1, 0, 0)),   # -X
        'top': Vector((0, 1, 0)),     # +Y
        'bottom': Vector((0, -1, 0)), # -Y
    }
    
    # Local face normals (before rotation)
    local_face_normals = {
        'front': Vector((0, 0, 1)),
        'back': Vector((0, 0, -1)),
        'right': Vector((1, 0, 0)),
        'left': Vector((-1, 0, 0)),
        'top': Vector((0, 1, 0)),
        'bottom': Vector((0, -1, 0)),
    }
    
    def find_closest_global_direction(rotated_normal):
        """Find which global direction is closest to the rotated normal."""
        best_match = 'front'
        best_dot = -2
        
        for direction, global_vec in global_directions.items():
            dot = rotated_normal.dot(global_vec)
            if dot > best_dot:
                best_dot = dot
                best_match = direction
        
        return best_match
    
    # MIRRORING DISABLED FOR NOW - just collect all faces
    
    # Collect all faces by direction (considering rotation)
    faces_by_direction = {
        'front': [], 'back': [], 'left': [], 'right': [], 'top': [], 'bottom': []
    }
    
    num_reoriented = 0
    
    for block_idx, (block_center, block_size, quat) in enumerate(blocks):
        # Check if block is rotated (quaternion not identity)
        is_rotated = not (abs(quat.w - 1.0) < 0.001 and 
                         abs(quat.x) < 0.001 and 
                         abs(quat.y) < 0.001 and 
                         abs(quat.z) < 0.001)
        
        face_data = {
            'front': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'back': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.y * scale))},
            'right': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'left': {'width': max(1, int(block_size.z * scale)), 'height': max(1, int(block_size.y * scale))},
            'top': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
            'bottom': {'width': max(1, int(block_size.x * scale)), 'height': max(1, int(block_size.z * scale))},
        }
        
        # Map local face names to actual global directions
        face_direction_mapping = {}
        
        if is_rotated:
            # Rotate each local face normal to find its actual global direction
            rotation_matrix = quat.to_matrix()
            
            for local_face, local_normal in local_face_normals.items():
                # Apply rotation to local normal
                rotated_normal = rotation_matrix @ local_normal
                # Find closest global direction
                global_dir = find_closest_global_direction(rotated_normal)
                face_direction_mapping[local_face] = global_dir
                
                if local_face != global_dir:
                    num_reoriented += 1
        else:
            # No rotation: local face = global direction
            for face in face_data.keys():
                face_direction_mapping[face] = face
        
        # Add faces to the appropriate direction grids
        for local_face, data in face_data.items():
            target_direction = face_direction_mapping[local_face]
            
            faces_by_direction[target_direction].append({
                'block_idx': block_idx,
                'local_face': local_face,  # Which face slot in the block JSON
                'width': data['width'],
                'height': data['height'],
                'color': face_colors[target_direction]  # Use target direction color
            })
    
    if num_reoriented > 0:
        print(f"  Reoriented {num_reoriented} faces based on rotation")
    
    # Create grid for each direction
    grids = {}
    current_y = 0
    texture_width = 0
    
    for direction in ['top', 'front', 'back', 'left', 'right', 'bottom']:
        faces = faces_by_direction[direction]
        if not faces:
            grids[direction] = {'cell_w': 1, 'cell_h': 1, 'cols': 1, 'offset_y': current_y, 'faces': []}
            continue
        
        # Max dimensions for grid cells (+ 2px padding for 1px border on each side)
        max_w = max(f['width'] for f in faces) + 2
        max_h = max(f['height'] for f in faces) + 2
        
        # Grid layout (squarish)
        cols = max(1, int(math.sqrt(len(faces))))
        rows = (len(faces) + cols - 1) // cols
        
        grid_width = cols * max_w
        grid_height = rows * max_h
        
        grids[direction] = {
            'cell_w': max_w,
            'cell_h': max_h,
            'cols': cols,
            'offset_y': current_y,
            'faces': faces
        }
        
        texture_width = max(texture_width, grid_width)
        current_y += grid_height
        
        print(f"  {direction}: {len(faces)} faces in {cols}×{rows} grid (cell {max_w}×{max_h})")
    
    total_height = current_y
    
    # Place faces in grids WITH 1px OFFSET INSIDE CELLS
    all_block_uvs = [{} for _ in blocks]
    
    for direction, grid in grids.items():
        for idx, face_data in enumerate(grid['faces']):
            row = idx // grid['cols']
            col = idx % grid['cols']
            
            # Base position of cell
            cell_u = col * grid['cell_w']
            cell_v = grid['offset_y'] + row * grid['cell_h']
            
            # Place face with 1px offset inside cell
            u = cell_u + 1
            v = cell_v + 1
            
            block_idx = face_data['block_idx']
            local_face = face_data['local_face']  # Use local face name for JSON
            
            all_block_uvs[block_idx][local_face] = {
                'u': u,
                'v': v,
                'width': face_data['width'],
                'height': face_data['height'],
                'color': face_data['color'],
                'mirror': {'x': False, 'y': False}
            }
    
    print(f"  Grid layout: {texture_width}×{total_height}px")
    print(f"  Total faces: {sum(len(g['faces']) for g in grids.values())}")
    
    return texture_width, total_height, all_block_uvs, grids


def create_orthogonal_projection_texture(blocks, output_path, enable_mirror=False, mirror_axis='X'):
    """
    Create texture PNG with orthogonal projection layout.
    Each direction gets a 2D projection that maintains spatial relationships.
    """
    if not TEXTURE_AVAILABLE:
        return None
    
    texture_width, texture_height, all_block_uvs, projections = calculate_projection_based_uvs(
        blocks, enable_mirror, mirror_axis
    )
    
    # Create image with dark gray background
    img = Image.new('RGBA', (texture_width, texture_height), (64, 64, 64, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw projection boundaries for each direction (for debugging)
    for direction, proj in projections.items():
        offset_y = proj['offset_y']
        width = proj.get('width', 0)
        height = proj.get('height', 0)
        
        if height == 0:
            continue
        
        # Draw projection boundary
        draw.rectangle([0, offset_y, width-1, offset_y + height-1], 
                      outline=(128, 128, 128, 128))
    
    # Draw all block faces with their colors
    faces_drawn = 0
    faces_skipped = 0
    
    for block_idx, block_uvs in enumerate(all_block_uvs):
        for face_name, face_data in block_uvs.items():
            u = face_data['u']
            v = face_data['v']
            w = face_data['width']
            h = face_data['height']
            color = face_data['color']
            
            # Validate dimensions
            if w <= 0 or h <= 0:
                faces_skipped += 1
                continue
            
            # Draw the face
            try:
                draw.rectangle([u, v, u+w-1, v+h-1], fill=color)
                faces_drawn += 1
            except Exception as e:
                print(f"      ERROR drawing block {block_idx} {face_name}: {e}")
                faces_skipped += 1
    
    print(f"    Drew {faces_drawn} faces" + (f", skipped {faces_skipped}" if faces_skipped > 0 else ""))
    
    png_path = f"{output_path}_texture.png"
    img.save(png_path)
    
    print(f"    Texture: {os.path.basename(png_path)} ({texture_width}×{texture_height}px)")
    
    return png_path, texture_width, texture_height, all_block_uvs


def patch_block_uvs_from_projections(box_node, block_idx, all_block_uvs):
    """Patch UVs with orthogonal projection or overflow coordinates, including mirror flags."""
    if block_idx >= len(all_block_uvs):
        print(f"    WARNING: block_idx {block_idx} out of range")
        return box_node
    
    if 'textureLayout' not in box_node['shape']:
        print("    DEBUG: No textureLayout in box_node!")
        return box_node
    
    block_uvs = all_block_uvs[block_idx]
    
    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        if face_name in box_node['shape']['textureLayout'] and face_name in block_uvs:
            uv_data = block_uvs[face_name]
            box_node['shape']['textureLayout'][face_name]['offset']['x'] = uv_data['u']
            box_node['shape']['textureLayout'][face_name]['offset']['y'] = uv_data['v']
            
            # Apply mirror flags if present
            if 'mirror' in uv_data:
                box_node['shape']['textureLayout'][face_name]['mirror']['x'] = uv_data['mirror']['x']
                box_node['shape']['textureLayout'][face_name]['mirror']['y'] = uv_data['mirror']['y']
    
    return box_node

def create_colored_texture(blocks, output_path):
    """Create compact texture PNG with scale=1."""
    if not TEXTURE_AVAILABLE:
        return None
    
    png_width, png_height, all_face_uvs, projections, bounds = calculate_all_uvs_compact(blocks)
    
    img = Image.new('RGBA', (png_width, png_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw projection boundaries
    for proj_name, proj in projections.items():
        x = proj['offset_x']
        w = proj['width']
        h = proj['height']
        draw.rectangle([x, 0, x+w, h], outline=(128, 128, 128, 64))
    
    # Draw face rectangles
    total_faces = 0
    for block_uvs in all_face_uvs:
        for face_name, face_data in block_uvs.items():
            u = face_data['u']
            v = face_data['v']
            w = face_data['width']
            h = face_data['height']
            color = face_data['color']
            
            if w > 0 and h > 0:
                draw.rectangle([u, v, u+w, v+h], fill=color, outline=(255, 255, 255, 192))
                total_faces += 1
    
    png_path = f"{output_path}_texture.png"
    img.save(png_path)
    
    print(f"    Texture: {os.path.basename(png_path)} ({png_width}×{png_height}px)")
    print(f"    Scale: 1 pixel = 1 unit")
    print(f"    Drew {total_faces} block faces")
    
    return png_path, projections, bounds


def export_proportional_texture(blocks, output_path):
    """Export proportional texture."""
    if not TEXTURE_AVAILABLE or not blocks:
        return None, None, None
    
    print(f"  Creating proportional texture...")
    png_path, projections, bounds = create_colored_texture(blocks, output_path)
    return png_path, projections, bounds

def patch_block_uvs(box_node, block_center, block_size, projections, bounds):
    """Patch UVs with scale=1 and compact projections."""
    print(f"    DEBUG: Patching UVs for block at {block_center}")
    
    if 'textureLayout' not in box_node['shape']:
        print("    DEBUG: No textureLayout in box_node!")
        return box_node
    
    scale = 1  # 1 pixel = 1 unit
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    
    face_to_proj = {
        'front': ('nord', block_size.x, block_size.z),
        'back': ('sud', block_size.x, block_size.z),
        'right': ('est', block_size.y, block_size.z),
        'left': ('ouest', block_size.y, block_size.z),
        'top': ('haut', block_size.x, block_size.y),
        'bottom': ('bas', block_size.x, block_size.y),
    }
    
    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        if face_name in box_node['shape']['textureLayout']:
            proj_name, face_w, face_h = face_to_proj[face_name]
            proj = projections[proj_name]
            
            # Calculate position (scale=1)
            if face_name in ['front', 'back']:
                local_u = (block_center.x - face_w/2 - min_x) * scale
                local_v = (block_center.z - face_h/2 - min_z) * scale
            elif face_name in ['right', 'left']:
                local_u = (block_center.y - face_w/2 - min_y) * scale
                local_v = (block_center.z - face_h/2 - min_z) * scale
            else:
                local_u = (block_center.x - face_w/2 - min_x) * scale
                local_v = (block_center.y - face_h/2 - min_y) * scale
            
            u = int(proj['offset_x'] + local_u)
            v = int(local_v)
            
            print(f"    DEBUG: {face_name} UV = ({u}, {v})")
            
            box_node['shape']['textureLayout'][face_name]['offset']['x'] = u
            box_node['shape']['textureLayout'][face_name]['offset']['y'] = v
    
    return box_node



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
# Integer rounding for blocks
# ============================================================================

def round_blocks_to_integer(blocks, scale_multiplier=1):
    """
    Round block sizes and positions to create integer dimensions.
    
    Args:
        blocks: list of (center, size, quaternion) tuples
        scale_multiplier: multiply all dimensions by this factor first
    
    Returns:
        list of (center, size, quaternion) with integer sizes
    """
    rounded_blocks = []
    
    for center, size, quat in blocks:
        # Apply scale multiplier
        scaled_size = Vector((
            size.x * scale_multiplier,
            size.y * scale_multiplier,
            size.z * scale_multiplier,
        ))
        scaled_center = Vector((
            center.x * scale_multiplier,
            center.y * scale_multiplier,
            center.z * scale_multiplier,
        ))
        
        # Round sizes to nearest integer (minimum 1 for non-zero dimensions)
        new_size = Vector((
            max(1, round(scaled_size.x)) if scaled_size.x > 0.1 else 0.0,
            max(1, round(scaled_size.y)) if scaled_size.y > 0.1 else 0.0,
            max(1, round(scaled_size.z)) if scaled_size.z > 0.1 else 0.0,
        ))
        
        # Adjust center to align with integer grid
        # Round center to nearest 0.5 to keep blocks aligned
        new_center = Vector((
            round(scaled_center.x * 2) / 2,
            round(scaled_center.y * 2) / 2,
            round(scaled_center.z * 2) / 2,
        ))
        
        rounded_blocks.append((new_center, new_size, quat))
    
    return rounded_blocks


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


def export_mesh(obj, node_id, max_blocks, adjust_orientation, plane_threshold, enable_mirror=False, mirror_axis='X', merge_threshold=0.95, scale_multiplier=1, integer_only=False):
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
    
    # Apply integer rounding if requested
    if integer_only or scale_multiplier > 1:
        print(f"  Applying scale multiplier: {scale_multiplier}x")
        if integer_only:
            print(f"  Rounding to integer sizes...")
        blocks = round_blocks_to_integer(blocks, scale_multiplier)
        print(f"  After rounding: {len(blocks)} blocks")

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

    return group, blocks


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
        default=True,
    )

    export_selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected mesh objects",
        default=True,
    )

    enable_mirror: BoolProperty(
        name="Enable Mirroring",
        description="Split mesh in half and mirror blocks to double coverage",
        default=True,
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

    export_textures: BoolProperty(
        name="Export Textures",
        description="Generate texture atlas (requires Pillow)",
        default=True,
    )
    
    scale_multiplier: IntProperty(
        name="Scale Multiplier",
        description="Multiply mesh dimensions by this factor (1-10). Higher values = larger integer sizes",
        default=1,
        min=1,
        max=10,
    )
    
    integer_only: BoolProperty(
        name="Integer Sizes Only",
        description="Force block sizes to be whole numbers (better for textures)",
        default=True,
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
        print("HYTALE BLOCKY MODEL EXPORT v18.8")
        print(f"  Objects       : {len(objects)}")
        print(f"  Max blocks    : {self.max_blocks}")
        print(f"  Plane thresh  : {self.plane_threshold}")
        print(f"  Adjust orient : {self.adjust_orientation}")
        print(f"  Merge thresh  : {self.merge_threshold * 100:.0f}%")
        if self.scale_multiplier > 1 or self.integer_only:
            print(f"  Scale mult.   : {self.scale_multiplier}x")
            print(f"  Integer only  : {self.integer_only}")
        if self.enable_mirror:
            print(f"  Mirror axis   : {self.mirror_axis}")
            print(f"  Mirror UVs    : Grid-based (parallel faces)")
        print("=" * 60)

        top_nodes = []
        all_blocks = []  # Store blocks for each object
        for idx, obj in enumerate(objects):
            print(f"\n[{idx+1}/{len(objects)}] {obj.name}")
            group, blocks = export_mesh(
                obj,
                node_id=idx + 1,
                max_blocks=self.max_blocks,
                adjust_orientation=self.adjust_orientation,
                plane_threshold=self.plane_threshold,
                enable_mirror=self.enable_mirror,
                mirror_axis=self.mirror_axis,
                merge_threshold=self.merge_threshold,
                scale_multiplier=self.scale_multiplier,
                integer_only=self.integer_only,
            )
            top_nodes.append(group)
        all_blocks.append(blocks)  # Store blocks for texture export

        # Prepare filepath for JSON and texture
        filepath = bpy.path.abspath(self.filepath)
        
        # UV Calculation (always done if export_textures is enabled)
        if self.export_textures:
            print("\n" + "="*60)
            print("UV MAPPING GENERATION (PRIORITY-BASED + OVERFLOW)")
            print("="*60)
            base = os.path.splitext(filepath)[0]
            
            for idx, obj in enumerate(objects):
                try:
                    blocks = all_blocks[idx]
                    
                    # Convert Blender blocks to Hytale coordinates
                    from mathutils import Vector
                    hytale_blocks = []
                    for center, size, quat in blocks:
                        # Blender (X,Y,Z) → Hytale (X,Z,Y)
                        h_center = Vector((center.x, center.z, center.y))
                        h_size = Vector((size.x, size.z, size.y))
                        hytale_blocks.append((h_center, h_size, quat))
                    
                    print(f"  Processing {obj.name}: {len(hytale_blocks)} blocks")
                    
                    # Create PNG texture with priority-based placement (needs Pillow)
                    if TEXTURE_AVAILABLE:
                        result = create_orthogonal_projection_texture(
                            hytale_blocks, f"{base}_{obj.name}",
                            enable_mirror=self.enable_mirror,
                            mirror_axis=self.mirror_axis
                        )
                        if result:
                            png_path, texture_width, texture_height, all_block_uvs = result
                            print(f"  ✓ Texture: {os.path.basename(png_path)}")
                            
                            # Patch UVs in the JSON nodes
                            for i, box in enumerate(top_nodes[idx].get("children", [])):
                                if i < len(all_block_uvs):
                                    patch_block_uvs_from_projections(box, i, all_block_uvs)
                    else:
                        print(f"  ⚠ Pillow not installed - no texture PNG created")
                    
                except Exception as e:
                    print(f"  ✗ {obj.name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Write JSON (AFTER patching UVs if textures enabled)
        output = {"nodes": top_nodes, "format": "prop", "lod": "auto"}
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
        
        # Size Options
        box_size = layout.box()
        box_size.label(text="Size & Scale:", icon='EMPTY_ARROWS')
        box_size.prop(self, "scale_multiplier")
        box_size.prop(self, "integer_only")
        if self.integer_only:
            box_size.label(text="Rounds sizes to whole numbers (better for textures)", icon='INFO')
        if self.scale_multiplier > 1:
            box_size.label(text=f"All dimensions multiplied by {self.scale_multiplier}x", icon='INFO')
        
        layout.separator()
        
        # Textures
        box = layout.box()
        box.label(text="Textures:", icon='TEXTURE')
        box.prop(self, "export_textures")
        if self.export_textures:
            if not TEXTURE_AVAILABLE:
                row = box.row()
                row.alert = True
                row.label(text="⚠ Install Pillow", icon='ERROR')
            else:
                box.label(text="6 orthogonal projections of entire mesh", icon='INFO')
        
        layout.separator()
        tip = layout.box()
        tip.label(text="💡 Tips:", icon='SETTINGS')
        tip.label(text="• Max blocks: start low (4-16), increase for detail")
        tip.label(text="• Plane threshold 0 = never flatten to plane")
        tip.label(text="• Scale multiplier: use 2-5x for larger textures")
        tip.label(text="• Integer only: prevents texture misalignment issues")
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