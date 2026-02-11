# Hytale Blocky Model Exporter v18

Blender addon to export meshes to the `.blockymodel` format used by Hytale / Blockbench.

---

## Installation

1. In Blender: **Edit → Preferences → Add-ons → Install**
2. Select `hytale_exporter_v18.py`
3. Enable the **"Export Hytale Blocky Model"** addon
4. The exporter will appear under **File → Export → Hytale Blocky Model (.blockymodel)**

---

## Usage

Select one or more mesh objects, then go to **File → Export → Hytale Blocky Model**.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| **Selected Only** | Off | Export only the currently selected objects |
| **Max Blocks per Mesh** | 8 | Maximum number of blocks generated per mesh |
| **Plane Threshold** | 0.5 | Dimensions below this value are collapsed to 0 (flat plane). Set to 0 to disable |
| **Adjust Orientation** | Off | Uses PCA to rotate each block for a tighter fit to the local surface geometry |

---

## Algorithm

### Overview

For each mesh object, the exporter:

1. **Computes the AABB** (axis-aligned bounding box) of the mesh
2. **Divides that space** into a proportional grid — axes with greater extent receive more subdivisions — capped at `Max Blocks`
3. **For each cell**, samples the mesh faces (vertices, edge midpoints, face centers, barycentric sub-samples) to verify that a face actually passes through the cell — empty cells are discarded
4. **Computes a tight block** around the samples found inside the cell
5. **Applies the plane threshold**: any dimension smaller than the threshold is set to 0
6. If **Adjust Orientation** is enabled: PCA is run on the cell's sample points to find a rotation that minimises block volume while preserving surface coverage
7. **Merges adjacent coplanar planes** (same orientation, same position on the flat axis) using Union-Find to reduce the total block count

### Key design points

- **No floating blocks**: cells are validated by actual face intersection, with no tolerance margin that could bleed into empty neighbouring cells
- **Plane blocks**: when geometry in a cell is very thin, the block is flattened to a plane (size 0 on the thin axis). Adjacent coplanar planes are then automatically merged
- **Coordinate space conversion**: Blender is Z-up, Hytale is Y-up. Positions, sizes, and rotations are properly converted via matrix conjugation — not a naive component swap

---

## Output Format

JSON `.blockymodel` file:

```json
{
  "nodes": [
    {
      "id": "1",
      "name": "MeshName",
      "position": { "x": 0, "y": 0, "z": 0 },
      "orientation": { "x": 0, "y": 0, "z": 0, "w": 1 },
      "shape": { "type": "none" },
      "children": [
        {
          "id": "1_0",
          "name": "MeshName_block_0",
          "position": { "x": 1.5, "y": 0.0, "z": -2.0 },
          "orientation": { "x": 0, "y": 0, "z": 0, "w": 1 },
          "shape": {
            "type": "box",
            "settings": {
              "size": { "x": 10, "y": 5, "z": 3 },
              "isStaticBox": true
            }
          }
        }
      ]
    }
  ],
  "format": "prop",
  "lod": "auto"
}
```

Each mesh becomes a **group node** containing its blocks as children.

---

## Tips

- **Start with a low block count** (4–8) to validate the overall shape, then increase for more detail
- **Adjust Orientation** is most useful on diagonal or curved surfaces, but can be slow on dense meshes
- **Plane Threshold at 0** keeps all blocks as volumes (no planes are generated)
- For a mesh that is very elongated on one axis (e.g. a sword), the grid will automatically allocate more subdivisions along that axis

---

## Requirements

- Blender 2.80+
- Hytale / Blockbench (`.blockymodel` format)
- `numpy` (included in Blender's bundled Python)