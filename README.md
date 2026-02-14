# Hytale Blocky Model Exporter v18.1

Blender addon to export meshes to the `.blockymodel` format used by Hytale / Blockbench, with optional texture atlas generation.

---

## Installation

1. In Blender: **Edit → Preferences → Add-ons → Install**
2. Select `export2blockymodel.py`
3. Enable the **"Export Hytale Blocky Model"** addon
4. **(Optional)** For texture export, install Pillow:

### Installing Pillow

#### Windows - Easy Method
1. Download `pip-install-Pillow-Windows.py`
2. Open Blender **as Administrator** (right-click Blender icon → Run as Administrator)
3. Go to **Scripting** tab
4. Click **New** → **Open Text File** → select `pip-install-Pillow-Windows.py`
5. Click **Run Script** (or press Alt+P)
6. Wait for installation to complete
7. **Restart Blender**

#### Windows/Mac/Linux - Manual Method
```bash
# Find Blender's Python path:
# Windows: C:\Program Files\Blender Foundation\Blender 3.X\3.X\python\bin\python.exe
# Mac: /Applications/Blender.app/Contents/Resources/3.X/python/bin/python3
# Linux: ~/.local/share/blender/3.X/python/bin/python3

# Then install Pillow:
<python_path> -m pip install Pillow
```

5. The exporter will appear under **File → Export → Hytale Blocky Model (.blockymodel)**

---

## Two Versions Available

### Main Version (`export2blockymodel.py`)
- **Recommended** for most use cases
- Renders mesh from 6 angles using orthographic camera
- Creates texture atlas with actual rendered views
- Best for models with complex materials/lighting

### Plan B (`export2blockymodel_planB_template.py`)  
- Uses original mesh textures and UVs
- Preserves exact texture appearance
- Best for models with precise UV mapping
- **Note**: Currently a template - requires completion

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
| **Enable Mirroring** | Off | Split mesh in half and mirror blocks along selected axis |
| **Mirror Axis** | X | Axis to mirror along (X/Y/Z) |
| **Merge Threshold** | 95% | Minimum volume overlap ratio to merge blocks |
| **Export Textures** ⭐ | Off | Generate texture atlas from mesh materials |
| **Atlas Resolution** ⭐ | 512 | Resolution per tile (total atlas is 4×3 tiles) |

⭐ = New in v18.1

---

## Algorithm

### Geometry Export

For each mesh object, the exporter:

1. **Computes the AABB** (axis-aligned bounding box) of the mesh
2. **Divides that space** into a proportional grid — axes with greater extent receive more subdivisions — capped at `Max Blocks`
3. **For each cell**, samples the mesh faces (vertices, edge midpoints, face centers, barycentric sub-samples) to verify that a face actually passes through the cell — empty cells are discarded
4. **Computes a tight block** around the samples found inside the cell
5. **Applies the plane threshold**: any dimension smaller than the threshold is set to 0
6. If **Adjust Orientation** is enabled: PCA is run on the cell's sample points to find a rotation that minimises block volume while preserving surface coverage
7. **Merges adjacent coplanar planes** (same orientation, same position on the flat axis) using Union-Find to reduce the total block count
8. **Merges adjacent/overlapping volumes** based on the merge threshold
9. If **Enable Mirroring**: mirrors blocks across the selected axis and performs a final merge

### Texture Export ⭐

When **Export Textures** is enabled:

1. **6-Axis Rendering**: Creates orthographic camera renders from 6 directions (+X, -X, +Y, -Y, +Z, -Z)
   - Positions camera to capture full mesh from each angle
   - Uses orthographic projection for consistent scale
   - Renders with Cycles engine and transparent background
   - Each render captures the actual appearance of the mesh

2. **Atlas Creation**: Combines the 6 renders into a 3×2 grid atlas:
   ```
        [ +Y ]
   [-X][+Z][+X][-Z]
        [-Y ]
   ```

3. **UV Mapping**: Generates UV coordinates mapping each cube face to the corresponding atlas region

4. **Export Location**:
   - If project is saved: Exports to same folder as .blockymodel
   - If project unsaved: Exports to `/export/` subfolder in temp directory
   - PNG files are never created in the scene folder randomly

5. **Proportional Sizing**: Atlas resolution is calculated based on total cube surface area
   - Larger meshes get higher resolution atlases
   - "One square for one square" principle maintained

**Atlas Sizes:**
- Resolution 256: 1024×768 atlas
- Resolution 512: 2048×1536 atlas (default)
- Resolution 1024: 4096×3072 atlas
- Resolution 2048: 8192×6144 atlas

### Key Design Points

- **No floating blocks**: cells are validated by actual face intersection, with no tolerance margin that could bleed into empty neighbouring cells
- **Plane blocks**: when geometry in a cell is very thin, the block is flattened to a plane (size 0 on the thin axis). Adjacent coplanar planes are then automatically merged
- **Coordinate space conversion**: Blender is Z-up, Hytale is Y-up. Positions, sizes, and rotations are properly converted via matrix conjugation — not a naive component swap
- **Real camera rendering**: Uses orthographic camera positioned at 6 angles to capture actual mesh appearance (not texture baking)
- **Smart export location**: PNG files are placed in proper export folder, never scattered in scene directory
- **Proportional UVs**: UV mapping reflects actual 3D position and scale of each cube face

---

## Output Format

### Without Textures

Single JSON `.blockymodel` file:

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

### With Textures ⭐

Multiple files:

```
model.blockymodel       # JSON with UV coordinates
model_Cube.png          # 2048×1536 texture atlas
model_Sphere.png        # 2048×1536 texture atlas
```

UV coordinates in JSON:
```json
{
  "shape": {
    "textureLayout": {
      "front": {
        "offset": {"x": 25, "y": 33},
        "mirror": {"x": false, "y": false},
        "angle": 0
      }
    }
  }
}
```

---

## Tips

### Geometry

- **Start with a low block count** (4–8) to validate the overall shape, then increase for more detail
- **Adjust Orientation** is most useful on diagonal or curved surfaces, but can be slow on dense meshes
- **Plane Threshold at 0** keeps all blocks as volumes (no planes are generated)
- **Mirroring** is useful for symmetric models to save computation time
- For a mesh that is very elongated on one axis (e.g. a sword), the grid will automatically allocate more subdivisions along that axis

### Textures ⭐

- **Resolution guide**:
  - 256: Low-poly/stylized (fast rendering)
  - 512: Standard quality (recommended)
  - 1024: High detail (slower)
  - 2048: Hero assets only (slowest)

- **Rendering tips**:
  - Good scene lighting produces better texture atlases
  - Materials with emission work well
  - Transparent materials render with alpha channel
  - Orthographic renders preserve scale accurately

- **File locations**:
  - Always check the export folder if project unsaved
  - PNG files never scatter in scene directory
  - Use same name as .blockymodel for easy organization

- **Performance**: 
  - 512 res: ~5-10s per object (6 renders)
  - 1024 res: ~15-25s per object
  - 2048 res: ~45-90s per object
  - Time depends on scene complexity and lighting

- **Plan B option**:
  - Use `export2blockymodel_planB_template.py` if you need exact UV preservation
  - Better for models with precise texture mapping
  - Requires completion of template code

---

## Troubleshooting

### Windows: "⚠ Install Pillow" Appears

**Problem**: Texture export option shows warning message

**Solution**: 
1. Download `pip-install-Pillow-Windows.py` from the addon folder
2. Open Blender **as Administrator**
3. Go to Scripting tab
4. Open the script and click Run Script
5. Restart Blender

### PNG Files Created in Wrong Location

**Problem**: Texture files appear in random locations

**Solution**: This was fixed in v18.1. Textures now export to:
- Same folder as .blockymodel (if project is saved)
- `/export/` subfolder in temp directory (if project unsaved)

Never in the scene folder unless the .blockymodel is also saved there.

### Textures Look Wrong/Incorrect

**Problem**: Atlas doesn't match expected appearance

**Possible causes**:
1. **Materials missing**: Mesh has no materials → will use colored fallback
2. **Lighting issues**: Scene lighting affects render → use good lighting or add materials with emission
3. **Camera angle**: Mesh orientation may not align with cardinal axes → rotate mesh if needed

**Solution**: 
- Ensure mesh has materials assigned
- Check scene lighting
- Try different atlas resolutions
- Consider using Plan B version for exact UV preservation

### Blocks Appear in Wrong Locations

**Problem**: Blocks are floating or disconnected

**Solution**: This was fixed in v18 by using face sampling. Make sure you're using the latest version.

---

## Requirements

- **Blender 2.80+**
- **Hytale / Blockbench** (`.blockymodel` format)
- **numpy** (included in Blender's bundled Python)
- **Pillow** (optional, for texture export)

---

## Version History

### v18.1 (Current)
- ✨ Added texture atlas export with 6-axis orthographic rendering
- ✨ Real camera-based renders (not texture baking)
- ✨ Smart export location (no scattered PNG files)
- ✨ UV coordinate generation based on 3D positions
- ✨ Proportional atlas sizing
- ✨ Integrated texture module (no separate files needed)
- ✨ Windows Pillow installation script
- 📦 Plan B template for original texture/UV preservation

### v18.0
- ✨ Advanced block merging (face contact, containment, overlap)
- ✨ Mesh mirroring feature
- ✨ PCA orientation fitting
- 🐛 Fixed floating blocks via face sampling
- 🐛 Fixed quaternion coordinate conversion
- 🐛 Fixed planar block merging

---

## License

MIT License - See LICENSE file for details

---

## Credits

Created by Claude (Anthropic)