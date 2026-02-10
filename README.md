# Hytale Blocky Model Exporter for Blender

A Blender addon that converts standard mesh geometry into Hytale's proprietary .blockymodel format. Instead of simple voxelization, it uses an intelligent subdivision and shrinking algorithm to approximate mesh shapes with the fewest possible blocks, making them ready for import into tools like Blockbench.

> Vibe-coded with [Claude.ai](https://claude.ai).

---

## Why Does This Exist?

Creating "blocky" assets for Hytale or Minecraft-style aesthetics usually requires modeling with cubes from scratch. Converting existing high-fidelity meshes (like sculpted statues or organic shapes) into this format manually is incredibly tedious.
This addon automates that workflow by "voxelizing" your Blender mesh, intelligently rotating and resizing blocks to fit the original silhouette, and exporting a clean JSON file that Hytale model editors can read.

---

## Files

| File | Description |
|------|-------------|
| `export2blockymodel.py` | **Main addon** — analyzes mesh geometry, subdivides it into blocks, optimizes the structure, and exports a `.blockymodel` JSON file. |

---

## Features

- **Smart Subdivision :** Automatically calculates a proportional grid based on the mesh's bounding box.
- **Tight Fitting :** "Shrink-wraps" blocks to the mesh volume (AABB) rather than using a fixed grid.
- **Planar Optimization :** Automatically detects and merges adjacent flat blocks (planes) to reduce element count.
- **PCA Orientation :** Optional logic to rotate blocks to align with local geometry flow (using Principal Component Analysis).
- **Coordinate Conversion :** Automatically handles the transformation from Blender (Z-up) to Hytale (Y-up).
- **Batch Export :** Can export single objects or all objects in the scene as a hierarchy.

---

## Requirements

- Blender **2.80** or newer (including Blender 4.x)
- Standard Python libraries (included with Blender): `json`, `math`, `numpy`.

---

## Installation

### Option 1 — Install via Blender Preferences (recommended)

1. Download or clone this repository.
2. Open Blender and go to **Edit > Preferences > Add-ons**.
3. Click **Install...** and select either `export2blockymodel.py`.
4. Enable the addon by checking the checkbox next to its name.

### Option 2 — Manual installation

Copy the script(s) to your Blender addons directory:

- **Windows:** `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
- **macOS:** `~/Library/Application Support/Blender/<version>/scripts/addons/`
- **Linux:** `~/.config/blender/<version>/scripts/addons/`

Then enable the addon in **Edit > Preferences > Add-ons**.

---

## Usage

### Importing a COLLADA file (`dea2obj2import.py`)

1. Select the mesh object(s) you wish to export in the 3D Viewport.
2. Go to **File > Export > Hytale Blocky Model (.blockymodel)**.
3. Adjust the export settings in the side panel :
   - **Max Blocks per Mesh :** (Default: 8) Controls the resolution. Higher numbers create more detailed (but heavier) models.
   - **Plane Threshold :** (Default: 0.5) Dimensions smaller than this value are snapped to 0, creating flat 2D planes (ideal for leaves, wings, or cloth).
   - **Adjust Orientation :** (Default: False) If enabled, uses PCA to rotate blocks to better fit diagonal or curved surfaces. Note: This is computationally heavier.
   - **Selected Only :** Check this to export only the currently selected objects.
4. Click **Export Blocky Model**.
5. Import the resulting file into **Blockbench** (via *File > Open Model*) to texture or animate it.

---

## How It Works

1. **AABB Analysis :** Calculates the global bounding box of the mesh.
2. **Grid Generation :** Creates a proportional grid ($N_x \times N_y \times N_z$) aiming for the target `Max Blocks` count.
3. **Barycentric Sampling :** Populates mesh faces with sample points to accurately detect which grid cells contain geometry (preventing "bleeding" into empty space).
4. **Trimming :** Calculates the tightest possible bounding box for the geometry inside each cell.
5. **Planar Merge :** A post-processing pass uses a Union-Find algorithm to merge adjacent coplanar blocks, significantly reducing the final block count.
6. **JSON Generation :** Outputs a structured JSON file with `Box` and `Group` nodes compatible with Hytale's format.

---

## Known Limitations

- **Geometry Only :** This tool exports shape and rotation. UV maps and Textures are not exported and must be applied in Blockbench.
- **Approximation :** This is a lossy process. The result is a blocky approximation, not a 1:1 replica of the mesh.
- **Complex Geometry :** High-poly meshes with "Adjust Orientation" enabled may take several seconds to process due to the matrix calculations required for every block.

---

## Contributing

Contributions and bug reports are welcome.

---

## License

See [LICENSE](LICENSE) for details.

---

## Credits

- Addon development: vibe-coded with [Claude.ai](https://claude.ai)
