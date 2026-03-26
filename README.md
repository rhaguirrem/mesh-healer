# Mesh Heal

Desktop and CLI app to detect, heal, and combine triangulated solids.

It uses existing geometry libraries instead of custom kernels:
- `trimesh`
- `manifold3d` as the primary solid boolean backend
- `pymeshfix` (MeshFix)
- `pyvista` / VTK as a fallback boolean backend
- `ezdxf` for DXF 3DFACE ingestion
- `PySide6` for the desktop GUI
- `pyvistaqt` for optional embedded 3D preview

## What it detects

- Overlapped/duplicate triangles
- Degenerate triangles
- Non-manifold edges
- Boundary edges (open holes)
- Watertightness and winding consistency

## Supported inputs

- DXF with triangulated `3DFACE` entities
- Leapfrog / ARANZ binary `.msh`
- STL, OBJ, PLY and other mesh formats supported by `trimesh`

## Supported outputs

- DXF written as triangulated `3DFACE` entities
	- exported as connected `POLYFACE` when writing DXF
- Leapfrog / ARANZ binary `.msh`
- STL, OBJ, PLY, VTK and other formats supported by `trimesh`

## Healing strategy

1. Remove duplicate triangles
2. Remove degenerate triangles
3. Merge near-identical vertices
4. Split disconnected solids and heal each component independently
5. Run optional cleanup passes from installed mesh libraries
6. Run MeshFix repair
7. Fix normals and fill remaining small holes
8. Run a final sanitation pass to remove any duplicate or degenerate triangles introduced during repair
9. If multiple healed solids remain and each is watertight, union them before export so overlapping components do not survive as stacked triangles
10. Write the cleaned output mesh file

Optional heal mode:
- Rebuild the input triangle by triangle before repair, union coplanar triangle patches in 2D, and retriangulate them back into 3D before the normal repair pass
- Experimental localized self-intersection repair preprocesses the mesh, detects intersecting triangle regions, repairs those regions independently, and stitches them back into the global cleanup pass
- Optional watertight repair attempts to close open boundaries and fill holes before export
- Optional surface-only return mode stops after watertight repair and exports that repaired surface directly
- Advanced external backend mode can run OpenMeshCraft arrangements first, then pass the exact surface through either FastTetWild or OpenMeshCraft CDT before the normal cleanup pass

## Boolean operations

- `union`
- `intersection`
- `clip` (`left - right`)

Boolean operations use `manifold3d` through `trimesh` when available, with a VTK fallback through `pyvista`.

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

## Usage

```bash
uv run python mesh_heal.py input.dxf healed.stl --report report.json
```

Legacy CLI usage without a subcommand still maps to `heal`.

### Explicit heal command

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --report report.json
```

To enable the triangle-by-triangle rebuild filter:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --rebuild-triangles --report report.json
```

To enable the experimental localized self-intersection pass:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --localized-intersection-repair --report report.json
```

To attempt hole filling and watertight closure:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --make-watertight --report report.json
```

To stop after watertight repair and export the repaired surface directly:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --return-surface-after-watertight --report report.json
```

To run the advanced exact plus tetrahedral backend path:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --advanced-backend omc-ftetwild --exact-arrangements-exe C:/tools/OpenMeshCraft-Arrangements.exe --tetra-backend-exe C:/tools/FloatTetwild_bin.exe --report report.json
```

Supported advanced backends:
- `none`
- `omc-ftetwild` for OpenMeshCraft arrangements followed by FastTetWild
- `omc-cdt` for OpenMeshCraft arrangements followed by OpenMeshCraft CDT

The advanced backends are not bundled with this project. They require external builds and executables:
- OpenMeshCraft arrangements executable: `OpenMeshCraft-Arrangements`
- OpenMeshCraft CDT executable: `OpenMeshCraft-CDT`
- FastTetWild executable: `FloatTetwild_bin`

You can either pass executable paths on the command line or set environment variables:
- `OPENMESHCRAFT_ARRANGEMENTS_EXE`
- `OPENMESHCRAFT_CDT_EXE`
- `FASTTETWILD_EXE`

Output format is inferred from the output extension. For example:

```bash
uv run python mesh_heal.py heal input.msh healed.dxf
uv run python mesh_heal.py heal input.dxf healed.msh
```

### Boolean command

```bash
uv run python mesh_heal.py boolean left.stl right.stl result.stl --operation union --report boolean_report.json
```

### GUI

```bash
uv run python mesh_heal_gui.py
```

The GUI has two tabs:
- `Heal` for single-solid cleanup and repair
- `Boolean` for union, intersection, and clip across one ordered list of solids, including batch runs with multiple operations selected at once

The Heal tab also includes an experimental localized self-intersection repair option and advanced external backend selectors for meshes where a global MeshFix-style pass is not enough.

In the GUI Heal tab, choosing an input file auto-fills the output path with the same filename plus `_healed` in the same folder by default.

When a heal input contains multiple disconnected solids, the tool keeps them separated during repair, heals each one independently, and writes them back together into the same output file.

When the GUI Boolean tab runs multiple operations in one pass, it writes suffixed outputs such as `result_union.stl`, `result_intersection.stl`, and matching report files.

The GUI Boolean tab lets you add files iteratively from different folders into one ordered list.

For Boolean operations in the GUI:
- `union` unions all solids in the list
- `intersection` intersects all solids in the list
- `clip` uses the first solid as the base and subtracts each following solid in order

The GUI also includes a manual 3D preview pane:
- Preview is opt-in and never auto-loads
- Preview loading happens in a worker thread so the UI stays responsive
- Large meshes are decimated to a bounded preview size before rendering
- Very large files are skipped instead of forcing an unsafe preview load

## Notes for DXF

- DXF loader reads triangulated `3DFACE` entities and `POLYFACE` meshes.
- DXF export writes a connected `POLYFACE` mesh instead of isolated `3DFACE` entities.
- `POLYLINE` polygon meshes other than `POLYFACE` are rejected with a clear error message for now.
- If your DXF uses other entity types for surfaces, pre-convert to triangulated faces.

## Notes for MSH

- `.msh` support targets Leapfrog / ARANZ binary meshes with `Location Double 3` and `Tri Integer 3` blocks.
- Unsupported `.msh` variants fail with an explicit parse error.

## Notes for Maptek Vulcan

- `.00t` load/save is not implemented.
- There is no suitable free public library or documented specification available in this workspace to support `.00t` safely.
- If you need `.00t`, the practical path is a proprietary Maptek Vulcan SDK or an external conversion step into DXF / MSH / STL.
