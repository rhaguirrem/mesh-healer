# Mesh Heal

Desktop and CLI app to detect, heal, and combine triangulated meshes.

It uses existing geometry libraries instead of a custom kernel:
- `trimesh`
- `manifold3d` as the primary solid boolean backend
- `pymeshfix` (MeshFix)
- `pyvista` / VTK as a fallback boolean backend
- `ezdxf` for DXF 3DFACE and POLYFACE ingestion
- `PySide6` for the desktop GUI
- `pyvistaqt` for optional embedded 3D preview

## Supported inputs

- DXF with triangulated `3DFACE` or `POLYFACE` entities
- Leapfrog / ARANZ binary `.msh`
- STL, OBJ, PLY, and other mesh formats supported by `trimesh`

DXF files that contain only unsupported `POLYLINE` geometry are recognized and skipped with a structured skip report instead of failing the whole run.

## Supported outputs

- DXF written as triangulated `POLYFACE`
- Leapfrog / ARANZ binary `.msh`
- STL, OBJ, PLY, VTK, and other formats supported by `trimesh`

## Guided heal workflow

The current workflow is intentionally narrow:

1. Load a mesh.
2. Declare whether the result is meant to stay open or become closed.
3. Optionally load external problem hints for bad points, edges, or triangles.
4. Rebuild locally from the problematic regions first.
5. After each local repair stage, sanitize duplicate and degenerate triangles before continuing.
6. If the mesh is meant to be solid, bias the pipeline toward watertight closure.

This is the primary workflow in both the GUI and CLI.

Manual Repair and Autoresearch are deprecated. OpenMeshCraft backends are also deprecated.

## Healing strategy

The core pipeline performs these stages:

1. Remove duplicate triangles.
2. Remove degenerate triangles.
3. Merge near-identical vertices.
4. Rebuild triangle soup locally when hints or rebuild mode request it.
5. Run optional local repair passes such as non-manifold edge repair or localized self-intersection repair.
6. Sanitize the intermediate mesh after each local stage.
7. Heal disconnected components.
8. Optionally attempt watertight closure.
9. Run a final sanitation pass and fix normals.

Optional modes:
- Triangle rebuild from external problem hints.
- Experimental non-manifold edge sleeve repair.
- Experimental localized self-intersection repair.
- Optional distance-hull generation.
- Optional surface-shell generation.
- Optional watertight repair.
- Optional CGAL Alpha Wrap fallback.

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

## CLI usage

Default heal run:

```bash
uv run python mesh_heal.py input.dxf healed.stl --report report.json
```

Explicit intended solid:

```bash
uv run python mesh_heal.py heal input.dxf healed.msh --intended-mesh-type solid --report heal_report.json
```

Guided rebuild from one or more external hint files:

```bash
uv run python mesh_heal.py heal input.dxf healed.msh --intended-mesh-type solid --hints leapfrog_hints_a.json leapfrog_hints_b.dxf --report heal_report.json
```

Localized self-intersection repair:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --localized-intersection-repair --report report.json
```

Non-manifold edge sleeve repair:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --nonmanifold-edge-repair --report report.json
```

Explicit watertight repair:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --make-watertight --report report.json
```

Distance hull:

```bash
uv run python mesh_heal.py heal input.dxf distance_hull.stl --distance-model distance-hull --distance-offset 2.5 --report report.json
```

Surface shell:

```bash
uv run python mesh_heal.py heal input_surface.obj thickened_surface.stl --distance-model surface-shell --distance-offset 2.5 --report report.json
```

`distance-hull` extracts a new enclosing iso-surface from an unsigned distance field. `surface-shell` thickens the current mesh itself by offsetting it on both sides and stitching any open boundary rims.

Surface return after watertight repair:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --return-surface-after-watertight --report report.json
```

The deprecated `autoresearch` command still parses, but now returns a structured deprecation report instead of running candidate search.

## Advanced backends

Supported advanced backends:
- `none`
- `cgal-alpha-wrap`
- `cgal-repair`

OpenMeshCraft backends are deprecated and should not be used for new repair runs.

CGAL Alpha Wrap example:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --advanced-backend cgal-alpha-wrap --cgal-backend-exe C:/tools/mesh_heal_cgal_alpha_wrap.exe --report report.json
```

CGAL repair example:

```bash
uv run python mesh_heal.py heal input.dxf healed.stl --advanced-backend cgal-repair --cgal-backend-exe C:/tools/mesh_heal_cgal_alpha_wrap.exe --report report.json
```

The CGAL sidecar project lives in `advanced_backends/cgal_alpha_wrap` and builds with CMake against a local CGAL installation.

On Windows, `build_advanced_backends.ps1` can provision the optional backend toolchain and build `mesh_heal_cgal_alpha_wrap.exe` into `C:\Tools\MeshHealBackends`.

You can either pass the executable path on the command line or set:
- `CGAL_ALPHA_WRAP_EXE`

Confirm CGAL package licensing before redistributing that backend.

## Boolean operations

Supported operations:
- `union`
- `intersection`
- `clip` (`left - right`)

Example:

```bash
uv run python mesh_heal.py boolean left.stl right.stl result.stl --operation union --report boolean_report.json
```

Boolean operations use `manifold3d` through `trimesh` when available, with a VTK fallback through `pyvista`.

## GUI

```bash
uv run python mesh_heal_gui.py
```

The GUI now opens directly into the Heal window with an embedded preview pane. The main supported flow is guided healing, not manual mesh editing.
