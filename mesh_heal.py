import argparse
from collections import defaultdict
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh


def _try_import_ezdxf():
    try:
        import ezdxf  # type: ignore

        return ezdxf
    except Exception:
        return None


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception:
        return None


def _try_import_pymeshfix():
    try:
        import pymeshfix  # type: ignore

        return pymeshfix
    except Exception:
        return None


def _try_import_pyvista():
    try:
        import pyvista as pv  # type: ignore

        return pv
    except Exception:
        return None


def _try_import_shapely():
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon  # type: ignore
        from shapely.ops import triangulate, unary_union  # type: ignore

        return {
            "GeometryCollection": GeometryCollection,
            "MultiPolygon": MultiPolygon,
            "Polygon": Polygon,
            "triangulate": triangulate,
            "unary_union": unary_union,
        }
    except Exception:
        return None


LEAPFROG_MSH_PREAMBLE = bytes.fromhex("ff0ff0001bde8342cac0f33f")


@dataclass
class MeshReport:
    vertices: int
    faces: int
    watertight: bool
    winding_consistent: bool
    euler_number: int
    volume: float
    area: float
    degenerate_faces: int
    duplicate_faces: int
    nonmanifold_edges: int
    boundary_edges: int


@dataclass
class PreviewPayload:
    source: str
    original_vertices: int
    original_faces: int
    preview_vertices: int
    preview_faces: int
    decimated: bool
    vertices: np.ndarray
    faces: np.ndarray


StatusCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, int], None]]

HEAL_PREPROCESS_STEP_COUNT = 3
HEAL_COMPONENT_STEP_COUNT = 3

BOOLEAN_OPERATIONS = ("union", "intersection", "clip")
ADVANCED_HEAL_BACKENDS = ("none", "omc-ftetwild", "omc-cdt")

OPENMESHCRAFT_ARRANGEMENTS_CANDIDATES = (
    "OpenMeshCraft-Arrangements.exe",
    "OpenMeshCraft-Arrangements",
)
OPENMESHCRAFT_CDT_CANDIDATES = (
    "OpenMeshCraft-CDT.exe",
    "OpenMeshCraft-CDT",
)
FASTTETWILD_CANDIDATES = (
    "FloatTetwild_bin.exe",
    "FloatTetwild_bin",
)

OPENMESHCRAFT_ARRANGEMENTS_ENV = "OPENMESHCRAFT_ARRANGEMENTS_EXE"
OPENMESHCRAFT_CDT_ENV = "OPENMESHCRAFT_CDT_EXE"
FASTTETWILD_ENV = "FASTTETWILD_EXE"


def emit_status(status_callback: StatusCallback, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def emit_progress(progress_callback: ProgressCallback, current: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(current, total)


class ProgressTracker:
    def __init__(self, progress_callback: ProgressCallback, total_steps: int):
        self.progress_callback = progress_callback
        self.total_steps = max(1, int(total_steps))
        self.current_step = 0
        emit_progress(self.progress_callback, self.current_step, self.total_steps)

    def advance(self, steps: int = 1) -> None:
        self.current_step = min(self.total_steps, self.current_step + steps)
        emit_progress(self.progress_callback, self.current_step, self.total_steps)


def normalize_boolean_operation(operation: str) -> str:
    operation_name = operation.lower()
    if operation_name == "difference":
        operation_name = "clip"
    if operation_name not in BOOLEAN_OPERATIONS:
        raise ValueError(f"Unsupported boolean operation: {operation}")
    return operation_name


def derive_operation_path(base_path: Path, operation: str, multi_operation: bool) -> Path:
    operation_name = normalize_boolean_operation(operation)
    if not multi_operation:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{operation_name}{base_path.suffix}")


def derive_healed_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_healed{input_path.suffix}")


def normalize_advanced_heal_backend(backend: str) -> str:
    backend_name = backend.lower().strip()
    if backend_name not in ADVANCED_HEAL_BACKENDS:
        raise ValueError(f"Unsupported advanced heal backend: {backend}")
    return backend_name


def resolve_external_executable(
    explicit_path: Optional[Path | str],
    env_var: str,
    candidates: Sequence[str],
) -> Optional[str]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return str(path.resolve())
        raise FileNotFoundError(f"Executable not found: {path}")

    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return str(env_path.resolve())
        raise FileNotFoundError(f"Executable from {env_var} not found: {env_path}")

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def run_external_command(
    command: Sequence[str],
    cwd: Path,
    status_callback: StatusCallback = None,
) -> dict:
    emit_status(status_callback, f"Running external backend: {' '.join(command)}")
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        message = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"External backend failed: {message}")
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def load_surface_from_vtk(path: Path) -> trimesh.Trimesh:
    pv = _try_import_pyvista()
    if pv is None:
        raise RuntimeError("VTK surface extraction requires pyvista. Install requirements first.")

    dataset = pv.read(str(path))
    surface = dataset.extract_surface().triangulate().clean()
    return polydata_to_mesh(surface)


def export_temporary_triangle_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def ensure_watertight_for_advanced_backend(mesh: trimesh.Trimesh, stage: str) -> None:
    if mesh.is_watertight:
        return

    _, boundary_edges = detect_edge_stats(mesh)
    raise RuntimeError(
        f"Advanced tetrahedral backend requires a closed watertight surface before {stage}. "
        f"Found {boundary_edges} boundary edges. Open surfaces get capped into an enclosing shell."
    )


def run_openmeshcraft_arrangements_backend(
    mesh: trimesh.Trimesh,
    executable_path: Optional[Path | str] = None,
    tree_leaf: Optional[int] = None,
    tree_adaptive: Optional[float] = None,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    resolved_executable = resolve_external_executable(
        explicit_path=executable_path,
        env_var=OPENMESHCRAFT_ARRANGEMENTS_ENV,
        candidates=OPENMESHCRAFT_ARRANGEMENTS_CANDIDATES,
    )
    if resolved_executable is None:
        raise RuntimeError(
            "OpenMeshCraft arrangements executable not found. Set OPENMESHCRAFT_ARRANGEMENTS_EXE, "
            "pass --exact-arrangements-exe, or place OpenMeshCraft-Arrangements on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="mesh_heal_omc_arr_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "arrangement_input.obj"
        export_temporary_triangle_mesh(mesh, input_path)

        command = [resolved_executable, str(input_path), "-r"]
        if tree_leaf is not None:
            command.append(f"--tree_leaf={int(tree_leaf)}")
        if tree_adaptive is not None:
            command.append(f"--tree_adaptive={float(tree_adaptive)}")

        process_report = run_external_command(command, cwd=temp_dir, status_callback=status_callback)
        output_path = temp_dir / input_path.name
        if not output_path.exists():
            raise RuntimeError("OpenMeshCraft arrangements did not produce an explicit output mesh.")

        arranged_mesh = load_mesh(output_path)
        return arranged_mesh, {
            "backend": "openmeshcraft-arrangements",
            "executable": resolved_executable,
            "input": str(input_path),
            "output": str(output_path),
            "process": process_report,
        }


def run_ftetwild_surface_backend(
    mesh: trimesh.Trimesh,
    executable_path: Optional[Path | str] = None,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    resolved_executable = resolve_external_executable(
        explicit_path=executable_path,
        env_var=FASTTETWILD_ENV,
        candidates=FASTTETWILD_CANDIDATES,
    )
    if resolved_executable is None:
        raise RuntimeError(
            "FastTetWild executable not found. Set FASTTETWILD_EXE, pass --tetra-backend-exe, "
            "or place FloatTetwild_bin on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="mesh_heal_ftet_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "ftet_input.obj"
        tet_output_path = temp_dir / "ftet_output.msh"
        tracked_surface_path = Path(f"{tet_output_path}__tracked_surface.stl")
        export_temporary_triangle_mesh(mesh, input_path)

        command = [
            resolved_executable,
            "-i",
            str(input_path),
            "-o",
            str(tet_output_path),
            "--manifold-surface",
        ]
        process_report = run_external_command(command, cwd=temp_dir, status_callback=status_callback)

        if not tracked_surface_path.exists():
            raise RuntimeError("FastTetWild did not produce the expected tracked surface STL output.")

        surface_mesh = load_mesh(tracked_surface_path)
        return surface_mesh, {
            "backend": "ftetwild",
            "executable": resolved_executable,
            "input": str(input_path),
            "output_tetmesh": str(tet_output_path),
            "output_surface_stl": str(tracked_surface_path),
            "process": process_report,
        }


def run_openmeshcraft_cdt_backend(
    mesh: trimesh.Trimesh,
    executable_path: Optional[Path | str] = None,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    resolved_executable = resolve_external_executable(
        explicit_path=executable_path,
        env_var=OPENMESHCRAFT_CDT_ENV,
        candidates=OPENMESHCRAFT_CDT_CANDIDATES,
    )
    if resolved_executable is None:
        raise RuntimeError(
            "OpenMeshCraft CDT executable not found. Set OPENMESHCRAFT_CDT_EXE, pass --tetra-backend-exe, "
            "or place OpenMeshCraft-CDT on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="mesh_heal_omc_cdt_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "cdt_input.obj"
        vtk_output_path = temp_dir / "cdt_input.vtk"
        export_temporary_triangle_mesh(mesh, input_path)

        command = [resolved_executable, str(input_path), "-r"]
        process_report = run_external_command(command, cwd=temp_dir, status_callback=status_callback)

        if not vtk_output_path.exists():
            raise RuntimeError("OpenMeshCraft CDT did not produce the expected VTK output.")

        surface_mesh = load_surface_from_vtk(vtk_output_path)
        return surface_mesh, {
            "backend": "openmeshcraft-cdt",
            "executable": resolved_executable,
            "input": str(input_path),
            "output_vtk": str(vtk_output_path),
            "process": process_report,
        }


def run_advanced_exact_tetrahedral_backend(
    mesh: trimesh.Trimesh,
    advanced_backend: str,
    exact_arrangements_executable: Optional[Path | str] = None,
    tetra_backend_executable: Optional[Path | str] = None,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    backend_name = normalize_advanced_heal_backend(advanced_backend)
    if backend_name == "none":
        return mesh, {"backend": backend_name, "skipped": True}

    ensure_watertight_for_advanced_backend(mesh, stage="exact intersection")

    emit_status(status_callback, "Running exact intersection backend")
    arranged_mesh, exact_report = run_openmeshcraft_arrangements_backend(
        mesh,
        executable_path=exact_arrangements_executable,
        status_callback=status_callback,
    )
    ensure_watertight_for_advanced_backend(arranged_mesh, stage="tetrahedralization")

    emit_status(status_callback, "Running tetrahedral backend")
    if backend_name == "omc-ftetwild":
        surface_mesh, tetra_report = run_ftetwild_surface_backend(
            arranged_mesh,
            executable_path=tetra_backend_executable,
            status_callback=status_callback,
        )
    elif backend_name == "omc-cdt":
        surface_mesh, tetra_report = run_openmeshcraft_cdt_backend(
            arranged_mesh,
            executable_path=tetra_backend_executable,
            status_callback=status_callback,
        )
    else:
        raise ValueError(f"Unsupported advanced backend: {advanced_backend}")

    return surface_mesh, {
        "backend": backend_name,
        "skipped": False,
        "exact_intersection": exact_report,
        "tetrahedralization": tetra_report,
    }


def as_mesh(mesh_or_scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Scene):
        if not mesh_or_scene.geometry:
            raise ValueError("Scene is empty")
        meshes = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene contains no triangle mesh geometry")
        merged = trimesh.util.concatenate(meshes)
        return merged
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise ValueError("Unsupported mesh type")


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def face_key(vertices: np.ndarray, decimals: int = 8) -> Tuple[Tuple[float, float, float], ...]:
    pts = np.round(vertices, decimals=decimals)
    tuples = [tuple(p.tolist()) for p in pts]
    tuples.sort()
    return tuple(tuples)


def signed_polygon_area_2d(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area_2d(points: np.ndarray) -> float:
    return abs(signed_polygon_area_2d(points))


def ensure_ccw_2d(points: np.ndarray) -> np.ndarray:
    if signed_polygon_area_2d(points) < 0.0:
        return points[::-1].copy()
    return points.copy()


def bbox_overlap_2d(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
    eps: float,
) -> bool:
    return not (
        first[2] < second[0] - eps
        or second[2] < first[0] - eps
        or first[3] < second[1] - eps
        or second[3] < first[1] - eps
    )


def point_inside_halfspace_2d(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray, eps: float) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    return cross >= -eps


def line_intersection_2d(start: np.ndarray, end: np.ndarray, clip_start: np.ndarray, clip_end: np.ndarray) -> np.ndarray:
    x1, y1 = start
    x2, y2 = end
    x3, y3 = clip_start
    x4, y4 = clip_end
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if math.isclose(denom, 0.0, abs_tol=1e-20):
        return end.copy()

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
    return np.array([px, py], dtype=float)


def convex_polygon_intersection_2d(subject: np.ndarray, clip: np.ndarray, eps: float) -> np.ndarray:
    output = subject.copy()
    clip_ccw = ensure_ccw_2d(clip)
    for index in range(len(clip_ccw)):
        clip_start = clip_ccw[index]
        clip_end = clip_ccw[(index + 1) % len(clip_ccw)]
        input_points = output.copy()
        if len(input_points) == 0:
            break

        new_output: List[np.ndarray] = []
        start = input_points[-1]
        for end in input_points:
            end_inside = point_inside_halfspace_2d(end, clip_start, clip_end, eps)
            start_inside = point_inside_halfspace_2d(start, clip_start, clip_end, eps)
            if end_inside:
                if not start_inside:
                    new_output.append(line_intersection_2d(start, end, clip_start, clip_end))
                new_output.append(end.copy())
            elif start_inside:
                new_output.append(line_intersection_2d(start, end, clip_start, clip_end))
            start = end

        if not new_output:
            return np.empty((0, 2), dtype=float)
        output = np.asarray(new_output, dtype=float)

    return output


def project_points_to_2d(points: np.ndarray, drop_axis: int) -> np.ndarray:
    if drop_axis == 0:
        return points[:, [1, 2]]
    if drop_axis == 1:
        return points[:, [0, 2]]
    return points[:, [0, 1]]


def triangle_plane_bucket(
    vertices: np.ndarray,
    merge_eps: float = 1e-8,
    decimals: int = 8,
) -> Optional[Tuple[Tuple[float, float, float, float], np.ndarray, float, int]]:
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return None

    normal = normal / norm
    for component in normal:
        if math.isclose(float(component), 0.0, abs_tol=1e-15):
            continue
        if component < 0.0:
            normal = -normal
        break

    dist = -float(np.dot(normal, vertices[0]))
    dist_decimals = max(0, int(round(-math.log10(max(merge_eps, 1e-12)))))
    plane_key = (
        round(float(normal[0]), decimals),
        round(float(normal[1]), decimals),
        round(float(normal[2]), decimals),
        round(dist, dist_decimals),
    )
    drop_axis = int(np.argmax(np.abs(normal)))
    return plane_key, normal, dist, drop_axis


def unproject_points_from_plane_2d(
    points_2d: np.ndarray,
    normal: np.ndarray,
    dist: float,
    drop_axis: int,
) -> np.ndarray:
    coords_3d = np.zeros((len(points_2d), 3), dtype=float)
    if drop_axis == 0:
        coords_3d[:, 1:] = points_2d
        coords_3d[:, 0] = -(normal[1] * coords_3d[:, 1] + normal[2] * coords_3d[:, 2] + dist) / normal[0]
    elif drop_axis == 1:
        coords_3d[:, 0] = points_2d[:, 0]
        coords_3d[:, 2] = points_2d[:, 1]
        coords_3d[:, 1] = -(normal[0] * coords_3d[:, 0] + normal[2] * coords_3d[:, 2] + dist) / normal[1]
    else:
        coords_3d[:, :2] = points_2d
        coords_3d[:, 2] = -(normal[0] * coords_3d[:, 0] + normal[1] * coords_3d[:, 1] + dist) / normal[2]
    return coords_3d


def iter_polygon_components(geometry, geometry_collection_type, multi_polygon_type):
    if geometry.is_empty:
        return
    if isinstance(geometry, multi_polygon_type):
        for polygon in geometry.geoms:
            yield polygon
        return
    if isinstance(geometry, geometry_collection_type):
        for item in geometry.geoms:
            yield from iter_polygon_components(item, geometry_collection_type, multi_polygon_type)
        return
    yield geometry


def rebuild_mesh_without_triangle_overlaps(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    shapely_tools = _try_import_shapely()
    if shapely_tools is None:
        raise RuntimeError("Triangle rebuild requires shapely. Install requirements first.")

    polygon_type = shapely_tools["Polygon"]
    multi_polygon_type = shapely_tools["MultiPolygon"]
    geometry_collection_type = shapely_tools["GeometryCollection"]
    unary_union = shapely_tools["unary_union"]
    triangulate = shapely_tools["triangulate"]

    work = mesh.copy()
    work = merge_nearby_vertices(work, merge_eps=merge_eps)

    exact_faces_seen: set[Tuple[Tuple[float, float, float], ...]] = set()
    plane_buckets: Dict[
        Tuple[float, float, float, float],
        Dict[str, object],
    ] = {}
    duplicate_faces_removed = 0
    degenerate_faces_removed = 0
    accepted_faces = 0

    emit_status(status_callback, "Rebuilding mesh triangle by triangle")
    for face_index, face in enumerate(np.asarray(work.faces, dtype=np.int64)):
        triangle = np.asarray(work.vertices[face], dtype=float)
        if triangle_area(triangle[0], triangle[1], triangle[2]) <= area_eps:
            degenerate_faces_removed += 1
            continue

        dedup_key = face_key(triangle, decimals=dedup_decimals)
        if dedup_key in exact_faces_seen:
            duplicate_faces_removed += 1
            continue

        plane_bucket = triangle_plane_bucket(triangle, merge_eps=merge_eps, decimals=dedup_decimals)
        if plane_bucket is None:
            degenerate_faces_removed += 1
            continue

        plane_key, normal, dist, drop_axis = plane_bucket
        triangle_2d = ensure_ccw_2d(project_points_to_2d(triangle, drop_axis))
        triangle_polygon = polygon_type(triangle_2d)
        if triangle_polygon.is_empty or float(triangle_polygon.area) <= area_eps:
            degenerate_faces_removed += 1
            continue

        exact_faces_seen.add(dedup_key)
        accepted_faces += 1
        plane_bucket_state = plane_buckets.setdefault(
            plane_key,
            {
                "normal": normal,
                "dist": dist,
                "drop_axis": drop_axis,
                "polygons": [],
            },
        )
        plane_bucket_state["polygons"].append(triangle_polygon)

    rebuilt_vertices: List[List[float]] = []
    rebuilt_faces: List[List[int]] = []
    merged_polygons = 0

    emit_status(status_callback, "Unioning coplanar triangle patches")
    for plane_bucket_state in plane_buckets.values():
        polygons = plane_bucket_state["polygons"]
        if not polygons:
            continue

        merged_geometry = unary_union(polygons)
        normal = np.asarray(plane_bucket_state["normal"], dtype=float)
        dist = float(plane_bucket_state["dist"])
        drop_axis = int(plane_bucket_state["drop_axis"])
        polygon_tolerance = max(merge_eps, 1e-9)

        for merged_polygon in iter_polygon_components(merged_geometry, geometry_collection_type, multi_polygon_type):
            if merged_polygon.is_empty or float(merged_polygon.area) <= area_eps:
                continue
            merged_polygons += 1
            polygon_buffer = merged_polygon.buffer(polygon_tolerance)
            for tri_polygon in triangulate(merged_polygon):
                if tri_polygon.is_empty or float(tri_polygon.area) <= area_eps:
                    continue
                if not polygon_buffer.covers(tri_polygon):
                    continue

                coords_2d = np.asarray(tri_polygon.exterior.coords[:-1], dtype=float)
                if coords_2d.shape != (3, 2):
                    continue
                coords_3d = unproject_points_from_plane_2d(coords_2d, normal=normal, dist=dist, drop_axis=drop_axis)
                if triangle_area(coords_3d[0], coords_3d[1], coords_3d[2]) <= area_eps:
                    continue

                tri_normal = np.cross(coords_3d[1] - coords_3d[0], coords_3d[2] - coords_3d[0])
                if float(np.dot(tri_normal, normal)) < 0.0:
                    coords_3d = coords_3d[[0, 2, 1]]

                base = len(rebuilt_vertices)
                rebuilt_vertices.extend(coords_3d.tolist())
                rebuilt_faces.append([base, base + 1, base + 2])

    if rebuilt_faces:
        rebuilt_mesh = trimesh.Trimesh(
            vertices=np.asarray(rebuilt_vertices, dtype=float),
            faces=np.asarray(rebuilt_faces, dtype=np.int64),
            process=False,
        )
        rebuilt_mesh = merge_nearby_vertices(rebuilt_mesh, merge_eps=merge_eps)
        rebuilt_mesh = remove_duplicate_faces(rebuilt_mesh, decimals=dedup_decimals)
        rebuilt_mesh = remove_degenerate_faces(rebuilt_mesh, eps=area_eps)
    else:
        rebuilt_mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )

    return rebuilt_mesh, {
        "input_faces": int(len(work.faces)),
        "accepted_faces": int(accepted_faces),
        "output_faces": int(len(rebuilt_mesh.faces)),
        "duplicate_faces_removed": int(duplicate_faces_removed),
        "degenerate_faces_removed": int(degenerate_faces_removed),
        "coplanar_groups_processed": int(len(plane_buckets)),
        "merged_polygons": int(merged_polygons),
    }


def detect_duplicate_faces(mesh: trimesh.Trimesh, decimals: int = 8) -> Tuple[int, Dict[Tuple, List[int]]]:
    groups: Dict[Tuple, List[int]] = {}
    for i, face in enumerate(mesh.faces):
        key = face_key(mesh.vertices[face], decimals=decimals)
        groups.setdefault(key, []).append(i)
    dup_count = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    return dup_count, groups


def detect_degenerate_faces(mesh: trimesh.Trimesh, eps: float = 1e-12) -> np.ndarray:
    bad = []
    for i, face in enumerate(mesh.faces):
        v0, v1, v2 = mesh.vertices[face]
        if triangle_area(v0, v1, v2) <= eps:
            bad.append(i)
    return np.array(bad, dtype=np.int64)


def remove_degenerate_faces(mesh: trimesh.Trimesh, eps: float = 1e-12) -> trimesh.Trimesh:
    degenerate = detect_degenerate_faces(mesh, eps=eps)
    if len(degenerate) == 0:
        return mesh

    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[degenerate] = False
    return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[keep_mask].copy(), process=False)


def merge_nearby_vertices(mesh: trimesh.Trimesh, merge_eps: float = 1e-8) -> trimesh.Trimesh:
    if merge_eps <= 0.0:
        raise ValueError("merge_eps must be greater than zero.")

    digits = max(0, int(round(-math.log10(merge_eps))))
    mesh.merge_vertices(digits_vertex=digits)
    mesh.remove_unreferenced_vertices()
    return mesh


def detect_edge_stats(mesh: trimesh.Trimesh) -> Tuple[int, int]:
    unique_edges = mesh.edges_unique
    if unique_edges.shape[0] == 0:
        return 0, 0
    edge_inv = mesh.edges_unique_inverse
    counts = np.bincount(edge_inv, minlength=unique_edges.shape[0])
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return nonmanifold, boundary


def mesh_report(mesh: trimesh.Trimesh, area_eps: float = 1e-12) -> MeshReport:
    deg = detect_degenerate_faces(mesh, eps=area_eps)
    dup, _ = detect_duplicate_faces(mesh)
    nonmanifold, boundary = detect_edge_stats(mesh)
    return MeshReport(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        volume=float(mesh.volume) if np.isfinite(mesh.volume) else float("nan"),
        area=float(mesh.area),
        degenerate_faces=int(len(deg)),
        duplicate_faces=int(dup),
        nonmanifold_edges=nonmanifold,
        boundary_edges=boundary,
    )


def emit_mesh_topology_status(status_callback: StatusCallback, label: str, report: MeshReport) -> None:
    if report.watertight:
        emit_status(status_callback, f"{label} surface is closed/watertight (boundary edges: {report.boundary_edges})")
        return

    emit_status(status_callback, f"{label} surface is open (boundary edges: {report.boundary_edges})")


def attempt_watertight_repair(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    before = mesh_report(mesh, area_eps=area_eps)
    if before.watertight:
        emit_status(status_callback, "Mesh is already closed/watertight; skipping watertight repair")
        return mesh, {
            "requested": True,
            "skipped": True,
            "reason": "already_watertight",
            "before": asdict(before),
            "after": asdict(before),
        }

    emit_status(status_callback, "Attempting watertight repair")
    components = split_disconnected_components(mesh)
    if len(components) > 1:
        emit_status(status_callback, f"Attempting watertight repair on {len(components)} disconnected solids")

    repaired_components: List[trimesh.Trimesh] = []
    for index, component in enumerate(components, start=1):
        work = component.copy()
        if len(components) > 1:
            emit_status(status_callback, f"Sealing solid {index}/{len(components)}")
        try:
            work = heal_with_meshfix(work)
        except Exception as exc:
            emit_status(status_callback, f"Watertight MeshFix stage skipped for solid {index} ({exc})")

        try:
            trimesh.repair.fill_holes(work)
        except Exception as exc:
            emit_status(status_callback, f"Hole-filling stage skipped for solid {index} ({exc})")

        work = merge_nearby_vertices(work, merge_eps=merge_eps)
        work = remove_duplicate_faces(work, decimals=dedup_decimals)
        work = remove_degenerate_faces(work, eps=area_eps)
        work.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(work)
        repaired_components.append(work)

    if not repaired_components:
        combined = make_empty_mesh()
    elif len(repaired_components) == 1:
        combined = repaired_components[0]
    else:
        combined = combine_meshes(repaired_components)

    after = mesh_report(combined, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Watertight repair output", after)
    return combined, {
        "requested": True,
        "skipped": False,
        "before": asdict(before),
        "after": asdict(after),
        "component_count": int(len(components)),
        "made_watertight": bool(after.watertight),
        "boundary_edges_removed": int(before.boundary_edges - after.boundary_edges),
    }


def make_empty_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.empty((0, 3), dtype=float),
        faces=np.empty((0, 3), dtype=np.int64),
        process=False,
    )


def bbox_overlap_3d(first_min: np.ndarray, first_max: np.ndarray, second_min: np.ndarray, second_max: np.ndarray, eps: float) -> bool:
    return not (
        first_max[0] < second_min[0] - eps
        or second_max[0] < first_min[0] - eps
        or first_max[1] < second_min[1] - eps
        or second_max[1] < first_min[1] - eps
        or first_max[2] < second_min[2] - eps
        or second_max[2] < first_min[2] - eps
    )


def estimate_local_intersection_edge_length(mesh: trimesh.Trimesh, ratio: float = 0.01) -> float:
    if len(mesh.vertices) == 0:
        return 0.0
    bounds = np.asarray(mesh.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return diagonal * ratio


def preprocess_for_localized_intersections(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    target_edge_length: Optional[float] = None,
    max_subdivide_iter: int = 2,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    work = mesh.copy()
    report = {
        "input_faces": int(len(work.faces)),
        "input_vertices": int(len(work.vertices)),
        "target_edge_length": 0.0,
        "max_edge_length": 0.0,
        "subdivided": False,
        "output_faces": int(len(work.faces)),
        "output_vertices": int(len(work.vertices)),
    }
    if len(work.faces) == 0:
        return work, report

    if target_edge_length is None or target_edge_length <= 0.0:
        target_edge_length = estimate_local_intersection_edge_length(work)
    if target_edge_length <= 0.0:
        return work, report

    max_edge_length = (4.0 / 3.0) * target_edge_length
    report["target_edge_length"] = float(target_edge_length)
    report["max_edge_length"] = float(max_edge_length)

    edges_unique = np.asarray(work.edges_unique, dtype=np.int64)
    if len(edges_unique) == 0:
        return work, report

    edge_lengths = np.linalg.norm(work.vertices[edges_unique[:, 0]] - work.vertices[edges_unique[:, 1]], axis=1)
    if len(edge_lengths) == 0 or float(edge_lengths.max()) <= max_edge_length:
        return work, report

    emit_status(status_callback, "Preprocessing mesh for localized intersection repair")
    try:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            np.asarray(work.vertices, dtype=float),
            np.asarray(work.faces, dtype=np.int64),
            max_edge=max_edge_length,
            max_iter=max_subdivide_iter,
        )
    except ValueError as exc:
        report["subdivide_error"] = str(exc)
        return work, report
    work = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)
    report["subdivided"] = True
    report["output_faces"] = int(len(work.faces))
    report["output_vertices"] = int(len(work.vertices))
    return work, report


def build_spatial_candidate_face_pairs(
    mesh: trimesh.Trimesh,
    cell_size: float,
    max_cells_per_face: int = 4096,
) -> List[Tuple[int, int]]:
    triangles = np.asarray(mesh.triangles, dtype=float)
    if len(triangles) == 0:
        return []

    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    origin = mins.min(axis=0)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    pairs: set[Tuple[int, int]] = set()

    for face_index in range(len(triangles)):
        min_key = np.floor((mins[face_index] - origin) / cell_size).astype(int)
        max_key = np.floor((maxs[face_index] - origin) / cell_size).astype(int)
        span = (max_key - min_key) + 1
        cell_count = int(np.prod(span))
        if cell_count > max_cells_per_face:
            center_key = tuple(np.floor((((mins[face_index] + maxs[face_index]) * 0.5) - origin) / cell_size).astype(int).tolist())
            bucket = buckets.setdefault(center_key, [])
            for other_face in bucket:
                pairs.add((other_face, face_index) if other_face < face_index else (face_index, other_face))
            bucket.append(face_index)
            continue

        for x in range(int(min_key[0]), int(max_key[0]) + 1):
            for y in range(int(min_key[1]), int(max_key[1]) + 1):
                for z in range(int(min_key[2]), int(max_key[2]) + 1):
                    key = (x, y, z)
                    bucket = buckets.setdefault(key, [])
                    for other_face in bucket:
                        pairs.add((other_face, face_index) if other_face < face_index else (face_index, other_face))
                    bucket.append(face_index)

    return sorted(pairs)


def triangle_polydata(vertices: np.ndarray):
    pv = _try_import_pyvista()
    if pv is None:
        raise RuntimeError("Localized intersection repair requires pyvista. Install requirements first.")
    return pv.PolyData(np.asarray(vertices, dtype=float), np.array([3, 0, 1, 2], dtype=np.int64))


def triangles_intersect_exact(triangle_a: np.ndarray, triangle_b: np.ndarray, eps: float = 1e-9) -> bool:
    poly_a = triangle_polydata(triangle_a)
    poly_b = triangle_polydata(triangle_b)
    try:
        _, collision_count = poly_a.collision(
            poly_b,
            contact_mode=0,
            box_tolerance=max(eps, 1e-9),
            cell_tolerance=0.0,
        )
    except Exception:
        return False
    return collision_count > 0


def detect_self_intersection_pairs(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    cell_size: Optional[float] = None,
    max_candidate_pairs: int = 25000,
    status_callback: StatusCallback = None,
) -> Tuple[List[Tuple[int, int]], dict]:
    if _try_import_pyvista() is None:
        raise RuntimeError("Localized intersection repair requires pyvista. Install requirements first.")

    work = mesh.copy()
    work.remove_unreferenced_vertices()
    if len(work.faces) == 0:
        return [], {
            "candidate_pairs": 0,
            "intersecting_pairs": 0,
            "intersecting_faces": 0,
            "skipped": False,
            "cell_size": 0.0,
        }

    if cell_size is None or cell_size <= 0.0:
        cell_size = max(estimate_local_intersection_edge_length(work), merge_eps * 10.0)
    cell_size = max(cell_size, merge_eps * 10.0, 1e-9)

    emit_status(status_callback, "Detecting localized self-intersections")
    candidate_pairs = build_spatial_candidate_face_pairs(work, cell_size=cell_size)
    report = {
        "candidate_pairs": int(len(candidate_pairs)),
        "intersecting_pairs": 0,
        "intersecting_faces": 0,
        "skipped": False,
        "cell_size": float(cell_size),
    }
    if len(candidate_pairs) > max_candidate_pairs:
        report["skipped"] = True
        report["skip_reason"] = f"candidate pair count {len(candidate_pairs)} exceeded limit {max_candidate_pairs}"
        return [], report

    faces = np.asarray(work.faces, dtype=np.int64)
    triangles = np.asarray(work.triangles, dtype=float)
    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    intersecting_pairs: List[Tuple[int, int]] = []

    for first_face, second_face in candidate_pairs:
        if first_face == second_face:
            continue
        if np.intersect1d(faces[first_face], faces[second_face]).size > 0:
            continue
        if not bbox_overlap_3d(mins[first_face], maxs[first_face], mins[second_face], maxs[second_face], merge_eps):
            continue
        if triangles_intersect_exact(triangles[first_face], triangles[second_face], eps=merge_eps):
            intersecting_pairs.append((int(first_face), int(second_face)))

    intersecting_faces = {face_index for pair in intersecting_pairs for face_index in pair}
    report["intersecting_pairs"] = int(len(intersecting_pairs))
    report["intersecting_faces"] = int(len(intersecting_faces))
    return intersecting_pairs, report


def partition_intersecting_face_components(
    mesh: trimesh.Trimesh,
    intersecting_faces: Sequence[int],
    intersecting_pairs: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    if not intersecting_faces:
        return []

    remaining = set(int(face_index) for face_index in intersecting_faces)
    vertex_to_faces: Dict[int, List[int]] = defaultdict(list)
    face_neighbors: Dict[int, set[int]] = defaultdict(set)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for face_index in remaining:
        for vertex_index in faces[face_index]:
            vertex_to_faces[int(vertex_index)].append(face_index)
    for first_face, second_face in intersecting_pairs:
        face_neighbors[int(first_face)].add(int(second_face))
        face_neighbors[int(second_face)].add(int(first_face))

    components: List[List[int]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            face_index = stack.pop()
            for vertex_index in faces[face_index]:
                for neighbor_face in vertex_to_faces[int(vertex_index)]:
                    if neighbor_face in remaining:
                        remaining.remove(neighbor_face)
                        stack.append(neighbor_face)
                        component.append(neighbor_face)
            for neighbor_face in face_neighbors.get(face_index, ()):
                if neighbor_face in remaining:
                    remaining.remove(neighbor_face)
                    stack.append(neighbor_face)
                    component.append(neighbor_face)
        components.append(sorted(component))

    return components


def extract_face_submesh(mesh: trimesh.Trimesh, face_indices: Sequence[int]) -> trimesh.Trimesh:
    if not face_indices:
        return make_empty_mesh()
    submesh = mesh.submesh([np.asarray(sorted(face_indices), dtype=np.int64)], append=True, repair=False)
    extracted = as_mesh(submesh)
    extracted.remove_unreferenced_vertices()
    return extracted


def remove_mesh_faces(mesh: trimesh.Trimesh, face_indices: Sequence[int]) -> trimesh.Trimesh:
    if len(mesh.faces) == 0 or not face_indices:
        return mesh.copy()

    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[np.asarray(sorted(set(int(face_index) for face_index in face_indices)), dtype=np.int64)] = False
    if not np.any(keep_mask):
        return make_empty_mesh()

    trimmed = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float).copy(),
        faces=np.asarray(mesh.faces[keep_mask], dtype=np.int64).copy(),
        process=False,
    )
    trimmed.remove_unreferenced_vertices()
    return trimmed


def repair_localized_self_intersections(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    preprocessed_mesh, preprocessing_report = preprocess_for_localized_intersections(
        mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
    )
    intersecting_pairs, detection_report = detect_self_intersection_pairs(
        preprocessed_mesh,
        merge_eps=merge_eps,
        status_callback=status_callback,
    )
    if detection_report.get("skipped"):
        return preprocessed_mesh, {
            "preprocessing": preprocessing_report,
            "detection": detection_report,
            "components": [],
            "repaired_components": 0,
            "skipped": True,
        }

    intersecting_faces = sorted({face_index for pair in intersecting_pairs for face_index in pair})
    components = partition_intersecting_face_components(preprocessed_mesh, intersecting_faces, intersecting_pairs)
    if not components:
        return preprocessed_mesh, {
            "preprocessing": preprocessing_report,
            "detection": detection_report,
            "components": [],
            "repaired_components": 0,
            "skipped": False,
        }

    repaired_components: List[trimesh.Trimesh] = []
    component_reports = []
    for index, component_faces in enumerate(components, start=1):
        emit_status(status_callback, f"Repairing localized intersection region {index}/{len(components)}")
        component_mesh = extract_face_submesh(preprocessed_mesh, component_faces)
        rebuild_report = None
        working_component = component_mesh
        try:
            working_component, rebuild_report = rebuild_mesh_without_triangle_overlaps(
                component_mesh,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=None,
            )
        except Exception as exc:
            rebuild_report = {"skipped": True, "reason": str(exc)}

        try:
            repaired_mesh = heal_mesh(
                working_component,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=None,
            )
        except Exception as exc:
            repaired_mesh = component_mesh.copy()
            repaired_mesh.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(repaired_mesh)
            component_reports.append(
                {
                    "component": index,
                    "input_faces": int(len(component_mesh.faces)),
                    "output_faces": int(len(repaired_mesh.faces)),
                    "rebuild": rebuild_report,
                    "fallback_reason": str(exc),
                }
            )
            repaired_components.append(repaired_mesh)
            continue

        if len(repaired_mesh.faces) == 0:
            repaired_mesh = component_mesh.copy()
            repaired_mesh.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(repaired_mesh)

        component_reports.append(
            {
                "component": index,
                "input_faces": int(len(component_mesh.faces)),
                "output_faces": int(len(repaired_mesh.faces)),
                "rebuild": rebuild_report,
            }
        )
        repaired_components.append(repaired_mesh)

    untouched_mesh = remove_mesh_faces(preprocessed_mesh, intersecting_faces)
    meshes_to_combine = [untouched_mesh, *repaired_components]
    non_empty_meshes = [candidate for candidate in meshes_to_combine if len(candidate.faces) > 0]
    if not non_empty_meshes:
        combined = make_empty_mesh()
    elif len(non_empty_meshes) == 1:
        combined = non_empty_meshes[0]
    else:
        combined = combine_meshes(non_empty_meshes)

    combined = finalize_healed_mesh(
        combined,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=True,
        status_callback=status_callback,
    )
    return combined, {
        "preprocessing": preprocessing_report,
        "detection": detection_report,
        "components": component_reports,
        "repaired_components": int(len(repaired_components)),
        "skipped": False,
    }


def load_dxf_as_trimesh(path: Path) -> trimesh.Trimesh:
    ezdxf = _try_import_ezdxf()
    if ezdxf is None:
        raise RuntimeError("DXF support requires ezdxf. Install requirements first.")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    saw_polyline = False

    def add_triangle(p0, p1, p2):
        base = len(vertices)
        vertices.extend([
            [float(p0[0]), float(p0[1]), float(p0[2])],
            [float(p1[0]), float(p1[1]), float(p1[2])],
            [float(p2[0]), float(p2[1]), float(p2[2])],
        ])
        faces.append([base, base + 1, base + 2])

    def add_polyface(polyline) -> None:
        poly_vertices, poly_faces = polyline.indexed_faces()
        index_by_location = {
            tuple(float(coord) for coord in vertex.dxf.location): index
            for index, vertex in enumerate(poly_vertices)
        }
        base = len(vertices)
        vertices.extend([list(map(float, vertex.dxf.location)) for vertex in poly_vertices])
        for face in poly_faces:
            face_indices = [index_by_location[tuple(float(coord) for coord in vertex.dxf.location)] for vertex in face]
            if len(face_indices) < 3:
                continue
            for offset in range(1, len(face_indices) - 1):
                faces.append([
                    base + face_indices[0],
                    base + face_indices[offset],
                    base + face_indices[offset + 1],
                ])

    for e in msp:
        t = e.dxftype()
        if t == "3DFACE":
            pts = [
                e.dxf.vtx0,
                e.dxf.vtx1,
                e.dxf.vtx2,
                e.dxf.vtx3,
            ]
            p0, p1, p2, p3 = pts
            add_triangle(p0, p1, p2)
            if not (
                math.isclose(p2[0], p3[0], rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(p2[1], p3[1], rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(p2[2], p3[2], rel_tol=0.0, abs_tol=1e-12)
            ):
                add_triangle(p0, p2, p3)
        elif t == "POLYLINE":
            saw_polyline = True
            try:
                if e.is_poly_face_mesh:
                    add_polyface(e)
                    continue
                if e.is_polygon_mesh:
                    continue
            except Exception:
                pass

    if not faces:
        if saw_polyline:
            raise ValueError(
                "DXF does not contain triangulated 3DFACE entities. Found POLYLINE entities, which are not "
                "supported yet. Export triangulated 3DFACE or use STL/OBJ/PLY/MSH instead."
            )
        raise ValueError("No triangulated 3DFACE entities found in DXF.")

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    return mesh


def load_leapfrog_msh(path: Path) -> trimesh.Trimesh:
    raw = path.read_bytes()
    marker = b"[binary]"
    marker_index = raw.find(marker)
    if marker_index < 0:
        raise ValueError("Unsupported .msh file: missing [binary] section.")

    binary_offset = marker_index + len(marker)
    while binary_offset < len(raw) and raw[binary_offset] in (9, 10, 13, 32):
        binary_offset += 1

    header = raw[:marker_index].decode("ascii", errors="replace")
    loc_match = re.search(r"Location\s+Double\s+3\s+(\d+)\s*;", header)
    tri_match = re.search(r"Tri\s+Integer\s+3\s+(\d+)\s*;", header)
    if loc_match is None or tri_match is None:
        raise ValueError("Unsupported .msh file: missing Location/Tri index entries.")

    vertex_count = int(loc_match.group(1))
    face_count = int(tri_match.group(1))

    vertex_values = vertex_count * 3
    face_values = face_count * 3
    vertex_bytes = vertex_values * 8
    face_bytes = face_values * 4
    expected_payload = vertex_bytes + face_bytes
    extra_bytes = len(raw) - binary_offset - expected_payload
    if extra_bytes < 0:
        raise ValueError("Incomplete .msh file: binary payload is shorter than header declares.")

    vertices_offset = binary_offset + extra_bytes
    vertices = np.frombuffer(raw, dtype="<f8", count=vertex_values, offset=vertices_offset).reshape(vertex_count, 3)
    faces_offset = vertices_offset + vertex_bytes
    faces = np.frombuffer(raw, dtype="<i4", count=face_values, offset=faces_offset).reshape(face_count, 3)

    if faces.size == 0 or vertices.size == 0:
        raise ValueError("Empty .msh mesh data.")

    min_index = int(faces.min())
    max_index = int(faces.max())
    if min_index == 1 and max_index == vertex_count:
        faces = faces - 1
    elif min_index < 0 or max_index >= vertex_count:
        raise ValueError("Unsupported .msh file: triangle indices are out of range.")

    return trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)


def load_mesh(path: Path) -> trimesh.Trimesh:
    ext = path.suffix.lower()
    if ext == ".dxf":
        return load_dxf_as_trimesh(path)
    if ext == ".msh":
        return load_leapfrog_msh(path)
    if ext == ".00t":
        raise ValueError(
            "Maptek Vulcan .00t is not supported in this build. A proprietary Maptek SDK or documented file "
            "specification is required to load it safely."
        )
    loaded = trimesh.load(str(path), force="mesh")
    return as_mesh(loaded)


def _coerce_path_list(paths: Path | Sequence[Path], label: str) -> List[Path]:
    if isinstance(paths, Path):
        normalized = [paths]
    else:
        normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError(f"At least one {label} mesh is required.")
    return normalized


def combine_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("At least one mesh is required.")
    if len(meshes) == 1:
        return meshes[0]
    combined = trimesh.util.concatenate([mesh.copy() for mesh in meshes])
    combined = as_mesh(combined)
    combined.remove_unreferenced_vertices()
    return combined


def load_meshes(paths: Path | Sequence[Path], label: str, status_callback: StatusCallback = None) -> Tuple[List[Path], trimesh.Trimesh]:
    normalized_paths = _coerce_path_list(paths, label)
    meshes = []
    if len(normalized_paths) == 1:
        emit_status(status_callback, f"Loading {label} mesh {normalized_paths[0].name}")
    else:
        emit_status(status_callback, f"Loading {len(normalized_paths)} {label} meshes")
    for index, path in enumerate(normalized_paths, start=1):
        if len(normalized_paths) > 1:
            emit_status(status_callback, f"Loading {label} mesh {index}/{len(normalized_paths)}: {path.name}")
        meshes.append(load_mesh(path))
    if len(meshes) > 1:
        emit_status(status_callback, f"Combining {len(meshes)} {label} meshes")
    return normalized_paths, combine_meshes(meshes)


def load_individual_meshes(
    paths: Path | Sequence[Path],
    label: str,
    status_callback: StatusCallback = None,
) -> Tuple[List[Path], List[trimesh.Trimesh]]:
    normalized_paths = _coerce_path_list(paths, label)
    meshes: List[trimesh.Trimesh] = []
    emit_status(status_callback, f"Loading {len(normalized_paths)} {label} meshes")
    for index, path in enumerate(normalized_paths, start=1):
        emit_status(status_callback, f"Loading {label} mesh {index}/{len(normalized_paths)}: {path.name}")
        meshes.append(load_mesh(path))
    return normalized_paths, meshes


def prepare_boolean_meshes(
    meshes: Sequence[trimesh.Trimesh],
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> List[trimesh.Trimesh]:
    prepared_meshes: List[trimesh.Trimesh] = []
    for index, mesh in enumerate(meshes, start=1):
        emit_status(status_callback, f"Preparing solid {index}/{len(meshes)}")
        prepared_meshes.append(
            heal_mesh(
                mesh,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=None,
            )
        )
    return prepared_meshes


def run_iterative_boolean_operation(
    meshes: Sequence[trimesh.Trimesh],
    operation: str,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> trimesh.Trimesh:
    operation_name = normalize_boolean_operation(operation)
    if len(meshes) < 2:
        raise ValueError("At least two solids are required for boolean operations.")

    result_mesh = meshes[0].copy()
    for index, next_mesh in enumerate(meshes[1:], start=2):
        emit_status(status_callback, f"Applying {operation_name} with solid {index}/{len(meshes)}")
        result_mesh = run_boolean_operation(
            result_mesh,
            next_mesh,
            operation=operation_name,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
    return result_mesh


def export_dxf_polyface(mesh: trimesh.Trimesh, path: Path) -> None:
    ezdxf = _try_import_ezdxf()
    if ezdxf is None:
        raise RuntimeError("DXF export requires ezdxf. Install requirements first.")

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    polyface = msp.add_polyface()
    polyface.append_faces(
        (
            [
                tuple(map(float, vertices[face[0]])),
                tuple(map(float, vertices[face[1]])),
                tuple(map(float, vertices[face[2]])),
            ]
            for face in faces
        )
    )
    doc.saveas(str(path))


def export_leapfrog_msh(mesh: trimesh.Trimesh, path: Path) -> None:
    vertices = np.asarray(mesh.vertices, dtype="<f8")
    faces = np.asarray(mesh.faces, dtype="<i4")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Cannot export an empty mesh.")

    header = (
        "%ARANZ-1.0\n\n"
        "[index]\n"
        f"Location Double 3 {len(vertices)};\n"
        f"Tri Integer 3 {len(faces)};\n\n"
        "[binary]"
    ).encode("ascii")
    payload = b"".join(
        [
            header,
            LEAPFROG_MSH_PREAMBLE,
            vertices.tobytes(order="C"),
            faces.tobytes(order="C"),
        ]
    )
    path.write_bytes(payload)


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".dxf":
        export_dxf_polyface(mesh, path)
        return
    if ext == ".msh":
        export_leapfrog_msh(mesh, path)
        return
    if ext == ".00t":
        raise ValueError(
            "Maptek Vulcan .00t export is not supported in this build. A proprietary Maptek SDK or documented file "
            "specification is required to write it safely."
        )
    mesh.export(str(path))


def mesh_to_polydata(mesh: trimesh.Trimesh):
    pv = _try_import_pyvista()
    if pv is None:
        raise RuntimeError("Boolean operations require pyvista to be installed.")

    faces = np.hstack([
        np.full((len(mesh.faces), 1), 3, dtype=np.int64),
        np.asarray(mesh.faces, dtype=np.int64),
    ]).reshape(-1)
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)
    return poly.triangulate().clean()


def polydata_to_mesh(poly) -> trimesh.Trimesh:
    surface = poly.extract_surface(algorithm="dataset_surface").triangulate().clean()
    points = np.asarray(surface.points)
    faces = np.asarray(surface.faces)
    if len(points) == 0 or len(faces) == 0:
        raise ValueError("Boolean operation produced an empty mesh.")
    tri_faces = faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=points.copy(), faces=tri_faces.copy(), process=False)


def heal_with_pyvista(mesh: trimesh.Trimesh, merge_eps: float = 1e-8) -> trimesh.Trimesh:
    pv = _try_import_pyvista()
    if pv is None:
        return mesh

    poly = mesh_to_polydata(mesh)
    cleaned = poly.clean(tolerance=merge_eps, absolute=True)
    if cleaned.n_points == 0 or cleaned.n_cells == 0:
        return mesh
    return polydata_to_mesh(cleaned)


def remove_duplicate_faces(mesh: trimesh.Trimesh, decimals: int = 8) -> trimesh.Trimesh:
    _, groups = detect_duplicate_faces(mesh, decimals=decimals)
    keep = []
    for idxs in groups.values():
        keep.append(idxs[0])
    keep = np.array(sorted(keep), dtype=np.int64)
    new = mesh.submesh([keep], append=True, repair=False)
    return as_mesh(new)


def union_watertight_components(
    components: Sequence[trimesh.Trimesh],
    status_callback: StatusCallback = None,
) -> trimesh.Trimesh:
    if not components:
        raise ValueError("At least one mesh component is required.")
    if len(components) == 1:
        return components[0]

    normalized_components = [as_mesh(component.copy()) for component in components if len(component.faces) > 0]
    if len(normalized_components) <= 1:
        return combine_meshes(normalized_components)
    if not all(component.is_watertight for component in normalized_components):
        emit_status(status_callback, "Skipping overlap-resolution union because at least one healed solid is open")
        return combine_meshes(normalized_components)

    try:
        unioned = trimesh.boolean.union([component.copy() for component in normalized_components], engine="manifold")
    except Exception as exc:
        emit_status(status_callback, f"Overlap-resolution union skipped ({exc})")
        return combine_meshes(normalized_components)

    result = as_mesh(unioned)
    result.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(result)
    return result


def finalize_healed_mesh(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    resolve_component_overlaps: bool = False,
    status_callback: StatusCallback = None,
) -> trimesh.Trimesh:
    work = mesh.copy()

    emit_status(status_callback, "Finalizing healed mesh")
    try:
        work = heal_with_pyvista(work, merge_eps=merge_eps)
    except Exception as exc:
        emit_status(status_callback, f"PyVista cleanup skipped ({exc})")
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)

    if resolve_component_overlaps:
        components = split_disconnected_components(work)
        if len(components) > 1:
            emit_status(status_callback, f"Resolving overlap across {len(components)} healed solids")
            work = union_watertight_components(components, status_callback=status_callback)
            work = merge_nearby_vertices(work, merge_eps=merge_eps)
            work = remove_duplicate_faces(work, decimals=dedup_decimals)
            work = remove_degenerate_faces(work, eps=area_eps)

    work.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(work)
    return work


def heal_with_open3d(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    o3d = _try_import_open3d()
    if o3d is None:
        return mesh

    omesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )

    omesh.remove_duplicated_vertices()
    omesh.remove_degenerate_triangles()
    omesh.remove_duplicated_triangles()
    omesh.remove_non_manifold_edges()
    omesh.compute_triangle_normals()
    omesh.compute_vertex_normals()

    v = np.asarray(omesh.vertices)
    f = np.asarray(omesh.triangles)
    if len(v) == 0 or len(f) == 0:
        return mesh

    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def heal_with_meshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    pymeshfix = _try_import_pymeshfix()
    if pymeshfix is None:
        return mesh

    cleaner = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    try:
        cleaner.clean()
    except Exception:
        pass
    try:
        cleaner.join_closest_components()
    except Exception:
        pass
    try:
        cleaner.repair(joincomp=True, remove_smallest_components=False)
    except TypeError:
        cleaner.repair(verbose=False, joincomp=True, remove_smallest_components=False)
    except Exception:
        try:
            cleaner.repair(joincomp=False, remove_smallest_components=False)
        except TypeError:
            cleaner.repair(verbose=False, joincomp=False, remove_smallest_components=False)
    v = getattr(cleaner, "points", getattr(cleaner, "v", None))
    f = getattr(cleaner, "faces", getattr(cleaner, "f", None))
    if v is None or f is None or len(v) == 0 or len(f) == 0:
        return mesh
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def split_disconnected_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    work = mesh.copy()
    work.remove_unreferenced_vertices()
    if len(work.faces) == 0:
        return [work]

    parts = work.split(only_watertight=False)
    components: List[trimesh.Trimesh] = []
    for part in parts:
        component = as_mesh(part)
        if len(component.faces) == 0:
            continue
        component.remove_unreferenced_vertices()
        components.append(component)

    return components or [work]


def estimate_heal_mesh_steps(mesh: trimesh.Trimesh) -> int:
    preprocessed_mesh = preprocess_heal_mesh(mesh)
    component_count = len(split_disconnected_components(preprocessed_mesh))
    recombine_steps = 1 if component_count > 1 else 0
    return HEAL_PREPROCESS_STEP_COUNT + component_count * HEAL_COMPONENT_STEP_COUNT + recombine_steps


def preprocess_heal_mesh(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_tracker: Optional[ProgressTracker] = None,
) -> trimesh.Trimesh:
    work = mesh.copy()

    emit_status(status_callback, "Removing duplicate triangles")
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    if progress_tracker is not None:
        progress_tracker.advance()

    emit_status(status_callback, "Removing degenerate triangles")
    work = remove_degenerate_faces(work, eps=area_eps)
    if progress_tracker is not None:
        progress_tracker.advance()

    emit_status(status_callback, "Merging nearby vertices")
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    if progress_tracker is not None:
        progress_tracker.advance()

    return work


def heal_single_mesh(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_tracker: Optional[ProgressTracker] = None,
) -> trimesh.Trimesh:
    work = mesh.copy()

    emit_status(status_callback, "Running Open3D cleanup")
    try:
        work = heal_with_open3d(work)
    except Exception as exc:
        emit_status(status_callback, f"Open3D cleanup skipped ({exc})")
    if progress_tracker is not None:
        progress_tracker.advance()

    emit_status(status_callback, "Running MeshFix repair")
    try:
        work = heal_with_meshfix(work)
    except Exception as exc:
        emit_status(status_callback, f"MeshFix repair skipped ({exc})")
    if progress_tracker is not None:
        progress_tracker.advance()

    emit_status(status_callback, "Fixing normals and filling holes")
    trimesh.repair.fix_normals(work)
    trimesh.repair.fill_holes(work)
    work.remove_unreferenced_vertices()
    if progress_tracker is not None:
        progress_tracker.advance()

    return work


def heal_mesh(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_tracker: Optional[ProgressTracker] = None,
) -> trimesh.Trimesh:
    preprocessed_mesh = preprocess_heal_mesh(
        mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
        progress_tracker=progress_tracker,
    )
    components = split_disconnected_components(preprocessed_mesh)
    if len(components) == 1:
        healed = heal_single_mesh(
            components[0],
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
            progress_tracker=progress_tracker,
        )
        return finalize_healed_mesh(
            healed,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            resolve_component_overlaps=False,
            status_callback=status_callback,
        )

    emit_status(status_callback, f"Detected {len(components)} disconnected solids")
    healed_components: List[trimesh.Trimesh] = []
    for index, component in enumerate(components, start=1):
        emit_status(status_callback, f"Healing solid {index}/{len(components)}")
        try:
            healed_component = heal_single_mesh(
                component,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=status_callback,
                progress_tracker=progress_tracker,
            )
        except Exception as exc:
            emit_status(status_callback, f"Solid {index} repair failed ({exc}); keeping the original solid")
            healed_component = component.copy()
            healed_component.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(healed_component)

        if len(healed_component.faces) == 0:
            emit_status(status_callback, f"Solid {index} became empty after repair; keeping the original solid")
            healed_component = component.copy()
            healed_component.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(healed_component)

        healed_components.append(healed_component)

    emit_status(status_callback, f"Recombining {len(healed_components)} healed solids")
    if progress_tracker is not None:
        progress_tracker.advance()
    combined = combine_meshes(healed_components)
    return finalize_healed_mesh(
        combined,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=True,
        status_callback=status_callback,
    )


def boolean_meshes(
    left: trimesh.Trimesh,
    right: trimesh.Trimesh,
    operation: str,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> trimesh.Trimesh:
    operation_name = normalize_boolean_operation(operation)

    emit_status(status_callback, "Preparing left solid")
    left_ready = heal_mesh(
        left,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=None,
    )
    emit_status(status_callback, "Preparing right solid")
    right_ready = heal_mesh(
        right,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=None,
    )

    return run_boolean_operation(
        left_ready,
        right_ready,
        operation=operation_name,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
    )


def run_boolean_operation(
    left_ready: trimesh.Trimesh,
    right_ready: trimesh.Trimesh,
    operation: str,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> trimesh.Trimesh:
    operation_name = normalize_boolean_operation(operation)

    emit_status(status_callback, "Converting solids for boolean kernel")
    try:
        emit_status(status_callback, f"Running {operation_name} boolean with manifold")
        if operation_name == "union":
            result_mesh = trimesh.boolean.union([left_ready.copy(), right_ready.copy()], engine="manifold")
        elif operation_name == "intersection":
            result_mesh = trimesh.boolean.intersection([left_ready.copy(), right_ready.copy()], engine="manifold")
        else:
            result_mesh = trimesh.boolean.difference([left_ready.copy(), right_ready.copy()], engine="manifold")
        result_mesh = as_mesh(result_mesh)
        result_mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(result_mesh)
        return result_mesh
    except Exception as exc:
        emit_status(status_callback, f"Manifold backend unavailable ({exc}); falling back to VTK")

    left_poly = mesh_to_polydata(left_ready)
    right_poly = mesh_to_polydata(right_ready)

    emit_status(status_callback, f"Running {operation_name} boolean with VTK")
    if operation_name == "union":
        result_poly = left_poly.boolean_union(right_poly)
    elif operation_name == "intersection":
        result_poly = left_poly.boolean_intersection(right_poly)
    else:
        result_poly = left_poly.boolean_difference(right_poly)

    emit_status(status_callback, "Cleaning boolean result")
    result_mesh = polydata_to_mesh(result_poly)
    result_mesh = heal_mesh(
        result_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=None,
    )
    return result_mesh


def run_boolean_pipelines(
    left_paths: Path | Sequence[Path],
    right_paths: Path | Sequence[Path],
    output_path: Path,
    operations: Sequence[str],
    report_path: Optional[Path] = None,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    normalized_operations = [normalize_boolean_operation(operation) for operation in operations]
    if not normalized_operations:
        raise ValueError("At least one boolean operation is required.")

    left_input_paths, left_mesh = load_meshes(left_paths, "left", status_callback=status_callback)
    right_input_paths, right_mesh = load_meshes(right_paths, "right", status_callback=status_callback)

    emit_status(status_callback, "Computing input reports")
    left_report = mesh_report(left_mesh, area_eps=area_eps)
    right_report = mesh_report(right_mesh, area_eps=area_eps)

    emit_status(status_callback, "Preparing left solid")
    left_ready = heal_mesh(
        left_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=None,
    )
    emit_status(status_callback, "Preparing right solid")
    right_ready = heal_mesh(
        right_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=None,
    )

    multi_operation = len(normalized_operations) > 1
    results = []
    for operation_name in normalized_operations:
        emit_status(status_callback, f"Starting {operation_name} operation")
        result_mesh = run_boolean_operation(
            left_ready,
            right_ready,
            operation=operation_name,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        emit_status(status_callback, f"Computing {operation_name} report")
        result_report = mesh_report(result_mesh, area_eps=area_eps)

        operation_output_path = derive_operation_path(output_path, operation_name, multi_operation)
        operation_report_path = None
        if report_path is not None:
            operation_report_path = derive_operation_path(report_path, operation_name, multi_operation)

        emit_status(status_callback, f"Writing {operation_output_path.name}")
        operation_output_path.parent.mkdir(parents=True, exist_ok=True)
        export_mesh(result_mesh, operation_output_path)

        operation_report = {
            "mode": "boolean",
            "operation": operation_name,
            "left_input": str(left_input_paths[0]) if len(left_input_paths) == 1 else None,
            "right_input": str(right_input_paths[0]) if len(right_input_paths) == 1 else None,
            "left_inputs": [str(path) for path in left_input_paths],
            "right_inputs": [str(path) for path in right_input_paths],
            "output": str(operation_output_path),
            "left": asdict(left_report),
            "right": asdict(right_report),
            "result": asdict(result_report),
        }
        write_json_report(operation_report, operation_report_path)
        results.append(operation_report)

    if len(results) == 1:
        return results[0]

    summary = {
        "mode": "boolean-batch",
        "operations": normalized_operations,
        "left_input": str(left_input_paths[0]) if len(left_input_paths) == 1 else None,
        "right_input": str(right_input_paths[0]) if len(right_input_paths) == 1 else None,
        "left_inputs": [str(path) for path in left_input_paths],
        "right_inputs": [str(path) for path in right_input_paths],
        "output_base": str(output_path),
        "report_base": str(report_path) if report_path is not None else None,
        "results": results,
    }
    if report_path is not None:
        write_json_report(summary, report_path)
    return summary


def run_multi_input_boolean_pipelines(
    input_paths: Path | Sequence[Path],
    output_path: Path,
    operations: Sequence[str],
    report_path: Optional[Path] = None,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    normalized_operations = [normalize_boolean_operation(operation) for operation in operations]
    if not normalized_operations:
        raise ValueError("At least one boolean operation is required.")

    input_paths_list, input_meshes = load_individual_meshes(input_paths, "input", status_callback=status_callback)
    if len(input_paths_list) < 2:
        raise ValueError("Select at least two solids.")

    emit_status(status_callback, "Computing input reports")
    input_reports = [
        {
            "path": str(path),
            "report": asdict(mesh_report(mesh, area_eps=area_eps)),
        }
        for path, mesh in zip(input_paths_list, input_meshes)
    ]

    prepared_meshes = prepare_boolean_meshes(
        input_meshes,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
    )

    multi_operation = len(normalized_operations) > 1
    results = []
    for operation_name in normalized_operations:
        if operation_name == "clip":
            emit_status(status_callback, "Clip uses the first selected solid as the base and subtracts the remaining solids in order")
        emit_status(status_callback, f"Starting {operation_name} operation")
        result_mesh = run_iterative_boolean_operation(
            prepared_meshes,
            operation=operation_name,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        emit_status(status_callback, f"Computing {operation_name} report")
        result_report = mesh_report(result_mesh, area_eps=area_eps)

        operation_output_path = derive_operation_path(output_path, operation_name, multi_operation)
        operation_report_path = None
        if report_path is not None:
            operation_report_path = derive_operation_path(report_path, operation_name, multi_operation)

        emit_status(status_callback, f"Writing {operation_output_path.name}")
        operation_output_path.parent.mkdir(parents=True, exist_ok=True)
        export_mesh(result_mesh, operation_output_path)

        operation_report = {
            "mode": "boolean-multi-input",
            "operation": operation_name,
            "inputs": [str(path) for path in input_paths_list],
            "output": str(operation_output_path),
            "input_reports": input_reports,
            "result": asdict(result_report),
        }
        write_json_report(operation_report, operation_report_path)
        results.append(operation_report)

    if len(results) == 1:
        return results[0]

    summary = {
        "mode": "boolean-multi-input-batch",
        "operations": normalized_operations,
        "inputs": [str(path) for path in input_paths_list],
        "output_base": str(output_path),
        "report_base": str(report_path) if report_path is not None else None,
        "results": results,
    }
    if report_path is not None:
        write_json_report(summary, report_path)
    return summary


def write_json_report(report: dict, report_path: Optional[Path]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def make_preview_mesh(
    mesh: trimesh.Trimesh,
    max_faces: int = 120000,
    max_vertices: int = 80000,
) -> Tuple[trimesh.Trimesh, bool]:
    work = mesh.copy()
    decimated = False

    if len(work.faces) <= max_faces and len(work.vertices) <= max_vertices:
        return work, decimated

    total_faces = len(work.faces)
    step = max(1, math.ceil(total_faces / max_faces))
    while True:
        keep = np.arange(0, total_faces, step, dtype=np.int64)
        if len(keep) > max_faces:
            keep = keep[:max_faces]
        preview = work.submesh([keep], append=True, repair=False)
        preview = as_mesh(preview)
        preview.remove_unreferenced_vertices()
        decimated = True
        if len(preview.vertices) <= max_vertices or len(keep) <= 1000:
            return preview, decimated
        step += 1


def prepare_preview_payload(
    path: Path,
    max_file_bytes: int = 250_000_000,
    max_faces: int = 120000,
    max_vertices: int = 80000,
) -> PreviewPayload:
    if not path.exists():
        raise FileNotFoundError(f"Preview input not found: {path}")
    if path.stat().st_size > max_file_bytes:
        raise ValueError(
            f"Preview skipped: {path.name} is larger than the safe preview limit of {max_file_bytes // 1_000_000} MB."
        )

    mesh = load_mesh(path)
    preview_mesh, decimated = make_preview_mesh(mesh, max_faces=max_faces, max_vertices=max_vertices)
    return PreviewPayload(
        source=str(path),
        original_vertices=int(len(mesh.vertices)),
        original_faces=int(len(mesh.faces)),
        preview_vertices=int(len(preview_mesh.vertices)),
        preview_faces=int(len(preview_mesh.faces)),
        decimated=decimated,
        vertices=np.asarray(preview_mesh.vertices, dtype=float).copy(),
        faces=np.asarray(preview_mesh.faces, dtype=np.int64).copy(),
    )


def run_heal_pipeline(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    rebuild_triangles: bool = False,
    localized_intersection_repair: bool = False,
    make_watertight: bool = False,
    return_surface_after_watertight: bool = False,
    advanced_backend: str = "none",
    exact_arrangements_executable: Optional[Path] = None,
    tetra_backend_executable: Optional[Path] = None,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    emit_status(status_callback, f"Loading {input_path.name}")
    mesh_in = load_mesh(input_path)
    normalized_advanced_backend = normalize_advanced_heal_backend(advanced_backend)
    use_surface_return = bool(return_surface_after_watertight)
    use_watertight_step = bool(make_watertight or use_surface_return)
    use_advanced_backend = normalized_advanced_backend != "none" and not use_surface_return
    extra_steps = 0
    if rebuild_triangles:
        extra_steps += 1
    if localized_intersection_repair:
        extra_steps += 1
    if use_watertight_step:
        extra_steps += 1
    if use_advanced_backend:
        extra_steps += 1
    heal_steps = 0 if (use_advanced_backend or use_surface_return) else estimate_heal_mesh_steps(mesh_in)
    progress_tracker = ProgressTracker(progress_callback, 4 + extra_steps + heal_steps)
    progress_tracker.advance()

    emit_status(status_callback, "Computing input report")
    before = mesh_report(mesh_in, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Input", before)
    progress_tracker.advance()

    rebuild_report = None
    localized_intersection_report = None
    watertight_repair_report = None
    advanced_backend_report = None
    mesh_for_heal = mesh_in
    if use_surface_return and normalized_advanced_backend != "none":
        emit_status(status_callback, "Skipping advanced backend because return-surface-after-watertight is enabled")
    if rebuild_triangles:
        mesh_for_heal, rebuild_report = rebuild_mesh_without_triangle_overlaps(
            mesh_in,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        progress_tracker.advance()

    if localized_intersection_repair:
        mesh_for_heal, localized_intersection_report = repair_localized_self_intersections(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        progress_tracker.advance()

    if use_watertight_step and (use_advanced_backend or use_surface_return):
        mesh_for_heal, watertight_repair_report = attempt_watertight_repair(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        progress_tracker.advance()

    if use_advanced_backend:
        mesh_for_heal, advanced_backend_report = run_advanced_exact_tetrahedral_backend(
            mesh_for_heal,
            advanced_backend=normalized_advanced_backend,
            exact_arrangements_executable=exact_arrangements_executable,
            tetra_backend_executable=tetra_backend_executable,
            status_callback=status_callback,
        )
        progress_tracker.advance()

    if use_surface_return:
        emit_status(status_callback, "Returning repaired surface after watertight repair")
        mesh_out = mesh_for_heal
    elif use_advanced_backend:
        emit_status(status_callback, "Skipping legacy component healing after advanced backend")
        mesh_out = finalize_healed_mesh(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            resolve_component_overlaps=False,
            status_callback=status_callback,
        )
    else:
        mesh_out = heal_mesh(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
            progress_tracker=progress_tracker,
        )
        if use_watertight_step:
            mesh_out, watertight_repair_report = attempt_watertight_repair(
                mesh_out,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=status_callback,
            )
            progress_tracker.advance()
    emit_status(status_callback, "Computing output report")
    after = mesh_report(mesh_out, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Output", after)
    progress_tracker.advance()

    emit_status(status_callback, f"Writing {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_mesh(mesh_out, output_path)
    progress_tracker.advance()

    report = {
        "mode": "heal",
        "input": str(input_path),
        "output": str(output_path),
        "before": asdict(before),
        "after": asdict(after),
    }
    if rebuild_report is not None:
        report["triangle_rebuild"] = rebuild_report
    if localized_intersection_report is not None:
        report["localized_intersection_repair"] = localized_intersection_report
    if watertight_repair_report is not None:
        report["watertight_repair"] = watertight_repair_report
    if advanced_backend_report is not None:
        report["advanced_backend"] = advanced_backend_report
    write_json_report(report, report_path)
    return report


def run_boolean_pipeline(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    operation: str,
    report_path: Optional[Path] = None,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    return run_boolean_pipelines(
        left_paths=[left_path],
        right_paths=[right_path],
        output_path=output_path,
        operations=[operation],
        report_path=report_path,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
        progress_callback=progress_callback,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, heal, and boolean triangulated solids (DXF/MSH/STL/OBJ/PLY)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    heal_parser = subparsers.add_parser("heal", help="Detect and heal a triangulated solid")
    heal_parser.add_argument("input", type=Path, help="Input mesh file")
    heal_parser.add_argument("output", type=Path, help="Output healed mesh file")
    heal_parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    heal_parser.add_argument("--merge-eps", type=float, default=1e-8, help="Vertex merge tolerance")
    heal_parser.add_argument("--area-eps", type=float, default=1e-12, help="Degenerate triangle area threshold")
    heal_parser.add_argument("--dedup-decimals", type=int, default=8, help="Rounding decimals for duplicate-face detection")
    heal_parser.add_argument(
        "--rebuild-triangles",
        action="store_true",
        help="Rebuild the mesh triangle by triangle and skip duplicate or coplanar overlapping faces before healing",
    )
    heal_parser.add_argument(
        "--localized-intersection-repair",
        action="store_true",
        help="Experimental localized self-intersection repair pass inspired by region partitioning workflows",
    )
    heal_parser.add_argument(
        "--make-watertight",
        action="store_true",
        help="Attempt to fill holes and close open boundaries before export",
    )
    heal_parser.add_argument(
        "--return-surface-after-watertight",
        action="store_true",
        help="Stop after watertight surface repair and export that repaired surface directly",
    )
    heal_parser.add_argument(
        "--advanced-backend",
        choices=list(ADVANCED_HEAL_BACKENDS),
        default="none",
        help="Advanced exact+tetra backend: none, omc-ftetwild, or omc-cdt",
    )
    heal_parser.add_argument(
        "--exact-arrangements-exe",
        type=Path,
        default=None,
        help="Optional path to the OpenMeshCraft-Arrangements executable",
    )
    heal_parser.add_argument(
        "--tetra-backend-exe",
        type=Path,
        default=None,
        help="Optional path to the tetrahedral backend executable (FloatTetwild_bin or OpenMeshCraft-CDT)",
    )

    boolean_parser = subparsers.add_parser("boolean", help="Run boolean operations on two solids")
    boolean_parser.add_argument("left", type=Path, help="Left input solid")
    boolean_parser.add_argument("right", type=Path, help="Right input solid")
    boolean_parser.add_argument("output", type=Path, help="Output boolean mesh file")
    boolean_parser.add_argument(
        "--operation",
        choices=["union", "intersection", "clip"],
        required=True,
        help="Boolean operation to run; clip means left minus right",
    )
    boolean_parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    boolean_parser.add_argument("--merge-eps", type=float, default=1e-8, help="Vertex merge tolerance")
    boolean_parser.add_argument("--area-eps", type=float, default=1e-12, help="Degenerate triangle area threshold")
    boolean_parser.add_argument("--dedup-decimals", type=int, default=8, help="Rounding decimals for duplicate-face detection")

    return parser


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] not in {"heal", "boolean"}:
        argv = ["heal", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "heal":
            if not args.input.exists():
                raise FileNotFoundError(f"Input not found: {args.input}")
            report = run_heal_pipeline(
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
                merge_eps=args.merge_eps,
                area_eps=args.area_eps,
                dedup_decimals=args.dedup_decimals,
                rebuild_triangles=args.rebuild_triangles,
                localized_intersection_repair=args.localized_intersection_repair,
                make_watertight=args.make_watertight,
                return_surface_after_watertight=args.return_surface_after_watertight,
                advanced_backend=args.advanced_backend,
                exact_arrangements_executable=args.exact_arrangements_exe,
                tetra_backend_executable=args.tetra_backend_exe,
            )
        else:
            if not args.left.exists():
                raise FileNotFoundError(f"Input not found: {args.left}")
            if not args.right.exists():
                raise FileNotFoundError(f"Input not found: {args.right}")
            report = run_boolean_pipeline(
                left_path=args.left,
                right_path=args.right,
                output_path=args.output,
                operation=args.operation,
                report_path=args.report,
                merge_eps=args.merge_eps,
                area_eps=args.area_eps,
                dedup_decimals=args.dedup_decimals,
            )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
