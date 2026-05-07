import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import multiprocessing
import os
from queue import Empty
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:
    if exc.name == "scipy":
        raise ModuleNotFoundError(
            "Mesh Heal requires scipy. Install the project requirements into the interpreter you are using, "
            "or launch the app with c:/Projects/DG Edits/.venv/Scripts/python.exe .\\mesh_heal_gui.py"
        ) from exc
    raise

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


def _try_import_vtk_object():
    try:
        from vtkmodules.vtkCommonCore import vtkObject  # type: ignore

        return vtkObject
    except Exception:
        return None


def _try_import_vtk_output_window_types():
    try:
        from vtkmodules.vtkCommonCore import vtkOutputWindow, vtkStringOutputWindow  # type: ignore

        return vtkOutputWindow, vtkStringOutputWindow
    except Exception:
        return None, None


def _try_import_shapely():
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon  # type: ignore
        from shapely.ops import triangulate, unary_union  # type: ignore

        return {
            "GeometryCollection": GeometryCollection,
            "MultiPolygon": MultiPolygon,
            "Point": Point,
            "Polygon": Polygon,
            "triangulate": triangulate,
            "unary_union": unary_union,
        }
    except Exception:
        return None


@contextmanager
def suppress_vtk_warnings():
    vtk_object = _try_import_vtk_object()
    vtk_output_window, vtk_string_output_window = _try_import_vtk_output_window_types()
    if vtk_object is None:
        yield
        return

    try:
        previous_state = bool(vtk_object.GetGlobalWarningDisplay())
    except Exception:
        previous_state = True

    previous_output_window = None
    replacement_output_window = None
    if vtk_output_window is not None and vtk_string_output_window is not None:
        try:
            previous_output_window = vtk_output_window.GetInstance()
            replacement_output_window = vtk_string_output_window()
            replacement_output_window.PromptUserOff()
            vtk_output_window.SetInstance(replacement_output_window)
        except Exception:
            previous_output_window = None
            replacement_output_window = None

    try:
        vtk_object.GlobalWarningDisplayOff()
        yield
    finally:
        if vtk_output_window is not None and previous_output_window is not None:
            try:
                vtk_output_window.SetInstance(previous_output_window)
            except Exception:
                pass
        try:
            if previous_state:
                vtk_object.GlobalWarningDisplayOn()
            else:
                vtk_object.GlobalWarningDisplayOff()
        except Exception:
            pass


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
    hint_issues: list[dict] = field(default_factory=list)
    hint_paths: list[str] = field(default_factory=list)


class PreviewSkippedError(ValueError):
    pass


@dataclass(frozen=True)
class HealSearchCandidate:
    name: str
    rebuild_triangles: bool = False
    nonmanifold_edge_repair: bool = False
    localized_intersection_repair: bool = False
    point_cloud_rebuild: str = "none"
    distance_model: str = "none"
    make_watertight: bool = False
    distance_offset_ratio: float = 0.0
    distance_grid_spacing_ratio: float = 0.0
    aggressive: bool = False

    def enabled_step_count(self) -> int:
        count_enabled = 0
        count_enabled += int(self.rebuild_triangles)
        count_enabled += int(self.nonmanifold_edge_repair)
        count_enabled += int(self.localized_intersection_repair)
        count_enabled += int(self.point_cloud_rebuild != "none")
        count_enabled += int(self.distance_model != "none")
        count_enabled += int(self.make_watertight)
        return count_enabled


@dataclass(frozen=True)
class LeapfrogAcceptanceThresholds:
    max_mean_distance_normalized: float = 0.02
    max_p95_distance_normalized: float = 0.05
    max_component_count_delta: float = 5.0
    max_volume_ratio_delta: float = 0.25


class SkippedInputError(RuntimeError):
    def __init__(self, message: str, *, code: str = "skipped_input", details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


StatusCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, int], None]]

HEAL_PREPROCESS_STEP_COUNT = 3
HEAL_COMPONENT_STEP_COUNT = 3

BOOLEAN_OPERATIONS = ("union", "intersection", "clip")
ADVANCED_HEAL_BACKENDS = ("none", "cgal-alpha-wrap", "cgal-repair")
DEPRECATED_ADVANCED_HEAL_BACKENDS = ("omc-ftetwild", "omc-cdt")
ALL_ADVANCED_HEAL_BACKENDS = ADVANCED_HEAL_BACKENDS + DEPRECATED_ADVANCED_HEAL_BACKENDS
POINT_CLOUD_REBUILD_MODES = ("none", "triangle-centers-poisson")
DISTANCE_MODEL_MODES = ("none", "distance-hull", "surface-shell")
INTENDED_MESH_TYPES = ("auto", "solid", "surface")

AUTORESEARCH_SAMPLE_POINT_COUNT = 2048
AUTORESEARCH_FAST_SAMPLE_POINT_COUNT = 512
AUTORESEARCH_FAST_MAX_CANDIDATES = 5
AUTORESEARCH_FAST_TIME_BUDGET_SECONDS = 180.0
AUTORESEARCH_FAST_CANDIDATE_TIMEOUT_SECONDS = 150.0
AUTORESEARCH_HISTORY_LEDGER_LIMIT = 32
AUTORESEARCH_SELF_INTERSECTION_OPEN3D_MAX_FACES = 10000
AUTORESEARCH_SELF_INTERSECTION_MAX_CANDIDATE_PAIRS = 50000
AUTORESEARCH_SELF_INTERSECTION_TIMEOUT_SECONDS = 30.0
AUTORESEARCH_ERROR_SCORE = 1e18

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
CGAL_ALPHA_WRAP_CANDIDATES = (
    "mesh_heal_cgal_alpha_wrap.exe",
    "mesh_heal_cgal_alpha_wrap",
    "cgal-alpha-wrap.exe",
    "cgal-alpha-wrap",
    "advanced_backends/cgal_alpha_wrap/build/mesh_heal_cgal_alpha_wrap.exe",
    "advanced_backends/cgal_alpha_wrap/build/mesh_heal_cgal_alpha_wrap",
    "advanced_backends/cgal_alpha_wrap/build/Release/mesh_heal_cgal_alpha_wrap.exe",
)

OPENMESHCRAFT_ARRANGEMENTS_ENV = "OPENMESHCRAFT_ARRANGEMENTS_EXE"
OPENMESHCRAFT_CDT_ENV = "OPENMESHCRAFT_CDT_EXE"
FASTTETWILD_ENV = "FASTTETWILD_EXE"
CGAL_ALPHA_WRAP_ENV = "CGAL_ALPHA_WRAP_EXE"


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
    return input_path.with_name(f"{input_path.stem}_healed.msh")


def normalize_advanced_heal_backend(backend: str) -> str:
    backend_name = backend.lower().strip()
    if backend_name in DEPRECATED_ADVANCED_HEAL_BACKENDS:
        raise ValueError(
            "OpenMeshCraft backends are deprecated. Use the guided heal workflow or CGAL Alpha Wrap instead."
        )
    if backend_name not in ALL_ADVANCED_HEAL_BACKENDS:
        raise ValueError(f"Unsupported advanced heal backend: {backend}")
    return backend_name


def normalize_intended_mesh_type(intended_mesh_type: str) -> str:
    normalized = str(intended_mesh_type or "auto").strip().lower()
    if normalized not in INTENDED_MESH_TYPES:
        raise ValueError(f"Unsupported intended mesh type: {intended_mesh_type}")
    return normalized


def apply_intended_mesh_type_to_hint_summary(hint_summary: dict, intended_mesh_type: str) -> dict:
    normalized_intended_mesh_type = normalize_intended_mesh_type(intended_mesh_type)
    adjusted = dict(hint_summary)
    if normalized_intended_mesh_type == "surface":
        adjusted["force_make_watertight"] = False
        adjusted["prefer_proactive_safe"] = bool(
            adjusted.get("force_rebuild_triangles")
            or adjusted.get("force_nonmanifold_edge_repair")
            or adjusted.get("force_localized_intersection_repair")
        )
    return adjusted


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
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())
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


def run_cgal_backend(
    mesh: trimesh.Trimesh,
    mode: str,
    executable_path: Optional[Path | str] = None,
    alpha: Optional[float] = None,
    offset: Optional[float] = None,
    alpha_relative: Optional[float] = None,
    offset_relative: Optional[float] = None,
    repair_merge_boundary_vertices: bool = True,
    repair_merge_reversible_components: bool = True,
    repair_stitch_borders: bool = True,
    repair_duplicate_non_manifold_vertices: bool = True,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    resolved_executable = resolve_external_executable(
        explicit_path=executable_path,
        env_var=CGAL_ALPHA_WRAP_ENV,
        candidates=CGAL_ALPHA_WRAP_CANDIDATES,
    )
    if resolved_executable is None:
        raise RuntimeError(
            "CGAL Alpha Wrap executable not found. Set CGAL_ALPHA_WRAP_EXE, pass --cgal-backend-exe, "
            "or place mesh_heal_cgal_alpha_wrap on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="mesh_heal_cgal_wrap_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        mode_slug = mode.replace("-", "_")
        input_path = temp_dir / f"cgal_{mode_slug}_input.obj"
        output_path = temp_dir / f"cgal_{mode_slug}_output.obj"
        export_temporary_triangle_mesh(mesh, input_path)

        command = [resolved_executable, str(input_path), str(output_path), "--mode", mode]
        if mode == "alpha-wrap":
            if alpha is not None and alpha > 0.0:
                command.extend(["--alpha", str(float(alpha))])
            elif alpha_relative is not None and alpha_relative > 0.0:
                command.extend(["--alpha-relative", str(float(alpha_relative))])

            if offset is not None and offset > 0.0:
                command.extend(["--offset", str(float(offset))])
            elif offset_relative is not None and offset_relative > 0.0:
                command.extend(["--offset-relative", str(float(offset_relative))])
        elif mode == "repair":
            if not repair_merge_boundary_vertices:
                command.append("--skip-repair-merge-boundary-vertices")
            if not repair_merge_reversible_components:
                command.append("--skip-repair-merge-reversible-components")
            if not repair_stitch_borders:
                command.append("--skip-repair-stitch-borders")
            if not repair_duplicate_non_manifold_vertices:
                command.append("--skip-repair-duplicate-non-manifold-vertices")
        process_report = run_external_command(command, cwd=temp_dir, status_callback=status_callback)

        if not output_path.exists():
            raise RuntimeError(f"CGAL {mode} backend did not produce the expected OBJ output.")

        surface_mesh = load_mesh(output_path)
        return surface_mesh, {
            "backend": f"cgal-{mode}",
            "mode": mode,
            "requested_parameters": {
                "alpha": None if alpha is None else float(alpha),
                "offset": None if offset is None else float(offset),
                "alpha_relative": None if alpha_relative is None else float(alpha_relative),
                "offset_relative": None if offset_relative is None else float(offset_relative),
                "repair_merge_boundary_vertices": bool(repair_merge_boundary_vertices),
                "repair_merge_reversible_components": bool(repair_merge_reversible_components),
                "repair_stitch_borders": bool(repair_stitch_borders),
                "repair_duplicate_non_manifold_vertices": bool(repair_duplicate_non_manifold_vertices),
            },
            "executable": resolved_executable,
            "input": str(input_path),
            "output": str(output_path),
            "process": process_report,
        }


def run_advanced_exact_tetrahedral_backend(
    mesh: trimesh.Trimesh,
    advanced_backend: str,
    exact_arrangements_executable: Optional[Path | str] = None,
    tetra_backend_executable: Optional[Path | str] = None,
    cgal_backend_executable: Optional[Path | str] = None,
    cgal_alpha: Optional[float] = None,
    cgal_offset: Optional[float] = None,
    cgal_alpha_relative: Optional[float] = None,
    cgal_offset_relative: Optional[float] = None,
    cgal_repair_merge_boundary_vertices: bool = True,
    cgal_repair_merge_reversible_components: bool = True,
    cgal_repair_stitch_borders: bool = True,
    cgal_repair_duplicate_non_manifold_vertices: bool = True,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    backend_name = normalize_advanced_heal_backend(advanced_backend)
    if backend_name == "none":
        return mesh, {"backend": backend_name, "skipped": True}

    if backend_name in {"cgal-alpha-wrap", "cgal-repair"}:
        mode = "alpha-wrap" if backend_name == "cgal-alpha-wrap" else "repair"
        emit_status(status_callback, f"Running CGAL {mode} backend")
        surface_mesh, cgal_report = run_cgal_backend(
            mesh,
            mode=mode,
            executable_path=cgal_backend_executable,
            alpha=cgal_alpha,
            offset=cgal_offset,
            alpha_relative=cgal_alpha_relative,
            offset_relative=cgal_offset_relative,
            repair_merge_boundary_vertices=cgal_repair_merge_boundary_vertices,
            repair_merge_reversible_components=cgal_repair_merge_reversible_components,
            repair_stitch_borders=cgal_repair_stitch_borders,
            repair_duplicate_non_manifold_vertices=cgal_repair_duplicate_non_manifold_vertices,
            status_callback=status_callback,
        )
        return surface_mesh, {
            "backend": backend_name,
            "skipped": False,
            "cgal": cgal_report,
        }

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
    rebuilt_faces_seen: set[Tuple[Tuple[float, float, float], ...]] = set()
    generated_duplicate_faces_skipped = 0
    generated_overlap_faces_skipped = 0

    total_plane_buckets = int(len(plane_buckets))
    emit_status(status_callback, f"Unioning coplanar triangle patches across {total_plane_buckets:,} groups")
    progress_update_interval = max(1, total_plane_buckets // 20) if total_plane_buckets > 0 else 1
    for plane_bucket_index, plane_bucket_state in enumerate(plane_buckets.values(), start=1):
        polygons = plane_bucket_state["polygons"]
        if not polygons:
            continue

        merged_geometry = unary_union(polygons)
        normal = np.asarray(plane_bucket_state["normal"], dtype=float)
        dist = float(plane_bucket_state["dist"])
        drop_axis = int(plane_bucket_state["drop_axis"])
        polygon_tolerance = max(merge_eps, 1e-9)
        accepted_output_polygons: List[object] = []
        accepted_output_bounds: List[Tuple[float, float, float, float]] = []

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
                candidate_polygon = polygon_type(ensure_ccw_2d(coords_2d))
                if candidate_polygon.is_empty or float(candidate_polygon.area) <= area_eps:
                    continue

                candidate_bounds = tuple(float(value) for value in candidate_polygon.bounds)
                overlaps_previous_output = False
                for existing_polygon, existing_bounds in zip(accepted_output_polygons, accepted_output_bounds):
                    if (
                        candidate_bounds[2] < existing_bounds[0] - polygon_tolerance
                        or candidate_bounds[0] > existing_bounds[2] + polygon_tolerance
                        or candidate_bounds[3] < existing_bounds[1] - polygon_tolerance
                        or candidate_bounds[1] > existing_bounds[3] + polygon_tolerance
                    ):
                        continue
                    try:
                        overlap_area = float(candidate_polygon.intersection(existing_polygon).area)
                    except Exception:
                        overlap_area = 0.0
                    if overlap_area > area_eps:
                        overlaps_previous_output = True
                        break
                if overlaps_previous_output:
                    generated_overlap_faces_skipped += 1
                    continue

                coords_3d = unproject_points_from_plane_2d(coords_2d, normal=normal, dist=dist, drop_axis=drop_axis)
                if triangle_area(coords_3d[0], coords_3d[1], coords_3d[2]) <= area_eps:
                    continue

                tri_normal = np.cross(coords_3d[1] - coords_3d[0], coords_3d[2] - coords_3d[0])
                if float(np.dot(tri_normal, normal)) < 0.0:
                    coords_3d = coords_3d[[0, 2, 1]]

                rebuilt_face_key = face_key(coords_3d, decimals=dedup_decimals)
                if rebuilt_face_key in rebuilt_faces_seen:
                    generated_duplicate_faces_skipped += 1
                    continue

                base = len(rebuilt_vertices)
                rebuilt_vertices.extend(coords_3d.tolist())
                rebuilt_faces.append([base, base + 1, base + 2])
                rebuilt_faces_seen.add(rebuilt_face_key)
                accepted_output_polygons.append(candidate_polygon)
                accepted_output_bounds.append(candidate_bounds)

        if total_plane_buckets > 10 and (
            plane_bucket_index == total_plane_buckets
            or plane_bucket_index % progress_update_interval == 0
        ):
            percent_complete = 100.0 * float(plane_bucket_index) / float(total_plane_buckets)
            emit_status(
                status_callback,
                "Unioning coplanar triangle patches: processed "
                f"{plane_bucket_index:,}/{total_plane_buckets:,} groups ({percent_complete:.1f}%), "
                f"merged polygons {merged_polygons:,}, output faces {len(rebuilt_faces):,}",
            )

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
        "generated_duplicate_faces_skipped": int(generated_duplicate_faces_skipped),
        "generated_overlap_faces_skipped": int(generated_overlap_faces_skipped),
        "coplanar_groups_processed": int(len(plane_buckets)),
        "merged_polygons": int(merged_polygons),
    }


def normalize_face_index_selection(face_indices: Sequence[int], face_count: int) -> np.ndarray:
    if face_count < 0:
        raise ValueError("face_count must be non-negative.")
    if face_count == 0:
        return np.empty((0,), dtype=np.int64)

    normalized = sorted({int(index) for index in face_indices if 0 <= int(index) < face_count})
    return np.asarray(normalized, dtype=np.int64)


def face_unit_normal(vertices: np.ndarray, eps: float = 1e-15) -> Optional[np.ndarray]:
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    norm = float(np.linalg.norm(normal))
    if norm <= eps:
        return None
    return normal / norm


def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, normal))) >= 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)

    axis_u = np.cross(normal, candidate)
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm <= 1e-15:
        candidate = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_u = np.cross(normal, candidate)
        axis_u_norm = float(np.linalg.norm(axis_u))
        if axis_u_norm <= 1e-15:
            raise ValueError("Could not build a plane basis from the supplied normal.")
    axis_u = axis_u / axis_u_norm
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / float(np.linalg.norm(axis_v))
    return axis_u, axis_v


def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if len(points) < 3:
        raise ValueError("At least three points are required to fit a plane.")

    origin = np.mean(points, axis=0)
    centered = points - origin
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-15:
        raise ValueError("Could not compute a stable plane normal for the selected points.")
    normal = normal / normal_norm

    distances = centered @ normal
    max_distance = float(np.max(np.abs(distances))) if len(distances) else 0.0
    axis_u, axis_v = build_plane_basis(normal)
    if len(singular_values) >= 2 and singular_values[0] > 0.0:
        first_axis = np.asarray(vh[0], dtype=float)
        first_axis = first_axis - normal * float(np.dot(first_axis, normal))
        first_axis_norm = float(np.linalg.norm(first_axis))
        if first_axis_norm > 1e-15:
            axis_u = first_axis / first_axis_norm
            axis_v = np.cross(normal, axis_u)
            axis_v = axis_v / float(np.linalg.norm(axis_v))

    return origin, normal, axis_u, axis_v, max_distance


def project_points_to_plane_basis(
    points: np.ndarray,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    relative = np.asarray(points, dtype=float) - origin
    return np.column_stack((relative @ axis_u, relative @ axis_v))


def unproject_points_from_plane_basis(
    points_2d: np.ndarray,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    return origin + np.outer(points_2d[:, 0], axis_u) + np.outer(points_2d[:, 1], axis_v)


def sanitize_loop_points(points: np.ndarray, merge_eps: float = 1e-8) -> np.ndarray:
    sanitized: List[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if sanitized and float(np.linalg.norm(point - sanitized[-1])) <= merge_eps:
            continue
        sanitized.append(point)

    if len(sanitized) >= 2 and float(np.linalg.norm(sanitized[0] - sanitized[-1])) <= merge_eps:
        sanitized.pop()

    if len(sanitized) < 3:
        raise ValueError("The selected closed path needs at least three distinct points.")
    return np.asarray(sanitized, dtype=float)


def extract_boundary_loops(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if unique_edges.size == 0:
        return []

    edge_inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(edge_inverse, minlength=unique_edges.shape[0])
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return []

    adjacency: Dict[int, List[int]] = defaultdict(list)
    unused_edges: set[Tuple[int, int]] = set()
    for start, end in boundary_edges:
        a = int(start)
        b = int(end)
        adjacency[a].append(b)
        adjacency[b].append(a)
        unused_edges.add((min(a, b), max(a, b)))

    loops: List[np.ndarray] = []
    while unused_edges:
        start_edge = next(iter(unused_edges))
        start_vertex = start_edge[0]
        next_vertex = start_edge[1]
        path = [start_vertex, next_vertex]
        unused_edges.remove(start_edge)
        previous_vertex = start_vertex
        current_vertex = next_vertex
        closed = False

        while True:
            available = []
            for candidate in adjacency[current_vertex]:
                edge_key = (min(current_vertex, candidate), max(current_vertex, candidate))
                if edge_key in unused_edges:
                    available.append(candidate)

            if not available:
                break

            next_candidate = None
            for candidate in available:
                if candidate != previous_vertex:
                    next_candidate = candidate
                    break
            if next_candidate is None:
                next_candidate = available[0]

            edge_key = (min(current_vertex, next_candidate), max(current_vertex, next_candidate))
            unused_edges.remove(edge_key)
            if next_candidate == path[0]:
                closed = True
                break

            path.append(next_candidate)
            previous_vertex, current_vertex = current_vertex, next_candidate

        if closed and len(path) >= 3:
            loops.append(np.asarray(path, dtype=np.int64))

    return loops


def find_boundary_loop_near_point(mesh: trimesh.Trimesh, point: np.ndarray) -> Tuple[np.ndarray, dict]:
    loops = extract_boundary_loops(mesh)
    if not loops:
        raise ValueError("The current mesh has no closed boundary loops.")

    target = np.asarray(point, dtype=float)
    best_loop = None
    best_distance = float("inf")
    for loop in loops:
        loop_vertices = np.asarray(mesh.vertices[loop], dtype=float)
        distances = np.linalg.norm(loop_vertices - target, axis=1)
        distance = float(np.min(distances)) if len(distances) else float("inf")
        if distance < best_distance:
            best_distance = distance
            best_loop = loop

    if best_loop is None:
        raise ValueError("Could not find a usable boundary loop near the selected point.")

    loop_vertices = np.asarray(mesh.vertices[best_loop], dtype=float)
    edge_vectors = np.roll(loop_vertices, -1, axis=0) - loop_vertices
    loop_length = float(np.sum(np.linalg.norm(edge_vectors, axis=1)))
    return best_loop.copy(), {
        "boundary_loop_vertices": int(len(best_loop)),
        "distance_to_pick": best_distance,
        "loop_length": loop_length,
        "boundary_loop_count": int(len(loops)),
    }


def remove_faces_by_index(
    mesh: trimesh.Trimesh,
    face_indices: Sequence[int],
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
) -> Tuple[trimesh.Trimesh, dict]:
    normalized = normalize_face_index_selection(face_indices, len(mesh.faces))
    before = mesh_report(mesh, area_eps=area_eps)
    if len(normalized) == 0:
        return mesh.copy(), {
            "removed_faces": 0,
            "selected_faces": 0,
            "before": asdict(before),
            "after": asdict(before),
        }

    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[normalized] = False
    edited = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float).copy(),
        faces=np.asarray(mesh.faces[keep_mask], dtype=np.int64).copy(),
        process=False,
    )
    edited.remove_unreferenced_vertices()
    edited = merge_nearby_vertices(edited, merge_eps=merge_eps)
    edited = remove_duplicate_faces(edited, decimals=dedup_decimals)
    edited = remove_degenerate_faces(edited, eps=area_eps)
    edited.remove_unreferenced_vertices()

    after = mesh_report(edited, area_eps=area_eps)
    return edited, {
        "removed_faces": int(len(normalized)),
        "selected_faces": int(len(normalized)),
        "before": asdict(before),
        "after": asdict(after),
    }


def rebuild_region_inside_closed_path(
    mesh: trimesh.Trimesh,
    closed_path_points: Sequence[Sequence[float]] | np.ndarray,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    plane_tolerance: Optional[float] = None,
    normal_alignment: float = 0.95,
) -> Tuple[trimesh.Trimesh, dict]:
    shapely_tools = _try_import_shapely()
    if shapely_tools is None:
        raise RuntimeError("Closed-path retriangulation requires shapely. Install requirements first.")

    polygon_type = shapely_tools["Polygon"]
    point_type = shapely_tools["Point"]
    triangulate = shapely_tools["triangulate"]

    loop_points = sanitize_loop_points(np.asarray(closed_path_points, dtype=float), merge_eps=merge_eps)
    before = mesh_report(mesh, area_eps=area_eps)

    if plane_tolerance is None:
        plane_tolerance = max(merge_eps * 10.0, 1e-6)

    origin, normal, axis_u, axis_v, max_distance = fit_plane_to_points(loop_points)
    if max_distance > plane_tolerance:
        raise ValueError(
            f"The selected closed path is not planar enough for retriangulation. Maximum deviation was {max_distance:.6g}."
        )

    boundary_2d = sanitize_loop_points(
        project_points_to_plane_basis(loop_points, origin=origin, axis_u=axis_u, axis_v=axis_v),
        merge_eps=max(merge_eps, 1e-12),
    )
    boundary_2d = ensure_ccw_2d(boundary_2d)
    polygon = polygon_type(boundary_2d)
    if polygon.is_empty or float(polygon.area) <= area_eps:
        raise ValueError("The selected closed path does not enclose a valid area.")
    if not polygon.is_valid:
        raise ValueError("The selected closed path is self-intersecting or otherwise invalid.")

    polygon_buffer = polygon.buffer(max(merge_eps, 1e-9))
    removable_faces: List[int] = []
    centroid_points = np.asarray(mesh.triangles_center, dtype=float)
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    for face_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        triangle = np.asarray(mesh.vertices[face], dtype=float)
        distances = np.abs((triangle - origin) @ normal)
        if float(np.max(distances)) > plane_tolerance:
            continue

        face_normal = np.asarray(face_normals[face_index], dtype=float)
        face_normal_norm = float(np.linalg.norm(face_normal))
        if face_normal_norm <= 1e-15:
            computed_normal = face_unit_normal(triangle)
            if computed_normal is None:
                continue
            face_normal = computed_normal
        else:
            face_normal = face_normal / face_normal_norm

        if abs(float(np.dot(face_normal, normal))) < normal_alignment:
            continue

        centroid_2d = project_points_to_plane_basis(
            centroid_points[face_index].reshape(1, 3),
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )[0]
        if polygon_buffer.covers(point_type(float(centroid_2d[0]), float(centroid_2d[1]))):
            removable_faces.append(face_index)

    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    if removable_faces:
        keep_mask[np.asarray(removable_faces, dtype=np.int64)] = False

    combined_vertices = np.asarray(mesh.vertices, dtype=float).copy().tolist()
    combined_faces = np.asarray(mesh.faces[keep_mask], dtype=np.int64).copy().tolist()
    added_faces = 0
    for tri_polygon in triangulate(polygon):
        if tri_polygon.is_empty or float(tri_polygon.area) <= area_eps:
            continue
        if not polygon_buffer.covers(tri_polygon):
            continue

        coords_2d = np.asarray(tri_polygon.exterior.coords[:-1], dtype=float)
        if coords_2d.shape != (3, 2):
            continue

        coords_3d = unproject_points_from_plane_basis(coords_2d, origin=origin, axis_u=axis_u, axis_v=axis_v)
        if triangle_area(coords_3d[0], coords_3d[1], coords_3d[2]) <= area_eps:
            continue

        tri_normal = face_unit_normal(coords_3d)
        if tri_normal is None:
            continue
        if float(np.dot(tri_normal, normal)) < 0.0:
            coords_3d = coords_3d[[0, 2, 1]]

        base_index = len(combined_vertices)
        combined_vertices.extend(coords_3d.tolist())
        combined_faces.append([base_index, base_index + 1, base_index + 2])
        added_faces += 1

    rebuilt_mesh = trimesh.Trimesh(
        vertices=np.asarray(combined_vertices, dtype=float),
        faces=np.asarray(combined_faces, dtype=np.int64),
        process=False,
    )
    rebuilt_mesh = merge_nearby_vertices(rebuilt_mesh, merge_eps=merge_eps)
    rebuilt_mesh = remove_duplicate_faces(rebuilt_mesh, decimals=dedup_decimals)
    rebuilt_mesh = remove_degenerate_faces(rebuilt_mesh, eps=area_eps)
    rebuilt_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(rebuilt_mesh)

    after = mesh_report(rebuilt_mesh, area_eps=area_eps)
    return rebuilt_mesh, {
        "selected_loop_vertices": int(len(loop_points)),
        "path_planarity_max_distance": max_distance,
        "removed_faces": int(len(removable_faces)),
        "added_faces": int(added_faces),
        "before": asdict(before),
        "after": asdict(after),
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


def detect_nonmanifold_edges(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if unique_edges.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)

    edge_inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    edge_counts = np.bincount(edge_inverse, minlength=unique_edges.shape[0])
    nonmanifold_edge_indices = np.flatnonzero(edge_counts > 2)
    if len(nonmanifold_edge_indices) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return unique_edges[nonmanifold_edge_indices], edge_counts[nonmanifold_edge_indices]


def detect_edge_stats(mesh: trimesh.Trimesh) -> Tuple[int, int]:
    unique_edges = mesh.edges_unique
    if unique_edges.shape[0] == 0:
        return 0, 0
    edge_inv = mesh.edges_unique_inverse
    counts = np.bincount(edge_inv, minlength=unique_edges.shape[0])
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return nonmanifold, boundary


def safe_mesh_volume(mesh: trimesh.Trimesh) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        volume = mesh.volume
    return float(volume) if np.isfinite(volume) else float("nan")


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
        volume=safe_mesh_volume(mesh),
        area=float(mesh.area),
        degenerate_faces=int(len(deg)),
        duplicate_faces=int(dup),
        nonmanifold_edges=nonmanifold,
        boundary_edges=boundary,
    )


def estimate_mesh_scale(mesh: trimesh.Trimesh) -> float:
    if len(mesh.vertices) == 0:
        return 1.0

    bounds = np.asarray(mesh.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return max(diagonal, 1e-9)


def relative_metric_delta(reference_value: float, candidate_value: float, floor: float = 1e-9) -> Optional[float]:
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        return None
    scale = max(abs(float(reference_value)), floor)
    return float(abs(float(candidate_value) - float(reference_value)) / scale)


def sample_mesh_surface_points(mesh: trimesh.Trimesh, sample_count: int) -> np.ndarray:
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return np.zeros((0, 3), dtype=float)

    count_target = max(64, int(sample_count))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count_target)
        return np.asarray(points, dtype=float)
    except Exception:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) <= count_target:
            return vertices.copy()
        sample_indices = np.linspace(0, len(vertices) - 1, num=count_target, dtype=np.int64)
        return vertices[sample_indices]


def measure_shape_fidelity(
    reference_mesh: trimesh.Trimesh,
    candidate_mesh: trimesh.Trimesh,
    reference_report: MeshReport,
    candidate_report: MeshReport,
    reference_component_count: int,
    sample_point_count: int = AUTORESEARCH_SAMPLE_POINT_COUNT,
) -> dict:
    scale = max(estimate_mesh_scale(reference_mesh), estimate_mesh_scale(candidate_mesh), 1e-9)
    if len(candidate_mesh.faces) == 0 or len(candidate_mesh.vertices) == 0:
        return {
            "sample_count": 0,
            "mean_distance": scale,
            "p95_distance": scale,
            "max_distance": scale,
            "mean_distance_normalized": 1.0,
            "p95_distance_normalized": 1.0,
            "max_distance_normalized": 1.0,
            "centroid_drift": scale,
            "centroid_drift_normalized": 1.0,
            "extents_drift": scale,
            "extents_drift_normalized": 1.0,
            "area_ratio_delta": 1.0,
            "volume_ratio_delta": None,
            "component_count": 0,
            "component_count_delta": float(reference_component_count),
        }

    sample_count = min(
        max(64, int(sample_point_count)),
        max(256, min(len(reference_mesh.faces), len(candidate_mesh.faces)) * 4),
    )
    reference_points = sample_mesh_surface_points(reference_mesh, sample_count)
    candidate_points = sample_mesh_surface_points(candidate_mesh, sample_count)

    if len(reference_points) == 0 or len(candidate_points) == 0:
        symmetric_distances = np.full((1,), scale, dtype=float)
    else:
        reference_tree = cKDTree(reference_points)
        candidate_tree = cKDTree(candidate_points)
        reference_to_candidate = np.asarray(candidate_tree.query(reference_points, k=1)[0], dtype=float)
        candidate_to_reference = np.asarray(reference_tree.query(candidate_points, k=1)[0], dtype=float)
        symmetric_distances = np.concatenate([reference_to_candidate, candidate_to_reference])

    reference_vertices = np.asarray(reference_mesh.vertices, dtype=float)
    candidate_vertices = np.asarray(candidate_mesh.vertices, dtype=float)
    reference_centroid = np.mean(reference_vertices, axis=0) if len(reference_vertices) > 0 else np.zeros(3, dtype=float)
    candidate_centroid = np.mean(candidate_vertices, axis=0) if len(candidate_vertices) > 0 else np.zeros(3, dtype=float)
    centroid_drift = float(np.linalg.norm(candidate_centroid - reference_centroid))

    reference_extents = np.ptp(reference_vertices, axis=0) if len(reference_vertices) > 0 else np.zeros(3, dtype=float)
    candidate_extents = np.ptp(candidate_vertices, axis=0) if len(candidate_vertices) > 0 else np.zeros(3, dtype=float)
    extents_drift = float(np.linalg.norm(candidate_extents - reference_extents))

    volume_ratio_delta = None
    if reference_report.watertight and candidate_report.watertight:
        volume_ratio_delta = relative_metric_delta(reference_report.volume, candidate_report.volume)

    component_count = len(split_disconnected_components(candidate_mesh))

    return {
        "sample_count": int(len(reference_points) + len(candidate_points)),
        "mean_distance": float(np.mean(symmetric_distances)),
        "p95_distance": float(np.percentile(symmetric_distances, 95)),
        "max_distance": float(np.max(symmetric_distances)),
        "mean_distance_normalized": float(np.mean(symmetric_distances) / scale),
        "p95_distance_normalized": float(np.percentile(symmetric_distances, 95) / scale),
        "max_distance_normalized": float(np.max(symmetric_distances) / scale),
        "centroid_drift": centroid_drift,
        "centroid_drift_normalized": float(centroid_drift / scale),
        "extents_drift": extents_drift,
        "extents_drift_normalized": float(extents_drift / scale),
        "area_ratio_delta": relative_metric_delta(reference_report.area, candidate_report.area) or 0.0,
        "volume_ratio_delta": volume_ratio_delta,
        "component_count": int(component_count),
        "component_count_delta": float(abs(component_count - reference_component_count)),
    }


def validate_leapfrog_roundtrip(
    mesh: trimesh.Trimesh,
    area_eps: float = 1e-12,
    self_intersection_timeout_seconds: float = AUTORESEARCH_SELF_INTERSECTION_TIMEOUT_SECONDS,
) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="mesh_heal_autoresearch_") as temp_dir_name:
            temp_path = Path(temp_dir_name) / "candidate.msh"
            export_leapfrog_msh(mesh, temp_path)
            reloaded = load_leapfrog_msh(temp_path)
        reloaded_report = mesh_report(reloaded, area_eps=area_eps)
        self_intersections = validate_leapfrog_self_intersections(
            reloaded,
            timeout_seconds=self_intersection_timeout_seconds,
        )
        return {
            "exportable": True,
            "roundtrip_loadable": True,
            "vertices_match": bool(len(reloaded.vertices) == len(mesh.vertices)),
            "faces_match": bool(len(reloaded.faces) == len(mesh.faces)),
            "roundtrip_report": asdict(reloaded_report),
            "self_intersections": self_intersections,
        }
    except Exception as exc:
        return {
            "exportable": False,
            "roundtrip_loadable": False,
            "vertices_match": False,
            "faces_match": False,
            "roundtrip_report": None,
            "self_intersections": {
                "checked": False,
                "skipped": True,
                "skip_reason": "roundtrip_failed",
                "has_self_intersections": None,
                "intersecting_pairs": None,
                "intersecting_faces": None,
                "method": None,
            },
            "error": str(exc),
        }


def make_skipped_leapfrog_validation(skip_reason: str) -> dict:
    return {
        "exportable": False,
        "roundtrip_loadable": False,
        "vertices_match": False,
        "faces_match": False,
        "roundtrip_report": None,
        "self_intersections": {
            "checked": False,
            "skipped": True,
            "skip_reason": skip_reason,
            "has_self_intersections": None,
            "intersecting_pairs": None,
            "intersecting_faces": None,
            "method": None,
        },
        "skipped": True,
        "skip_reason": skip_reason,
    }


def _validate_single_mesh_self_intersections(mesh: trimesh.Trimesh, merge_eps: float = 1e-8) -> dict:
    work = mesh.copy()
    work.remove_unreferenced_vertices()
    if len(work.faces) == 0:
        return {
            "checked": True,
            "skipped": False,
            "skip_reason": None,
            "has_self_intersections": False,
            "intersecting_pairs": 0,
            "intersecting_faces": 0,
            "method": "empty-mesh",
        }

    open3d_error = None
    o3d = _try_import_open3d()
    if o3d is not None and len(work.faces) <= AUTORESEARCH_SELF_INTERSECTION_OPEN3D_MAX_FACES:
        try:
            omesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.asarray(work.vertices, dtype=float)),
                triangles=o3d.utility.Vector3iVector(np.asarray(work.faces, dtype=np.int32)),
            )
            intersecting_pairs = None
            intersecting_faces = None
            has_self_intersections = None
            if hasattr(omesh, "get_self_intersecting_triangles"):
                raw_pairs = np.asarray(omesh.get_self_intersecting_triangles())
                if raw_pairs.size == 0:
                    intersecting_pairs = 0
                    intersecting_faces = 0
                    has_self_intersections = False
                else:
                    normalized_pairs = np.atleast_2d(raw_pairs)
                    intersecting_pairs = int(len(normalized_pairs))
                    intersecting_faces = int(len(np.unique(normalized_pairs.reshape(-1))))
                    has_self_intersections = True
            if has_self_intersections is None and hasattr(omesh, "is_self_intersecting"):
                has_self_intersections = bool(omesh.is_self_intersecting())
            if has_self_intersections is None:
                raise RuntimeError("Open3D self-intersection API is unavailable in the installed version.")
            return {
                "checked": True,
                "skipped": False,
                "skip_reason": None,
                "has_self_intersections": bool(has_self_intersections),
                "intersecting_pairs": intersecting_pairs,
                "intersecting_faces": intersecting_faces,
                "method": "open3d",
            }
        except Exception as exc:
            open3d_error = str(exc)
    elif o3d is not None:
        open3d_error = (
            f"skipped_open3d_for_large_mesh:{len(work.faces)}>{AUTORESEARCH_SELF_INTERSECTION_OPEN3D_MAX_FACES}"
        )

    if _try_import_pyvista() is None:
        return {
            "checked": False,
            "skipped": True,
            "skip_reason": "self_intersection_validation_unavailable",
            "has_self_intersections": None,
            "intersecting_pairs": None,
            "intersecting_faces": None,
            "method": None,
            "open3d_error": open3d_error,
        }

    try:
        _, detection_report = detect_self_intersection_pairs(
            work,
            merge_eps=merge_eps,
            max_candidate_pairs=AUTORESEARCH_SELF_INTERSECTION_MAX_CANDIDATE_PAIRS,
            status_callback=None,
        )
        return {
            "checked": not bool(detection_report.get("skipped")),
            "skipped": bool(detection_report.get("skipped")),
            "skip_reason": detection_report.get("skip_reason"),
            "has_self_intersections": bool(detection_report.get("intersecting_pairs", 0) > 0),
            "intersecting_pairs": int(detection_report.get("intersecting_pairs", 0)),
            "intersecting_faces": int(detection_report.get("intersecting_faces", 0)),
            "candidate_pairs": int(detection_report.get("candidate_pairs", 0)),
            "method": "pyvista-exact",
            "open3d_error": open3d_error,
        }
    except Exception as exc:
        return {
            "checked": False,
            "skipped": True,
            "skip_reason": f"self_intersection_validation_failed: {exc}",
            "has_self_intersections": None,
            "intersecting_pairs": None,
            "intersecting_faces": None,
            "method": "pyvista-exact",
            "open3d_error": open3d_error,
        }


def _validate_leapfrog_self_intersections_direct(mesh: trimesh.Trimesh, merge_eps: float = 1e-8) -> dict:
    work = mesh.copy()
    work.remove_unreferenced_vertices()
    if len(work.faces) == 0:
        return _validate_single_mesh_self_intersections(work, merge_eps=merge_eps)

    components = split_disconnected_components(work)
    if len(components) <= 1:
        result = _validate_single_mesh_self_intersections(work, merge_eps=merge_eps)
        result["component_count"] = 1
        result["checked_components"] = 1 if result.get("checked") else 0
        result["components_with_self_intersections"] = 1 if result.get("has_self_intersections") else 0
        return result

    aggregate = {
        "checked": True,
        "skipped": False,
        "skip_reason": None,
        "has_self_intersections": False,
        "intersecting_pairs": 0,
        "intersecting_faces": 0,
        "method": "component-wise",
        "component_count": int(len(components)),
        "checked_components": 0,
        "components_with_self_intersections": 0,
        "component_reports": [],
    }

    for index, component in enumerate(components, start=1):
        component_result = _validate_single_mesh_self_intersections(component, merge_eps=merge_eps)
        aggregate["component_reports"].append(
            {
                "component": int(index),
                "faces": int(len(component.faces)),
                "checked": bool(component_result.get("checked")),
                "skipped": bool(component_result.get("skipped")),
                "has_self_intersections": component_result.get("has_self_intersections"),
                "intersecting_pairs": component_result.get("intersecting_pairs"),
                "intersecting_faces": component_result.get("intersecting_faces"),
                "method": component_result.get("method"),
                "skip_reason": component_result.get("skip_reason"),
            }
        )

        if component_result.get("checked"):
            aggregate["checked_components"] += 1
        if aggregate.get("intersecting_pairs") is not None:
            if component_result.get("intersecting_pairs") is None:
                aggregate["intersecting_pairs"] = None
            else:
                aggregate["intersecting_pairs"] += int(component_result.get("intersecting_pairs") or 0)
        if aggregate.get("intersecting_faces") is not None:
            if component_result.get("intersecting_faces") is None:
                aggregate["intersecting_faces"] = None
            else:
                aggregate["intersecting_faces"] += int(component_result.get("intersecting_faces") or 0)

        if component_result.get("has_self_intersections"):
            aggregate["has_self_intersections"] = True
            aggregate["components_with_self_intersections"] += 1
            return aggregate

        if component_result.get("skipped"):
            aggregate["checked"] = False
            aggregate["skipped"] = True
            if aggregate.get("skip_reason") is None:
                aggregate["skip_reason"] = component_result.get("skip_reason")

    return aggregate


def validate_leapfrog_self_intersections(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    timeout_seconds: float = AUTORESEARCH_SELF_INTERSECTION_TIMEOUT_SECONDS,
) -> dict:
    effective_timeout_seconds = max(0.0, float(timeout_seconds))
    if effective_timeout_seconds <= 0.0:
        return _validate_leapfrog_self_intersections_direct(mesh, merge_eps=merge_eps)

    try:
        return run_leapfrog_self_intersection_validation_with_timeout(
            mesh,
            merge_eps=merge_eps,
            timeout_seconds=effective_timeout_seconds,
        )
    except Exception as exc:
        return {
            "checked": False,
            "skipped": True,
            "skip_reason": f"self_intersection_validation_failed: {exc}",
            "has_self_intersections": None,
            "intersecting_pairs": None,
            "intersecting_faces": None,
            "method": "subprocess",
        }


def should_skip_leapfrog_roundtrip(report: MeshReport, fast_leapfrog: bool = False) -> Optional[str]:
    if not fast_leapfrog:
        return None
    if report.vertices == 0 or report.faces == 0:
        return "empty_output"
    if report.boundary_edges > 0:
        return "boundary_edges_present"
    if report.nonmanifold_edges > 0:
        return "nonmanifold_edges_present"
    if report.degenerate_faces > 0:
        return "degenerate_faces_present"
    if report.duplicate_faces > 0:
        return "duplicate_faces_present"
    return None


def evaluate_leapfrog_acceptance(
    report: MeshReport,
    leapfrog_validation: dict,
    fidelity: dict,
    thresholds: LeapfrogAcceptanceThresholds,
) -> dict:
    failed_checks: List[str] = []
    self_intersections = leapfrog_validation.get("self_intersections") or {}
    self_intersection_checked = bool(self_intersections.get("checked"))
    self_intersection_free = self_intersections.get("has_self_intersections") is False

    topology_checks = [
        (bool(leapfrog_validation.get("roundtrip_loadable")), "roundtrip_not_loadable"),
        (report.vertices > 0, "empty_vertices"),
        (report.faces > 0, "empty_faces"),
        (report.watertight, "not_watertight"),
        (report.winding_consistent, "inconsistent_winding"),
        (report.boundary_edges == 0, "boundary_edges_present"),
        (report.nonmanifold_edges == 0, "nonmanifold_edges_present"),
        (report.degenerate_faces == 0, "degenerate_faces_present"),
        (report.duplicate_faces == 0, "duplicate_faces_present"),
        (self_intersection_checked, "self_intersection_validation_incomplete"),
        (self_intersection_free, "self_intersections_present"),
    ]
    topology_ready = True
    for passed, reason in topology_checks:
        if not passed:
            topology_ready = False
            failed_checks.append(reason)

    fidelity_checks = [
        (
            float(fidelity.get("mean_distance_normalized") or 0.0)
            <= float(thresholds.max_mean_distance_normalized),
            "mean_distance_normalized_exceeded",
        ),
        (
            float(fidelity.get("p95_distance_normalized") or 0.0)
            <= float(thresholds.max_p95_distance_normalized),
            "p95_distance_normalized_exceeded",
        ),
        (
            float(fidelity.get("component_count_delta") or 0.0)
            <= float(thresholds.max_component_count_delta),
            "component_count_delta_exceeded",
        ),
    ]
    fidelity_ready = True
    for passed, reason in fidelity_checks:
        if not passed:
            fidelity_ready = False
            failed_checks.append(reason)

    volume_ratio_delta = fidelity.get("volume_ratio_delta")
    if volume_ratio_delta is not None and float(volume_ratio_delta) > float(thresholds.max_volume_ratio_delta):
        fidelity_ready = False
        failed_checks.append("volume_ratio_delta_exceeded")

    return {
        "ready": bool(topology_ready and fidelity_ready),
        "topology_ready": bool(topology_ready),
        "fidelity_ready": bool(fidelity_ready),
        "failed_checks": failed_checks,
        "thresholds": asdict(thresholds),
    }


def score_autoresearch_candidate(
    candidate: HealSearchCandidate,
    output_report: MeshReport,
    leapfrog_validation: dict,
    fidelity: dict,
    acceptance: dict,
    runtime_seconds: float = 0.0,
) -> dict:
    leapfrog_ready = bool(acceptance.get("ready"))
    self_intersections = leapfrog_validation.get("self_intersections") or {}

    topology_penalty = 0.0
    if output_report.vertices == 0 or output_report.faces == 0:
        topology_penalty += 500_000.0
    if not leapfrog_validation.get("roundtrip_loadable"):
        topology_penalty += 500_000.0
    if not output_report.watertight:
        topology_penalty += 250_000.0
    if not output_report.winding_consistent:
        topology_penalty += 50_000.0
    topology_penalty += float(output_report.boundary_edges) * 2_000.0
    topology_penalty += float(output_report.nonmanifold_edges) * 50_000.0
    topology_penalty += float(output_report.degenerate_faces) * 10_000.0
    topology_penalty += float(output_report.duplicate_faces) * 2_500.0
    if self_intersections.get("checked") is False:
        topology_penalty += 250_000.0
    if self_intersections.get("has_self_intersections"):
        intersecting_pairs = self_intersections.get("intersecting_pairs")
        topology_penalty += 300_000.0
        if intersecting_pairs is not None:
            topology_penalty += float(intersecting_pairs) * 10_000.0

    fidelity_penalty = 0.0
    fidelity_penalty += float(fidelity["mean_distance_normalized"]) * 10_000.0
    fidelity_penalty += float(fidelity["p95_distance_normalized"]) * 8_000.0
    fidelity_penalty += float(fidelity["max_distance_normalized"]) * 1_500.0
    fidelity_penalty += float(fidelity["centroid_drift_normalized"]) * 2_000.0
    fidelity_penalty += float(fidelity["extents_drift_normalized"]) * 1_500.0
    fidelity_penalty += float(fidelity["area_ratio_delta"]) * 2_000.0
    fidelity_penalty += float(fidelity["component_count_delta"]) * 750.0
    volume_ratio_delta = fidelity.get("volume_ratio_delta")
    if volume_ratio_delta is not None:
        fidelity_penalty += float(volume_ratio_delta) * 4_000.0

    complexity_penalty = float(candidate.enabled_step_count()) * 25.0
    if candidate.aggressive:
        complexity_penalty += 250.0

    runtime_penalty = min(max(float(runtime_seconds), 0.0), 600.0)
    total_score = topology_penalty + fidelity_penalty + complexity_penalty + runtime_penalty
    ranking = [
        0 if leapfrog_ready else 1,
        0 if acceptance.get("fidelity_ready") else 1,
        float(topology_penalty),
        float(fidelity_penalty),
        float(complexity_penalty),
        float(runtime_penalty),
        float(total_score),
    ]
    return {
        "leapfrog_ready": leapfrog_ready,
        "topology_penalty": float(topology_penalty),
        "fidelity_penalty": float(fidelity_penalty),
        "complexity_penalty": float(complexity_penalty),
        "runtime_penalty": float(runtime_penalty),
        "total": float(total_score),
        "ranking": ranking,
        "acceptance": acceptance,
    }


def serialize_heal_search_candidate(candidate: HealSearchCandidate) -> dict:
    return {
        "name": candidate.name,
        "rebuild_triangles": bool(candidate.rebuild_triangles),
        "nonmanifold_edge_repair": bool(candidate.nonmanifold_edge_repair),
        "localized_intersection_repair": bool(candidate.localized_intersection_repair),
        "point_cloud_rebuild": candidate.point_cloud_rebuild,
        "distance_model": candidate.distance_model,
        "make_watertight": bool(candidate.make_watertight),
        "distance_offset_ratio": float(candidate.distance_offset_ratio),
        "distance_grid_spacing_ratio": float(candidate.distance_grid_spacing_ratio),
        "aggressive": bool(candidate.aggressive),
        "enabled_step_count": int(candidate.enabled_step_count()),
    }


def estimate_nonmanifold_edge_repair_radius(
    mesh: trimesh.Trimesh,
    edge_lengths: np.ndarray,
    merge_eps: float,
) -> float:
    positive_lengths = np.asarray(edge_lengths, dtype=float)
    positive_lengths = positive_lengths[positive_lengths > 1e-15]
    if len(positive_lengths) == 0:
        return max(merge_eps * 100.0, 1e-6)

    diagonal = 0.0
    if len(mesh.vertices) > 0:
        bounds = np.asarray(mesh.bounds, dtype=float)
        diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))

    characteristic_length = float(np.median(positive_lengths))
    min_radius = max(merge_eps * 100.0, diagonal * 5e-4 if diagonal > 0.0 else 0.0, characteristic_length * 0.1)
    max_radius = diagonal * 0.02 if diagonal > 0.0 else characteristic_length
    radius = max(min_radius, characteristic_length * 0.25)
    if max_radius > 0.0:
        radius = min(radius, max_radius)
    return float(radius)


def repair_nonmanifold_edges_with_cylinders(
    mesh: trimesh.Trimesh,
    radius: float = 0.0,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    if radius < 0.0:
        raise ValueError("Non-manifold edge repair radius must be greater than or equal to zero.")

    before = mesh_report(mesh, area_eps=area_eps)
    nonmanifold_edges, edge_counts = detect_nonmanifold_edges(mesh)
    if len(nonmanifold_edges) == 0:
        emit_status(status_callback, "Mesh has no non-manifold edges; skipping cylindrical edge repair")
        return mesh, {
            "requested": True,
            "skipped": True,
            "reason": "no_nonmanifold_edges",
            "before": asdict(before),
            "after": asdict(before),
        }

    vertices = np.asarray(mesh.vertices, dtype=float)
    edge_points = vertices[nonmanifold_edges]
    edge_vectors = edge_points[:, 1] - edge_points[:, 0]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    base_radius = float(radius) if radius > 0.0 else estimate_nonmanifold_edge_repair_radius(mesh, edge_lengths, merge_eps)
    cylinder_sections = 24

    emit_status(
        status_callback,
        f"Detected {len(nonmanifold_edges)} non-manifold edges in the current mesh topology; repairing them with cylindrical sleeves",
    )
    emit_status(
        status_callback,
        "This non-manifold edge count is recomputed from the rebuilt mesh and can be much larger than the imported hint count.",
    )

    sleeves: List[trimesh.Trimesh] = []
    repaired_edge_count = 0
    skipped_short_edges = 0
    used_radii: List[float] = []
    for edge_index, _edge in enumerate(nonmanifold_edges):
        start = edge_points[edge_index, 0]
        end = edge_points[edge_index, 1]
        edge_length = float(edge_lengths[edge_index])
        if edge_length <= merge_eps:
            skipped_short_edges += 1
            continue

        direction = (end - start) / edge_length
        edge_radius = base_radius if radius > 0.0 else max(base_radius, edge_length * 0.1)
        extension = max(edge_radius, edge_length * 0.05)
        segment = np.vstack([start - direction * extension, end + direction * extension])
        sleeve = trimesh.creation.cylinder(
            radius=float(edge_radius),
            segment=segment,
            sections=cylinder_sections,
        )
        sleeves.append(as_mesh(sleeve))
        repaired_edge_count += 1
        used_radii.append(float(edge_radius))

    if not sleeves:
        emit_status(status_callback, "Cylindrical edge repair skipped because no valid non-manifold edges could be sleeved")
        return mesh, {
            "requested": True,
            "skipped": True,
            "reason": "no_valid_edge_segments",
            "before": asdict(before),
            "after": asdict(before),
            "requested_radius": float(radius),
            "base_radius": float(base_radius),
            "nonmanifold_edges_detected": int(len(nonmanifold_edges)),
            "nonmanifold_edge_face_counts": [int(count) for count in edge_counts.tolist()],
            "skipped_short_edges": int(skipped_short_edges),
        }

    sleeve_mesh = combine_meshes(sleeves)
    sleeve_mesh = finalize_healed_mesh(
        sleeve_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=True,
        status_callback=None,
    )

    union_backend = "manifold"
    union_fallback_reason = None
    try:
        unioned = trimesh.boolean.union([mesh.copy(), sleeve_mesh.copy()], engine="manifold")
        repaired = as_mesh(unioned)
        repaired.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(repaired)
    except Exception as manifold_exc:
        union_backend = "vtk"
        union_fallback_reason = f"manifold failed: {manifold_exc}"
        try:
            left_poly = mesh_to_polydata(mesh)
            right_poly = mesh_to_polydata(sleeve_mesh)
            repaired = polydata_to_mesh(left_poly.boolean_union(right_poly))
        except Exception as vtk_exc:
            union_backend = "concatenate"
            union_fallback_reason = f"{union_fallback_reason}; vtk failed: {vtk_exc}"
            repaired = combine_meshes([mesh, sleeve_mesh])

    repaired = finalize_healed_mesh(
        repaired,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=True,
        status_callback=None,
    )
    after = mesh_report(repaired, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Non-manifold edge repair output", after)
    return repaired, {
        "requested": True,
        "skipped": False,
        "before": asdict(before),
        "after": asdict(after),
        "requested_radius": float(radius),
        "base_radius": float(base_radius),
        "radius_range": [float(min(used_radii)), float(max(used_radii))] if used_radii else [0.0, 0.0],
        "nonmanifold_edges_detected": int(len(nonmanifold_edges)),
        "nonmanifold_edges_repaired": int(repaired_edge_count),
        "nonmanifold_edge_face_counts": [int(count) for count in edge_counts.tolist()],
        "skipped_short_edges": int(skipped_short_edges),
        "sleeve_faces": int(len(sleeve_mesh.faces)),
        "union_backend": union_backend,
        "union_fallback_reason": union_fallback_reason,
        "nonmanifold_edges_removed": int(before.nonmanifold_edges - after.nonmanifold_edges),
    }


def find_nearest_edge(mesh: trimesh.Trimesh, point: Sequence[float] | np.ndarray) -> dict:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("The current mesh has no edges to pick from.")

    vertices = np.asarray(mesh.vertices, dtype=float)
    target = np.asarray(point, dtype=float).reshape(3)
    starts = vertices[edges[:, 0]]
    ends = vertices[edges[:, 1]]
    segments = ends - starts
    lengths_sq = np.einsum("ij,ij->i", segments, segments)
    valid = lengths_sq > 1e-20
    factors = np.zeros(len(edges), dtype=float)
    factors[valid] = np.clip(
        np.einsum("ij,ij->i", np.broadcast_to(target, starts.shape) - starts, segments)[valid] / lengths_sq[valid],
        0.0,
        1.0,
    )
    closest_points = starts + segments * factors[:, None]
    deltas = closest_points - target
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    best_index = int(np.argmin(distances_sq))
    best_edge = edges[best_index]
    return {
        "edge_vertex_ids": [int(best_edge[0]), int(best_edge[1])],
        "distance": float(math.sqrt(max(0.0, distances_sq[best_index]))),
        "point": closest_points[best_index].astype(float).tolist(),
        "edge_points": vertices[best_edge].astype(float).tolist(),
    }


def collect_mesh_issues(
    mesh: trimesh.Trimesh,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    max_duplicate_groups: int = 250,
    max_degenerate_faces: int = 1000,
    max_nonmanifold_edges: int = 1000,
    max_boundary_loops: int = 250,
    max_detailed_faces: int = 200000,
) -> dict:
    issues: List[dict] = []
    truncated = {
        "duplicate_groups": False,
        "degenerate_faces": False,
        "nonmanifold_edges": False,
        "boundary_loops": False,
    }
    skipped = {
        "duplicate_groups": False,
        "degenerate_faces": False,
    }

    face_centers = np.asarray(mesh.triangles_center, dtype=float) if len(mesh.faces) > 0 else np.zeros((0, 3), dtype=float)

    run_expensive_face_checks = len(mesh.faces) <= max_detailed_faces
    duplicate_count = 0
    duplicate_groups: Dict[Tuple, List[int]] = {}
    degenerate_faces = np.zeros((0,), dtype=np.int64)
    if run_expensive_face_checks:
        duplicate_count, duplicate_groups = detect_duplicate_faces(mesh, decimals=dedup_decimals)
        degenerate_faces = detect_degenerate_faces(mesh, eps=area_eps)
    else:
        skipped["duplicate_groups"] = True
        skipped["degenerate_faces"] = True

    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    edge_counts = np.zeros((0,), dtype=np.int64)
    nonmanifold_edge_indices = np.zeros((0,), dtype=np.int64)
    boundary_loops: List[np.ndarray] = []
    boundary_edge_count = 0
    nonmanifold_edge_count = 0

    if len(unique_edges) > 0:
        edge_inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
        edge_counts = np.bincount(edge_inverse, minlength=len(unique_edges))
        nonmanifold_edge_indices = np.flatnonzero(edge_counts > 2)
        boundary_edge_count = int(np.sum(edge_counts == 1))
        nonmanifold_edge_count = int(len(nonmanifold_edge_indices))
        boundary_loops = extract_boundary_loops(mesh)

    report = MeshReport(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        volume=safe_mesh_volume(mesh),
        area=float(mesh.area),
        degenerate_faces=int(len(degenerate_faces)),
        duplicate_faces=int(duplicate_count),
        nonmanifold_edges=nonmanifold_edge_count,
        boundary_edges=boundary_edge_count,
    )

    duplicate_face_groups = [sorted(int(face_id) for face_id in face_ids) for face_ids in duplicate_groups.values() if len(face_ids) > 1]
    duplicate_face_groups.sort(key=lambda face_ids: (-len(face_ids), face_ids[0]))
    if len(duplicate_face_groups) > max_duplicate_groups:
        truncated["duplicate_groups"] = True
    for index, face_ids in enumerate(duplicate_face_groups[:max_duplicate_groups], start=1):
        focus_point = np.mean(face_centers[np.asarray(face_ids, dtype=np.int64)], axis=0) if face_ids else np.zeros(3, dtype=float)
        issues.append(
            {
                "id": f"duplicate-group-{index}",
                "category": "duplicate-faces",
                "label": f"Duplicate group {index}",
                "description": f"{len(face_ids)} triangles share the same geometry.",
                "face_ids": face_ids,
                "point": focus_point.astype(float).tolist(),
            }
        )

    if len(degenerate_faces) > max_degenerate_faces:
        truncated["degenerate_faces"] = True
    for face_id in degenerate_faces[:max_degenerate_faces]:
        face_index = int(face_id)
        issues.append(
            {
                "id": f"degenerate-face-{face_index}",
                "category": "degenerate-faces",
                "label": f"Degenerate face {face_index}",
                "description": "Triangle area is below the configured area epsilon.",
                "face_ids": [face_index],
                "point": face_centers[face_index].astype(float).tolist() if len(face_centers) > face_index else [0.0, 0.0, 0.0],
            }
        )

    if len(unique_edges) > 0:
        vertices = np.asarray(mesh.vertices, dtype=float)

        if len(nonmanifold_edge_indices) > max_nonmanifold_edges:
            truncated["nonmanifold_edges"] = True
        for edge_index in nonmanifold_edge_indices[:max_nonmanifold_edges]:
            edge = unique_edges[int(edge_index)]
            edge_points = vertices[edge]
            issues.append(
                {
                    "id": f"nonmanifold-edge-{int(edge_index)}",
                    "category": "nonmanifold-edges",
                    "label": f"Non-manifold edge {int(edge_index)}",
                    "description": f"Edge is shared by {int(edge_counts[int(edge_index)])} triangles.",
                    "edge_vertex_ids": [int(edge[0]), int(edge[1])],
                    "polyline_points": edge_points.astype(float).tolist(),
                    "point": np.mean(edge_points, axis=0).astype(float).tolist(),
                }
            )
        if len(boundary_loops) > max_boundary_loops:
            truncated["boundary_loops"] = True
        for index, loop in enumerate(boundary_loops[:max_boundary_loops], start=1):
            loop_vertices = np.asarray(loop, dtype=np.int64)
            loop_points = vertices[loop_vertices]
            closed_loop_points = np.vstack([loop_points, loop_points[0]]) if len(loop_points) > 0 else loop_points
            edge_vectors = np.roll(loop_points, -1, axis=0) - loop_points if len(loop_points) > 1 else np.zeros((0, 3), dtype=float)
            loop_length = float(np.sum(np.linalg.norm(edge_vectors, axis=1))) if len(edge_vectors) > 0 else 0.0
            issues.append(
                {
                    "id": f"boundary-loop-{index}",
                    "category": "boundary-loops",
                    "label": f"Boundary loop {index}",
                    "description": f"Open boundary loop with {len(loop_vertices)} vertices and length {loop_length:.3f}.",
                    "vertex_ids": [int(vertex_id) for vertex_id in loop_vertices.tolist()],
                    "polyline_points": closed_loop_points.astype(float).tolist(),
                    "point": np.mean(loop_points, axis=0).astype(float).tolist() if len(loop_points) > 0 else [0.0, 0.0, 0.0],
                }
            )

    summary = asdict(report)
    if skipped["duplicate_groups"]:
        summary["duplicate_faces"] = None
    if skipped["degenerate_faces"]:
        summary["degenerate_faces"] = None

    category_counts = {
        "duplicate_faces": None if skipped["duplicate_groups"] else int(duplicate_count),
        "duplicate_groups": int(len(duplicate_face_groups)),
        "degenerate_faces": None if skipped["degenerate_faces"] else int(len(degenerate_faces)),
        "nonmanifold_edges": int(nonmanifold_edge_count),
        "boundary_loops": int(len(boundary_loops)),
        "boundary_edges": int(report.boundary_edges),
    }
    return {
        "summary": summary,
        "categories": category_counts,
        "issue_count": int(len(issues)),
        "issues": issues,
        "truncated": truncated,
        "skipped": skipped,
    }


def emit_mesh_topology_status(status_callback: StatusCallback, label: str, report: MeshReport) -> None:
    if report.watertight:
        emit_status(status_callback, f"{label} surface is closed/watertight (boundary edges: {report.boundary_edges})")
        return

    emit_status(status_callback, f"{label} surface is open (boundary edges: {report.boundary_edges})")


def sanitize_intermediate_mesh(
    mesh: trimesh.Trimesh,
    *,
    stage: str,
    merge_eps: float,
    area_eps: float,
    dedup_decimals: int,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    before = mesh_report(mesh, area_eps=area_eps)
    work = merge_nearby_vertices(mesh, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)
    work.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(work)
    except Exception:
        pass
    after = mesh_report(work, area_eps=area_eps)

    removed_duplicate_faces = max(0, int(before.duplicate_faces) - int(after.duplicate_faces))
    removed_degenerate_faces = max(0, int(before.degenerate_faces) - int(after.degenerate_faces))
    repaired_invalid_triangles = bool(removed_duplicate_faces or removed_degenerate_faces)
    if repaired_invalid_triangles:
        emit_status(
            status_callback,
            f"Sanitized {stage}: removed {removed_duplicate_faces} duplicate and {removed_degenerate_faces} degenerate triangles",
        )
    else:
        emit_status(status_callback, f"Validated intermediate mesh after {stage}")

    return work, {
        "stage": stage,
        "before": asdict(before),
        "after": asdict(after),
        "removed_duplicate_faces": int(removed_duplicate_faces),
        "removed_degenerate_faces": int(removed_degenerate_faces),
        "repaired_invalid_triangles": repaired_invalid_triangles,
    }


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


def normalize_point_cloud_rebuild_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in POINT_CLOUD_REBUILD_MODES:
        raise ValueError(
            f"Unsupported point-cloud rebuild mode: {mode}. Expected one of: {', '.join(POINT_CLOUD_REBUILD_MODES)}"
        )
    return normalized


def normalize_distance_model_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DISTANCE_MODEL_MODES:
        raise ValueError(
            f"Unsupported distance model mode: {mode}. Expected one of: {', '.join(DISTANCE_MODEL_MODES)}"
        )
    return normalized


def resolve_distance_hull_spacing(
    bounds: np.ndarray,
    offset_distance: float,
    requested_spacing: float,
    merge_eps: float,
    max_grid_points: int = 4_000_000,
) -> Tuple[float, np.ndarray, float]:
    extents = np.asarray(bounds[1] - bounds[0], dtype=float)
    diagonal = float(np.linalg.norm(extents))
    positive_extent = extents[extents > 0.0]
    min_extent = float(positive_extent.min()) if len(positive_extent) > 0 else 0.0

    if requested_spacing > 0.0:
        spacing = float(requested_spacing)
    else:
        auto_candidates = [
            offset_distance / 8.0,
            diagonal / 192.0 if diagonal > 0.0 else 0.0,
            min_extent / 96.0 if min_extent > 0.0 else 0.0,
        ]
        positive_candidates = [candidate for candidate in auto_candidates if candidate > 0.0]
        spacing = min(positive_candidates) if positive_candidates else max(offset_distance / 4.0, 1.0)

    spacing = max(float(spacing), merge_eps * 100.0, 1e-6)
    padding = float(offset_distance + (3.0 * spacing))
    dimensions = np.maximum(3, np.ceil((extents + (2.0 * padding)) / spacing).astype(int) + 1)

    while int(np.prod(dimensions, dtype=np.int64)) > max_grid_points:
        spacing *= 1.25
        padding = float(offset_distance + (3.0 * spacing))
        dimensions = np.maximum(3, np.ceil((extents + (2.0 * padding)) / spacing).astype(int) + 1)

    return float(spacing), np.asarray(dimensions, dtype=np.int64), float(padding)


def build_distance_hull(
    mesh: trimesh.Trimesh,
    offset_distance: float,
    grid_spacing: float = 0.0,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    pv = _try_import_pyvista()
    if pv is None:
        raise RuntimeError("Distance hull generation requires pyvista. Install requirements first.")
    if offset_distance <= 0.0:
        raise ValueError("Distance hull offset must be greater than zero.")

    work = mesh.copy()
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)
    work.remove_unreferenced_vertices()

    report = {
        "mode": "distance-hull",
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "preprocessed_vertices": int(len(work.vertices)),
        "preprocessed_faces": int(len(work.faces)),
        "offset_distance": float(offset_distance),
        "requested_grid_spacing": float(grid_spacing),
        "grid_spacing": 0.0,
        "grid_dimensions": [0, 0, 0],
        "grid_points": 0,
        "padding": 0.0,
        "output_vertices": 0,
        "output_faces": 0,
        "watertight": False,
    }
    if len(work.faces) == 0:
        return make_empty_mesh(), report

    bounds = np.asarray(work.bounds, dtype=float)
    resolved_spacing, grid_dimensions, padding = resolve_distance_hull_spacing(
        bounds=bounds,
        offset_distance=offset_distance,
        requested_spacing=grid_spacing,
        merge_eps=merge_eps,
    )
    report["grid_spacing"] = float(resolved_spacing)
    report["grid_dimensions"] = [int(value) for value in grid_dimensions.tolist()]
    report["grid_points"] = int(np.prod(grid_dimensions, dtype=np.int64))
    report["padding"] = float(padding)

    padded_min = bounds[0] - padding
    emit_status(
        status_callback,
        "Sampling unsigned distance field on "
        f"{int(grid_dimensions[0])}x{int(grid_dimensions[1])}x{int(grid_dimensions[2])} grid",
    )
    distance_grid = pv.ImageData(
        dimensions=tuple(int(value) for value in grid_dimensions.tolist()),
        spacing=(resolved_spacing, resolved_spacing, resolved_spacing),
        origin=tuple(float(value) for value in padded_min.tolist()),
    )
    distance_grid = distance_grid.compute_implicit_distance(mesh_to_polydata(work))
    distance_grid["unsigned_distance"] = np.abs(np.asarray(distance_grid["implicit_distance"], dtype=float))

    emit_status(status_callback, "Extracting distance hull iso-surface")
    hull_poly = distance_grid.contour(isosurfaces=[float(offset_distance)], scalars="unsigned_distance")
    if hull_poly.n_points == 0 or hull_poly.n_cells == 0:
        raise RuntimeError("Distance hull generation produced an empty iso-surface. Increase grid resolution or offset.")

    hull_surface = hull_poly.extract_surface(algorithm="dataset_surface").triangulate().clean(
        tolerance=merge_eps,
        absolute=True,
    )
    if hull_surface.n_points == 0 or hull_surface.n_cells == 0:
        raise RuntimeError("Distance hull generation produced an empty surface after cleanup.")

    hull_mesh = polydata_to_mesh(hull_surface)
    hull_mesh = finalize_healed_mesh(
        hull_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=False,
        status_callback=None,
    )

    report["output_vertices"] = int(len(hull_mesh.vertices))
    report["output_faces"] = int(len(hull_mesh.faces))
    report["watertight"] = bool(hull_mesh.is_watertight)
    return hull_mesh, report


def _estimate_min_unique_edge_length(mesh: trimesh.Trimesh) -> float:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        return 0.0

    vertices = np.asarray(mesh.vertices, dtype=float)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    finite_lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if len(finite_lengths) == 0:
        return 0.0
    return float(np.min(finite_lengths))


def _compute_surface_shell_vertex_normals(
    mesh: trimesh.Trimesh,
    area_eps: float,
    fallback_scale: float,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=float)

    try:
        vertex_normals = np.array(mesh.vertex_normals, dtype=float, copy=True)
    except Exception:
        vertex_normals = np.zeros_like(vertices)

    if vertex_normals.shape != vertices.shape:
        vertex_normals = np.zeros_like(vertices)

    face_normals = np.asarray(mesh.face_normals, dtype=float)
    accumulated = np.zeros_like(vertices)
    for face, face_normal in zip(np.asarray(mesh.faces, dtype=np.int64), face_normals):
        normal = np.asarray(face_normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= area_eps:
            continue
        accumulated[face] += normal / length

    accumulated_lengths = np.linalg.norm(accumulated, axis=1)
    accumulated_valid = accumulated_lengths > max(area_eps, 1e-15)
    if np.any(accumulated_valid):
        vertex_normals[accumulated_valid] = accumulated[accumulated_valid] / accumulated_lengths[accumulated_valid][:, None]

    lengths = np.linalg.norm(vertex_normals, axis=1)
    invalid_mask = (~np.isfinite(vertex_normals).all(axis=1)) | (lengths <= max(area_eps, 1e-15))
    if np.any(invalid_mask):
        center = np.asarray(mesh.bounding_box.centroid if len(vertices) > 0 else np.zeros(3), dtype=float)
        fallback = vertices[invalid_mask] - center
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        usable_fallback = fallback_lengths > max(fallback_scale * 1e-9, 1e-15)
        if np.any(usable_fallback):
            fallback[usable_fallback] = fallback[usable_fallback] / fallback_lengths[usable_fallback][:, None]
        if np.any(~usable_fallback):
            fallback[~usable_fallback] = np.array([0.0, 0.0, 1.0], dtype=float)
        vertex_normals[invalid_mask] = fallback
        lengths = np.linalg.norm(vertex_normals, axis=1)

    return vertex_normals / np.maximum(lengths[:, None], 1e-15)


def build_surface_shell(
    mesh: trimesh.Trimesh,
    offset_distance: float,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    if offset_distance <= 0.0:
        raise ValueError("Surface shell offset must be greater than zero.")

    work = mesh.copy()
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)
    work.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(work)
    except Exception:
        pass

    boundary_loops = extract_boundary_loops(work)
    min_edge_length = _estimate_min_unique_edge_length(work)
    report = {
        "mode": "surface-shell",
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "preprocessed_vertices": int(len(work.vertices)),
        "preprocessed_faces": int(len(work.faces)),
        "offset_distance": float(offset_distance),
        "boundary_loops_detected": int(len(boundary_loops)),
        "boundary_loops_stitched": 0,
        "stitched_side_faces": 0,
        "min_input_edge_length": float(min_edge_length),
        "warnings": [],
        "assembly_validation": None,
        "output_vertices": 0,
        "output_faces": 0,
        "watertight": False,
    }
    if len(work.faces) == 0:
        return make_empty_mesh(), report

    if min_edge_length > 0.0 and offset_distance >= min_edge_length:
        report["warnings"].append(
            "Offset distance is at least the minimum input edge length; sharp features may self-intersect or collapse."
        )

    emit_status(status_callback, "Computing offset skins for surface shell")
    vertices = np.asarray(work.vertices, dtype=float)
    faces = np.asarray(work.faces, dtype=np.int64)
    shell_normals = _compute_surface_shell_vertex_normals(
        work,
        area_eps=area_eps,
        fallback_scale=max(estimate_mesh_scale(work), merge_eps),
    )
    outer_vertices = vertices + (shell_normals * float(offset_distance))
    inner_vertices = vertices - (shell_normals * float(offset_distance))
    vertex_count = int(len(vertices))

    outer_faces = faces.copy()
    inner_faces = (faces[:, ::-1] + vertex_count).copy()
    side_faces: List[List[int]] = []
    for loop in boundary_loops:
        loop_indices = [int(index) for index in np.asarray(loop, dtype=np.int64).tolist()]
        if len(loop_indices) < 3:
            continue
        report["boundary_loops_stitched"] += 1
        for edge_index in range(len(loop_indices)):
            start = loop_indices[edge_index]
            end = loop_indices[(edge_index + 1) % len(loop_indices)]
            side_faces.append([start, end, end + vertex_count])
            side_faces.append([start, end + vertex_count, start + vertex_count])

    report["stitched_side_faces"] = int(len(side_faces))
    combined_vertices = np.vstack([outer_vertices, inner_vertices])
    face_groups = [outer_faces, inner_faces]
    if side_faces:
        face_groups.append(np.asarray(side_faces, dtype=np.int64))
    combined_faces = np.vstack(face_groups)
    shell_mesh = trimesh.Trimesh(
        vertices=np.asarray(combined_vertices, dtype=float),
        faces=np.asarray(combined_faces, dtype=np.int64),
        process=False,
    )
    shell_mesh, assembly_validation = sanitize_intermediate_mesh(
        shell_mesh,
        stage="surface shell assembly",
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        status_callback=status_callback,
    )
    report["assembly_validation"] = assembly_validation
    shell_mesh = finalize_healed_mesh(
        shell_mesh,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=False,
        status_callback=None,
    )

    after = mesh_report(shell_mesh, area_eps=area_eps)
    if after.boundary_edges > 0:
        report["warnings"].append(
            f"Surface shell output still has {after.boundary_edges} boundary edges after stitching."
        )
    if after.nonmanifold_edges > 0 or after.degenerate_faces > 0 or after.duplicate_faces > 0:
        report["warnings"].append(
            "Surface shell output contains invalid topology; the chosen offset may have introduced self-intersections or collapsed features."
        )

    report["output_vertices"] = int(len(shell_mesh.vertices))
    report["output_faces"] = int(len(shell_mesh.faces))
    report["watertight"] = bool(after.watertight)
    return shell_mesh, report


def estimate_point_cloud_neighbor_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        finite = nearest[np.isfinite(nearest)]
        if len(finite) == 0:
            return 0.0
        return float(np.median(finite))

    tree = cKDTree(np.asarray(points, dtype=float))
    distances, _ = tree.query(np.asarray(points, dtype=float), k=2)
    if distances.ndim != 2 or distances.shape[1] < 2:
        return 0.0
    finite = distances[:, 1][np.isfinite(distances[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float(np.median(finite))


def orient_normals_from_mesh_centroid(points: np.ndarray, normals: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return normals

    oriented = np.asarray(normals, dtype=float).copy()
    directions = np.asarray(points, dtype=float) - np.asarray(centroid, dtype=float)
    dot_products = np.einsum("ij,ij->i", oriented, directions)
    flip_mask = dot_products < 0.0
    oriented[flip_mask] *= -1.0
    return oriented


def rebuild_solid_from_triangle_centers(
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    o3d = _try_import_open3d()
    pv = _try_import_pyvista()
    if o3d is None and pv is None:
        raise RuntimeError("Point-cloud rebuild requires open3d or pyvista. Install requirements first.")

    work = mesh.copy()
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    work = remove_degenerate_faces(work, eps=area_eps)
    work.remove_unreferenced_vertices()

    report = {
        "mode": "triangle-centers-poisson",
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "preprocessed_vertices": int(len(work.vertices)),
        "preprocessed_faces": int(len(work.faces)),
        "sample_points": 0,
        "degenerate_faces_removed": int(len(mesh.faces) - len(work.faces)) if len(mesh.faces) >= len(work.faces) else 0,
        "neighbor_distance": 0.0,
        "poisson_depth": 0,
        "density_trim_quantile": 0.05,
        "backend": "open3d" if o3d is not None else "pyvista",
        "output_vertices": 0,
        "output_faces": 0,
        "watertight": False,
    }
    if len(work.faces) == 0:
        return make_empty_mesh(), report

    emit_status(status_callback, "Building oriented point cloud from triangle centers")
    triangles = np.asarray(work.triangles, dtype=float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_mask = normal_lengths > area_eps
    if not np.any(valid_mask):
        return make_empty_mesh(), report

    triangles = triangles[valid_mask]
    normals = normals[valid_mask] / normal_lengths[valid_mask][:, None]
    points = triangles.mean(axis=1)
    report["sample_points"] = int(len(points))

    mesh_centroid = np.asarray(work.bounding_box.centroid if len(work.vertices) > 0 else np.zeros(3), dtype=float)
    normals = orient_normals_from_mesh_centroid(points, normals, mesh_centroid)

    neighbor_distance = estimate_point_cloud_neighbor_distance(points)
    report["neighbor_distance"] = float(neighbor_distance)
    point_count = int(len(points))
    if point_count < 10:
        raise RuntimeError("Point-cloud rebuild needs at least 10 valid triangle centers after preprocessing.")

    if o3d is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        point_cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))

        try:
            tangent_k = max(10, min(64, point_count - 1))
            point_cloud.orient_normals_consistent_tangent_plane(tangent_k)
        except Exception:
            point_cloud.normals = o3d.utility.Vector3dVector(
                orient_normals_from_mesh_centroid(np.asarray(point_cloud.points), np.asarray(point_cloud.normals), mesh_centroid)
            )

        poisson_depth = 8
        if point_count < 2000:
            poisson_depth = 7
        elif point_count > 20000:
            poisson_depth = 9
        report["poisson_depth"] = int(poisson_depth)

        emit_status(status_callback, "Reconstructing surface from oriented point cloud")
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=poisson_depth,
            linear_fit=True,
        )
        densities = np.asarray(densities, dtype=float)
        if len(densities) == 0 or len(np.asarray(poisson_mesh.triangles)) == 0:
            raise RuntimeError("Point-cloud rebuild produced an empty Poisson surface.")

        density_threshold = float(np.quantile(densities, report["density_trim_quantile"]))
        low_density_vertices = np.where(densities < density_threshold)[0]
        if len(low_density_vertices) > 0:
            poisson_mesh.remove_vertices_by_mask(low_density_vertices)

        bbox = point_cloud.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=float)
        padding = max(float(np.linalg.norm(extent)) * 0.05, neighbor_distance * 2.0, merge_eps * 100.0)
        crop_box = o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound - padding, bbox.max_bound + padding)
        poisson_mesh = poisson_mesh.crop(crop_box)

        poisson_mesh.remove_duplicated_vertices()
        poisson_mesh.remove_duplicated_triangles()
        poisson_mesh.remove_degenerate_triangles()
        poisson_mesh.remove_non_manifold_edges()
        poisson_mesh.remove_unreferenced_vertices()
        poisson_mesh.compute_triangle_normals()
        poisson_mesh.compute_vertex_normals()

        rebuilt = trimesh.Trimesh(
            vertices=np.asarray(poisson_mesh.vertices, dtype=float).copy(),
            faces=np.asarray(poisson_mesh.triangles, dtype=np.int64).copy(),
            process=False,
        )
    else:
        emit_status(status_callback, "Reconstructing surface from triangle centers with PyVista")
        point_poly = pv.PolyData(np.asarray(points, dtype=float))
        point_poly.point_data["Normals"] = np.asarray(normals, dtype=float)
        sample_spacing = max(neighbor_distance, merge_eps * 10.0)
        reconstructed = point_poly.reconstruct_surface(sample_spacing=sample_spacing)
        surface = reconstructed.extract_surface(algorithm="dataset_surface").triangulate().clean()
        if surface.n_points == 0 or surface.n_cells == 0:
            raise RuntimeError("Point-cloud rebuild produced an empty PyVista surface.")
        rebuilt = polydata_to_mesh(surface)
    rebuilt = finalize_healed_mesh(
        rebuilt,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        resolve_component_overlaps=False,
        status_callback=None,
    )

    report["output_vertices"] = int(len(rebuilt.vertices))
    report["output_faces"] = int(len(rebuilt.faces))
    report["watertight"] = bool(rebuilt.is_watertight)
    return rebuilt, report


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
        with suppress_vtk_warnings():
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
    return partition_face_components(mesh, intersecting_faces, extra_face_neighbors=intersecting_pairs)


def partition_face_components(
    mesh: trimesh.Trimesh,
    face_indices: Sequence[int],
    extra_face_neighbors: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[List[int]]:
    if not face_indices:
        return []

    remaining = set(int(face_index) for face_index in face_indices)
    vertex_to_faces: Dict[int, List[int]] = defaultdict(list)
    face_neighbors: Dict[int, set[int]] = defaultdict(set)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for face_index in remaining:
        for vertex_index in faces[face_index]:
            vertex_to_faces[int(vertex_index)].append(face_index)
    for first_face, second_face in extra_face_neighbors or ():
        first = int(first_face)
        second = int(second_face)
        if first not in remaining or second not in remaining:
            continue
        face_neighbors[first].add(second)
        face_neighbors[second].add(first)

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


def expand_face_selection(mesh: trimesh.Trimesh, face_indices: Sequence[int], rings: int = 1) -> List[int]:
    if len(mesh.faces) == 0 or not face_indices:
        return []

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_to_faces: Dict[int, List[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[int(vertex_index)].append(int(face_index))

    selected = {int(face_index) for face_index in face_indices if 0 <= int(face_index) < len(faces)}
    frontier = set(selected)
    for _ in range(max(0, int(rings))):
        next_frontier: set[int] = set()
        for face_index in frontier:
            for vertex_index in faces[face_index]:
                for neighbor_face in vertex_to_faces[int(vertex_index)]:
                    if neighbor_face in selected:
                        continue
                    selected.add(neighbor_face)
                    next_frontier.add(neighbor_face)
        if not next_frontier:
            break
        frontier = next_frontier

    return sorted(selected)


def collect_seed_face_indices_from_external_issues(
    mesh: trimesh.Trimesh,
    external_hints: Optional[dict],
) -> Tuple[List[int], dict]:
    if not external_hints:
        return [], {
            "seed_faces": 0,
            "seeded_issues": 0,
            "issue_categories": {},
        }

    issues = external_hints.get("issues") or []
    if len(mesh.faces) == 0 or not issues:
        return [], {
            "seed_faces": 0,
            "seeded_issues": 0,
            "issue_categories": {},
        }

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_to_faces: Dict[int, set[int]] = defaultdict(set)
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[int(vertex_index)].add(int(face_index))

    seed_faces: set[int] = set()
    seeded_issues = 0
    issue_categories: Dict[str, int] = defaultdict(int)

    for issue in issues:
        if not isinstance(issue, dict):
            continue

        issue_faces: set[int] = set(int(face_index) for face_index in issue.get("face_ids") or [])

        for vertex_id in (issue.get("vertex_ids") or []):
            issue_faces.update(vertex_to_faces.get(int(vertex_id), ()))

        edge_vertex_ids = [int(vertex_id) for vertex_id in issue.get("edge_vertex_ids") or []]
        if len(edge_vertex_ids) >= 2:
            first_vertex, second_vertex = edge_vertex_ids[:2]
            shared_faces = vertex_to_faces.get(first_vertex, set()) & vertex_to_faces.get(second_vertex, set())
            if shared_faces:
                issue_faces.update(shared_faces)
            else:
                issue_faces.update(vertex_to_faces.get(first_vertex, ()))
                issue_faces.update(vertex_to_faces.get(second_vertex, ()))

        sample_points: List[np.ndarray] = []
        for key in ("triangle_points", "closed_path_points", "polyline_points"):
            raw_points = issue.get(key) or []
            if not raw_points:
                continue
            try:
                points = np.asarray(raw_points, dtype=float)
            except Exception:
                continue
            if points.ndim == 1 and points.shape[0] == 3:
                points = points.reshape(1, 3)
            if points.ndim != 2 or points.shape[1] != 3:
                continue
            sample_points.extend(points)
        if not sample_points and issue.get("point") is not None:
            try:
                point = np.asarray(issue["point"], dtype=float).reshape(3)
            except Exception:
                point = None
            if point is not None:
                sample_points.append(point)

        if not issue_faces:
            for point in sample_points:
                try:
                    nearest_face = find_nearest_face(mesh, point)
                except Exception:
                    continue
                issue_faces.add(int(nearest_face["face_id"]))

        issue_faces = {face_index for face_index in issue_faces if 0 <= face_index < len(faces)}
        if not issue_faces:
            continue

        seed_faces.update(issue_faces)
        seeded_issues += 1
        category = str(issue.get("category") or "external-hints")
        issue_categories[category] += 1

    return sorted(seed_faces), {
        "seed_faces": int(len(seed_faces)),
        "seeded_issues": int(seeded_issues),
        "issue_categories": dict(issue_categories),
    }


def rebuild_mesh_from_external_issues(
    mesh: trimesh.Trimesh,
    external_hints: Optional[dict],
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    seed_neighbor_rings: int = 1,
    status_callback: StatusCallback = None,
) -> Tuple[trimesh.Trimesh, dict]:
    seed_faces, seed_report = collect_seed_face_indices_from_external_issues(mesh, external_hints)
    if not seed_faces:
        return mesh.copy(), {
            "skipped": True,
            "reason": "no_seed_faces_from_external_hints",
            **seed_report,
        }

    expanded_faces = expand_face_selection(mesh, seed_faces, rings=seed_neighbor_rings)
    components = partition_face_components(mesh, expanded_faces)
    if not components:
        return mesh.copy(), {
            "skipped": True,
            "reason": "seed_faces_did_not_form_components",
            **seed_report,
            "expanded_faces": int(len(expanded_faces)),
        }

    emit_status(
        status_callback,
        f"Rebuilding {len(components)} hint-seeded damage region(s) before the full triangle rebuild",
    )

    rebuilt_components: List[trimesh.Trimesh] = []
    component_reports: List[dict] = []
    for component_index, component_faces in enumerate(components, start=1):
        emit_status(status_callback, f"Rebuilding hint-seeded region {component_index}/{len(components)}")
        component_mesh = extract_face_submesh(mesh, component_faces)
        try:
            rebuilt_component, rebuild_report = rebuild_mesh_without_triangle_overlaps(
                component_mesh,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=None,
            )
        except Exception as exc:
            rebuilt_component = component_mesh.copy()
            rebuilt_component.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(rebuilt_component)
            rebuild_report = {
                "skipped": True,
                "reason": str(exc),
            }

        if len(rebuilt_component.faces) == 0 and len(component_mesh.faces) > 0:
            rebuilt_component = component_mesh.copy()
            rebuilt_component.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(rebuilt_component)

        rebuilt_components.append(rebuilt_component)
        component_reports.append(
            {
                "component": int(component_index),
                "input_faces": int(len(component_mesh.faces)),
                "output_faces": int(len(rebuilt_component.faces)),
                "rebuild": rebuild_report,
            }
        )

    untouched_mesh = remove_mesh_faces(mesh, expanded_faces)
    meshes_to_combine = [untouched_mesh, *rebuilt_components]
    non_empty_meshes = [candidate for candidate in meshes_to_combine if len(candidate.faces) > 0]
    if not non_empty_meshes:
        combined = make_empty_mesh()
    elif len(non_empty_meshes) == 1:
        combined = non_empty_meshes[0]
    else:
        combined = combine_meshes(non_empty_meshes)

    combined = merge_nearby_vertices(combined, merge_eps=merge_eps)
    combined = remove_duplicate_faces(combined, decimals=dedup_decimals)
    combined = remove_degenerate_faces(combined, eps=area_eps)
    if len(combined.faces) > 0:
        trimesh.repair.fix_normals(combined)

    return combined, {
        "skipped": False,
        **seed_report,
        "expanded_faces": int(len(expanded_faces)),
        "seed_neighbor_rings": int(max(0, int(seed_neighbor_rings))),
        "components": component_reports,
        "components_rebuilt": int(len(component_reports)),
        "untouched_faces": int(len(untouched_mesh.faces)),
        "output_faces": int(len(combined.faces)),
    }


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
    entity_counts: Dict[str, int] = defaultdict(int)
    polyline_counts: Dict[str, int] = defaultdict(int)
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
        entity_counts[t] += 1
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
                    polyline_counts["polyface_mesh"] += 1
                    add_polyface(e)
                    continue
                if e.is_polygon_mesh:
                    polyline_counts["polygon_mesh"] += 1
                    continue
                polyline_counts["polyline_curve"] += 1
            except Exception:
                polyline_counts["unclassified_polyline"] += 1

    if not faces:
        if saw_polyline:
            raise SkippedInputError(
                "DXF contains POLYLINE entities but no supported triangulated mesh entities; skipping file.",
                code="unsupported_dxf_polyline",
                details={
                    "format": "dxf",
                    "entity_counts": dict(entity_counts),
                    "polyline_counts": dict(polyline_counts),
                    "supported_mesh_entities": ["3DFACE", "POLYFACE"],
                    "reason": (
                        "Export triangulated 3DFACE or POLYFACE, or convert the file to STL/OBJ/PLY/MSH before healing."
                    ),
                },
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


def build_skipped_input_report(
    mode: str,
    input_path: Path,
    output_path: Optional[Path],
    skip_error: SkippedInputError,
    extra: Optional[dict] = None,
) -> dict:
    report = {
        "mode": mode,
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "skipped": True,
        "skip_code": skip_error.code,
        "skip_reason": str(skip_error),
        "skip_details": dict(skip_error.details),
    }
    if extra:
        report.update(extra)
    return report


def build_deprecated_feature_report(
    mode: str,
    message: str,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> dict:
    report = {
        "mode": mode,
        "input": str(input_path) if input_path is not None else None,
        "output": str(output_path) if output_path is not None else None,
        "deprecated": True,
        "status": "deprecated",
        "message": message,
    }
    if extra:
        report.update(extra)
    return report


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
    surface = poly.extract_surface(algorithm="dataset_surface")
    surface.clear_data()
    surface = surface.triangulate().clean()
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
        emit_status(
            status_callback,
            "Skipping overlap-resolution union because at least one healed solid is open; downstream watertight repair will still run when enabled.",
        )
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
    finalize_start = time.perf_counter()
    try:
        step_start = time.perf_counter()
        emit_status(status_callback, f"Finalization: PyVista cleanup on {len(work.faces):,} faces")
        work = heal_with_pyvista(work, merge_eps=merge_eps)
        emit_status(
            status_callback,
            f"Finalization: PyVista cleanup finished in {time.perf_counter() - step_start:.1f}s",
        )
    except Exception as exc:
        emit_status(status_callback, f"PyVista cleanup skipped ({exc})")
    step_start = time.perf_counter()
    emit_status(status_callback, f"Finalization: merging {len(work.vertices):,} vertices")
    work = merge_nearby_vertices(work, merge_eps=merge_eps)
    emit_status(
        status_callback,
        f"Finalization: vertex merge finished in {time.perf_counter() - step_start:.1f}s",
    )
    step_start = time.perf_counter()
    emit_status(status_callback, f"Finalization: removing duplicate faces from {len(work.faces):,} faces")
    work = remove_duplicate_faces(work, decimals=dedup_decimals)
    emit_status(
        status_callback,
        f"Finalization: duplicate-face removal finished in {time.perf_counter() - step_start:.1f}s",
    )
    step_start = time.perf_counter()
    emit_status(status_callback, f"Finalization: removing degenerate faces from {len(work.faces):,} faces")
    work = remove_degenerate_faces(work, eps=area_eps)
    emit_status(
        status_callback,
        f"Finalization: degenerate-face removal finished in {time.perf_counter() - step_start:.1f}s",
    )

    if resolve_component_overlaps:
        components = split_disconnected_components(work)
        if len(components) > 1:
            emit_status(status_callback, f"Resolving overlap across {len(components)} healed solids")
            step_start = time.perf_counter()
            work = union_watertight_components(components, status_callback=status_callback)
            emit_status(
                status_callback,
                f"Finalization: overlap-resolution union finished in {time.perf_counter() - step_start:.1f}s",
            )
            step_start = time.perf_counter()
            emit_status(status_callback, f"Finalization: merging vertices after overlap resolution")
            work = merge_nearby_vertices(work, merge_eps=merge_eps)
            emit_status(
                status_callback,
                f"Finalization: post-union vertex merge finished in {time.perf_counter() - step_start:.1f}s",
            )
            step_start = time.perf_counter()
            emit_status(status_callback, f"Finalization: removing duplicate faces after overlap resolution")
            work = remove_duplicate_faces(work, decimals=dedup_decimals)
            emit_status(
                status_callback,
                f"Finalization: post-union duplicate-face removal finished in {time.perf_counter() - step_start:.1f}s",
            )
            step_start = time.perf_counter()
            emit_status(status_callback, f"Finalization: removing degenerate faces after overlap resolution")
            work = remove_degenerate_faces(work, eps=area_eps)
            emit_status(
                status_callback,
                f"Finalization: post-union degenerate-face removal finished in {time.perf_counter() - step_start:.1f}s",
            )

    step_start = time.perf_counter()
    emit_status(status_callback, f"Finalization: dropping unreferenced vertices")
    work.remove_unreferenced_vertices()
    emit_status(
        status_callback,
        f"Finalization: unreferenced-vertex cleanup finished in {time.perf_counter() - step_start:.1f}s",
    )
    step_start = time.perf_counter()
    emit_status(status_callback, f"Finalization: fixing normals on {len(work.faces):,} faces")
    trimesh.repair.fix_normals(work)
    emit_status(
        status_callback,
        f"Finalization complete in {time.perf_counter() - finalize_start:.1f}s",
    )
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


def classify_autoresearch_topology_bucket(input_report: MeshReport) -> str:
    needs_watertight_repair = (not input_report.watertight) or input_report.boundary_edges > 0
    nonmanifold_present = input_report.nonmanifold_edges > 0
    nonmanifold_heavy = input_report.nonmanifold_edges >= max(8, input_report.boundary_edges * 2)

    if nonmanifold_heavy:
        return "nonmanifold-heavy"
    if needs_watertight_repair and nonmanifold_present:
        return "mixed-topology"
    if needs_watertight_repair:
        return "watertight-repair"
    if nonmanifold_present:
        return "nonmanifold-light"
    if input_report.duplicate_faces > 0:
        return "rebuild-damage"
    return "mostly-clean"


def discover_autoresearch_ledger_paths(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path],
    limit: int = AUTORESEARCH_HISTORY_LEDGER_LIMIT,
) -> List[Path]:
    roots: List[Path] = []
    seen_roots: set[Path] = set()
    for candidate_root in (
        report_path.parent if report_path is not None else None,
        output_path.parent,
        input_path.parent,
    ):
        if candidate_root is None:
            continue
        resolved_root = candidate_root.resolve()
        if resolved_root in seen_roots or not resolved_root.exists():
            continue
        seen_roots.add(resolved_root)
        roots.append(resolved_root)

    ledger_paths: List[Path] = []
    seen_ledgers: set[Path] = set()
    for root in roots:
        try:
            candidates = sorted(root.glob("*_ledger.tsv"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for ledger_path in candidates:
            resolved_path = ledger_path.resolve()
            if resolved_path in seen_ledgers:
                continue
            seen_ledgers.add(resolved_path)
            ledger_paths.append(resolved_path)
            if len(ledger_paths) >= max(1, int(limit)):
                return ledger_paths
    return ledger_paths


def load_autoresearch_history(ledger_paths: Sequence[Path], topology_bucket: str) -> dict:
    def parse_boolish(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def parse_optional_float(value: object) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def make_stats() -> dict:
        return {
            "attempts": 0,
            "completed": 0,
            "accepted": 0,
            "selected": 0,
            "score_sum": 0.0,
            "time_sum": 0.0,
        }

    global_stats: Dict[str, dict] = defaultdict(make_stats)
    bucket_stats: Dict[str, dict] = defaultdict(make_stats)

    for ledger_path in ledger_paths:
        try:
            with ledger_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    candidate_name = str(row.get("candidate_name") or "").strip()
                    if not candidate_name:
                        continue
                    row_bucket = str(row.get("input_topology_bucket") or "").strip()
                    target_stats = [global_stats[candidate_name]]
                    if row_bucket and row_bucket == topology_bucket:
                        target_stats.append(bucket_stats[candidate_name])
                    accepted = parse_boolish(row.get("acceptance_passed", row.get("leapfrog_ready")))
                    selected = parse_boolish(row.get("selected"))
                    completed = str(row.get("status") or "").strip().lower() == "completed"
                    total_score = parse_optional_float(row.get("total_score"))
                    total_seconds = parse_optional_float(row.get("total_seconds"))
                    for stats in target_stats:
                        stats["attempts"] += 1
                        stats["accepted"] += int(accepted)
                        stats["selected"] += int(selected)
                        if completed:
                            stats["completed"] += 1
                            if total_score is not None and math.isfinite(total_score):
                                stats["score_sum"] += float(total_score)
                            if total_seconds is not None and math.isfinite(total_seconds):
                                stats["time_sum"] += float(total_seconds)
        except OSError:
            continue

    candidate_history: Dict[str, dict] = {}
    for candidate_name in sorted(set(global_stats) | set(bucket_stats)):
        preferred_stats = bucket_stats.get(candidate_name)
        scope = "bucket"
        if preferred_stats is None or int(preferred_stats["attempts"]) == 0:
            preferred_stats = global_stats.get(candidate_name)
            scope = "global"
        if preferred_stats is None or int(preferred_stats["attempts"]) == 0:
            continue
        completed = max(1, int(preferred_stats["completed"]))
        attempts = int(preferred_stats["attempts"])
        candidate_history[candidate_name] = {
            "scope": scope,
            "attempts": attempts,
            "completed": int(preferred_stats["completed"]),
            "acceptance_rate": float(preferred_stats["accepted"]) / float(max(1, attempts)),
            "selected_rate": float(preferred_stats["selected"]) / float(max(1, attempts)),
            "avg_total_score": float(preferred_stats["score_sum"]) / float(completed),
            "avg_total_seconds": float(preferred_stats["time_sum"]) / float(completed),
        }

    return {
        "topology_bucket": topology_bucket,
        "ledger_paths": [str(path) for path in ledger_paths],
        "candidate_stats": candidate_history,
    }


def build_autoresearch_candidates(
    input_report: MeshReport,
    allow_aggressive_modes: bool = False,
    fast_leapfrog: bool = False,
    history: Optional[dict] = None,
    external_hint_summary: Optional[dict] = None,
    intended_mesh_type: str = "auto",
) -> List[HealSearchCandidate]:
    candidates: List[HealSearchCandidate] = []
    seen: set[Tuple] = set()

    def add_candidate(candidate: HealSearchCandidate) -> None:
        key = (
            candidate.rebuild_triangles,
            candidate.nonmanifold_edge_repair,
            candidate.localized_intersection_repair,
            candidate.point_cloud_rebuild,
            candidate.distance_model,
            candidate.make_watertight,
            round(candidate.distance_offset_ratio, 8),
            round(candidate.distance_grid_spacing_ratio, 8),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    normalized_intended_mesh_type = normalize_intended_mesh_type(intended_mesh_type)
    needs_watertight_repair = (not input_report.watertight) or input_report.boundary_edges > 0
    has_nonmanifold_edges = input_report.nonmanifold_edges > 0
    has_topology_damage = needs_watertight_repair or has_nonmanifold_edges or input_report.duplicate_faces > 0
    hint_summary = apply_intended_mesh_type_to_hint_summary(
        summarize_external_issue_payload(external_hint_summary),
        normalized_intended_mesh_type,
    )
    hinted_overlap_damage = bool(hint_summary.get("force_rebuild_triangles") or hint_summary.get("force_localized_intersection_repair"))
    hinted_nonmanifold_damage = bool(hint_summary.get("force_nonmanifold_edge_repair"))
    hinted_boundary_damage = bool(hint_summary.get("force_make_watertight"))
    if normalized_intended_mesh_type == "surface":
        needs_watertight_repair = False
    needs_watertight_repair = needs_watertight_repair or hinted_boundary_damage
    has_nonmanifold_edges = has_nonmanifold_edges or hinted_nonmanifold_damage
    has_topology_damage = has_topology_damage or hinted_overlap_damage or hinted_nonmanifold_damage or hinted_boundary_damage
    proactive_leapfrog_safe_candidate = HealSearchCandidate(
        name="proactive-safe",
        rebuild_triangles=True,
        nonmanifold_edge_repair=True,
        localized_intersection_repair=True,
        make_watertight=True,
    )

    if fast_leapfrog:
        add_candidate(HealSearchCandidate(name="baseline"))
        if needs_watertight_repair:
            add_candidate(HealSearchCandidate(name="watertight", make_watertight=True))
        if has_topology_damage:
            add_candidate(
                HealSearchCandidate(
                    name="rebuild-watertight",
                    rebuild_triangles=True,
                    make_watertight=True,
                )
            )
        if has_nonmanifold_edges:
            add_candidate(
                HealSearchCandidate(
                    name="nonmanifold-watertight",
                    nonmanifold_edge_repair=True,
                    make_watertight=True,
                )
            )
        add_candidate(
            HealSearchCandidate(
                name="localized-watertight",
                localized_intersection_repair=True,
                make_watertight=True,
            )
        )
        add_candidate(proactive_leapfrog_safe_candidate)
        if has_nonmanifold_edges or needs_watertight_repair:
            add_candidate(
                HealSearchCandidate(
                    name="full-safe",
                    rebuild_triangles=True,
                    nonmanifold_edge_repair=has_nonmanifold_edges,
                    localized_intersection_repair=True,
                    make_watertight=True,
                )
            )
        elif input_report.duplicate_faces > 0:
            add_candidate(HealSearchCandidate(name="rebuild", rebuild_triangles=True))
        return prioritize_autoresearch_candidates(
            candidates,
            input_report,
            fast_leapfrog=True,
            history=history,
            external_hint_summary=hint_summary,
        )

    add_candidate(HealSearchCandidate(name="baseline"))
    if needs_watertight_repair:
        add_candidate(HealSearchCandidate(name="watertight", make_watertight=True))
    if has_topology_damage:
        add_candidate(HealSearchCandidate(name="rebuild", rebuild_triangles=True))
        add_candidate(
            HealSearchCandidate(
                name="rebuild-watertight",
                rebuild_triangles=True,
                make_watertight=True,
            )
        )

    add_candidate(HealSearchCandidate(name="localized", localized_intersection_repair=True))
    add_candidate(
        HealSearchCandidate(
            name="localized-watertight",
            localized_intersection_repair=True,
            make_watertight=True,
        )
    )
    add_candidate(proactive_leapfrog_safe_candidate)

    if has_nonmanifold_edges:
        add_candidate(HealSearchCandidate(name="nonmanifold", nonmanifold_edge_repair=True))
        add_candidate(
            HealSearchCandidate(
                name="nonmanifold-watertight",
                nonmanifold_edge_repair=True,
                make_watertight=True,
            )
        )
        add_candidate(
            HealSearchCandidate(
                name="rebuild-nonmanifold-watertight",
                rebuild_triangles=True,
                nonmanifold_edge_repair=True,
                make_watertight=True,
            )
        )

    add_candidate(
        HealSearchCandidate(
            name="rebuild-localized-watertight",
            rebuild_triangles=True,
            localized_intersection_repair=True,
            make_watertight=True,
        )
    )

    if has_nonmanifold_edges or needs_watertight_repair:
        add_candidate(
            HealSearchCandidate(
                name="full-safe",
                rebuild_triangles=True,
                nonmanifold_edge_repair=has_nonmanifold_edges,
                localized_intersection_repair=True,
                make_watertight=True,
            )
        )

    if allow_aggressive_modes:
        add_candidate(
            HealSearchCandidate(
                name="aggressive-point-cloud",
                point_cloud_rebuild="triangle-centers-poisson",
                make_watertight=True,
                aggressive=True,
            )
        )
        add_candidate(
            HealSearchCandidate(
                name="aggressive-distance-hull",
                distance_model="distance-hull",
                distance_offset_ratio=0.0025,
                distance_grid_spacing_ratio=0.00125,
                aggressive=True,
            )
        )
        add_candidate(
            HealSearchCandidate(
                name="aggressive-rebuild-distance-hull",
                rebuild_triangles=True,
                distance_model="distance-hull",
                distance_offset_ratio=0.0025,
                distance_grid_spacing_ratio=0.00125,
                aggressive=True,
            )
        )

    return prioritize_autoresearch_candidates(
        candidates,
        input_report,
        fast_leapfrog=False,
        history=history,
        external_hint_summary=hint_summary,
    )


def prioritize_autoresearch_candidates(
    candidates: Sequence[HealSearchCandidate],
    input_report: MeshReport,
    fast_leapfrog: bool = False,
    history: Optional[dict] = None,
    external_hint_summary: Optional[dict] = None,
) -> List[HealSearchCandidate]:
    ordered = list(candidates)
    if not fast_leapfrog:
        return ordered

    topology_bucket = classify_autoresearch_topology_bucket(input_report)
    hint_summary = summarize_external_issue_payload(external_hint_summary)

    if topology_bucket == "nonmanifold-heavy":
        preferred_names = [
            "nonmanifold-watertight",
            "proactive-safe",
            "full-safe",
            "watertight",
            "baseline",
            "rebuild-watertight",
            "localized-watertight",
            "rebuild",
        ]
    elif topology_bucket in {"mixed-topology", "watertight-repair"}:
        preferred_names = [
            "watertight",
            "proactive-safe",
            "baseline",
            "rebuild-watertight",
            "localized-watertight",
            "nonmanifold-watertight",
            "full-safe",
            "rebuild",
        ]
    else:
        preferred_names = [
            "baseline",
            "proactive-safe",
            "watertight",
            "localized-watertight",
            "rebuild-watertight",
            "nonmanifold-watertight",
            "full-safe",
            "rebuild",
        ]

    if hint_summary.get("prefer_proactive_safe") and "proactive-safe" in preferred_names:
        preferred_names = ["proactive-safe", *[name for name in preferred_names if name != "proactive-safe"]]

    preferred_order = {name: index for index, name in enumerate(preferred_names)}
    history_stats = (history or {}).get("candidate_stats", {})

    def history_sort_key(candidate: HealSearchCandidate) -> Tuple[float, ...]:
        candidate_history = history_stats.get(candidate.name)
        if not candidate_history:
            return (1.0, 0.0, 0.0, AUTORESEARCH_ERROR_SCORE, 600.0, 0.0)
        return (
            0.0,
            -float(candidate_history.get("acceptance_rate", 0.0)),
            -float(candidate_history.get("selected_rate", 0.0)),
            float(candidate_history.get("avg_total_score", AUTORESEARCH_ERROR_SCORE)),
            float(candidate_history.get("avg_total_seconds", 600.0)),
            -float(candidate_history.get("attempts", 0.0)),
        )

    return sorted(
        ordered,
        key=lambda candidate: (
            history_sort_key(candidate),
            preferred_order.get(candidate.name, len(preferred_order)),
            candidate.enabled_step_count(),
            candidate.name,
        ),
    )




def find_nearest_face(mesh: trimesh.Trimesh, point: Sequence[float] | np.ndarray) -> dict:
    if len(mesh.faces) == 0:
        raise ValueError("The current mesh has no faces to pick from.")

    centers = np.asarray(mesh.triangles_center, dtype=float)
    target = np.asarray(point, dtype=float).reshape(3)
    deltas = centers - target
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    best_index = int(np.argmin(distances_sq))
    return {
        "face_id": best_index,
        "distance": float(math.sqrt(max(0.0, distances_sq[best_index]))),
        "point": centers[best_index].astype(float).tolist(),
    }


def find_nearest_vertex(mesh: trimesh.Trimesh, point: Sequence[float] | np.ndarray) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        raise ValueError("The current mesh has no vertices to pick from.")

    target = np.asarray(point, dtype=float).reshape(3)
    deltas = vertices - target
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    best_index = int(np.argmin(distances_sq))
    return {
        "vertex_id": best_index,
        "distance": float(math.sqrt(max(0.0, distances_sq[best_index]))),
        "point": vertices[best_index].astype(float).tolist(),
    }


def normalize_external_issue_category(category: str | None) -> str:
    normalized = str(category or "external-hints").strip().lower().replace("_", "-").replace(" ", "-")
    if normalized in {
        "nonmanifold",
        "non-manifold",
        "nonmanifold-edge",
        "non-manifold-edge",
        "nonmanifold-edges",
        "non-manifold-edges",
    }:
        return "nonmanifold-edges"
    if normalized in {
        "boundary-loop",
        "boundary-loops",
        "open-boundary",
        "open-boundaries",
        "hole",
        "holes",
        "contour",
        "contours",
    }:
        return "boundary-loops"
    if normalized in {
        "degenerate",
        "degenerate-face",
        "degenerate-faces",
    }:
        return "degenerate-faces"
    if normalized in {
        "duplicate",
        "duplicate-face",
        "duplicate-faces",
    }:
        return "duplicate-faces"
    if normalized in {
        "overlap",
        "overlaps",
        "overlapping-triangle",
        "overlapping-triangles",
        "self-intersection",
        "self-intersections",
        "intersection",
        "intersections",
    }:
        return "external-overlap-hints"
    return normalized or "external-hints"


def _coerce_issue_points(raw_points: object) -> Optional[np.ndarray]:
    if raw_points is None:
        return None
    try:
        points = np.asarray(raw_points, dtype=float)
    except Exception:
        return None

    if points.size == 0:
        return None
    if points.ndim == 1:
        if points.shape[0] != 3:
            return None
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    return points.astype(float)


def _normalize_issue_index_list(indices: object, upper_bound: int) -> List[int]:
    if indices is None:
        return []
    try:
        values = [int(index) for index in indices]
    except Exception:
        return []
    return sorted({index for index in values if 0 <= index < upper_bound})


def _map_points_to_vertex_ids(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    merge_eps: float = 1e-8,
) -> List[int]:
    vertex_ids: List[int] = []
    last_vertex_id = None
    for point in points:
        nearest = find_nearest_vertex(mesh, point)
        vertex_id = int(nearest["vertex_id"])
        if last_vertex_id == vertex_id and nearest["distance"] <= max(merge_eps * 10.0, 1e-9):
            continue
        vertex_ids.append(vertex_id)
        last_vertex_id = vertex_id

    if len(vertex_ids) >= 2 and vertex_ids[0] == vertex_ids[-1]:
        vertex_ids.pop()
    return vertex_ids


def normalize_external_issue_payload(
    payload: object,
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    source: Optional[str] = None,
) -> dict:
    raw_issues: List[dict] = []
    skipped_issues: List[dict] = []

    def append_group(entries: object, default_category: str) -> None:
        if not isinstance(entries, list):
            return
        for item in entries:
            if isinstance(item, dict):
                normalized_item = dict(item)
            else:
                normalized_item = {"point": item}
            normalized_item.setdefault("category", default_category)
            raw_issues.append(normalized_item)

    if isinstance(payload, list):
        raw_issues.extend(item for item in payload if isinstance(item, dict))
    elif isinstance(payload, dict):
        append_group(payload.get("issues"), "external-hints")
        append_group(payload.get("nonmanifold_edges"), "nonmanifold-edges")
        append_group(payload.get("overlapping_triangles"), "external-overlap-hints")
        append_group(payload.get("overlap_hints"), "external-overlap-hints")
        append_group(payload.get("self_intersections"), "external-overlap-hints")
        append_group(payload.get("boundary_loops"), "boundary-loops")
        append_group(payload.get("contours"), "boundary-loops")
        if not raw_issues and any(
            key in payload
            for key in (
                "category",
                "type",
                "kind",
                "face_ids",
                "vertex_ids",
                "point",
                "points",
                "polyline_points",
                "closed_path_points",
                "edge_vertex_ids",
                "triangle_points",
            )
        ):
            raw_issues.append(dict(payload))

    vertices = np.asarray(mesh.vertices, dtype=float)
    face_centers = np.asarray(mesh.triangles_center, dtype=float) if len(mesh.faces) > 0 else np.zeros((0, 3), dtype=float)
    normalized_issues: List[dict] = []
    category_counts: Dict[str, int] = defaultdict(int)

    for index, raw_issue in enumerate(raw_issues, start=1):
        if not isinstance(raw_issue, dict):
            skipped_issues.append({"index": index, "reason": "issue is not an object"})
            continue

        category = normalize_external_issue_category(
            raw_issue.get("category") or raw_issue.get("type") or raw_issue.get("kind")
        )
        label = str(raw_issue.get("label") or raw_issue.get("name") or f"Imported issue {index}")
        description = str(raw_issue.get("description") or raw_issue.get("message") or "Imported external repair hint.")

        point_array = _coerce_issue_points(raw_issue.get("point") or raw_issue.get("location") or raw_issue.get("coordinates"))
        polyline_points = _coerce_issue_points(raw_issue.get("polyline_points") or raw_issue.get("points"))
        closed_path_points = _coerce_issue_points(raw_issue.get("closed_path_points") or raw_issue.get("loop_points") or raw_issue.get("contour_points"))
        triangle_points = _coerce_issue_points(raw_issue.get("triangle_points") or raw_issue.get("triangle"))

        face_ids = _normalize_issue_index_list(raw_issue.get("face_ids"), len(mesh.faces))
        if not face_ids and triangle_points is not None and len(triangle_points) >= 3 and len(mesh.faces) > 0:
            nearest_face = find_nearest_face(mesh, np.mean(triangle_points[:3], axis=0))
            face_ids = [int(nearest_face["face_id"])]
            if point_array is None:
                point_array = np.asarray([nearest_face["point"]], dtype=float)
        if not face_ids and point_array is not None and len(mesh.faces) > 0 and category in {
            "duplicate-faces",
            "degenerate-faces",
            "external-overlap-hints",
        }:
            nearest_face = find_nearest_face(mesh, point_array[0])
            face_ids = [int(nearest_face["face_id"])]

        vertex_ids = _normalize_issue_index_list(raw_issue.get("vertex_ids"), len(mesh.vertices))
        if not vertex_ids and closed_path_points is not None:
            vertex_ids = _map_points_to_vertex_ids(mesh, closed_path_points, merge_eps=merge_eps)
        elif not vertex_ids and polyline_points is not None and category == "boundary-loops":
            vertex_ids = _map_points_to_vertex_ids(mesh, polyline_points, merge_eps=merge_eps)

        edge_vertex_ids = _normalize_issue_index_list(raw_issue.get("edge_vertex_ids"), len(mesh.vertices))
        if len(edge_vertex_ids) > 2:
            edge_vertex_ids = edge_vertex_ids[:2]
        if len(edge_vertex_ids) < 2:
            edge_vertex_ids = []
        if not edge_vertex_ids and point_array is not None and category == "nonmanifold-edges":
            edge_pick = find_nearest_edge(mesh, point_array[0])
            edge_vertex_ids = [int(edge_pick["edge_vertex_ids"][0]), int(edge_pick["edge_vertex_ids"][1])]
            if polyline_points is None:
                polyline_points = np.asarray(edge_pick["edge_points"], dtype=float)
            if point_array is None:
                point_array = np.asarray([edge_pick["point"]], dtype=float)

        if polyline_points is None and len(edge_vertex_ids) == 2:
            polyline_points = np.asarray(vertices[edge_vertex_ids], dtype=float)

        if point_array is None:
            if polyline_points is not None and len(polyline_points) > 0:
                point_array = np.asarray([np.mean(polyline_points, axis=0)], dtype=float)
            elif face_ids:
                point_array = np.asarray([np.mean(face_centers[np.asarray(face_ids, dtype=np.int64)], axis=0)], dtype=float)
            elif vertex_ids:
                point_array = np.asarray([np.mean(vertices[np.asarray(vertex_ids, dtype=np.int64)], axis=0)], dtype=float)

        issue = {
            "id": str(raw_issue.get("id") or f"external-issue-{index}"),
            "category": category,
            "label": label,
            "description": description,
            "source": "external",
        }
        if face_ids:
            issue["face_ids"] = face_ids
        if vertex_ids:
            issue["vertex_ids"] = vertex_ids
        if len(edge_vertex_ids) == 2:
            issue["edge_vertex_ids"] = edge_vertex_ids
        if polyline_points is not None and len(polyline_points) > 0:
            issue["polyline_points"] = polyline_points.astype(float).tolist()
        if closed_path_points is not None and len(closed_path_points) > 0:
            issue["closed_path_points"] = closed_path_points.astype(float).tolist()
        if triangle_points is not None and len(triangle_points) > 0:
            issue["triangle_points"] = triangle_points.astype(float).tolist()
        if point_array is not None and len(point_array) > 0:
            issue["point"] = point_array[0].astype(float).tolist()

        normalized_issues.append(issue)
        category_counts[category] += 1

    return {
        "source": str(source) if source is not None else None,
        "issues": normalized_issues,
        "issue_count": int(len(normalized_issues)),
        "categories": {key: int(value) for key, value in category_counts.items()},
        "raw_issue_count": int(len(raw_issues)),
        "skipped_issue_count": int(len(skipped_issues)),
        "skipped_issues": skipped_issues,
    }


def _infer_external_hint_category(
    layer_name: str,
    entity_type: str,
    is_closed: bool = False,
) -> str:
    normalized_layer = str(layer_name or "").strip().lower().replace("_", "-").replace(" ", "-")
    if "non" in normalized_layer and "manifold" in normalized_layer:
        return "nonmanifold-edges"
    if any(token in normalized_layer for token in ("overlap", "intersect", "collision", "stack")):
        return "external-overlap-hints"
    if any(token in normalized_layer for token in ("boundary", "contour", "loop", "hole", "open-edge")):
        return "boundary-loops"

    if entity_type in {"3DFACE", "SOLID", "TRACE"}:
        return "external-overlap-hints"
    if entity_type in {"LWPOLYLINE", "POLYLINE"}:
        return "boundary-loops" if is_closed else "nonmanifold-edges"
    if entity_type == "LINE":
        return "nonmanifold-edges"
    return "external-hints"


def _coerce_dxf_point_3d(point: Sequence[float] | np.ndarray, fallback_z: float = 0.0) -> List[float]:
    values = list(point)
    if len(values) >= 3:
        return [float(values[0]), float(values[1]), float(values[2])]
    if len(values) >= 2:
        return [float(values[0]), float(values[1]), float(fallback_z)]
    raise ValueError("DXF point does not contain enough coordinates.")


def load_external_issue_source(path: Path) -> object:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if suffix != ".dxf":
        raise ValueError("External repair hints currently support JSON or DXF files.")

    ezdxf = _try_import_ezdxf()
    if ezdxf is None:
        raise RuntimeError("DXF hint import requires ezdxf. Install requirements first.")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    issues: List[dict] = []
    entity_counts: Dict[str, int] = defaultdict(int)

    for entity_index, entity in enumerate(msp, start=1):
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        layer_name = str(getattr(entity.dxf, "layer", "") or "")

        try:
            if entity_type == "LINE":
                start = _coerce_dxf_point_3d(entity.dxf.start)
                end = _coerce_dxf_point_3d(entity.dxf.end)
                issues.append(
                    {
                        "id": f"dxf-line-{entity_index}",
                        "label": f"DXF line {entity_index}",
                        "category": _infer_external_hint_category(layer_name, entity_type, is_closed=False),
                        "polyline_points": [start, end],
                        "description": f"Imported from DXF LINE on layer '{layer_name or '0'}'.",
                    }
                )
            elif entity_type == "LWPOLYLINE":
                elevation = float(getattr(entity.dxf, "elevation", 0.0) or 0.0)
                points = [_coerce_dxf_point_3d(point[:2], fallback_z=elevation) for point in entity.get_points("xy")]
                is_closed = bool(entity.closed)
                if len(points) >= 2:
                    issue = {
                        "id": f"dxf-lwpolyline-{entity_index}",
                        "label": f"DXF polyline {entity_index}",
                        "category": _infer_external_hint_category(layer_name, entity_type, is_closed=is_closed),
                        "description": f"Imported from DXF LWPOLYLINE on layer '{layer_name or '0'}'.",
                    }
                    if is_closed and len(points) >= 3:
                        issue["closed_path_points"] = points
                    else:
                        issue["polyline_points"] = points
                    issues.append(issue)
            elif entity_type == "POLYLINE":
                if getattr(entity, "is_poly_face_mesh", False) or getattr(entity, "is_polygon_mesh", False):
                    continue
                points = [_coerce_dxf_point_3d(vertex.dxf.location) for vertex in entity.vertices]
                is_closed = bool(getattr(entity, "is_closed", False))
                if len(points) >= 2:
                    issue = {
                        "id": f"dxf-polyline-{entity_index}",
                        "label": f"DXF polyline {entity_index}",
                        "category": _infer_external_hint_category(layer_name, entity_type, is_closed=is_closed),
                        "description": f"Imported from DXF POLYLINE on layer '{layer_name or '0'}'.",
                    }
                    if is_closed and len(points) >= 3:
                        issue["closed_path_points"] = points
                    else:
                        issue["polyline_points"] = points
                    issues.append(issue)
            elif entity_type == "3DFACE":
                points = [
                    _coerce_dxf_point_3d(entity.dxf.vtx0),
                    _coerce_dxf_point_3d(entity.dxf.vtx1),
                    _coerce_dxf_point_3d(entity.dxf.vtx2),
                ]
                issues.append(
                    {
                        "id": f"dxf-3dface-{entity_index}",
                        "label": f"DXF 3DFACE {entity_index}",
                        "category": _infer_external_hint_category(layer_name, entity_type, is_closed=True),
                        "triangle_points": points,
                        "description": f"Imported from DXF 3DFACE on layer '{layer_name or '0'}'.",
                    }
                )
            elif entity_type in {"SOLID", "TRACE"}:
                points = [
                    _coerce_dxf_point_3d(entity.dxf.vtx0),
                    _coerce_dxf_point_3d(entity.dxf.vtx1),
                    _coerce_dxf_point_3d(entity.dxf.vtx2),
                    _coerce_dxf_point_3d(entity.dxf.vtx3),
                ]
                deduped_points: List[List[float]] = []
                for point in points:
                    if not deduped_points or any(abs(point[i] - deduped_points[-1][i]) > 1e-12 for i in range(3)):
                        deduped_points.append(point)
                issue = {
                    "id": f"dxf-{entity_type.lower()}-{entity_index}",
                    "label": f"DXF {entity_type} {entity_index}",
                    "category": _infer_external_hint_category(layer_name, entity_type, is_closed=True),
                    "description": f"Imported from DXF {entity_type} on layer '{layer_name or '0'}'.",
                }
                if len(deduped_points) == 3:
                    issue["triangle_points"] = deduped_points
                elif len(deduped_points) >= 4:
                    issue["closed_path_points"] = deduped_points
                else:
                    continue
                issues.append(issue)
        except Exception:
            continue

    return {
        "issues": issues,
        "source_format": "dxf",
        "entity_counts": dict(entity_counts),
    }


def load_and_normalize_external_issues(
    path: Path | Sequence[Path],
    mesh: trimesh.Trimesh,
    merge_eps: float = 1e-8,
) -> dict:
    hint_paths = [path] if isinstance(path, Path) else [Path(item) for item in path]
    if not hint_paths:
        return normalize_external_issue_payload([], mesh, merge_eps=merge_eps, source=None)

    if len(hint_paths) == 1:
        hint_path = hint_paths[0]
        payload = load_external_issue_source(hint_path)
        normalized = normalize_external_issue_payload(
            payload,
            mesh,
            merge_eps=merge_eps,
            source=str(hint_path),
        )
        normalized["path"] = str(hint_path)
        normalized["paths"] = [str(hint_path)]
        normalized["source_format"] = str(hint_path.suffix.lower().lstrip("."))
        normalized["source_formats"] = [normalized["source_format"]]
        return normalized

    combined_issues: List[dict] = []
    combined_categories: Dict[str, int] = defaultdict(int)
    combined_skipped_issues: List[dict] = []
    raw_issue_count = 0

    for path_index, hint_path in enumerate(hint_paths, start=1):
        payload = load_external_issue_source(hint_path)
        normalized = normalize_external_issue_payload(
            payload,
            mesh,
            merge_eps=merge_eps,
            source=str(hint_path),
        )
        path_text = str(hint_path)
        for issue in normalized.get("issues", []):
            combined_issue = dict(issue)
            combined_issue["id"] = f"hint-{path_index}-{combined_issue.get('id', path_index)}"
            combined_issue["source_path"] = path_text
            combined_issues.append(combined_issue)
        for category, count in (normalized.get("categories") or {}).items():
            combined_categories[str(category)] += int(count)
        for skipped in normalized.get("skipped_issues") or []:
            combined_skipped_issue = dict(skipped)
            combined_skipped_issue["source_path"] = path_text
            combined_skipped_issues.append(combined_skipped_issue)
        raw_issue_count += int(normalized.get("raw_issue_count", 0))

    source_formats = [str(hint_path.suffix.lower().lstrip(".")) for hint_path in hint_paths]
    return {
        "source": "multiple",
        "path": None,
        "paths": [str(hint_path) for hint_path in hint_paths],
        "source_format": "multiple",
        "source_formats": source_formats,
        "issues": combined_issues,
        "issue_count": int(len(combined_issues)),
        "categories": {key: int(value) for key, value in combined_categories.items()},
        "raw_issue_count": int(raw_issue_count),
        "skipped_issue_count": int(len(combined_skipped_issues)),
        "skipped_issues": combined_skipped_issues,
    }


def summarize_external_issue_payload(payload: Optional[dict]) -> dict:
    if not payload:
        return {
            "provided": False,
            "issue_count": 0,
            "categories": {},
            "has_overlap_hints": False,
            "has_nonmanifold_hints": False,
            "has_boundary_hints": False,
            "force_rebuild_triangles": False,
            "force_nonmanifold_edge_repair": False,
            "force_localized_intersection_repair": False,
            "force_make_watertight": False,
            "prefer_proactive_safe": False,
        }

    categories = {str(key): int(value) for key, value in (payload.get("categories") or {}).items()}
    issue_count = int(payload.get("issue_count", 0))
    overlap_count = int(categories.get("external-overlap-hints", 0)) + int(categories.get("duplicate-faces", 0))
    nonmanifold_count = int(categories.get("nonmanifold-edges", 0))
    boundary_count = int(categories.get("boundary-loops", 0))

    force_rebuild_triangles = (overlap_count + nonmanifold_count + boundary_count) > 0
    force_nonmanifold_edge_repair = nonmanifold_count > 0
    force_localized_intersection_repair = overlap_count > 0
    force_make_watertight = boundary_count > 0

    return {
        "provided": True,
        "path": payload.get("path"),
        "source_format": payload.get("source_format"),
        "issue_count": issue_count,
        "categories": categories,
        "has_overlap_hints": bool(overlap_count > 0),
        "has_nonmanifold_hints": bool(nonmanifold_count > 0),
        "has_boundary_hints": bool(boundary_count > 0),
        "force_rebuild_triangles": bool(force_rebuild_triangles),
        "force_nonmanifold_edge_repair": bool(force_nonmanifold_edge_repair),
        "force_localized_intersection_repair": bool(force_localized_intersection_repair),
        "force_make_watertight": bool(force_make_watertight),
        "prefer_proactive_safe": bool(
            force_rebuild_triangles or force_nonmanifold_edge_repair or force_localized_intersection_repair or force_make_watertight
        ),
    }
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


def derive_autoresearch_ledger_path(report_path: Optional[Path]) -> Optional[Path]:
    if report_path is None:
        return None
    return report_path.with_name(f"{report_path.stem}_ledger.tsv")


def write_mesh_snapshot(mesh: trimesh.Trimesh, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("wb") as handle:
        np.savez(
            handle,
            vertices=np.asarray(mesh.vertices, dtype=float),
            faces=np.asarray(mesh.faces, dtype=np.int64),
        )


def load_mesh_snapshot(snapshot_path: Path) -> trimesh.Trimesh:
    with np.load(snapshot_path) as snapshot:
        vertices = np.asarray(snapshot["vertices"], dtype=float)
        faces = np.asarray(snapshot["faces"], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _leapfrog_self_intersection_worker(snapshot_path: Path, merge_eps: float, result_queue) -> None:
    try:
        mesh = load_mesh_snapshot(snapshot_path)
        result_queue.put(
            {
                "ok": True,
                "result": _validate_leapfrog_self_intersections_direct(mesh, merge_eps=merge_eps),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def run_leapfrog_self_intersection_validation_with_timeout(
    mesh: trimesh.Trimesh,
    merge_eps: float,
    timeout_seconds: float,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="mesh_heal_self_intersection_") as temp_dir_name:
        snapshot_path = Path(temp_dir_name) / "mesh_snapshot.npz"
        write_mesh_snapshot(mesh, snapshot_path)

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_leapfrog_self_intersection_worker,
            args=(snapshot_path, float(merge_eps), result_queue),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(10.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            return {
                "checked": False,
                "skipped": True,
                "skip_reason": f"self_intersection_validation_timeout:{timeout_seconds:.1f}s",
                "has_self_intersections": None,
                "intersecting_pairs": None,
                "intersecting_faces": None,
                "method": "subprocess-timeout",
            }

        try:
            result = result_queue.get(timeout=1.0)
        except Empty:
            result = None

        if result is None:
            return {
                "checked": False,
                "skipped": True,
                "skip_reason": f"self_intersection_validation_no_result_exitcode:{process.exitcode}",
                "has_self_intersections": None,
                "intersecting_pairs": None,
                "intersecting_faces": None,
                "method": "subprocess",
            }
        if not result.get("ok"):
            return {
                "checked": False,
                "skipped": True,
                "skip_reason": f"self_intersection_validation_failed:{result.get('error')}",
                "has_self_intersections": None,
                "intersecting_pairs": None,
                "intersecting_faces": None,
                "method": "subprocess",
            }
        return dict(result["result"])


def candidate_needs_subprocess_timeout(candidate: HealSearchCandidate) -> bool:
    return bool(
        candidate.aggressive
        or candidate.localized_intersection_repair
        or candidate.nonmanifold_edge_repair
        or candidate.point_cloud_rebuild != "none"
        or candidate.distance_model != "none"
        or candidate.enabled_step_count() >= 2
    )


def _autoresearch_candidate_worker(
    input_path: Path,
    output_snapshot_path: Path,
    execution_kwargs: dict,
    result_queue,
) -> None:
    try:
        mesh_in = load_mesh(input_path)
        execution = execute_heal_strategy_on_mesh(
            mesh_in,
            merge_eps=float(execution_kwargs["merge_eps"]),
            area_eps=float(execution_kwargs["area_eps"]),
            dedup_decimals=int(execution_kwargs["dedup_decimals"]),
            external_hints=execution_kwargs.get("external_hints"),
            rebuild_triangles=bool(execution_kwargs["rebuild_triangles"]),
            nonmanifold_edge_repair=bool(execution_kwargs["nonmanifold_edge_repair"]),
            nonmanifold_edge_radius=float(execution_kwargs["nonmanifold_edge_radius"]),
            localized_intersection_repair=bool(execution_kwargs["localized_intersection_repair"]),
            point_cloud_rebuild=str(execution_kwargs["point_cloud_rebuild"]),
            distance_model=str(execution_kwargs["distance_model"]),
            distance_offset=float(execution_kwargs["distance_offset"]),
            distance_grid_spacing=float(execution_kwargs["distance_grid_spacing"]),
            make_watertight=bool(execution_kwargs["make_watertight"]),
            status_callback=None,
            progress_callback=None,
        )
        write_mesh_snapshot(execution["mesh"], output_snapshot_path)
        result_queue.put({
            "ok": True,
            "report": execution["report"],
        })
    except Exception as exc:
        result_queue.put({
            "ok": False,
            "error": str(exc),
        })


def run_heal_candidate_with_timeout(
    input_path: Path,
    output_snapshot_path: Path,
    execution_kwargs: dict,
    timeout_seconds: float,
) -> dict:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_autoresearch_candidate_worker,
        args=(input_path, output_snapshot_path, execution_kwargs, result_queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(10.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        raise TimeoutError(f"candidate exceeded hard timeout of {timeout_seconds:.1f}s")

    try:
        result = result_queue.get(timeout=1.0)
    except Empty:
        result = None

    if result is None:
        raise RuntimeError(f"candidate subprocess exited with code {process.exitcode} before returning a result")
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "candidate subprocess failed"))
    return {
        "mesh": load_mesh_snapshot(output_snapshot_path),
        "report": result["report"],
        "execution_mode": "subprocess",
    }


def write_autoresearch_ledger(report: dict, report_path: Optional[Path]) -> Optional[Path]:
    ledger_path = derive_autoresearch_ledger_path(report_path)
    if ledger_path is None:
        return None

    fieldnames = [
        "input",
        "output",
        "input_topology_bucket",
        "fast_leapfrog",
        "candidate_timeout_seconds",
        "self_intersection_timeout_seconds",
        "candidate_rank",
        "candidate_name",
        "selected",
        "status",
        "execution_mode",
        "timed_out",
        "error",
        "enabled_step_count",
        "rebuild_triangles",
        "nonmanifold_edge_repair",
        "localized_intersection_repair",
        "point_cloud_rebuild",
        "distance_model",
        "make_watertight",
        "heal_seconds",
        "leapfrog_validation_seconds",
        "fidelity_seconds",
        "total_seconds",
        "leapfrog_ready",
        "acceptance_passed",
        "acceptance_failed_checks",
        "roundtrip_loadable",
        "self_intersection_checked",
        "self_intersection_pairs",
        "self_intersection_faces",
        "watertight",
        "boundary_edges",
        "nonmanifold_edges",
        "degenerate_faces",
        "duplicate_faces",
        "mean_distance_normalized",
        "p95_distance_normalized",
        "area_ratio_delta",
        "volume_ratio_delta",
        "component_count_delta",
        "total_score",
    ]

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for candidate_summary in report.get("candidates", []):
            strategy = candidate_summary.get("strategy", {})
            after = candidate_summary.get("after", {})
            timing = candidate_summary.get("timing", {})
            fidelity = candidate_summary.get("fidelity", {})
            score = candidate_summary.get("score", {})
            acceptance = candidate_summary.get("acceptance", {})
            leapfrog_validation = candidate_summary.get("leapfrog_validation", {})
            self_intersections = leapfrog_validation.get("self_intersections", {})
            writer.writerow(
                {
                    "input": report.get("input"),
                    "output": report.get("output"),
                    "input_topology_bucket": report.get("search", {}).get("topology_bucket"),
                    "fast_leapfrog": report.get("search", {}).get("fast_leapfrog"),
                    "candidate_timeout_seconds": report.get("search", {}).get("candidate_timeout_seconds"),
                    "self_intersection_timeout_seconds": report.get("search", {}).get("self_intersection_timeout_seconds"),
                    "candidate_rank": candidate_summary.get("rank"),
                    "candidate_name": candidate_summary.get("name"),
                    "selected": candidate_summary.get("selected"),
                    "status": candidate_summary.get("status"),
                    "execution_mode": candidate_summary.get("execution_mode"),
                    "timed_out": candidate_summary.get("timed_out", False),
                    "error": candidate_summary.get("error"),
                    "enabled_step_count": strategy.get("enabled_step_count"),
                    "rebuild_triangles": strategy.get("rebuild_triangles"),
                    "nonmanifold_edge_repair": strategy.get("nonmanifold_edge_repair"),
                    "localized_intersection_repair": strategy.get("localized_intersection_repair"),
                    "point_cloud_rebuild": strategy.get("point_cloud_rebuild"),
                    "distance_model": strategy.get("distance_model"),
                    "make_watertight": strategy.get("make_watertight"),
                    "heal_seconds": timing.get("heal_seconds"),
                    "leapfrog_validation_seconds": timing.get("leapfrog_validation_seconds"),
                    "fidelity_seconds": timing.get("fidelity_seconds"),
                    "total_seconds": timing.get("total_seconds"),
                    "leapfrog_ready": score.get("leapfrog_ready"),
                    "acceptance_passed": acceptance.get("ready"),
                    "acceptance_failed_checks": ";".join(acceptance.get("failed_checks", [])),
                    "roundtrip_loadable": leapfrog_validation.get("roundtrip_loadable"),
                    "self_intersection_checked": self_intersections.get("checked"),
                    "self_intersection_pairs": self_intersections.get("intersecting_pairs"),
                    "self_intersection_faces": self_intersections.get("intersecting_faces"),
                    "watertight": after.get("watertight"),
                    "boundary_edges": after.get("boundary_edges"),
                    "nonmanifold_edges": after.get("nonmanifold_edges"),
                    "degenerate_faces": after.get("degenerate_faces"),
                    "duplicate_faces": after.get("duplicate_faces"),
                    "mean_distance_normalized": fidelity.get("mean_distance_normalized"),
                    "p95_distance_normalized": fidelity.get("p95_distance_normalized"),
                    "area_ratio_delta": fidelity.get("area_ratio_delta"),
                    "volume_ratio_delta": fidelity.get("volume_ratio_delta"),
                    "component_count_delta": fidelity.get("component_count_delta"),
                    "total_score": score.get("total"),
                }
            )
    return ledger_path


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
    external_hint_path: Optional[Path | Sequence[Path]] = None,
    max_file_bytes: int = 0,
    max_faces: int = 120000,
    max_vertices: int = 80000,
    allow_decimation: bool = True,
) -> PreviewPayload:
    if not path.exists():
        raise FileNotFoundError(f"Preview input not found: {path}")
    if max_file_bytes > 0 and path.stat().st_size > max_file_bytes:
        raise PreviewSkippedError(
            f"Preview skipped: {path.name} is larger than the safe preview limit of {max_file_bytes // 1_000_000} MB."
        )

    mesh = load_mesh(path)
    preview_hints: list[dict] = []
    hint_paths: list[str] = []
    if external_hint_path is not None:
        normalized_hints = load_and_normalize_external_issues(external_hint_path, mesh)
        preview_hints = [dict(issue) for issue in (normalized_hints.get("issues") or [])]
        hint_paths = [str(item) for item in (normalized_hints.get("paths") or [])]
    if allow_decimation:
        preview_mesh, decimated = make_preview_mesh(mesh, max_faces=max_faces, max_vertices=max_vertices)
    else:
        preview_mesh = mesh.copy()
        decimated = False
    return PreviewPayload(
        source=str(path),
        original_vertices=int(len(mesh.vertices)),
        original_faces=int(len(mesh.faces)),
        preview_vertices=int(len(preview_mesh.vertices)),
        preview_faces=int(len(preview_mesh.faces)),
        decimated=decimated,
        vertices=np.asarray(preview_mesh.vertices, dtype=float).copy(),
        faces=np.asarray(preview_mesh.faces, dtype=np.int64).copy(),
        hint_issues=preview_hints,
        hint_paths=hint_paths,
    )


def execute_heal_strategy_on_mesh(
    mesh_in: trimesh.Trimesh,
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    external_hints: Optional[dict] = None,
    rebuild_triangles: bool = False,
    nonmanifold_edge_repair: bool = False,
    nonmanifold_edge_radius: float = 0.0,
    localized_intersection_repair: bool = False,
    point_cloud_rebuild: str = "none",
    distance_model: str = "none",
    distance_offset: float = 0.0,
    distance_grid_spacing: float = 0.0,
    make_watertight: bool = False,
    return_surface_after_watertight: bool = False,
    advanced_backend: str = "none",
    exact_arrangements_executable: Optional[Path] = None,
    tetra_backend_executable: Optional[Path] = None,
    cgal_backend_executable: Optional[Path] = None,
    cgal_alpha: Optional[float] = None,
    cgal_offset: Optional[float] = None,
    cgal_alpha_relative: Optional[float] = None,
    cgal_offset_relative: Optional[float] = None,
    cgal_repair_merge_boundary_vertices: bool = True,
    cgal_repair_merge_reversible_components: bool = True,
    cgal_repair_stitch_borders: bool = True,
    cgal_repair_duplicate_non_manifold_vertices: bool = True,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    normalized_advanced_backend = normalize_advanced_heal_backend(advanced_backend)
    normalized_point_cloud_rebuild = normalize_point_cloud_rebuild_mode(point_cloud_rebuild)
    normalized_distance_model = normalize_distance_model_mode(distance_model)
    use_surface_return = bool(return_surface_after_watertight)
    use_watertight_step = bool(make_watertight or use_surface_return)
    use_advanced_backend = normalized_advanced_backend != "none" and not use_surface_return
    use_distance_model = normalized_distance_model != "none"
    use_hint_seeded_rebuild = bool(rebuild_triangles and external_hints and (external_hints.get("issues") or []))
    if use_distance_model and distance_offset <= 0.0:
        raise ValueError("Distance-model offset must be greater than zero when distance model generation is enabled.")

    extra_steps = 0
    if use_hint_seeded_rebuild:
        extra_steps += 1
    if rebuild_triangles:
        extra_steps += 1
    if nonmanifold_edge_repair:
        extra_steps += 1
    if localized_intersection_repair:
        extra_steps += 1
    if normalized_point_cloud_rebuild != "none":
        extra_steps += 1
    if use_distance_model:
        extra_steps += 1
    if use_watertight_step:
        extra_steps += 1
    if use_advanced_backend:
        extra_steps += 1
    heal_steps = 0 if (use_advanced_backend or use_surface_return or use_distance_model) else estimate_heal_mesh_steps(mesh_in)
    progress_tracker = ProgressTracker(progress_callback, 2 + extra_steps + heal_steps)

    emit_status(status_callback, "Computing input report")
    before = mesh_report(mesh_in, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Input", before)
    progress_tracker.advance()

    rebuild_report = None
    hint_seeded_rebuild_report = None
    nonmanifold_edge_repair_report = None
    localized_intersection_report = None
    point_cloud_rebuild_report = None
    distance_model_report = None
    watertight_repair_report = None
    advanced_backend_report = None
    intermediate_validations: List[dict] = []
    mesh_for_heal = mesh_in.copy()
    if use_surface_return and normalized_advanced_backend != "none":
        emit_status(status_callback, "Skipping advanced backend because return-surface-after-watertight is enabled")
    if use_hint_seeded_rebuild:
        mesh_for_heal, hint_seeded_rebuild_report = rebuild_mesh_from_external_issues(
            mesh_for_heal,
            external_hints,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="hint-seeded rebuild",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()
    if rebuild_triangles:
        mesh_for_heal, rebuild_report = rebuild_mesh_without_triangle_overlaps(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="triangle rebuild",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if nonmanifold_edge_repair:
        mesh_for_heal, nonmanifold_edge_repair_report = repair_nonmanifold_edges_with_cylinders(
            mesh_for_heal,
            radius=nonmanifold_edge_radius,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="non-manifold edge repair",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if localized_intersection_repair:
        mesh_for_heal, localized_intersection_report = repair_localized_self_intersections(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="localized self-intersection repair",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if normalized_point_cloud_rebuild != "none":
        mesh_for_heal, point_cloud_rebuild_report = rebuild_solid_from_triangle_centers(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="point-cloud rebuild",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if use_watertight_step and (use_advanced_backend or use_surface_return):
        mesh_for_heal, watertight_repair_report = attempt_watertight_repair(
            mesh_for_heal,
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage="watertight repair",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if use_advanced_backend:
        mesh_for_heal, advanced_backend_report = run_advanced_exact_tetrahedral_backend(
            mesh_for_heal,
            advanced_backend=normalized_advanced_backend,
            exact_arrangements_executable=exact_arrangements_executable,
            tetra_backend_executable=tetra_backend_executable,
            cgal_backend_executable=cgal_backend_executable,
            cgal_alpha=cgal_alpha,
            cgal_offset=cgal_offset,
            cgal_alpha_relative=cgal_alpha_relative,
            cgal_offset_relative=cgal_offset_relative,
            cgal_repair_merge_boundary_vertices=cgal_repair_merge_boundary_vertices,
            cgal_repair_merge_reversible_components=cgal_repair_merge_reversible_components,
            cgal_repair_stitch_borders=cgal_repair_stitch_borders,
            cgal_repair_duplicate_non_manifold_vertices=cgal_repair_duplicate_non_manifold_vertices,
            status_callback=status_callback,
        )
        mesh_for_heal, validation_report = sanitize_intermediate_mesh(
            mesh_for_heal,
            stage=f"advanced backend {normalized_advanced_backend}",
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            status_callback=status_callback,
        )
        intermediate_validations.append(validation_report)
        progress_tracker.advance()

    if use_distance_model:
        if normalized_distance_model == "distance-hull":
            mesh_out, distance_model_report = build_distance_hull(
                mesh_for_heal,
                offset_distance=distance_offset,
                grid_spacing=distance_grid_spacing,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=status_callback,
            )
        else:
            mesh_out, distance_model_report = build_surface_shell(
                mesh_for_heal,
                offset_distance=distance_offset,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
                status_callback=status_callback,
            )
        progress_tracker.advance()
    elif use_surface_return:
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
            emit_status(
                status_callback,
                "Legacy component healing finished with solid/watertight intent still active; attempting watertight repair on the combined result.",
            )
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

    report = {
        "before": asdict(before),
        "after": asdict(after),
    }
    if hint_seeded_rebuild_report is not None:
        report["hint_seeded_triangle_rebuild"] = hint_seeded_rebuild_report
    if rebuild_report is not None:
        report["triangle_rebuild"] = rebuild_report
    if nonmanifold_edge_repair_report is not None:
        report["nonmanifold_edge_repair"] = nonmanifold_edge_repair_report
    if localized_intersection_report is not None:
        report["localized_intersection_repair"] = localized_intersection_report
    if point_cloud_rebuild_report is not None:
        report["point_cloud_rebuild"] = point_cloud_rebuild_report
    if distance_model_report is not None:
        report["distance_model"] = distance_model_report
    if watertight_repair_report is not None:
        report["watertight_repair"] = watertight_repair_report
    if advanced_backend_report is not None:
        report["advanced_backend"] = advanced_backend_report
    if intermediate_validations:
        report["intermediate_validations"] = intermediate_validations
    return {
        "mesh": mesh_out,
        "report": report,
    }


def run_heal_pipeline(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    external_hint_path: Optional[Path | Sequence[Path]] = None,
    intended_mesh_type: str = "auto",
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    rebuild_triangles: bool = True,
    nonmanifold_edge_repair: bool = False,
    nonmanifold_edge_radius: float = 0.0,
    localized_intersection_repair: bool = False,
    point_cloud_rebuild: str = "none",
    distance_model: str = "none",
    distance_offset: float = 0.0,
    distance_grid_spacing: float = 0.0,
    make_watertight: bool = False,
    return_surface_after_watertight: bool = False,
    advanced_backend: str = "none",
    exact_arrangements_executable: Optional[Path] = None,
    tetra_backend_executable: Optional[Path] = None,
    cgal_backend_executable: Optional[Path] = None,
    cgal_alpha: Optional[float] = None,
    cgal_offset: Optional[float] = None,
    cgal_alpha_relative: Optional[float] = None,
    cgal_offset_relative: Optional[float] = None,
    cgal_repair_merge_boundary_vertices: bool = True,
    cgal_repair_merge_reversible_components: bool = True,
    cgal_repair_stitch_borders: bool = True,
    cgal_repair_duplicate_non_manifold_vertices: bool = True,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    emit_status(status_callback, f"Loading {input_path.name}")
    normalized_intended_mesh_type = normalize_intended_mesh_type(intended_mesh_type)
    try:
        mesh_in = load_mesh(input_path)
    except SkippedInputError as exc:
        emit_status(status_callback, f"Skipping {input_path.name}: {exc}")
        report = build_skipped_input_report("heal", input_path, output_path, exc)
        write_json_report(report, report_path)
        return report

    external_hints = None
    external_hint_summary = summarize_external_issue_payload(None)
    if external_hint_path is not None:
        hint_paths = [external_hint_path] if isinstance(external_hint_path, Path) else [Path(path) for path in external_hint_path]
        if len(hint_paths) == 1:
            emit_status(status_callback, f"Loading external repair hints from {hint_paths[0].name}")
        else:
            emit_status(status_callback, f"Loading external repair hints from {len(hint_paths)} files")
        external_hints = load_and_normalize_external_issues(hint_paths, mesh_in, merge_eps=merge_eps)
        external_hint_summary = summarize_external_issue_payload(external_hints)
    external_hint_summary = apply_intended_mesh_type_to_hint_summary(external_hint_summary, normalized_intended_mesh_type)

    effective_rebuild_triangles = bool(rebuild_triangles or external_hint_summary["force_rebuild_triangles"])
    effective_nonmanifold_edge_repair = bool(nonmanifold_edge_repair or external_hint_summary["force_nonmanifold_edge_repair"])
    effective_localized_intersection_repair = bool(localized_intersection_repair or external_hint_summary["force_localized_intersection_repair"])
    effective_make_watertight = bool(
        make_watertight
        or external_hint_summary["force_make_watertight"]
        or normalized_intended_mesh_type == "solid"
    )

    execution = execute_heal_strategy_on_mesh(
        mesh_in,
        merge_eps=merge_eps,
        area_eps=area_eps,
        dedup_decimals=dedup_decimals,
        external_hints=external_hints,
        rebuild_triangles=effective_rebuild_triangles,
        nonmanifold_edge_repair=effective_nonmanifold_edge_repair,
        nonmanifold_edge_radius=nonmanifold_edge_radius,
        localized_intersection_repair=effective_localized_intersection_repair,
        point_cloud_rebuild=point_cloud_rebuild,
        distance_model=distance_model,
        distance_offset=distance_offset,
        distance_grid_spacing=distance_grid_spacing,
        make_watertight=effective_make_watertight,
        return_surface_after_watertight=return_surface_after_watertight,
        advanced_backend=advanced_backend,
        exact_arrangements_executable=exact_arrangements_executable,
        tetra_backend_executable=tetra_backend_executable,
        cgal_backend_executable=cgal_backend_executable,
        cgal_alpha=cgal_alpha,
        cgal_offset=cgal_offset,
        cgal_alpha_relative=cgal_alpha_relative,
        cgal_offset_relative=cgal_offset_relative,
        cgal_repair_merge_boundary_vertices=cgal_repair_merge_boundary_vertices,
        cgal_repair_merge_reversible_components=cgal_repair_merge_reversible_components,
        cgal_repair_stitch_borders=cgal_repair_stitch_borders,
        cgal_repair_duplicate_non_manifold_vertices=cgal_repair_duplicate_non_manifold_vertices,
        status_callback=status_callback,
        progress_callback=progress_callback,
    )
    mesh_out = execution["mesh"]
    core_report = execution["report"]

    emit_status(status_callback, f"Writing {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_mesh(mesh_out, output_path)

    report = {
        "mode": "heal",
        "input": str(input_path),
        "output": str(output_path),
        "intended_mesh_type": normalized_intended_mesh_type,
        "external_hints": external_hints,
        "external_hint_summary": external_hint_summary,
        "effective_options": {
            "intended_mesh_type": normalized_intended_mesh_type,
            "rebuild_triangles": effective_rebuild_triangles,
            "nonmanifold_edge_repair": effective_nonmanifold_edge_repair,
            "localized_intersection_repair": effective_localized_intersection_repair,
            "make_watertight": effective_make_watertight,
        },
        **core_report,
    }
    write_json_report(report, report_path)
    return report


def run_autoresearch_pipeline(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    external_hint_path: Optional[Path] = None,
    intended_mesh_type: str = "auto",
    merge_eps: float = 1e-8,
    area_eps: float = 1e-12,
    dedup_decimals: int = 8,
    nonmanifold_edge_radius: float = 0.0,
    allow_aggressive_modes: bool = False,
    fast_leapfrog: bool = False,
    max_candidates: int = 0,
    time_budget_seconds: float = 0.0,
    candidate_timeout_seconds: float = 0.0,
    self_intersection_timeout_seconds: float = 0.0,
    fidelity_sample_point_count: int = 0,
    max_mean_distance_normalized: float = 0.02,
    max_p95_distance_normalized: float = 0.05,
    max_component_count_delta: float = 5.0,
    max_volume_ratio_delta: float = 0.25,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    normalized_intended_mesh_type = normalize_intended_mesh_type(intended_mesh_type)
    message = (
        "The autoresearch workflow is deprecated. Use the guided heal workflow with an intended mesh type "
        "and optional external problem hints instead."
    )
    emit_status(status_callback, message)
    report = build_deprecated_feature_report(
        "autoresearch-heal",
        message,
        input_path=input_path,
        output_path=output_path,
        extra={
            "intended_mesh_type": normalized_intended_mesh_type,
        },
    )
    write_json_report(report, report_path)
    return report

    emit_status(status_callback, f"Loading {input_path.name}")
    try:
        mesh_in = load_mesh(input_path)
    except SkippedInputError as exc:
        emit_status(status_callback, f"Skipping {input_path.name}: {exc}")
        report = build_skipped_input_report(
            "autoresearch-heal",
            input_path,
            output_path,
            exc,
            extra={
                "search": {
                    "fast_leapfrog": bool(fast_leapfrog),
                    "requested_max_candidates": int(max_candidates),
                    "time_budget_seconds": float(max(0.0, time_budget_seconds)),
                    "candidate_timeout_seconds": float(max(0.0, candidate_timeout_seconds)),
                    "self_intersection_timeout_seconds": float(max(0.0, self_intersection_timeout_seconds)),
                },
                "selected_candidate": None,
                "best_overall_candidate": None,
                "candidates": [],
            },
        )
        write_json_report(report, report_path)
        return report

    external_hints = None
    external_hint_summary = summarize_external_issue_payload(None)
    if external_hint_path is not None:
        emit_status(status_callback, f"Loading external repair hints from {external_hint_path.name}")
        external_hints = load_and_normalize_external_issues(external_hint_path, mesh_in, merge_eps=merge_eps)
        external_hint_summary = summarize_external_issue_payload(external_hints)
    external_hint_summary = apply_intended_mesh_type_to_hint_summary(external_hint_summary, normalized_intended_mesh_type)

    before = mesh_report(mesh_in, area_eps=area_eps)
    emit_mesh_topology_status(status_callback, "Input", before)

    effective_max_candidates = max(0, int(max_candidates))
    if fast_leapfrog and effective_max_candidates == 0:
        effective_max_candidates = AUTORESEARCH_FAST_MAX_CANDIDATES
    effective_time_budget_seconds = max(0.0, float(time_budget_seconds))
    if fast_leapfrog and effective_time_budget_seconds <= 0.0:
        effective_time_budget_seconds = AUTORESEARCH_FAST_TIME_BUDGET_SECONDS
    effective_candidate_timeout_seconds = max(0.0, float(candidate_timeout_seconds))
    if fast_leapfrog and effective_candidate_timeout_seconds <= 0.0:
        effective_candidate_timeout_seconds = AUTORESEARCH_FAST_CANDIDATE_TIMEOUT_SECONDS
    effective_self_intersection_timeout_seconds = max(0.0, float(self_intersection_timeout_seconds))
    if effective_self_intersection_timeout_seconds <= 0.0:
        effective_self_intersection_timeout_seconds = AUTORESEARCH_SELF_INTERSECTION_TIMEOUT_SECONDS
    effective_fidelity_sample_point_count = max(0, int(fidelity_sample_point_count))
    if effective_fidelity_sample_point_count == 0:
        effective_fidelity_sample_point_count = (
            AUTORESEARCH_FAST_SAMPLE_POINT_COUNT if fast_leapfrog else AUTORESEARCH_SAMPLE_POINT_COUNT
        )
    acceptance_thresholds = LeapfrogAcceptanceThresholds(
        max_mean_distance_normalized=float(max_mean_distance_normalized),
        max_p95_distance_normalized=float(max_p95_distance_normalized),
        max_component_count_delta=float(max_component_count_delta),
        max_volume_ratio_delta=float(max_volume_ratio_delta),
    )
    topology_bucket = classify_autoresearch_topology_bucket(before)
    history_ledgers = discover_autoresearch_ledger_paths(input_path, output_path, report_path)
    history = load_autoresearch_history(history_ledgers, topology_bucket)

    all_candidates = build_autoresearch_candidates(
        before,
        allow_aggressive_modes=allow_aggressive_modes and not fast_leapfrog,
        fast_leapfrog=fast_leapfrog,
        history=history,
        external_hint_summary=external_hint_summary,
        intended_mesh_type=normalized_intended_mesh_type,
    )
    candidates = all_candidates[:effective_max_candidates] if effective_max_candidates > 0 else all_candidates
    reference_component_count = len(
        split_disconnected_components(
            preprocess_heal_mesh(
                mesh_in,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
            )
        )
    )
    mesh_scale = estimate_mesh_scale(mesh_in)
    progress_tracker = ProgressTracker(progress_callback, len(candidates) + 2)
    progress_tracker.advance()

    best_overall_mesh: Optional[trimesh.Trimesh] = None
    best_overall_report: Optional[dict] = None
    best_overall_sort_key: Optional[Tuple[float, ...]] = None
    best_mesh: Optional[trimesh.Trimesh] = None
    best_report: Optional[dict] = None
    best_sort_key: Optional[Tuple[float, ...]] = None
    candidate_summaries: List[dict] = []
    search_started_at = time.perf_counter()
    stopped_early = False
    stop_reason = None

    for index, candidate in enumerate(candidates, start=1):
        elapsed_before_candidate = time.perf_counter() - search_started_at
        if effective_time_budget_seconds > 0.0 and best_report is not None and elapsed_before_candidate >= effective_time_budget_seconds:
            stopped_early = True
            stop_reason = (
                f"time budget of {effective_time_budget_seconds:.1f}s reached after evaluating {index - 1} candidates"
            )
            emit_status(status_callback, f"Stopping autoresearch early: {stop_reason}")
            break

        emit_status(status_callback, f"Evaluating candidate {index}/{len(candidates)}: {candidate.name}")
        candidate_started_at = time.perf_counter()

        def candidate_status(message: str, candidate_name: str = candidate.name) -> None:
            emit_status(status_callback, f"[{candidate_name}] {message}")

        distance_offset = 0.0
        if candidate.distance_model != "none":
            distance_offset = max(mesh_scale * candidate.distance_offset_ratio, merge_eps * 250.0)
        distance_grid_spacing = 0.0
        if candidate.distance_model != "none" and candidate.distance_grid_spacing_ratio > 0.0:
            distance_grid_spacing = max(mesh_scale * candidate.distance_grid_spacing_ratio, merge_eps * 100.0)

        execution_mode = "in-process"
        try:
            heal_started_at = time.perf_counter()
            execution_kwargs = {
                "merge_eps": float(merge_eps),
                "area_eps": float(area_eps),
                "dedup_decimals": int(dedup_decimals),
                "external_hints": external_hints,
                "rebuild_triangles": bool(candidate.rebuild_triangles),
                "nonmanifold_edge_repair": bool(candidate.nonmanifold_edge_repair),
                "nonmanifold_edge_radius": float(nonmanifold_edge_radius),
                "localized_intersection_repair": bool(candidate.localized_intersection_repair),
                "point_cloud_rebuild": candidate.point_cloud_rebuild,
                "distance_model": candidate.distance_model,
                "distance_offset": float(distance_offset),
                "distance_grid_spacing": float(distance_grid_spacing),
                "make_watertight": bool(candidate.make_watertight),
            }
            if effective_candidate_timeout_seconds > 0.0 and candidate_needs_subprocess_timeout(candidate):
                execution_mode = "subprocess"
                emit_status(
                    status_callback,
                    f"Candidate {candidate.name} is running in an isolated subprocess with a {effective_candidate_timeout_seconds:.1f}s hard timeout",
                )
                with tempfile.TemporaryDirectory(prefix="mesh_heal_candidate_") as temp_dir_name:
                    output_snapshot_path = Path(temp_dir_name) / "candidate_output.npz"
                    execution = run_heal_candidate_with_timeout(
                        input_path=input_path,
                        output_snapshot_path=output_snapshot_path,
                        execution_kwargs=execution_kwargs,
                        timeout_seconds=effective_candidate_timeout_seconds,
                    )
            else:
                execution = execute_heal_strategy_on_mesh(
                    mesh_in,
                    merge_eps=merge_eps,
                    area_eps=area_eps,
                    dedup_decimals=dedup_decimals,
                    external_hints=external_hints,
                    rebuild_triangles=candidate.rebuild_triangles,
                    nonmanifold_edge_repair=candidate.nonmanifold_edge_repair,
                    nonmanifold_edge_radius=nonmanifold_edge_radius,
                    localized_intersection_repair=candidate.localized_intersection_repair,
                    point_cloud_rebuild=candidate.point_cloud_rebuild,
                    distance_model=candidate.distance_model,
                    distance_offset=distance_offset,
                    distance_grid_spacing=distance_grid_spacing,
                    make_watertight=candidate.make_watertight,
                    status_callback=candidate_status,
                    progress_callback=None,
                )
                execution["execution_mode"] = execution_mode
            heal_seconds = time.perf_counter() - heal_started_at
            mesh_out = execution["mesh"]
            candidate_report = execution["report"]
            after = MeshReport(**candidate_report["after"])
            execution_mode = execution.get("execution_mode", execution_mode)
            leapfrog_started_at = time.perf_counter()
            skip_reason = should_skip_leapfrog_roundtrip(after, fast_leapfrog=fast_leapfrog)
            if skip_reason is None:
                leapfrog_validation = validate_leapfrog_roundtrip(
                    mesh_out,
                    area_eps=area_eps,
                    self_intersection_timeout_seconds=effective_self_intersection_timeout_seconds,
                )
            else:
                leapfrog_validation = make_skipped_leapfrog_validation(skip_reason)
            leapfrog_seconds = time.perf_counter() - leapfrog_started_at
            fidelity_started_at = time.perf_counter()
            fidelity = measure_shape_fidelity(
                mesh_in,
                mesh_out,
                reference_report=before,
                candidate_report=after,
                reference_component_count=reference_component_count,
                sample_point_count=effective_fidelity_sample_point_count,
            )
            fidelity_seconds = time.perf_counter() - fidelity_started_at
            candidate_total_seconds = time.perf_counter() - candidate_started_at
            acceptance = evaluate_leapfrog_acceptance(
                after,
                leapfrog_validation,
                fidelity,
                acceptance_thresholds,
            )
            score = score_autoresearch_candidate(
                candidate,
                after,
                leapfrog_validation,
                fidelity,
                acceptance,
                runtime_seconds=candidate_total_seconds,
            )
            summary = {
                "name": candidate.name,
                "strategy": serialize_heal_search_candidate(candidate),
                "after": candidate_report["after"],
                "leapfrog_validation": leapfrog_validation,
                "fidelity": fidelity,
                "acceptance": acceptance,
                "status": "completed",
                "execution_mode": execution_mode,
                "timed_out": False,
                "timing": {
                    "heal_seconds": float(heal_seconds),
                    "leapfrog_validation_seconds": float(leapfrog_seconds),
                    "fidelity_seconds": float(fidelity_seconds),
                    "total_seconds": float(candidate_total_seconds),
                },
                "score": score,
            }
            sort_key = tuple(float(value) for value in score["ranking"])
            candidate_report_summary = {
                "name": candidate.name,
                "strategy": serialize_heal_search_candidate(candidate),
                "heal_report": candidate_report,
                "leapfrog_validation": leapfrog_validation,
                "fidelity": fidelity,
                "acceptance": acceptance,
                "status": "completed",
                "execution_mode": summary["execution_mode"],
                "timed_out": False,
                "timing": summary["timing"],
                "score": score,
            }
            if best_overall_sort_key is None or sort_key < best_overall_sort_key:
                best_overall_sort_key = sort_key
                best_overall_mesh = mesh_out.copy()
                best_overall_report = candidate_report_summary
            if score["leapfrog_ready"] and (best_sort_key is None or sort_key < best_sort_key):
                best_sort_key = sort_key
                best_mesh = mesh_out.copy()
                best_report = candidate_report_summary
            emit_status(
                status_callback,
                (
                    f"Candidate {candidate.name} finished in {candidate_total_seconds:.1f}s; "
                    f"watertight={after.watertight}, boundaries={after.boundary_edges}, "
                    f"nonmanifold={after.nonmanifold_edges}, "
                    f"self_intersections={leapfrog_validation.get('self_intersections', {}).get('intersecting_pairs')}, "
                    f"acceptance_ready={score['leapfrog_ready']}"
                ),
            )
        except TimeoutError as exc:
            candidate_total_seconds = time.perf_counter() - candidate_started_at
            summary = {
                "name": candidate.name,
                "strategy": serialize_heal_search_candidate(candidate),
                "error": str(exc),
                "status": "timed_out",
                "execution_mode": "subprocess",
                "timed_out": True,
                "acceptance": {
                    "ready": False,
                    "topology_ready": False,
                    "fidelity_ready": False,
                    "failed_checks": ["candidate_timed_out"],
                    "thresholds": asdict(acceptance_thresholds),
                },
                "timing": {
                    "heal_seconds": float(candidate_total_seconds),
                    "leapfrog_validation_seconds": 0.0,
                    "fidelity_seconds": 0.0,
                    "total_seconds": float(candidate_total_seconds),
                },
                "score": {
                    "leapfrog_ready": False,
                    "topology_penalty": AUTORESEARCH_ERROR_SCORE,
                    "fidelity_penalty": AUTORESEARCH_ERROR_SCORE,
                    "complexity_penalty": AUTORESEARCH_ERROR_SCORE,
                    "runtime_penalty": float(min(candidate_total_seconds, 600.0)),
                    "total": AUTORESEARCH_ERROR_SCORE,
                    "ranking": [
                        1.0,
                        1.0,
                        AUTORESEARCH_ERROR_SCORE,
                        AUTORESEARCH_ERROR_SCORE,
                        AUTORESEARCH_ERROR_SCORE,
                        float(min(candidate_total_seconds, 600.0)),
                        AUTORESEARCH_ERROR_SCORE,
                    ],
                },
            }
            emit_status(status_callback, f"Candidate {candidate.name} timed out after {candidate_total_seconds:.1f}s: {exc}")
        except Exception as exc:
            candidate_total_seconds = time.perf_counter() - candidate_started_at
            summary = {
                "name": candidate.name,
                "strategy": serialize_heal_search_candidate(candidate),
                "error": str(exc),
                "status": "failed",
                "execution_mode": execution_mode,
                "timed_out": False,
                "acceptance": {
                    "ready": False,
                    "topology_ready": False,
                    "fidelity_ready": False,
                    "failed_checks": ["candidate_failed"],
                    "thresholds": asdict(acceptance_thresholds),
                },
                "timing": {
                    "heal_seconds": float(candidate_total_seconds),
                    "leapfrog_validation_seconds": 0.0,
                    "fidelity_seconds": 0.0,
                    "total_seconds": float(candidate_total_seconds),
                },
                "score": {
                    "leapfrog_ready": False,
                    "topology_penalty": AUTORESEARCH_ERROR_SCORE,
                    "fidelity_penalty": AUTORESEARCH_ERROR_SCORE,
                    "complexity_penalty": AUTORESEARCH_ERROR_SCORE,
                    "runtime_penalty": float(min(candidate_total_seconds, 600.0)),
                    "total": AUTORESEARCH_ERROR_SCORE,
                    "ranking": [
                        1.0,
                        1.0,
                        AUTORESEARCH_ERROR_SCORE,
                        AUTORESEARCH_ERROR_SCORE,
                        AUTORESEARCH_ERROR_SCORE,
                        float(min(candidate_total_seconds, 600.0)),
                        AUTORESEARCH_ERROR_SCORE,
                    ],
                },
            }
            emit_status(status_callback, f"Candidate {candidate.name} failed after {candidate_total_seconds:.1f}s: {exc}")
        candidate_summaries.append(summary)
        progress_tracker.advance()

    if best_overall_mesh is None or best_overall_report is None:
        raise RuntimeError("Autoresearch could not produce any candidate output.")

    selected_report = best_report if best_report is not None else best_overall_report

    candidate_summaries.sort(key=lambda item: tuple(float(value) for value in item["score"]["ranking"]))
    for rank, summary in enumerate(candidate_summaries, start=1):
        summary["rank"] = int(rank)
        summary["selected"] = bool(summary["name"] == selected_report["name"])

    if best_mesh is None or best_report is None:
        total_search_seconds = time.perf_counter() - search_started_at
        emit_status(status_callback, "No candidate satisfied Leapfrog acceptance thresholds.")
        report = {
            "status": "no_accepted_candidate",
            "mode": "autoresearch-heal",
            "input": str(input_path),
            "output": str(output_path),
            "intended_mesh_type": normalized_intended_mesh_type,
            "before": asdict(before),
            "search": {
                "fast_leapfrog": bool(fast_leapfrog),
                "allow_aggressive_modes": bool(allow_aggressive_modes and not fast_leapfrog),
                "external_hint_summary": external_hint_summary,
                "requested_max_candidates": int(max_candidates),
                "evaluated_candidates": int(len(candidate_summaries)),
                "generated_candidates": int(len(all_candidates)),
                "candidate_limit": int(effective_max_candidates),
                "time_budget_seconds": float(effective_time_budget_seconds),
                "candidate_timeout_seconds": float(effective_candidate_timeout_seconds),
                "self_intersection_timeout_seconds": float(effective_self_intersection_timeout_seconds),
                "fidelity_sample_point_count": int(effective_fidelity_sample_point_count),
                "topology_bucket": topology_bucket,
                "history": history,
                "acceptance_thresholds": asdict(acceptance_thresholds),
                "stopped_early": bool(stopped_early),
                "stop_reason": stop_reason,
                "total_seconds": float(total_search_seconds),
                "candidate_order": [candidate.name for candidate in candidates],
            },
            "external_hints": external_hints,
            "selected_candidate": None,
            "best_overall_candidate": best_overall_report,
            "candidates": candidate_summaries,
            "failure": {
                "reason": "No candidate satisfied Leapfrog acceptance thresholds.",
                "best_overall_candidate": best_overall_report["name"],
                "failed_checks": best_overall_report.get("acceptance", {}).get("failed_checks", []),
            },
        }
        write_json_report(report, report_path)
        ledger_path = write_autoresearch_ledger(report, report_path)
        if ledger_path is not None:
            report["ledger"] = str(ledger_path)
            write_json_report(report, report_path)
        return report

    emit_status(
        status_callback,
        f"Selected {best_report['name']} (Leapfrog-accepted: {'yes' if best_report['score']['leapfrog_ready'] else 'no'})",
    )
    emit_status(status_callback, f"Writing {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_mesh(best_mesh, output_path)
    progress_tracker.advance()

    total_search_seconds = time.perf_counter() - search_started_at

    report = {
        "status": "completed",
        "mode": "autoresearch-heal",
        "input": str(input_path),
        "output": str(output_path),
        "intended_mesh_type": normalized_intended_mesh_type,
        "before": asdict(before),
        "search": {
            "fast_leapfrog": bool(fast_leapfrog),
            "allow_aggressive_modes": bool(allow_aggressive_modes and not fast_leapfrog),
            "external_hint_summary": external_hint_summary,
            "requested_max_candidates": int(max_candidates),
            "evaluated_candidates": int(len(candidate_summaries)),
            "generated_candidates": int(len(all_candidates)),
            "candidate_limit": int(effective_max_candidates),
            "time_budget_seconds": float(effective_time_budget_seconds),
            "candidate_timeout_seconds": float(effective_candidate_timeout_seconds),
            "self_intersection_timeout_seconds": float(effective_self_intersection_timeout_seconds),
            "fidelity_sample_point_count": int(effective_fidelity_sample_point_count),
            "topology_bucket": topology_bucket,
            "history": history,
            "acceptance_thresholds": asdict(acceptance_thresholds),
            "stopped_early": bool(stopped_early),
            "stop_reason": stop_reason,
            "total_seconds": float(total_search_seconds),
            "candidate_order": [candidate.name for candidate in candidates],
        },
        "external_hints": external_hints,
        "selected_candidate": best_report,
        "best_overall_candidate": best_overall_report,
        "candidates": candidate_summaries,
    }
    write_json_report(report, report_path)
    ledger_path = write_autoresearch_ledger(report, report_path)
    if ledger_path is not None:
        report["ledger"] = str(ledger_path)
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
    heal_parser.add_argument("--hints", type=Path, nargs="+", default=None, help="Optional JSON or DXF file set with external repair hints")
    heal_parser.add_argument("--merge-eps", type=float, default=1e-8, help="Vertex merge tolerance")
    heal_parser.add_argument("--area-eps", type=float, default=1e-12, help="Degenerate triangle area threshold")
    heal_parser.add_argument("--dedup-decimals", type=int, default=8, help="Rounding decimals for duplicate-face detection")
    heal_parser.add_argument(
        "--rebuild-triangles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild the mesh triangle by triangle before healing; when external hints are supplied, the rebuild starts from those problematic regions first",
    )
    heal_parser.add_argument(
        "--nonmanifold-edge-repair",
        action="store_true",
        help="Experimental repair that sleeves each non-manifold edge with a cylindrical hull and unions it into the solid",
    )
    heal_parser.add_argument(
        "--nonmanifold-edge-radius",
        type=float,
        default=0.0,
        help="Optional radius for non-manifold edge sleeve repair; default 0 chooses an automatic radius",
    )
    heal_parser.add_argument(
        "--localized-intersection-repair",
        action="store_true",
        help="Experimental localized self-intersection repair pass inspired by region partitioning workflows",
    )
    heal_parser.add_argument(
        "--point-cloud-rebuild",
        choices=list(POINT_CLOUD_REBUILD_MODES),
        default="none",
        help="Experimental rebuild from triangle centers plus normals using an oriented point-cloud surface reconstruction",
    )
    heal_parser.add_argument(
        "--distance-model",
        choices=list(DISTANCE_MODEL_MODES),
        default="none",
        help="Generate either a distance-field hull or a stitched offset surface shell around the current surface",
    )
    heal_parser.add_argument(
        "--distance-offset",
        type=float,
        default=0.0,
        help="Offset distance for the selected distance model; required when --distance-model is not none",
    )
    heal_parser.add_argument(
        "--distance-grid-spacing",
        type=float,
        default=0.0,
        help="Optional sampling grid spacing for distance-hull generation only; default 0 chooses automatically",
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
        choices=list(ALL_ADVANCED_HEAL_BACKENDS),
        default="none",
        help="Advanced backend: none, cgal-alpha-wrap, or cgal-repair. OpenMeshCraft backends remain parseable only to surface an explicit deprecation error.",
    )
    heal_parser.add_argument(
        "--exact-arrangements-exe",
        type=Path,
        default=None,
        help="Deprecated OpenMeshCraft path override; retained only for backwards-compatible argument parsing",
    )
    heal_parser.add_argument(
        "--tetra-backend-exe",
        type=Path,
        default=None,
        help="Deprecated tetra backend path override; retained only for backwards-compatible argument parsing",
    )
    heal_parser.add_argument(
        "--cgal-backend-exe",
        type=Path,
        default=None,
        help="Optional path to the CGAL backend executable",
    )
    heal_parser.add_argument(
        "--cgal-alpha",
        type=float,
        default=None,
        help="Optional absolute CGAL Alpha Wrap alpha value; only used with --advanced-backend cgal-alpha-wrap",
    )
    heal_parser.add_argument(
        "--cgal-offset",
        type=float,
        default=None,
        help="Optional absolute CGAL Alpha Wrap offset value; only used with --advanced-backend cgal-alpha-wrap",
    )
    heal_parser.add_argument(
        "--cgal-alpha-relative",
        type=float,
        default=1.0 / 50.0,
        help="Relative CGAL Alpha Wrap alpha as a fraction of the input bbox diagonal when --cgal-alpha is omitted",
    )
    heal_parser.add_argument(
        "--cgal-offset-relative",
        type=float,
        default=1.0 / 30.0,
        help="Relative CGAL Alpha Wrap offset as a fraction of alpha when --cgal-offset is omitted",
    )
    heal_parser.add_argument(
        "--cgal-repair-merge-boundary-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable merge_duplicated_vertices_in_boundary_cycles in CGAL repair mode",
    )
    heal_parser.add_argument(
        "--cgal-repair-merge-reversible-components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable merge_reversible_connected_components in CGAL repair mode",
    )
    heal_parser.add_argument(
        "--cgal-repair-stitch-borders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stitch_borders in CGAL repair mode",
    )
    heal_parser.add_argument(
        "--cgal-repair-duplicate-non-manifold-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable duplicate_non_manifold_vertices in CGAL repair mode",
    )

    autoresearch_parser = subparsers.add_parser(
        "autoresearch",
        help="Deprecated: use the guided heal workflow instead",
    )
    autoresearch_parser.add_argument("input", type=Path, help="Input mesh file")
    autoresearch_parser.add_argument("output", type=Path, help="Output healed mesh file")
    autoresearch_parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    autoresearch_parser.add_argument("--hints", type=Path, default=None, help="Optional JSON or DXF file with external repair hints")
    autoresearch_parser.add_argument("--merge-eps", type=float, default=1e-8, help="Vertex merge tolerance")
    autoresearch_parser.add_argument("--area-eps", type=float, default=1e-12, help="Degenerate triangle area threshold")
    autoresearch_parser.add_argument(
        "--dedup-decimals",
        type=int,
        default=8,
        help="Rounding decimals for duplicate-face detection",
    )
    autoresearch_parser.add_argument(
        "--nonmanifold-edge-radius",
        type=float,
        default=0.0,
        help="Optional radius for non-manifold edge sleeve candidates; default 0 chooses automatically",
    )
    autoresearch_parser.add_argument(
        "--allow-aggressive-modes",
        action="store_true",
        help="Include point-cloud and distance-hull fallback candidates when safe repair combinations are not enough",
    )
    autoresearch_parser.add_argument(
        "--fast-leapfrog",
        action="store_true",
        help="Use a reduced, Leapfrog-oriented candidate set with lower sampling and a default soft time budget",
    )
    autoresearch_parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on how many autoresearch candidates to evaluate; default 0 evaluates the full selected set",
    )
    autoresearch_parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=0.0,
        help="Optional soft total search budget; once exceeded, the search stops after the current candidate if a best result exists",
    )
    autoresearch_parser.add_argument(
        "--candidate-timeout-seconds",
        type=float,
        default=0.0,
        help="Hard timeout for isolated heavy candidates; default 0 disables it unless --fast-leapfrog enables the safe default",
    )
    autoresearch_parser.add_argument(
        "--self-intersection-timeout-seconds",
        type=float,
        default=0.0,
        help="Bounded timeout for Leapfrog self-intersection validation; default 0 uses the built-in validation timeout",
    )
    autoresearch_parser.add_argument(
        "--fidelity-samples",
        type=int,
        default=0,
        help="Optional surface sample count used for fidelity scoring; default 0 chooses automatically",
    )
    autoresearch_parser.add_argument(
        "--max-mean-distance-normalized",
        type=float,
        default=0.02,
        help="Leapfrog acceptance threshold for normalized mean surface drift",
    )
    autoresearch_parser.add_argument(
        "--max-p95-distance-normalized",
        type=float,
        default=0.05,
        help="Leapfrog acceptance threshold for normalized p95 surface drift",
    )
    autoresearch_parser.add_argument(
        "--max-component-count-delta",
        type=float,
        default=5.0,
        help="Leapfrog acceptance threshold for component-count drift",
    )
    autoresearch_parser.add_argument(
        "--max-volume-ratio-delta",
        type=float,
        default=0.25,
        help="Leapfrog acceptance threshold for relative volume drift when both meshes are watertight",
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
    if argv and argv[0] not in {"heal", "autoresearch", "boolean"}:
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
                external_hint_path=args.hints,
                merge_eps=args.merge_eps,
                area_eps=args.area_eps,
                dedup_decimals=args.dedup_decimals,
                rebuild_triangles=args.rebuild_triangles,
                nonmanifold_edge_repair=args.nonmanifold_edge_repair,
                nonmanifold_edge_radius=args.nonmanifold_edge_radius,
                localized_intersection_repair=args.localized_intersection_repair,
                point_cloud_rebuild=args.point_cloud_rebuild,
                distance_model=args.distance_model,
                distance_offset=args.distance_offset,
                distance_grid_spacing=args.distance_grid_spacing,
                make_watertight=args.make_watertight,
                return_surface_after_watertight=args.return_surface_after_watertight,
                advanced_backend=args.advanced_backend,
                exact_arrangements_executable=args.exact_arrangements_exe,
                tetra_backend_executable=args.tetra_backend_exe,
                cgal_backend_executable=args.cgal_backend_exe,
                cgal_alpha=args.cgal_alpha,
                cgal_offset=args.cgal_offset,
                cgal_alpha_relative=args.cgal_alpha_relative,
                cgal_offset_relative=args.cgal_offset_relative,
                cgal_repair_merge_boundary_vertices=args.cgal_repair_merge_boundary_vertices,
                cgal_repair_merge_reversible_components=args.cgal_repair_merge_reversible_components,
                cgal_repair_stitch_borders=args.cgal_repair_stitch_borders,
                cgal_repair_duplicate_non_manifold_vertices=args.cgal_repair_duplicate_non_manifold_vertices,
            )
        elif args.command == "autoresearch":
            if not args.input.exists():
                raise FileNotFoundError(f"Input not found: {args.input}")
            report = run_autoresearch_pipeline(
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
                external_hint_path=args.hints,
                merge_eps=args.merge_eps,
                area_eps=args.area_eps,
                dedup_decimals=args.dedup_decimals,
                nonmanifold_edge_radius=args.nonmanifold_edge_radius,
                allow_aggressive_modes=args.allow_aggressive_modes,
                fast_leapfrog=args.fast_leapfrog,
                max_candidates=args.max_candidates,
                time_budget_seconds=args.time_budget_seconds,
                candidate_timeout_seconds=args.candidate_timeout_seconds,
                self_intersection_timeout_seconds=args.self_intersection_timeout_seconds,
                fidelity_sample_point_count=args.fidelity_samples,
                max_mean_distance_normalized=args.max_mean_distance_normalized,
                max_p95_distance_normalized=args.max_p95_distance_normalized,
                max_component_count_delta=args.max_component_count_delta,
                max_volume_ratio_delta=args.max_volume_ratio_delta,
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
    if args.command == "autoresearch" and report.get("selected_candidate") is None:
        raise SystemExit("Autoresearch did not find a candidate that satisfied Leapfrog acceptance thresholds.")


if __name__ == "__main__":
    main()
