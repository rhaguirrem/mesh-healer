import json
import importlib
import multiprocessing
import os
from queue import Empty
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
os.environ.setdefault("QT_OPENGL", "software")

from PySide6.QtCore import QObject, QPointF, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QListWidget,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import shiboken6
except Exception:
    shiboken6 = None

import mesh_heal

try:
    from pyvistaqt import QtInteractor
except Exception:
    QtInteractor = None

try:
    import pyvista as pv
except Exception:
    pv = None

try:
    ti = importlib.import_module("taichi")
except Exception:
    ti = None


MESH_FILTER = "Mesh Files (*.dxf *.msh *.stl *.obj *.ply *.00t);;All Files (*.*)"
OUTPUT_FILTER = "Leapfrog Mesh Files (*.msh);;DXF Files (*.dxf);;Mesh Files (*.stl *.obj *.ply *.vtk);;Maptek Vulcan Files (*.00t);;All Files (*.*)"
JSON_FILTER = "JSON Files (*.json);;All Files (*.*)"
HINT_FILTER = "Hint Files (*.json *.dxf);;JSON Files (*.json);;DXF Files (*.dxf);;All Files (*.*)"
INTENDED_MESH_TYPE_OPTIONS = (
    ("Auto", "auto"),
    ("Solid", "solid"),
    ("Surface", "surface"),
)


def suggest_autoresearch_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_autoresearch.msh")


def suggest_autoresearch_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_report.json")


def create_intended_mesh_type_combo(default_value: str = "auto") -> QComboBox:
    combo = QComboBox()
    for label, value in INTENDED_MESH_TYPE_OPTIONS:
        combo.addItem(label, value)
    index = combo.findData(mesh_heal.normalize_intended_mesh_type(default_value))
    combo.setCurrentIndex(max(0, index))
    return combo


def describe_intended_mesh_type(intended_mesh_type: str) -> str:
    normalized = mesh_heal.normalize_intended_mesh_type(intended_mesh_type)
    if normalized == "solid":
        return "Solid"
    if normalized == "surface":
        return "Surface"
    return "Auto"


def configure_qt_opengl_backend() -> None:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)


def create_plotter_or_placeholder(parent: QWidget, unavailable_message: str, min_height: int = 320):
    if QtInteractor is None:
        placeholder = QLabel(unavailable_message)
        placeholder.setWordWrap(True)
        placeholder.setFrameShape(QFrame.StyledPanel)
        placeholder.setMinimumHeight(min_height)
        return None, placeholder, unavailable_message

    try:
        plotter = QtInteractor(parent, multi_samples=0)
        plotter.setMinimumHeight(min_height)
        plotter.set_background("#fafafa")
        plotter.view_isometric()
        return plotter, plotter.interactor, None
    except Exception as exc:
        placeholder = QLabel(f"{unavailable_message}\n\nRenderer initialization failed: {exc}")
        placeholder.setWordWrap(True)
        placeholder.setFrameShape(QFrame.StyledPanel)
        placeholder.setMinimumHeight(min_height)
        return None, placeholder, str(exc)


def create_viewer_placeholder(message: str, min_height: int) -> QLabel:
    placeholder = QLabel(message)
    placeholder.setWordWrap(True)
    placeholder.setFrameShape(QFrame.StyledPanel)
    placeholder.setMinimumHeight(min_height)
    return placeholder


_TAICHI_INITIALIZED = False
_TAICHI_INIT_ERROR = None


def ensure_taichi_initialized() -> str | None:
    global _TAICHI_INITIALIZED, _TAICHI_INIT_ERROR
    if _TAICHI_INITIALIZED:
        return None
    if _TAICHI_INIT_ERROR is not None:
        return _TAICHI_INIT_ERROR
    if ti is None:
        _TAICHI_INIT_ERROR = (
            "Taichi is not available in the active interpreter. Use a Python 3.13 environment for the Taichi preview, "
            "because Taichi does not currently ship wheels for Python 3.14."
        )
        return _TAICHI_INIT_ERROR
    try:
        ti.init(arch=ti.cpu)
    except Exception as exc:
        _TAICHI_INIT_ERROR = f"Taichi initialization failed: {exc}"
        return _TAICHI_INIT_ERROR
    _TAICHI_INITIALIZED = True
    return None


def _rotate_preview_vertices(
    vertices: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
    *,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float32)
    if origin is None:
        origin = np.mean(array, axis=0, dtype=np.float32) if len(array) > 0 else np.zeros(3, dtype=np.float32)
    centered = array - np.asarray(origin, dtype=np.float32)
    azimuth = np.deg2rad(float(azimuth_degrees))
    elevation = np.deg2rad(float(elevation_degrees))
    cos_azimuth = float(np.cos(azimuth))
    sin_azimuth = float(np.sin(azimuth))
    cos_elevation = float(np.cos(elevation))
    sin_elevation = float(np.sin(elevation))

    rotate_z = np.asarray(
        [
            [cos_azimuth, -sin_azimuth, 0.0],
            [sin_azimuth, cos_azimuth, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotate_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_elevation, -sin_elevation],
            [0.0, sin_elevation, cos_elevation],
        ],
        dtype=np.float32,
    )
    return centered @ rotate_z.T @ rotate_x.T


def _build_preview_projection_state(
    vertices: np.ndarray,
    width: int,
    height: int,
    *,
    azimuth_degrees: float,
    elevation_degrees: float,
    zoom: float,
    pan_offset: tuple[float, float],
) -> dict:
    vertices = np.asarray(vertices, dtype=np.float32)
    origin = np.mean(vertices, axis=0, dtype=np.float32) if len(vertices) > 0 else np.zeros(3, dtype=np.float32)
    rotated = _rotate_preview_vertices(
        vertices,
        azimuth_degrees,
        elevation_degrees,
        origin=origin,
    )
    projected = rotated[:, :2] if len(rotated) > 0 else np.zeros((0, 2), dtype=np.float32)
    if len(projected) == 0:
        return {
            "origin": origin,
            "azimuth_degrees": float(azimuth_degrees),
            "elevation_degrees": float(elevation_degrees),
            "scale": 1.0,
            "projection_center_xy": np.zeros(2, dtype=np.float32),
            "width": int(width),
            "height": int(height),
            "pan_offset": (float(pan_offset[0]), float(pan_offset[1])),
        }

    min_xy = np.min(projected, axis=0)
    max_xy = np.max(projected, axis=0)
    center_xy = (min_xy + max_xy) * 0.5
    span_xy = np.maximum(max_xy - min_xy, 1e-6)
    margin = 0.08
    scale = min(
        (width * (1.0 - 2.0 * margin)) / float(span_xy[0]),
        (height * (1.0 - 2.0 * margin)) / float(span_xy[1]),
    )
    scale *= max(0.1, float(zoom))
    return {
        "origin": origin,
        "azimuth_degrees": float(azimuth_degrees),
        "elevation_degrees": float(elevation_degrees),
        "scale": float(scale),
        "projection_center_xy": center_xy.astype(np.float32),
        "width": int(width),
        "height": int(height),
        "pan_offset": (float(pan_offset[0]), float(pan_offset[1])),
    }


def _project_preview_overlay_points(points: Sequence[Sequence[float]], projection_state: dict) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if len(array) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rotated = _rotate_preview_vertices(
        array,
        projection_state["azimuth_degrees"],
        projection_state["elevation_degrees"],
        origin=np.asarray(projection_state["origin"], dtype=np.float32),
    )
    projected = rotated[:, :2]
    normalized = (projected - np.asarray(projection_state["projection_center_xy"], dtype=np.float32)) * float(projection_state["scale"])
    screen = np.empty((len(array), 2), dtype=np.float32)
    screen[:, 0] = normalized[:, 0] + float(projection_state["width"]) * 0.5 + float(projection_state["pan_offset"][0])
    screen[:, 1] = float(projection_state["height"]) * 0.5 - normalized[:, 1] + float(projection_state["pan_offset"][1])
    return screen


def _compute_preview_projection(
    vertices: np.ndarray,
    width: int,
    height: int,
    *,
    azimuth_degrees: float = 45.0,
    elevation_degrees: float = 35.26438968,
    zoom: float = 1.0,
    pan_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    projection_state = _build_preview_projection_state(
        vertices,
        width,
        height,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
        zoom=zoom,
        pan_offset=pan_offset,
    )
    rotated = _rotate_preview_vertices(
        vertices,
        azimuth_degrees,
        elevation_degrees,
        origin=np.asarray(projection_state["origin"], dtype=np.float32),
    )
    projected = rotated[:, :2]
    center_xy = np.asarray(projection_state["projection_center_xy"], dtype=np.float32)
    scale = float(projection_state["scale"])
    normalized = (projected - center_xy) * scale
    screen = np.empty((len(vertices), 2), dtype=np.float32)
    screen[:, 0] = normalized[:, 0] + width * 0.5 + float(pan_offset[0])
    screen[:, 1] = height * 0.5 - normalized[:, 1] + float(pan_offset[1])
    return screen, rotated[:, 2].astype(np.float32, copy=False), rotated


def _compute_face_intensity(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float32)

    face_vertices = np.asarray(vertices[faces], dtype=np.float32)
    edges_a = face_vertices[:, 1] - face_vertices[:, 0]
    edges_b = face_vertices[:, 2] - face_vertices[:, 0]
    normals = np.cross(edges_a, edges_b)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 1e-9
    normals[valid] /= normal_lengths[valid][:, None]

    light_direction = np.asarray([0.35, 0.45, 0.82], dtype=np.float32)
    light_direction /= np.linalg.norm(light_direction)
    lambert = np.abs(normals @ light_direction)
    intensity = 0.22 + 0.78 * np.clip(lambert, 0.0, 1.0)
    intensity[~valid] = 0.35
    return intensity.astype(np.float32, copy=False)


def _hint_overlay_colors(category: str) -> tuple[QColor, QColor]:
    normalized = str(category or "external-hints")
    if normalized == "nonmanifold-edges":
        return QColor(209, 66, 44, 230), QColor(209, 66, 44, 90)
    if normalized == "boundary-loops":
        return QColor(33, 122, 196, 230), QColor(33, 122, 196, 70)
    if normalized == "external-overlap-hints":
        return QColor(138, 47, 156, 235), QColor(138, 47, 156, 80)
    return QColor(44, 148, 97, 230), QColor(44, 148, 97, 70)


def draw_hint_overlay(qimage: QImage, payload: mesh_heal.PreviewPayload, projection_state: dict) -> None:
    if not payload.hint_issues:
        return

    painter = QPainter(qimage)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for issue in payload.hint_issues:
        outline_color, fill_color = _hint_overlay_colors(str(issue.get("category") or "external-hints"))
        painter.setPen(QPen(outline_color, 2.0))

        triangle_points = issue.get("triangle_points") or []
        if len(triangle_points) >= 3:
            projected = _project_preview_overlay_points(triangle_points[:3], projection_state)
            polygon = QPolygonF([QPointF(float(point[0]), float(point[1])) for point in projected])
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            continue

        closed_path_points = issue.get("closed_path_points") or []
        if len(closed_path_points) >= 3:
            projected = _project_preview_overlay_points(closed_path_points, projection_state)
            polygon = QPolygonF([QPointF(float(point[0]), float(point[1])) for point in projected])
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            continue

        polyline_points = issue.get("polyline_points") or []
        if len(polyline_points) >= 2:
            projected = _project_preview_overlay_points(polyline_points, projection_state)
            polygon = QPolygonF([QPointF(float(point[0]), float(point[1])) for point in projected])
            painter.drawPolyline(polygon)

        point = issue.get("point")
        if point is not None:
            projected = _project_preview_overlay_points([point], projection_state)
            if len(projected) > 0:
                painter.setBrush(outline_color)
                painter.drawEllipse(QPointF(float(projected[0, 0]), float(projected[0, 1])), 4.0, 4.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.end()


if ti is not None:
    @ti.func
    def _taichi_edge_value(a, b, p):
        return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])

    @ti.kernel
    def taichi_clear_buffers(
        color: ti.types.ndarray(dtype=ti.f32, ndim=3),  # type: ignore[reportInvalidTypeForm]
        depth_buffer: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore[reportInvalidTypeForm]
    ):
        for y, x in depth_buffer:
            depth_buffer[y, x] = -1.0e30
            color[y, x, 0] = 0.965
            color[y, x, 1] = 0.953
            color[y, x, 2] = 0.925

    @ti.kernel
    def taichi_rasterize_triangles(
        screen: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore[reportInvalidTypeForm]
        depths: ti.types.ndarray(dtype=ti.f32, ndim=1),  # type: ignore[reportInvalidTypeForm]
        faces: ti.types.ndarray(dtype=ti.i32, ndim=2),  # type: ignore[reportInvalidTypeForm]
        face_intensity: ti.types.ndarray(dtype=ti.f32, ndim=1),  # type: ignore[reportInvalidTypeForm]
        color: ti.types.ndarray(dtype=ti.f32, ndim=3),  # type: ignore[reportInvalidTypeForm]
        depth_buffer: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore[reportInvalidTypeForm]
        width: ti.i32,  # type: ignore[reportInvalidTypeForm]
        height: ti.i32,  # type: ignore[reportInvalidTypeForm]
        face_count: ti.i32,  # type: ignore[reportInvalidTypeForm]
    ):
        for face_id in range(face_count):
            i0 = faces[face_id, 0]
            i1 = faces[face_id, 1]
            i2 = faces[face_id, 2]
            p0 = ti.Vector([screen[i0, 0], screen[i0, 1]])
            p1 = ti.Vector([screen[i1, 0], screen[i1, 1]])
            p2 = ti.Vector([screen[i2, 0], screen[i2, 1]])
            z0 = depths[i0]
            z1 = depths[i1]
            z2 = depths[i2]
            area = _taichi_edge_value(p0, p1, p2)
            if ti.abs(area) < 1.0e-6:
                continue

            min_px = ti.min(p0[0], p1[0])
            min_px = ti.min(min_px, p2[0])
            max_px = ti.max(p0[0], p1[0])
            max_px = ti.max(max_px, p2[0])
            min_py = ti.min(p0[1], p1[1])
            min_py = ti.min(min_py, p2[1])
            max_py = ti.max(p0[1], p1[1])
            max_py = ti.max(max_py, p2[1])

            min_x = ti.max(0, ti.cast(ti.floor(min_px), ti.i32))
            max_x = ti.min(width - 1, ti.cast(ti.ceil(max_px), ti.i32))
            min_y = ti.max(0, ti.cast(ti.floor(min_py), ti.i32))
            max_y = ti.min(height - 1, ti.cast(ti.ceil(max_py), ti.i32))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    sample = ti.Vector([ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5])
                    w0 = _taichi_edge_value(p1, p2, sample)
                    w1 = _taichi_edge_value(p2, p0, sample)
                    w2 = _taichi_edge_value(p0, p1, sample)
                    inside = (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0 and area > 0.0) or (w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0 and area < 0.0)
                    if inside:
                        bary0 = w0 / area
                        bary1 = w1 / area
                        bary2 = w2 / area
                        depth = bary0 * z0 + bary1 * z1 + bary2 * z2
                        if depth > depth_buffer[y, x]:
                            depth_buffer[y, x] = depth
                            intensity = face_intensity[face_id]
                            color[y, x, 0] = 0.788 * intensity
                            color[y, x, 1] = 0.604 * intensity
                            color[y, x, 2] = 0.322 * intensity
else:
    def taichi_clear_buffers(*_args, **_kwargs):
        raise RuntimeError("Taichi preview renderer is unavailable in this interpreter.")

    def taichi_rasterize_triangles(*_args, **_kwargs):
        raise RuntimeError("Taichi preview renderer is unavailable in this interpreter.")


def render_preview_payload_taichi(
    payload: mesh_heal.PreviewPayload,
    width: int,
    height: int,
    *,
    azimuth_degrees: float = 45.0,
    elevation_degrees: float = 35.26438968,
    zoom: float = 1.0,
    pan_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    error = ensure_taichi_initialized()
    if error is not None:
        raise RuntimeError(error)
    vertices = np.asarray(payload.vertices, dtype=np.float32)
    faces = np.asarray(payload.faces, dtype=np.int32)
    image_height = max(int(height), 320)
    image_width = max(int(width), 320)
    if len(vertices) == 0 or len(faces) == 0:
        return np.full((image_height, image_width, 3), fill_value=(246, 241, 234), dtype=np.uint8)

    screen, depths, rotated_vertices = _compute_preview_projection(
        vertices,
        image_width,
        image_height,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
        zoom=zoom,
        pan_offset=pan_offset,
    )
    face_intensity = _compute_face_intensity(rotated_vertices, faces)
    color = np.empty((image_height, image_width, 3), dtype=np.float32)
    depth_buffer = np.empty((image_height, image_width), dtype=np.float32)
    taichi_clear_buffers(color, depth_buffer)
    taichi_rasterize_triangles(
        np.ascontiguousarray(screen, dtype=np.float32),
        np.ascontiguousarray(depths, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(face_intensity, dtype=np.float32),
        color,
        depth_buffer,
        image_width,
        image_height,
        len(faces),
    )
    return np.ascontiguousarray(np.clip(color * 255.0, 0.0, 255.0).astype(np.uint8))


class TaichiPreviewWidget(QLabel):
    def __init__(self, parent: QWidget | None = None, min_height: int = 320):
        super().__init__(parent)
        self.current_payload = None
        self.azimuth_degrees = 45.0
        self.elevation_degrees = 35.26438968
        self.zoom = 1.0
        self.pan_offset = np.zeros(2, dtype=np.float32)
        self.last_drag_position = None
        self.drag_mode = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(min_height)
        self.setStyleSheet("background-color: #f6f1ea; color: #4a4336;")
        self.setText("Preview viewer is idle. Load a preview to render the whole mesh.")
        self.setToolTip("Left drag rotates. Right drag pans. Mouse wheel zooms. Double-click resets the view.")

    def set_preview_payload(self, payload: mesh_heal.PreviewPayload) -> None:
        self.current_payload = payload
        self.reset_view()
        self.render_current_payload()

    def clear_preview(self, message: str) -> None:
        self.current_payload = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def reset_view(self) -> None:
        self.azimuth_degrees = 45.0
        self.elevation_degrees = 35.26438968
        self.zoom = 1.0
        self.pan_offset = np.zeros(2, dtype=np.float32)

    def mousePressEvent(self, event):
        if self.current_payload is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_mode = "rotate"
            self.last_drag_position = event.position()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.drag_mode = "pan"
            self.last_drag_position = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.current_payload is None or self.last_drag_position is None or self.drag_mode is None:
            super().mouseMoveEvent(event)
            return
        current_position = event.position()
        delta = current_position - self.last_drag_position
        self.last_drag_position = current_position
        if self.drag_mode == "rotate":
            self.azimuth_degrees += float(delta.x()) * 0.5
            self.elevation_degrees = float(np.clip(self.elevation_degrees - float(delta.y()) * 0.5, -89.0, 89.0))
        elif self.drag_mode == "pan":
            self.pan_offset[0] += float(delta.x())
            self.pan_offset[1] += float(delta.y())
        self.render_current_payload()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.last_drag_position = None
            self.drag_mode = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self.current_payload is None:
            super().wheelEvent(event)
            return
        delta_steps = event.angleDelta().y() / 120.0
        if delta_steps != 0.0:
            self.zoom = float(np.clip(self.zoom * (1.15 ** delta_steps), 0.1, 20.0))
            self.render_current_payload()
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.current_payload is not None and event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            self.render_current_payload()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_payload is not None:
            QTimer.singleShot(0, self.render_current_payload)

    def render_current_payload(self) -> None:
        if self.current_payload is None:
            return
        error = ensure_taichi_initialized()
        if error is not None:
            self.clear_preview(error)
            return
        projection_state = _build_preview_projection_state(
            np.asarray(self.current_payload.vertices, dtype=np.float32),
            max(int(self.width()), 320),
            max(int(self.height()), 320),
            azimuth_degrees=self.azimuth_degrees,
            elevation_degrees=self.elevation_degrees,
            zoom=self.zoom,
            pan_offset=(float(self.pan_offset[0]), float(self.pan_offset[1])),
        )
        image = render_preview_payload_taichi(
            self.current_payload,
            self.width(),
            self.height(),
            azimuth_degrees=self.azimuth_degrees,
            elevation_degrees=self.elevation_degrees,
            zoom=self.zoom,
            pan_offset=(float(self.pan_offset[0]), float(self.pan_offset[1])),
        )
        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        draw_hint_overlay(qimage, self.current_payload, projection_state)
        self.setPixmap(QPixmap.fromImage(qimage))
        self.setText("")


def create_taichi_preview_or_placeholder(parent: QWidget, unavailable_message: str, min_height: int = 320):
    error = ensure_taichi_initialized()
    if error is not None:
        placeholder = QLabel(f"{unavailable_message}\n\n{error}")
        placeholder.setWordWrap(True)
        placeholder.setFrameShape(QFrame.StyledPanel)
        placeholder.setMinimumHeight(min_height)
        return None, placeholder, error
    widget = TaichiPreviewWidget(parent, min_height=min_height)
    return widget, widget, None


def is_qt_object_alive(obj) -> bool:
    if obj is None:
        return False
    if shiboken6 is not None:
        try:
            return bool(shiboken6.isValid(obj))
        except Exception:
            return False
    try:
        obj.parent()
    except RuntimeError:
        return False
    except Exception:
        return False
    return True


def run_task_in_subprocess(task, kwargs, message_queue):
    try:
        result = task(
            status_callback=lambda message: message_queue.put(("log", message)),
            progress_callback=lambda current, total: message_queue.put(("progress", current, total)),
            **kwargs,
        )
        message_queue.put(("completed", result))
    except Exception:
        message_queue.put(("failed", traceback.format_exc()))


class TaskWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    completed = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, task, kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs

    @Slot()
    def run(self):
        ctx = multiprocessing.get_context("spawn")
        message_queue = ctx.Queue()
        process = ctx.Process(target=run_task_in_subprocess, args=(self.task, self.kwargs, message_queue))
        process.start()

        try:
            while True:
                try:
                    message = message_queue.get(timeout=0.1)
                except Empty:
                    if not process.is_alive():
                        break
                    continue

                kind = message[0]
                if kind == "log":
                    self.log.emit(message[1])
                elif kind == "progress":
                    self.progress.emit(message[1], message[2])
                elif kind == "completed":
                    self.completed.emit(message[1])
                    break
                elif kind == "failed":
                    self.failed.emit(message[1])
                    break

            process.join()
            while True:
                try:
                    message = message_queue.get_nowait()
                except Empty:
                    break

                kind = message[0]
                if kind == "log":
                    self.log.emit(message[1])
                elif kind == "progress":
                    self.progress.emit(message[1], message[2])
                elif kind == "completed":
                    self.completed.emit(message[1])
                elif kind == "failed":
                    self.failed.emit(message[1])

            if process.exitcode not in (0, None):
                self.failed.emit(f"Task process exited with code {process.exitcode}.")
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            message_queue.close()
            self.finished.emit()


class PreviewWorker(QObject):
    completed = Signal(object)
    skipped = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        path: Path,
        hint_paths: Sequence[Path] | None = None,
        allow_decimation: bool = False,
        max_faces: int = 120000,
        max_vertices: int = 80000,
    ):
        super().__init__()
        self.path = path
        self.hint_paths = [Path(item) for item in (hint_paths or [])]
        self.allow_decimation = allow_decimation
        self.max_faces = max_faces
        self.max_vertices = max_vertices

    @Slot()
    def run(self):
        try:
            payload = mesh_heal.prepare_preview_payload(
                self.path,
                external_hint_path=self.hint_paths if self.hint_paths else None,
                max_faces=self.max_faces,
                max_vertices=self.max_vertices,
                allow_decimation=self.allow_decimation,
            )
            self.completed.emit(payload)
        except mesh_heal.PreviewSkippedError as exc:
            self.skipped.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class IssueAnalysisWorker(QObject):
    completed = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)

    def __init__(self, token: int, mesh, area_eps: float, dedup_decimals: int):
        super().__init__()
        self.token = token
        self.mesh = mesh.copy()
        self.area_eps = area_eps
        self.dedup_decimals = dedup_decimals

    @Slot()
    def run(self):
        try:
            payload = mesh_heal.collect_mesh_issues(
                self.mesh,
                area_eps=self.area_eps,
                dedup_decimals=self.dedup_decimals,
            )
            self.completed.emit(self.token, payload)
        except Exception:
            self.failed.emit(self.token, traceback.format_exc())
        finally:
            self.finished.emit(self.token)


class FilePicker(QWidget):
    path_changed = Signal(str)

    def __init__(
        self,
        label: str,
        button_text: str,
        file_filter: str,
        save_dialog: bool = False,
        multi_select: bool = False,
    ):
        super().__init__()
        self.file_filter = file_filter
        self.save_dialog = save_dialog
        self.multi_select = multi_select
        self.selected_paths: list[str] = []
        self.help_tooltip = ""

        self.label = QLabel(label)
        self.line_edit = QLineEdit()
        if self.multi_select:
            self.line_edit.setReadOnly(True)
        self.browse_button = QPushButton(button_text)
        self.browse_button.clicked.connect(self.choose_path)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def _compose_line_edit_tooltip(self, value: str = "") -> str:
        value = value.strip()
        if not self.help_tooltip:
            return value
        if not value:
            return self.help_tooltip
        return f"{self.help_tooltip}\n\nCurrent value:\n{value}"

    def _refresh_line_edit_tooltip(self) -> None:
        tooltip_value = ""
        if self.multi_select and self.selected_paths:
            tooltip_value = "\n".join(self.selected_paths)
        elif not self.multi_select:
            tooltip_value = self.line_edit.text()
        self.line_edit.setToolTip(self._compose_line_edit_tooltip(tooltip_value))

    def set_help_tooltip(self, text: str) -> None:
        self.help_tooltip = text.strip()
        self.setToolTip(self.help_tooltip)
        self.label.setToolTip(self.help_tooltip)
        self.browse_button.setToolTip(self.help_tooltip)
        self._refresh_line_edit_tooltip()

    def choose_path(self):
        if self.save_dialog:
            path, _ = QFileDialog.getSaveFileName(self, self.label.text(), self.line_edit.text(), self.file_filter)
            if path:
                self.selected_paths = []
                self.line_edit.setText(path)
                self._refresh_line_edit_tooltip()
                self.path_changed.emit(path)
            return

        if self.multi_select:
            paths, _ = QFileDialog.getOpenFileNames(self, self.label.text(), "", self.file_filter)
            if paths:
                self.set_paths(paths)
        else:
            path, _ = QFileDialog.getOpenFileName(self, self.label.text(), self.line_edit.text(), self.file_filter)
            if path:
                self.selected_paths = []
                self.line_edit.setText(path)
                self._refresh_line_edit_tooltip()
                self.path_changed.emit(path)

    def set_paths(self, paths: list[str]) -> None:
        self.selected_paths = [path for path in paths if path]
        if not self.selected_paths:
            self.line_edit.clear()
            self._refresh_line_edit_tooltip()
            return
        if len(self.selected_paths) == 1:
            display_text = self.selected_paths[0]
        else:
            names = ", ".join(Path(path).name for path in self.selected_paths)
            display_text = f"{len(self.selected_paths)} files selected: {names}"
        self.line_edit.setText(display_text)
        self._refresh_line_edit_tooltip()

    def paths(self) -> list[str]:
        if self.multi_select:
            return list(self.selected_paths)
        value = self.text()
        return [value] if value else []

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, value: str) -> None:
        self.selected_paths = []
        self.line_edit.setText(value)
        self._refresh_line_edit_tooltip()

    def setEnabled(self, enabled: bool) -> None:
        self.label.setEnabled(enabled)
        self.line_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)


class FileListPicker(QWidget):
    def __init__(self, label: str, button_text: str, file_filter: str):
        super().__init__()
        self.file_filter = file_filter
        self.last_directory = ""

        self.label = QLabel(label)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setMinimumHeight(140)

        self.add_button = QPushButton(button_text)
        self.add_button.clicked.connect(self.add_files)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(self.move_up)
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(self.move_down)

        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.add_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.move_up_button)
        controls.addWidget(self.move_down_button)
        controls.addStretch(1)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.list_widget, 1)
        body.addLayout(controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addLayout(body)

    def add_files(self):
        start_dir = self.last_directory
        current = self.current_path()
        if current is not None:
            start_dir = str(Path(current).parent)
        paths, _ = QFileDialog.getOpenFileNames(self, self.label.text(), start_dir, self.file_filter)
        if not paths:
            return
        existing = set(self.paths())
        for path in paths:
            if path in existing:
                continue
            self.list_widget.addItem(path)
            existing.add(path)
        self.last_directory = str(Path(paths[-1]).parent)

    def paths(self) -> list[str]:
        return [self.list_widget.item(index).text() for index in range(self.list_widget.count())]

    def current_path(self) -> str | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.text()

    def remove_selected(self):
        for item in list(self.list_widget.selectedItems()):
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

    def clear(self):
        self.list_widget.clear()

    def move_up(self):
        rows = sorted({self.list_widget.row(item) for item in self.list_widget.selectedItems()})
        for row in rows:
            if row <= 0:
                continue
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            item.setSelected(True)

    def move_down(self):
        rows = sorted({self.list_widget.row(item) for item in self.list_widget.selectedItems()}, reverse=True)
        for row in rows:
            if row >= self.list_widget.count() - 1:
                continue
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            item.setSelected(True)

    def setEnabled(self, enabled: bool) -> None:
        self.label.setEnabled(enabled)
        self.list_widget.setEnabled(enabled)
        self.add_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.move_up_button.setEnabled(enabled)
        self.move_down_button.setEnabled(enabled)


class PreviewPane(QGroupBox):
    def __init__(self):
        super().__init__("Preview")
        self.thread = None
        self.worker = None
        self.pending_preview_label = "Preview"
        self.plotter_init_error = None
        self.plotter = None
        self.viewer_widget = None

        layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Preview is manual and uses Taichi to render the full mesh. Large previews can take longer and use more memory."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.preview_status = QLabel("No preview loaded")
        layout.addWidget(self.preview_status)

        self.preview_progress = QProgressBar()
        self.preview_progress.setRange(0, 1)
        self.preview_progress.setValue(0)
        layout.addWidget(self.preview_progress)

        self.viewer_layout = QVBoxLayout()
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_placeholder = create_viewer_placeholder(
            "Preview viewer is idle. Load a preview to initialize the embedded renderer.",
            min_height=320,
        )
        self.viewer_widget = self.viewer_placeholder
        self.viewer_layout.addWidget(self.viewer_widget)
        layout.addLayout(self.viewer_layout, 1)

        self.preview_log = QPlainTextEdit()
        self.preview_log.setReadOnly(True)
        self.preview_log.setPlaceholderText("Preview log")
        self.preview_log.setMaximumBlockCount(200)
        layout.addWidget(self.preview_log)

    def available(self) -> bool:
        return self.plotter is not None

    def set_viewer_widget(self, widget: QWidget) -> None:
        if self.viewer_widget is not None:
            self.viewer_layout.removeWidget(self.viewer_widget)
            self.viewer_widget.setParent(None)
        self.viewer_widget = widget
        self.viewer_layout.addWidget(widget)

    def ensure_plotter(self) -> bool:
        if self.plotter is not None:
            return True

        plotter, viewer_widget, error = create_taichi_preview_or_placeholder(
            self,
            "Embedded Taichi preview is unavailable. The rest of the GUI still works without it.",
            min_height=320,
        )
        self.plotter = plotter
        self.plotter_init_error = error
        self.viewer_placeholder = viewer_widget if plotter is None else None
        self.set_viewer_widget(viewer_widget)
        if plotter is None:
            self.preview_status.setText("Preview unavailable")
            if error:
                self.append_log(f"Preview renderer unavailable: {error}")
            return False
        return True

    def append_log(self, message: str):
        self.preview_log.appendPlainText(message)

    def request_preview(
        self,
        path_text: str,
        label: str,
        hint_path_texts: Sequence[str] | None = None,
        allow_decimation: bool = False,
        max_faces: int = 120000,
        max_vertices: int = 80000,
    ):
        if not path_text:
            QMessageBox.warning(self, "Missing path", f"Select a file before previewing {label.lower()}.")
            return
        if not self.available() and not self.ensure_plotter():
            QMessageBox.information(
                self,
                "Preview unavailable",
                self.plotter_init_error or "Embedded preview needs Taichi in a Python 3.13 environment. The rest of the GUI still works without it.",
            )
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Preview busy", "Wait for the current preview to finish loading.")
            return

        path = Path(path_text)
        self.pending_preview_label = label
        self.preview_status.setText(f"Loading {label}: {path.name}")
        self.append_log(f"Loading preview for {path}")
        self.preview_progress.setRange(0, 0)

        self.thread = QThread(self)
        self.worker = PreviewWorker(
            path,
            hint_paths=[Path(value) for value in (hint_path_texts or []) if str(value).strip()],
            allow_decimation=allow_decimation,
            max_faces=max_faces,
            max_vertices=max_vertices,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self.on_preview_completed, Qt.ConnectionType.QueuedConnection)
        self.worker.skipped.connect(self.show_preview_skipped)
        self.worker.failed.connect(self.show_preview_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.reset_progress)
        self.thread.start()

    def reset_progress(self):
        self.preview_progress.setRange(0, 1)
        self.preview_progress.setValue(1)
        self.thread = None
        self.worker = None

    @Slot(object)
    def on_preview_completed(self, payload: mesh_heal.PreviewPayload):
        self.show_preview(payload, self.pending_preview_label)

    def show_preview(self, payload: mesh_heal.PreviewPayload, label: str):
        self.plotter.set_preview_payload(payload)
        note = "decimated" if payload.decimated else "full-resolution"
        hint_note = f" | {len(payload.hint_issues):,} hint overlay(s)" if payload.hint_issues else ""
        self.preview_status.setText(
            f"{label}: {Path(payload.source).name} | original {payload.original_faces:,} faces -> preview "
            f"{payload.preview_faces:,} faces ({note}){hint_note}"
        )
        self.append_log(self.preview_status.text())

    def show_preview_error(self, details: str):
        self.preview_status.setText("Preview failed")
        self.append_log(details)
        QMessageBox.warning(self, "Preview failed", details)

    def show_preview_skipped(self, message: str):
        self.preview_status.setText("Preview skipped")
        self.append_log(message)


class BaseOperationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.controls = []
        self.progress_received = False

    def create_status_widgets(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Task log")

        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlaceholderText("Run summary")

    def set_running(self, running: bool):
        alive_controls = []
        for control in self.controls:
            if not is_qt_object_alive(control):
                continue
            control.setEnabled(not running)
            alive_controls.append(control)
        self.controls = alive_controls
        if running:
            self.progress_received = False
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)

    def append_log(self, message: str):
        self.log_view.appendPlainText(message)

    def show_summary(self, result: dict):
        self.summary_view.setPlainText(json.dumps(result, indent=2))

    def show_error(self, details: str):
        self.append_log(details)
        QMessageBox.critical(self, "Operation failed", details)

    def update_progress(self, current: int, total: int):
        if total <= 0:
            return
        if not self.progress_received:
            self.progress_received = True
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(max(0, min(current, total)))

    def start_worker(self, task, kwargs):
        self.log_view.clear()
        self.summary_view.clear()
        self.set_running(True)

        self.thread = QThread(self)
        self.worker = TaskWorker(task, kwargs)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.show_summary)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.set_running(False))
        self.thread.start()


class HealTab(BaseOperationTab):
    def __init__(self, preview_pane: PreviewPane | None):
        super().__init__()
        self.preview_pane = preview_pane
        self.last_suggested_output = ""
        self.healed_output_ready = False

        self.input_picker = FilePicker("Input", "Browse...", MESH_FILTER)
        self.output_picker = FilePicker("Output", "Save...", OUTPUT_FILTER, save_dialog=True)
        self.report_picker = FilePicker("Report", "Save...", JSON_FILTER, save_dialog=True)
        self.hints_picker = FilePicker("Hints", "Browse...", HINT_FILTER, multi_select=True)
        self.intended_mesh_type_combo = create_intended_mesh_type_combo("auto")
        self.input_picker.line_edit.textChanged.connect(self.update_default_output_path)
        self.input_picker.line_edit.textChanged.connect(self.refresh_preview_buttons)
        self.output_picker.line_edit.textChanged.connect(self.invalidate_output_preview)

        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)
        self.rebuild_triangles_check = QCheckBox("Start from problematic areas and rebuild triangles first")
        self.rebuild_triangles_check.setChecked(True)
        self.nonmanifold_edge_repair_check = QCheckBox("Experimental cylindrical non-manifold edge repair")
        self.nonmanifold_edge_radius_edit = QLineEdit("0.0")
        self.nonmanifold_edge_radius_edit.setPlaceholderText("0 = auto")
        self.localized_intersection_repair_check = QCheckBox("Experimental localized self-intersection repair")
        self.point_cloud_rebuild_combo = QComboBox()
        self.point_cloud_rebuild_combo.addItem("None", "none")
        self.point_cloud_rebuild_combo.addItem("Triangle centers + normals (Poisson)", "triangle-centers-poisson")
        self.distance_model_combo = QComboBox()
        self.distance_model_combo.addItem("None", "none")
        self.distance_model_combo.addItem("Distance hull", "distance-hull")
        self.distance_model_combo.addItem("Surface shell", "surface-shell")
        self.distance_model_combo.currentIndexChanged.connect(self.update_distance_model_controls)
        self.distance_offset_edit = QLineEdit("1.0")
        self.distance_grid_spacing_edit = QLineEdit("0.0")
        self.distance_grid_spacing_edit.setPlaceholderText("0 = auto")
        self.make_watertight_check = QCheckBox("Attempt to fill holes and make watertight")
        self.return_surface_after_watertight_check = QCheckBox("Return repaired surface after hole filling")
        self.advanced_backend_combo = QComboBox()
        self.advanced_backend_combo.addItem("None", "none")
        self.advanced_backend_combo.addItem("CGAL Alpha Wrap", "cgal-alpha-wrap")
        self.advanced_backend_combo.currentIndexChanged.connect(self.update_advanced_backend_controls)
        self.cgal_backend_exe_edit = QLineEdit()
        self.cgal_backend_exe_edit.setPlaceholderText("Optional CGAL backend executable")
        self.cgal_alpha_edit = QLineEdit("0.0")
        self.cgal_alpha_edit.setPlaceholderText("0 = auto")
        self.cgal_offset_edit = QLineEdit("0.0")
        self.cgal_offset_edit.setPlaceholderText("0 = auto")
        self.cgal_alpha_relative_edit = QLineEdit("0.02")
        self.cgal_offset_relative_edit = QLineEdit("0.03333333333333333")
        self.run_button = QPushButton("Run Heal")
        self.run_button.clicked.connect(self.run_task)
        self.preview_input_button = None
        self.preview_output_button = None
        preview_buttons = None
        if self.preview_pane is not None:
            self.preview_input_button = QPushButton("Preview Input")
            self.preview_output_button = QPushButton("Preview Output")
            self.preview_input_button.clicked.connect(
                lambda: self.preview_pane.request_preview(
                    self.input_picker.text(),
                    "Heal input",
                    hint_path_texts=self.hints_picker.paths(),
                    allow_decimation=False,
                    max_faces=600000,
                    max_vertices=1200000,
                )
            )
            self.preview_output_button.clicked.connect(
                lambda: self.preview_pane.request_preview(
                    self.output_picker.text(),
                    "Heal output",
                    hint_path_texts=self.hints_picker.paths(),
                    allow_decimation=False,
                    max_faces=600000,
                    max_vertices=1200000,
                )
            )
            preview_buttons = QWidget()
            preview_buttons_layout = QHBoxLayout(preview_buttons)
            preview_buttons_layout.setContentsMargins(0, 0, 0, 0)
            preview_buttons_layout.addWidget(self.preview_input_button)
            preview_buttons_layout.addWidget(self.preview_output_button)

        workflow_note = QLabel(
            "Workflow: load a mesh, declare whether it is meant to stay open or become closed, optionally load external problem hints, then rebuild locally from the damaged regions before the global cleanup. For solids, the pipeline automatically closes residual openings after repair."
        )
        workflow_note.setWordWrap(True)

        options_group = QGroupBox("Healing")
        form = QFormLayout(options_group)
        form.addRow(workflow_note)
        form.addRow(self.input_picker)
        form.addRow(self.output_picker)
        form.addRow(self.report_picker)
        form.addRow(self.hints_picker)
        form.addRow("Intended result", self.intended_mesh_type_combo)
        form.addRow("Merge epsilon", self.merge_eps_edit)
        form.addRow("Area epsilon", self.area_eps_edit)
        form.addRow("Dedup decimals", self.dedup_spin)
        form.addRow(self.rebuild_triangles_check)
        form.addRow(self.nonmanifold_edge_repair_check)
        form.addRow("Non-manifold edge radius", self.nonmanifold_edge_radius_edit)
        form.addRow(self.localized_intersection_repair_check)
        form.addRow("Point-cloud rebuild", self.point_cloud_rebuild_combo)
        form.addRow("Distance model", self.distance_model_combo)
        form.addRow("Distance offset", self.distance_offset_edit)
        form.addRow("Distance grid spacing", self.distance_grid_spacing_edit)
        form.addRow(self.make_watertight_check)
        form.addRow(self.return_surface_after_watertight_check)
        form.addRow("Advanced backend", self.advanced_backend_combo)
        form.addRow("CGAL backend exe", self.cgal_backend_exe_edit)
        form.addRow("CGAL alpha", self.cgal_alpha_edit)
        form.addRow("CGAL offset", self.cgal_offset_edit)
        form.addRow("CGAL alpha relative", self.cgal_alpha_relative_edit)
        form.addRow("CGAL offset relative", self.cgal_offset_relative_edit)
        if preview_buttons is not None:
            form.addRow("Preview", preview_buttons)
        form.addRow(self.run_button)

        self.create_status_widgets()

        layout = QVBoxLayout(self)
        layout.addWidget(options_group)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.progress_bar)

        panels = QGridLayout()
        panels.addWidget(QLabel("Log"), 0, 0)
        panels.addWidget(QLabel("Summary"), 0, 1)
        panels.addWidget(self.log_view, 1, 0)
        panels.addWidget(self.summary_view, 1, 1)
        layout.addLayout(panels)

        self.controls = [
            self.input_picker,
            self.output_picker,
            self.report_picker,
            self.hints_picker,
            self.intended_mesh_type_combo,
            self.merge_eps_edit,
            self.area_eps_edit,
            self.dedup_spin,
            self.rebuild_triangles_check,
            self.nonmanifold_edge_repair_check,
            self.nonmanifold_edge_radius_edit,
            self.localized_intersection_repair_check,
            self.point_cloud_rebuild_combo,
            self.distance_model_combo,
            self.distance_offset_edit,
            self.distance_grid_spacing_edit,
            self.make_watertight_check,
            self.return_surface_after_watertight_check,
            self.advanced_backend_combo,
            self.cgal_backend_exe_edit,
            self.cgal_alpha_edit,
            self.cgal_offset_edit,
            self.cgal_alpha_relative_edit,
            self.cgal_offset_relative_edit,
            self.run_button,
        ]
        if self.preview_input_button is not None:
            self.controls.append(self.preview_input_button)
        if self.preview_output_button is not None:
            self.controls.append(self.preview_output_button)
        self.update_distance_model_controls()
        self.update_advanced_backend_controls()
        self.refresh_preview_buttons()

    def update_default_output_path(self, input_path_text: str):
        if not input_path_text:
            self.last_suggested_output = ""
            return
        suggested_output = str(mesh_heal.derive_healed_output_path(Path(input_path_text)))
        current_output = self.output_picker.text()
        if current_output and current_output != self.last_suggested_output:
            return
        self.output_picker.setText(suggested_output)
        self.last_suggested_output = suggested_output

    def invalidate_output_preview(self, _path_text: str):
        self.healed_output_ready = False
        self.refresh_preview_buttons()

    def refresh_preview_buttons(self, *_args):
        if self.preview_pane is None:
            return
        input_path = self.input_picker.text()
        input_ready = bool(input_path) and Path(input_path).exists()
        output_ready = self.healed_output_ready and bool(self.output_picker.text())
        if is_qt_object_alive(self.preview_input_button):
            self.preview_input_button.setEnabled(input_ready)
        if is_qt_object_alive(self.preview_output_button):
            self.preview_output_button.setEnabled(output_ready)

    def update_distance_model_controls(self, *_args):
        distance_mode = str(self.distance_model_combo.currentData())
        uses_distance_model = distance_mode != "none"
        uses_distance_hull = distance_mode == "distance-hull"
        self.distance_offset_edit.setEnabled(uses_distance_model)
        self.distance_grid_spacing_edit.setEnabled(uses_distance_hull)
        self.distance_offset_edit.setToolTip(
            "Per-side offset used by Distance hull and Surface shell. Surface shell offsets along mesh normals on both sides."
        )
        self.distance_grid_spacing_edit.setToolTip(
            "Only used for Distance hull. Surface shell offsets the existing mesh directly and ignores grid spacing."
        )

    def update_advanced_backend_controls(self, *_args):
        backend_name = str(self.advanced_backend_combo.currentData())
        uses_cgal = backend_name == "cgal-alpha-wrap"
        uses_cgal_alpha_wrap = backend_name == "cgal-alpha-wrap"
        self.cgal_backend_exe_edit.setEnabled(uses_cgal)
        self.cgal_alpha_edit.setEnabled(uses_cgal_alpha_wrap)
        self.cgal_offset_edit.setEnabled(uses_cgal_alpha_wrap)
        self.cgal_alpha_relative_edit.setEnabled(uses_cgal_alpha_wrap)
        self.cgal_offset_relative_edit.setEnabled(uses_cgal_alpha_wrap)
        alpha_wrap_tooltip = "CGAL Alpha Wrap can be tuned here. Absolute alpha and offset override the relative values when set above 0."
        inactive_tooltip = "These parameters are only used when Advanced backend is set to CGAL Alpha Wrap."
        tooltip = alpha_wrap_tooltip if uses_cgal_alpha_wrap else inactive_tooltip
        for widget in (
            self.cgal_alpha_edit,
            self.cgal_offset_edit,
            self.cgal_alpha_relative_edit,
            self.cgal_offset_relative_edit,
        ):
            widget.setToolTip(tooltip)

        self.hints_picker.set_help_tooltip(
            "Optional JSON or DXF hint file set with external repair hints, such as Leapfrog-exported non-manifold polylines or overlap shapes. Automatic heal merges all selected hint files and uses them to force the relevant repair passes even when the internal detector stays quiet."
        )
        self.intended_mesh_type_combo.setToolTip(
            "Declare whether the mesh is expected to end as a closed solid or an open surface. Solid intent automatically favors closed-output repair; surface intent avoids treating open boundaries as automatic hole-fill damage."
        )

    def set_running(self, running: bool):
        super().set_running(running)
        if not running:
            self.refresh_preview_buttons()

    def show_summary(self, result: dict):
        super().show_summary(result)
        output_path = result.get("output")
        self.healed_output_ready = bool(output_path) and Path(output_path).exists()
        self.refresh_preview_buttons()

    def run_task(self):
        input_path = self.input_picker.text()
        output_path = self.output_picker.text()
        if not input_path or not output_path:
            QMessageBox.warning(self, "Missing path", "Input and output are required.")
            return

        try:
            hint_paths = [Path(path_text) for path_text in self.hints_picker.paths() if path_text.strip()]
            kwargs = {
                "input_path": Path(input_path),
                "output_path": Path(output_path),
                "report_path": Path(self.report_picker.text()) if self.report_picker.text() else None,
                "external_hint_path": hint_paths if hint_paths else None,
                "intended_mesh_type": str(self.intended_mesh_type_combo.currentData()),
                "merge_eps": float(self.merge_eps_edit.text()),
                "area_eps": float(self.area_eps_edit.text()),
                "dedup_decimals": int(self.dedup_spin.value()),
                "rebuild_triangles": bool(self.rebuild_triangles_check.isChecked()),
                "nonmanifold_edge_repair": bool(self.nonmanifold_edge_repair_check.isChecked()),
                "nonmanifold_edge_radius": float(self.nonmanifold_edge_radius_edit.text()),
                "localized_intersection_repair": bool(self.localized_intersection_repair_check.isChecked()),
                "point_cloud_rebuild": str(self.point_cloud_rebuild_combo.currentData()),
                "distance_model": str(self.distance_model_combo.currentData()),
                "distance_offset": float(self.distance_offset_edit.text()),
                "distance_grid_spacing": float(self.distance_grid_spacing_edit.text()),
                "make_watertight": bool(self.make_watertight_check.isChecked()),
                "return_surface_after_watertight": bool(self.return_surface_after_watertight_check.isChecked()),
                "advanced_backend": str(self.advanced_backend_combo.currentData()),
                "cgal_backend_executable": Path(self.cgal_backend_exe_edit.text()) if self.cgal_backend_exe_edit.text().strip() else None,
                "cgal_alpha": float(self.cgal_alpha_edit.text()),
                "cgal_offset": float(self.cgal_offset_edit.text()),
                "cgal_alpha_relative": float(self.cgal_alpha_relative_edit.text()),
                "cgal_offset_relative": float(self.cgal_offset_relative_edit.text()),
            }
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid numeric input",
                "Merge epsilon, area epsilon, non-manifold edge radius, distance offset, distance grid spacing, and CGAL alpha-wrap parameters must be valid numbers.",
            )
            return

        self.healed_output_ready = False
        self.start_worker(mesh_heal.run_heal_pipeline, kwargs)


class SurfaceShellBatchTab(BaseOperationTab):
    def __init__(self):
        super().__init__()

        self.selected_meshes_label = QLabel("Selected meshes")
        self.items_tree = QTreeWidget()
        self.items_tree.setColumnCount(3)
        self.items_tree.setHeaderLabels(["Input mesh", "Offset", "Output file"])
        self.items_tree.setRootIsDecorated(False)
        self.items_tree.setAlternatingRowColors(True)
        self.items_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.items_tree.setMinimumHeight(220)
        header = self.items_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.add_files_button = QPushButton("Add Files...")
        self.add_files_button.clicked.connect(self.add_files)
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_items)

        list_buttons = QWidget()
        list_buttons_layout = QVBoxLayout(list_buttons)
        list_buttons_layout.setContentsMargins(0, 0, 0, 0)
        list_buttons_layout.addWidget(self.add_files_button)
        list_buttons_layout.addWidget(self.remove_selected_button)
        list_buttons_layout.addWidget(self.clear_button)
        list_buttons_layout.addStretch(1)

        list_body = QWidget()
        list_body_layout = QHBoxLayout(list_body)
        list_body_layout.setContentsMargins(0, 0, 0, 0)
        list_body_layout.addWidget(self.items_tree, 1)
        list_body_layout.addWidget(list_buttons)

        self.default_offset_edit = QLineEdit("1.0")
        self.apply_offset_edit = QLineEdit("1.0")
        self.apply_offset_button = QPushButton("Apply To Selected")
        self.apply_offset_button.clicked.connect(self.apply_selected_offset)

        apply_offset_widget = QWidget()
        apply_offset_layout = QHBoxLayout(apply_offset_widget)
        apply_offset_layout.setContentsMargins(0, 0, 0, 0)
        apply_offset_layout.addWidget(self.apply_offset_edit)
        apply_offset_layout.addWidget(self.apply_offset_button)

        self.output_directory_edit = QLineEdit()
        self.output_directory_edit.setPlaceholderText("Optional. Leave empty to write next to each input mesh.")
        self.output_directory_edit.textChanged.connect(self.refresh_output_paths)
        self.output_directory_button = QPushButton("Browse...")
        self.output_directory_button.clicked.connect(self.choose_output_directory)

        output_directory_widget = QWidget()
        output_directory_layout = QHBoxLayout(output_directory_widget)
        output_directory_layout.setContentsMargins(0, 0, 0, 0)
        output_directory_layout.addWidget(self.output_directory_edit, 1)
        output_directory_layout.addWidget(self.output_directory_button)

        self.intended_mesh_type_combo = create_intended_mesh_type_combo("surface")
        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)
        self.run_button = QPushButton("Run Batch Surface Shell")
        self.run_button.clicked.connect(self.run_task)

        note = QLabel(
            "Add multiple mesh files, assign each row its own per-side offset, and the batch runner will generate DXF outputs named with the pattern name_buffer_offset.dxf. The output directory is optional; when blank, each file is written beside its source mesh."
        )
        note.setWordWrap(True)

        options_group = QGroupBox("Batch Surface Shell")
        form = QFormLayout(options_group)
        form.addRow(note)
        form.addRow(self.selected_meshes_label)
        form.addRow(list_body)
        form.addRow("Default offset for new rows", self.default_offset_edit)
        form.addRow("Offset for selected rows", apply_offset_widget)
        form.addRow("Output directory", output_directory_widget)
        form.addRow("Intended result", self.intended_mesh_type_combo)
        form.addRow("Merge epsilon", self.merge_eps_edit)
        form.addRow("Area epsilon", self.area_eps_edit)
        form.addRow("Dedup decimals", self.dedup_spin)
        form.addRow(self.run_button)

        self.create_status_widgets()

        layout = QVBoxLayout(self)
        layout.addWidget(options_group)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.progress_bar)

        panels = QGridLayout()
        panels.addWidget(QLabel("Log"), 0, 0)
        panels.addWidget(QLabel("Summary"), 0, 1)
        panels.addWidget(self.log_view, 1, 0)
        panels.addWidget(self.summary_view, 1, 1)
        layout.addLayout(panels)

        self.controls = [
            self.selected_meshes_label,
            self.items_tree,
            self.add_files_button,
            self.remove_selected_button,
            self.clear_button,
            self.default_offset_edit,
            self.apply_offset_edit,
            self.apply_offset_button,
            self.output_directory_edit,
            self.output_directory_button,
            self.intended_mesh_type_combo,
            self.merge_eps_edit,
            self.area_eps_edit,
            self.dedup_spin,
            self.run_button,
        ]

    def choose_output_directory(self):
        start_dir = self.output_directory_edit.text().strip()
        if not start_dir and self.items_tree.topLevelItemCount() > 0:
            start_dir = str(Path(self.items_tree.topLevelItem(0).text(0)).parent)
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", start_dir)
        if directory:
            self.output_directory_edit.setText(directory)

    def add_files(self):
        start_dir = self.output_directory_edit.text().strip()
        paths, _ = QFileDialog.getOpenFileNames(self, "Select mesh files", start_dir, MESH_FILTER)
        if not paths:
            return

        existing_paths = {self.items_tree.topLevelItem(index).text(0) for index in range(self.items_tree.topLevelItemCount())}
        try:
            default_offset = mesh_heal.format_surface_shell_offset_token(float(self.default_offset_edit.text()))
        except ValueError:
            QMessageBox.warning(self, "Invalid offset", "Default offset must be a number greater than zero.")
            return

        for path in paths:
            if path in existing_paths:
                continue
            item = QTreeWidgetItem([path, default_offset, ""])
            self.items_tree.addTopLevelItem(item)
            existing_paths.add(path)

        self.refresh_output_paths()

    def remove_selected(self):
        for item in list(self.items_tree.selectedItems()):
            row = self.items_tree.indexOfTopLevelItem(item)
            if row >= 0:
                self.items_tree.takeTopLevelItem(row)

    def clear_items(self):
        self.items_tree.clear()

    def apply_selected_offset(self):
        selected_items = self.items_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Missing selection", "Select at least one row before applying an offset.")
            return

        try:
            offset_value = mesh_heal.format_surface_shell_offset_token(float(self.apply_offset_edit.text()))
        except ValueError:
            QMessageBox.warning(self, "Invalid offset", "Offset must be a number greater than zero.")
            return

        for item in selected_items:
            item.setText(1, offset_value)
        self.refresh_output_paths()

    def refresh_output_paths(self, *_args):
        output_directory_text = self.output_directory_edit.text().strip()
        output_directory = Path(output_directory_text) if output_directory_text else None
        for index in range(self.items_tree.topLevelItemCount()):
            item = self.items_tree.topLevelItem(index)
            input_path = Path(item.text(0))
            try:
                offset_value = float(item.text(1))
                output_path = mesh_heal.derive_surface_shell_output_path(
                    input_path,
                    offset_value,
                    output_directory=output_directory,
                )
            except ValueError:
                output_path = Path("Invalid offset")
            item.setText(2, str(output_path))

    def run_task(self):
        if self.items_tree.topLevelItemCount() == 0:
            QMessageBox.warning(self, "Missing input", "Add at least one mesh file to the batch list.")
            return

        try:
            items = []
            for index in range(self.items_tree.topLevelItemCount()):
                item = self.items_tree.topLevelItem(index)
                input_path = Path(item.text(0))
                offset_value = float(item.text(1))
                output_path = Path(item.text(2))
                if offset_value <= 0.0:
                    raise ValueError
                items.append(
                    {
                        "input_path": input_path,
                        "distance_offset": offset_value,
                        "output_path": output_path,
                    }
                )

            kwargs = {
                "items": items,
                "output_directory": Path(self.output_directory_edit.text()) if self.output_directory_edit.text().strip() else None,
                "intended_mesh_type": str(self.intended_mesh_type_combo.currentData()),
                "merge_eps": float(self.merge_eps_edit.text()),
                "area_eps": float(self.area_eps_edit.text()),
                "dedup_decimals": int(self.dedup_spin.value()),
            }
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid numeric input",
                "Every batch offset must be greater than zero, and merge epsilon plus area epsilon must be valid numbers.",
            )
            return

        self.start_worker(mesh_heal.run_surface_shell_batch_pipeline, kwargs)


class AutoresearchTab(BaseOperationTab):
    def __init__(self, preview_pane: PreviewPane | None):
        super().__init__()
        self.preview_pane = preview_pane
        self.last_suggested_output = ""
        self.last_suggested_report = ""
        self.output_ready = False

        self.input_picker = FilePicker("Input", "Browse...", MESH_FILTER)
        self.output_picker = FilePicker("Output", "Save...", OUTPUT_FILTER, save_dialog=True)
        self.report_picker = FilePicker("Report", "Save...", JSON_FILTER, save_dialog=True)
        self.hints_picker = FilePicker("Hints", "Browse...", HINT_FILTER)
        self.intended_mesh_type_combo = create_intended_mesh_type_combo("auto")
        self.input_picker.line_edit.textChanged.connect(self.update_default_output_path)
        self.input_picker.line_edit.textChanged.connect(self.refresh_preview_buttons)
        self.output_picker.line_edit.textChanged.connect(self.update_default_report_path)
        self.output_picker.line_edit.textChanged.connect(self.invalidate_output_preview)
        self.report_picker.line_edit.textChanged.connect(self.update_ledger_path_display)

        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)
        self.nonmanifold_edge_radius_edit = QLineEdit("0.0")
        self.nonmanifold_edge_radius_edit.setPlaceholderText("0 = auto")
        self.allow_aggressive_modes_check = QCheckBox("Allow aggressive fallback modes")
        self.fast_leapfrog_check = QCheckBox("Fast Leapfrog mode")
        self.fast_leapfrog_check.setChecked(True)

        self.max_candidates_spin = QSpinBox()
        self.max_candidates_spin.setRange(0, 999)
        self.max_candidates_spin.setValue(0)
        self.max_candidates_spin.setSpecialValueText("0 = auto/full set")

        self.time_budget_seconds_edit = QLineEdit("0.0")
        self.time_budget_seconds_edit.setPlaceholderText("0 = no soft cap / fast default")
        self.candidate_timeout_seconds_edit = QLineEdit("0.0")
        self.candidate_timeout_seconds_edit.setPlaceholderText("0 = disabled / fast default")
        self.self_intersection_timeout_seconds_edit = QLineEdit("0.0")
        self.self_intersection_timeout_seconds_edit.setPlaceholderText("0 = built-in validation timeout")

        self.fidelity_samples_spin = QSpinBox()
        self.fidelity_samples_spin.setRange(0, 500000)
        self.fidelity_samples_spin.setSingleStep(256)
        self.fidelity_samples_spin.setValue(0)
        self.fidelity_samples_spin.setSpecialValueText("0 = auto")

        self.max_mean_distance_normalized_edit = QLineEdit("0.02")
        self.max_p95_distance_normalized_edit = QLineEdit("0.05")
        self.max_component_count_delta_edit = QLineEdit("5.0")
        self.max_volume_ratio_delta_edit = QLineEdit("0.25")

        self.ledger_path_label = QLabel("No TSV ledger until a JSON report path is set.")
        self.ledger_path_label.setWordWrap(True)

        self.run_button = QPushButton("Run Autoresearch")
        self.run_button.clicked.connect(self.run_task)
        self.preview_input_button = None
        self.preview_output_button = None
        preview_buttons = None
        if self.preview_pane is not None:
            self.preview_input_button = QPushButton("Preview Input")
            self.preview_output_button = QPushButton("Preview Output")
            self.preview_input_button.clicked.connect(
                lambda: self.preview_pane.request_preview(
                    self.input_picker.text(),
                    "Autoresearch input",
                    hint_path_texts=self.hints_picker.paths(),
                    allow_decimation=False,
                    max_faces=600000,
                    max_vertices=1200000,
                )
            )
            self.preview_output_button.clicked.connect(
                lambda: self.preview_pane.request_preview(
                    self.output_picker.text(),
                    "Autoresearch output",
                    hint_path_texts=self.hints_picker.paths(),
                    allow_decimation=False,
                    max_faces=600000,
                    max_vertices=1200000,
                )
            )
            preview_buttons = QWidget()
            preview_buttons_layout = QHBoxLayout(preview_buttons)
            preview_buttons_layout.setContentsMargins(0, 0, 0, 0)
            preview_buttons_layout.addWidget(self.preview_input_button)
            preview_buttons_layout.addWidget(self.preview_output_button)

        paths_group = QGroupBox("Autoresearch")
        paths_form = QFormLayout(paths_group)
        paths_form.addRow(self.input_picker)
        paths_form.addRow(self.output_picker)
        paths_form.addRow(self.report_picker)
        paths_form.addRow(self.hints_picker)
        paths_form.addRow("Intended result", self.intended_mesh_type_combo)
        paths_form.addRow("TSV ledger", self.ledger_path_label)
        if preview_buttons is not None:
            paths_form.addRow("Preview", preview_buttons)

        search_group = QGroupBox("Search")
        search_form = QFormLayout(search_group)
        search_form.addRow("Merge epsilon", self.merge_eps_edit)
        search_form.addRow("Area epsilon", self.area_eps_edit)
        search_form.addRow("Dedup decimals", self.dedup_spin)
        search_form.addRow("Non-manifold edge radius", self.nonmanifold_edge_radius_edit)
        search_form.addRow(self.allow_aggressive_modes_check)
        search_form.addRow(self.fast_leapfrog_check)
        search_form.addRow("Max candidates", self.max_candidates_spin)
        search_form.addRow("Soft time budget (s)", self.time_budget_seconds_edit)
        search_form.addRow("Per-candidate timeout (s)", self.candidate_timeout_seconds_edit)
        search_form.addRow("Self-intersection timeout (s)", self.self_intersection_timeout_seconds_edit)
        search_form.addRow("Fidelity samples", self.fidelity_samples_spin)

        acceptance_group = QGroupBox("Leapfrog Acceptance")
        acceptance_form = QFormLayout(acceptance_group)
        acceptance_form.addRow("Max mean drift (normalized)", self.max_mean_distance_normalized_edit)
        acceptance_form.addRow("Max p95 drift (normalized)", self.max_p95_distance_normalized_edit)
        acceptance_form.addRow("Max component-count delta", self.max_component_count_delta_edit)
        acceptance_form.addRow("Max volume-ratio delta", self.max_volume_ratio_delta_edit)

        note = QLabel(
            "Fast Leapfrog mode uses the bounded Leapfrog-oriented search path. When max candidates, time budget, "
            "candidate timeout, or fidelity samples are left at 0, the pipeline applies its built-in defaults. "
            "It also keeps a proactive repair candidate so overlapping triangles or hidden non-manifold defects can still be rebuilt even when the initial diagnostics miss them. "
            "If a JSON report path is provided, a sibling TSV ledger is written automatically for batch comparison."
        )
        note.setWordWrap(True)

        self.create_status_widgets()

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(paths_group)
        controls_layout.addWidget(search_group)
        controls_layout.addWidget(acceptance_group)
        controls_layout.addWidget(note)
        controls_layout.addWidget(self.run_button)

        diagnostics_panel = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_panel)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.addWidget(QLabel("Status"))
        diagnostics_layout.addWidget(self.progress_bar)
        diagnostics_layout.addWidget(QLabel("Summary"))
        diagnostics_layout.addWidget(self.summary_view, 1)
        diagnostics_layout.addWidget(QLabel("Log"))
        diagnostics_layout.addWidget(self.log_view, 2)

        layout = QHBoxLayout(self)
        layout.addWidget(controls_panel, 3)
        layout.addWidget(diagnostics_panel, 2)

        self.controls = [
            self.input_picker,
            self.output_picker,
            self.report_picker,
            self.hints_picker,
            self.intended_mesh_type_combo,
            self.merge_eps_edit,
            self.area_eps_edit,
            self.dedup_spin,
            self.nonmanifold_edge_radius_edit,
            self.allow_aggressive_modes_check,
            self.fast_leapfrog_check,
            self.max_candidates_spin,
            self.time_budget_seconds_edit,
            self.candidate_timeout_seconds_edit,
            self.self_intersection_timeout_seconds_edit,
            self.fidelity_samples_spin,
            self.max_mean_distance_normalized_edit,
            self.max_p95_distance_normalized_edit,
            self.max_component_count_delta_edit,
            self.max_volume_ratio_delta_edit,
            self.run_button,
        ]
        if self.preview_input_button is not None:
            self.controls.append(self.preview_input_button)
        if self.preview_output_button is not None:
            self.controls.append(self.preview_output_button)

        self.apply_tooltips()
        self.update_ledger_path_display(self.report_picker.text())
        self.refresh_preview_buttons()

    def apply_tooltips(self):
        self.input_picker.set_help_tooltip(
            "Mesh volume to evaluate. Supports the same mesh formats as the heal pipeline, such as DXF and MSH."
        )
        self.output_picker.set_help_tooltip(
            "Healed output mesh to write when autoresearch selects a winning candidate. Use MSH for Leapfrog import."
        )
        self.report_picker.set_help_tooltip(
            "Optional JSON report with input diagnostics, candidate rankings, timing, Leapfrog validation, and the selected result."
        )
        self.hints_picker.set_help_tooltip(
            "Optional JSON or DXF file with external repair hints, such as Leapfrog-exported non-manifold polylines or overlap shapes. Autoresearch uses these hints to prioritize and keep the relevant repair candidates."
        )
        self.intended_mesh_type_combo.setToolTip(
            "Declare whether the mesh is expected to end as a closed solid or an open surface. This changes how automatic watertight candidates are prioritized and how boundary hints are interpreted."
        )
        self.ledger_path_label.setToolTip(
            "Sibling TSV ledger derived from the JSON report path. It stores one row per evaluated candidate for batch comparison."
        )
        self.merge_eps_edit.setToolTip(
            "Vertex merge tolerance used during cleanup. Larger values merge nearby vertices more aggressively."
        )
        self.area_eps_edit.setToolTip(
            "Minimum triangle area treated as valid. Smaller faces under this threshold are considered degenerate and removed."
        )
        self.dedup_spin.setToolTip(
            "Decimal precision used when detecting duplicate triangles and nearly identical vertices."
        )
        self.nonmanifold_edge_radius_edit.setToolTip(
            "Radius used by non-manifold edge repair when that candidate is enabled. Set 0 to let the pipeline estimate it automatically."
        )
        self.allow_aggressive_modes_check.setToolTip(
            "Include reconstruction-style fallback candidates, such as point-cloud or distance-field rebuilds. These are slower and can change shape more."
        )
        self.fast_leapfrog_check.setToolTip(
            "Use the bounded Leapfrog-oriented candidate set and defaults. This favors fast validation of closed, manifold, round-trip-safe outputs and still keeps a proactive repair branch even when the initial diagnostics look clean."
        )
        self.max_candidates_spin.setToolTip(
            "Maximum number of candidates to evaluate. Set 0 to let the pipeline choose its built-in full or fast set."
        )
        self.time_budget_seconds_edit.setToolTip(
            "Soft total search budget in seconds. The search stops before starting another candidate once this budget has been exceeded. Set 0 to disable it."
        )
        self.candidate_timeout_seconds_edit.setToolTip(
            "Hard timeout for heavy candidates that run in isolated subprocesses. Set 0 to disable the timeout or use fast-mode defaults."
        )
        self.self_intersection_timeout_seconds_edit.setToolTip(
            "Bounded timeout for Leapfrog self-intersection validation after round-trip export and reload. Raise it on very large solids when Leapfrog still reports self intersections."
        )
        self.fidelity_samples_spin.setToolTip(
            "Number of sampled points used to compare candidate geometry against the input. Higher values are slower but more stable. Set 0 for automatic defaults."
        )
        self.max_mean_distance_normalized_edit.setToolTip(
            "Acceptance threshold for average geometric drift, normalized by model size. Lower values preserve shape more strictly."
        )
        self.max_p95_distance_normalized_edit.setToolTip(
            "Acceptance threshold for 95th-percentile geometric drift, normalized by model size. This limits localized distortions and spikes."
        )
        self.max_component_count_delta_edit.setToolTip(
            "Maximum allowed change in disconnected solid count compared with the normalized reference mesh."
        )
        self.max_volume_ratio_delta_edit.setToolTip(
            "Maximum allowed relative volume change when the input volume is reliable. Use a larger value if strong closure changes are expected."
        )
        self.run_button.setToolTip(
            "Run the autoresearch search loop, rank candidates, validate Leapfrog round-tripping, and write the selected mesh plus reports."
        )
        if self.preview_input_button is not None:
            self.preview_input_button.setToolTip(
                "Load the full input mesh in the embedded Taichi preview."
            )
        if self.preview_output_button is not None:
            self.preview_output_button.setToolTip(
                "Load the full selected output mesh in the embedded Taichi preview after a run completes."
            )

    def update_default_output_path(self, input_path_text: str):
        if not input_path_text:
            self.last_suggested_output = ""
            self.last_suggested_report = ""
            self.update_ledger_path_display(self.report_picker.text())
            return
        suggested_output = str(suggest_autoresearch_output_path(Path(input_path_text)))
        current_output = self.output_picker.text()
        if current_output and current_output != self.last_suggested_output:
            return
        self.output_picker.setText(suggested_output)
        self.last_suggested_output = suggested_output

    def update_default_report_path(self, output_path_text: str):
        if not output_path_text:
            self.last_suggested_report = ""
            self.update_ledger_path_display(self.report_picker.text())
            return
        suggested_report = str(suggest_autoresearch_report_path(Path(output_path_text)))
        current_report = self.report_picker.text()
        if current_report and current_report != self.last_suggested_report:
            self.update_ledger_path_display(current_report)
            return
        self.report_picker.setText(suggested_report)
        self.last_suggested_report = suggested_report
        self.update_ledger_path_display(suggested_report)

    def update_ledger_path_display(self, report_path_text: str):
        report_path_text = report_path_text.strip()
        if not report_path_text:
            self.ledger_path_label.setText("No TSV ledger until a JSON report path is set.")
            return
        ledger_path = mesh_heal.derive_autoresearch_ledger_path(Path(report_path_text))
        if ledger_path is None:
            self.ledger_path_label.setText("No TSV ledger will be written.")
            return
        self.ledger_path_label.setText(str(ledger_path))

    def invalidate_output_preview(self, _path_text: str):
        self.output_ready = False
        self.refresh_preview_buttons()

    def refresh_preview_buttons(self, *_args):
        if self.preview_pane is None:
            return
        input_path = self.input_picker.text()
        input_ready = bool(input_path) and Path(input_path).exists()
        output_ready = self.output_ready and bool(self.output_picker.text())
        if is_qt_object_alive(self.preview_input_button):
            self.preview_input_button.setEnabled(input_ready)
        if is_qt_object_alive(self.preview_output_button):
            self.preview_output_button.setEnabled(output_ready)

    def set_running(self, running: bool):
        super().set_running(running)
        if not running:
            self.refresh_preview_buttons()

    def show_summary(self, result: dict):
        super().show_summary(result)
        output_path = result.get("output")
        selected_candidate = result.get("selected_candidate")
        self.output_ready = bool(selected_candidate) and bool(output_path) and Path(output_path).exists()
        if selected_candidate is None:
            failure = result.get("failure") or {}
            reason = failure.get("reason", "Autoresearch did not find a Leapfrog-accepted candidate.")
            self.append_log(reason)
        self.refresh_preview_buttons()

    def run_task(self):
        input_path = self.input_picker.text()
        output_path = self.output_picker.text()
        if not input_path or not output_path:
            QMessageBox.warning(self, "Missing path", "Input and output are required.")
            return

        try:
            kwargs = {
                "input_path": Path(input_path),
                "output_path": Path(output_path),
                "report_path": Path(self.report_picker.text()) if self.report_picker.text() else None,
                "external_hint_path": Path(self.hints_picker.text()) if self.hints_picker.text().strip() else None,
                "intended_mesh_type": str(self.intended_mesh_type_combo.currentData()),
                "merge_eps": float(self.merge_eps_edit.text()),
                "area_eps": float(self.area_eps_edit.text()),
                "dedup_decimals": int(self.dedup_spin.value()),
                "nonmanifold_edge_radius": float(self.nonmanifold_edge_radius_edit.text()),
                "allow_aggressive_modes": bool(self.allow_aggressive_modes_check.isChecked()),
                "fast_leapfrog": bool(self.fast_leapfrog_check.isChecked()),
                "max_candidates": int(self.max_candidates_spin.value()),
                "time_budget_seconds": float(self.time_budget_seconds_edit.text()),
                "candidate_timeout_seconds": float(self.candidate_timeout_seconds_edit.text()),
                "self_intersection_timeout_seconds": float(self.self_intersection_timeout_seconds_edit.text()),
                "fidelity_sample_point_count": int(self.fidelity_samples_spin.value()),
                "max_mean_distance_normalized": float(self.max_mean_distance_normalized_edit.text()),
                "max_p95_distance_normalized": float(self.max_p95_distance_normalized_edit.text()),
                "max_component_count_delta": float(self.max_component_count_delta_edit.text()),
                "max_volume_ratio_delta": float(self.max_volume_ratio_delta_edit.text()),
            }
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid numeric input",
                "All numeric autoresearch settings must be valid numbers.",
            )
            return

        self.output_ready = False
        self.start_worker(mesh_heal.run_autoresearch_pipeline, kwargs)


class BooleanTab(BaseOperationTab):
    def __init__(self, preview_pane: PreviewPane | None):
        super().__init__()
        self.preview_pane = preview_pane

        self.input_picker = FileListPicker("Selected solids", "Add Files...", MESH_FILTER)
        self.output_picker = FilePicker("Output", "Save...", OUTPUT_FILTER, save_dialog=True)
        self.report_picker = FilePicker("Report", "Save...", JSON_FILTER, save_dialog=True)

        self.operation_checks = {
            "union": QCheckBox("Union"),
            "intersection": QCheckBox("Intersection"),
            "clip": QCheckBox("Clip (Left - Right)"),
        }
        self.operation_checks["union"].setChecked(True)

        operation_widget = QWidget()
        operation_layout = QVBoxLayout(operation_widget)
        operation_layout.setContentsMargins(0, 0, 0, 0)
        for operation in mesh_heal.BOOLEAN_OPERATIONS:
            operation_layout.addWidget(self.operation_checks[operation])

        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)
        self.run_button = QPushButton("Run Boolean")
        self.run_button.clicked.connect(self.run_task)
        self.preview_input_button = None
        self.preview_output_button = None
        preview_buttons = None
        if self.preview_pane is not None:
            self.preview_input_button = QPushButton("Preview Selected Input")
            self.preview_output_button = QPushButton("Preview Output")
            self.preview_input_button.clicked.connect(self.preview_input)
            self.preview_output_button.clicked.connect(self.preview_output)
            preview_buttons = QWidget()
            preview_buttons_layout = QHBoxLayout(preview_buttons)
            preview_buttons_layout.setContentsMargins(0, 0, 0, 0)
            preview_buttons_layout.addWidget(self.preview_input_button)
            preview_buttons_layout.addWidget(self.preview_output_button)

        options_group = QGroupBox("Boolean Operations")
        form = QFormLayout(options_group)
        form.addRow(self.input_picker)
        form.addRow("Operations", operation_widget)
        form.addRow(self.output_picker)
        form.addRow(self.report_picker)
        form.addRow("Merge epsilon", self.merge_eps_edit)
        form.addRow("Area epsilon", self.area_eps_edit)
        form.addRow("Dedup decimals", self.dedup_spin)
        if preview_buttons is not None:
            form.addRow("Preview", preview_buttons)
        form.addRow(self.run_button)

        self.create_status_widgets()

        layout = QVBoxLayout(self)
        layout.addWidget(options_group)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.progress_bar)

        panels = QGridLayout()
        panels.addWidget(QLabel("Log"), 0, 0)
        panels.addWidget(QLabel("Summary"), 0, 1)
        panels.addWidget(self.log_view, 1, 0)
        panels.addWidget(self.summary_view, 1, 1)
        layout.addLayout(panels)

        note = QLabel(
            "Union and intersection run across the solids in the list. Clip uses the first selected solid as the "
            "base and subtracts each following solid in order, so use Move Up and Move Down when order matters. "
            "You can keep adding files from different folders with Add Files...."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.controls = [
            self.input_picker,
            self.output_picker,
            self.report_picker,
            self.merge_eps_edit,
            self.area_eps_edit,
            self.dedup_spin,
            self.run_button,
        ]
        if self.preview_input_button is not None:
            self.controls.append(self.preview_input_button)
        if self.preview_output_button is not None:
            self.controls.append(self.preview_output_button)
        self.controls.extend(self.operation_checks.values())

    def selected_operations(self) -> list[str]:
        return [operation for operation, checkbox in self.operation_checks.items() if checkbox.isChecked()]

    def preview_output(self):
        if self.preview_pane is None:
            return
        operations = self.selected_operations()
        if len(operations) != 1:
            QMessageBox.information(
                self,
                "Select one operation",
                "Preview Output needs exactly one selected operation. For batch runs, preview the generated file directly.",
            )
            return

        output_path = self.output_picker.text()
        if not output_path:
            QMessageBox.warning(self, "Missing path", "Select an output path before previewing the result.")
            return

        resolved_path = mesh_heal.derive_operation_path(Path(output_path), operations[0], False)
        self.preview_pane.request_preview(str(resolved_path), "Boolean output")

    def preview_input(self):
        if self.preview_pane is None:
            return
        current_path = self.input_picker.current_path()
        if current_path is None:
            QMessageBox.warning(self, "Missing selection", "Select one input file in the list before previewing.")
            return
        self.preview_pane.request_preview(current_path, "Boolean input")

    def run_task(self):
        input_paths = self.input_picker.paths()
        output_path = self.output_picker.text()
        operations = self.selected_operations()
        if not input_paths or not output_path:
            QMessageBox.warning(self, "Missing path", "At least two input solids and an output path are required.")
            return
        if len(input_paths) < 2:
            QMessageBox.warning(self, "Need more solids", "Select at least two solids for boolean operations.")
            return
        if not operations:
            QMessageBox.warning(self, "Missing operation", "Select at least one boolean operation.")
            return

        try:
            kwargs = {
                "input_paths": [Path(path) for path in input_paths],
                "output_path": Path(output_path),
                "operations": operations,
                "report_path": Path(self.report_picker.text()) if self.report_picker.text() else None,
                "merge_eps": float(self.merge_eps_edit.text()),
                "area_eps": float(self.area_eps_edit.text()),
                "dedup_decimals": int(self.dedup_spin.value()),
            }
        except ValueError:
            QMessageBox.warning(self, "Invalid numeric input", "Merge epsilon and area epsilon must be valid numbers.")
            return

        self.start_worker(mesh_heal.run_multi_input_boolean_pipelines, kwargs)


class ManualRepairEditorWindow(QMainWindow):
    state_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Heal Editor")
        self.resize(1680, 980)

        self.current_path: Path | None = None
        self.original_mesh = None
        self.current_mesh = None
        self.undo_stack: list[dict] = []
        self.action_history: list[dict] = []
        self.selected_face_ids: set[int] = set()
        self.loop_vertex_ids: list[int] = []
        self.pick_mode = "none"
        self.plotter_init_error = None
        self.plotter = None
        self.viewer_widget = None
        self.render_update_pending = False
        self.analysis_area_eps = 1e-12
        self.analysis_dedup_decimals = 8
        self.loaded_mesh_intended_type = "auto"
        self.issue_payload: dict = {"issues": [], "issue_count": 0, "categories": {}, "truncated": {}}
        self.external_issue_payload: dict = {
            "issues": [],
            "issue_count": 0,
            "categories": {},
            "source": None,
            "skipped_issue_count": 0,
        }
        self.cached_mesh_summary: dict = {
            "vertices": 0,
            "faces": 0,
            "intended_mesh_type": "auto",
            "boundary_edges": None,
            "duplicate_faces": None,
            "degenerate_faces": None,
            "nonmanifold_edges": None,
        }
        self.issue_analysis_status = "No analysis run"
        self.issue_thread = None
        self.issue_worker = None
        self.issue_analysis_token = 0
        self.focus_overlay_points: np.ndarray | None = None
        self.focus_point: np.ndarray | None = None
        self.tool_panel = None

        central = QWidget()
        layout = QVBoxLayout(central)

        self.mode_status = QLabel("Interaction mode: camera")
        self.mesh_status = QLabel("No mesh loaded")
        self.selection_status = QLabel("Selected faces: 0 | contour vertices: 0")

        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.mode_status, 1)
        status_layout.addWidget(self.mesh_status, 2)
        status_layout.addWidget(self.selection_status, 1)

        self.viewer_host = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_host)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_placeholder = create_viewer_placeholder(
            "Editor viewport is idle. Load a mesh from the floating tool panel to start editing.",
            min_height=520,
        )
        self.viewer_widget = self.viewer_placeholder
        self.viewer_layout.addWidget(self.viewer_widget)

        self.issue_tree = QTreeWidget()
        self.issue_tree.setColumnCount(2)
        self.issue_tree.setHeaderLabels(["Issue", "Details"])
        self.issue_tree.itemSelectionChanged.connect(self.on_issue_selection_changed)

        issue_panel = QWidget()
        issue_layout = QVBoxLayout(issue_panel)
        issue_layout.setContentsMargins(0, 0, 0, 0)
        issue_note = QLabel(
            "Repair browser lists concrete problems in the current triangulation. Selecting an item highlights it and focuses the viewport."
        )
        issue_note.setWordWrap(True)
        issue_layout.addWidget(issue_note)
        issue_layout.addWidget(self.issue_tree, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Editor log")
        self.log_view.setMaximumBlockCount(400)

        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlaceholderText("Operation summary")

        diagnostics = QTabWidget()
        diagnostics.addTab(issue_panel, "Repair Browser")
        diagnostics.addTab(self.log_view, "Log")
        diagnostics.addTab(self.summary_view, "Summary")

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.viewer_host)
        split.addWidget(diagnostics)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 260])

        layout.addWidget(status_row)
        layout.addWidget(split, 1)
        self.setCentralWidget(central)

    def register_tool_panel(self, tool_panel: QMainWindow) -> None:
        self.tool_panel = tool_panel

    def closeEvent(self, event):
        if self.tool_panel is not None:
            self.tool_panel.close()
        super().closeEvent(event)

    def has_mesh(self) -> bool:
        return self.current_mesh is not None

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def selected_face_count(self) -> int:
        return len(self.selected_face_ids)

    def contour_vertex_count(self) -> int:
        return len(self.loop_vertex_ids)

    def issue_count(self) -> int:
        return int(self.build_combined_issue_payload().get("issue_count", 0))

    def external_issue_count(self) -> int:
        return int(self.external_issue_payload.get("issue_count", 0))

    def build_combined_issue_payload(self) -> dict:
        internal_issues = list(self.issue_payload.get("issues", []))
        external_issues = list(self.external_issue_payload.get("issues", []))
        combined_issues = [*internal_issues, *external_issues]
        category_counts: dict[str, int] = {}
        for payload in (self.issue_payload, self.external_issue_payload):
            for key, value in (payload.get("categories") or {}).items():
                category_counts[str(key)] = category_counts.get(str(key), 0) + int(value)
        return {
            "issues": combined_issues,
            "issue_count": int(len(combined_issues)),
            "categories": category_counts,
            "truncated": dict(self.issue_payload.get("truncated", {})),
            "skipped": dict(self.issue_payload.get("skipped", {})),
            "internal_issue_count": int(len(internal_issues)),
            "external_issue_count": int(len(external_issues)),
            "external_source": self.external_issue_payload.get("source"),
            "external_skipped_issue_count": int(self.external_issue_payload.get("skipped_issue_count", 0)),
        }

    def append_log(self, message: str):
        self.log_view.appendPlainText(message)

    def show_summary(self, result: dict):
        self.summary_view.setPlainText(json.dumps(result, indent=2))

    def set_viewer_widget(self, widget: QWidget) -> None:
        if self.viewer_widget is not None:
            self.viewer_layout.removeWidget(self.viewer_widget)
            self.viewer_widget.setParent(None)
        self.viewer_widget = widget
        self.viewer_layout.addWidget(widget)

    def ensure_plotter(self) -> bool:
        if self.plotter is not None:
            return True

        plotter, viewer_widget, error = create_plotter_or_placeholder(
            self.viewer_host,
            "The editor viewport needs pyvistaqt. Manual tools remain visible, but interactive picking is unavailable until the renderer loads.",
            min_height=520,
        )
        self.plotter = plotter
        self.plotter_init_error = error
        self.viewer_placeholder = viewer_widget if plotter is None else None
        self.set_viewer_widget(viewer_widget)
        if self.plotter is None:
            self.mode_status.setText("Interaction mode: viewer unavailable")
            if error:
                self.append_log(f"Editor renderer unavailable: {error}")
            self.state_changed.emit()
            return False

        self.plotter.set_background("#f8f3eb")
        self.plotter.view_isometric()
        return True

    def ensure_mesh_loaded(self) -> bool:
        if self.current_mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Load a mesh before using manual repair tools.")
            return False
        if self.plotter is None and not self.ensure_plotter():
            QMessageBox.warning(
                self,
                "Preview unavailable",
                self.plotter_init_error or "The manual editor needs pyvistaqt for interactive picking.",
            )
            return False
        return True

    def update_status_labels(self):
        if self.current_mesh is None:
            self.mesh_status.setText("No mesh loaded")
        else:
            summary = self.cached_mesh_summary
            parts = [
                f"Mesh: {int(summary.get('vertices', 0)):,} vertices",
                f"{int(summary.get('faces', 0)):,} faces",
                f"intent {describe_intended_mesh_type(str(summary.get('intended_mesh_type', 'auto'))).lower()}",
            ]
            for key, label in (
                ("boundary_edges", "boundary edges"),
                ("duplicate_faces", "duplicates"),
                ("degenerate_faces", "degenerate"),
                ("nonmanifold_edges", "non-manifold"),
            ):
                value = summary.get(key)
                if value is not None:
                    parts.append(f"{label} {int(value):,}")
            parts.append(self.issue_analysis_status)
            self.mesh_status.setText(" | ".join(parts))
        self.selection_status.setText(
            f"Selected faces: {len(self.selected_face_ids):,} | contour vertices: {len(self.loop_vertex_ids):,} | issues: {self.issue_count():,}"
        )
        self.state_changed.emit()

    def clear_focus_overlay(self) -> None:
        self.focus_overlay_points = None
        self.focus_point = None

    def update_basic_mesh_summary(self) -> None:
        if self.current_mesh is None:
            self.cached_mesh_summary = {
                "vertices": 0,
                "faces": 0,
                "intended_mesh_type": self.loaded_mesh_intended_type,
                "boundary_edges": None,
                "duplicate_faces": None,
                "degenerate_faces": None,
                "nonmanifold_edges": None,
            }
            return
        self.cached_mesh_summary = {
            "vertices": int(len(self.current_mesh.vertices)),
            "faces": int(len(self.current_mesh.faces)),
            "intended_mesh_type": self.loaded_mesh_intended_type,
            "boundary_edges": None,
            "duplicate_faces": None,
            "degenerate_faces": None,
            "nonmanifold_edges": None,
        }

    def clear_selection_state(self):
        self.selected_face_ids.clear()
        self.loop_vertex_ids.clear()
        self.clear_focus_overlay()

    def disable_picking(self):
        self.pick_mode = "none"
        self.mode_status.setText("Interaction mode: camera")
        if self.plotter is None:
            return
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

    def schedule_render_scene(self, reset_camera: bool = False, focus_point: np.ndarray | None = None):
        if self.render_update_pending:
            return

        self.render_update_pending = True

        def _run() -> None:
            self.render_update_pending = False
            self.render_scene(reset_camera=reset_camera, focus_point=focus_point)

        QTimer.singleShot(0, _run)

    def push_undo_state(self, label: str):
        if self.current_mesh is None:
            return
        self.undo_stack.append(
            {
                "label": label,
                "mesh": self.current_mesh.copy(),
                "selected_face_ids": sorted(self.selected_face_ids),
                "loop_vertex_ids": list(self.loop_vertex_ids),
            }
        )
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.state_changed.emit()

    def record_action(self, action: str, result: dict):
        self.action_history.append({"action": action, "result": result})
        self.show_summary(
            {
                "mode": "manual-repair-step",
                "action": action,
                "result": result,
            }
        )

    def focus_camera_on_point(self, point: Sequence[float] | np.ndarray) -> None:
        if self.plotter is None:
            return
        target = np.asarray(point, dtype=float).reshape(3)
        try:
            camera = self.plotter.camera
            position = np.asarray(camera.position, dtype=float)
            focal_point = np.asarray(camera.focal_point, dtype=float)
            offset = position - focal_point
            if float(np.linalg.norm(offset)) <= 1e-9:
                offset = np.array([1.0, 1.0, 1.0], dtype=float)
            camera.focal_point = target.tolist()
            camera.position = (target + offset).tolist()
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def render_scene(self, reset_camera: bool = False, focus_point: np.ndarray | None = None):
        if self.plotter is None or self.current_mesh is None or pv is None:
            self.update_status_labels()
            return

        try:
            self.plotter.clear()
            mesh_poly = mesh_heal.mesh_to_polydata(self.current_mesh)
            mesh_poly.cell_data["face_id"] = np.arange(len(self.current_mesh.faces), dtype=np.int64)
            self.plotter.add_mesh(
                mesh_poly,
                name="manual-mesh",
                color="#d0a45c",
                smooth_shading=False,
                show_edges=True,
                edge_color="#5d4f37",
                line_width=1,
                pickable=True,
                lighting=True,
            )

            if self.selected_face_ids:
                selected = np.asarray(sorted(self.selected_face_ids), dtype=np.int64)
                selected_mesh = self.current_mesh.submesh([selected], append=True, repair=False)
                selected_mesh = mesh_heal.as_mesh(selected_mesh)
                if len(selected_mesh.faces) > 0:
                    self.plotter.add_mesh(
                        mesh_heal.mesh_to_polydata(selected_mesh),
                        name="manual-selected-faces",
                        color="#cf3d2e",
                        opacity=0.88,
                        show_edges=True,
                        edge_color="#6a1e17",
                        line_width=2,
                        pickable=False,
                    )

            if self.loop_vertex_ids:
                loop_points = np.asarray(self.current_mesh.vertices[self.loop_vertex_ids], dtype=float)
                self.plotter.add_mesh(
                    pv.PolyData(loop_points),
                    name="manual-loop-points",
                    color="#177e89",
                    point_size=13,
                    render_points_as_spheres=True,
                    pickable=False,
                )
                if len(loop_points) >= 2:
                    closed_points = np.vstack([loop_points, loop_points[0]])
                    line_poly = pv.PolyData(closed_points)
                    line_poly.lines = np.hstack((
                        np.asarray([len(closed_points)], dtype=np.int64),
                        np.arange(len(closed_points), dtype=np.int64),
                    ))
                    self.plotter.add_mesh(
                        line_poly,
                        name="manual-loop-line",
                        color="#177e89",
                        line_width=4,
                        pickable=False,
                    )

            if self.focus_overlay_points is not None and len(self.focus_overlay_points) > 0:
                focus_poly = pv.PolyData(self.focus_overlay_points)
                self.plotter.add_mesh(
                    focus_poly,
                    name="issue-focus-points",
                    color="#0a6c74",
                    point_size=14,
                    render_points_as_spheres=True,
                    pickable=False,
                )
                if len(self.focus_overlay_points) >= 2:
                    line_poly = pv.PolyData(self.focus_overlay_points)
                    line_poly.lines = np.hstack((
                        np.asarray([len(self.focus_overlay_points)], dtype=np.int64),
                        np.arange(len(self.focus_overlay_points), dtype=np.int64),
                    ))
                    self.plotter.add_mesh(
                        line_poly,
                        name="issue-focus-line",
                        color="#0a6c74",
                        line_width=5,
                        pickable=False,
                    )

            if self.focus_point is not None:
                self.plotter.add_mesh(
                    pv.PolyData(np.asarray([self.focus_point], dtype=float)),
                    name="issue-focus-marker",
                    color="#0a6c74",
                    point_size=18,
                    render_points_as_spheres=True,
                    pickable=False,
                )

            if reset_camera:
                self.plotter.view_isometric()
                self.plotter.reset_camera()
            elif focus_point is not None:
                self.focus_camera_on_point(focus_point)
        except Exception as exc:
            self.disable_picking()
            self.append_log(f"Manual repair render failed: {exc}")
            QMessageBox.warning(self, "Manual repair render failed", str(exc))
        finally:
            self.update_status_labels()

    def set_issue_tree_message(self, title: str, details: str = "") -> None:
        self.issue_tree.blockSignals(True)
        self.issue_tree.clear()
        self.issue_tree.addTopLevelItem(QTreeWidgetItem([title, details]))
        self.issue_tree.blockSignals(False)

    def populate_issue_tree(self, payload: dict) -> None:
        self.issue_tree.blockSignals(True)
        self.issue_tree.clear()
        skipped = payload.get("skipped", {})
        if skipped.get("duplicate_groups") or skipped.get("degenerate_faces"):
            notes = []
            if skipped.get("duplicate_groups"):
                notes.append("duplicate-face scan skipped on this large mesh")
            if skipped.get("degenerate_faces"):
                notes.append("degenerate-face scan skipped on this large mesh")
            self.issue_tree.addTopLevelItem(QTreeWidgetItem(["Analysis limits applied", "; ".join(notes)]))

        grouped: dict[str, list[dict]] = {}
        for issue in payload.get("issues", []):
            grouped.setdefault(str(issue.get("category", "other")), []).append(issue)

        category_labels = [
            ("duplicate-faces", "Duplicate Face Groups"),
            ("degenerate-faces", "Degenerate Faces"),
            ("nonmanifold-edges", "Non-Manifold Edges"),
            ("boundary-loops", "Boundary Loops"),
            ("external-overlap-hints", "Imported Overlap Hints"),
            ("external-hints", "Imported Repair Hints"),
        ]
        seen_categories: set[str] = set()
        for key, label in category_labels:
            category_issues = grouped.get(key, [])
            if not category_issues:
                continue
            seen_categories.add(key)
            root = QTreeWidgetItem([f"{label} ({len(category_issues):,})", ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.issue_tree.addTopLevelItem(root)
            for issue in category_issues:
                child = QTreeWidgetItem([str(issue.get("label", key)), str(issue.get("description", ""))])
                child.setData(0, Qt.ItemDataRole.UserRole, issue)
                root.addChild(child)
            root.setExpanded(True)

        for key in sorted(grouped):
            if key in seen_categories:
                continue
            category_issues = grouped[key]
            label = key.replace("-", " ").replace("_", " ").title()
            root = QTreeWidgetItem([f"{label} ({len(category_issues):,})", ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.issue_tree.addTopLevelItem(root)
            for issue in category_issues:
                child = QTreeWidgetItem([str(issue.get("label", key)), str(issue.get("description", ""))])
                child.setData(0, Qt.ItemDataRole.UserRole, issue)
                root.addChild(child)
            root.setExpanded(True)

        if self.issue_tree.topLevelItemCount() == 0:
            self.issue_tree.addTopLevelItem(QTreeWidgetItem(["No repair issues detected", ""]))
        self.issue_tree.blockSignals(False)

    def refresh_issue_browser(self, area_eps: float | None = None, dedup_decimals: int | None = None):
        if area_eps is not None:
            self.analysis_area_eps = area_eps
        if dedup_decimals is not None:
            self.analysis_dedup_decimals = dedup_decimals
        if self.current_mesh is None:
            self.issue_payload = {"issues": [], "issue_count": 0, "categories": {}, "truncated": {}}
            self.cached_mesh_summary = {
                "vertices": 0,
                "faces": 0,
                "boundary_edges": None,
                "duplicate_faces": None,
                "degenerate_faces": None,
                "nonmanifold_edges": None,
            }
            self.issue_analysis_status = "No analysis run"
            self.set_issue_tree_message("No mesh loaded")
            self.update_status_labels()
            return
        self.update_basic_mesh_summary()
        self.issue_payload = {"issues": [], "issue_count": 0, "categories": {}, "truncated": {}}
        self.issue_analysis_status = "Analyzing errors..."
        self.set_issue_tree_message("Analyzing repair issues...", "The editor stays usable while this runs.")
        self.issue_analysis_token += 1
        token = self.issue_analysis_token

        thread = QThread(self)
        worker = IssueAnalysisWorker(
            token=token,
            mesh=self.current_mesh,
            area_eps=self.analysis_area_eps,
            dedup_decimals=self.analysis_dedup_decimals,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.completed.connect(self.on_issue_analysis_completed)
        worker.failed.connect(self.on_issue_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self.on_issue_analysis_finished(token))

        self.issue_thread = thread
        self.issue_worker = worker
        thread.start()
        self.update_status_labels()

    def on_issue_analysis_completed(self, token: int, payload: dict):
        if token != self.issue_analysis_token:
            return
        self.issue_payload = payload
        summary = payload.get("summary", {})
        self.cached_mesh_summary = {
            "vertices": int(summary.get("vertices", len(self.current_mesh.vertices) if self.current_mesh is not None else 0)),
            "faces": int(summary.get("faces", len(self.current_mesh.faces) if self.current_mesh is not None else 0)),
            "boundary_edges": summary.get("boundary_edges"),
            "duplicate_faces": summary.get("duplicate_faces"),
            "degenerate_faces": summary.get("degenerate_faces"),
            "nonmanifold_edges": summary.get("nonmanifold_edges"),
        }
        skipped = payload.get("skipped", {})
        skipped_notes = []
        if skipped.get("duplicate_groups"):
            skipped_notes.append("duplicates skipped")
        if skipped.get("degenerate_faces"):
            skipped_notes.append("degenerates skipped")
        status = f"analysis ready: {int(payload.get('issue_count', 0)):,} browser items"
        if skipped_notes:
            status = f"{status} ({', '.join(skipped_notes)})"
        self.issue_analysis_status = status
        self.populate_issue_tree(self.build_combined_issue_payload())
        self.update_status_labels()

    def on_issue_analysis_failed(self, token: int, details: str):
        if token != self.issue_analysis_token:
            return
        self.issue_payload = {"issues": [], "issue_count": 0, "categories": {}, "truncated": {}}
        self.issue_analysis_status = "analysis failed"
        if self.external_issue_count() > 0:
            self.populate_issue_tree(self.build_combined_issue_payload())
        else:
            self.set_issue_tree_message("Issue analysis failed", "See the Log tab for details.")
        self.append_log(details)
        self.update_status_labels()

    def on_issue_analysis_finished(self, token: int):
        if token == self.issue_analysis_token:
            self.issue_thread = None
            self.issue_worker = None

    def on_issue_selection_changed(self):
        selected_items = self.issue_tree.selectedItems()
        if not selected_items:
            return
        issue = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(issue, dict):
            return
        self.apply_issue_focus(issue)

    def apply_issue_focus(self, issue: dict):
        self.clear_focus_overlay()
        face_ids = [int(face_id) for face_id in issue.get("face_ids", [])]
        vertex_ids = [int(vertex_id) for vertex_id in issue.get("vertex_ids", [])]
        polyline_points = issue.get("polyline_points")
        point = issue.get("point")

        if face_ids:
            self.selected_face_ids = set(face_ids)
        else:
            self.selected_face_ids.clear()

        if vertex_ids:
            self.loop_vertex_ids = vertex_ids
        elif issue.get("category") != "boundary-loops":
            self.loop_vertex_ids.clear()

        if polyline_points:
            self.focus_overlay_points = np.asarray(polyline_points, dtype=float)
        if point is not None:
            self.focus_point = np.asarray(point, dtype=float)

        self.append_log(f"Focused repair issue: {issue.get('label', 'issue')}")
        self.render_scene(reset_camera=False, focus_point=self.focus_point)

    def load_mesh_from_path(
        self,
        path: Path,
        area_eps: float = 1e-12,
        dedup_decimals: int = 8,
        intended_mesh_type: str = "auto",
    ):
        try:
            self.disable_picking()
            mesh = mesh_heal.load_mesh(path)
        except mesh_heal.SkippedInputError as exc:
            self.append_log(f"Skipped unsupported input: {path} ({exc})")
            QMessageBox.information(self, "Input skipped", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        normalized_intended_mesh_type = mesh_heal.normalize_intended_mesh_type(intended_mesh_type)
        self.current_path = path
        self.original_mesh = mesh.copy()
        self.current_mesh = mesh.copy()
        self.loaded_mesh_intended_type = normalized_intended_mesh_type
        self.undo_stack.clear()
        self.action_history.clear()
        self.clear_selection_state()
        self.external_issue_payload = {
            "issues": [],
            "issue_count": 0,
            "categories": {},
            "source": None,
            "skipped_issue_count": 0,
        }
        self.cached_mesh_summary = {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "intended_mesh_type": normalized_intended_mesh_type,
            "boundary_edges": None,
            "duplicate_faces": None,
            "degenerate_faces": None,
            "nonmanifold_edges": None,
        }
        self.issue_payload = {"issues": [], "issue_count": 0, "categories": {}, "truncated": {}}
        self.issue_analysis_status = "analysis queued"
        self.append_log(
            f"Loaded mesh for manual repair: {path} | intended result: {describe_intended_mesh_type(normalized_intended_mesh_type)}"
        )
        self.ensure_plotter()
        self.set_issue_tree_message("Mesh loaded", "Run analysis or wait for the background repair scan to finish.")
        try:
            self.render_scene(reset_camera=True)
        except Exception as exc:
            QMessageBox.warning(self, "Manual repair load warning", str(exc))
            self.append_log(str(exc))
        self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
        self.show_summary(
            {
                "mode": "manual-repair-load",
                "input": str(path),
                "intended_mesh_type": normalized_intended_mesh_type,
                "report": self.cached_mesh_summary,
                "issues": {"status": self.issue_analysis_status},
            }
        )

    def save_mesh_to_path(self, output_path: Path, report_path: Path | None = None):
        if self.current_mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Load and edit a mesh before saving.")
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh_heal.export_mesh(self.current_mesh, output_path)
            summary = {
                "mode": "manual-repair",
                "input": str(self.current_path) if self.current_path is not None else None,
                "output": str(output_path),
                "before": self.original_mesh and mesh_heal.mesh_report(self.original_mesh, area_eps=self.analysis_area_eps).__dict__,
                "after": mesh_heal.mesh_report(self.current_mesh, area_eps=self.analysis_area_eps).__dict__,
                "issues": self.build_combined_issue_payload(),
                "issue_sources": {
                    "internal": self.issue_payload,
                    "external": self.external_issue_payload,
                },
                "actions": list(self.action_history),
            }
            if report_path is not None:
                mesh_heal.write_json_report(summary, report_path)
            self.show_summary(summary)
            self.append_log(f"Saved edited mesh to {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def analyze_current_mesh(self, area_eps: float, dedup_decimals: int):
        if self.current_mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Load a mesh before running repair analysis.")
            return
        self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
        self.append_log(
            f"Analyzed mesh: {self.issue_payload.get('issue_count', 0):,} issue entries across the current triangulation"
        )

    def load_external_issues_from_path(self, path: Path, merge_eps: float = 1e-8):
        if self.current_mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Load a mesh before importing external repair hints.")
            return
        try:
            payload = mesh_heal.load_external_issue_source(path)
            normalized = mesh_heal.normalize_external_issue_payload(
                payload,
                self.current_mesh,
                merge_eps=merge_eps,
                source=str(path),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Hint import failed", str(exc))
            self.append_log(str(exc))
            return

        self.external_issue_payload = normalized
        combined_payload = self.build_combined_issue_payload()
        if combined_payload.get("issue_count", 0) > 0:
            self.populate_issue_tree(combined_payload)
        else:
            self.set_issue_tree_message("No imported repair hints", "The file did not contain any usable issue entries.")
        self.show_summary(
            {
                "mode": "manual-repair-import-hints",
                "source": str(path),
                "source_format": path.suffix.lower().lstrip("."),
                "imported_hints": normalized,
                "issues": combined_payload,
            }
        )
        self.append_log(f"Loaded {normalized.get('issue_count', 0):,} external repair hints from {path}")
        skipped_count = int(normalized.get("skipped_issue_count", 0))
        if skipped_count > 0:
            self.append_log(f"Skipped {skipped_count:,} imported hint entries that could not be interpreted")
        self.update_status_labels()

    def clear_external_issues(self):
        if self.external_issue_count() == 0:
            return
        self.external_issue_payload = {
            "issues": [],
            "issue_count": 0,
            "categories": {},
            "source": None,
            "skipped_issue_count": 0,
        }
        if self.current_mesh is None:
            self.set_issue_tree_message("No mesh loaded")
        else:
            combined_payload = self.build_combined_issue_payload()
            if combined_payload.get("issue_count", 0) > 0:
                self.populate_issue_tree(combined_payload)
            else:
                self.set_issue_tree_message("No repair issues detected", "")
        self.append_log("Cleared imported repair hints")
        self.update_status_labels()

    def undo_last_edit(self):
        if not self.undo_stack:
            return
        self.disable_picking()
        state = self.undo_stack.pop()
        self.current_mesh = state["mesh"].copy()
        self.selected_face_ids = set(state["selected_face_ids"])
        self.loop_vertex_ids = list(state["loop_vertex_ids"])
        self.clear_focus_overlay()
        self.append_log(f"Undid manual repair step: {state['label']}")
        self.refresh_issue_browser()
        self.render_scene(reset_camera=False)

    def reset_to_loaded_mesh(self):
        if self.original_mesh is None:
            return
        self.disable_picking()
        self.current_mesh = self.original_mesh.copy()
        self.undo_stack.clear()
        self.action_history.clear()
        self.clear_selection_state()
        self.append_log("Reset manual repair mesh back to the loaded state")
        self.refresh_issue_browser()
        self.render_scene(reset_camera=True)

    def clear_face_selection(self):
        self.selected_face_ids.clear()
        self.clear_focus_overlay()
        self.append_log("Cleared selected faces")
        self.render_scene(reset_camera=False)

    def clear_loop_selection(self):
        self.loop_vertex_ids.clear()
        self.clear_focus_overlay()
        if self.pick_mode == "contour-pick":
            self.disable_picking()
        self.append_log("Cleared contour selection")
        self.render_scene(reset_camera=False)

    def extract_face_ids(self, picked_cells) -> list[int]:
        if picked_cells is None or pv is None:
            return []

        face_ids: list[int] = []

        def collect_ids(dataset) -> None:
            if dataset is None:
                return
            if isinstance(dataset, pv.MultiBlock):
                for block in dataset:
                    collect_ids(block)
                return
            for name in ("face_id", "vtkOriginalCellIds"):
                try:
                    if name in dataset.cell_data:
                        values = np.asarray(dataset.cell_data[name], dtype=np.int64).reshape(-1)
                        face_ids.extend(int(value) for value in values)
                        return
                except Exception:
                    continue

        collect_ids(picked_cells)
        return sorted({face_id for face_id in face_ids if face_id >= 0})

    def nearest_vertex_id(self, point: np.ndarray, picker) -> int:
        if self.current_mesh is None:
            raise ValueError("No mesh loaded")
        if picker is not None and hasattr(picker, "GetPointId"):
            try:
                point_id = int(picker.GetPointId())
                if 0 <= point_id < len(self.current_mesh.vertices):
                    return point_id
            except Exception:
                pass
        vertices = np.asarray(self.current_mesh.vertices, dtype=float)
        target = np.asarray(point, dtype=float)
        distances = np.linalg.norm(vertices - target, axis=1)
        return int(np.argmin(distances))

    def nearest_face_id(self, point: Sequence[float] | np.ndarray) -> int:
        if self.current_mesh is None or len(self.current_mesh.faces) == 0:
            raise ValueError("No mesh loaded")
        centers = np.asarray(self.current_mesh.triangles_center, dtype=float)
        target = np.asarray(point, dtype=float).reshape(3)
        distances = np.linalg.norm(centers - target, axis=1)
        return int(np.argmin(distances))

    def append_loop_vertex(self, vertex_id: int) -> None:
        if self.loop_vertex_ids and self.loop_vertex_ids[-1] == vertex_id:
            return
        self.loop_vertex_ids.append(vertex_id)

    def append_loop_edge(self, edge_vertex_ids: Sequence[int]) -> None:
        if self.current_mesh is None:
            return
        start, end = int(edge_vertex_ids[0]), int(edge_vertex_ids[1])
        if not self.loop_vertex_ids:
            self.loop_vertex_ids.extend([start, end])
            return
        last_vertex = self.loop_vertex_ids[-1]
        if last_vertex == start:
            self.append_loop_vertex(end)
            return
        if last_vertex == end:
            self.append_loop_vertex(start)
            return
        vertices = np.asarray(self.current_mesh.vertices, dtype=float)
        last_point = vertices[last_vertex]
        start_distance = float(np.linalg.norm(vertices[start] - last_point))
        end_distance = float(np.linalg.norm(vertices[end] - last_point))
        ordered = (start, end) if start_distance <= end_distance else (end, start)
        self.append_loop_vertex(ordered[0])
        self.append_loop_vertex(ordered[1])

    def start_rectangle_face_selection(self):
        if not self.ensure_mesh_loaded():
            return
        self.disable_picking()
        self.pick_mode = "face-selection"
        self.mode_status.setText("Interaction mode: rectangle face selection")
        self.plotter.enable_cell_picking(
            callback=self.on_faces_picked,
            through=False,
            show=True,
            show_message="Press R in the viewport and drag a rectangle to select triangles.",
            style="surface",
            color="#cf3d2e",
            line_width=3,
        )
        self.append_log("Rectangle face selection is active. Press R and drag to add triangles.")

    def on_faces_picked(self, picked_cells):
        face_ids = self.extract_face_ids(picked_cells)
        self.disable_picking()
        if not face_ids:
            self.append_log("Rectangle selection finished with no triangles picked")
            self.schedule_render_scene(reset_camera=False)
            return
        before_count = len(self.selected_face_ids)
        self.selected_face_ids.update(face_ids)
        added = len(self.selected_face_ids) - before_count
        self.append_log(f"Added {added:,} triangles to the selection ({len(self.selected_face_ids):,} total)")
        self.schedule_render_scene(reset_camera=False)

    def start_triangle_pick(self):
        if not self.ensure_mesh_loaded():
            return
        self.disable_picking()
        self.pick_mode = "triangle-pick"
        self.mode_status.setText("Interaction mode: triangle click selection")
        self.plotter.enable_point_picking(
            callback=self.on_triangle_picked,
            left_clicking=True,
            picker="point",
            show_message="Left click a triangle to toggle it in the selection.",
            show_point=False,
            use_picker=True,
            clear_on_no_selection=False,
        )
        self.append_log("Triangle click selection is active. Left click triangles to toggle them.")

    def on_triangle_picked(self, point, _picker):
        try:
            face_id = self.nearest_face_id(point)
            if face_id in self.selected_face_ids:
                self.selected_face_ids.remove(face_id)
                self.append_log(f"Removed triangle {face_id} from the selection")
            else:
                self.selected_face_ids.add(face_id)
                self.append_log(f"Selected triangle {face_id}")
        except Exception as exc:
            QMessageBox.warning(self, "Triangle selection failed", str(exc))
            self.append_log(str(exc))
            self.disable_picking()
        finally:
            self.schedule_render_scene(reset_camera=False)

    def start_boundary_loop_pick(self):
        if not self.ensure_mesh_loaded():
            return
        self.disable_picking()
        self.pick_mode = "boundary-loop"
        self.mode_status.setText("Interaction mode: boundary loop pick")
        self.plotter.enable_point_picking(
            callback=self.on_boundary_loop_picked,
            left_clicking=True,
            picker="point",
            show_message="Left click near an open boundary edge to select the nearest boundary loop.",
            show_point=False,
            use_picker=True,
            clear_on_no_selection=False,
        )
        self.append_log("Boundary loop picking is active. Left click near a hole edge to capture the nearest loop.")

    def on_boundary_loop_picked(self, point, picker):
        try:
            vertex_id = self.nearest_vertex_id(point, picker)
            loop, report = mesh_heal.find_boundary_loop_near_point(
                self.current_mesh,
                np.asarray(self.current_mesh.vertices[vertex_id], dtype=float),
            )
            self.loop_vertex_ids = [int(index) for index in loop.tolist()]
            self.clear_focus_overlay()
            self.append_log(
                f"Selected boundary loop with {report['boundary_loop_vertices']:,} vertices and length {report['loop_length']:.3f}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Boundary loop selection failed", str(exc))
            self.append_log(str(exc))
        finally:
            self.disable_picking()
            self.schedule_render_scene(reset_camera=False)

    def start_contour_pick(self):
        if not self.ensure_mesh_loaded():
            return
        self.disable_picking()
        self.pick_mode = "contour-pick"
        self.mode_status.setText("Interaction mode: contour picking")
        self.plotter.enable_point_picking(
            callback=self.on_contour_point_picked,
            left_clicking=True,
            picker="point",
            show_message="Left click vertices or edges in order around the patch you want to retriangulate.",
            show_point=False,
            use_picker=True,
            clear_on_no_selection=False,
        )
        self.append_log("Contour picking is active. Left click vertices or edges in order around the patch.")

    def on_contour_point_picked(self, point, picker):
        try:
            vertex_id = self.nearest_vertex_id(point, picker)
            vertex_point = np.asarray(self.current_mesh.vertices[vertex_id], dtype=float)
            vertex_distance = float(np.linalg.norm(vertex_point - np.asarray(point, dtype=float)))
            edge_pick = mesh_heal.find_nearest_edge(self.current_mesh, point)
            bounds = np.asarray(self.current_mesh.bounds, dtype=float)
            scale = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-6)
            use_edge = edge_pick["distance"] <= max(scale * 0.01, vertex_distance * 1.2)
            if use_edge:
                self.append_loop_edge(edge_pick["edge_vertex_ids"])
                self.append_log(
                    f"Added edge {edge_pick['edge_vertex_ids'][0]}-{edge_pick['edge_vertex_ids'][1]} to the contour ({len(self.loop_vertex_ids):,} vertices in path)"
                )
            else:
                self.append_loop_vertex(vertex_id)
                self.append_log(f"Added contour vertex {vertex_id} ({len(self.loop_vertex_ids):,} vertices in path)")
            self.clear_focus_overlay()
        except Exception as exc:
            QMessageBox.warning(self, "Contour picking failed", str(exc))
            self.append_log(str(exc))
            self.disable_picking()
        finally:
            self.schedule_render_scene(reset_camera=False)

    def delete_selected_faces(self, merge_eps: float, area_eps: float, dedup_decimals: int):
        if not self.ensure_mesh_loaded() or not self.selected_face_ids:
            return
        self.push_undo_state("delete selected faces")
        new_mesh, result = mesh_heal.remove_faces_by_index(
            self.current_mesh,
            sorted(self.selected_face_ids),
            merge_eps=merge_eps,
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
        )
        self.current_mesh = new_mesh
        self.selected_face_ids.clear()
        self.clear_focus_overlay()
        self.record_action("delete-selected-faces", result)
        self.append_log(f"Deleted {result['removed_faces']:,} selected triangles")
        self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
        self.render_scene(reset_camera=False)

    def drop_duplicate_faces(self, merge_eps: float, area_eps: float, dedup_decimals: int):
        if not self.ensure_mesh_loaded():
            return
        duplicate_count, _ = mesh_heal.detect_duplicate_faces(self.current_mesh, decimals=dedup_decimals)
        if duplicate_count == 0:
            self.append_log("No exact duplicate triangles were found")
            return
        self.push_undo_state("drop duplicate faces")
        before = mesh_heal.mesh_report(self.current_mesh, area_eps=area_eps).__dict__
        self.current_mesh = mesh_heal.remove_duplicate_faces(self.current_mesh, decimals=dedup_decimals)
        self.current_mesh = mesh_heal.merge_nearby_vertices(self.current_mesh, merge_eps=merge_eps)
        self.current_mesh = mesh_heal.remove_degenerate_faces(self.current_mesh, eps=area_eps)
        self.current_mesh.remove_unreferenced_vertices()
        result = {
            "duplicate_faces_removed": int(duplicate_count),
            "before": before,
            "after": mesh_heal.mesh_report(self.current_mesh, area_eps=area_eps).__dict__,
        }
        self.record_action("drop-duplicate-faces", result)
        self.append_log(f"Removed {duplicate_count:,} exact duplicate triangles")
        self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
        self.render_scene(reset_camera=False)

    def drop_degenerate_faces(self, area_eps: float):
        if not self.ensure_mesh_loaded():
            return
        degenerate_faces = mesh_heal.detect_degenerate_faces(self.current_mesh, eps=area_eps)
        if len(degenerate_faces) == 0:
            self.append_log("No degenerate triangles were found")
            return
        self.push_undo_state("drop degenerate faces")
        before = mesh_heal.mesh_report(self.current_mesh, area_eps=area_eps).__dict__
        self.current_mesh = mesh_heal.remove_degenerate_faces(self.current_mesh, eps=area_eps)
        self.current_mesh.remove_unreferenced_vertices()
        result = {
            "degenerate_faces_removed": int(len(degenerate_faces)),
            "before": before,
            "after": mesh_heal.mesh_report(self.current_mesh, area_eps=area_eps).__dict__,
        }
        self.record_action("drop-degenerate-faces", result)
        self.append_log(f"Removed {len(degenerate_faces):,} degenerate triangles")
        self.refresh_issue_browser(area_eps=area_eps)
        self.render_scene(reset_camera=False)

    def remove_overlaps(self, merge_eps: float, area_eps: float, dedup_decimals: int):
        if not self.ensure_mesh_loaded():
            return
        try:
            self.push_undo_state("auto remove overlaps")
            new_mesh, result = mesh_heal.rebuild_mesh_without_triangle_overlaps(
                self.current_mesh,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
            )
            self.current_mesh = new_mesh
            self.selected_face_ids.clear()
            self.clear_focus_overlay()
            self.record_action("auto-remove-overlaps", result)
            self.append_log(
                f"Auto overlap cleanup rebuilt the mesh from {result['input_faces']:,} to {result['output_faces']:,} faces"
            )
            self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
            self.render_scene(reset_camera=False)
        except Exception as exc:
            QMessageBox.warning(self, "Overlap cleanup failed", str(exc))
            self.append_log(str(exc))

    def rebuild_inside_loop(self, merge_eps: float, area_eps: float, dedup_decimals: int):
        if not self.ensure_mesh_loaded():
            return
        if len(self.loop_vertex_ids) < 3:
            QMessageBox.warning(self, "Incomplete loop", "Pick at least three contour vertices before rebuilding.")
            return
        try:
            self.push_undo_state("rebuild inside loop")
            loop_points = np.asarray(self.current_mesh.vertices[self.loop_vertex_ids], dtype=float)
            new_mesh, result = mesh_heal.rebuild_region_inside_closed_path(
                self.current_mesh,
                loop_points,
                merge_eps=merge_eps,
                area_eps=area_eps,
                dedup_decimals=dedup_decimals,
            )
            self.current_mesh = new_mesh
            self.selected_face_ids.clear()
            self.record_action("rebuild-inside-loop", result)
            self.append_log(
                f"Rebuilt region inside the contour: removed {result['removed_faces']:,} faces and added {result['added_faces']:,} faces"
            )
            self.loop_vertex_ids.clear()
            self.clear_focus_overlay()
            self.disable_picking()
            self.refresh_issue_browser(area_eps=area_eps, dedup_decimals=dedup_decimals)
            self.render_scene(reset_camera=False)
        except ValueError as exc:
            QMessageBox.warning(self, "Rebuild failed", str(exc))
            self.append_log(str(exc))
        except Exception as exc:
            QMessageBox.warning(self, "Rebuild failed", str(exc))
            self.append_log(str(exc))


class EditToolTab(QWidget):
    def __init__(self, editor: ManualRepairEditorWindow):
        super().__init__()
        self.editor = editor
        self.last_suggested_output = ""

        self.input_picker = FilePicker("Input", "Browse...", MESH_FILTER)
        self.output_picker = FilePicker("Output", "Save...", OUTPUT_FILTER, save_dialog=True)
        self.report_picker = FilePicker("Report", "Save...", JSON_FILTER, save_dialog=True)
        self.hints_picker = FilePicker("Hints", "Browse...", HINT_FILTER)
        self.intended_mesh_type_combo = create_intended_mesh_type_combo("auto")
        self.input_picker.line_edit.textChanged.connect(self.update_default_output_path)

        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)

        self.status_label = QLabel("No mesh loaded in the editor")
        self.status_label.setWordWrap(True)

        self.focus_editor_button = QPushButton("Focus Editor")
        self.focus_editor_button.clicked.connect(self.focus_editor)
        self.load_button = QPushButton("Load Mesh")
        self.load_button.clicked.connect(self.load_mesh)
        self.save_button = QPushButton("Save Mesh")
        self.save_button.clicked.connect(self.save_mesh)
        self.analyze_button = QPushButton("Analyze Errors")
        self.analyze_button.clicked.connect(self.analyze_mesh)
        self.load_hints_button = QPushButton("Load Hints")
        self.load_hints_button.clicked.connect(self.load_external_hints)
        self.clear_hints_button = QPushButton("Clear Hints")
        self.clear_hints_button.clicked.connect(self.editor.clear_external_issues)

        self.pick_triangle_button = QPushButton("Pick Triangles")
        self.pick_triangle_button.clicked.connect(self.start_triangle_pick)
        self.box_select_button = QPushButton("Box Select")
        self.box_select_button.clicked.connect(self.start_box_select)
        self.clear_faces_button = QPushButton("Clear Triangles")
        self.clear_faces_button.clicked.connect(self.editor.clear_face_selection)
        self.delete_faces_button = QPushButton("Delete Selected")
        self.delete_faces_button.clicked.connect(self.delete_selected_faces)

        self.pick_boundary_loop_button = QPushButton("Pick Boundary Loop")
        self.pick_boundary_loop_button.clicked.connect(self.editor.start_boundary_loop_pick)
        self.pick_contour_button = QPushButton("Pick Contour")
        self.pick_contour_button.clicked.connect(self.editor.start_contour_pick)
        self.clear_contour_button = QPushButton("Clear Contour")
        self.clear_contour_button.clicked.connect(self.editor.clear_loop_selection)
        self.rebuild_contour_button = QPushButton("Rebuild Inside Contour")
        self.rebuild_contour_button.clicked.connect(self.rebuild_inside_contour)

        self.remove_duplicates_button = QPushButton("Drop Duplicates")
        self.remove_duplicates_button.clicked.connect(self.drop_duplicates)
        self.remove_degenerate_button = QPushButton("Drop Degenerate")
        self.remove_degenerate_button.clicked.connect(self.drop_degenerate)
        self.remove_overlaps_button = QPushButton("Auto Remove Overlaps")
        self.remove_overlaps_button.clicked.connect(self.remove_overlaps)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.editor.undo_last_edit)
        self.reset_button = QPushButton("Reset Loaded Mesh")
        self.reset_button.clicked.connect(self.editor.reset_to_loaded_mesh)

        form_group = QGroupBox("Editing Session")
        form = QFormLayout(form_group)
        form.addRow(self.input_picker)
        form.addRow(self.output_picker)
        form.addRow(self.report_picker)
        form.addRow(self.hints_picker)
        form.addRow("Intended result", self.intended_mesh_type_combo)
        form.addRow("Merge epsilon", self.merge_eps_edit)
        form.addRow("Area epsilon", self.area_eps_edit)
        form.addRow("Dedup decimals", self.dedup_spin)

        self.intended_mesh_type_combo.setToolTip(
            "Declare whether the loaded mesh should be treated as a closed solid or an open surface during manual repair and later automated healing."
        )

        file_actions = QWidget()
        file_actions_layout = QHBoxLayout(file_actions)
        file_actions_layout.setContentsMargins(0, 0, 0, 0)
        file_actions_layout.addWidget(self.focus_editor_button)
        file_actions_layout.addWidget(self.load_button)
        file_actions_layout.addWidget(self.save_button)
        file_actions_layout.addWidget(self.analyze_button)
        form.addRow("File", file_actions)

        hint_actions = QWidget()
        hint_actions_layout = QHBoxLayout(hint_actions)
        hint_actions_layout.setContentsMargins(0, 0, 0, 0)
        hint_actions_layout.addWidget(self.load_hints_button)
        hint_actions_layout.addWidget(self.clear_hints_button)
        form.addRow("Hints", hint_actions)

        triangle_actions = QWidget()
        triangle_layout = QHBoxLayout(triangle_actions)
        triangle_layout.setContentsMargins(0, 0, 0, 0)
        triangle_layout.addWidget(self.pick_triangle_button)
        triangle_layout.addWidget(self.box_select_button)
        triangle_layout.addWidget(self.clear_faces_button)
        triangle_layout.addWidget(self.delete_faces_button)
        form.addRow("Triangles", triangle_actions)

        contour_actions = QWidget()
        contour_layout = QHBoxLayout(contour_actions)
        contour_layout.setContentsMargins(0, 0, 0, 0)
        contour_layout.addWidget(self.pick_boundary_loop_button)
        contour_layout.addWidget(self.pick_contour_button)
        contour_layout.addWidget(self.clear_contour_button)
        contour_layout.addWidget(self.rebuild_contour_button)
        form.addRow("Contour", contour_actions)

        cleanup_actions = QWidget()
        cleanup_layout = QHBoxLayout(cleanup_actions)
        cleanup_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.addWidget(self.remove_duplicates_button)
        cleanup_layout.addWidget(self.remove_degenerate_button)
        cleanup_layout.addWidget(self.remove_overlaps_button)
        cleanup_layout.addWidget(self.undo_button)
        cleanup_layout.addWidget(self.reset_button)
        form.addRow("Repair", cleanup_actions)

        note = QLabel(
            "Manual editing is the default workflow. Use Pick Triangles for point-click face toggling, Box Select for bulk selection, and Pick Contour to click vertices or edges before rebuilding the triangulation inside a closed patch. Import a JSON or DXF hint file when Leapfrog reports overlaps or non-manifold locations that the local detector did not flag."
        )
        note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(form_group)
        layout.addWidget(note)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.editor.state_changed.connect(self.update_from_editor)
        self.update_from_editor()

    def update_default_output_path(self, input_path_text: str):
        if not input_path_text:
            self.last_suggested_output = ""
            return
        input_path = Path(input_path_text)
        suggested_output = str(input_path.with_name(f"{input_path.stem}_manual_repair{input_path.suffix}"))
        current_output = self.output_picker.text()
        if current_output and current_output != self.last_suggested_output:
            return
        self.output_picker.setText(suggested_output)
        self.last_suggested_output = suggested_output

    def current_settings(self) -> tuple[float, float, int]:
        return float(self.merge_eps_edit.text()), float(self.area_eps_edit.text()), int(self.dedup_spin.value())

    def require_settings(self) -> tuple[float, float, int] | None:
        try:
            return self.current_settings()
        except ValueError:
            QMessageBox.warning(self, "Invalid numeric input", "Merge epsilon and area epsilon must be valid numbers.")
            return None

    def focus_editor(self):
        self.editor.showNormal()
        self.editor.raise_()
        self.editor.activateWindow()

    def load_mesh(self):
        input_path_text = self.input_picker.text()
        if not input_path_text:
            QMessageBox.warning(self, "Missing path", "Select an input mesh before loading.")
            return
        settings = self.require_settings()
        if settings is None:
            return
        _, area_eps, dedup_decimals = settings
        self.editor.load_mesh_from_path(
            Path(input_path_text),
            area_eps=area_eps,
            dedup_decimals=dedup_decimals,
            intended_mesh_type=str(self.intended_mesh_type_combo.currentData()),
        )
        self.focus_editor()

    def save_mesh(self):
        output_path_text = self.output_picker.text()
        if not output_path_text:
            QMessageBox.warning(self, "Missing path", "Choose an output path before saving.")
            return
        report_path = Path(self.report_picker.text()) if self.report_picker.text() else None
        self.editor.save_mesh_to_path(Path(output_path_text), report_path=report_path)

    def analyze_mesh(self):
        settings = self.require_settings()
        if settings is None:
            return
        _, area_eps, dedup_decimals = settings
        self.editor.analyze_current_mesh(area_eps=area_eps, dedup_decimals=dedup_decimals)

    def load_external_hints(self):
        hints_path_text = self.hints_picker.text()
        if not hints_path_text:
            QMessageBox.warning(self, "Missing path", "Choose a JSON or DXF hints file before importing external repair hints.")
            return
        settings = self.require_settings()
        if settings is None:
            return
        merge_eps, _, _ = settings
        self.editor.load_external_issues_from_path(Path(hints_path_text), merge_eps=merge_eps)
        self.focus_editor()

    def start_triangle_pick(self):
        self.editor.start_triangle_pick()
        self.focus_editor()

    def start_box_select(self):
        self.editor.start_rectangle_face_selection()
        self.focus_editor()

    def delete_selected_faces(self):
        settings = self.require_settings()
        if settings is None:
            return
        merge_eps, area_eps, dedup_decimals = settings
        self.editor.delete_selected_faces(merge_eps=merge_eps, area_eps=area_eps, dedup_decimals=dedup_decimals)

    def rebuild_inside_contour(self):
        settings = self.require_settings()
        if settings is None:
            return
        merge_eps, area_eps, dedup_decimals = settings
        self.editor.rebuild_inside_loop(merge_eps=merge_eps, area_eps=area_eps, dedup_decimals=dedup_decimals)

    def drop_duplicates(self):
        settings = self.require_settings()
        if settings is None:
            return
        merge_eps, area_eps, dedup_decimals = settings
        self.editor.drop_duplicate_faces(merge_eps=merge_eps, area_eps=area_eps, dedup_decimals=dedup_decimals)

    def drop_degenerate(self):
        settings = self.require_settings()
        if settings is None:
            return
        _, area_eps, _ = settings
        self.editor.drop_degenerate_faces(area_eps=area_eps)

    def remove_overlaps(self):
        settings = self.require_settings()
        if settings is None:
            return
        merge_eps, area_eps, dedup_decimals = settings
        self.editor.remove_overlaps(merge_eps=merge_eps, area_eps=area_eps, dedup_decimals=dedup_decimals)

    def update_from_editor(self):
        has_mesh = self.editor.has_mesh()
        self.save_button.setEnabled(has_mesh)
        self.analyze_button.setEnabled(has_mesh)
        self.pick_triangle_button.setEnabled(has_mesh)
        self.box_select_button.setEnabled(has_mesh)
        self.clear_faces_button.setEnabled(has_mesh and self.editor.selected_face_count() > 0)
        self.delete_faces_button.setEnabled(has_mesh and self.editor.selected_face_count() > 0)
        self.pick_boundary_loop_button.setEnabled(has_mesh)
        self.pick_contour_button.setEnabled(has_mesh)
        self.clear_contour_button.setEnabled(has_mesh and self.editor.contour_vertex_count() > 0)
        self.rebuild_contour_button.setEnabled(has_mesh and self.editor.contour_vertex_count() >= 3)
        self.load_hints_button.setEnabled(has_mesh)
        self.clear_hints_button.setEnabled(has_mesh and self.editor.external_issue_count() > 0)
        self.remove_duplicates_button.setEnabled(has_mesh)
        self.remove_degenerate_button.setEnabled(has_mesh)
        self.remove_overlaps_button.setEnabled(has_mesh)
        self.undo_button.setEnabled(self.editor.can_undo())
        self.reset_button.setEnabled(has_mesh)

        if not has_mesh:
            self.status_label.setText("No mesh loaded in the editor")
            return
        self.status_label.setText(
            f"Mesh loaded as {describe_intended_mesh_type(self.editor.loaded_mesh_intended_type).lower()}. {self.editor.issue_count():,} issue entries ({self.editor.external_issue_count():,} imported) | {self.editor.selected_face_count():,} selected triangles | {self.editor.contour_vertex_count():,} contour vertices"
        )


class ToolPanelWindow(QMainWindow):
    def __init__(self, editor: ManualRepairEditorWindow):
        super().__init__()
        self.editor = editor
        self.setWindowTitle("Mesh Heal Tools")
        self.resize(1040, 900)

        tabs = QTabWidget()
        tabs.addTab(EditToolTab(editor), "Edit")
        tabs.addTab(HealTab(None), "Heal")
        tabs.addTab(SurfaceShellBatchTab(), "Surface Shell Batch")
        tabs.addTab(AutoresearchTab(None), "Autoresearch")
        tabs.addTab(BooleanTab(None), "Boolean")
        tabs.setCurrentIndex(0)
        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        if self.editor is not None:
            self.editor.close()
        super().closeEvent(event)


class GuidedHealWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Heal")
        self.resize(1680, 980)

        central = QWidget()
        layout = QVBoxLayout(central)

        controls_row = QHBoxLayout()
        controls_row.addStretch(1)
        self.preview_toggle_button = QPushButton("Hide Preview")
        self.preview_toggle_button.setCheckable(True)
        self.preview_toggle_button.toggled.connect(self.toggle_preview_panel)
        controls_row.addWidget(self.preview_toggle_button)
        layout.addLayout(controls_row)

        self.preview_pane = PreviewPane()
        self.preview_pane.setMinimumWidth(520)
        self.preview_container = QWidget()
        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_pane)

        tabs = QTabWidget()
        tabs.addTab(HealTab(self.preview_pane), "Heal")
        tabs.addTab(SurfaceShellBatchTab(), "Surface Shell Batch")
        tabs.addTab(BooleanTab(self.preview_pane), "Boolean")
        tabs.setCurrentIndex(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(tabs)
        self.main_splitter.addWidget(self.preview_container)
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setSizes([1040, 640])
        layout.addWidget(self.main_splitter, 1)
        self.setCentralWidget(central)

    def toggle_preview_panel(self, collapsed: bool):
        self.preview_container.setVisible(not collapsed)
        if collapsed:
            self.preview_toggle_button.setText("Show Preview")
            self.main_splitter.setSizes([1, 0])
        else:
            self.preview_toggle_button.setText("Hide Preview")
            self.main_splitter.setSizes([1040, 640])


def main() -> None:
    configure_qt_opengl_backend()
    app = QApplication(sys.argv)
    window = GuidedHealWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()