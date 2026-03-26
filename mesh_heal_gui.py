import json
import multiprocessing
from queue import Empty
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QListWidget,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import mesh_heal

try:
    from pyvistaqt import QtInteractor
except Exception:
    QtInteractor = None

try:
    import pyvista as pv
except Exception:
    pv = None


MESH_FILTER = "Mesh Files (*.dxf *.msh *.stl *.obj *.ply *.00t);;All Files (*.*)"
OUTPUT_FILTER = "DXF Files (*.dxf);;Leapfrog Mesh Files (*.msh);;Mesh Files (*.stl *.obj *.ply *.vtk);;Maptek Vulcan Files (*.00t);;All Files (*.*)"
JSON_FILTER = "JSON Files (*.json);;All Files (*.*)"


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
    failed = Signal(str)
    finished = Signal()

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    @Slot()
    def run(self):
        try:
            payload = mesh_heal.prepare_preview_payload(self.path)
            self.completed.emit(payload)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


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

    def choose_path(self):
        if self.save_dialog:
            path, _ = QFileDialog.getSaveFileName(self, self.label.text(), self.line_edit.text(), self.file_filter)
            if path:
                self.selected_paths = []
                self.line_edit.setText(path)
                self.line_edit.setToolTip(path)
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
                self.line_edit.setToolTip(path)
                self.path_changed.emit(path)

    def set_paths(self, paths: list[str]) -> None:
        self.selected_paths = [path for path in paths if path]
        if not self.selected_paths:
            self.line_edit.clear()
            self.line_edit.setToolTip("")
            return
        if len(self.selected_paths) == 1:
            display_text = self.selected_paths[0]
        else:
            names = ", ".join(Path(path).name for path in self.selected_paths)
            display_text = f"{len(self.selected_paths)} files selected: {names}"
        self.line_edit.setText(display_text)
        self.line_edit.setToolTip("\n".join(self.selected_paths))

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
        self.line_edit.setToolTip(value)

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

        layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Preview is manual and safety-capped. Large files are decimated before rendering."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.preview_status = QLabel("No preview loaded")
        layout.addWidget(self.preview_status)

        self.preview_progress = QProgressBar()
        self.preview_progress.setRange(0, 1)
        self.preview_progress.setValue(0)
        layout.addWidget(self.preview_progress)

        if QtInteractor is None:
            self.plotter = None
            self.viewer_placeholder = QLabel(
                "3D preview backend is not installed. Install pyvistaqt to enable embedded preview."
            )
            self.viewer_placeholder.setWordWrap(True)
            self.viewer_placeholder.setFrameShape(QFrame.StyledPanel)
            self.viewer_placeholder.setMinimumHeight(320)
            layout.addWidget(self.viewer_placeholder, 1)
        else:
            self.plotter = QtInteractor(self)
            self.plotter.setMinimumHeight(320)
            self.plotter.set_background("#fafafa")
            layout.addWidget(self.plotter.interactor, 1)

        self.preview_log = QPlainTextEdit()
        self.preview_log.setReadOnly(True)
        self.preview_log.setPlaceholderText("Preview log")
        self.preview_log.setMaximumBlockCount(200)
        layout.addWidget(self.preview_log)

    def available(self) -> bool:
        return self.plotter is not None

    def append_log(self, message: str):
        self.preview_log.appendPlainText(message)

    def request_preview(self, path_text: str, label: str):
        if not path_text:
            QMessageBox.warning(self, "Missing path", f"Select a file before previewing {label.lower()}.")
            return
        if not self.available():
            QMessageBox.information(
                self,
                "Preview unavailable",
                "Embedded preview needs pyvistaqt. The rest of the GUI still works without it.",
            )
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Preview busy", "Wait for the current preview to finish loading.")
            return

        path = Path(path_text)
        self.preview_status.setText(f"Loading {label}: {path.name}")
        self.append_log(f"Loading preview for {path}")
        self.preview_progress.setRange(0, 0)

        self.thread = QThread(self)
        self.worker = PreviewWorker(path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(lambda payload: self.show_preview(payload, label))
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

    def show_preview(self, payload: mesh_heal.PreviewPayload, label: str):
        faces = np.hstack([
            np.full((len(payload.faces), 1), 3, dtype=np.int64),
            payload.faces,
        ]).reshape(-1)
        if pv is None:
            raise RuntimeError("pyvista is required to render embedded previews.")
        poly = pv.PolyData(payload.vertices, faces)
        self.plotter.clear()
        self.plotter.add_mesh(
            poly,
            color="#c99a52",
            smooth_shading=False,
            show_edges=False,
            lighting=True,
        )
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        note = "decimated" if payload.decimated else "full-resolution"
        self.preview_status.setText(
            f"{label}: {Path(payload.source).name} | original {payload.original_faces:,} faces -> preview "
            f"{payload.preview_faces:,} faces ({note})"
        )
        self.append_log(self.preview_status.text())

    def show_preview_error(self, details: str):
        self.preview_status.setText("Preview failed")
        self.append_log(details)
        QMessageBox.warning(self, "Preview failed", details)


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
        for control in self.controls:
            control.setEnabled(not running)
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
    def __init__(self, preview_pane: PreviewPane):
        super().__init__()
        self.preview_pane = preview_pane
        self.last_suggested_output = ""
        self.healed_output_ready = False

        self.input_picker = FilePicker("Input", "Browse...", MESH_FILTER)
        self.output_picker = FilePicker("Output", "Save...", OUTPUT_FILTER, save_dialog=True)
        self.report_picker = FilePicker("Report", "Save...", JSON_FILTER, save_dialog=True)
        self.input_picker.line_edit.textChanged.connect(self.update_default_output_path)
        self.input_picker.line_edit.textChanged.connect(self.refresh_preview_buttons)
        self.output_picker.line_edit.textChanged.connect(self.invalidate_output_preview)

        self.merge_eps_edit = QLineEdit("1e-8")
        self.area_eps_edit = QLineEdit("1e-12")
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(0, 16)
        self.dedup_spin.setValue(8)
        self.rebuild_triangles_check = QCheckBox("Rebuild triangles before healing")
        self.localized_intersection_repair_check = QCheckBox("Experimental localized self-intersection repair")
        self.make_watertight_check = QCheckBox("Attempt to fill holes and make watertight")
        self.return_surface_after_watertight_check = QCheckBox("Return repaired surface after hole filling")
        self.advanced_backend_combo = QComboBox()
        self.advanced_backend_combo.addItem("None", "none")
        self.advanced_backend_combo.addItem("OpenMeshCraft + FastTetWild", "omc-ftetwild")
        self.advanced_backend_combo.addItem("OpenMeshCraft + OpenMeshCraft CDT", "omc-cdt")
        self.exact_arrangements_exe_edit = QLineEdit()
        self.exact_arrangements_exe_edit.setPlaceholderText("Optional OpenMeshCraft-Arrangements executable")
        self.tetra_backend_exe_edit = QLineEdit()
        self.tetra_backend_exe_edit.setPlaceholderText("Optional tetra backend executable")
        self.run_button = QPushButton("Run Heal")
        self.run_button.clicked.connect(self.run_task)
        self.preview_input_button = QPushButton("Preview Input")
        self.preview_input_button.clicked.connect(
            lambda: self.preview_pane.request_preview(self.input_picker.text(), "Heal input")
        )
        self.preview_output_button = QPushButton("Preview Output")
        self.preview_output_button.clicked.connect(
            lambda: self.preview_pane.request_preview(self.output_picker.text(), "Heal output")
        )

        preview_buttons = QWidget()
        preview_buttons_layout = QHBoxLayout(preview_buttons)
        preview_buttons_layout.setContentsMargins(0, 0, 0, 0)
        preview_buttons_layout.addWidget(self.preview_input_button)
        preview_buttons_layout.addWidget(self.preview_output_button)

        options_group = QGroupBox("Healing")
        form = QFormLayout(options_group)
        form.addRow(self.input_picker)
        form.addRow(self.output_picker)
        form.addRow(self.report_picker)
        form.addRow("Merge epsilon", self.merge_eps_edit)
        form.addRow("Area epsilon", self.area_eps_edit)
        form.addRow("Dedup decimals", self.dedup_spin)
        form.addRow(self.rebuild_triangles_check)
        form.addRow(self.localized_intersection_repair_check)
        form.addRow(self.make_watertight_check)
        form.addRow(self.return_surface_after_watertight_check)
        form.addRow("Advanced backend", self.advanced_backend_combo)
        form.addRow("Exact backend exe", self.exact_arrangements_exe_edit)
        form.addRow("Tetra backend exe", self.tetra_backend_exe_edit)
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
            self.merge_eps_edit,
            self.area_eps_edit,
            self.dedup_spin,
            self.rebuild_triangles_check,
            self.localized_intersection_repair_check,
            self.make_watertight_check,
            self.return_surface_after_watertight_check,
            self.advanced_backend_combo,
            self.exact_arrangements_exe_edit,
            self.tetra_backend_exe_edit,
            self.preview_input_button,
            self.preview_output_button,
            self.run_button,
        ]
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
        input_path = self.input_picker.text()
        input_ready = bool(input_path) and Path(input_path).exists()
        output_ready = self.healed_output_ready and bool(self.output_picker.text())
        self.preview_input_button.setEnabled(input_ready)
        self.preview_output_button.setEnabled(output_ready)

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
            kwargs = {
                "input_path": Path(input_path),
                "output_path": Path(output_path),
                "report_path": Path(self.report_picker.text()) if self.report_picker.text() else None,
                "merge_eps": float(self.merge_eps_edit.text()),
                "area_eps": float(self.area_eps_edit.text()),
                "dedup_decimals": int(self.dedup_spin.value()),
                "rebuild_triangles": bool(self.rebuild_triangles_check.isChecked()),
                "localized_intersection_repair": bool(self.localized_intersection_repair_check.isChecked()),
                "make_watertight": bool(self.make_watertight_check.isChecked()),
                "return_surface_after_watertight": bool(self.return_surface_after_watertight_check.isChecked()),
                "advanced_backend": str(self.advanced_backend_combo.currentData()),
                "exact_arrangements_executable": Path(self.exact_arrangements_exe_edit.text()) if self.exact_arrangements_exe_edit.text().strip() else None,
                "tetra_backend_executable": Path(self.tetra_backend_exe_edit.text()) if self.tetra_backend_exe_edit.text().strip() else None,
            }
        except ValueError:
            QMessageBox.warning(self, "Invalid numeric input", "Merge epsilon and area epsilon must be valid numbers.")
            return

        self.healed_output_ready = False
        self.start_worker(mesh_heal.run_heal_pipeline, kwargs)


class BooleanTab(BaseOperationTab):
    def __init__(self, preview_pane: PreviewPane):
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
        self.preview_input_button = QPushButton("Preview Selected Input")
        self.preview_input_button.clicked.connect(self.preview_input)
        self.preview_output_button = QPushButton("Preview Output")
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
            self.preview_input_button,
            self.preview_output_button,
            self.run_button,
        ]
        self.controls.extend(self.operation_checks.values())

    def selected_operations(self) -> list[str]:
        return [operation for operation, checkbox in self.operation_checks.items() if checkbox.isChecked()]

    def preview_output(self):
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Heal")
        self.resize(1480, 820)

        preview_pane = PreviewPane()
        tabs = QTabWidget()
        tabs.addTab(HealTab(preview_pane), "Heal")
        tabs.addTab(BooleanTab(preview_pane), "Boolean")

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(tabs, 3)
        layout.addWidget(preview_pane, 2)
        self.setCentralWidget(container)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()