#!/usr/bin/env python3
"""PyQt GUI to browse images and run YOLO predictions with adjustable confidence."""

from __future__ import annotations

import csv
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path

try:
    from PyQt5.QtCore import QLibraryInfo, QPoint, QProcess, QProcessEnvironment, QRect, QSize, QTimer, Qt
    from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSizePolicy,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    print(f"PyQt5 is required. Install with: {sys.executable} -m pip install PyQt5")
    raise SystemExit(f"Failed to import PyQt5: {exc}")

# Force Qt plugin lookup to PyQt5's plugins to avoid cv2 Qt plugin conflicts.
_qt_plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
if _qt_plugins:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins
    os.environ["QT_PLUGIN_PATH"] = _qt_plugins
    # os.environ["QT_QPA_PLATFORM"] = "wayland"  # or "xcb" on X11, "windows" on Windows, etc.

import cv2
try:
    from ultralytics import YOLO
except Exception as exc:
    msg = str(exc)
    if "mpf_ln" in msg and "mpmath" in msg:
        print("Detected broken mpmath/sympy mix in .venv-rocm.")
        print(
            "Fix with:\n"
            "rm -rf .venv-rocm/lib/python3.14/site-packages/mpmath "
            ".venv-rocm/lib/python3.14/site-packages/mpmath-*.dist-info"
        )
    raise

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    FigureCanvas = None
    Figure = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
TRAINABLE_MODEL_SUFFIXES = {".pt", ".yaml", ".yml"}


class FloatingTrainingPlotWindow(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(980, 560)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.loss_ax = self.figure.add_subplot(1, 2, 1)
            self.metric_ax = self.figure.add_subplot(1, 2, 2)
            layout.addWidget(self.canvas, 1)
            self.clear_plot()
        else:
            self.figure = None
            self.canvas = None
            self.loss_ax = None
            self.metric_ax = None
            layout.addWidget(QLabel("Matplotlib not available. Plot window disabled."))

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def clear_plot(self) -> None:
        if not HAS_MATPLOTLIB or self.loss_ax is None or self.metric_ax is None:
            return
        self.loss_ax.clear()
        self.metric_ax.clear()
        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.3)
        self.metric_ax.set_title("Validation Metrics")
        self.metric_ax.set_xlabel("Epoch")
        self.metric_ax.set_ylabel("Score")
        self.metric_ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


class PolygonAnnotationCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image: QImage | None = None
        self._draw_rect = QRect()
        self._annotations: list[dict] = []
        self._current_points: list[tuple[float, float]] = []
        self._current_class_id = 0
        self.on_changed = None

    def set_current_class(self, class_id: int) -> None:
        self._current_class_id = max(0, int(class_id))

    def set_image_and_annotations(self, image_path: Path, annotations: list[dict]) -> bool:
        frame = cv2.imread(str(image_path))
        if frame is None:
            self._image = None
            self._annotations = []
            self._current_points = []
            self.update()
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        self._image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()
        self._annotations = [
            {"class_id": int(a.get("class_id", 0)), "points": [tuple(p) for p in a.get("points", [])]}
            for a in annotations
        ]
        self._current_points = []
        self.update()
        return True

    def clear_current_polygon(self) -> None:
        self._current_points = []
        self.update()

    def remove_last_current_point(self) -> bool:
        if not self._current_points:
            return False
        self._current_points.pop()
        self.update()
        return True

    def remove_last_polygon(self) -> None:
        if self._annotations:
            self._annotations.pop()
            self._emit_changed()
            self.update()

    def finish_current_polygon(self) -> bool:
        if len(self._current_points) < 3:
            return False
        self._annotations.append({"class_id": self._current_class_id, "points": list(self._current_points)})
        self._current_points = []
        self._emit_changed()
        self.update()
        return True

    def annotations(self) -> list[dict]:
        return [
            {"class_id": int(a["class_id"]), "points": [tuple(p) for p in a["points"]]}
            for a in self._annotations
        ]

    def sizeHint(self) -> QSize:
        return QSize(960, 600)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        if event.button() == Qt.LeftButton:
            point = self._widget_to_image(event.pos())
            if point is not None:
                if len(self._current_points) >= 3:
                    first_widget = self._image_to_widget(self._current_points[0])
                    dx = event.pos().x() - first_widget.x()
                    dy = event.pos().y() - first_widget.y()
                    if (dx * dx + dy * dy) <= 100:
                        self.finish_current_polygon()
                        return
                self._current_points.append(point)
                self.update()
        elif event.button() == Qt.RightButton:
            if self._current_points:
                self._current_points.pop()
                self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.finish_current_polygon()

    def paintEvent(self, event) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._image is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a folder and choose an image to label")
            return

        img_w = self._image.width()
        img_h = self._image.height()
        area = self.rect()
        scale = min(area.width() / img_w, area.height() / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        x = (area.width() - draw_w) // 2
        y = (area.height() - draw_h) // 2
        self._draw_rect = QRect(x, y, draw_w, draw_h)
        painter.drawImage(self._draw_rect, self._image)

        for poly in self._annotations:
            self._draw_polygon(painter, poly.get("points", []), int(poly.get("class_id", 0)))
        self._draw_polygon(painter, self._current_points, self._current_class_id, in_progress=True)

    def _color_for_class(self, class_id: int):
        palette = [
            Qt.green,
            Qt.red,
            Qt.yellow,
            Qt.cyan,
            Qt.magenta,
            Qt.blue,
            Qt.white,
        ]
        return palette[class_id % len(palette)]

    def _draw_polygon(self, painter: QPainter, points: list[tuple[float, float]], class_id: int, in_progress=False) -> None:
        if self._image is None or len(points) < 2:
            return
        color = self._color_for_class(class_id)
        pen = QPen(color, 2)
        painter.setPen(pen)
        widget_points = [self._image_to_widget(p) for p in points]
        for i in range(len(widget_points) - 1):
            painter.drawLine(widget_points[i], widget_points[i + 1])
        if not in_progress:
            painter.drawLine(widget_points[-1], widget_points[0])
        for p in widget_points:
            painter.drawEllipse(p, 3, 3)

    def _widget_to_image(self, point: QPoint) -> tuple[float, float] | None:
        if self._image is None or self._draw_rect.width() <= 0 or self._draw_rect.height() <= 0:
            return None
        if not self._draw_rect.contains(point):
            return None
        rel_x = (point.x() - self._draw_rect.x()) / self._draw_rect.width()
        rel_y = (point.y() - self._draw_rect.y()) / self._draw_rect.height()
        x = rel_x * self._image.width()
        y = rel_y * self._image.height()
        return (max(0.0, min(x, self._image.width() - 1)), max(0.0, min(y, self._image.height() - 1)))

    def _image_to_widget(self, point: tuple[float, float]) -> QPoint:
        if self._image is None:
            return QPoint(0, 0)
        x, y = point
        draw_x = self._draw_rect.x() + int((x / self._image.width()) * self._draw_rect.width())
        draw_y = self._draw_rect.y() + int((y / self._image.height()) * self._draw_rect.height())
        return QPoint(draw_x, draw_y)

    def _emit_changed(self) -> None:
        if callable(self.on_changed):
            self.on_changed()


class ModelViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Criarte Model Studio")
        self.resize(1300, 850)

        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.image_paths: list[Path] = []
        self.index = -1  # image index for folder mode
        self.source_mode = "none"  # one of: none, images, video
        self.video_path: Path | None = None
        self.video_capture = None
        self.video_frame_count = 0
        self.video_frame_idx = -1
        self.video_fps = 30.0
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._advance_video_loop)
        self.current_qimage: QImage | None = None
        self.train_process: QProcess | None = None
        self.train_plot_timer = QTimer(self)
        self.train_plot_timer.timeout.connect(self._update_optimize_training_plot)
        self.train_results_csv: Path | None = None
        self.model_train_process: QProcess | None = None
        self.model_export_process: QProcess | None = None
        self.converter_process: QProcess | None = None
        self.model_train_plot_timer = QTimer(self)
        self.model_train_plot_timer.timeout.connect(self._update_model_training_plot)
        self.model_train_results_csv: Path | None = None
        self.optimize_plot_window: FloatingTrainingPlotWindow | None = None
        self.model_plot_window: FloatingTrainingPlotWindow | None = None
        self.plot_windows: list[FloatingTrainingPlotWindow] = []
        self.label_image_paths: list[Path] = []
        self.label_index = -1
        self.label_annotations: dict[str, list[dict]] = {}
        self.label_class_names: list[str] = ["object"]

        self._apply_professional_theme()
        self._build_ui()
        self._apply_argument_tooltips()
        self._refresh_nav_buttons()

    def _apply_professional_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #323232;
                color: #EEEEEE;
                font-family: "Segoe UI", "Noto Sans", sans-serif;
                font-size: 13px;
            }
            QMainWindow {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #323232,
                    stop: 1 #323232
                );
            }
            QScrollArea {
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QWidget#headerCard {
                background-color: rgba(50, 50, 50, 215);
                border: 1px solid #CF0A0A;
                border-radius: 18px;
            }
            QLabel#headerTitle {
                font-size: 20px;
                font-weight: 700;
                color: #EEEEEE;
            }
            QLabel#headerSubtitle {
                font-size: 12px;
                color: #DC5F00;
            }
            QTabWidget::pane {
                border: 1px solid #CF0A0A;
                border-radius: 14px;
                background: rgba(50, 50, 50, 220);
                top: -1px;
            }
            QTabBar::tab {
                background: rgba(50, 50, 50, 200);
                color: #EEEEEE;
                padding: 9px 16px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border: 1px solid #DC5F00;
            }
            QTabBar::tab:selected {
                background: #CF0A0A;
                color: #EEEEEE;
                font-weight: 600;
            }
            QPushButton {
                background-color: rgba(50, 50, 50, 210);
                border: 1px solid #DC5F00;
                border-radius: 11px;
                padding: 8px 12px;
                font-weight: 600;
                color: #EEEEEE;
            }
            QPushButton:hover {
                background-color: #DC5F00;
                color: #323232;
            }
            QPushButton#primaryButton {
                background-color: #CF0A0A;
                border: 1px solid #DC5F00;
                color: #EEEEEE;
            }
            QPushButton#primaryButton:hover {
                background-color: #DC5F00;
                color: #323232;
            }
            QPushButton#dangerButton {
                background-color: #CF0A0A;
                border: 1px solid #DC5F00;
                color: #EEEEEE;
            }
            QPushButton#dangerButton:hover {
                background-color: #DC5F00;
                color: #323232;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background-color: rgba(50, 50, 50, 220);
                border: 1px solid #DC5F00;
                border-radius: 10px;
                padding: 6px;
                color: #EEEEEE;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #CF0A0A;
            }
            QCheckBox {
                spacing: 8px;
                color: #EEEEEE;
            }
            QSlider::groove:horizontal {
                border-radius: 5px;
                height: 6px;
                background: #323232;
                border: 1px solid #DC5F00;
            }
            QSlider::handle:horizontal {
                background: #CF0A0A;
                border: 1px solid #DC5F00;
                width: 15px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #323232;
                width: 12px;
                margin: 3px;
            }
            QScrollBar::handle:vertical {
                background: #CF0A0A;
                border: 1px solid #DC5F00;
                border-radius: 6px;
                min-height: 24px;
            }
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        header = QWidget()
        header.setObjectName("headerCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(2)
        header_title = QLabel("Criarte Model Studio")
        header_title.setObjectName("headerTitle")
        header_title.setAlignment(Qt.AlignCenter)
        header_subtitle = QLabel("Inference, pseudo-label optimization, and OBB training")
        header_subtitle.setObjectName("headerSubtitle")
        header_subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_title, alignment=Qt.AlignHCenter)
        header_layout.addWidget(header_subtitle, alignment=Qt.AlignHCenter)
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.inference_tab = self._create_scroll_tab()
        self.labeling_tab = self._create_scroll_tab()
        self.optimize_tab = self._create_scroll_tab()
        self.training_tab = self._create_scroll_tab()
        self.converter_tab = self._create_scroll_tab()
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.labeling_tab, "Labeling")
        self.tabs.addTab(self.optimize_tab, "Optimize")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.converter_tab, "Converter")

        self._build_inference_tab()
        self._build_labeling_tab()
        self._build_optimize_tab()
        self._build_training_tab()
        self._build_converter_tab()

    def _create_scroll_tab(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        container_layout.addWidget(scroll)

        container._scroll_content = content  # type: ignore[attr-defined]
        return container

    @staticmethod
    def _tab_content(tab_container: QWidget) -> QWidget:
        return tab_container._scroll_content  # type: ignore[attr-defined]

    def _build_inference_tab(self) -> None:
        layout = QVBoxLayout(self._tab_content(self.inference_tab))
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        layout.addLayout(controls)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls.addWidget(self.load_model_btn)

        self.load_folder_btn = QPushButton("Load Image Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        controls.addWidget(self.load_folder_btn)

        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        controls.addWidget(self.load_video_btn)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_image)
        controls.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)
        controls.addWidget(self.next_btn)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        self.conf_slider.sliderReleased.connect(self._rerun_current)
        controls.addWidget(QLabel("Confidence"))
        controls.addWidget(self.conf_slider)

        self.conf_label = QLabel("25%")
        self.conf_label.setFixedWidth(50)
        controls.addWidget(self.conf_label)

        self.model_label = QLabel("Model: not loaded")
        layout.addWidget(self.model_label)

        self.info_label = QLabel("Image: none")
        layout.addWidget(self.info_label)

        self.classification_label = QLabel("Classification: -")
        self.classification_label.setWordWrap(True)
        layout.addWidget(self.classification_label)

        self.image_label = QLabel("Load an image folder or a video to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: rgba(50, 50, 50, 220); "
            "color: #EEEEEE; "
            "border-radius: 14px; "
            "border: 1px solid #CF0A0A;"
        )
        self.image_label.setMinimumHeight(0)
        # Prevent pixmap size hints from growing the scrollable tab content on each frame/image update.
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.image_label, 1)

    def _build_labeling_tab(self) -> None:
        layout = QVBoxLayout(self._tab_content(self.labeling_tab))
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self.label_load_folder_btn = QPushButton("Load Image Folder")
        self.label_load_folder_btn.clicked.connect(self._label_load_folder)
        controls.addWidget(self.label_load_folder_btn)
        self.label_prev_btn = QPushButton("Previous")
        self.label_prev_btn.clicked.connect(self._label_prev_image)
        controls.addWidget(self.label_prev_btn)
        self.label_next_btn = QPushButton("Next")
        self.label_next_btn.clicked.connect(self._label_next_image)
        controls.addWidget(self.label_next_btn)
        self.label_finish_poly_btn = QPushButton("Finish Polygon")
        self.label_finish_poly_btn.clicked.connect(self._label_finish_polygon)
        controls.addWidget(self.label_finish_poly_btn)
        self.label_undo_point_btn = QPushButton("Undo Point")
        self.label_undo_point_btn.clicked.connect(self._label_undo_point)
        controls.addWidget(self.label_undo_point_btn)
        self.label_undo_poly_btn = QPushButton("Delete Last Polygon")
        self.label_undo_poly_btn.clicked.connect(self._label_delete_last_polygon)
        controls.addWidget(self.label_undo_poly_btn)
        self.label_clear_poly_btn = QPushButton("Clear Current Polygon")
        self.label_clear_poly_btn.clicked.connect(self._label_clear_current_polygon)
        controls.addWidget(self.label_clear_poly_btn)
        layout.addLayout(controls)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class"))
        self.label_class_combo = QComboBox()
        self.label_class_combo.addItems(self.label_class_names)
        self.label_class_combo.currentIndexChanged.connect(self._label_class_changed)
        class_row.addWidget(self.label_class_combo)
        self.label_new_class_edit = QLineEdit()
        self.label_new_class_edit.setPlaceholderText("New class name")
        class_row.addWidget(self.label_new_class_edit)
        self.label_add_class_btn = QPushButton("Add Class")
        self.label_add_class_btn.clicked.connect(self._label_add_class)
        class_row.addWidget(self.label_add_class_btn)
        layout.addLayout(class_row)

        self.label_info = QLabel("Labeling: no folder loaded")
        self.label_info.setWordWrap(True)
        layout.addWidget(self.label_info)

        hint = QLabel(
            "Left click: add polygon point | Right click: remove last point | "
            "Click first point, double-click, or 'Finish Polygon': close polygon"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.label_canvas = PolygonAnnotationCanvas()
        self.label_canvas.on_changed = self._label_canvas_changed
        layout.addWidget(self.label_canvas, 1)

        split_form = QFormLayout()
        self.label_train_pct = QSpinBox()
        self.label_train_pct.setRange(0, 100)
        self.label_train_pct.setValue(70)
        split_form.addRow("Train %", self.label_train_pct)
        self.label_test_pct = QSpinBox()
        self.label_test_pct.setRange(0, 100)
        self.label_test_pct.setValue(20)
        split_form.addRow("Test %", self.label_test_pct)
        self.label_valid_pct = QSpinBox()
        self.label_valid_pct.setRange(0, 100)
        self.label_valid_pct.setValue(10)
        split_form.addRow("Valid %", self.label_valid_pct)
        self.label_export_dir = QLineEdit("data/labeled_polygon_dataset")
        self._add_dir_selector_row(split_form, "Export Folder", self.label_export_dir, "Select export output folder")
        layout.addLayout(split_form)

        export_row = QHBoxLayout()
        self.label_export_btn = QPushButton("Export Labeled Dataset")
        self.label_export_btn.setObjectName("primaryButton")
        self.label_export_btn.clicked.connect(self._export_labeled_dataset)
        export_row.addWidget(self.label_export_btn)
        layout.addLayout(export_row)

        self.label_log = QTextEdit()
        self.label_log.setReadOnly(True)
        self.label_log.setMinimumHeight(140)
        self.label_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label_log)

        self._label_refresh_nav()
        self.label_canvas.set_current_class(0)

    def _label_class_changed(self, index: int) -> None:
        self.label_canvas.set_current_class(index)

    def _label_add_class(self) -> None:
        name = self.label_new_class_edit.text().strip()
        if not name:
            return
        if name in self.label_class_names:
            QMessageBox.information(self, "Class Exists", f"Class '{name}' already exists.")
            return
        self.label_class_names.append(name)
        self.label_class_combo.addItem(name)
        self.label_class_combo.setCurrentIndex(len(self.label_class_names) - 1)
        self.label_new_class_edit.clear()

    def _label_current_path(self) -> Path | None:
        if not (0 <= self.label_index < len(self.label_image_paths)):
            return None
        return self.label_image_paths[self.label_index]

    def _label_store_current_annotations(self) -> None:
        path = self._label_current_path()
        if path is None:
            return
        self.label_annotations[str(path)] = self.label_canvas.annotations()

    def _label_load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder for Labeling")
        if not folder:
            return
        folder_path = Path(folder)
        self.label_image_paths = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        self.label_annotations = {}
        if not self.label_image_paths:
            QMessageBox.warning(self, "No Images", "No supported images found in this folder.")
            self.label_index = -1
            self._label_refresh_nav()
            return
        self.label_index = 0
        self._label_show_image(self.label_index)

    def _label_show_image(self, index: int) -> None:
        if not (0 <= index < len(self.label_image_paths)):
            return
        image_path = self.label_image_paths[index]
        existing = self.label_annotations.get(str(image_path), [])
        ok = self.label_canvas.set_image_and_annotations(image_path, existing)
        if not ok:
            QMessageBox.warning(self, "Image Error", f"Failed to read:\n{image_path}")
            return
        self.label_index = index
        self._label_refresh_info()
        self._label_refresh_nav()

    def _label_refresh_info(self) -> None:
        path = self._label_current_path()
        if path is None:
            self.label_info.setText("Labeling: no folder loaded")
            return
        count = len(self.label_canvas.annotations())
        self.label_info.setText(
            f"Image: {self.label_index + 1}/{len(self.label_image_paths)} - {path.name} | Polygons: {count}"
        )

    def _label_refresh_nav(self) -> None:
        has_images = len(self.label_image_paths) > 0
        self.label_prev_btn.setEnabled(has_images and self.label_index > 0)
        self.label_next_btn.setEnabled(has_images and self.label_index < len(self.label_image_paths) - 1)
        self.label_finish_poly_btn.setEnabled(has_images)
        self.label_undo_point_btn.setEnabled(has_images)
        self.label_undo_poly_btn.setEnabled(has_images)
        self.label_clear_poly_btn.setEnabled(has_images)

    def _label_prev_image(self) -> None:
        if self.label_index <= 0:
            return
        self._label_store_current_annotations()
        self._label_show_image(self.label_index - 1)

    def _label_next_image(self) -> None:
        if self.label_index >= len(self.label_image_paths) - 1:
            return
        self._label_store_current_annotations()
        self._label_show_image(self.label_index + 1)

    def _label_finish_polygon(self) -> None:
        if not self.label_canvas.finish_current_polygon():
            QMessageBox.information(self, "Polygon", "A polygon needs at least 3 points.")
        self._label_refresh_info()

    def _label_undo_point(self) -> None:
        self.label_canvas.remove_last_current_point()
        self._label_refresh_info()

    def _label_delete_last_polygon(self) -> None:
        self.label_canvas.remove_last_polygon()
        self._label_refresh_info()

    def _label_clear_current_polygon(self) -> None:
        self.label_canvas.clear_current_polygon()

    def _label_canvas_changed(self) -> None:
        self._label_refresh_info()

    def _write_yolo_seg_label(self, label_file: Path, annotations: list[dict], width: int, height: int) -> None:
        lines = []
        for ann in annotations:
            points = ann.get("points", [])
            if len(points) < 3:
                continue
            class_id = int(ann.get("class_id", 0))
            coords = []
            for x, y in points:
                xn = max(0.0, min(float(x) / max(1, width), 1.0))
                yn = max(0.0, min(float(y) / max(1, height), 1.0))
                coords.append(f"{xn:.6f}")
                coords.append(f"{yn:.6f}")
            lines.append(f"{class_id} " + " ".join(coords))
        label_file.write_text("\n".join(lines), encoding="utf-8")

    def _export_labeled_dataset(self) -> None:
        self._label_store_current_annotations()
        if not self.label_image_paths:
            QMessageBox.warning(self, "Export", "Load images first.")
            return
        train_pct = self.label_train_pct.value()
        test_pct = self.label_test_pct.value()
        valid_pct = self.label_valid_pct.value()
        if train_pct + test_pct + valid_pct != 100:
            QMessageBox.warning(self, "Export", "Train/Test/Valid percentages must sum to 100.")
            return

        labeled = []
        for path in self.label_image_paths:
            anns = self.label_annotations.get(str(path), [])
            if anns:
                labeled.append((path, anns))
        if not labeled:
            QMessageBox.warning(self, "Export", "No labeled polygons found.")
            return

        random.Random(42).shuffle(labeled)
        n = len(labeled)
        n_train = int(n * train_pct / 100)
        n_test = int(n * test_pct / 100)
        n_valid = n - n_train - n_test
        splits = {
            "train": labeled[:n_train],
            "test": labeled[n_train : n_train + n_test],
            "valid": labeled[n_train + n_test : n_train + n_test + n_valid],
        }

        out_root = Path(self.label_export_dir.text().strip() or "data/labeled_polygon_dataset")
        if not out_root.is_absolute():
            out_root = self._project_root() / out_root
        for split in ("train", "test", "valid"):
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        exported = 0
        for split_name, items in splits.items():
            for image_path, anns in items:
                img_out = out_root / "images" / split_name / image_path.name
                shutil.copy2(image_path, img_out)
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                label_out = out_root / "labels" / split_name / f"{image_path.stem}.txt"
                self._write_yolo_seg_label(label_out, anns, w, h)
                exported += 1

        data_yaml = out_root / "data.yaml"
        yaml_lines = [
            f"path: {out_root}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            f"nc: {len(self.label_class_names)}",
            "names:",
        ]
        for idx, name in enumerate(self.label_class_names):
            yaml_lines.append(f"  {idx}: {name}")
        data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        self.label_log.append(
            f"Exported {exported} labeled images to {out_root}\n"
            f"Split: train={len(splits['train'])}, test={len(splits['test'])}, valid={len(splits['valid'])}"
        )
        self.statusBar().showMessage("Labeled dataset exported", 5000)

    def _build_optimize_tab(self) -> None:
        layout = QVBoxLayout(self._tab_content(self.optimize_tab))
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        self.optimize_mode_tabs = QTabWidget()
        layout.addWidget(self.optimize_mode_tabs)
        self.optimize_amd_tab = QWidget()
        self.optimize_nvidia_tab = QWidget()
        self.optimize_cpu_tab = QWidget()
        self.optimize_mode_tabs.addTab(self.optimize_amd_tab, "AMD")
        self.optimize_mode_tabs.addTab(self.optimize_nvidia_tab, "NVIDIA")
        self.optimize_mode_tabs.addTab(self.optimize_cpu_tab, "CPU")
        self.optimize_mode_tabs.currentChanged.connect(self._on_optimize_mode_changed)

        preset_actions = QHBoxLayout()
        self.optimize_preset_amd_btn = QPushButton("AMD Safe")
        self.optimize_preset_amd_btn.clicked.connect(lambda: self._apply_optimize_preset("amd_safe"))
        preset_actions.addWidget(self.optimize_preset_amd_btn)
        self.optimize_preset_nvidia_btn = QPushButton("NVIDIA Fast")
        self.optimize_preset_nvidia_btn.clicked.connect(lambda: self._apply_optimize_preset("nvidia_fast"))
        preset_actions.addWidget(self.optimize_preset_nvidia_btn)
        self.optimize_preset_cpu_btn = QPushButton("CPU Stable")
        self.optimize_preset_cpu_btn.clicked.connect(lambda: self._apply_optimize_preset("cpu_stable"))
        preset_actions.addWidget(self.optimize_preset_cpu_btn)
        layout.addLayout(preset_actions)

        amd_layout = QFormLayout(self.optimize_amd_tab)
        self.optimize_amd_hsa_edit = QLineEdit("10.3.0")
        amd_layout.addRow("HSA GFX Override", self.optimize_amd_hsa_edit)
        self.optimize_amd_amp_check = QCheckBox("Enable AMP")
        self.optimize_amd_amp_check.setChecked(False)
        amd_layout.addRow(self.optimize_amd_amp_check)
        self.optimize_amd_val_check = QCheckBox("Enable validation during train")
        self.optimize_amd_val_check.setChecked(False)
        amd_layout.addRow(self.optimize_amd_val_check)

        nvidia_layout = QFormLayout(self.optimize_nvidia_tab)
        self.optimize_nvidia_amp_check = QCheckBox("Enable AMP")
        self.optimize_nvidia_amp_check.setChecked(True)
        nvidia_layout.addRow(self.optimize_nvidia_amp_check)
        self.optimize_nvidia_val_check = QCheckBox("Enable validation during train")
        self.optimize_nvidia_val_check.setChecked(True)
        nvidia_layout.addRow(self.optimize_nvidia_val_check)

        cpu_layout = QFormLayout(self.optimize_cpu_tab)
        self.optimize_cpu_amp_check = QCheckBox("Enable AMP")
        self.optimize_cpu_amp_check.setChecked(False)
        cpu_layout.addRow(self.optimize_cpu_amp_check)
        self.optimize_cpu_val_check = QCheckBox("Enable validation during train")
        self.optimize_cpu_val_check.setChecked(True)
        cpu_layout.addRow(self.optimize_cpu_val_check)

        form = QFormLayout()
        layout.addLayout(form)

        self.teacher_model_edit = QLineEdit()
        self._add_file_selector_row(
            form,
            "Teacher Model (--teacher-model)",
            self.teacher_model_edit,
            "Select teacher model",
            "Model Files (*.pt *.onnx *.engine *.torchscript);;All Files (*)",
        )

        self.images_dir_edit = QLineEdit()
        self._add_dir_selector_row(form, "Unlabeled Images (--images-dir)", self.images_dir_edit, "Select image folder")

        self.output_dir_edit = QLineEdit("data/pseudo_labeled_yolo")
        self._add_dir_selector_row(form, "Output Dataset (--output-dir)", self.output_dir_edit, "Select output folder")

        self.conf_train_spin = QDoubleSpinBox()
        self.conf_train_spin.setRange(0.0, 1.0)
        self.conf_train_spin.setDecimals(3)
        self.conf_train_spin.setSingleStep(0.05)
        self.conf_train_spin.setValue(0.7)
        form.addRow("Pseudo Conf (--conf)", self.conf_train_spin)

        self.imgsz_train_spin = QSpinBox()
        self.imgsz_train_spin.setRange(64, 4096)
        self.imgsz_train_spin.setValue(640)
        form.addRow("Image Size (--imgsz)", self.imgsz_train_spin)

        self.device_train_edit = QLineEdit("cpu")
        form.addRow("Device (--device)", self.device_train_edit)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 0.5)
        self.val_ratio_spin.setDecimals(3)
        self.val_ratio_spin.setSingleStep(0.01)
        self.val_ratio_spin.setValue(0.15)
        form.addRow("Val Ratio (--val-ratio)", self.val_ratio_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        form.addRow("Seed (--seed)", self.seed_spin)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 10_000_000)
        self.max_images_spin.setValue(0)
        form.addRow("Max Images (--max-images)", self.max_images_spin)

        self.train_model_edit = QLineEdit()
        self._add_file_selector_row(
            form,
            "Optimize Base Model (--train-model)",
            self.train_model_edit,
            "Select optimization base model",
            "Trainable Model Files (*.pt *.yaml *.yml);;All Files (*)",
        )

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs (--epochs)", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        form.addRow("Batch (--batch)", self.batch_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 128)
        self.workers_spin.setValue(4)
        form.addRow("Workers (--workers)", self.workers_spin)

        self.project_edit = QLineEdit("runs/train")
        form.addRow("Project (--project)", self.project_edit)

        self.run_name_edit = QLineEdit("pseudo_label_train")
        form.addRow("Run Name (--name)", self.run_name_edit)

        self.skip_train_check = QCheckBox("Skip train and only pseudo-label (--skip-train)")
        form.addRow(self.skip_train_check)

        self.export_onnx_name_edit = QLineEdit("optimized_model")
        form.addRow("New ONNX Name (--export-onnx-name)", self.export_onnx_name_edit)

        self.export_onnx_imgsz_spin = QSpinBox()
        self.export_onnx_imgsz_spin.setRange(0, 4096)
        self.export_onnx_imgsz_spin.setValue(0)
        form.addRow("ONNX Img Size (--export-onnx-imgsz)", self.export_onnx_imgsz_spin)

        actions = QHBoxLayout()
        self.start_train_btn = QPushButton("Start Pseudo-Label + Train")
        self.start_train_btn.setObjectName("primaryButton")
        self.start_train_btn.clicked.connect(self.start_training_pipeline)
        actions.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("Stop")
        self.stop_train_btn.setObjectName("dangerButton")
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.clicked.connect(self.stop_training_pipeline)
        actions.addWidget(self.stop_train_btn)
        layout.addLayout(actions)

        layout.addWidget(QLabel("A floating Matplotlib window opens automatically when training starts."))

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(0)
        self.training_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.training_log, 1)
        self._on_optimize_mode_changed(self.optimize_mode_tabs.currentIndex())

    def _build_training_tab(self) -> None:
        layout = QVBoxLayout(self._tab_content(self.training_tab))
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        form = QFormLayout()
        layout.addLayout(form)

        self.train_platform_combo = QComboBox()
        self.train_platform_combo.addItems(["AMD (ROCm)", "NVIDIA (CUDA)"])
        form.addRow("Platform", self.train_platform_combo)

        self.train_task_combo = QComboBox()
        self.train_task_combo.addItems(["obb", "detect"])
        form.addRow("Task", self.train_task_combo)

        self.obb_data_edit = QLineEdit(self._default_dataset_yaml("obb"))
        self._add_file_selector_row(
            form,
            "Dataset YAML (--data)",
            self.obb_data_edit,
            "Select dataset yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )

        self.obb_model_edit = QLineEdit("yolo11n-obb.pt")
        self._add_file_selector_row(
            form,
            "Model (--model)",
            self.obb_model_edit,
            "Select model",
            "Trainable Model Files (*.pt *.yaml *.yml);;All Files (*)",
        )

        self.obb_epochs_spin = QSpinBox()
        self.obb_epochs_spin.setRange(1, 10000)
        self.obb_epochs_spin.setValue(100)
        form.addRow("Epochs (--epochs)", self.obb_epochs_spin)

        self.obb_imgsz_spin = QSpinBox()
        self.obb_imgsz_spin.setRange(64, 4096)
        self.obb_imgsz_spin.setValue(640)
        form.addRow("Image Size (--imgsz)", self.obb_imgsz_spin)

        self.obb_batch_spin = QSpinBox()
        self.obb_batch_spin.setRange(1, 1024)
        self.obb_batch_spin.setValue(8)
        form.addRow("Batch (--batch)", self.obb_batch_spin)

        self.obb_device_edit = QLineEdit("0")
        form.addRow("Device (--device)", self.obb_device_edit)

        self.obb_workers_spin = QSpinBox()
        self.obb_workers_spin.setRange(0, 128)
        self.obb_workers_spin.setValue(8)
        form.addRow("Workers (--workers)", self.obb_workers_spin)

        self.obb_project_edit = QLineEdit("runs/obb")
        form.addRow("Project (--project)", self.obb_project_edit)

        self.obb_name_edit = QLineEdit("sandwich_panel_obb_gui")
        form.addRow("Run Name (--name)", self.obb_name_edit)

        self.obb_amp_check = QCheckBox("Enable AMP (--amp)")
        self.obb_amp_check.setChecked(False)
        form.addRow(self.obb_amp_check)

        self.obb_val_check = QCheckBox("Enable validation during train (--val)")
        self.obb_val_check.setChecked(False)
        form.addRow(self.obb_val_check)

        self.obb_hsa_edit = QLineEdit("10.3.0")
        form.addRow("AMD HSA GFX Override (--hsa-gfx)", self.obb_hsa_edit)

        actions = QHBoxLayout()
        self.start_model_train_btn = QPushButton("Start Training")
        self.start_model_train_btn.setObjectName("primaryButton")
        self.start_model_train_btn.clicked.connect(self.start_model_training)
        actions.addWidget(self.start_model_train_btn)

        self.stop_model_train_btn = QPushButton("Stop")
        self.stop_model_train_btn.setObjectName("dangerButton")
        self.stop_model_train_btn.setEnabled(False)
        self.stop_model_train_btn.clicked.connect(self.stop_model_training)
        actions.addWidget(self.stop_model_train_btn)
        layout.addLayout(actions)

        self.model_training_log = QTextEdit()
        self.model_training_log.setReadOnly(True)
        self.model_training_log.setMinimumHeight(0)
        self.model_training_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.model_training_log, 1)
        self.train_task_combo.currentTextChanged.connect(self._on_training_task_changed)
        self._on_training_task_changed(self.train_task_combo.currentText())

    def _build_converter_tab(self) -> None:
        layout = QVBoxLayout(self._tab_content(self.converter_tab))
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        layout.addLayout(form)

        self.converter_input_edit = QLineEdit()
        self._add_file_selector_row(
            form,
            "Input PT Model",
            self.converter_input_edit,
            "Select .pt model",
            "PyTorch Model Files (*.pt);;All Files (*)",
        )
        self.converter_input_edit.textChanged.connect(self._sync_converter_output_path)

        self.converter_output_edit = QLineEdit()
        output_row = QWidget()
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.addWidget(self.converter_output_edit)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._pick_converter_output_file)
        output_row_layout.addWidget(output_browse)
        form.addRow("Output ONNX File", output_row)

        self.converter_task_combo = QComboBox()
        self.converter_task_combo.addItems(["obb", "detect", "segment", "classify", "pose"])
        form.addRow("Task", self.converter_task_combo)

        self.converter_imgsz_spin = QSpinBox()
        self.converter_imgsz_spin.setRange(0, 4096)
        self.converter_imgsz_spin.setValue(640)
        self.converter_imgsz_spin.setToolTip("Set 0 to keep model default")
        form.addRow("Image Size (imgsz)", self.converter_imgsz_spin)

        self.converter_opset_spin = QSpinBox()
        self.converter_opset_spin.setRange(9, 22)
        self.converter_opset_spin.setValue(17)
        form.addRow("ONNX Opset", self.converter_opset_spin)

        self.converter_runtime_combo = QComboBox()
        self.converter_runtime_combo.addItems(["Auto", "AMD (.venv-rocm)", "NVIDIA (.venv)", "Current Python"])
        form.addRow("Runtime", self.converter_runtime_combo)

        actions = QHBoxLayout()
        self.start_converter_btn = QPushButton("Convert PT to ONNX")
        self.start_converter_btn.setObjectName("primaryButton")
        self.start_converter_btn.clicked.connect(self.start_model_conversion)
        actions.addWidget(self.start_converter_btn)

        self.stop_converter_btn = QPushButton("Stop")
        self.stop_converter_btn.setObjectName("dangerButton")
        self.stop_converter_btn.setEnabled(False)
        self.stop_converter_btn.clicked.connect(self.stop_model_conversion)
        actions.addWidget(self.stop_converter_btn)
        layout.addLayout(actions)

        self.converter_log = QTextEdit()
        self.converter_log.setReadOnly(True)
        self.converter_log.setMinimumHeight(0)
        self.converter_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.converter_log, 1)

    def _add_file_selector_row(
        self,
        form: QFormLayout,
        label: str,
        edit: QLineEdit,
        caption: str,
        file_filter: str,
    ) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit)
        browse = QPushButton("Browse")
        browse.clicked.connect(
            lambda: self._pick_file_into_lineedit(edit, caption=caption, file_filter=file_filter)
        )
        row_layout.addWidget(browse)
        form.addRow(label, row)

    def _add_dir_selector_row(self, form: QFormLayout, label: str, edit: QLineEdit, caption: str) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._pick_dir_into_lineedit(edit, caption=caption))
        row_layout.addWidget(browse)
        form.addRow(label, row)

    def _pick_file_into_lineedit(self, edit: QLineEdit, caption: str, file_filter: str) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, caption, "", file_filter)
        if file_name:
            edit.setText(file_name)

    def _pick_save_file_into_lineedit(self, edit: QLineEdit, caption: str, file_filter: str) -> None:
        file_name, _ = QFileDialog.getSaveFileName(self, caption, edit.text().strip(), file_filter)
        if file_name:
            edit.setText(file_name)

    def _pick_converter_output_file(self) -> None:
        self._pick_save_file_into_lineedit(
            self.converter_output_edit,
            "Select output .onnx file",
            "ONNX Files (*.onnx);;All Files (*)",
        )

    def _sync_converter_output_path(self, input_path: str) -> None:
        input_path = input_path.strip()
        if not input_path:
            return
        if self.converter_output_edit.text().strip():
            return
        out_path = str(Path(input_path).with_suffix(".onnx"))
        self.converter_output_edit.setText(out_path)

    def _pick_dir_into_lineedit(self, edit: QLineEdit, caption: str) -> None:
        dir_name = QFileDialog.getExistingDirectory(self, caption)
        if dir_name:
            edit.setText(dir_name)

    def _apply_argument_tooltips(self) -> None:
        # Inference
        self.load_model_btn.setToolTip("Load a model file used for inference.")
        self.load_folder_btn.setToolTip("Load a folder of images for sequential inference.")
        self.load_video_btn.setToolTip("Load a video file for frame-by-frame inference.")
        self.prev_btn.setToolTip("Go to previous image/frame.")
        self.next_btn.setToolTip("Go to next image/frame.")
        self.conf_slider.setToolTip("Confidence threshold for predictions (0-100%).")

        # Labeling
        self.label_load_folder_btn.setToolTip("Load images to annotate with polygons.")
        self.label_class_combo.setToolTip("Current class ID used for new polygons.")
        self.label_new_class_edit.setToolTip("Type a new class name, then click Add Class.")
        self.label_add_class_btn.setToolTip("Add the typed class name to the class list.")
        self.label_finish_poly_btn.setToolTip("Close current polygon (minimum 3 points).")
        self.label_undo_point_btn.setToolTip("Remove the last point from current polygon.")
        self.label_undo_poly_btn.setToolTip("Delete the most recently created polygon.")
        self.label_clear_poly_btn.setToolTip("Clear all points from current polygon.")
        self.label_train_pct.setToolTip("Percentage of labeled images exported to train split.")
        self.label_test_pct.setToolTip("Percentage of labeled images exported to test split.")
        self.label_valid_pct.setToolTip("Percentage of labeled images exported to validation split.")
        self.label_export_dir.setToolTip("Output folder for exported dataset (images, labels, data.yaml).")
        self.label_export_btn.setToolTip("Export labeled polygons as YOLO-seg dataset files.")

        # Optimize tab platform controls
        self.optimize_preset_amd_btn.setToolTip("Apply conservative AMD defaults for stability.")
        self.optimize_preset_nvidia_btn.setToolTip("Apply faster NVIDIA defaults.")
        self.optimize_preset_cpu_btn.setToolTip("Apply CPU-safe defaults.")
        self.optimize_amd_hsa_edit.setToolTip("AMD only: HSA_OVERRIDE_GFX_VERSION passed to training.")
        self.optimize_amd_amp_check.setToolTip("AMD only: enable mixed precision (faster, may be unstable).")
        self.optimize_amd_val_check.setToolTip("AMD only: run validation during training.")
        self.optimize_nvidia_amp_check.setToolTip("NVIDIA only: enable mixed precision training.")
        self.optimize_nvidia_val_check.setToolTip("NVIDIA only: run validation during training.")
        self.optimize_cpu_amp_check.setToolTip("CPU only: enable mixed precision (usually keep disabled).")
        self.optimize_cpu_val_check.setToolTip("CPU only: run validation during training.")

        # Optimize tab arguments (pseudo_label_and_train.py)
        self.teacher_model_edit.setToolTip("--teacher-model: model used to generate pseudo labels.")
        self.images_dir_edit.setToolTip("--images-dir: unlabeled source images directory.")
        self.output_dir_edit.setToolTip("--output-dir: destination dataset directory for pseudo labels.")
        self.conf_train_spin.setToolTip("--conf: minimum confidence for pseudo-label detections.")
        self.imgsz_train_spin.setToolTip("--imgsz: image size for pseudo-label inference and training.")
        self.device_train_edit.setToolTip("--device: training/inference device (cpu, 0, 0,1...).")
        self.val_ratio_spin.setToolTip("--val-ratio: validation split ratio for generated dataset.")
        self.seed_spin.setToolTip("--seed: random seed used for data split reproducibility.")
        self.max_images_spin.setToolTip("--max-images: limit number of unlabeled images (0 means all).")
        self.train_model_edit.setToolTip("--train-model: base trainable model/config (.pt/.yaml/.yml).")
        self.epochs_spin.setToolTip("--epochs: number of training epochs.")
        self.batch_spin.setToolTip("--batch: batch size for training.")
        self.workers_spin.setToolTip("--workers: dataloader worker processes.")
        self.project_edit.setToolTip("--project: output project directory for training runs.")
        self.run_name_edit.setToolTip("--name: run name subfolder inside project directory.")
        self.skip_train_check.setToolTip("--skip-train: only create pseudo labels, skip model training.")
        self.export_onnx_name_edit.setToolTip("--export-onnx-name: output ONNX filename after training.")
        self.export_onnx_imgsz_spin.setToolTip("--export-onnx-imgsz: ONNX export image size (0 uses --imgsz).")
        self.start_train_btn.setToolTip("Start pseudo-label generation and optional training pipeline.")
        self.stop_train_btn.setToolTip("Stop current pseudo-label/training process.")

        # Training tab arguments
        self.train_platform_combo.setToolTip("Choose runtime stack for training: AMD ROCm or NVIDIA CUDA.")
        self.train_task_combo.setToolTip("YOLO task for training command (detect or obb).")
        self.obb_data_edit.setToolTip("--data: dataset YAML path.")
        self.obb_model_edit.setToolTip("--model: trainable model/config (.pt/.yaml/.yml).")
        self.obb_epochs_spin.setToolTip("--epochs: number of training epochs.")
        self.obb_imgsz_spin.setToolTip("--imgsz: image size used for training.")
        self.obb_batch_spin.setToolTip("--batch: batch size for training.")
        self.obb_device_edit.setToolTip("--device: training device (cpu, 0, 0,1...).")
        self.obb_workers_spin.setToolTip("--workers: dataloader worker processes.")
        self.obb_project_edit.setToolTip("--project: output project directory for run artifacts.")
        self.obb_name_edit.setToolTip("--name: run name subfolder.")
        self.obb_amp_check.setToolTip("--amp: enable automatic mixed precision.")
        self.obb_val_check.setToolTip("--val: run validation during training.")
        self.obb_hsa_edit.setToolTip("--hsa-gfx: AMD HSA override value for ROCm environment.")
        self.start_model_train_btn.setToolTip("Start the selected training task.")
        self.stop_model_train_btn.setToolTip("Stop current training process.")

        # Converter tab arguments
        self.converter_input_edit.setToolTip("Input .pt model path to convert.")
        self.converter_output_edit.setToolTip("Output .onnx file path.")
        self.converter_task_combo.setToolTip("Task used to load/export model metadata.")
        self.converter_imgsz_spin.setToolTip("imgsz used during export (0 keeps model default).")
        self.converter_opset_spin.setToolTip("ONNX opset version used for export.")
        self.converter_runtime_combo.setToolTip("Python/runtime environment used for conversion.")
        self.start_converter_btn.setToolTip("Run .pt to .onnx conversion.")
        self.stop_converter_btn.setToolTip("Stop current conversion process.")

    def _project_root(self) -> Path:
        here = Path(__file__).resolve()
        for candidate in [here.parent, *here.parents]:
            if (candidate / ".git").exists():
                return candidate
        return here.parents[2]

    def _resolve_path(self, path_text: str) -> Path:
        p = Path(path_text).expanduser()
        if p.is_absolute():
            return p
        return self._project_root() / p

    def _resolve_existing_path(self, path_text: str) -> Path | None:
        p = Path(path_text).expanduser()
        candidates = [p] if p.is_absolute() else [p, self._project_root() / p]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _default_dataset_yaml(self, task: str) -> str:
        root = self._project_root()
        task = (task or "").strip().lower()
        if task == "detect":
            candidates = [
                root / "data" / "SandwichPanel.v9i.yolov8" / "data.yaml",
                root / "Backup" / "Data" / "SandwichPanel.v9i.yolov8" / "data.yaml",
            ]
            patterns = ("*yolov8*/data.yaml", "*detect*/data.yaml")
        else:
            candidates = [
                root / "data" / "SandwichPanel.v8i.yolov8-obb" / "data.yaml",
                root / "data" / "SandwichPanel.v7i.yolov8-obb" / "data.yaml",
                root / "Backup" / "Data" / "SandwichPanel.v8i.yolov8-obb" / "data.yaml",
                root / "Backup" / "Data" / "SandwichPanel.v7i.yolov8-obb" / "data.yaml",
            ]
            patterns = ("*obb*/data.yaml",)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        search_roots = [root / "data", root / "Backup" / "Data", root]
        for base in search_roots:
            if not base.exists():
                continue
            for pattern in patterns:
                found = sorted(base.glob(f"**/{pattern}"))
                for path in found:
                    if path.is_file():
                        if task == "detect" and "obb" in str(path).lower():
                            continue
                        return str(path)

        for base in search_roots:
            if not base.exists():
                continue
            fallback = sorted(base.glob("**/data.yaml"))
            for path in fallback:
                if path.is_file():
                    if task == "detect" and "obb" in str(path).lower():
                        continue
                    return str(path)

        return str(candidates[0])

    @staticmethod
    def _is_trainable_model_file(path_text: str) -> bool:
        suffix = Path(path_text).suffix.lower()
        return suffix in TRAINABLE_MODEL_SUFFIXES

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)

    @staticmethod
    def _safe_float(value: str | None) -> float:
        try:
            if value is None or value == "":
                return float("nan")
            return float(value)
        except Exception:
            return float("nan")

    def _on_optimize_mode_changed(self, index: int) -> None:
        if index == 0:  # AMD
            if self.device_train_edit.text().strip().lower() == "cpu":
                self.device_train_edit.setText("0")
        elif index == 1:  # NVIDIA
            if self.device_train_edit.text().strip().lower() == "cpu":
                self.device_train_edit.setText("0")
        else:  # CPU
            self.device_train_edit.setText("cpu")

    def _on_training_task_changed(self, task: str) -> None:
        task = (task or "").strip().lower()
        if task not in {"obb", "detect"}:
            task = "obb"
        self.start_model_train_btn.setText(f"Start {task.upper()} Training")

        obb_default = self._default_dataset_yaml("obb")
        detect_default = self._default_dataset_yaml("detect")
        current_data = self.obb_data_edit.text().strip()
        current_data_exists = self._resolve_existing_path(current_data) is not None

        legacy_obb_defaults = {
            "data/SandwichPanel.v7i.yolov8-obb/data.yaml",
            "data/SandwichPanel.v8i.yolov8-obb/data.yaml",
        }
        legacy_detect_defaults = {
            "data/SandwichPanel.v9i.yolov8/data.yaml",
        }

        project_text = self.obb_project_edit.text().strip()
        model_text = self.obb_model_edit.text().strip()
        if task == "detect":
            if (
                current_data in legacy_obb_defaults
                or current_data == obb_default
                or (current_data and not current_data_exists)
            ):
                self.obb_data_edit.setText(detect_default)
            if project_text == "runs/obb":
                self.obb_project_edit.setText("runs/detect")
            if model_text == "yolo11n-obb.pt":
                self.obb_model_edit.setText("yolo11n.pt")
        else:
            if (
                current_data in legacy_detect_defaults
                or current_data == detect_default
                or (current_data and not current_data_exists)
            ):
                self.obb_data_edit.setText(obb_default)
            if project_text == "runs/detect":
                self.obb_project_edit.setText("runs/obb")
            if model_text == "yolo11n.pt":
                self.obb_model_edit.setText("yolo11n-obb.pt")

    def _default_training_project_and_name(self) -> tuple[str, str]:
        task = self.train_task_combo.currentText().strip().lower()
        if task == "detect":
            return "runs/detect", "sandwich_panel_detect_gui"
        return "runs/obb", "sandwich_panel_obb_gui"

    def _validate_training_task_model_combo(self, task: str, model_path: str) -> bool:
        name = Path(model_path).name.lower()
        if task == "detect" and "obb" in name:
            QMessageBox.warning(
                self,
                "Task/Model Mismatch",
                "Task is set to detect, but the selected model appears to be OBB.\n"
                "Choose a detect model (e.g. yolo11n.pt) or switch task to obb.",
            )
            return False
        return True

    def _optimize_platform_args(self) -> list[str]:
        index = self.optimize_mode_tabs.currentIndex()
        if index == 0:
            args = [
                "--platform",
                "amd",
                "--hsa-gfx",
                self.optimize_amd_hsa_edit.text().strip() or "10.3.0",
                "--amp",
                "true" if self.optimize_amd_amp_check.isChecked() else "false",
                "--train-val",
                "true" if self.optimize_amd_val_check.isChecked() else "false",
            ]
            return args
        if index == 1:
            return [
                "--platform",
                "nvidia",
                "--amp",
                "true" if self.optimize_nvidia_amp_check.isChecked() else "false",
                "--train-val",
                "true" if self.optimize_nvidia_val_check.isChecked() else "false",
            ]
        return [
            "--platform",
            "cpu",
            "--amp",
            "true" if self.optimize_cpu_amp_check.isChecked() else "false",
            "--train-val",
            "true" if self.optimize_cpu_val_check.isChecked() else "false",
        ]

    def _current_optimize_mode(self) -> str:
        index = self.optimize_mode_tabs.currentIndex()
        if index == 0:
            return "amd"
        if index == 1:
            return "nvidia"
        return "cpu"

    def _select_optimize_python(self, mode: str) -> str:
        root = self._project_root()
        rocm_py = root / ".venv-rocm" / "bin" / "python"
        cuda_py = root / ".venv" / "bin" / "python"

        if mode == "amd":
            return str(rocm_py) if rocm_py.exists() else ""
        if mode == "nvidia":
            return str(cuda_py) if cuda_py.exists() else ""

        if cuda_py.exists():
            return str(cuda_py)
        if rocm_py.exists():
            return str(rocm_py)
        return sys.executable

    def _validate_optimize_runtime(self, mode: str, python_bin: str) -> bool:
        if not python_bin:
            if mode == "amd":
                QMessageBox.warning(
                    self,
                    "Missing AMD Runtime",
                    "AMD mode requires .venv-rocm/bin/python.\nCreate the ROCm environment first.",
                )
                return False
            if mode == "nvidia":
                QMessageBox.warning(
                    self,
                    "Missing NVIDIA Runtime",
                    "NVIDIA mode requires .venv/bin/python.\nCreate the CUDA environment first.",
                )
                return False
            return False

        if not Path(python_bin).exists():
            QMessageBox.warning(self, "Missing Python", f"Python runtime not found:\n{python_bin}")
            return False
        return True

    def _apply_optimize_preset(self, preset: str) -> None:
        if preset == "amd_safe":
            self.optimize_mode_tabs.setCurrentIndex(0)
            self.device_train_edit.setText("0")
            self.optimize_amd_hsa_edit.setText("10.3.0")
            self.optimize_amd_amp_check.setChecked(False)
            self.optimize_amd_val_check.setChecked(False)
            self.workers_spin.setValue(4)
            self.statusBar().showMessage("Optimize preset applied: AMD Safe", 3000)
            return

        if preset == "nvidia_fast":
            self.optimize_mode_tabs.setCurrentIndex(1)
            self.device_train_edit.setText("0")
            self.optimize_nvidia_amp_check.setChecked(True)
            self.optimize_nvidia_val_check.setChecked(True)
            self.workers_spin.setValue(8)
            self.statusBar().showMessage("Optimize preset applied: NVIDIA Fast", 3000)
            return

        if preset == "cpu_stable":
            self.optimize_mode_tabs.setCurrentIndex(2)
            self.device_train_edit.setText("cpu")
            self.optimize_cpu_amp_check.setChecked(False)
            self.optimize_cpu_val_check.setChecked(True)
            self.workers_spin.setValue(0)
            self.statusBar().showMessage("Optimize preset applied: CPU Stable", 3000)

    def _extract_results_csv_from_output(self, text: str) -> Path | None:
        markers = ("Logging results to", "Results saved to", "Saving to")
        for raw_line in text.splitlines():
            line = self._strip_ansi(raw_line)
            for marker in markers:
                if marker in line:
                    run_dir_str = line.split(marker, 1)[1].strip().strip('"').strip("'")
                    run_dir = Path(run_dir_str.rstrip("/"))
                    # Ultralytics usually logs the run directory. Be tolerant if file path is printed.
                    if run_dir.name == "results.csv":
                        return run_dir
                    return run_dir / "results.csv"
        return None

    def _fallback_find_results_csv(self, project_value: str, run_name: str) -> Path | None:
        project_dir = Path(project_value)
        if not project_dir.is_absolute():
            project_dir = self._project_root() / project_dir
        if not project_dir.exists():
            return None

        run_name = run_name.lower()
        candidates = []
        for path in project_dir.rglob("results.csv"):
            parent_path = str(path.parent).lower()
            if run_name in parent_path:
                candidates.append(path)
        if not candidates:
            candidates = list(project_dir.rglob("results.csv"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _refresh_results_csv_path(self, current_path: Path | None, project_value: str, run_name: str) -> Path | None:
        newest = self._fallback_find_results_csv(project_value, run_name)
        if newest is None:
            return current_path
        if current_path is None:
            return newest
        try:
            current_mtime = current_path.stat().st_mtime if current_path.exists() else -1
            newest_mtime = newest.stat().st_mtime
        except Exception:
            return newest
        return newest if newest_mtime >= current_mtime else current_path

    def _create_plot_window(self, title: str) -> FloatingTrainingPlotWindow | None:
        if not HAS_MATPLOTLIB:
            return None
        win = FloatingTrainingPlotWindow(title)
        self.plot_windows.append(win)
        win.show()
        return win

    def _update_plot_window(self, csv_path: Path | None, plot_window: FloatingTrainingPlotWindow | None) -> None:
        if not HAS_MATPLOTLIB or plot_window is None:
            return

        if csv_path is None or not csv_path.exists():
            return

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return
        if not rows:
            return

        x = list(range(1, len(rows) + 1))
        row0 = rows[0]
        loss_keys = [k for k in row0 if k.startswith("train/") and k.endswith("_loss")]
        metric_keys = [k for k in row0 if k.startswith("metrics/")]

        plot_window.clear_plot()
        loss_ax = plot_window.loss_ax
        metric_ax = plot_window.metric_ax
        if loss_ax is None or metric_ax is None:
            return

        for key in loss_keys:
            if key in rows[0]:
                values = [self._safe_float(row.get(key)) for row in rows]
                if not all(math.isnan(v) for v in values):
                    loss_ax.plot(x, values, label=key.replace("train/", ""))
        if loss_ax.lines:
            loss_ax.legend(loc="best", fontsize=8)

        for key in metric_keys:
            if key in rows[0]:
                values = [self._safe_float(row.get(key)) for row in rows]
                if not all(math.isnan(v) for v in values):
                    metric_ax.plot(x, values, label=key.replace("metrics/", ""))
        if metric_ax.lines:
            metric_ax.legend(loc="best", fontsize=8)

        plot_window.canvas.draw_idle()

    def _update_optimize_training_plot(self) -> None:
        self.train_results_csv = self._refresh_results_csv_path(
            self.train_results_csv,
            self.project_edit.text().strip() or "runs/train",
            self.run_name_edit.text().strip() or "pseudo_label_train",
        )
        self._update_plot_window(self.train_results_csv, self.optimize_plot_window)

    def _update_model_training_plot(self) -> None:
        default_project, default_run_name = self._default_training_project_and_name()
        self.model_train_results_csv = self._refresh_results_csv_path(
            self.model_train_results_csv,
            self.obb_project_edit.text().strip() or default_project,
            self.obb_name_edit.text().strip() or default_run_name,
        )
        self._update_plot_window(self.model_train_results_csv, self.model_plot_window)

    def _select_training_python(self) -> str:
        root = self._project_root()
        rocm_py = root / ".venv-rocm" / "bin" / "python"
        cuda_py = root / ".venv" / "bin" / "python"
        platform_is_amd = self.train_platform_combo.currentText().startswith("AMD")
        if platform_is_amd and rocm_py.exists():
            return str(rocm_py)
        if (not platform_is_amd) and cuda_py.exists():
            return str(cuda_py)
        if rocm_py.exists():
            return str(rocm_py)
        if cuda_py.exists():
            return str(cuda_py)
        return sys.executable

    def _resolve_project_dir(self, project_value: str) -> Path:
        project_dir = Path(project_value)
        if not project_dir.is_absolute():
            project_dir = self._project_root() / project_dir
        return project_dir

    def _find_best_weights(self, project_value: str, run_name: str) -> Path | None:
        project_dir = self._resolve_project_dir(project_value)
        expected = project_dir / run_name / "weights" / "best.pt"
        if expected.exists():
            return expected
        if not project_dir.exists():
            return None
        candidates = list(project_dir.rglob("best.pt"))
        if not candidates:
            return None
        run_name = run_name.lower().strip()
        if run_name:
            filtered = [p for p in candidates if run_name in str(p.parent.parent).lower()]
            if filtered:
                candidates = filtered
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _start_model_export_onnx(self) -> None:
        if self.model_export_process is not None and self.model_export_process.state() != QProcess.NotRunning:
            return

        default_project, default_run_name = self._default_training_project_and_name()
        project_value = self.obb_project_edit.text().strip() or default_project
        run_name = self.obb_name_edit.text().strip() or default_run_name
        best_pt = self._find_best_weights(project_value, run_name)
        if best_pt is None:
            self.model_training_log.append("\n[ONNX export skipped: best.pt not found]")
            return

        python_bin = self._select_training_python()
        if not Path(python_bin).exists():
            self.model_training_log.append(f"\n[ONNX export skipped: Python runtime not found: {python_bin}]")
            return

        export_code = (
            "from ultralytics import YOLO; import sys; "
            "out=YOLO(sys.argv[1]).export(format='onnx', imgsz=int(sys.argv[2]), opset=17, simplify=False); "
            "print(out)"
        )
        args = ["-c", export_code, str(best_pt), str(self.obb_imgsz_spin.value())]
        self.model_training_log.append(f"\n$ {python_bin} {' '.join(args)}")

        self.model_export_process = QProcess(self)
        self.model_export_process.setWorkingDirectory(str(self._project_root()))
        self.model_export_process.setProgram(python_bin)
        self.model_export_process.setArguments(args)
        self.model_export_process.setProcessChannelMode(QProcess.MergedChannels)
        self.model_export_process.readyReadStandardOutput.connect(self._on_model_export_output)
        self.model_export_process.finished.connect(self._on_model_export_finished)
        self.model_export_process.start()
        self.statusBar().showMessage("Exporting ONNX from best.pt ...", 5000)

    def _on_model_export_output(self) -> None:
        if self.model_export_process is None:
            return
        text = bytes(self.model_export_process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.model_training_log.insertPlainText(text)
            self.model_training_log.verticalScrollBar().setValue(
                self.model_training_log.verticalScrollBar().maximum()
            )

    def _on_model_export_finished(self, exit_code: int, exit_status: int) -> None:
        _ = exit_status
        if exit_code == 0:
            self.model_training_log.append("\n[ONNX export finished successfully]")
            self.statusBar().showMessage("ONNX export finished", 5000)
        else:
            self.model_training_log.append(f"\n[ONNX export failed, exit code {exit_code}]")
            self.statusBar().showMessage("ONNX export failed", 5000)

    def _select_converter_python(self) -> str:
        root = self._project_root()
        rocm_py = root / ".venv-rocm" / "bin" / "python"
        cuda_py = root / ".venv" / "bin" / "python"
        mode = self.converter_runtime_combo.currentText()

        if mode.startswith("AMD"):
            return str(rocm_py) if rocm_py.exists() else ""
        if mode.startswith("NVIDIA"):
            return str(cuda_py) if cuda_py.exists() else ""
        if mode.startswith("Current"):
            return sys.executable

        if rocm_py.exists():
            return str(rocm_py)
        if cuda_py.exists():
            return str(cuda_py)
        return sys.executable

    def start_model_conversion(self) -> None:
        if self.converter_process is not None and self.converter_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Converter", "A conversion process is already running.")
            return

        src = self.converter_input_edit.text().strip()
        dst = self.converter_output_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "Missing Input", "Please choose a .pt model file.")
            return
        if Path(src).suffix.lower() != ".pt":
            QMessageBox.warning(self, "Invalid Input", "Input model must be a .pt file.")
            return
        src_path = self._resolve_existing_path(src)
        if src_path is None:
            QMessageBox.warning(self, "Missing Input", f"Input model not found:\n{src}")
            return
        if not dst:
            dst = str(src_path.with_suffix(".onnx"))
            self.converter_output_edit.setText(dst)
        if Path(dst).suffix.lower() != ".onnx":
            QMessageBox.warning(self, "Invalid Output", "Output path must end with .onnx.")
            return
        dst_path = self._resolve_path(dst)
        if not dst_path.parent.exists():
            QMessageBox.warning(
                self,
                "Missing Output Folder",
                "Output folder does not exist. Please choose an existing folder.",
            )
            return

        python_bin = self._select_converter_python()
        if not python_bin:
            QMessageBox.warning(self, "Missing Runtime", "Selected runtime is not available.")
            return
        if not Path(python_bin).exists():
            QMessageBox.warning(self, "Missing Python", f"Python runtime not found:\n{python_bin}")
            return

        export_code = (
            "from pathlib import Path; import sys; from ultralytics import YOLO; "
            "src=Path(sys.argv[1]); dst=Path(sys.argv[2]); imgsz=int(sys.argv[3]); "
            "task=sys.argv[4]; opset=int(sys.argv[5]); "
            "model=YOLO(str(src), task=task); "
            "kwargs={'format':'onnx', 'opset':opset, 'simplify':False, **({'imgsz': imgsz} if imgsz>0 else {})}; "
            "out=Path(model.export(**kwargs)); "
            "_=out.replace(dst) if out.resolve()!=dst.resolve() else None; "
            "print(dst)"
        )
        args = [
            "-c",
            export_code,
            str(src_path),
            str(dst_path),
            str(self.converter_imgsz_spin.value()),
            self.converter_task_combo.currentText(),
            str(self.converter_opset_spin.value()),
        ]

        self.converter_log.clear()
        self.converter_log.append(f"$ {python_bin} {' '.join(args)}")

        self.converter_process = QProcess(self)
        self.converter_process.setWorkingDirectory(str(self._project_root()))
        self.converter_process.setProgram(python_bin)
        self.converter_process.setArguments(args)
        self.converter_process.setProcessChannelMode(QProcess.MergedChannels)
        self.converter_process.readyReadStandardOutput.connect(self._on_model_conversion_output)
        self.converter_process.finished.connect(self._on_model_conversion_finished)
        self.converter_process.start()

        self.start_converter_btn.setEnabled(False)
        self.stop_converter_btn.setEnabled(True)
        self.statusBar().showMessage("Model conversion started", 3000)

    def stop_model_conversion(self) -> None:
        if self.converter_process is None or self.converter_process.state() == QProcess.NotRunning:
            return
        self.converter_process.kill()
        self.converter_log.append("\n[Model conversion stopped]")
        self.start_converter_btn.setEnabled(True)
        self.stop_converter_btn.setEnabled(False)
        self.statusBar().showMessage("Model conversion stopped", 3000)

    def _on_model_conversion_output(self) -> None:
        if self.converter_process is None:
            return
        text = bytes(self.converter_process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.converter_log.insertPlainText(text)
            self.converter_log.verticalScrollBar().setValue(self.converter_log.verticalScrollBar().maximum())

    def _on_model_conversion_finished(self, exit_code: int, exit_status: int) -> None:
        _ = exit_status
        self.start_converter_btn.setEnabled(True)
        self.stop_converter_btn.setEnabled(False)
        if exit_code == 0:
            self.converter_log.append("\n[Model conversion finished successfully]")
            self.statusBar().showMessage("Model conversion finished", 5000)
        else:
            self.converter_log.append(f"\n[Model conversion failed, exit code {exit_code}]")
            self.statusBar().showMessage("Model conversion failed", 5000)

    def start_training_pipeline(self) -> None:
        if self.train_process is not None and self.train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Training", "A training process is already running.")
            return

        teacher_model = self.teacher_model_edit.text().strip()
        images_dir = self.images_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not teacher_model:
            QMessageBox.warning(self, "Missing Input", "Please choose --teacher-model.")
            return
        if not images_dir:
            QMessageBox.warning(self, "Missing Input", "Please choose --images-dir.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing Input", "Please set --output-dir.")
            return

        teacher_model_path = self._resolve_existing_path(teacher_model)
        if teacher_model_path is None:
            QMessageBox.warning(self, "Missing Input", f"Teacher model not found:\n{teacher_model}")
            return
        images_dir_path = self._resolve_existing_path(images_dir)
        if images_dir_path is None or not images_dir_path.is_dir():
            QMessageBox.warning(self, "Missing Input", f"Images directory not found:\n{images_dir}")
            return
        output_dir_path = self._resolve_path(output_dir)

        mode = self._current_optimize_mode()
        python_bin = self._select_optimize_python(mode)
        if not self._validate_optimize_runtime(mode, python_bin):
            return

        script_path = Path(__file__).resolve().parent / "pseudo_label_and_train.py"
        args = [
            str(script_path),
            "--teacher-model",
            str(teacher_model_path),
            "--images-dir",
            str(images_dir_path),
            "--output-dir",
            str(output_dir_path),
            "--conf",
            str(self.conf_train_spin.value()),
            "--imgsz",
            str(self.imgsz_train_spin.value()),
            "--device",
            self.device_train_edit.text().strip() or "cpu",
            "--val-ratio",
            str(self.val_ratio_spin.value()),
            "--seed",
            str(self.seed_spin.value()),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch",
            str(self.batch_spin.value()),
            "--workers",
            str(self.workers_spin.value()),
            "--project",
            self.project_edit.text().strip() or "runs/train",
            "--name",
            self.run_name_edit.text().strip() or "pseudo_label_train",
        ]
        args.extend(self._optimize_platform_args())

        if self.max_images_spin.value() > 0:
            args.extend(["--max-images", str(self.max_images_spin.value())])

        train_model = self.train_model_edit.text().strip()
        if train_model:
            if not self._is_trainable_model_file(train_model):
                QMessageBox.warning(
                    self,
                    "Invalid Train Model",
                    "Optimize Base Model must be .pt or .yaml/.yml.\n"
                    "ONNX models are inference-only and cannot be trained.",
                )
                return
            train_model_path = self._resolve_existing_path(train_model)
            args.extend(["--train-model", str(train_model_path) if train_model_path is not None else train_model])

        if self.skip_train_check.isChecked():
            args.append("--skip-train")

        onnx_name = self.export_onnx_name_edit.text().strip()
        if onnx_name:
            args.extend(["--export-onnx-name", onnx_name])
        if self.export_onnx_imgsz_spin.value() > 0:
            args.extend(["--export-onnx-imgsz", str(self.export_onnx_imgsz_spin.value())])

        self.training_log.clear()
        self.training_log.append(f"$ {python_bin} {' '.join(args)}")
        self.train_results_csv = None
        run_name = self.run_name_edit.text().strip() or "pseudo_label_train"
        self.optimize_plot_window = self._create_plot_window(f"Optimize Plot: {run_name}")

        self.train_process = QProcess(self)
        self.train_process.setWorkingDirectory(str(self._project_root()))
        self.train_process.setProgram(python_bin)
        self.train_process.setArguments(args)
        self.train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self._on_training_output)
        self.train_process.finished.connect(self._on_training_finished)
        self.train_process.start()

        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_plot_timer.start(2000)
        self.statusBar().showMessage("Training pipeline started", 3000)

    def stop_training_pipeline(self) -> None:
        if self.train_process is None or self.train_process.state() == QProcess.NotRunning:
            return
        self.train_process.kill()
        self.train_plot_timer.stop()
        self.training_log.append("\n[Training process stopped]")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.statusBar().showMessage("Training pipeline stopped", 3000)

    def _on_training_output(self) -> None:
        if self.train_process is None:
            return
        text = bytes(self.train_process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.training_log.insertPlainText(text)
            self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
            csv_path = self._extract_results_csv_from_output(text)
            if csv_path is not None:
                self.train_results_csv = csv_path
            self._update_optimize_training_plot()

    def _on_training_finished(self, exit_code: int, exit_status: int) -> None:
        _ = exit_status
        self.train_plot_timer.stop()
        self._update_optimize_training_plot()
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        if exit_code == 0:
            self.training_log.append("\n[Training pipeline finished successfully]")
            self.statusBar().showMessage("Training pipeline finished", 5000)
        else:
            self.training_log.append(f"\n[Training pipeline failed, exit code {exit_code}]")
            self.statusBar().showMessage("Training pipeline failed", 5000)

    def start_model_training(self) -> None:
        if self.model_train_process is not None and self.model_train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Training", "A model training process is already running.")
            return

        data_yaml = self.obb_data_edit.text().strip()
        task = self.train_task_combo.currentText().strip().lower() or "obb"
        if task not in {"obb", "detect"}:
            task = "obb"
        default_model = "yolo11n-obb.pt" if task == "obb" else "yolo11n.pt"
        default_project, default_run_name = self._default_training_project_and_name()
        model = self.obb_model_edit.text().strip() or default_model
        if not self._validate_training_task_model_combo(task, model):
            return
        if not data_yaml:
            QMessageBox.warning(self, "Missing Input", "Please choose --data yaml.")
            return
        data_yaml_path = self._resolve_existing_path(data_yaml)
        if data_yaml_path is None:
            QMessageBox.warning(self, "Missing Input", f"Dataset yaml not found:\n{data_yaml}")
            return
        data_yaml = str(data_yaml_path)
        model_path = self._resolve_existing_path(model)
        if model_path is not None:
            model = str(model_path)

        root = self._project_root()
        platform_is_amd = self.train_platform_combo.currentText().startswith("AMD")
        device_value = self.obb_device_edit.text().strip() or "0"
        workers_value = str(self.obb_workers_spin.value())
        project_value = self.obb_project_edit.text().strip() or default_project
        run_name = self.obb_name_edit.text().strip() or default_run_name
        args: list[str]
        process_env: QProcessEnvironment | None = None

        if platform_is_amd and task == "obb":
            script_path = root / "scripts" / "bash" / "train_obb_amd.sh"
            if not script_path.exists():
                QMessageBox.warning(self, "Missing Script", f"Script not found:\n{script_path}")
                return
            args = [
                str(script_path),
                "--data",
                data_yaml,
                "--model",
                model,
                "--epochs",
                str(self.obb_epochs_spin.value()),
                "--imgsz",
                str(self.obb_imgsz_spin.value()),
                "--batch",
                str(self.obb_batch_spin.value()),
                "--device",
                device_value,
                "--workers",
                workers_value,
                "--project",
                project_value,
                "--name",
                run_name,
                "--hsa-gfx",
                self.obb_hsa_edit.text().strip() or "10.3.0",
                "--val",
                "True" if self.obb_val_check.isChecked() else "False",
                "--amp",
                "True" if self.obb_amp_check.isChecked() else "False",
            ]
            program = "bash"
        elif platform_is_amd:
            python_bin = self._select_training_python()
            if not Path(python_bin).exists():
                QMessageBox.warning(self, "Missing Python", f"Python runtime not found:\n{python_bin}")
                return
            program = python_bin
            args = [
                "-c",
                "from ultralytics.cfg import entrypoint; entrypoint()",
                task,
                "train",
                f"data={data_yaml}",
                f"model={model}",
                f"epochs={self.obb_epochs_spin.value()}",
                f"imgsz={self.obb_imgsz_spin.value()}",
                f"batch={self.obb_batch_spin.value()}",
                f"device={device_value}",
                f"workers={workers_value}",
                f"project={project_value}",
                f"name={run_name}",
                f"val={'True' if self.obb_val_check.isChecked() else 'False'}",
                f"amp={'True' if self.obb_amp_check.isChecked() else 'False'}",
            ]
            process_env = QProcessEnvironment.systemEnvironment()
            process_env.insert("HSA_OVERRIDE_GFX_VERSION", self.obb_hsa_edit.text().strip() or "10.3.0")
        else:
            yolo_bin = root / ".venv" / "bin" / "yolo"
            program = str(yolo_bin) if yolo_bin.exists() else "yolo"
            args = [
                task,
                "train",
                f"data={data_yaml}",
                f"model={model}",
                f"epochs={self.obb_epochs_spin.value()}",
                f"imgsz={self.obb_imgsz_spin.value()}",
                f"batch={self.obb_batch_spin.value()}",
                f"device={device_value}",
                f"workers={workers_value}",
                f"project={project_value}",
                f"name={run_name}",
                f"val={'True' if self.obb_val_check.isChecked() else 'False'}",
                f"amp={'True' if self.obb_amp_check.isChecked() else 'False'}",
            ]

        self.model_training_log.clear()
        self.model_training_log.append(f"$ {program} {' '.join(args)}")
        self.model_train_results_csv = None
        self.model_plot_window = self._create_plot_window(f"Training Plot: {run_name}")

        self.model_train_process = QProcess(self)
        self.model_train_process.setWorkingDirectory(str(root))
        if process_env is not None:
            self.model_train_process.setProcessEnvironment(process_env)
        self.model_train_process.setProgram(program)
        self.model_train_process.setArguments(args)
        self.model_train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.model_train_process.readyReadStandardOutput.connect(self._on_model_training_output)
        self.model_train_process.finished.connect(self._on_model_training_finished)
        self.model_train_process.start()

        self.start_model_train_btn.setEnabled(False)
        self.stop_model_train_btn.setEnabled(True)
        self.model_train_plot_timer.start(2000)
        self.statusBar().showMessage("Model training started", 3000)

    def stop_model_training(self) -> None:
        if self.model_train_process is None or self.model_train_process.state() == QProcess.NotRunning:
            return
        self.model_train_process.kill()
        self.model_train_plot_timer.stop()
        self.model_training_log.append("\n[Model training stopped]")
        self.start_model_train_btn.setEnabled(True)
        self.stop_model_train_btn.setEnabled(False)
        self.statusBar().showMessage("Model training stopped", 3000)

    def _on_model_training_output(self) -> None:
        if self.model_train_process is None:
            return
        text = bytes(self.model_train_process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.model_training_log.insertPlainText(text)
            self.model_training_log.verticalScrollBar().setValue(
                self.model_training_log.verticalScrollBar().maximum()
            )
            csv_path = self._extract_results_csv_from_output(text)
            if csv_path is not None:
                self.model_train_results_csv = csv_path
            self._update_model_training_plot()

    def _on_model_training_finished(self, exit_code: int, exit_status: int) -> None:
        _ = exit_status
        self.model_train_plot_timer.stop()
        self._update_model_training_plot()
        self.start_model_train_btn.setEnabled(True)
        self.stop_model_train_btn.setEnabled(False)
        if exit_code == 0:
            self.model_training_log.append("\n[Model training finished successfully]")
            self.statusBar().showMessage("Model training finished", 5000)
            self._start_model_export_onnx()
        else:
            self.model_training_log.append(f"\n[Model training failed, exit code {exit_code}]")
            self.statusBar().showMessage("Model training failed", 5000)

    def _refresh_nav_buttons(self) -> None:
        if self.source_mode == "images":
            has_images = len(self.image_paths) > 0
            self.prev_btn.setEnabled(has_images and self.index > 0)
            self.next_btn.setEnabled(has_images and self.index < len(self.image_paths) - 1)
            return

        if self.source_mode == "video" and self.video_capture is not None:
            self.prev_btn.setEnabled(self.video_frame_idx > 0)
            self.next_btn.setEnabled(self.video_frame_idx < self.video_frame_count - 1)
            return

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def _on_confidence_changed(self, value: int) -> None:
        self.conf_label.setText(f"{value}%")

    def _rerun_current(self) -> None:
        if self.source_mode == "images" and self.index >= 0:
            self.show_image(self.index)
        elif self.source_mode == "video" and self.video_frame_idx >= 0:
            self.show_video_frame(self.video_frame_idx)

    def load_model(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.onnx *.engine *.torchscript);;All Files (*)",
        )
        if not file_name:
            return
        try:
            self.model_path = Path(file_name)
            if self.model_path.suffix.lower() == ".onnx":
                task = self._read_onnx_task(self.model_path)
                if task is not None:
                    self.model = YOLO(file_name, task=task)
                    self.statusBar().showMessage(f"ONNX model loaded with task={task}", 3000)
                else:
                    # OBB ONNX often fails if task is auto-guessed as detect.
                    fallback_task = "obb" if "obb" in self.model_path.name.lower() else "detect"
                    self.model = YOLO(file_name, task=fallback_task)
                    self.statusBar().showMessage(
                        f"ONNX metadata task not found, using task={fallback_task}",
                        5000,
                    )
            else:
                self.model = YOLO(file_name)
            self.model_label.setText(f"Model: {self.model_path}")
            self.statusBar().showMessage("Model loaded", 3000)
            self._rerun_current()
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", f"Failed to load model:\n{exc}")

    @staticmethod
    def _read_onnx_task(model_path: Path) -> str | None:
        try:
            import onnx
        except Exception:
            return None

        try:
            model = onnx.load(str(model_path), load_external_data=False)
            metadata = {p.key: p.value for p in model.metadata_props}
            task = metadata.get("task", "").strip().lower()
            if task in {"detect", "segment", "classify", "pose", "obb"}:
                return task
        except Exception:
            return None
        return None

    def _clear_video(self) -> None:
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = None
        self.video_path = None
        self.video_frame_count = 0
        self.video_frame_idx = -1
        self.video_fps = 30.0

    def load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.image_paths = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "No supported images found in this folder.")
            return

        self._clear_video()
        self.source_mode = "images"
        self.index = 0
        self.show_image(self.index)

    def load_video(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All Files (*)",
        )
        if not file_name:
            return

        video_path = Path(file_name)
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            QMessageBox.warning(self, "Video Error", "Unsupported video extension.")
            return

        self._clear_video()
        self.image_paths = []
        self.index = -1
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(self, "Video Error", f"Failed to open video:\n{video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.video_capture = cap
        self.video_path = video_path
        self.video_frame_count = max(1, frame_count)
        self.video_frame_idx = 0
        self.video_fps = fps if fps > 0 else 30.0
        self.source_mode = "video"
        self._start_video_loop()
        self.show_video_frame(0)

    def _start_video_loop(self) -> None:
        if self.source_mode != "video":
            return
        interval_ms = max(1, int(1000.0 / self.video_fps))
        self.video_timer.start(interval_ms)

    def _advance_video_loop(self) -> None:
        if self.source_mode != "video" or self.video_capture is None:
            self.video_timer.stop()
            return
        if self.video_frame_count <= 0:
            return

        next_idx = self.video_frame_idx + 1
        if next_idx >= self.video_frame_count:
            next_idx = 0  # loop video
        self.show_video_frame(next_idx)

    def prev_image(self) -> None:
        if self.source_mode == "images" and self.index > 0:
            self.index -= 1
            self.show_image(self.index)
        elif self.source_mode == "video" and self.video_frame_idx > 0:
            self.show_video_frame(self.video_frame_idx - 1)

    def next_image(self) -> None:
        if self.source_mode == "images" and self.index < len(self.image_paths) - 1:
            self.index += 1
            self.show_image(self.index)
        elif self.source_mode == "video" and self.video_frame_idx < self.video_frame_count - 1:
            self.show_video_frame(self.video_frame_idx + 1)

    def _class_name(self, result, class_id: int) -> str:
        names = result.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _build_summary(self, result) -> str:
        if hasattr(result, "probs") and result.probs is not None:
            top1 = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf.item()) * 100.0
            return f"Class: {self._class_name(result, top1)} ({top1_conf:.1f}%)"

        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            lines = [f"OBB detections: {len(obb)}"]
            for i, box in enumerate(obb[:10]):
                class_id = int(box.cls.item())
                conf = float(box.conf.item()) * 100.0
                lines.append(f"{i + 1}. {self._class_name(result, class_id)} ({conf:.1f}%)")
            if len(obb) > 10:
                lines.append(f"... +{len(obb) - 10} more")
            return " | ".join(lines)

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return f"No detections above {self.conf_slider.value()}%"

        lines = [f"Detections: {len(boxes)}"]
        for i, box in enumerate(boxes[:10]):
            class_id = int(box.cls.item())
            conf = float(box.conf.item()) * 100.0
            lines.append(f"{i + 1}. {self._class_name(result, class_id)} ({conf:.1f}%)")
        if len(boxes) > 10:
            lines.append(f"... +{len(boxes) - 10} more")
        return " | ".join(lines)

    def _infer_frame(self, frame):
        summary = "Model not loaded"
        display = frame
        if self.model is not None:
            try:
                conf = self.conf_slider.value() / 100.0
                predict_kwargs = {
                    "source": frame,
                    "conf": conf,
                    "verbose": False,
                }
                # ONNX models may try CUDAExecutionProvider by default, which breaks on AMD/CPU-only stacks.
                if self.model_path is not None and self.model_path.suffix.lower() == ".onnx":
                    predict_kwargs["device"] = "cpu"
                result = self.model.predict(**predict_kwargs)[0]
                display = result.plot()
                summary = self._build_summary(result)
            except Exception as exc:
                summary = f"Inference error: {exc}"
        return display, summary

    def _render_frame(self, bgr_frame, summary: str, info_text: str) -> None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        bytes_per_line = channels * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.current_qimage = qimg
        self.info_label.setText(info_text)
        self.classification_label.setText(f"Classification: {summary}")
        self._refresh_image_view()
        self._refresh_nav_buttons()

    def show_image(self, index: int) -> None:
        if not (0 <= index < len(self.image_paths)):
            return

        image_path = self.image_paths[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            QMessageBox.warning(self, "Image Error", f"Failed to read:\n{image_path}")
            return

        display, summary = self._infer_frame(frame)
        self._render_frame(
            display,
            summary,
            f"Image: {index + 1}/{len(self.image_paths)} - {image_path.name}",
        )

    def show_video_frame(self, frame_idx: int) -> None:
        if self.video_capture is None or self.video_path is None:
            return
        if frame_idx < 0 or frame_idx >= self.video_frame_count:
            return

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            QMessageBox.warning(self, "Video Error", "Failed to read requested frame.")
            return

        self.video_frame_idx = frame_idx
        display, summary = self._infer_frame(frame)
        self._render_frame(
            display,
            summary,
            f"Video: {self.video_path.name} | Frame: {frame_idx + 1}/{self.video_frame_count}",
        )

    def _refresh_image_view(self) -> None:
        if self.current_qimage is None:
            return
        target_size = self.image_label.size()
        pixmap = QPixmap.fromImage(self.current_qimage).scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_image_view()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.train_process is not None and self.train_process.state() != QProcess.NotRunning:
            self.train_process.kill()
        if self.model_train_process is not None and self.model_train_process.state() != QProcess.NotRunning:
            self.model_train_process.kill()
        if self.model_export_process is not None and self.model_export_process.state() != QProcess.NotRunning:
            self.model_export_process.kill()
        if self.converter_process is not None and self.converter_process.state() != QProcess.NotRunning:
            self.converter_process.kill()
        self.train_plot_timer.stop()
        self.model_train_plot_timer.stop()
        self._clear_video()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    viewer = ModelViewer()
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
