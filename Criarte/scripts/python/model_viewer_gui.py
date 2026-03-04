#!/usr/bin/env python3
"""PyQt GUI to browse images and run YOLO predictions with adjustable confidence."""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

try:
    from PyQt5.QtCore import QLibraryInfo, QProcess, QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
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
    os.environ["QT_QPA_PLATFORM"] = "wayland"  # or "xcb" on X11, "windows" on Windows, etc.

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
        self.model_train_plot_timer = QTimer(self)
        self.model_train_plot_timer.timeout.connect(self._update_model_training_plot)
        self.model_train_results_csv: Path | None = None
        self.optimize_plot_window: FloatingTrainingPlotWindow | None = None
        self.model_plot_window: FloatingTrainingPlotWindow | None = None
        self.plot_windows: list[FloatingTrainingPlotWindow] = []

        self._apply_professional_theme()
        self._build_ui()
        self._refresh_nav_buttons()

    def _apply_professional_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f4f7fa;
                color: #1f2a37;
                font-family: "Segoe UI", "Noto Sans", sans-serif;
                font-size: 13px;
            }
            QMainWindow {
                background-color: #e9eef4;
            }
            QWidget#headerCard {
                background-color: #ffffff;
                border: 1px solid #d7e1ea;
                border-radius: 12px;
            }
            QLabel#headerTitle {
                font-size: 20px;
                font-weight: 700;
                color: #10263e;
            }
            QLabel#headerSubtitle {
                font-size: 12px;
                color: #58708a;
            }
            QTabWidget::pane {
                border: 1px solid #cfd8e3;
                border-radius: 10px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #dde5ef;
                color: #1f2a37;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                font-weight: 600;
            }
            QPushButton {
                background-color: #dce7f3;
                border: 1px solid #c2d0e0;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #cfdeef;
            }
            QPushButton#primaryButton {
                background-color: #0a66c2;
                border: 1px solid #0858a8;
                color: #ffffff;
            }
            QPushButton#primaryButton:hover {
                background-color: #0858a8;
            }
            QPushButton#dangerButton {
                background-color: #d33f49;
                border: 1px solid #b2313a;
                color: #ffffff;
            }
            QPushButton#dangerButton:hover {
                background-color: #b2313a;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background-color: #ffffff;
                border: 1px solid #c8d4e1;
                border-radius: 8px;
                padding: 6px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QSlider::groove:horizontal {
                border-radius: 4px;
                height: 6px;
                background: #d0dbe7;
            }
            QSlider::handle:horizontal {
                background: #0a66c2;
                border: 1px solid #0858a8;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
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
        header_subtitle = QLabel("Inference, pseudo-label optimization, and OBB training")
        header_subtitle.setObjectName("headerSubtitle")
        header_layout.addWidget(header_title)
        header_layout.addWidget(header_subtitle)
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.inference_tab = self._create_scroll_tab()
        self.optimize_tab = self._create_scroll_tab()
        self.training_tab = self._create_scroll_tab()
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.optimize_tab, "Optimize")
        self.tabs.addTab(self.training_tab, "Training")

        self._build_inference_tab()
        self._build_optimize_tab()
        self._build_training_tab()

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
            "background-color: #122231; color: #dbe8f5; border-radius: 10px; border: 1px solid #203a54;"
        )
        self.image_label.setMinimumHeight(0)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

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

        self.obb_data_edit = QLineEdit("data/SandwichPanel.v7i.yolov8-obb/data.yaml")
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
        self.start_model_train_btn = QPushButton("Start OBB Training")
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

    def _pick_dir_into_lineedit(self, edit: QLineEdit, caption: str) -> None:
        dir_name = QFileDialog.getExistingDirectory(self, caption)
        if dir_name:
            edit.setText(dir_name)

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

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
        self.model_train_results_csv = self._refresh_results_csv_path(
            self.model_train_results_csv,
            self.obb_project_edit.text().strip() or "runs/obb",
            self.obb_name_edit.text().strip() or "sandwich_panel_obb_gui",
        )
        self._update_plot_window(self.model_train_results_csv, self.model_plot_window)

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

        mode = self._current_optimize_mode()
        python_bin = self._select_optimize_python(mode)
        if not self._validate_optimize_runtime(mode, python_bin):
            return

        script_path = Path(__file__).resolve().parent / "pseudo_label_and_train.py"
        args = [
            str(script_path),
            "--teacher-model",
            teacher_model,
            "--images-dir",
            images_dir,
            "--output-dir",
            output_dir,
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
            args.extend(["--train-model", train_model])

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
        model = self.obb_model_edit.text().strip() or "yolo11n-obb.pt"
        if not data_yaml:
            QMessageBox.warning(self, "Missing Input", "Please choose --data yaml.")
            return

        root = self._project_root()
        platform_is_amd = self.train_platform_combo.currentText().startswith("AMD")
        args: list[str]

        if platform_is_amd:
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
                self.obb_device_edit.text().strip() or "0",
                "--workers",
                str(self.obb_workers_spin.value()),
                "--project",
                self.obb_project_edit.text().strip() or "runs/obb",
                "--name",
                self.obb_name_edit.text().strip() or "sandwich_panel_obb_gui",
                "--hsa-gfx",
                self.obb_hsa_edit.text().strip() or "10.3.0",
                "--val",
                "True" if self.obb_val_check.isChecked() else "False",
                "--amp",
                "True" if self.obb_amp_check.isChecked() else "False",
            ]
            program = "bash"
        else:
            yolo_bin = root / ".venv" / "bin" / "yolo"
            program = str(yolo_bin) if yolo_bin.exists() else "yolo"
            args = [
                "obb",
                "train",
                f"data={data_yaml}",
                f"model={model}",
                f"epochs={self.obb_epochs_spin.value()}",
                f"imgsz={self.obb_imgsz_spin.value()}",
                f"batch={self.obb_batch_spin.value()}",
                f"device={self.obb_device_edit.text().strip() or '0'}",
                f"workers={self.obb_workers_spin.value()}",
                f"project={self.obb_project_edit.text().strip() or 'runs/obb'}",
                f"name={self.obb_name_edit.text().strip() or 'sandwich_panel_obb_gui'}",
                f"val={'True' if self.obb_val_check.isChecked() else 'False'}",
                f"amp={'True' if self.obb_amp_check.isChecked() else 'False'}",
            ]

        self.model_training_log.clear()
        self.model_training_log.append(f"$ {program} {' '.join(args)}")
        self.model_train_results_csv = None
        run_name = self.obb_name_edit.text().strip() or "sandwich_panel_obb_gui"
        self.model_plot_window = self._create_plot_window(f"Training Plot: {run_name}")

        self.model_train_process = QProcess(self)
        self.model_train_process.setWorkingDirectory(str(root))
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
            self.model = YOLO(file_name)
            self.model_path = Path(file_name)
            self.model_label.setText(f"Model: {self.model_path}")
            self.statusBar().showMessage("Model loaded", 3000)
            self._rerun_current()
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", f"Failed to load model:\n{exc}")

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
