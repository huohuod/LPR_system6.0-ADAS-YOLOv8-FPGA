"""
PySide6主界面模块 - v5.3
macOS 深色风格，中性暖灰配色
"""
import sys
import os
import time
from datetime import datetime
from typing import List, Optional
from collections import deque

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QFrame, QToolBar, QSizePolicy,
    QFileDialog
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QAction, QDragEnterEvent, QDropEvent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import ConfigManager
from src.core.engine import LPREngine
from src.pipeline.async_pipeline import AsyncPipeline
from src.pipeline.stages import FramePacket
from src.utils.draw_text import draw_chinese_text
from src.ui.performance_monitor import PerformanceMonitor

# ── 全局样式 ──────────────────────────────────────────────
# 色板: 纯中性灰，无蓝无紫
# 背景层: #1c1c1e → #2c2c2e → #3a3a3c
# 文字层: #e5e5ea → #aeaeb2 → #636366
# 强调色: #30d158 (绿，仅用于运行状态)
_FONT = "'PingFang SC', 'Microsoft YaHei', 'SF Pro Text', 'Helvetica Neue', sans-serif"

_STYLE = f"""
QMainWindow, QWidget {{
    background-color: #1c1c1e;
    color: #e5e5ea;
    font-family: {_FONT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}

QPushButton {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: none;
    padding: 7px 18px;
    border-radius: 8px;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: #3a3a3c; }}
QPushButton:pressed {{ background-color: #1c1c1e; }}
QPushButton:disabled {{ background-color: #2c2c2e; color: #636366; }}

QComboBox {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: none;
    padding: 5px 10px;
    border-radius: 8px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    selection-background-color: #3a3a3c;
    selection-color: #fff;
    border: none;
    border-radius: 8px;
}}

QTextEdit {{
    background-color: #2c2c2e;
    color: #aeaeb2;
    border: none;
    border-radius: 8px;
    font-family: 'Menlo', 'Consolas', 'PingFang SC', monospace;
    font-size: 12px;
    padding: 8px;
}}

QToolBar {{
    background-color: #2c2c2e;
    border-bottom: 1px solid #3a3a3c;
    spacing: 2px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: #aeaeb2;
    border: none;
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 12px;
}}
QToolBar QToolButton:hover {{
    background-color: #3a3a3c;
    color: #e5e5ea;
}}
QToolBar::separator {{
    width: 1px;
    background-color: #3a3a3c;
    margin: 6px 4px;
}}

QStatusBar {{
    background-color: #1c1c1e;
    color: #636366;
    font-size: 12px;
    border-top: 1px solid #2c2c2e;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: #3a3a3c;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background-color: #48484a; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""


class CameraThread(QThread):
    frame_ready = Signal(object)

    def __init__(self, source: int):
        super().__init__()
        self.source = source
        self.cap = None
        self.running = False
        self._frame = None
        self._lock = None

    def run(self) -> None:
        self._lock = self._lock or __import__('threading').Lock()
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.source, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = self.cap.read()
                    if ret:
                        break
                    self.cap.release()
                    self.cap = None
            except Exception:
                continue

        if self.cap is None or not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {self.source}")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        if self._lock:
            with self._lock:
                return self._frame.copy() if self._frame is not None else None
        return None

    def stop(self) -> None:
        self.running = False
        self.wait(3000)


class VideoFileThread(QThread):
    frame_ready = Signal(object)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.cap = None
        self.running = False
        self._frame = None
        self._lock = None
        self._fps = 30.0

    def run(self) -> None:
        self._lock = __import__('threading').Lock()
        # 处理中文路径兼容性
        for backend_name, backend_id in [('FFMPEG', cv2.CAP_FFMPEG), ('DSHOW', cv2.CAP_DSHOW), ('ANY', cv2.CAP_ANY)]:
            try:
                self.cap = cv2.VideoCapture(self.file_path, backend_id)
                if self.cap.isOpened():
                    ret, test = self.cap.read()
                    if ret and test is not None:
                        print(f"[VIDEO] 已加载: {self.file_path} ({backend_name})")
                        break
                    self.cap.release(); self.cap = None
            except:
                self.cap = None
        if self.cap is None or not self.cap.isOpened():
            print(f"[VIDEO] 无法打开文件: {self.file_path}")
            return
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / self._fps
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(frame_interval)
        self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        if self._lock:
            with self._lock:
                return self._frame.copy() if self._frame is not None else None
        return None

    def stop(self) -> None:
        self.running = False
        self.wait(3000)


# ── 辅助组件 ──────────────────────────────────────────────

class SectionCard(QFrame):
    """macOS 风格圆角卡片"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #2c2c2e;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(10)

        header = QLabel(title)
        header.setStyleSheet(f"color: #636366; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; border: none; font-family: {_FONT};")
        layout.addWidget(header)

        self.content = QVBoxLayout()
        self.content.setSpacing(8)
        layout.addLayout(self.content)

    def add_widget(self, w):
        self.content.addWidget(w)

    def add_layout(self, l):
        self.content.addLayout(l)


class StatusIndicator(QWidget):
    """状态指示灯 + 数值"""

    def __init__(self, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self._unit = unit

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        name = QLabel(label)
        name.setStyleSheet(f"color: #636366; font-size: 12px; border: none; font-family: {_FONT};")
        layout.addWidget(name)

        layout.addStretch()

        self._value = QLabel("--")
        self._value.setStyleSheet(f"color: #e5e5ea; font-size: 13px; font-weight: 600; border: none; font-family: {_FONT};")
        layout.addWidget(self._value)

    def set_value(self, v: float):
        self._value.setText(f"{v:.0f}{self._unit}")

    def set_color(self, color: str):
        self._value.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600; border: none; font-family: {_FONT};")


# ── 主窗口 ────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, engine: Optional[LPREngine] = None,
                 pipeline: Optional[AsyncPipeline] = None,
                 config_manager: Optional[ConfigManager] = None,
                 source: int = 0):
        super().__init__()
        self.engine = engine
        self.pipeline = pipeline
        self.cfg = config_manager
        self.source = source

        self.camera_thread: Optional[CameraThread] = None
        self.video_file_thread: Optional[VideoFileThread] = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0.0
        self.detection_log: List[str] = []
        self._last_log_index = 0
        self.current_source_type = 'none'
        self.video_file_path = ''

        self.fps_timestamps: deque = deque(maxlen=60)
        self._latest_result_frame = None
        self._latest_detections = []

        self._setup_ui()
        self._setup_toolbar()
        self._setup_timer()
        self.setAcceptDrops(True)

    # ── UI 构建 ──────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setWindowTitle("LPR System")
        self.setGeometry(60, 60, 1480, 880)
        self.setMinimumSize(1100, 650)
        self.setStyleSheet(_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 左侧：视频区 ──
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border-radius: 12px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: #3a3a3c;
                font-size: 15px;
                font-weight: 500;
                letter-spacing: 3px;
                border-radius: 12px;
            }
        """)
        self._set_no_signal()
        video_layout.addWidget(self.video_label)
        root.addWidget(video_frame, stretch=1)

        # ── 右侧：控制面板 ──
        panel = QWidget()
        panel.setFixedWidth(320)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(8)

        # 标题
        title_bar = QFrame()
        title_bar.setStyleSheet("QFrame { background-color: #2c2c2e; border-radius: 12px; }")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 14, 16, 14)

        title_text = QVBoxLayout()
        title_text.setSpacing(2)
        t1 = QLabel("LPR System")
        t1.setStyleSheet(f"color: #f5f5f7; font-size: 17px; font-weight: 600; border: none; font-family: {_FONT};")
        t2 = QLabel("License Plate Recognition")
        t2.setStyleSheet(f"color: #636366; font-size: 11px; border: none; font-family: {_FONT};")
        title_text.addWidget(t1)
        title_text.addWidget(t2)
        title_layout.addLayout(title_text)
        title_layout.addStretch()

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(10, 10)
        self.status_dot.setStyleSheet("background-color: #48484a; border-radius: 5px; border: none;")
        title_layout.addWidget(self.status_dot)

        pl.addWidget(title_bar)

        # 视频源
        src_card = SectionCard("Source")
        src_row = QHBoxLayout()
        src_row.setSpacing(6)

        self.btn_import = QPushButton("File")
        self.btn_import.setFixedHeight(32)
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.clicked.connect(self.import_video_file)
        src_row.addWidget(self.btn_import)

        self.cam_combo = QComboBox()
        self.cam_combo.setFixedHeight(32)
        for i in range(4):
            self.cam_combo.addItem(f"Camera {i}", i)
        src_row.addWidget(self.cam_combo, stretch=1)

        src_card.add_layout(src_row)

        self.lbl_source = QLabel("No source selected")
        self.lbl_source.setStyleSheet(f"color: #636366; font-size: 11px; border: none; font-family: {_FONT};")
        self.lbl_source.setWordWrap(True)
        src_card.add_widget(self.lbl_source)
        pl.addWidget(src_card)

        # 控制按钮
        self.btn_toggle = QPushButton("Start Detection")
        self.btn_toggle.setFixedHeight(44)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                background-color: #1c1c1e;
                color: #30d158;
                border: 1.5px solid #30d158;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0.5px;
                font-family: {_FONT};
            }}
            QPushButton:hover {{
                background-color: #1a3d25;
            }}
        """)
        self.btn_toggle.clicked.connect(self.toggle_capture)
        pl.addWidget(self.btn_toggle)

        # 性能
        perf_card = SectionCard("Performance")
        self.perf_monitor = PerformanceMonitor()
        perf_card.add_widget(self.perf_monitor)

        self.fps_indicator = StatusIndicator("FPS")
        self.lat_indicator = StatusIndicator("Latency", "ms")
        self.det_indicator = StatusIndicator("Total")

        perf_card.add_widget(self.fps_indicator)
        perf_card.add_widget(self.lat_indicator)
        perf_card.add_widget(self.det_indicator)
        pl.addWidget(perf_card)

        # 日志
        log_card = SectionCard("Log")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        log_card.add_widget(self.log_text)
        pl.addWidget(log_card, stretch=1)

        root.addWidget(panel)

        self.statusBar().showMessage("Ready")

    def _set_no_signal(self):
        self.video_label.setText("NO SIGNAL")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: #2c2c2e;
                font-size: 15px;
                font-weight: 500;
                letter-spacing: 4px;
                border-radius: 12px;
            }
        """)

    def _setup_toolbar(self) -> None:
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        settings_act = QAction("Settings", self)
        settings_act.triggered.connect(self.open_settings)
        tb.addAction(settings_act)

        tb.addSeparator()

        cap_act = QAction("Screenshot", self)
        cap_act.triggered.connect(self.take_screenshot)
        tb.addAction(cap_act)

        tb.addSeparator()

        about_act = QAction("About", self)
        about_act.triggered.connect(self.show_about)
        tb.addAction(about_act)

    def _setup_timer(self) -> None:
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.on_frame)
        self.display_timer.setInterval(33)

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self._update_log)
        self.log_timer.start(500)

    # ── 事件 ──────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
                self._load_video_file(path)
                break

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_capture()
        elif key == Qt.Key_S:
            self.take_screenshot()
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    # ── 视频源 ────────────────────────────────────────────

    def import_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;所有 (*)"
        )
        if path:
            self._load_video_file(path)
            self.toggle_capture()

    def _load_video_file(self, path: str) -> None:
        self.video_file_path = path
        self.current_source_type = 'video_file'
        name = os.path.basename(path)
        self.lbl_source.setText(name)

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()
            self.statusBar().showMessage(f"{name}  {w}x{h} @ {fps:.0f}fps")
        else:
            self.statusBar().showMessage(f"无法打开: {name}")

    # ── 检测控制 ──────────────────────────────────────────

    def toggle_capture(self) -> None:
        if self.is_running:
            self.stop_capture()
        else:
            self.start_detection()

    def _set_running_style(self):
        self.btn_toggle.setText("Stop Detection")
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                background-color: #1c1c1e;
                color: #ff453a;
                border: 1.5px solid #ff453a;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0.5px;
                font-family: {_FONT};
            }}
            QPushButton:hover {{
                background-color: #3d1a1a;
            }}
        """)
        self.status_dot.setStyleSheet("background-color: #30d158; border-radius: 5px; border: none;")

    def _set_stopped_style(self):
        self.btn_toggle.setText("Start Detection")
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                background-color: #1c1c1e;
                color: #30d158;
                border: 1.5px solid #30d158;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0.5px;
                font-family: {_FONT};
            }}
            QPushButton:hover {{
                background-color: #1a3d25;
            }}
        """)
        self.status_dot.setStyleSheet("background-color: #48484a; border-radius: 5px; border: none;")

    def start_detection(self) -> None:
        if not self.engine or not self.engine.is_ready:
            self.statusBar().showMessage("Engine not ready")
            return

        if self.current_source_type == 'none':
            self.current_source_type = 'camera'
            self.source = self.cam_combo.currentData()
            self.lbl_source.setText(f"Camera {self.source}")

        if self.current_source_type == 'camera':
            self.camera_thread = CameraThread(self.source)
            self.camera_thread.start()
        elif self.current_source_type == 'video_file':
            if not self.video_file_path:
                self.statusBar().showMessage("请先导入视频")
                return
            self.video_file_thread = VideoFileThread(self.video_file_path)
            self.video_file_thread.start()

        if self.pipeline:
            self.pipeline.start()

        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_timestamps.clear()
        self.detection_log.clear()
        self._last_log_index = 0
        self.log_text.clear()
        self._latest_result_frame = None
        self._latest_detections = []

        self.display_timer.start(33)
        self._set_running_style()
        self.btn_import.setEnabled(False)
        self.statusBar().showMessage("Running")

    def stop_capture(self) -> None:
        self.is_running = False
        self.display_timer.stop()

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        if self.video_file_thread:
            self.video_file_thread.stop()
            self.video_file_thread = None
        if self.pipeline:
            self.pipeline.stop()

        self._set_stopped_style()
        self.btn_import.setEnabled(True)
        self._set_no_signal()
        self.statusBar().showMessage("Stopped")

    # ── 帧处理 ────────────────────────────────────────────

    def on_frame(self) -> None:
        if not self.is_running:
            return

        raw_frame = None
        if self.current_source_type == 'camera' and self.camera_thread:
            raw_frame = self.camera_thread.get_frame()
        elif self.current_source_type == 'video_file' and self.video_file_thread:
            raw_frame = self.video_file_thread.get_frame()

        if raw_frame is None:
            return

        self.frame_count += 1
        self.fps_timestamps.append(time.time())

        if self.pipeline and self.pipeline.is_running:
            packet = FramePacket(raw_frame)
            self.pipeline.submit(packet)
            result_packet = self.pipeline.get_result(timeout=0.001)
            if result_packet is not None:
                self._latest_result_frame = result_packet.result_frame
                self._latest_detections = result_packet.detections
        elif self.engine and self.engine.is_ready:
            result_frame, detections, _ = self.engine.detect(raw_frame)
            self._latest_result_frame = result_frame
            self._latest_detections = detections

        if self._latest_result_frame is not None:
            fps = self._calc_fps()

            display = self._latest_result_frame.copy()
            cv2.putText(display, f"{fps:.0f}", (12, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            pixmap = self._cv2_to_pixmap(display, self.video_label.size())
            self.video_label.setPixmap(pixmap)

            self.fps_indicator.set_value(fps)
            if self.engine:
                lat = self.engine.stats['avg_latency']
                self.lat_indicator.set_value(lat)
                if lat < 30:
                    self.lat_indicator.set_color("#30d158")
                elif lat < 80:
                    self.lat_indicator.set_color("#ff9f0a")
                else:
                    self.lat_indicator.set_color("#ff453a")
                self.det_indicator.set_value(self.engine.stats['detection_count'])

            for det in self._latest_detections[:5]:
                plate = det.get('plate_no', '')
                ts = datetime.now().strftime("%H:%M:%S")
                conf = det.get('confidence', det.get('score', 0))
                if plate:
                    self.detection_log.append(f"[{ts}] {plate}  {conf:.2f}")
                else:
                    self.detection_log.append(f"[{ts}] target  {conf:.2f}")
            if len(self.detection_log) > 80:
                self.detection_log = self.detection_log[-80:]

    def _calc_fps(self) -> float:
        if len(self.fps_timestamps) < 2:
            return 0.0
        elapsed = self.fps_timestamps[-1] - self.fps_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.fps_timestamps) - 1) / elapsed

    def _cv2_to_pixmap(self, cv_img: np.ndarray, size) -> QPixmap:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    # ── 功能 ──────────────────────────────────────────────

    def take_screenshot(self) -> None:
        if self._latest_result_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存截图", f"lpr_{ts}.jpg",
            "JPEG (*.jpg);;PNG (*.png)"
        )
        if path:
            cv2.imwrite(path, self._latest_result_frame)
            self.statusBar().showMessage(f"Saved: {path}")

    def open_settings(self) -> None:
        from src.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            model_cfg = dlg.get_model_config()
            color_cfg = dlg.get_color_config()

            if self.engine:
                self.engine.model_config.conf_threshold = model_cfg['conf']
                self.engine.model_config.iou_threshold = model_cfg['iou']
                self.engine.model_config.img_size = model_cfg['img_size']

            if self.cfg:
                self.cfg.config.model.conf_threshold = model_cfg['conf']
                self.cfg.config.model.iou_threshold = model_cfg['iou']
                self.cfg.config.model.img_size = model_cfg['img_size']
                self.cfg.config.color.brightness = color_cfg['brightness']
                self.cfg.config.color.contrast = color_cfg['contrast']
                self.cfg.config.color.saturation = color_cfg['saturation']
                self.cfg.config.color.wb_r = color_cfg['wb_r']
                self.cfg.config.color.wb_g = color_cfg['wb_g']
                self.cfg.config.color.wb_b = color_cfg['wb_b']
                self.cfg.save()

            self.statusBar().showMessage("Settings saved")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<h3 style='color:#f5f5f7'>LPR System v5.3</h3>"
            "<p style='color:#aeaeb2'>License Plate Recognition</p>"
            "<p style='color:#aeaeb2'>YOLO + HyperLPR3 OCR</p>"
            "<p style='color:#636366; font-size:11px'>Space: Start/Stop &nbsp;&middot;&nbsp; S: Screenshot &nbsp;&middot;&nbsp; Esc: Quit</p>"
        )

    def _update_log(self) -> None:
        if self._last_log_index >= len(self.detection_log):
            return
        for line in self.detection_log[self._last_log_index:]:
            self.log_text.append(line)
        self._last_log_index = len(self.detection_log)
        if len(self.detection_log) > 80:
            self.detection_log = self.detection_log[-80:]
            self._last_log_index = len(self.detection_log)
        sb = self.log_text.verticalScrollBar()
        if sb.maximum() > 0:
            sb.setValue(sb.maximum())

    def closeEvent(self, event) -> None:
        self.stop_capture()
        event.accept()
