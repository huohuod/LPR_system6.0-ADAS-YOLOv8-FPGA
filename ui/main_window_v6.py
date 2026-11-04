"""
LPR System v6.0 — Apple macOS Dark Mode
"""
import sys, os, time, math
from datetime import datetime
from typing import List
from collections import deque
import cv2, numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QApplication, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFrame, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTabWidget, QSlider,
    QLineEdit, QHeaderView, QAbstractItemView, QFileDialog,
    QProgressBar, QScrollArea, QInputDialog,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPointF, QSize
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QDragEnterEvent, QDropEvent, QLinearGradient, QPainterPath,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline.stages import FramePacket
from src.utils.quad_splitter import QuadSplitter

# ── macOS Dark Mode Color Tokens ──
C_WIN = "#1c1c1e"; C_SIDEBAR = "#252528"; C_SEC = "#2c2c2e"; C_TER = "#3a3a3c"
C_QUA = "#48484a"; C_WHITE = "#ffffff"; C_LGRAY = "#e5e5ea"; C_MGRAY = "#8e8e93"
C_DGRAY = "#636366"; C_BLUE = "#0a84ff"; C_PURPLE = "#bf5af2"; C_GREEN = "#30d158"
C_ORANGE = "#ff9f0a"; C_RED = "#ff453a"; C_DIV = "#38383a"
FF = "'PingFang SC', '-apple-system', 'SF Pro Text', 'Helvetica Neue', 'Microsoft YaHei', sans-serif"
FM = "'SF Mono', 'Menlo', 'Consolas', monospace"

STYLE = f"""
* {{ font-family: {FF}; }}
QMainWindow {{ background: {C_WIN}; }}
QWidget {{ background: {C_WIN}; color: {C_WHITE}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QPushButton {{ background: {C_SEC}; color: {C_WHITE}; border: none; padding: 6px 16px; border-radius: 6px; }}
QPushButton:hover {{ background: {C_TER}; }}
QPushButton:pressed {{ background: {C_QUA}; }}
QPushButton:disabled {{ background: {C_SEC}; color: {C_DGRAY}; }}
QComboBox {{ background: {C_SEC}; color: {C_WHITE}; border: none; padding: 6px 10px; border-radius: 6px; }}
QComboBox:hover {{ background: {C_TER}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: {C_SEC}; color: {C_WHITE}; selection-background-color: {C_TER}; border: none; border-radius: 8px; outline: none; padding: 4px; }}
QLineEdit {{ background: {C_SEC}; color: {C_WHITE}; border: none; padding: 7px 12px; border-radius: 6px; selection-background-color: {C_BLUE}; }}
QLineEdit:focus {{ background: {C_TER}; }}
QTextEdit {{ background: {C_SEC}; color: {C_MGRAY}; border: none; border-radius: 8px; font-family: {FM}; font-size: 12px; padding: 8px; selection-background-color: {C_BLUE}; }}
QTableWidget {{ background: {C_SEC}; color: {C_WHITE}; border: none; border-radius: 8px; gridline-color: {C_DIV}; selection-background-color: {C_TER}; font-size: 12px; }}
QTableWidget::item {{ padding: 6px 10px; border-bottom: 1px solid {C_DIV}; }}
QTableWidget::item:hover {{ background: {C_TER}; }}
QHeaderView::section {{ background: {C_SEC}; color: {C_MGRAY}; border: none; border-bottom: 1px solid {C_DIV}; padding: 8px 10px; font-weight: 600; font-size: 11px; }}
QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar::tab {{ background: transparent; color: {C_MGRAY}; padding: 8px 20px; border: none; border-radius: 6px; margin-right: 2px; }}
QTabBar::tab:hover {{ background: {C_SEC}; color: {C_LGRAY}; }}
QTabBar::tab:selected {{ background: {C_TER}; color: {C_WHITE}; }}
QSlider::groove:horizontal {{ background: {C_QUA}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; border: none; }}
QSlider::handle:horizontal:hover {{ background: #e0e0e0; }}
QSlider::sub-page:horizontal {{ background: {C_BLUE}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {C_QUA}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #5a5a5c; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QProgressBar {{ background: {C_QUA}; border: none; border-radius: 3px; height: 4px; text-align: center; }}
QProgressBar::chunk {{ border-radius: 3px; }}
"""

# ── Threads ──
class CameraThread(QThread):
    frame_ready = Signal(object)
    def __init__(self, source):
        super().__init__(); self.source = source; self.cap = None; self.running = False; self._frame = None; self._lock = None
    def run(self):
        self._lock = __import__('threading').Lock()
        for b in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
            try:
                self.cap = cv2.VideoCapture(self.source, b)
                if self.cap.isOpened(): self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); r, _ = self.cap.read()
                if self.cap and self.cap.isOpened() and r: break
                if self.cap: self.cap.release(); self.cap = None
            except: continue
        if not self.cap or not self.cap.isOpened(): return
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock: self._frame = frame
            else: time.sleep(0.005)
        if self.cap: self.cap.release()
    def get_frame(self):
        if self._lock:
            with self._lock: return self._frame.copy() if self._frame is not None else None
        return None
    def stop(self): self.running = False; self.wait(3000)

class VideoFileThread(QThread):
    frame_ready = Signal(object)
    def __init__(self, fp):
        super().__init__(); self.fp = fp; self.cap = None; self.running = False; self._frame = None; self._lock = None; self._fps = 30.0
    def run(self):
        self._lock = __import__('threading').Lock()
        # 处理中文路径: OpenCV 在某些 Windows 环境下对非 ASCII 路径兼容性差
        # 依次尝试 FFMPEG → DSHOW → 默认后端
        for backend_name, backend_id in [('FFMPEG', cv2.CAP_FFMPEG), ('DSHOW', cv2.CAP_DSHOW), ('ANY', cv2.CAP_ANY)]:
            try:
                self.cap = cv2.VideoCapture(self.fp, backend_id)
                if self.cap.isOpened():
                    ret, test = self.cap.read()
                    if ret and test is not None:
                        print(f"[VIDEO] 已加载: {self.fp} ({backend_name})")
                        break
                    self.cap.release(); self.cap = None
            except:
                self.cap = None
        if self.cap is None or not self.cap.isOpened():
            print(f"[VIDEO] 无法打开文件: {self.fp}")
            return
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[VIDEO] {w}x{h} @ {self._fps:.0f}fps")
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            with self._lock: self._frame = frame
            time.sleep(1.0 / self._fps)
        self.cap.release()
    def get_frame(self):
        if self._lock:
            with self._lock: return self._frame.copy() if self._frame is not None else None
        return None
    def stop(self): self.running = False; self.wait(3000)

# ── macOS-style Toggle ──
class MacToggle(QWidget):
    clicked = Signal()
    def __init__(self, checked=False, parent=None):
        super().__init__(parent); self.setFixedSize(38, 22); self._on = checked; self.setCursor(Qt.PointingHandCursor)
    def isChecked(self): return self._on
    def setChecked(self, v): self._on = v; self.update()
    def mousePressEvent(self, e): self._on = not self._on; self.update(); self.clicked.emit()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); th = 16; ty = (h - th) // 2
        if self._on:
            p.setPen(Qt.NoPen); p.setBrush(QColor(C_GREEN))
            p.drawRoundedRect(QRect(0, ty, w, th), th // 2, th // 2)
            p.setBrush(QColor(255, 255, 255)); p.drawEllipse(QRect(w - 18, ty + 1, 14, 14))
        else:
            p.setPen(Qt.NoPen); p.setBrush(QColor(C_QUA))
            p.drawRoundedRect(QRect(0, ty, w, th), th // 2, th // 2)
            p.setBrush(QColor(C_DGRAY)); p.drawEllipse(QRect(3, ty + 1, 14, 14))
        p.end()

# ── Painted sidebar icons (SF Symbol style) ──
def _paint_icon(p, kind, x, y, size, color):
    """Draw simple geometric icons resembling SF Symbols."""
    p.setPen(Qt.NoPen); p.setBrush(QColor(color))
    s = size; cx = x + s // 2; cy = y + s // 2; r = s // 2 - 1
    if kind == "monitor":
        # Rounded rect (screen) with small triangle
        p.drawRoundedRect(QRect(x + 1, y + 2, s - 2, s - 6), 3, 3)
        p.setBrush(QColor(C_WIN))
        p.drawRoundedRect(QRect(x + 3, y + 4, s - 6, s - 10), 2, 2)
        p.setBrush(QColor(color))
        # Small play triangle
        pts = [QPointF(x + s // 2 - 2, y + s // 2 - 3), QPointF(x + s // 2 - 2, y + s // 2 + 3), QPointF(x + s // 2 + 2, y + s // 2)]
        p.drawPolygon(pts)
    elif kind == "chart":
        # Three vertical bars
        bw = 3; gap = 2
        heights = [s * 0.4, s * 0.7, s * 0.55]
        start_x = cx - (3 * bw + 2 * gap) // 2
        for i, h in enumerate(heights):
            bx = start_x + i * (bw + gap)
            by = y + s - int(h) - 2
            p.drawRoundedRect(QRect(bx, by, bw, int(h)), 1, 1)
    elif kind == "list":
        # Three horizontal lines
        lw = s - 6; ly_start = y + 3
        for i in range(3):
            ly = ly_start + i * (s - 6) // 2
            p.drawRoundedRect(QRect(x + 3, ly, lw, 3), 1, 1)
    elif kind == "gear":
        # Circle with hole
        p.drawEllipse(QRect(x + 2, y + 2, s - 4, s - 4))
        p.setBrush(QColor(C_SIDEBAR)); p.drawEllipse(QRect(cx - 3, cy - 3, 6, 6))
        p.setBrush(QColor(color))
        # Small notches
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            nx = cx + int((r - 1) * math.cos(rad)); ny = cy + int((r - 1) * math.sin(rad))
            p.drawEllipse(QRect(nx - 2, ny - 2, 4, 4))

# ── Sidebar ──
class Sidebar(QWidget):
    page_changed = Signal(int)
    ITEMS = [("monitor", "监控"), ("chart", "统计"), ("list", "记录"), ("gear", "设置")]
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedWidth(200); self._active = 0
        self.setStyleSheet(f"background: {C_SIDEBAR};")
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Section header
        p.setPen(QColor(C_DGRAY)); p.setFont(QFont(FF, 11, QFont.DemiBold))
        p.drawText(QRect(20, 16, w - 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "导航")
        # Items
        item_h = 32; start_y = 44; padding = 8
        for i, (kind, label) in enumerate(self.ITEMS):
            iy = start_y + i * item_h
            if i == self._active:
                p.setPen(Qt.NoPen); p.setBrush(QColor(C_TER))
                p.drawRoundedRect(QRect(padding, iy, w - padding * 2, item_h - 2), 6, 6)
            icon_color = C_WHITE if i == self._active else C_MGRAY
            _paint_icon(p, kind, 16, iy + 4, 20, icon_color)
            p.setPen(QColor(C_WHITE) if i == self._active else QColor(C_LGRAY))
            p.setFont(QFont(FF, 13)); p.drawText(QRect(44, iy, w - 50, item_h - 2), Qt.AlignLeft | Qt.AlignVCenter, label)
        p.end()
    def mousePressEvent(self, e):
        item_h = 32; start_y = 44; idx = (e.position().y() - start_y) / item_h
        if 0 <= int(idx) < len(self.ITEMS):
            self._active = int(idx); self.update(); self.page_changed.emit(self._active)

# ── Card (borderless) ──
class Card(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background: {C_SEC}; border-radius: 10px; }}")
        self._l = QVBoxLayout(self); self._l.setContentsMargins(16, 12, 16, 14); self._l.setSpacing(8)
        if title:
            lbl = QLabel(title); lbl.setStyleSheet(f"color: {C_MGRAY}; font-size: 11px; font-weight: 600; border: none; font-family: {FF}; letter-spacing: 0.3px;")
            self._l.addWidget(lbl)
    def add(self, w): self._l.addWidget(w)
    def addl(self, l): self._l.addLayout(l)

# ── Performance Panel ──
class PerfRing(QWidget):
    """Circular progress ring — Apple Activity Monitor style."""
    def __init__(self, label, color=C_BLUE, parent=None):
        super().__init__(parent); self.setFixedSize(64, 64)
        self._val = 0.0; self._label = label; self._color = color
    def set_value(self, v): self._val = min(v, 100); self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); cx, cy = w // 2, h // 2 - 6; r = 22
        # Background ring
        p.setPen(QPen(QColor(C_QUA), 5, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QRect(cx - r, cy - r, r * 2, r * 2))
        # Value arc
        span = -int(self._val / 100 * 360 * 16)
        c = C_RED if self._val > 85 else (C_ORANGE if self._val > 60 else QColor(self._color))
        p.setPen(QPen(QColor(c), 5, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRect(cx - r, cy - r, r * 2, r * 2), 90 * 16, span)
        # Percentage text
        p.setPen(QColor(C_WHITE)); p.setFont(QFont(FF, 10, QFont.DemiBold))
        p.drawText(QRect(cx - r, cy - 6, r * 2, 14), Qt.AlignCenter, f"{self._val:.0f}%")
        # Label
        p.setPen(QColor(C_MGRAY)); p.setFont(QFont(FF, 9))
        p.drawText(QRect(0, h - 14, w, 12), Qt.AlignCenter, self._label)
        p.end()

class FPSChart(QWidget):
    """Mini sparkline showing FPS history — last 60 seconds."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedHeight(48); self._data = deque(maxlen=60)
    def add_value(self, v): self._data.append(v); self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if len(self._data) < 2: p.end(); return
        data = list(self._data); n = len(data); mx = max(max(data), 1) * 1.2
        # Fill area
        path = QPainterPath(); path.moveTo(0, h)
        for i, v in enumerate(data):
            x = int(i / (n - 1) * w); y = h - int(v / mx * (h - 4)) - 2
            if i == 0: path.lineTo(x, y)
            else: path.lineTo(x, y)
        path.lineTo(w, h); path.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h); grad.setColorAt(0, QColor(C_BLUE + "30")); grad.setColorAt(1, QColor(C_BLUE + "05"))
        p.setPen(Qt.NoPen); p.setBrush(QBrush(grad)); p.drawPath(path)
        # Line
        pen = QPen(QColor(C_BLUE), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        first = True
        for i, v in enumerate(data):
            x = int(i / (n - 1) * w); y = h - int(v / mx * (h - 4)) - 2
            if first: p.drawPoint(x, y); first = False
            else: p.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y
        p.end()

class MetricGauge(QWidget):
    """Compact metric with label, big value, and unit."""
    def __init__(self, label, unit="", color=C_BLUE, parent=None):
        super().__init__(parent); self._u = unit; self._c = color
        self._val = QLabel("--"); self._val.setAlignment(Qt.AlignCenter)
        self._val.setStyleSheet(f"color: {C_WHITE}; font-size: 22px; font-weight: 700; border: none; font-family: {FF};")
        self._unit = QLabel(unit); self._unit.setAlignment(Qt.AlignCenter)
        self._unit.setStyleSheet(f"color: {C_DGRAY}; font-size: 10px; border: none; font-family: {FF};")
        self._name = QLabel(label); self._name.setAlignment(Qt.AlignCenter)
        self._name.setStyleSheet(f"color: {C_MGRAY}; font-size: 10px; border: none; font-family: {FF}; letter-spacing: 0.3px;")
        l = QVBoxLayout(self); l.setContentsMargins(4, 4, 4, 4); l.setSpacing(0)
        l.addWidget(self._name); l.addWidget(self._val); l.addWidget(self._unit)
    def set_value(self, v, fmt=".0f"):
        self._val.setText(f"{v:{fmt}}"); self._val.setStyleSheet(f"color: {self._c}; font-size: 22px; font-weight: 700; border: none; font-family: {FF};")
    def set_color(self, c): self._c = c; self._val.setStyleSheet(f"color: {c}; font-size: 22px; font-weight: 700; border: none; font-family: {FF};")

class PerfPanel(QWidget):
    """Full performance monitoring panel with rings, gauges, sparkline, and bars."""
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(12)

        # ── Section: Gauges (FPS / Latency / Count) ──
        gauge_row = QHBoxLayout(); gauge_row.setSpacing(0)
        self.fps_gauge = MetricGauge("FPS", "", C_BLUE)
        self.lat_gauge = MetricGauge("延迟", "ms", C_GREEN)
        self.det_gauge = MetricGauge("检测", "", C_ORANGE)
        for g in [self.fps_gauge, self.lat_gauge, self.det_gauge]:
            gauge_row.addWidget(g)
        root.addLayout(gauge_row)

        # ── Section: FPS Sparkline ──
        spark_card = QWidget(); spark_card.setStyleSheet(f"background: {C_SEC}; border-radius: 8px;")
        spark_l = QVBoxLayout(spark_card); spark_l.setContentsMargins(12, 8, 12, 8); spark_l.setSpacing(4)
        spark_header = QHBoxLayout()
        spark_title = QLabel("FPS 实时曲线"); spark_title.setStyleSheet(f"color: {C_MGRAY}; font-size: 10px; font-weight: 500; border: none; font-family: {FF};")
        spark_header.addWidget(spark_title); spark_header.addStretch()
        self.fps_value_label = QLabel("--"); self.fps_value_label.setStyleSheet(f"color: {C_WHITE}; font-size: 11px; font-weight: 600; border: none; font-family: {FF};")
        spark_header.addWidget(self.fps_value_label); spark_l.addLayout(spark_header)
        self.fps_spark = FPSChart(); spark_l.addWidget(self.fps_spark)
        root.addWidget(spark_card)

        # ── Section: Resource Rings ──
        ring_card = QWidget(); ring_card.setStyleSheet(f"background: {C_SEC}; border-radius: 8px;")
        ring_l = QVBoxLayout(ring_card); ring_l.setContentsMargins(12, 8, 12, 12); ring_l.setSpacing(4)
        ring_title = QLabel("系统资源"); ring_title.setStyleSheet(f"color: {C_MGRAY}; font-size: 10px; font-weight: 500; border: none; font-family: {FF};")
        ring_l.addWidget(ring_title)
        ring_row = QHBoxLayout(); ring_row.setSpacing(8)
        self.cpu_ring = PerfRing("CPU", C_BLUE); self.mem_ring = PerfRing("内存", C_PURPLE)
        self.gpu_ring = PerfRing("GPU", C_GREEN); self.vram_ring = PerfRing("显存", C_ORANGE)
        for r in [self.cpu_ring, self.mem_ring, self.gpu_ring, self.vram_ring]:
            ring_row.addWidget(r)
        ring_l.addLayout(ring_row)
        root.addWidget(ring_card)

        # ── Section: GPU info (if available) ──
        self.gpu_info = QLabel("GPU: --"); self.gpu_info.setStyleSheet(f"color: {C_DGRAY}; font-size: 10px; border: none; font-family: {FF};")
        root.addWidget(self.gpu_info)
        root.addStretch()

    def update_perf(self, cpu, mem, gpu_util, gpu_mem, fps, latency, det_count):
        self.cpu_ring.set_value(cpu)
        self.mem_ring.set_value(mem)
        self.gpu_ring.set_value(gpu_util)
        self.vram_ring.set_value(gpu_mem)
        self.fps_gauge.set_value(fps, ".1f")
        self.fps_gauge.set_color(C_BLUE if fps >= 24 else (C_ORANGE if fps >= 15 else C_RED))
        self.lat_gauge.set_value(latency, ".0f")
        self.lat_gauge.set_color(C_GREEN if latency < 30 else (C_ORANGE if latency < 80 else C_RED))
        self.det_gauge.set_value(det_count)
        self.fps_spark.add_value(fps)
        self.fps_value_label.setText(f"{fps:.1f}")
        if gpu_util > 0 or gpu_mem > 0:
            self.gpu_info.setText(f"GPU 利用率 {gpu_util:.0f}%  |  显存占用 {gpu_mem:.1f}%")

# ── GPU monitor helper ──
def _get_gpu_stats():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes): name = name.decode()
        pynvml.nvmlShutdown()
        return name, u.gpu, (m.used / m.total) * 100
    except Exception:
        return None, 0.0, 0.0

# ── Video placeholder ──
class VideoPlaceholder(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(640, 480)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#000000"))
        cx, cy = w // 2, h // 2 - 16
        p.setPen(QPen(QColor(C_QUA), 1.5)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRect(cx - 40, cy - 40, 80, 80))
        p.setPen(QPen(QColor(C_QUA), 1)); p.drawLine(cx - 52, cy, cx + 52, cy); p.drawLine(cx, cy - 52, cx, cy + 52)
        p.setPen(QColor(C_MGRAY)); p.setFont(QFont(FF, 13)); p.drawText(QRect(0, cy + 52, w, 20), Qt.AlignCenter, "等待视频信号")
        p.setFont(QFont(FF, 11)); p.setPen(QColor(C_DGRAY)); p.drawText(QRect(0, cy + 76, w, 18), Qt.AlignCenter, "拖入视频文件 或 选择摄像头")
        p.end()

# ── Chart widgets ──
class BarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(200); self._data = []; self._labels = []
    def set_data(self, data, labels=None): self._data = data[:12]; self._labels = labels or [str(i) for i in range(len(data))]; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        if not self._data or all(v == 0 for v in self._data):
            p.setPen(QColor(C_MGRAY)); p.setFont(QFont(FF, 13)); p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "等待检测数据"); p.end(); return
        ml, mr, mt, mb = 44, 16, 24, 32; cw, ch = w - ml - mr, h - mt - mb
        n = len(self._data); bw = max(int(cw / n * 0.5), 4); gap = cw / n; mx = max(self._data) * 1.15
        g = QPen(QColor(C_DIV), 1, Qt.DashLine); p.setPen(g)
        for i in range(5): y = int(mt + ch * i / 4); p.drawLine(ml, y, w - mr, y); p.setPen(QColor(C_DGRAY)); p.setFont(QFont(FF, 9)); p.drawText(QRect(0, y - 7, ml - 6, 14), Qt.AlignRight | Qt.AlignVCenter, f"{mx*(1-i/4):.0f}"); p.setPen(g)
        grad = QLinearGradient(0, mt, 0, mt + ch); grad.setColorAt(0, QColor(C_BLUE)); grad.setColorAt(1, QColor(C_BLUE + "50"))
        for i, val in enumerate(self._data):
            bx = int(ml + gap * i + (gap - bw) / 2); bh = int(val / mx * ch) if mx > 0 else 0
            p.setPen(Qt.NoPen); p.setBrush(QBrush(grad)); p.drawRoundedRect(QRect(bx, mt + ch - bh, bw, bh), 3, 3)
        p.setPen(QColor(C_DGRAY)); p.setFont(QFont(FF, 9))
        for i, lbl in enumerate(self._labels[:n]): bx = int(ml + gap * i + gap / 2); p.drawText(QRect(bx - 20, h - mb + 4, 40, 16), Qt.AlignCenter, lbl)
        p.end()

class HBarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(180); self._labels = []; self._values = []
        self._colors = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE, C_DGRAY]
    def set_data(self, labels, values): self._labels = labels; self._values = values; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height(); n = len(self._labels)
        if n == 0: p.setPen(QColor(C_MGRAY)); p.setFont(QFont(FF, 13)); p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "等待检测数据"); p.end(); return
        ml, mr, mt, mb = 56, 40, 16, 16; ch = h - mt - mb; bh = max(int(ch / n * 0.5), 8); gap = ch / n; mx = max(max(self._values), 1) * 1.15
        for i in range(n):
            by = int(mt + gap * i + (gap - bh) / 2); bw = int(self._values[i] / mx * (w - ml - mr))
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(self._colors[i % len(self._colors)]))); p.drawRoundedRect(QRect(ml, by, bw, bh), 3, 3)
            p.setPen(QColor(C_LGRAY)); p.setFont(QFont(FF, 11)); p.drawText(QRect(0, by, ml - 8, bh), Qt.AlignRight | Qt.AlignVCenter, self._labels[i])
            p.setPen(QColor(C_WHITE)); p.setFont(QFont(FF, 10, QFont.DemiBold)); p.drawText(QRect(ml + bw + 6, by, 36, bh), Qt.AlignLeft | Qt.AlignVCenter, f"{self._values[i]:.0f}")
        p.end()

class PairedChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(180); self._t = []; self._y = []; self._l = []
    def set_data(self, t, y, labels=None): self._t = t[:12]; self._y = y[:12]; self._l = labels or [str(i) for i in range(len(t))]; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        if not self._t or not self._y: p.setPen(QColor(C_MGRAY)); p.setFont(QFont(FF, 13)); p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "等待检测数据"); p.end(); return
        ml, mr, mt, mb = 40, 12, 28, 32; cw = w - ml - mr; ch = h - mt - mb
        n = min(len(self._t), len(self._y)); gap = cw / n; sw = max(int(gap * 0.25), 3); tbw = sw * 2 + 2
        mx = max(max(self._t + self._y), 1) * 1.15
        g = QPen(QColor(C_DIV), 1, Qt.DashLine)
        for i in range(5): y = int(mt + ch * i / 4); p.setPen(g); p.drawLine(ml, y, w - mr, y); p.setPen(QColor(C_DGRAY)); p.setFont(QFont(FF, 9)); p.drawText(QRect(0, y - 7, ml - 6, 14), Qt.AlignRight | Qt.AlignVCenter, f"{mx*(1-i/4):.0f}")
        for i in range(n):
            bx = int(ml + gap * i + (gap - tbw) / 2)
            h1 = int(self._t[i] / mx * ch) if mx > 0 else 0; p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(C_BLUE))); p.drawRoundedRect(QRect(bx, mt + ch - h1, sw, h1), 2, 2)
            h2 = int(self._y[i] / mx * ch) if mx > 0 else 0; p.setBrush(QBrush(QColor(C_PURPLE))); p.drawRoundedRect(QRect(bx + sw + 2, mt + ch - h2, sw, h2), 2, 2)
        lx = w - mr - 120
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(C_BLUE))); p.drawRoundedRect(QRect(lx, 8, 10, 10), 2, 2); p.setPen(QColor(C_LGRAY)); p.setFont(QFont(FF, 10)); p.drawText(lx + 14, 16, "今天")
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(C_PURPLE))); p.drawRoundedRect(QRect(lx + 52, 8, 10, 10), 2, 2); p.setPen(QColor(C_LGRAY)); p.drawText(lx + 66, 16, "昨天")
        p.setPen(QColor(C_DGRAY)); p.setFont(QFont(FF, 9))
        for i in range(n): bx = int(ml + gap * i + gap / 2); p.drawText(QRect(bx - 16, h - mb + 4, 32, 16), Qt.AlignCenter, self._l[i] if i < len(self._l) else "")
        p.end()

# ══════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════

class MonitorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); root = QHBoxLayout(self); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(16)
        self._video_path = ''  # 导入文件路径
        # Video
        self.vid_stack = QStackedWidget(); self.vid_stack.setMinimumSize(640, 480)
        self.vid_stack.setStyleSheet("QStackedWidget { background: #000; border-radius: 10px; }")
        self.vid_ph = VideoPlaceholder(); self.vid_stack.addWidget(self.vid_ph)
        self.vid_lbl = QLabel(); self.vid_lbl.setAlignment(Qt.AlignCenter); self.vid_lbl.setStyleSheet("background: #000; border-radius: 10px;")
        self.vid_stack.addWidget(self.vid_lbl); self.vid_stack.setCurrentIndex(0)
        root.addWidget(self.vid_stack, stretch=1)
        # Right panel
        panel = QWidget(); panel.setFixedWidth(300); pl = QVBoxLayout(panel); pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(12)
        # Title
        tc = Card(); tl = QHBoxLayout(); tl.setSpacing(10)
        col = QVBoxLayout(); col.setSpacing(1)
        t1 = QLabel("车辆辅助驾驶系统"); t1.setStyleSheet(f"color: {C_WHITE}; font-size: 17px; font-weight: 600; border: none; font-family: {FF};")
        t2 = QLabel("v6.0 · 智能监控平台"); t2.setStyleSheet(f"color: {C_MGRAY}; font-size: 11px; border: none; font-family: {FF};")
        col.addWidget(t1); col.addWidget(t2); tl.addLayout(col); tl.addStretch()
        self.dot = QLabel(); self.dot.setFixedSize(8, 8); self.dot.setStyleSheet(f"background: {C_DGRAY}; border-radius: 4px;"); tl.addWidget(self.dot)
        tc.addl(tl); pl.addWidget(tc)
        # Source — 两行独立选择，带图标
        sc = Card("视频来源")
        self._src_is_file = False
        row_style = "QWidget { background: transparent; }"
        # 行 1: 摄像头
        self.src_cam_row = QWidget(); self.src_cam_row.setCursor(Qt.PointingHandCursor)
        self.src_cam_row.installEventFilter(self)
        scr = QHBoxLayout(self.src_cam_row); scr.setContentsMargins(6, 4, 6, 4); scr.setSpacing(8)
        self.src_cam_icon = QLabel("📷"); self.src_cam_icon.setStyleSheet("font-size: 16px; border: none;")
        scr.addWidget(self.src_cam_icon)
        lbl_cam = QLabel("摄像头"); lbl_cam.setStyleSheet(f"color: {C_LGRAY}; font-size: 13px; border: none; font-family: {FF};")
        scr.addWidget(lbl_cam)
        scr.addStretch()
        self.cam_cb = QComboBox(); self.cam_cb.setFixedHeight(28)
        for i in range(4): self.cam_cb.addItem(f"摄像头 {i}", i)
        scr.addWidget(self.cam_cb)
        self.btn_clear_cam = QPushButton("清除"); self.btn_clear_cam.setFixedHeight(26); self.btn_clear_cam.setCursor(Qt.PointingHandCursor)
        self.btn_clear_cam.setStyleSheet(f"QPushButton {{ background: {C_TER}; color: {C_MGRAY}; border-radius: 5px; padding: 0 8px; font-size: 11px; }} QPushButton:hover {{ color: {C_RED}; }}")
        self.btn_clear_cam.clicked.connect(self._on_clear_cam)
        scr.addWidget(self.btn_clear_cam)
        sc.add(self.src_cam_row)
        # 分隔
        div = QFrame(); div.setFixedHeight(1); div.setStyleSheet(f"background: {C_DIV}; border: none;"); sc.add(div)
        # 行 2: 导入文件
        self.src_file_row = QWidget(); self.src_file_row.setCursor(Qt.PointingHandCursor)
        self.src_file_row.installEventFilter(self)
        sfr = QHBoxLayout(self.src_file_row); sfr.setContentsMargins(6, 4, 6, 4); sfr.setSpacing(8)
        self.src_file_icon = QLabel("📁"); self.src_file_icon.setStyleSheet("font-size: 16px; border: none;")
        sfr.addWidget(self.src_file_icon)
        lbl_file = QLabel("导入文件"); lbl_file.setStyleSheet(f"color: {C_LGRAY}; font-size: 13px; border: none; font-family: {FF};")
        sfr.addWidget(lbl_file)
        sfr.addStretch()
        self.lbl_filename = QLabel("未选择"); self.lbl_filename.setStyleSheet(f"color: {C_DGRAY}; font-size: 11px; border: none;")
        sfr.addWidget(self.lbl_filename)
        self.btn_file = QPushButton("选择"); self.btn_file.setFixedHeight(26); self.btn_file.setCursor(Qt.PointingHandCursor)
        self.btn_file.setStyleSheet(f"QPushButton {{ background: {C_TER}; color: {C_LGRAY}; border-radius: 5px; padding: 0 10px; font-size: 11px; }} QPushButton:hover {{ background: {C_QUA}; }}")
        self.btn_file.clicked.connect(self._on_select_file)
        sfr.addWidget(self.btn_file)
        self.btn_clear_file = QPushButton("清除"); self.btn_clear_file.setFixedHeight(26); self.btn_clear_file.setCursor(Qt.PointingHandCursor)
        self.btn_clear_file.setStyleSheet(f"QPushButton {{ background: {C_TER}; color: {C_MGRAY}; border-radius: 5px; padding: 0 8px; font-size: 11px; }} QPushButton:hover {{ color: {C_RED}; }}")
        self.btn_clear_file.clicked.connect(self._on_clear_file)
        sfr.addWidget(self.btn_clear_file)
        sc.add(self.src_file_row)
        pl.addWidget(sc)
        # Quad mode
        qc = Card("四路模式"); qr = QHBoxLayout(); qr.setSpacing(8)
        self.lbl_quad = QLabel("4路分割识别"); self.lbl_quad.setStyleSheet(f"color: {C_LGRAY}; font-size: 12px; border: none; font-family: {FF};"); qr.addWidget(self.lbl_quad); qr.addStretch()
        self.sw_quad = MacToggle(); qr.addWidget(self.sw_quad); qc.addl(qr); pl.addWidget(qc)
        # Toggle
        self.btn_toggle = QPushButton("开始检测"); self.btn_toggle.setFixedHeight(40); self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet(f"QPushButton {{ background: {C_BLUE}; color: #fff; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; font-family: {FF}; }} QPushButton:hover {{ background: #0975e0; }}")
        pl.addWidget(self.btn_toggle)
        # Performance
        pc = Card("性能监控"); self.perf = PerfPanel(); pc.add(self.perf); pl.addWidget(pc)
        # Log
        lc = Card("运行日志"); self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(80)
        self.log.setStyleSheet(f"background: {C_SEC}; color: {C_MGRAY}; border: none; border-radius: 8px; font-family: {FM}; font-size: 12px; padding: 8px;")
        lc.add(self.log); pl.addWidget(lc, stretch=1); root.addWidget(panel)

    def eventFilter(self, obj, event):
        """处理来源行点击"""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.MouseButtonPress:
            if obj is self.src_cam_row:
                self._set_src_mode(False)
            elif obj is self.src_file_row:
                self._set_src_mode(True)
        return super().eventFilter(obj, event)

    def _set_src_mode(self, is_file: bool):
        """切换摄像头 / 文件模式 — 高亮当前行"""
        self._src_is_file = is_file
        active = f"background: {C_TER}; border-radius: 6px;"
        inactive = "background: transparent;"
        if is_file:
            self.src_cam_row.setStyleSheet(inactive)
            self.src_file_row.setStyleSheet(active)
        else:
            self.src_cam_row.setStyleSheet(active)
            self.src_file_row.setStyleSheet(inactive)

    def _on_clear_cam(self):
        """清除摄像头选择，回到默认"""
        self._set_src_mode(False)
        self._video_path = ''
        self.lbl_filename.setText("未选择")

    def _on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;所有 (*)")
        if path:
            self._video_path = path
            name = os.path.basename(path)
            self.lbl_filename.setText(name if len(name) <= 18 else name[:16] + "…")
            self._set_src_mode(True)  # 选了文件自动切到文件模式

    def _on_clear_file(self):
        self._video_path = ''
        self.lbl_filename.setText("未选择")

    def _apply_running(self):
        self.btn_toggle.setText("停止检测")
        self.btn_toggle.setStyleSheet(f"QPushButton {{ background: {C_RED}; color: #fff; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; font-family: {FF}; }} QPushButton:hover {{ background: #e0342d; }}")
        self.dot.setStyleSheet(f"background: {C_GREEN}; border-radius: 4px;"); self.vid_stack.setCurrentIndex(1)
    def _apply_stopped(self):
        self.btn_toggle.setText("开始检测")
        self.btn_toggle.setStyleSheet(f"QPushButton {{ background: {C_BLUE}; color: #fff; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; font-family: {FF}; }} QPushButton:hover {{ background: #0975e0; }}")
        self.dot.setStyleSheet(f"background: {C_DGRAY}; border-radius: 4px;"); self.vid_stack.setCurrentIndex(0)


class StatsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); root = QVBoxLayout(self); root.setContentsMargins(20, 20, 20, 20); root.setSpacing(16)
        # Stat cards
        top = QHBoxLayout(); top.setSpacing(12); self.sc = []; self.sc_vals = []
        for title, val, color in [("今日检测数", "0", C_BLUE), ("识别成功率", "--", C_PURPLE), ("平均延迟", "--", C_GREEN), ("累计检测", "0", C_ORANGE)]:
            c = Card(); c.setFixedHeight(90); cl = QHBoxLayout(c); cl.setContentsMargins(4, 12, 16, 12); cl.setSpacing(10)
            stripe = QWidget(); stripe.setFixedWidth(3); stripe.setStyleSheet(f"background: {color}; border-radius: 1px;"); cl.addWidget(stripe)
            vl = QVBoxLayout(); vl.setSpacing(4)
            t = QLabel(title); t.setStyleSheet(f"color: {C_MGRAY}; font-size: 11px; font-weight: 500; border: none; font-family: {FF};"); vl.addWidget(t)
            v = QLabel(val); v.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: 700; border: none; font-family: {FF};"); vl.addWidget(v); vl.addStretch(); cl.addLayout(vl, stretch=1); top.addWidget(c); self.sc.append(c); self.sc_vals.append(v)
        root.addLayout(top)
        self._last_vals = {"records": 0, "fps": 0, "lat": 0, "det": 0}
        # Charts
        tc = Card("检测趋势"); self.bc = BarChart(); tc.add(self.bc); root.addWidget(tc)
        br = QHBoxLayout(); br.setSpacing(12)
        vc = Card("车辆类型分布"); self.hb = HBarChart(); vc.add(self.hb); br.addWidget(vc, stretch=1)
        hc = Card("时段分布对比"); self.pc = PairedChart(); hc.add(self.pc); br.addWidget(hc, stretch=1)
        root.addLayout(br); root.addStretch()

    def update_stats(self, engine_stats, records_count=0):
        if not engine_stats: return
        fps_hist = list(engine_stats.get('fps_history', []))[-30:]
        fps = sum(fps_hist) / max(len(fps_hist), 1)
        lat = engine_stats.get('avg_latency', 0)
        det = engine_stats.get('detection_count', 0)
        vals = self.sc_vals
        if len(vals) >= 4:
            vals[0].setText(str(records_count))
            vals[1].setText(f"{fps:.1f}" if fps > 0 else "--")
            vals[2].setText(f"{lat:.0f}" if lat > 0 else "--")
            vals[3].setText(str(det))
        self.bc.set_data(list(engine_stats.get('fps_history', []))[-12:])
        plates = engine_stats.get('plate_colors', {})
        if plates: self.hb.set_data(list(plates.keys()), list(plates.values()))
        vh = engine_stats.get('hourly', {}); yh = engine_stats.get('hourly_yesterday', {})
        if vh or yh: self.pc.set_data(list(vh.values()), list(yh.values()), list(vh.keys()))


class RecordsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); root = QVBoxLayout(self); root.setContentsMargins(20, 20, 20, 20); root.setSpacing(12)
        tb = QHBoxLayout(); tb.setSpacing(8)
        self.search = QLineEdit(); self.search.setPlaceholderText("搜索记录..."); self.search.setFixedHeight(32); tb.addWidget(self.search, stretch=1)
        self.btn_exp = QPushButton("导出CSV"); self.btn_exp.setFixedHeight(32)
        self.btn_exp.setStyleSheet(f"QPushButton {{ background: {C_BLUE}; color: #fff; border: none; border-radius: 6px; padding: 0 14px; }} QPushButton:hover {{ background: #0975e0; }}")
        tb.addWidget(self.btn_exp)
        self.btn_clr = QPushButton("清空"); self.btn_clr.setFixedHeight(32)
        self.btn_clr.setStyleSheet(f"QPushButton {{ background: {C_TER}; color: {C_LGRAY}; border: none; border-radius: 6px; padding: 0 14px; }} QPushButton:hover {{ background: {C_QUA}; }}")
        tb.addWidget(self.btn_clr); root.addLayout(tb)
        self.table = QTableWidget(); self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["碰撞提醒", "置信度", "时间"])
        self.table.horizontalHeader().setStretchLastSection(False)
        for c in range(3): self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False); self.table.setShowGrid(False)
        root.addWidget(self.table, stretch=1)
        self._data = []

    def add_collision_record(self, distance, confidence, ts):
        self._data.insert(0, (ts, distance, confidence))
        self.table.insertRow(0)
        warn = QTableWidgetItem(f"碰撞预警 {distance:.1f}m")
        warn.setForeground(QColor(C_RED))
        warn.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, 0, warn)
        conf_item = QTableWidgetItem(f"{confidence:.2f}")
        conf_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, 1, conf_item)
        time_item = QTableWidgetItem(ts)
        time_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, 2, time_item)
        if self.table.rowCount() > 50: self.table.removeRow(self.table.rowCount() - 1)

    def filter_records(self, text):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            self.table.setRowHidden(i, text.lower() not in item.text().lower() if item else True)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出记录", f"records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV (*.csv)")
        if not path: return
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write("碰撞提醒,置信度,时间\n")
            for i in range(self.table.rowCount()):
                row = [self.table.item(i, j).text() if self.table.item(i, j) else "" for j in range(3)]
                f.write(",".join(row) + "\n")

    def clear_records(self):
        self.table.setRowCount(0); self._data.clear()


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); root = QVBoxLayout(self); root.setContentsMargins(20, 20, 20, 20); root.setSpacing(12)
        self.tabs = QTabWidget()
        self._model = {}; self._image = {}; self._sys = {}; self._eng = {}
        mt = QWidget(); ml = QVBoxLayout(mt); ml.setContentsMargins(16, 16, 16, 16); ml.setSpacing(16)
        ml.addWidget(self._sl_row("置信度阈值", 5, 95, 40, self._model, "conf")); ml.addWidget(self._sl_row("IoU阈值", 10, 95, 45, self._model, "iou"))
        sz = QHBoxLayout(); sz.setSpacing(12); sl = QLabel("输入尺寸"); sl.setStyleSheet(f"color: {C_LGRAY}; border: none; font-family: {FF};"); sz.addWidget(sl); sz.addStretch()
        cb = QComboBox(); cb.setFixedWidth(100); [cb.addItem(str(s), s) for s in [320, 416, 480, 640, 800]]; cb.setCurrentText("640"); self._model["img_size"] = cb; sz.addWidget(cb); ml.addLayout(sz)
        ml.addWidget(self._sw_row("FP16加速", True, self._eng, "fp16")); ml.addWidget(self._sw_row("倾斜校正", False, self._eng, "tilt"))
        ml.addWidget(self._sw_row("车牌检测", True, self._eng, "plate")); ml.addWidget(self._sw_row("车辆检测", True, self._eng, "vehicle"))
        ml.addWidget(self._sw_row("红绿灯检测", True, self._eng, "traffic_light")); ml.addWidget(self._sw_row("行人检测", True, self._eng, "pedestrian"))
        ml.addWidget(self._sw_row("违规检测", True, self._eng, "violation")); ml.addWidget(self._sw_row("多尺度检测", True, self._eng, "multiscale"))
        ml.addWidget(self._sw_row("车流量统计", True, self._eng, "flow_count")); ml.addWidget(self._sw_row("斑马线检测", True, self._eng, "crosswalk"))
        ml.addStretch(); self.tabs.addTab(mt, "模型参数")
        it = QWidget(); il = QVBoxLayout(it); il.setContentsMargins(16, 16, 16, 16); il.setSpacing(16)
        il.addWidget(self._sl_row("亮度", -100, 100, 0, self._image, "brightness")); il.addWidget(self._sl_row("对比度", -100, 100, 0, self._image, "contrast"))
        il.addWidget(self._sl_row("饱和度", -100, 100, 0, self._image, "saturation"))
        il.addWidget(self._sl_row("红通道", -50, 50, 0, self._image, "wb_r")); il.addWidget(self._sl_row("绿通道", -50, 50, 0, self._image, "wb_g"))
        il.addWidget(self._sl_row("蓝通道", -50, 50, 0, self._image, "wb_b"))
        il.addStretch(); self.tabs.addTab(it, "图像校正")
        st = QWidget(); sl2 = QVBoxLayout(st); sl2.setContentsMargins(16, 16, 16, 16); sl2.setSpacing(16)
        sr = QHBoxLayout(); sr.setSpacing(12); srl = QLabel("默认来源"); srl.setStyleSheet(f"color: {C_LGRAY}; border: none; font-family: {FF};"); sr.addWidget(srl); sr.addStretch()
        sc = QComboBox(); sc.setFixedWidth(140); [sc.addItem(f"摄像头 {i}", i) for i in range(4)]; sc.addItem("文件", -1); self._sys["source"] = sc; sr.addWidget(sc); sl2.addLayout(sr)
        sl2.addWidget(self._sw_row("启动时自动检测", False, self._sys, "auto_detect")); sl2.addWidget(self._sw_row("保存检测截图", True, self._sys, "save_screenshot"))
        lr = QHBoxLayout(); lr.setSpacing(12); ll = QLabel("日志级别"); ll.setStyleSheet(f"color: {C_LGRAY}; border: none; font-family: {FF};"); lr.addWidget(ll); lr.addStretch()
        lc = QComboBox(); lc.setFixedWidth(140); [lc.addItem(l, l) for l in ["DEBUG", "INFO", "WARNING", "ERROR"]]; lc.setCurrentText("INFO"); self._sys["log_level"] = lc; lr.addWidget(lc); sl2.addLayout(lr)
        sl2.addStretch(); self.tabs.addTab(st, "系统设置")
        root.addWidget(self.tabs, stretch=1)
        br = QHBoxLayout(); br.setSpacing(10); br.addStretch()
        self.btn_reset = QPushButton("重置默认"); self.btn_reset.setFixedSize(100, 32)
        self.btn_reset.setStyleSheet(f"QPushButton {{ background: {C_TER}; color: {C_LGRAY}; border: none; border-radius: 6px; }} QPushButton:hover {{ background: {C_QUA}; }}")
        br.addWidget(self.btn_reset)
        self.btn_save = QPushButton("保存设置"); self.btn_save.setFixedSize(100, 32)
        self.btn_save.setStyleSheet(f"QPushButton {{ background: {C_BLUE}; color: #fff; border: none; border-radius: 6px; font-weight: 600; }} QPushButton:hover {{ background: #0975e0; }}")
        br.addWidget(self.btn_save); root.addLayout(br)

    def _sl_row(self, label, mn, mx, default, store=None, key=None):
        w = QWidget(); wl = QHBoxLayout(w); wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(12)
        lbl = QLabel(label); lbl.setStyleSheet(f"color: {C_LGRAY}; font-size: 13px; border: none; font-family: {FF};"); wl.addWidget(lbl)
        sl = QSlider(Qt.Horizontal); sl.setRange(int(mn), int(mx)); sl.setValue(int(default)); sl.setFixedWidth(200); wl.addWidget(sl, stretch=1)
        vl = QLabel(str(default)); vl.setFixedWidth(36); vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vl.setStyleSheet(f"color: {C_WHITE}; font-size: 12px; font-weight: 500; border: none; font-family: {FF};"); wl.addWidget(vl)
        sl.valueChanged.connect(lambda v, l=vl: l.setText(str(v)))
        if store is not None and key is not None: store[key] = sl
        return w

    def _sw_row(self, label, default=False, store=None, key=None):
        w = QWidget(); wl = QHBoxLayout(w); wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(12)
        lbl = QLabel(label); lbl.setStyleSheet(f"color: {C_LGRAY}; font-size: 13px; border: none; font-family: {FF};"); wl.addWidget(lbl); wl.addStretch()
        sw = MacToggle(); sw.setChecked(default); wl.addWidget(sw)
        if store is not None and key is not None: store[key] = sw
        return w

    def load_from_config(self, config):
        m = config.model; c = config.color
        if "conf" in self._model: self._model["conf"].setValue(int(m.conf_threshold * 100))
        if "iou" in self._model: self._model["iou"].setValue(int(m.iou_threshold * 100))
        if "img_size" in self._model:
            idx = self._model["img_size"].findData(m.img_size)
            if idx >= 0: self._model["img_size"].setCurrentIndex(idx)
        if "fp16" in self._eng: self._eng["fp16"].setChecked(m.fp16)
        if "brightness" in self._image: self._image["brightness"].setValue(c.brightness)
        if "contrast" in self._image: self._image["contrast"].setValue(c.contrast)
        if "saturation" in self._image: self._image["saturation"].setValue(c.saturation)
        if "wb_r" in self._image: self._image["wb_r"].setValue(c.wb_r)
        if "wb_g" in self._image: self._image["wb_g"].setValue(c.wb_g)
        if "wb_b" in self._image: self._image["wb_b"].setValue(c.wb_b)

    def save_to_config(self, config):
        config.model.conf_threshold = self._model["conf"].value() / 100.0 if "conf" in self._model else config.model.conf_threshold
        config.model.iou_threshold = self._model["iou"].value() / 100.0 if "iou" in self._model else config.model.iou_threshold
        config.model.img_size = self._model["img_size"].currentData() if "img_size" in self._model else config.model.img_size
        config.model.fp16 = self._eng["fp16"].isChecked() if "fp16" in self._eng else config.model.fp16
        config.color.brightness = self._image["brightness"].value() if "brightness" in self._image else 0
        config.color.contrast = self._image["contrast"].value() if "contrast" in self._image else 0
        config.color.saturation = self._image["saturation"].value() if "saturation" in self._image else 0
        config.color.wb_r = self._image["wb_r"].value() if "wb_r" in self._image else 0
        config.color.wb_g = self._image["wb_g"].value() if "wb_g" in self._image else 0
        config.color.wb_b = self._image["wb_b"].value() if "wb_b" in self._image else 0

    def apply_engine_toggles(self, engine):
        if not engine: return
        engine.enable_plate = self._eng["plate"].isChecked() if "plate" in self._eng else True
        engine.enable_vehicle = self._eng["vehicle"].isChecked() if "vehicle" in self._eng else True
        engine.enable_traffic_light = self._eng["traffic_light"].isChecked() if "traffic_light" in self._eng else True
        engine.enable_pedestrian = self._eng["pedestrian"].isChecked() if "pedestrian" in self._eng else True
        engine.enable_violation = self._eng["violation"].isChecked() if "violation" in self._eng else True
        engine.enable_multiscale = self._eng["multiscale"].isChecked() if "multiscale" in self._eng else True
        engine.enable_flow_count = self._eng["flow_count"].isChecked() if "flow_count" in self._eng else True
        engine.enable_crosswalk = self._eng["crosswalk"].isChecked() if "crosswalk" in self._eng else True
    def reset_defaults(self):
        for k, w in self._model.items():
            if k == "conf": w.setValue(40)
            elif k == "iou": w.setValue(45)
            elif k == "img_size": idx = w.findData(640); w.setCurrentIndex(idx if idx >= 0 else 0)
        for k, w in self._image.items(): w.setValue(0)
        for k, w in self._eng.items(): w.setChecked(k != "tilt")
        for k, w in self._sys.items():
            if isinstance(w, MacToggle): w.setChecked(k == "save_screenshot")
            elif isinstance(w, QComboBox): w.setCurrentIndex(0)

# ══════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, engine=None, pipeline=None, config_manager=None, source=0):
        super().__init__()
        self.engine = engine; self.pipeline = pipeline; self.cfg = config_manager; self.source = source
        self.cam_thread = None; self.file_thread = None; self.is_running = False
        self.frame_count = 0; self.fps_ts = deque(maxlen=60); self.log_lines = []; self._log_idx = 0
        self.src_type = 'none'; self.src_path = ''; self._result_frame = None; self._detections = []
        self._last_collision_seq = 0  # 碰撞记录已同步到的序号
        self.quad_splitter = QuadSplitter(); self.quad_mode = False
        self._calib_mode = False  # 深度标定模式开关
        self.setWindowTitle("LPR System v6.0"); self.setGeometry(80, 80, 1400, 860); self.setMinimumSize(1100, 650)
        self.setStyleSheet(STYLE); self.setAcceptDrops(True)
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self.sidebar = Sidebar(); self.sidebar.page_changed.connect(lambda i: self.pages.setCurrentIndex(i)); root.addWidget(self.sidebar)
        self.pages = QStackedWidget()
        self.pg_mon = MonitorPage(); self.pg_sta = StatsPage(); self.pg_rec = RecordsPage(); self.pg_set = SettingsPage()
        for pg in [self.pg_mon, self.pg_sta, self.pg_rec, self.pg_set]: self.pages.addWidget(pg)
        root.addWidget(self.pages, stretch=1)
        self.pg_mon.vid_lbl.installEventFilter(self)  # 深度标定点击
        self.pg_mon.btn_toggle.clicked.connect(self.toggle)
        self.pg_mon.sw_quad.clicked.connect(self._toggle_quad)
        self.disp_timer = QTimer(); self.disp_timer.timeout.connect(self.on_frame); self.disp_timer.setInterval(33)
        self.log_timer = QTimer(); self.log_timer.timeout.connect(self._flush_log); self.log_timer.start(500)
        self.perf_timer = QTimer(); self.perf_timer.timeout.connect(self._on_perf_tick); self.perf_timer.setInterval(1000)
        self._gpu_name = None; self._cpu_val = 0; self._mem_val = 0; self._gpu_util = 0; self._gpu_mem_val = 0
        self._load_settings()
        self.pg_set.btn_save.clicked.connect(self._save_settings)
        self.pg_set.btn_reset.clicked.connect(self._reset_settings)
        self.pg_mon.cam_cb.setCurrentIndex(self.source)  # 同步下拉框到 CLI --source 参数
        self.pg_mon._set_src_mode(False)  # 默认摄像头模式
        self.pg_rec.btn_exp.clicked.connect(self.pg_rec.export_csv)
        self.pg_rec.btn_clr.clicked.connect(self.pg_rec.clear_records)
        self.pg_rec.search.textChanged.connect(self.pg_rec.filter_records)
        self.perf_timer.timeout.connect(lambda: self.pg_sta.update_stats(
            self.engine.stats if self.engine else {}, self.pg_rec.table.rowCount()))

    def _load_settings(self):
        if self.cfg: self.pg_set.load_from_config(self.cfg.config)

    def _save_settings(self):
        if self.cfg:
            self.pg_set.save_to_config(self.cfg.config)
            self.cfg.save()
        self._apply_engine_toggles()
        if self.engine:
            self.engine._sync_color_config()

    def _reset_settings(self):
        self.pg_set.reset_defaults()
        self._apply_engine_toggles()

    def _apply_engine_toggles(self):
        if self.engine: self.pg_set.apply_engine_toggles(self.engine)

    def _toggle_quad(self):
        self.quad_mode = self.pg_mon.sw_quad.isChecked()

    def _detect_quad(self, raw):
        """四路分割检测: 切割→分别检测→绘制→合并
        Q1/Q2 完整检测; Q3/Q4 只识别车辆 + 辅助线(车道线), 不识别车牌等"""
        quads = self.quad_splitter.split(raw)
        all_dets = []
        result_quads = []
        eng = self.engine
        # 保存检测开关, 供 Q3/Q4 临时切换后恢复
        saved = {k: getattr(eng, k) for k in (
            'enable_vehicle', 'enable_plate', 'enable_traffic_light',
            'enable_pedestrian', 'enable_crosswalk', 'enable_lane',
            'enable_flow_count', 'enable_violation', 'enable_depth')}
        for i, q in enumerate(quads):
            if q.size == 0:
                result_quads.append(q); continue
            if i >= 2:
                # Q3/Q4 (下两路): 只识别车辆, 关掉车道线/车牌/红绿灯/行人/斑马线/车流/违规
                eng.enable_vehicle = True
                eng.enable_lane = False
                eng.enable_depth = True
                eng.enable_plate = False
                eng.enable_traffic_light = False
                eng.enable_pedestrian = False
                eng.enable_crosswalk = False
                eng.enable_flow_count = False
                eng.enable_violation = False
            eng._lane_slot = i   # 车道检测按象限隔离平滑状态, 避免闪烁
            if eng.vehicle_tracker:
                eng.vehicle_tracker.reset()
            if eng.ped_tracker:
                eng.ped_tracker.reset()
            if eng.traffic_tracker:
                eng.traffic_tracker.reset()
            rf, det, _ = eng.detect(q)
            if i >= 2:
                for k, v in saved.items():
                    setattr(eng, k, v)
            for d in det:
                d['quad'] = i
                d['quad_label'] = self.quad_splitter.quad_labels[i]
            all_dets.extend(det)
            result_quads.append(rf)
        eng._lane_slot = 0   # 恢复单路槽位
        combined = self.quad_splitter.combine(result_quads, raw.shape)
        self.quad_splitter.draw_grid(combined)
        return combined, all_dets

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')): e.acceptProposedAction(); return
    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
                # 切换到文件模式并加载
                self.pg_mon._set_src_mode(True)
                self.pg_mon._video_path = p
                self.pg_mon.lbl_src.setText(os.path.basename(p))
                self.toggle()
                break
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space: self.toggle()
        elif e.key() == Qt.Key_S: self.screenshot()
        elif e.key() == Qt.Key_C: self._toggle_calib()
        elif e.key() == Qt.Key_Escape: self.close()
        else: super().keyPressEvent(e)

    def import_file(self): pass  # 已被 MonitorPage._on_select_file 替代
    def _load_file(self, path): self.src_path = path; self.src_type = 'video_file'; self.pg_mon.lbl_src.setText(os.path.basename(path))
    def toggle(self): (self.stop if self.is_running else self.start)()

    def start(self):
        # 先清理任何残留的旧线程和画面
        if self.cam_thread: self.cam_thread.stop(); self.cam_thread = None
        if self.file_thread: self.file_thread.stop(); self.file_thread = None
        self._result_frame = None; self._detections = []
        # 根据来源模式决定 src_type
        is_file = getattr(self.pg_mon, '_src_is_file', False)
        if is_file:
            # 视频文件模式自动关闭四路分屏
            self.quad_mode = False
            self.pg_mon.sw_quad.setChecked(False)
            self.src_path = getattr(self.pg_mon, '_video_path', '')
            if not self.src_path: return
            self.src_type = 'video_file'
            self.file_thread = VideoFileThread(self.src_path); self.file_thread.start()
        else:
            self.src_type = 'camera'
            self.source = self.pg_mon.cam_cb.currentData()
            self.cam_thread = CameraThread(self.source); self.cam_thread.start()
        if self.pipeline: self.pipeline.start()
        self.is_running = True; self.frame_count = 0; self.fps_ts.clear(); self.log_lines.clear(); self._log_idx = 0; self.pg_mon.log.clear()
        self.disp_timer.start(); self.perf_timer.start(); self.pg_mon._apply_running()

    def stop(self):
        self.is_running = False; self.disp_timer.stop(); self.perf_timer.stop()
        if self.cam_thread: self.cam_thread.stop(); self.cam_thread = None
        if self.file_thread: self.file_thread.stop(); self.file_thread = None
        if self.pipeline: self.pipeline.stop()
        self.pg_mon._apply_stopped(); self.pg_mon.btn_file.setEnabled(True)

    def on_frame(self):
        if not self.is_running: return
        raw = None
        if self.src_type == 'camera' and self.cam_thread: raw = self.cam_thread.get_frame()
        elif self.src_type == 'video_file' and self.file_thread: raw = self.file_thread.get_frame()
        if raw is None: return
        self.frame_count += 1; self.fps_ts.append(time.time())
        # 四路模式
        if self.quad_mode and self.engine and self.engine.is_ready:
            self._result_frame, self._detections = self._detect_quad(raw)
        # video 文件 / pipeline 未就绪 → 引擎直出 (可靠)
        elif self.engine and self.engine.is_ready:
            rf, det, _ = self.engine.detect(raw); self._result_frame = rf; self._detections = det
        if self._result_frame is not None:
            fps = self._calc_fps(); d = self._result_frame.copy()
            cv2.putText(d, f"{fps:.0f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            rgb = cv2.cvtColor(d, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.pg_mon.vid_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(self.pg_mon.vid_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.pg_mon.perf.fps_gauge.set_value(fps, ".1f")
            self.pg_mon.perf.fps_gauge.set_color(C_BLUE if fps >= 24 else (C_ORANGE if fps >= 15 else C_RED))
            self.pg_mon.perf.fps_spark.add_value(fps)
            self.pg_mon.perf.fps_value_label.setText(f"{fps:.1f}")
            if self.engine:
                lat = self.engine.stats['avg_latency']
                self.pg_mon.perf.lat_gauge.set_value(lat, ".0f")
                self.pg_mon.perf.lat_gauge.set_color(C_GREEN if lat < 30 else (C_ORANGE if lat < 80 else C_RED))
                self.pg_mon.perf.det_gauge.set_value(self.engine.stats['detection_count'])
            # 同步碰撞预警记录 (车距<2m) 到记录页
            if self.engine and getattr(self.engine, 'collision_log', None):
                log = self.engine.collision_log
                if log and log[-1]['seq'] > self._last_collision_seq:
                    for rec in log:
                        if rec['seq'] > self._last_collision_seq:
                            self.pg_rec.add_collision_record(rec['distance'], rec['confidence'], rec['time'])
                    self._last_collision_seq = log[-1]['seq']
            if len(self.log_lines) > 80: self.log_lines = self.log_lines[-80:]

    def _calc_fps(self):
        if len(self.fps_ts) < 2: return 0.0
        el = self.fps_ts[-1] - self.fps_ts[0]; return (len(self.fps_ts) - 1) / el if el > 0 else 0.0

    def screenshot(self):
        if self._result_frame is None: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S"); path, _ = QFileDialog.getSaveFileName(self, "保存截图", f"lpr_{ts}.jpg", "JPEG (*.jpg);;PNG (*.png)")
        if path: cv2.imwrite(path, self._result_frame)

    # ── 深度标定 ──────────────────────────────────────────
    def _toggle_calib(self):
        if not (self.engine and self.engine.depth_estimator):
            self.log_lines.append("[标定] 深度估计器未加载, 无法标定")
            return
        self._calib_mode = not self._calib_mode
        if self._calib_mode:
            self.log_lines.append("[标定] 已进入标定模式: 点击画面中一个已知距离的地面点")
        else:
            self.log_lines.append("[标定] 已退出标定模式")

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.pg_mon.vid_lbl and event.type() == QEvent.MouseButtonPress and self._calib_mode:
            pos = event.position()
            self._on_calib_click(int(pos.x()), int(pos.y()))
            return True
        return super().eventFilter(obj, event)

    def _map_click_to_frame(self, lx: int, ly: int):
        """把视频标签上的点击坐标映射回原帧坐标 (KeepAspectRatio 留白缩放)"""
        frame = self._result_frame
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        lbl = self.pg_mon.vid_lbl
        lw, lh = lbl.width(), lbl.height()
        if lw <= 0 or lh <= 0:
            return None
        scale = min(lw / fw, lh / fh)
        dw, dh = fw * scale, fh * scale
        ox, oy = (lw - dw) / 2, (lh - dh) / 2
        x = (lx - ox) / scale
        y = (ly - oy) / scale
        if not (0 <= x < fw and 0 <= y < fh):
            return None
        return int(x), int(y)

    def _on_calib_click(self, lx: int, ly: int):
        pt = self._map_click_to_frame(lx, ly)
        if pt is None:
            return
        px, py = pt
        meters, ok = QInputDialog.getDouble(self, "深度标定", "该地面点到相机的实际距离 (米):", 10.0, 0.1, 200.0, 1)
        if not ok:
            return
        ok, msg = self.engine.calibrate_depth(self._result_frame, px, py, meters)
        self.log_lines.append(f"[标定] {msg}")
        if ok:
            self._calib_mode = False
            self.log_lines.append("[标定] 已退出标定模式")

    def _flush_log(self):
        if self._log_idx >= len(self.log_lines): return
        for line in self.log_lines[self._log_idx:]: self.pg_mon.log.append(line)
        self._log_idx = len(self.log_lines); sb = self.pg_mon.log.verticalScrollBar()
        if sb.maximum() > 0: sb.setValue(sb.maximum())

    def _on_perf_tick(self):
        import psutil
        self._cpu_val = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        self._mem_val = mem.percent
        if self._gpu_name is None:
            name, _, _ = _get_gpu_stats()
            if name: self._gpu_name = name; self.pg_mon.perf.gpu_info.setText(f"GPU: {name}")
        _, self._gpu_util, self._gpu_mem_val = _get_gpu_stats()
        self.pg_mon.perf.cpu_ring.set_value(self._cpu_val)
        self.pg_mon.perf.mem_ring.set_value(self._mem_val)
        self.pg_mon.perf.gpu_ring.set_value(self._gpu_util)
        self.pg_mon.perf.vram_ring.set_value(self._gpu_mem_val)

    def closeEvent(self, e): self.stop(); e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(STYLE); w = MainWindow(); w.show(); sys.exit(app.exec())
