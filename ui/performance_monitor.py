"""
性能监视器组件 - v6.0 macOS 深色风格
"""
import psutil

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer

_FONT = "'PingFang SC', 'Microsoft YaHei', 'SF Pro Text', 'Helvetica Neue', sans-serif"


def _try_get_gpu_info():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = util.gpu
        gpu_mem = (mem.used / mem.total) * 100
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        pynvml.nvmlShutdown()
        return gpu_util, gpu_mem, gpu_name
    except Exception:
        return None, None, None


class MetricBar(QWidget):

    def __init__(self, label: str, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.name_label = QLabel(label)
        self.name_label.setFixedWidth(42)
        self.name_label.setStyleSheet(f"color: #636366; font-size: 12px; background: transparent; border: none; font-family: {_FONT};")
        layout.addWidget(self.name_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.setStyleSheet("""
            QProgressBar {
                background-color: #3a3a3c;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #e5e5ea;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.bar, stretch=1)

        self.value_label = QLabel("--")
        self.value_label.setFixedWidth(38)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet(f"color: #e5e5ea; font-size: 12px; font-weight: 600; background: transparent; border: none; font-family: {_FONT};")
        layout.addWidget(self.value_label)

    def set_value(self, value: float):
        self.value_label.setText(f"{value:.0f}%")
        self.bar.setValue(int(min(value, 100)))

        if value > 85:
            color = "#ff453a"
        elif value > 60:
            color = "#ff9f0a"
        else:
            color = "#e5e5ea"

        self.value_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600; background: transparent; border: none; font-family: {_FONT};")
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #3a3a3c;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)


class PerformanceMonitor(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.cpu_bar = MetricBar("CPU")
        layout.addWidget(self.cpu_bar)

        self.mem_bar = MetricBar("MEM")
        layout.addWidget(self.mem_bar)

        self.gpu_bar = MetricBar("GPU")
        layout.addWidget(self.gpu_bar)

        self.vram_bar = MetricBar("VRAM")
        layout.addWidget(self.vram_bar)

        self._gpu_name = ""
        self._init_gpu()

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self._refresh()

    def _init_gpu(self):
        _, _, name = _try_get_gpu_info()
        if name:
            self._gpu_name = name

    def _refresh(self):
        cpu = psutil.cpu_percent(interval=0)
        self.cpu_bar.set_value(cpu)

        mem = psutil.virtual_memory()
        self.mem_bar.set_value(mem.percent)

        gpu_util, gpu_mem, _ = _try_get_gpu_info()
        if gpu_util is not None:
            self.gpu_bar.set_value(gpu_util)
            self.vram_bar.set_value(gpu_mem)
        else:
            self.gpu_bar.set_value(0)
            self.vram_bar.set_value(0)
