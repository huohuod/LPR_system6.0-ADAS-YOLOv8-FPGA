"""
设置对话框 - v6.0 macOS 深色风格
"""
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QGridLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QComboBox, QPushButton, QDialogButtonBox,
    QCheckBox
)
from PySide6.QtCore import Qt

_FONT = "'PingFang SC', 'Microsoft YaHei', 'SF Pro Text', 'Helvetica Neue', sans-serif"

_STYLE = f"""
QDialog {{
    background-color: #1c1c1e;
    color: #e5e5ea;
    font-family: {_FONT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QTabWidget::pane {{
    border: none;
    background-color: #1c1c1e;
}}
QTabBar::tab {{
    background-color: #2c2c2e;
    color: #636366;
    padding: 8px 24px;
    border: none;
    border-radius: 8px;
    margin-right: 4px;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background-color: #3a3a3c;
    color: #e5e5ea;
}}
QGroupBox {{
    color: #636366;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: none;
    border-top: 1px solid #3a3a3c;
    margin-top: 14px;
    padding-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 2px;
    padding: 0 4px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: none;
    padding: 5px 8px;
    border-radius: 8px;
}}
QComboBox {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: none;
    padding: 5px 8px;
    border-radius: 8px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    selection-background-color: #3a3a3c;
    selection-color: #fff;
    border: none;
    border-radius: 8px;
}}
QPushButton {{
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: none;
    padding: 7px 20px;
    border-radius: 8px;
}}
QPushButton:hover {{ background-color: #3a3a3c; }}
QCheckBox {{ color: #e5e5ea; background: transparent; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    background-color: #2c2c2e;
    border: none;
    border-radius: 4px;
}}
QCheckBox::indicator:checked {{
    background-color: #30d158;
}}
"""


class SettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 440)
        self.setStyleSheet(_STYLE)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        tabs = QTabWidget()

        # ── 模型 tab ──
        model_tab = QWidget()
        ml = QVBoxLayout(model_tab)
        ml.setContentsMargins(4, 12, 4, 12)

        mg = QGroupBox("DETECTION")
        mgl = QGridLayout(mg)
        mgl.setSpacing(12)
        mgl.setColumnMinimumWidth(0, 120)

        mgl.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.4)
        mgl.addWidget(self.conf_spin, 0, 1)

        mgl.addWidget(QLabel("IoU Threshold:"), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.10, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        mgl.addWidget(self.iou_spin, 1, 1)

        mgl.addWidget(QLabel("Input Size:"), 2, 0)
        self.size_combo = QComboBox()
        for s in [320, 416, 480, 640, 800]:
            self.size_combo.addItem(str(s), s)
        self.size_combo.setCurrentText('640')
        mgl.addWidget(self.size_combo, 2, 1)

        mgl.addWidget(QLabel("Tilt Correction:"), 3, 0)
        self.tilt_cb = QCheckBox("Enable")
        mgl.addWidget(self.tilt_cb, 3, 1)

        mgl.addWidget(QLabel("FP16 Acceleration:"), 4, 0)
        self.fp16_cb = QCheckBox("Enable")
        self.fp16_cb.setChecked(True)
        mgl.addWidget(self.fp16_cb, 4, 1)

        ml.addWidget(mg)
        ml.addStretch()
        tabs.addTab(model_tab, "Model")

        # ── 颜色 tab ──
        color_tab = QWidget()
        cl = QVBoxLayout(color_tab)
        cl.setContentsMargins(4, 12, 4, 12)

        cg = QGroupBox("COLOR CALIBRATION")
        cgl = QGridLayout(cg)
        cgl.setSpacing(12)
        cgl.setColumnMinimumWidth(0, 120)

        cgl.addWidget(QLabel("Brightness:"), 0, 0)
        self.bright_spin = QSpinBox()
        self.bright_spin.setRange(-100, 100)
        cgl.addWidget(self.bright_spin, 0, 1)

        cgl.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(-100, 100)
        cgl.addWidget(self.contrast_spin, 1, 1)

        cgl.addWidget(QLabel("Saturation:"), 2, 0)
        self.sat_spin = QSpinBox()
        self.sat_spin.setRange(-100, 100)
        cgl.addWidget(self.sat_spin, 2, 1)

        cgl.addWidget(QLabel("White Balance R:"), 3, 0)
        self.wb_r = QSpinBox()
        self.wb_r.setRange(-50, 50)
        cgl.addWidget(self.wb_r, 3, 1)

        cgl.addWidget(QLabel("White Balance G:"), 4, 0)
        self.wb_g = QSpinBox()
        self.wb_g.setRange(-50, 50)
        cgl.addWidget(self.wb_g, 4, 1)

        cgl.addWidget(QLabel("White Balance B:"), 5, 0)
        self.wb_b = QSpinBox()
        self.wb_b.setRange(-50, 50)
        cgl.addWidget(self.wb_b, 5, 1)

        cl.addWidget(cg)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(self._reset)
        cl.addWidget(reset_btn)
        cl.addStretch()
        tabs.addTab(color_tab, "Color")

        layout.addWidget(tabs)

        # 底部按钮
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Save).setText("Save")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _reset(self) -> None:
        self.bright_spin.setValue(0)
        self.contrast_spin.setValue(0)
        self.sat_spin.setValue(0)
        self.wb_r.setValue(0)
        self.wb_g.setValue(0)
        self.wb_b.setValue(0)

    def get_model_config(self) -> dict:
        return {
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'img_size': self.size_combo.currentData(),
            'enable_tilt_correction': self.tilt_cb.isChecked(),
            'fp16': self.fp16_cb.isChecked(),
        }

    def get_color_config(self) -> dict:
        return {
            'brightness': self.bright_spin.value(),
            'contrast': self.contrast_spin.value(),
            'saturation': self.sat_spin.value(),
            'wb_r': self.wb_r.value(),
            'wb_g': self.wb_g.value(),
            'wb_b': self.wb_b.value(),
        }
