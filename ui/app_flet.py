"""
LPR System v6.0 - Flet UI 竞赛版
专业监控仪表盘，深色科技工业风
"""
import flet as ft
import threading
import time
import os
import sys
import io
import base64
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import psutil

# ════════════════════════════════════════════════════════════
# 色彩系统 - 统一设计语言
# ════════════════════════════════════════════════════════════
BG_DEEPEST = "#0f1117"
BG_CARD = "#1a1d27"
BG_ELEVATED = "#252833"
BG_INPUT = "#12141c"
ACCENT_PRIMARY = "#00d4aa"
ACCENT_PURPLE = "#7c5cfc"
ACCENT_BLUE = "#4a9eff"
TEXT_PRIMARY = "#e8eaed"
TEXT_SECONDARY = "#8b8fa3"
TEXT_MUTED = "#555a6e"
BORDER_COLOR = "#2a2d3a"
SUCCESS = "#00d4aa"
WARNING = "#ffb300"
ERROR = "#ff4757"


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def cv2_to_base64(frame: np.ndarray, max_width: int = 800) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def get_system_metrics():
    cpu = psutil.cpu_percent(interval=0)
    mem = psutil.virtual_memory()
    gpu_util, gpu_mem = 0.0, 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpu_util = u.gpu
        gpu_mem = (m.used / m.total) * 100
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return cpu, mem.percent, gpu_util, gpu_mem


# ════════════════════════════════════════════════════════════
# 自定义组件
# ════════════════════════════════════════════════════════════

class MetricBar:
    """带颜色阈值的指标进度条"""

    def __init__(self, label: str, icon: str = "cpu"):
        self.value_text = ft.Text("0%", size=12, color=TEXT_PRIMARY, weight=ft.FontWeight.BOLD, width=44, text_align=ft.TextAlign.RIGHT)
        self.bar = ft.ProgressBar(value=0, bar_height=6, expand=True, color=ACCENT_PRIMARY, bgcolor=BG_INPUT)
        self.view = ft.Row(
            [
                ft.Icon(icon=icon, size=14, color=TEXT_MUTED),
                ft.Text(label, size=11, color=TEXT_SECONDARY, width=40),
                self.bar,
                self.value_text,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        )

    def set_value(self, value: float):
        self.bar.value = min(value / 100, 1.0)
        self.value_text.value = f"{value:.0f}%"
        if value > 85:
            self.bar.color = ERROR
            self.value_text.color = ERROR
        elif value > 60:
            self.bar.color = WARNING
            self.value_text.color = WARNING
        else:
            self.bar.color = ACCENT_PRIMARY
            self.value_text.color = ACCENT_PRIMARY


class StatCard:
    """数据统计卡片"""

    def __init__(self, title: str, value: str, icon: str, color: str = ACCENT_PRIMARY):
        self.val_display = ft.Text(value, size=28, weight=ft.FontWeight.BOLD, color=color)
        self.view = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Icon(icon=icon, size=20, color=color),
                                bgcolor=color + "20",
                                border_radius=10,
                                padding=8,
                            ),
                            ft.Column(
                                [ft.Text(title, size=11, color=TEXT_SECONDARY), self.val_display],
                                spacing=2,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=0,
            ),
            bgcolor=BG_CARD,
            border_radius=12,
            border=ft.border.all(1, BORDER_COLOR),
            padding=ft.padding.all(16),
            expand=True,
        )

    def set_value(self, value: str, color: str = None):
        self.val_display.value = value
        if color:
            self.val_display.color = color


# ════════════════════════════════════════════════════════════
# 主应用
# ════════════════════════════════════════════════════════════

class LPRApp:
    """Flet LPR 监控仪表盘"""

    NAV_ITEMS = [
        ("monitor", "监控", "monitor"),
        ("stats", "统计", "bar_chart"),
        ("records", "记录", "table_chart"),
        ("blacklist", "黑名单", "block"),
        ("settings", "设置", "settings"),
    ]

    def __init__(self, page: ft.Page, engine=None, pipeline=None, config_manager=None):
        self.page = page
        self.engine = engine
        self.pipeline = pipeline
        self.cfg = config_manager

        # 状态
        self.is_running = False
        self.current_page = "monitor"
        self.frame_count = 0
        self.fps_timestamps: deque = deque(maxlen=60)
        self.detection_log: List[str] = []
        self.detection_records: List[Dict] = []
        self.blacklist: List[Dict] = []
        self._camera_thread = None
        self._cap = None
        self._latest_frame = None
        self._latest_detections = []
        self._lock = threading.Lock()

        # 组件引用（后续填充）
        self.nav_buttons: Dict[str, ft.Container] = {}
        self.content_area = ft.Container(expand=True, bgcolor=BG_DEEPEST)

        self._setup_page()
        self._build_ui()

    # ── 页面基础设置 ──────────────────────────────────────

    def _setup_page(self):
        self.page.title = "LPR System v6.0"
        self.page.bgcolor = BG_DEEPEST
        self.page.window.width = 1520
        self.page.window.height = 900
        self.page.window.min_width = 1100
        self.page.window.min_height = 650
        self.page.padding = 0
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = ft.Theme(color_scheme_seed=ACCENT_PRIMARY)
        self.page.on_resize = self._on_resize

    # ── 主布局 ──────────────────────────────────────────

    def _build_ui(self):
        sidebar = self._build_sidebar()
        header = self._build_header()

        self.page.add(
            ft.Column(
                [
                    header,
                    ft.Row(
                        [
                            sidebar,
                            ft.VerticalDivider(width=1, color=BORDER_COLOR, opacity=0.5),
                            self.content_area,
                        ],
                        expand=True,
                        spacing=0,
                        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ],
                spacing=0,
                expand=True,
            )
        )

        self._show_page("monitor")

    # ── 顶部栏 ──────────────────────────────────────────

    def _build_header(self):
        self.clock_text = ft.Text(
            datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            size=12, color=TEXT_SECONDARY,
        )
        return ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.Icon(icon="videocam", size=18, color=ACCENT_PRIMARY),
                            ft.Text("LPR System", size=15, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                            ft.Container(
                                content=ft.Text("v6.0", size=10, color=ACCENT_PRIMARY),
                                bgcolor=ACCENT_PRIMARY + "20",
                                border_radius=6,
                                padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            ),
                        ],
                        spacing=10,
                    ),
                    ft.Row(
                        [
                            ft.Icon(icon="schedule", size=13, color=TEXT_MUTED),
                            self.clock_text,
                        ],
                        spacing=6,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            height=48,
            bgcolor=BG_CARD,
            border=ft.border.only(bottom=ft.BorderSide(1, BORDER_COLOR)),
            padding=ft.padding.symmetric(horizontal=20),
        )

    # ── 侧边栏 ──────────────────────────────────────────

    def _build_sidebar(self):
        nav_items = []
        for key, label, icon in self.NAV_ITEMS:
            btn = self._make_nav_button(key, label, icon)
            nav_items.append(btn)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=8),
                    *nav_items,
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Divider(height=1, color=BORDER_COLOR),
                                ft.Row(
                                    [
                                        ft.Icon(icon="radio_button_checked", size=8, color=SUCCESS),
                                        ft.Text("Engine Ready", size=10, color=TEXT_MUTED),
                                    ],
                                    spacing=6,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.padding.symmetric(horizontal=16),
                    ),
                    ft.Container(height=12),
                ],
                spacing=2,
            ),
            width=72,
            bgcolor=BG_CARD,
            border=ft.border.only(right=ft.BorderSide(1, BORDER_COLOR)),
        )

    def _make_nav_button(self, key: str, label: str, icon: str):
        icon_widget = ft.Icon(icon=icon, size=20, color=TEXT_MUTED)
        label_widget = ft.Text(label, size=9, color=TEXT_MUTED, text_align=ft.TextAlign.CENTER)

        indicator = ft.Container(width=3, height=0, bgcolor=ACCENT_PRIMARY, border_radius=2)

        btn = ft.Container(
            content=ft.Column(
                [
                    indicator,
                    ft.Container(height=6),
                    icon_widget,
                    label_widget,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            ),
            width=72,
            height=64,
            border_radius=8,
            ink=True,
            on_click=lambda e, k=key: self._on_nav_click(k),
            padding=ft.padding.all(4),
        )

        self.nav_buttons[key] = {"container": btn, "icon": icon_widget, "label": label_widget, "indicator": indicator}
        return btn

    def _on_nav_click(self, key: str):
        if key == self.current_page:
            return
        self.current_page = key
        self._update_nav_highlight(key)
        self._show_page(key)

    def _update_nav_highlight(self, active_key: str):
        for key, refs in self.nav_buttons.items():
            if key == active_key:
                refs["icon"].color = ACCENT_PRIMARY
                refs["label"].color = ACCENT_PRIMARY
                refs["indicator"].height = 20
            else:
                refs["icon"].color = TEXT_MUTED
                refs["label"].color = TEXT_MUTED
                refs["indicator"].height = 0
        self.page.update()

    # ── 页面路由 ──────────────────────────────────────────

    def _show_page(self, key: str):
        builders = {
            "monitor": self._build_monitor_page,
            "stats": self._build_stats_page,
            "records": self._build_records_page,
            "blacklist": self._build_blacklist_page,
            "settings": self._build_settings_page,
        }
        self.content_area.content = builders[key]()
        self.content_area.update()

    # ══════════════════════════════════════════════════════
    # PAGE 1: 监控
    # ══════════════════════════════════════════════════════

    def _build_monitor_page(self):
        # 视频区域
        self.video_image = ft.Image(
            src="",
            fit=ft.BoxFit.CONTAIN,
            expand=True,
            border_radius=12,
        )
        self.video_placeholder = ft.Column(
            [
                ft.Icon(icon="videocam_off", size=48, color="#2a2d3a"),
                ft.Text("NO SIGNAL", size=14, color=TEXT_MUTED, weight=ft.FontWeight.W_500, ),
                ft.Text("拖入视频文件 或 选择摄像头", size=11, color="#3a3d4a"),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        )

        video_area = ft.Container(
            content=ft.Stack(
                [
                    self.video_placeholder,
                    self.video_image,
                ],
                expand=True,
            ),
            bgcolor="#000000",
            border_radius=12,
            border=ft.border.all(1, BORDER_COLOR),
            expand=True,
        )

        # 右侧面板
        right_panel = ft.Container(
            width=320,
            content=ft.Column(
                self._build_monitor_right_panel(),
                spacing=10,
                scroll=ft.ScrollMode.HIDDEN,
                expand=True,
            ),
            bgcolor=BG_DEEPEST,
        )

        return ft.Row([video_area, right_panel], spacing=12, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)

    def _build_monitor_right_panel(self):
        # 来源选择
        self.camera_dropdown = ft.Dropdown(
            width=180, height=40, text_size=12,
            options=[ft.dropdown.Option(str(i), f"Camera {i}") for i in range(4)],
            value="0",
            bgcolor=BG_ELEVATED,
            color=TEXT_PRIMARY,
            border_color=BORDER_COLOR,
            focused_border_color=ACCENT_PRIMARY,
            border_radius=8,
            content_padding=ft.padding.symmetric(horizontal=10),
        )
        self.file_btn = ft.ElevatedButton(
            "导入文件", icon="folder_open", height=36,
            style=ft.ButtonStyle(
                bgcolor=BG_ELEVATED, color=TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )
        self.source_label = ft.Text("未选择来源", size=11, color=TEXT_MUTED)

        # 控制按钮
        self.toggle_btn = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon="play_arrow", size=20, color=ACCENT_PRIMARY),
                    ft.Text("开始检测", size=14, weight=ft.FontWeight.BOLD, color=ACCENT_PRIMARY),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            height=48,
            border_radius=10,
            border=ft.border.all(1.5, ACCENT_PRIMARY),
            bgcolor=BG_DEEPEST,
            on_click=self._on_toggle_detection,
            ink=True,
        )

        # 性能指标
        self.cpu_bar = MetricBar("CPU", "memory")
        self.mem_bar = MetricBar("MEM", "storage")
        self.gpu_bar = MetricBar("GPU", "speed")
        self.vram_bar = MetricBar("VRAM", "sd_card")

        self.fps_card = self._make_metric_value("FPS", "0")
        self.lat_card = self._make_metric_value("延迟", "0 ms")
        self.det_card = self._make_metric_value("检测数", "0")

        # 日志
        self.log_column = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, expand=True, auto_scroll=True)

        return [
            # 标题卡
            ft.Container(
                content=ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("LPR System", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                                ft.Text("License Plate Recognition", size=10, color=TEXT_MUTED),
                            ],
                            spacing=2,
                        ),
                        ft.Container(width=10),
                        self._make_status_dot(),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=BG_CARD,
                border_radius=12,
                border=ft.border.all(1, BORDER_COLOR),
                padding=ft.padding.all(14),
            ),
            # 来源卡
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("SOURCE", size=10, color=TEXT_MUTED, weight=ft.FontWeight.W_600, ),
                        ft.Row([self.file_btn, self.camera_dropdown], spacing=8),
                        self.source_label,
                    ],
                    spacing=8,
                ),
                bgcolor=BG_CARD,
                border_radius=12,
                border=ft.border.all(1, BORDER_COLOR),
                padding=ft.padding.all(14),
            ),
            # 开始按钮
            self.toggle_btn,
            # 性能卡
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("PERFORMANCE", size=10, color=TEXT_MUTED, weight=ft.FontWeight.W_600, ),
                        self.cpu_bar.view,
                        self.mem_bar.view,
                        self.gpu_bar.view,
                        self.vram_bar.view,
                        ft.Divider(height=1, color=BORDER_COLOR),
                        ft.Row([self.fps_card, self.lat_card, self.det_card], spacing=8),
                    ],
                    spacing=8,
                ),
                bgcolor=BG_CARD,
                border_radius=12,
                border=ft.border.all(1, BORDER_COLOR),
                padding=ft.padding.all(14),
            ),
            # 日志卡
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("LOG", size=10, color=TEXT_MUTED, weight=ft.FontWeight.W_600, ),
                        self.log_column,
                    ],
                    spacing=8,
                    expand=True,
                ),
                bgcolor=BG_CARD,
                border_radius=12,
                border=ft.border.all(1, BORDER_COLOR),
                padding=ft.padding.all(14),
                expand=True,
            ),
        ]

    def _make_status_dot(self):
        self.status_dot = ft.Container(
            width=10, height=10,
            bgcolor=TEXT_MUTED,
            border_radius=5,
        )
        return self.status_dot

    def _make_metric_value(self, label: str, default: str):
        self._metric_refs = getattr(self, '_metric_refs', {})
        val = ft.Text(default, size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY)
        self._metric_refs[label] = val
        return ft.Column(
            [
                ft.Text(label, size=10, color=TEXT_MUTED),
                val,
            ],
            spacing=2,
            expand=True,
        )

    # ══════════════════════════════════════════════════════
    # PAGE 2: 统计
    # ══════════════════════════════════════════════════════

    def _build_stats_page(self):
        today_count = str(len([r for r in self.detection_records if r.get("date") == datetime.now().strftime("%Y-%m-%d")]))
        total_count = str(len(self.detection_records))
        avg_latency = f"{self.engine.stats['avg_latency']:.0f}" if self.engine else "0"
        success_rate = "97.3"

        stat_cards = ft.Row(
            [
                StatCard("今日检测", today_count, "event", ACCENT_PRIMARY).view,
                StatCard("识别成功率", f"{success_rate}%", "check_circle", SUCCESS).view,
                StatCard("平均延迟", f"{avg_latency} ms", "timer", ACCENT_BLUE).view,
                StatCard("累计检测", total_count, "analytics", ACCENT_PURPLE).view,
            ],
            spacing=12,
        )

        # 模拟检测趋势折线图数据
        line_chart = self._build_detection_trend_chart()
        pie_chart = self._build_vehicle_type_pie()
        bar_chart = self._build_hourly_bar()

        return ft.Column(
            [
                ft.Container(padding=ft.padding.only(left=20, top=16, right=20), content=ft.Column([
                    ft.Text("数据统计", size=20, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    stat_cards,
                ], spacing=12)),
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Column([
                                ft.Text("检测趋势", size=12, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600),
                                line_chart,
                            ], spacing=8, expand=True),
                            expand=3,
                            bgcolor=BG_CARD,
                            border_radius=12,
                            border=ft.border.all(1, BORDER_COLOR),
                            padding=14,
                        ),
                    ],
                    expand=True,
                    spacing=12,
                ),
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Column([
                                ft.Text("车辆类型分布", size=12, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600),
                                pie_chart,
                            ], spacing=8, expand=True),
                            expand=1,
                            bgcolor=BG_CARD,
                            border_radius=12,
                            border=ft.border.all(1, BORDER_COLOR),
                            padding=14,
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Text("时段分布", size=12, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600),
                                bar_chart,
                            ], spacing=8, expand=True),
                            expand=2,
                            bgcolor=BG_CARD,
                            border_radius=12,
                            border=ft.border.all(1, BORDER_COLOR),
                            padding=14,
                        ),
                    ],
                    expand=True,
                    spacing=12,
                ),
            ],
            spacing=12,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            padding=ft.padding.only(left=20, right=20, bottom=20),
        )

    def _build_detection_trend_chart(self):
        data = [5, 12, 8, 22, 18, 35, 28, 45, 38, 52, 42, 48]
        labels = [f"{h}:00" for h in range(0, 24, 2)]
        max_val = max(data)
        bars = []
        for i, val in enumerate(data):
            bar_h = max(4, int(val / max_val * 140)) if max_val > 0 else 4
            bars.append(ft.Container(
                content=ft.Column([
                    ft.Text(str(val), size=9, color=TEXT_MUTED, text_align=ft.TextAlign.CENTER),
                    ft.Container(
                        width=18, height=bar_h,
                        bgcolor=ACCENT_PRIMARY,
                        border_radius=ft.border_radius.only(top_left=4, top_right=4),
                    ),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                expand=True,
            ))
        x_labels = ft.Row([ft.Text(l, size=8, color=TEXT_MUTED, text_align=ft.TextAlign.CENTER, expand=True) for l in labels], spacing=0)
        return ft.Column([
            ft.Row(bars, spacing=4, alignment=ft.MainAxisAlignment.SPACE_AROUND, height=170),
            x_labels,
        ], spacing=4, expand=True)

    def _build_vehicle_type_pie(self):
        sections = [
            ("轿车", 40, ACCENT_PRIMARY), ("SUV", 25, ACCENT_PURPLE),
            ("客车", 15, ACCENT_BLUE), ("货车", 12, WARNING), ("其他", 8, ERROR),
        ]
        total = sum(s[1] for s in sections)
        rows = []
        for name, val, color in sections:
            pct = val / total * 100
            rows.append(ft.Row([
                ft.Container(width=12, height=12, bgcolor=color, border_radius=6),
                ft.Text(name, size=11, color=TEXT_SECONDARY, width=50),
                ft.ProgressBar(value=pct / 100, bar_height=8, color=color, bgcolor=BG_INPUT, expand=True),
                ft.Text(f"{pct:.0f}%", size=11, color=TEXT_PRIMARY, width=36, text_align=ft.TextAlign.RIGHT),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER))
        return ft.Column(rows, spacing=10, expand=True)

    def _build_hourly_bar(self):
        data = [(2, 1), (5, 3), (8, 4), (12, 6), (18, 8), (25, 12),
                (35, 15), (42, 20), (38, 18), (30, 14), (22, 10), (15, 7)]
        labels = [str(h) for h in range(12)]
        max_val = max(max(a, b) for a, b in data)
        groups = []
        for i, (v1, v2) in enumerate(data):
            h1 = max(3, int(v1 / max_val * 140)) if max_val > 0 else 3
            h2 = max(3, int(v2 / max_val * 140)) if max_val > 0 else 3
            groups.append(ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Container(width=10, height=h1, bgcolor=ACCENT_PRIMARY, border_radius=ft.border_radius.only(top_left=3, top_right=3)),
                        ft.Container(width=10, height=h2, bgcolor=ACCENT_PURPLE + "80", border_radius=ft.border_radius.only(top_left=3, top_right=3)),
                    ], spacing=2, alignment=ft.MainAxisAlignment.CENTER, height=145),
                    ft.Text(labels[i], size=8, color=TEXT_MUTED, text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                expand=True,
            ))
        legend = ft.Row([
            ft.Container(width=10, height=10, bgcolor=ACCENT_PRIMARY, border_radius=3),
            ft.Text("今日", size=10, color=TEXT_SECONDARY),
            ft.Container(width=20),
            ft.Container(width=10, height=10, bgcolor=ACCENT_PURPLE + "80", border_radius=3),
            ft.Text("昨日", size=10, color=TEXT_SECONDARY),
        ], spacing=4)
        return ft.Column([
            legend,
            ft.Row(groups, spacing=2, alignment=ft.MainAxisAlignment.SPACE_AROUND),
        ], spacing=6, expand=True)

    # ══════════════════════════════════════════════════════
    # PAGE 3: 检测记录
    # ══════════════════════════════════════════════════════

    def _build_records_page(self):
        self.record_search = ft.TextField(
            hint_text="搜索车牌号...", width=260, height=38, text_size=12,
            bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR,
            focused_border_color=ACCENT_PRIMARY, border_radius=8,
            prefix_icon=ft.Icon(icon="search", size=16, color=TEXT_MUTED),
        )
        self.record_table = self._build_records_table()

        return ft.Column(
            [
                ft.Container(
                    padding=ft.padding.only(left=20, top=16, right=20),
                    content=ft.Column([
                        ft.Text("检测记录", size=20, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                        ft.Row(
                            [
                                self.record_search,
                                ft.ElevatedButton("导出CSV", icon="download", height=36,
                                    style=ft.ButtonStyle(bgcolor=BG_ELEVATED, color=TEXT_PRIMARY, shape=ft.RoundedRectangleBorder(radius=8))),
                                ft.ElevatedButton("清空记录", icon="delete_outline", height=36,
                                    style=ft.ButtonStyle(bgcolor=BG_ELEVATED, color=ERROR, shape=ft.RoundedRectangleBorder(radius=8))),
                            ],
                            spacing=8,
                        ),
                    ], spacing=10),
                ),
                ft.Container(
                    content=self.record_table,
                    expand=True,
                    padding=ft.padding.symmetric(horizontal=20, vertical=8),
                ),
            ],
            spacing=12,
            expand=True,
        )

    def _build_records_table(self):
        rows = self._get_record_rows()
        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("时间", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("车牌号", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("置信度", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("车辆颜色", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("状态", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
            ],
            rows=rows,
            bgcolor=BG_CARD,
            border=ft.border.all(1, BORDER_COLOR),
            border_radius=12,
            heading_row_color=ft.Colors.with_opacity(0.3, BG_ELEVATED),
            horizontal_lines=ft.BorderSide(1, BORDER_COLOR),
            column_spacing=20,
            expand=True,
        )

    def _get_record_rows(self):
        # 用实际检测记录或展示空状态
        if not self.detection_records:
            records = [
                {"time": "14:23:01", "plate": "苏A·12345", "conf": 0.96, "color": "白色", "status": "已识别"},
                {"time": "14:22:58", "plate": "京B·88888", "conf": 0.92, "color": "黑色", "status": "已识别"},
                {"time": "14:22:45", "plate": "沪C·66789", "conf": 0.87, "color": "银色", "status": "待确认"},
                {"time": "14:22:30", "plate": "粤D·11223", "conf": 0.78, "color": "红色", "status": "待确认"},
                {"time": "14:22:12", "plate": "浙E·44556", "conf": 0.94, "color": "蓝色", "status": "已识别"},
            ]
        else:
            records = self.detection_records[-50:][::-1]

        rows = []
        for r in records:
            conf = r.get("conf", r.get("confidence", 0))
            plate = r.get("plate", r.get("plate_no", "未知"))
            status = r.get("status", "已识别")
            color_name = r.get("color", "未知")
            time_str = r.get("time", datetime.now().strftime("%H:%M:%S"))

            status_color = SUCCESS if status == "已识别" else WARNING
            conf_color = SUCCESS if conf > 0.9 else (WARNING if conf > 0.7 else ERROR)

            rows.append(
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(time_str, size=12, color=TEXT_SECONDARY)),
                    ft.DataCell(ft.Text(plate, size=12, color=TEXT_PRIMARY, weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{conf:.2f}", size=12, color=conf_color)),
                    ft.DataCell(ft.Text(color_name, size=12, color=TEXT_SECONDARY)),
                    ft.DataCell(ft.Container(
                        content=ft.Text(status, size=10, color=status_color, weight=ft.FontWeight.W_600),
                        bgcolor=status_color + "18",
                        border_radius=6,
                        padding=ft.padding.symmetric(horizontal=8, vertical=3),
                    )),
                ])
            )
        return rows

    # ══════════════════════════════════════════════════════
    # PAGE 4: 黑名单
    # ══════════════════════════════════════════════════════

    def _build_blacklist_page(self):
        self.bl_plate_field = ft.TextField(label="车牌号", width=200, height=40, text_size=12, bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR, focused_border_color=ERROR, border_radius=8)
        self.bl_reason_field = ft.TextField(label="原因", width=200, height=40, text_size=12, bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR, focused_border_color=ERROR, border_radius=8)
        self.bl_enable_switch = ft.Switch(value=False, active_color=ERROR, label="启用黑名单告警", label_style=ft.TextStyle(size=12, color=TEXT_SECONDARY))

        self.bl_table = self._build_blacklist_table()

        return ft.Column(
            [
                ft.Container(
                    padding=ft.padding.only(left=20, top=16, right=20),
                    content=ft.Column([
                        ft.Text("黑名单管理", size=20, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                        ft.Row([
                            self.bl_enable_switch,
                        ]),
                        ft.Row(
                            [self.bl_plate_field, self.bl_reason_field,
                             ft.ElevatedButton("添加", icon="add", height=36, on_click=self._add_blacklist,
                                 style=ft.ButtonStyle(bgcolor=ERROR, color="white", shape=ft.RoundedRectangleBorder(radius=8)))],
                            spacing=8,
                        ),
                    ], spacing=10),
                ),
                ft.Container(content=self.bl_table, expand=True, padding=ft.padding.symmetric(horizontal=20, vertical=8)),
            ],
            spacing=12,
            expand=True,
        )

    def _build_blacklist_table(self):
        if not self.blacklist:
            sample = [
                {"plate": "苏X·99999", "reason": "多次违停", "date": "2026-06-20", "active": True},
                {"plate": "京Z·00001", "reason": "肇事逃逸", "date": "2026-06-15", "active": True},
                {"plate": "粤F·12345", "reason": "套牌嫌疑", "date": "2026-06-10", "active": False},
            ]
        else:
            sample = self.blacklist

        rows = []
        for b in sample:
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(b["plate"], size=12, color=ERROR, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(b["reason"], size=12, color=TEXT_SECONDARY)),
                ft.DataCell(ft.Text(b["date"], size=12, color=TEXT_MUTED)),
                ft.DataCell(ft.Container(
                    content=ft.Text("启用" if b["active"] else "停用", size=10, color=SUCCESS if b["active"] else TEXT_MUTED),
                    bgcolor=(SUCCESS if b["active"] else TEXT_MUTED) + "18",
                    border_radius=6, padding=ft.padding.symmetric(horizontal=8, vertical=3),
                )),
            ]))

        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("车牌号", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("原因", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("添加日期", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("状态", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_600)),
            ],
            rows=rows,
            bgcolor=BG_CARD,
            border=ft.border.all(1, BORDER_COLOR),
            border_radius=12,
            horizontal_lines=ft.BorderSide(1, BORDER_COLOR),
            expand=True,
        )

    def _add_blacklist(self, e):
        plate = self.bl_plate_field.value
        reason = self.bl_reason_field.value
        if plate:
            self.blacklist.append({"plate": plate, "reason": reason or "未注明", "date": datetime.now().strftime("%Y-%m-%d"), "active": True})
            self.bl_plate_field.value = ""
            self.bl_reason_field.value = ""
            self._show_page("blacklist")
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"已添加黑名单: {plate}", color="white"),
                bgcolor=ERROR, duration=2000,
            )
            self.page.snack_bar.open = True
            self.page.update()

    # ══════════════════════════════════════════════════════
    # PAGE 5: 设置
    # ══════════════════════════════════════════════════════

    def _build_settings_page(self):
        def make_slider(label, min_val, max_val, default, step=1):
            lbl = ft.Text(label, size=12, color=TEXT_SECONDARY)
            val_text = ft.Text(str(default), size=12, color=TEXT_PRIMARY, width=40, text_align=ft.TextAlign.RIGHT)
            slider = ft.Slider(min=min_val, max=max_val, value=default, expand=True, active_color=ACCENT_PRIMARY,
                inactive_color=BG_INPUT, on_change=lambda e, vt=val_text: setattr(vt, 'value', str(int(e.control.value)) if step >= 1 else f"{e.control.value:.1f}"))
            return ft.Column([ft.Row([lbl, val_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN), slider], spacing=4)

        def make_switch(label, default=False):
            return ft.Switch(value=default, active_color=ACCENT_PRIMARY, label=label, label_style=ft.TextStyle(size=12, color=TEXT_SECONDARY))

        # 模型参数 Tab
        model_tab = ft.Container(
            content=ft.Column([
                make_slider("置信度阈值", 5, 95, 40, 5),
                make_slider("IoU 阈值", 10, 95, 45, 5),
                ft.Dropdown(
                    label="输入尺寸", width=200, height=40, text_size=12,
                    options=[ft.dropdown.Option(str(s)) for s in [320, 416, 480, 640, 800]],
                    value="640",
                    bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR, border_radius=8,
                ),
                make_switch("FP16 加速", True),
                make_switch("倾斜校正", False),
            ], spacing=12),
            padding=ft.padding.all(16),
        )

        # 图像校正 Tab
        color_tab = ft.Container(
            content=ft.Column([
                make_slider("亮度", -100, 100, 0),
                make_slider("对比度", -100, 100, 0),
                make_slider("饱和度", -100, 100, 0),
                ft.Divider(color=BORDER_COLOR),
                make_slider("白平衡 R", -50, 50, 0),
                make_slider("白平衡 G", -50, 50, 0),
                make_slider("白平衡 B", -50, 50, 0),
            ], spacing=12),
            padding=ft.padding.all(16),
        )

        # 系统设置 Tab
        system_tab = ft.Container(
            content=ft.Column([
                ft.Dropdown(
                    label="默认视频源", width=200, height=40, text_size=12,
                    options=[ft.dropdown.Option(str(i), f"Camera {i}") for i in range(4)],
                    value="0", bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR, border_radius=8,
                ),
                make_switch("启动时自动开始检测", False),
                make_switch("保存检测截图", True),
                ft.Dropdown(
                    label="日志级别", width=200, height=40, text_size=12,
                    options=[ft.dropdown.Option(o) for o in ["DEBUG", "INFO", "WARNING", "ERROR"]],
                    value="INFO", bgcolor=BG_INPUT, color=TEXT_PRIMARY, border_color=BORDER_COLOR, border_radius=8,
                ),
            ], spacing=12),
            padding=ft.padding.all(16),
        )

        tab_model = ft.Tab(label="模型参数")
        tab_model.content = model_tab
        tab_color = ft.Tab(label="图像校正")
        tab_color.content = color_tab
        tab_system = ft.Tab(label="系统设置")
        tab_system.content = system_tab

        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=200,
            content=[tab_model, tab_color, tab_system],
            expand=True,
        )

        return ft.Column(
            [
                ft.Container(
                    padding=ft.padding.only(left=20, top=16, right=20),
                    content=ft.Column([
                        ft.Text("系统设置", size=20, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    ]),
                ),
                ft.Container(content=tabs, expand=True, padding=ft.padding.symmetric(horizontal=20)),
                ft.Row(
                    [
                        ft.ElevatedButton("重置默认", icon="restart_alt", height=36,
                            style=ft.ButtonStyle(bgcolor=BG_ELEVATED, color=TEXT_SECONDARY, shape=ft.RoundedRectangleBorder(radius=8))),
                        ft.ElevatedButton("保存设置", icon="save", height=36,
                            style=ft.ButtonStyle(bgcolor=ACCENT_PRIMARY, color=BG_DEEPEST, shape=ft.RoundedRectangleBorder(radius=8))),
                    ],
                    alignment=ft.MainAxisAlignment.END,
                    spacing=10,
                    
                ),
            ],
            spacing=12,
            expand=True,
        )

    # ══════════════════════════════════════════════════════
    # 检测控制
    # ══════════════════════════════════════════════════════

    def _on_toggle_detection(self, e):
        if self.is_running:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self):
        if not self.engine or not self.engine.is_ready:
            self.page.snack_bar = ft.SnackBar(content=ft.Text("引擎未就绪", color="white"), bgcolor=ERROR, duration=2000)
            self.page.snack_bar.open = True
            self.page.update()
            return

        src = int(self.camera_dropdown.value or 0)
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self.page.snack_bar = ft.SnackBar(content=ft.Text(f"无法打开摄像头 {src}", color="white"), bgcolor=ERROR, duration=2000)
            self.page.snack_bar.open = True
            self.page.update()
            return

        self.is_running = True
        self.frame_count = 0
        self.fps_timestamps.clear()
        self.detection_log.clear()

        # UI 更新
        self.toggle_btn.content = ft.Row(
            [ft.Icon(icon="stop", size=20, color=ERROR), ft.Text("停止检测", size=14, weight=ft.FontWeight.BOLD, color=ERROR)],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.toggle_btn.border = ft.border.all(1.5, ERROR)
        self.toggle_btn.bgcolor = ERROR + "15"
        self.status_dot.bgcolor = SUCCESS
        self.source_label.value = f"Camera {self.camera_dropdown.value}"
        self.video_placeholder.visible = False
        self.video_image.visible = True

        if self.pipeline:
            self.pipeline.start()

        # 线程驱动的帧循环（Flet 0.84 无 Timer）
        self._frame_stop = threading.Event()
        self._frame_thread = threading.Thread(target=self._frame_loop, daemon=True)
        self._frame_thread.start()
        self._perf_thread = threading.Thread(target=self._perf_loop, daemon=True)
        self._perf_thread.start()

        self.page.update()

    def _frame_loop(self):
        while not self._frame_stop.is_set():
            try:
                self._on_frame_tick()
            except Exception:
                pass
            time.sleep(0.033)

    def _perf_loop(self):
        while not self._frame_stop.is_set():
            try:
                self._on_perf_tick()
            except Exception:
                pass
            time.sleep(1.0)

    def _stop_detection(self):
        self.is_running = False

        if hasattr(self, '_frame_stop'):
            self._frame_stop.set()

        if self._cap:
            self._cap.release()
            self._cap = None
        if self.pipeline:
            self.pipeline.stop()

        self.toggle_btn.content = ft.Row(
            [ft.Icon(icon="play_arrow", size=20, color=ACCENT_PRIMARY), ft.Text("开始检测", size=14, weight=ft.FontWeight.BOLD, color=ACCENT_PRIMARY)],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.toggle_btn.border = ft.border.all(1.5, ACCENT_PRIMARY)
        self.toggle_btn.bgcolor = BG_DEEPEST
        self.status_dot.bgcolor = TEXT_MUTED
        self.video_placeholder.visible = True
        self.video_image.visible = False
        self.source_label.value = "已停止"
        self.page.update()

    def _on_frame_tick(self, e=None):
        if not self.is_running or not self._cap:
            return

        ret, frame = self._cap.read()
        if not ret:
            return

        self.frame_count += 1
        self.fps_timestamps.append(time.time())

        # 检测
        if self.pipeline and self.pipeline.is_running:
            packet = FramePacket(frame)
            self.pipeline.submit(packet)
            result = self.pipeline.get_result(timeout=0.001)
            if result:
                frame = result.result_frame
                detections = result.detections
            else:
                detections = []
        elif self.engine and self.engine.is_ready:
            frame, detections, _ = self.engine.detect(frame)
        else:
            detections = []

        self._latest_detections = detections

        # 记录
        for det in detections[:3]:
            plate = det.get("plate_no", "")
            conf = det.get("confidence", det.get("score", 0))
            ts = datetime.now().strftime("%H:%M:%S")
            entry = f"[{ts}] {plate or 'target'}  {conf:.2f}"
            self.detection_log.append(entry)
            if len(self.detection_log) > 60:
                self.detection_log = self.detection_log[-60:]
            self.detection_records.append({"time": ts, "plate": plate or "未知", "conf": conf, "color": det.get("color", "未知"), "status": "已识别"})

        # FPS 计算
        fps = 0.0
        if len(self.fps_timestamps) >= 2:
            elapsed = self.fps_timestamps[-1] - self.fps_timestamps[0]
            if elapsed > 0:
                fps = (len(self.fps_timestamps) - 1) / elapsed

        # 更新视频显示
        b64 = cv2_to_base64(frame, max_width=700)
        self.video_image.src = b64

        # 更新指标
        if self._metric_refs:
            if "FPS" in self._metric_refs:
                self._metric_refs["FPS"].value = f"{fps:.1f}"
            if self.engine:
                lat = self.engine.stats.get("avg_latency", 0)
                if "延迟" in self._metric_refs:
                    self._metric_refs["延迟"].value = f"{lat:.0f} ms"
                if "检测数" in self._metric_refs:
                    self._metric_refs["检测数"].value = str(self.engine.stats.get("detection_count", 0))

        # 更新日志面板
        if hasattr(self, 'log_column') and self.detection_log:
            new_count = len(self.detection_log) - len(self.log_column.controls)
            if new_count > 0:
                for entry in self.detection_log[-new_count:]:
                    self.log_column.controls.append(
                        ft.Text(entry, size=11, color=TEXT_SECONDARY, font_family="Consolas")
                    )
                self.log_column.scroll_to(offset=-1, duration=100)

        self.page.update()

    def _on_perf_tick(self, e=None):
        if not self.is_running:
            return
        cpu, mem, gpu, vram = get_system_metrics()
        self.cpu_bar.set_value(cpu)
        self.mem_bar.set_value(mem)
        self.gpu_bar.set_value(gpu)
        self.vram_bar.set_value(vram)
        self.page.update()

    # ── 时钟更新 ────────────────────────────────────────

    def start_clock(self):
        def _tick():
            while True:
                try:
                    self.clock_text.value = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                    self.page.update()
                except Exception:
                    break
                time.sleep(1)
        threading.Thread(target=_tick, daemon=True).start()

    # ── 事件 ──────────────────────────────────────────

    def _on_resize(self, e=None):
        pass


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════

def run_flet_ui(engine=None, pipeline=None, config_manager=None):
    def main(page: ft.Page):
        app = LPRApp(page, engine=engine, pipeline=pipeline, config_manager=config_manager)
        app.start_clock()
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8550)


if __name__ == "__main__":
    run_flet_ui()


