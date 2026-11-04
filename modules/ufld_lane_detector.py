"""
UFLD 车道线检测 — ONNX 深度学习模型 (无需标定)
- 模型: Ultra Fast Lane Detection (ResNet-18 主干, TuSimple 训练)
- 输入: 288x800 RGB, ImageNet 归一化
- 输出: [1, 101, 56, 4] = (100网格+1无车道类, 56行锚点, 4条车道)
- 解码: 每行锚点 softmax + 期望 → 车道线 x 坐标
"""
import os
import numpy as np
import cv2
import time
from collections import deque
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from .lane_detector import LaneLine, LaneResult

INPUT_W, INPUT_H = 800, 288
GRIDING_NUM = 200          # 水平网格数 (CULane)
CLS_NUM_PER_LANE = 18      # 行锚点数 (CULane)
NUM_LANES = 4              # 最多 4 条车道
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CULane 行锚点 (18 个, y 从 121 到 287)
ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

# HUD 直行图标 (正常行驶时显示在提示框上方, 代码绘制)
HUD_ICON_HEIGHT = 80

_HUD_FONT_CACHE = {}


def _hud_font(size, mono=True):
    """加载 HUD 字体 (等宽 Consolas / 中文雅黑), 带缓存"""
    key = (size, mono)
    if key in _HUD_FONT_CACHE:
        return _HUD_FONT_CACHE[key]
    if mono:
        candidates = [r"C:\Windows\Fonts\consola.ttf",
                      r"C:\Windows\Fonts\cour.ttf",
                      r"C:\Windows\Fonts\lucon.ttf"]
    else:
        candidates = [r"C:\Windows\Fonts\msyh.ttc",
                      r"C:\Windows\Fonts\simhei.ttf",
                      r"C:\Windows\Fonts\simsun.ttc"]
    font = None
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _HUD_FONT_CACHE[key] = font
    return font


def _font_metrics(font):
    """返回 (ascent, descent), 兼容默认字体"""
    try:
        return font.getmetrics()
    except Exception:
        return 0, 0


class UFLDLaneDetector:
    def __init__(self, model_path: str,
                 ldws_threshold: float = 80,
                 straight_px: float = 30,
                 gentle_px: float = 100,
                 y_min_ratio: float = 0.35,     # 丢弃该比例以上的点(上部路标/树/灯)
                 turn_px: float = 400,          # 转弯(真正急弯)阈值
                 smooth_alpha: float = 0.5,
                 smooth_max_jump: int = 150):
        self.ldws_threshold = ldws_threshold
        self.straight_px = straight_px
        self.gentle_px = gentle_px
        self.y_min_ratio = y_min_ratio
        self.turn_px = turn_px
        self.conf_threshold = 0.35   # 置信度阈值, 低于此判定为停止线/路口误检 (降低减少漏检)
        self.smooth_alpha = smooth_alpha
        self.smooth_max_jump = smooth_max_jump
        self.miss_max = 8             # 连续漏检多少帧后才真正消失 (抗闪烁)
        self._slots = {}              # lane_slot → {'left_fit','right_fit','left_miss','right_miss'}
        self._stop_line_time = -5.0   # 上次检测到停止线的时间 (用于 5 秒滞回)
        self._load(model_path)
        self._hud_icon = self._make_icon()
        self._hud_icon_warn = self._make_warn_icon()

    def _load(self, path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] UFLD 车道检测模型加载成功: {path}")
        print(f"[INFO] 实际执行提供器: {self.session.get_providers()}")

    def _make_icon(self):
        """用 OpenCV 画「直行」图标 (蓝色圆 + 白色直行箭头), 返回带 alpha 的 BGRA"""
        S = 200                       # 高分辨率绘制, 缩小抗锯齿
        blue = (179, 102, 0)          # BGR 交通蓝
        white = (255, 255, 255)
        c = S // 2
        r = int(S * 0.48)             # 圆半径

        # 透明区也填蓝色 (避免缩小时边缘出现黑边)
        bgr = np.full((S, S, 3), blue, np.uint8)
        # 白色直行箭头: 三角箭头 + 矩形箭杆
        apex_y = int(S * 0.18)
        base_y = int(S * 0.46)
        half_w = int(S * 0.16)
        shaft_w = int(S * 0.07)
        shaft_bottom = int(S * 0.82)
        head = np.array([[c, apex_y], [c - half_w, base_y], [c + half_w, base_y]], np.int32)
        cv2.fillPoly(bgr, [head], white)
        cv2.rectangle(bgr, (c - shaft_w, base_y), (c + shaft_w, shaft_bottom), white, -1)

        # alpha: 圆内不透明, 圆外透明
        mask = np.zeros((S, S), np.uint8)
        cv2.circle(mask, (c, c), r, 255, -1)

        icon = np.dstack([bgr, mask])
        scale = HUD_ICON_HEIGHT / icon.shape[0]
        return cv2.resize(icon, (int(icon.shape[1] * scale), HUD_ICON_HEIGHT),
                          interpolation=cv2.INTER_AREA)

    def _make_warn_icon(self):
        """画「减速慢行」图标 (黄色警告三角 + 黑色感叹号), 返回带 alpha 的 BGRA"""
        S = 200
        yellow = (0, 215, 255)   # BGR 黄
        black = (0, 0, 0)
        c = S // 2

        # 透明区也填黄 (避免缩小时边缘黑边)
        bgr = np.full((S, S, 3), yellow, np.uint8)
        mask = np.zeros((S, S), np.uint8)

        # 警告三角 (尖朝上)
        tri = np.array([[c, int(S * 0.12)],
                        [int(S * 0.14), int(S * 0.84)],
                        [int(S * 0.86), int(S * 0.84)]], np.int32)
        cv2.fillPoly(mask, [tri], 255)
        cv2.fillPoly(bgr, [tri], yellow)
        cv2.polylines(bgr, [tri], True, black, int(S * 0.025))   # 黑色描边

        # 感叹号: 竖条 + 圆点
        bar_w = int(S * 0.07)
        cv2.rectangle(bgr, (c - bar_w, int(S * 0.32)), (c + bar_w, int(S * 0.58)), black, -1)
        cv2.circle(bgr, (c, int(S * 0.68)), int(S * 0.055), black, -1)

        icon = np.dstack([bgr, mask])
        scale = HUD_ICON_HEIGHT / icon.shape[0]
        return cv2.resize(icon, (int(icon.shape[1] * scale), HUD_ICON_HEIGHT),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def _enhance_contrast(frame):
        """CLAHE 对比度增强 (LAB 亮度通道), 强化道路黑白对比"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _detect_horizontal_line(frame_bgr):
        """检测停止线(水平白线), 返回 (x1,y1,x2,y2) 或 None
        参数: 长线(>=800px)+近水平(|斜率|<0.1)+画面下半部分"""
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 60]))
        lines = cv2.HoughLinesP(white, 1, np.pi / 180, 50,
                                minLineLength=800, maxLineGap=50)
        if lines is None:
            return None
        h = frame_bgr.shape[0]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            if abs((y2 - y1) / dx) < 0.1 and 0.5 * h < (y1 + y2) / 2 < 0.9 * h:
                return (int(x1), int(y1), int(x2), int(y2))
        return None

    def detect(self, frame_bgr: np.ndarray, clip_y=None, hide_lane=False, warn_level='green', crosswalk_detected=False, lane_slot=0) -> LaneResult:
        h, w = frame_bgr.shape[:2]
        result = LaneResult()

        # 对比度增强 (CLAHE) — 强化道路黑白对比
        frame_bgr = self._enhance_contrast(frame_bgr)

        # 预处理
        img = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW

        out = self.session.run(None, {self.input_name: img})[0]  # [1, 101, 56, 4]

        lanes = self._decode(out[0], w, h, y_min=self.y_min_ratio * h)
        left, right = self._select_ego(lanes, w)

        # 检测到的最高点 (遮挡时只画到检测到的位置, 顶部相连形成四边形)
        y_top = self._detect_top(left, right, h)

        # 先拟合原始检测; 急弯判定在平滑前做, 视为"不可靠"交给平滑的漏检保持 (避免单帧误触发闪烁)
        left_fit = self._fit(left)
        right_fit = self._fit(right)
        if left_fit is not None and right_fit is not None:
            near_c = (self._x_at(left_fit.fit, h - 1) + self._x_at(right_fit.fit, h - 1)) / 2
            far_c = (self._x_at(left_fit.fit, y_top) + self._x_at(right_fit.fit, y_top)) / 2
            if abs(far_c - near_c) > self.turn_px:
                left_fit = None
                right_fit = None

        result.left = self._smooth(left_fit, 'left', h, lane_slot)
        result.right = self._smooth(right_fit, 'right', h, lane_slot)
        result.detected = result.left is not None or result.right is not None

        if result.left is not None and result.right is not None:
            y_eval = h - 1
            lx = self._x_at(result.left.fit, y_eval)
            rx = self._x_at(result.right.fit, y_eval)
            result.lane_center_x = (lx + rx) / 2
            result.vehicle_offset = result.lane_center_x - w / 2
            result.curvature_radius = self._curvature(result.left.fit, result.right.fit, y_eval)
            result.ldws = self._ldws(result.vehicle_offset)
            result.lkas = self._lkas(result.left.fit, result.right.fit, h, y_top)

        # 检测停止线, 有则隐藏车道辅助线 (5 秒滞回: 检测到后 5 秒内持续隐藏)
        stop_line = None  # 临时禁用停止线检测 (测试车道线)
        now = time.monotonic()
        if stop_line is not None:
            self._stop_line_time = now
        if hide_lane or (now - self._stop_line_time) < 5.0:
            result.left = None
            result.right = None
            result.detected = False
            result.ldws = "NO LANE"
            result.lkas = "To Be Determined"

        result.frame = self._draw(frame_bgr, result, w, h, y_top, clip_y, warn_level, crosswalk_detected)

        # 标记检测到的停止线 (红线圈, 便于测试)
        if stop_line is not None:
            x1, y1, x2, y2 = stop_line
            cv2.line(result.frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(result.frame, "Stop Line", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

    # ── 解码 ────────────────────────────────────────────
    def _decode(self, out, w, h, y_min=0.0):
        """out: [101, 56, 4] → 4 条车道, 每条 [(x,y) 原图坐标]; y<y_min 的点丢弃(上部路标/树/灯)"""
        lanes = []
        for l in range(NUM_LANES):
            pts = []
            for r in range(CLS_NUM_PER_LANE):
                logits = out[:, r, l].astype(np.float32)
                probs = self._softmax(logits)
                if int(np.argmax(probs)) == GRIDING_NUM:
                    continue  # 无车道
                conf = float(np.max(probs))
                if conf < self.conf_threshold:
                    continue  # 低置信度, 停止线/路口误检
                loc = float(np.sum(probs[:GRIDING_NUM] * np.arange(GRIDING_NUM)))
                x = loc / (GRIDING_NUM - 1) * (INPUT_W - 1) / INPUT_W * w
                y = ROW_ANCHOR[r] / INPUT_H * h
                if y < y_min:
                    continue
                pts.append((x, y))
            lanes.append(pts)
        return lanes

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _select_ego(self, lanes, w):
        """从 4 条车道里选本车车道的左右边界 (离中心最近的两条)"""
        center = w / 2
        scored = []
        for pts in lanes:
            if len(pts) < 3:
                continue
            avg_x = np.mean([p[0] for p in pts])
            scored.append((abs(avg_x - center), avg_x, pts))
        scored.sort(key=lambda x: x[0])

        left_pts = right_pts = None
        for _, avg_x, pts in scored:
            if avg_x < center and left_pts is None:
                left_pts = pts
            elif avg_x > center and right_pts is None:
                right_pts = pts
            if left_pts is not None and right_pts is not None:
                break
        return left_pts, right_pts

    @staticmethod
    def _fit(pts):
        if pts is None or len(pts) < 3:
            return None
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        fit = np.polyfit(ys, xs, 2)
        # 曲率限制: 二次项过大(发散)则退回线性拟合, 避免两线交叉
        if abs(fit[0]) > 0.0005:
            lin = np.polyfit(ys, xs, 1)
            fit = np.array([0.0, lin[0], lin[1]], dtype=np.float32)
        # 斜率检查: 太水平(停止线/路口误检)则丢弃, 不显示
        if abs(fit[1]) > 1.5:
            return None
        return LaneLine(fit=fit)

    def _smooth(self, line, side, h, slot=0):
        """EMA 时序平滑 (按槽位独立): 检测到→指数平均+跳变限制; 漏检→保持旧值, 连续 miss_max 帧才消失"""
        st = self._slots.setdefault(slot, {
            'left_fit': None, 'right_fit': None,
            'left_miss': 0, 'right_miss': 0,
        })
        fit_key = side + '_fit'
        miss_key = side + '_miss'
        prev = st[fit_key]

        if line is not None:
            new_fit = np.asarray(line.fit, dtype=np.float32)
            if prev is not None:
                # 用画面底部(近端) x 位置判断跳变, 过大突变视为误检/跨象限串扰
                if abs(self._x_at(new_fit, h) - self._x_at(prev, h)) > self.smooth_max_jump:
                    st[miss_key] += 1
                    if st[miss_key] <= self.miss_max:
                        return LaneLine(fit=prev)
                    st[fit_key] = None
                    return None
                new_fit = self.smooth_alpha * new_fit + (1 - self.smooth_alpha) * prev
            st[fit_key] = new_fit
            st[miss_key] = 0
            return LaneLine(fit=new_fit)

        # 漏检: 保持旧值, 连续 miss_max 帧后才真正消失
        st[miss_key] += 1
        if prev is not None and st[miss_key] <= self.miss_max:
            return LaneLine(fit=prev)
        st[fit_key] = None
        return None

    def reset_smoothing(self):
        """清空所有槽位的时序平滑状态 (手动重置用)"""
        self._slots.clear()

    @staticmethod
    def _x_at(fit, y):
        return fit[0] * y ** 2 + fit[1] * y + fit[2]

    @staticmethod
    def _gradient_blend(out, overlay, y_top, h, alpha_base=0.35):
        """垂直渐变混合: 顶部(远)透明, 底部(近)实"""
        ys = np.arange(h, dtype=np.float32)
        alpha = np.clip((ys - y_top) / max(1, h - 1 - y_top), 0, 1) * alpha_base
        alpha3 = alpha[:, None, None]
        return (out.astype(np.float32) * (1 - alpha3) + overlay.astype(np.float32) * alpha3).astype(np.uint8)

    @staticmethod
    def _detect_top(left_pts, right_pts, h):
        """检测到的最高点 (最小 y), 用于只画检测到的部分"""
        ys = []
        if left_pts:
            ys.extend(p[1] for p in left_pts)
        if right_pts:
            ys.extend(p[1] for p in right_pts)
        return int(min(ys)) if ys else int(h * 0.35)

    def _curvature(self, lf, rf, y_eval):
        A = (abs(lf[0]) + abs(rf[0])) / 2
        return float('inf') if A < 1e-7 else 1.0 / (2 * A)

    def _ldws(self, offset):
        if abs(offset) < self.ldws_threshold:
            return "Good Lane Keeping"
        return "Please Keep Left" if offset < 0 else "Please Keep Right"

    def _lkas(self, lf, rf, h, y_top):
        y_bottom = h - 1
        near = (self._x_at(lf, y_bottom) + self._x_at(rf, y_bottom)) / 2
        far = (self._x_at(lf, y_top) + self._x_at(rf, y_top)) / 2
        delta = far - near
        if abs(delta) < self.straight_px:
            return "Keep Straight Ahead"
        d = "Left" if delta < 0 else "Right"
        return f"Gentle {d} Curve Ahead" if abs(delta) < self.gentle_px else f"Hard {d} Curve Ahead"

    # ── 绘制 ────────────────────────────────────────────
    def _draw(self, frame, result, w, h, y_top, clip_y=None, warn_level='green', crosswalk_detected=False):
        out = frame.copy()
        if clip_y is not None:
            y_top = max(y_top, int(clip_y))   # 截断到前车位置, 不画前车之后(更远)的部分
        ploty = np.linspace(y_top, h - 1, 60).astype(int)

        has_l = result.left is not None
        has_r = result.right is not None
        if has_l or has_r:
            lane_color = {'green': (0, 255, 0),
                          'orange': (0, 165, 255),
                          'red': (0, 0, 255)}.get(warn_level, (0, 255, 0))
            overlay = out.copy()

            if has_l and has_r:
                # 左右都在: 填车道区域 + 两条边界线 (带交叉截断)
                lx = self._x_at(result.left.fit, ploty).astype(int)
                rx = self._x_at(result.right.fit, ploty).astype(int)
                valid = lx < rx
                cut = 0
                for i in range(len(ploty) - 1, -1, -1):
                    if not valid[i]:
                        cut = i + 1
                        break
                p2 = ploty[cut:]
                lx2 = lx[cut:]
                rx2 = rx[cut:]
                if len(p2) >= 2:
                    pts = np.concatenate([np.column_stack([lx2, p2]),
                                          np.column_stack([rx2, p2])[::-1]]).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], lane_color)
                    cv2.polylines(overlay, [np.column_stack([lx2, p2]).astype(np.int32)], False, lane_color, 3)
                    cv2.polylines(overlay, [np.column_stack([rx2, p2]).astype(np.int32)], False, lane_color, 3)
            elif has_l:
                # 只有左线: 单独画左边界
                lx = self._x_at(result.left.fit, ploty).astype(int)
                cv2.polylines(overlay, [np.column_stack([lx, ploty]).astype(np.int32)], False, lane_color, 3)
            else:
                # 只有右线: 单独画右边界
                rx = self._x_at(result.right.fit, ploty).astype(int)
                cv2.polylines(overlay, [np.column_stack([rx, ploty]).astype(np.int32)], False, lane_color, 3)

            # 渐变: 底部(近)实, 顶部(远)渐透明
            out = self._gradient_blend(out, overlay, y_top, h, alpha_base=0.35)

            # 中心线 + 车辆基准线 + offset 只在左右都在时画
            if has_l and has_r:
                cx = int(result.lane_center_x)
                cv2.line(out, (cx, h - 1), (cx, y_top), (255, 0, 255), 1)
                vx = w // 2
                cv2.line(out, (vx, h - 1), (vx, h - 41), (255, 255, 0), 2)
                cv2.putText(out, f"offset:{result.vehicle_offset:+.0f}px", (vx + 6, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        self._draw_hud(out, result, warn_level, crosswalk_detected)
        return out

    def _draw_hud(self, out, result, warn_level, crosswalk_detected):
        """左上角 HUD: 图标 + 文本框; 正常=直行图标+「正常行驶」, 减速慢行=警示图标+「减速慢行」 (PIL 渲染, 支持中文)"""
        warning = warn_level == 'red' or crosswalk_detected
        if warning:
            lines = [("减速慢行", 'cn', (255, 0, 0))]
        else:
            lines = [("正常行驶", 'cn', (0, 255, 0))]

        mono_size, cn_size = 9, 10
        line_h, pad = 13, 10
        x0 = 12
        gap = 6

        mono_font = _hud_font(mono_size, mono=True)
        cn_font = _hud_font(cn_size, mono=False)

        # 按文本实际宽度定框 (等宽字体保证两行对齐)
        widths = []
        for text, kind, _ in lines:
            font = mono_font if kind == 'mono' else cn_font
            bb = font.getbbox(text)
            widths.append(bb[2] - bb[0])
        box_w = max(widths) + pad * 2
        box_h = pad * 2 + line_h * len(lines)

        h, w = out.shape[:2]

        # 图标: 正常行驶显示直行图标, 减速慢行显示警示图标; 贴左上角, 提示框下移
        y0 = 12
        hud_icon = self._hud_icon if not warning else self._hud_icon_warn
        if hud_icon is not None:
            ih, iw = hud_icon.shape[:2]
            ix0, iy0 = x0, y0
            ix1, iy1 = min(x0 + iw, w), min(y0 + ih, h)
            if ix1 > ix0 and iy1 > iy0:
                ic = hud_icon[:iy1 - iy0, :ix1 - ix0]
                if ic.shape[2] == 4:
                    alpha = ic[..., 3].astype(np.float32) / 255.0
                    alpha3 = alpha[..., None]
                    region = out[iy0:iy1, ix0:ix1].astype(np.float32)
                    blended = ic[..., :3].astype(np.float32) * alpha3 + region * (1.0 - alpha3)
                    out[iy0:iy1, ix0:ix1] = blended.astype(np.uint8)
                else:
                    out[iy0:iy1, ix0:ix1] = ic
            y0 = min(y0 + ih + gap, h)

        # 文字框水平居中对齐图标
        box_x = int(x0 + HUD_ICON_HEIGHT / 2 - box_w / 2)
        box_x = max(0, min(box_x, w - box_w))
        x1 = min(box_x + box_w, w)
        y1 = min(y0 + box_h, h)
        crop = out[y0:y1, box_x:x1]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        # 红色细矩形边框 (PIL RGB: 红 = 255,0,0)
        draw.rectangle([0, 0, crop.shape[1] - 1, crop.shape[0] - 1],
                       outline=(255, 0, 0), width=1)

        for i, (text, kind, color) in enumerate(lines):
            font = mono_font if kind == 'mono' else cn_font
            asc, desc = _font_metrics(font)
            ty = pad + i * line_h + max(0, (line_h - (asc + desc)) // 2)
            draw.text((pad, ty), text, font=font, fill=color)

        out[y0:y1, box_x:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
