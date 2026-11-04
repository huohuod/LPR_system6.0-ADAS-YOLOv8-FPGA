"""
车道线检测 v3.1 — 直接霍夫 + 二次多项式拟合 (免标定, 支持弯道)
- 颜色(白/黄) + Canny → 霍夫找斜线 → 斜率过滤斑马线(横线)
- 分左右、锁本车车道 → 二次多项式拟合 x = A*y^2 + B*y + C (能顺着弯道)
- 时序平滑 + 异常帧丢弃
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LaneLine:
    """一条车道线, 二次多项式 x = A*y^2 + B*y + C (前视画面坐标)"""
    fit: Optional[np.ndarray] = None   # [A, B, C]
    line_type: str = 'unknown'


@dataclass
class LaneResult:
    detected: bool = False
    left: Optional[LaneLine] = None
    right: Optional[LaneLine] = None
    lane_center_x: float = 0.0
    vehicle_offset: float = 0.0      # 偏右为负, 偏左为正
    curvature_radius: float = 0.0
    ldws: str = "NO LANE"
    lkas: str = "To Be Determined"   # 车道保持辅助提示
    frame: Optional[np.ndarray] = None


class LaneDetector:
    def __init__(self,
                 hough_thresh: int = 30,
                 min_line_len: int = 50,
                 max_line_gap: int = 80,
                 slope_min: float = 0.4,
                 slope_max: float = 5.0,
                 max_curve: float = 0.0005,   # 二次项系数上限, 过滤离谱弯度
                 smooth_alpha: float = 0.5,
                 smooth_max_jump: int = 150,
                 ldws_threshold: float = 80):
        self.hough_thresh = hough_thresh
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.max_curve = max_curve
        self.smooth_alpha = smooth_alpha
        self.smooth_max_jump = smooth_max_jump
        self.ldws_threshold = ldws_threshold
        self._slots = {}   # lane_slot → {'left': fit, 'right': fit}  (按象限隔离平滑状态)

    # ── 主入口 ──────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray, lane_slot: int = 0) -> LaneResult:
        h, w = frame_bgr.shape[:2]
        result = LaneResult()

        # 去摩尔纹(二次采集画面): 轻度高斯只作用于检测的二值图, 不影响最终绘制
        det_input = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        binary = self._binary(det_input)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, self.hough_thresh,
                                minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)

        y_top = int(h * 0.35)
        y_bottom = h - 1
        left = right = None

        if lines is not None:
            left_pts, right_pts = self._separate(lines, w, h)
            left = self._fit(left_pts)
            right = self._fit(right_pts)

        left = self._smooth(left, 'left', y_bottom, lane_slot)
        right = self._smooth(right, 'right', y_bottom, lane_slot)

        if left is not None:
            left.line_type = self._classify(binary, left, y_top, y_bottom)
        if right is not None:
            right.line_type = self._classify(binary, right, y_top, y_bottom)

        result.left = left
        result.right = right
        result.detected = left is not None or right is not None

        if left is not None and right is not None:
            left_bx = self._x_at(left.fit, y_bottom)
            right_bx = self._x_at(right.fit, y_bottom)
            result.lane_center_x = (left_bx + right_bx) / 2
            result.vehicle_offset = result.lane_center_x - w / 2
            result.curvature_radius = self._curvature(left.fit, right.fit, y_bottom)
            result.ldws = self._ldws(result.vehicle_offset)
            result.lkas = self._lkas(left.fit, right.fit, y_top, y_bottom)

        result.frame = self._draw(frame_bgr, result, w, h, y_top, y_bottom)
        return result

    @staticmethod
    def _x_at(fit, y):
        return fit[0] * y ** 2 + fit[1] * y + fit[2]

    # ── 二值化 ──────────────────────────────────────────
    def _binary(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 60]))
        yellow = cv2.inRange(hls, np.array([15, 100, 100]), np.array([35, 255, 255]))
        color = cv2.bitwise_or(white, yellow)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.bitwise_or(color, edges)

    # ── 分离左右 + 本车车道锁定 ─────────────────────────
    def _separate(self, lines, w, h):
        center = w / 2
        left_segs, right_segs = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            if not (self.slope_min < abs(slope) < self.slope_max):
                continue
            if y2 != y1:
                dxdy = (x2 - x1) / (y2 - y1)
                bottom_x = x1 + dxdy * (h - 1 - y1)
            else:
                bottom_x = float(x1)
            if slope < 0:
                left_segs.append((x1, y1, x2, y2, bottom_x))
            else:
                right_segs.append((x1, y1, x2, y2, bottom_x))

        left_segs = self._select_ego(left_segs, center, is_left=True)
        right_segs = self._select_ego(right_segs, center, is_left=False)

        left_pts = [p for s in left_segs for p in [(s[0], s[1]), (s[2], s[3])]]
        right_pts = [p for s in right_segs for p in [(s[0], s[1]), (s[2], s[3])]]
        return left_pts, right_pts

    def _select_ego(self, segs, center, is_left):
        """选离画面中心最近的一段(本车车道边界), 过滤相邻车道/路沿"""
        if not segs:
            return segs
        scored = []
        for s in segs:
            bx = s[4]
            dist = (center - bx) if is_left else (bx - center)
            if dist > 0:
                scored.append((dist, s))
        if not scored:
            return segs
        scored.sort(key=lambda x: x[0])
        keep = scored[:8]   # 取离中心最近的 8 条线段
        return [s for _, s in keep]

    def _fit(self, pts):
        if len(pts) < 5:
            return None
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        fit = np.polyfit(ys, xs, 2)   # x = A*y^2 + B*y + C
        if abs(fit[0]) > self.max_curve:
            return None                # 弯度离谱, 判为噪声
        return LaneLine(fit=fit)

    def _smooth(self, line, side, y_bottom, slot=0):
        st = self._slots.setdefault(slot, {'left': None, 'right': None})
        prev = st[side]
        if line is None:
            return LaneLine(fit=np.array(prev, dtype=np.float32)) if prev is not None else None
        fit = line.fit
        if prev is not None:
            new_x = self._x_at(fit, y_bottom)
            prev_x = self._x_at(np.array(prev, dtype=np.float32), y_bottom)
            if abs(new_x - prev_x) > self.smooth_max_jump:
                fit = np.array(prev, dtype=np.float32)
            else:
                fit = self.smooth_alpha * fit + (1 - self.smooth_alpha) * np.array(prev, dtype=np.float32)
        st[side] = fit
        return LaneLine(fit=fit)

    def _classify(self, binary, line, y_top, y_bottom):
        n = 30
        hits = 0
        h, w = binary.shape
        for i in range(n):
            y = int(y_bottom - i * (y_bottom - y_top) / (n - 1))
            x = int(self._x_at(line.fit, y))
            if 0 <= y < h and 0 <= x < w:
                win = binary[max(0, y - 3):y + 4, max(0, x - 10):x + 11]
                if win.any():
                    hits += 1
        ratio = hits / n
        return 'solid' if ratio > 0.55 else ('dashed' if ratio < 0.35 else 'unknown')

    def _curvature(self, left_fit, right_fit, y_bottom):
        # 用二次项系数估曲率半径 (像素量级)
        A = (abs(left_fit[0]) + abs(right_fit[0])) / 2
        if A < 1e-7:
            return float('inf')
        return 1.0 / (2 * A)

    def _ldws(self, offset):
        if abs(offset) < self.ldws_threshold:
            return "Good Lane Keeping"
        return "Please Keep Left" if offset < 0 else "Please Keep Right"

    def _lkas(self, left_fit, right_fit, y_top, y_bottom, straight_px=30, gentle_px=100):
        """LKAS: 用远近车道中心偏移判断前方弯道方向与缓急"""
        near_c = (self._x_at(left_fit, y_bottom) + self._x_at(right_fit, y_bottom)) / 2
        far_c = (self._x_at(left_fit, y_top) + self._x_at(right_fit, y_top)) / 2
        delta = far_c - near_c          # >0 远在右(右弯), <0 远在左(左弯)
        if abs(delta) < straight_px:
            return "Keep Straight Ahead"
        direction = "Left" if delta < 0 else "Right"
        return f"Gentle {direction} Curve Ahead" if abs(delta) < gentle_px \
            else f"Hard {direction} Curve Ahead"

    # ── 绘制 ────────────────────────────────────────────
    def _draw(self, frame, result, w, h, y_top, y_bottom):
        out = frame.copy()
        ploty = np.linspace(y_top, y_bottom, 30).astype(int)

        if result.left is not None and result.right is not None:
            left_x = self._x_at(result.left.fit, ploty).astype(int)
            right_x = self._x_at(result.right.fit, ploty).astype(int)
            overlay = out.copy()
            pts = np.concatenate([np.column_stack([left_x, ploty]),
                                  np.column_stack([right_x, ploty])[::-1]]).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

            for line, tag in [(result.left, 'L'), (result.right, 'R')]:
                x = self._x_at(line.fit, ploty).astype(int)
                color = (0, 255, 0) if line.line_type == 'solid' else (0, 200, 255)
                cv2.polylines(out, [np.column_stack([x, ploty]).astype(np.int32)], False, color, 3)
                cv2.putText(out, f"{tag}:{line.line_type}", (int(x[-1]), y_bottom - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cx = int(result.lane_center_x)
            cv2.line(out, (cx, y_bottom), (cx, y_top), (255, 0, 255), 1)
            vx = w // 2
            cv2.line(out, (vx, y_bottom), (vx, y_bottom - 40), (255, 255, 0), 2)
            cv2.putText(out, f"offset:{result.vehicle_offset:+.0f}px", (vx + 6, y_bottom - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        color = (0, 255, 0) if result.ldws == "Good Lane Keeping" else (0, 0, 255)
        cv2.putText(out, result.ldws, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(out, result.lkas, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out
