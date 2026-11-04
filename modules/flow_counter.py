"""
车流量统计模块 - 虚线圈法
在画面中设置虚拟检测线，通过跟踪目标中心点与检测线的交叉关系统计车流量
"""
import numpy as np
from typing import Dict, Tuple, List, Set
from collections import defaultdict


class VirtualLine:
    """虚拟检测线"""

    def __init__(self, y: int, name: str = "line"):
        """
        Args:
            y: 检测线的y坐标 (水平线)
            name: 线的名称
        """
        self.y = y
        self.name = name
        self.in_count = 0   # 从上往下穿越
        self.out_count = 0  # 从下往上穿越
        self._crossed_in: Set[int] = set()   # 已计数的track_id
        self._crossed_out: Set[int] = set()

    def check_crossing(self, track_id: int, prev_y: float, curr_y: float):
        """
        检测是否穿越检测线

        Args:
            track_id: 跟踪目标ID
            prev_y: 前一帧中心点y坐标
            curr_y: 当前帧中心点y坐标
        """
        if prev_y <= self.y and curr_y > self.y:
            # 从上往下穿越 (进入)
            if track_id not in self._crossed_in:
                self.in_count += 1
                self._crossed_in.add(track_id)
        elif prev_y >= self.y and curr_y < self.y:
            # 从下往上穿越 (离开)
            if track_id not in self._crossed_out:
                self.out_count += 1
                self._crossed_out.add(track_id)

    def reset(self):
        self.in_count = 0
        self.out_count = 0
        self._crossed_in.clear()
        self._crossed_out.clear()

    @property
    def total(self) -> int:
        return self.in_count + self.out_count


class FlowCounter:
    """车流量统计器"""

    def __init__(self, frame_height: int = 540):
        self.frame_height = frame_height
        self.lines: List[VirtualLine] = []
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        self._prev_missed: Dict[int, int] = {}

        # 默认在画面中间偏下设一条检测线
        self._auto_setup()

    def _auto_setup(self):
        """自动设置默认检测线"""
        line_y = int(self.frame_height * 0.6)
        self.lines.append(VirtualLine(line_y, "main"))

    def set_lines(self, line_ys: List[int]):
        """手动设置检测线位置"""
        self.lines = [VirtualLine(y, f"line_{i}") for i, y in enumerate(line_ys)]

    def update(self, tracked_objects: Dict[int, Dict]) -> Dict[str, Dict]:
        """
        更新流量统计

        Args:
            tracked_objects: {track_id: {box, positions, ...}}

        Returns:
            {line_name: {in_count, out_count, total}}
        """
        for tid, track in tracked_objects.items():
            # 用 box 中心点 (始终可用), 不用 positions (需预热)
            box = track.get('box')
            if not box:
                continue
            curr_cx = (box[0] + box[2]) / 2.0
            curr_cy = (box[1] + box[3]) / 2.0

            if tid in self._prev_positions:
                prev_cx, prev_cy = self._prev_positions[tid]
                for line in self.lines:
                    line.check_crossing(tid, prev_cy, curr_cy)

            # 首帧就记录, 第二帧即可检测跨越
            self._prev_positions[tid] = (curr_cx, curr_cy)

        # 清理已消失的目标 (超过 5 帧未出现才清除, 容忍短暂丢失)
        active_ids = set(tracked_objects.keys())
        for tid in list(self._prev_positions.keys()):
            if tid not in active_ids:
                self._prev_missed[tid] = self._prev_missed.get(tid, 0) + 1
                if self._prev_missed[tid] > 15:
                    del self._prev_positions[tid]
                    del self._prev_missed[tid]
            else:
                self._prev_missed[tid] = 0

        return {
            line.name: {
                'in_count': line.in_count,
                'out_count': line.out_count,
                'total': line.total,
                'y': line.y,
            }
            for line in self.lines
        }

    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制检测线"""
        w = frame.shape[1]
        for line in self.lines:
            cv2.line(frame, (0, line.y), (w, line.y), (0, 255, 255), 1)
        return frame

    def reset(self):
        for line in self.lines:
            line.reset()
        self._prev_positions.clear()


# 需要 import cv2
import cv2
