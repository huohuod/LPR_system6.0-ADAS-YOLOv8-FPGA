"""
红绿灯检测与颜色分类模块
使用YOLOv8检测交通灯，HSV空间分类红/黄/绿状态
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class TrafficLightClassifier:
    """交通灯颜色分类器"""

    # HSV 范围定义
    RANGES = {
        'red': [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([160, 70, 50]), np.array([180, 255, 255])),
        ],
        'yellow': [
            (np.array([15, 100, 100]), np.array([35, 255, 255])),
        ],
        'green': [
            (np.array([40, 150, 80]), np.array([80, 255, 255])),
        ],
    }

    LABELS = {
        'red': 'RED',
        'yellow': 'YELLOW',
        'green': 'GREEN',
    }

    @classmethod
    def classify(cls, img: np.ndarray) -> Tuple[str, float]:
        """分类交通灯颜色，返回 (颜色名, 置信度)"""
        if img is None or img.size == 0:
            return 'unknown', 0.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w

        scores = {}
        for color, ranges in cls.RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            scores[color] = np.sum(mask > 0) / total_pixels

        if max(scores.values()) < 0.05:
            return 'unknown', 0.0

        best_color = max(scores, key=scores.get)
        confidence = min(scores[best_color] * 5, 1.0)
        return best_color, confidence


class TrafficLightTracker:
    """交通灯跟踪器 - 稳定状态输出"""

    def __init__(self, stability_frames: int = 5):
        self.tracks = {}  # id -> {box, color_history, stable_color, age}
        self.next_id = 0
        self.stability_frames = stability_frames
        self._current_state = 'unknown'  # 当前路口整体灯态

    def reset(self):
        """清空跟踪状态 (四路分屏每路独立检测用)"""
        self.tracks = {}
        self.next_id = 0
        self._current_state = 'unknown'

    def update(self, detections: List[Dict]) -> List[Dict]:
        """更新跟踪，返回稳定后的检测结果"""
        if not detections:
            # 老化后清空
            expired = []
            for tid, t in self.tracks.items():
                t['age'] += 1
                if t['age'] > 8:
                    expired.append(tid)
            for tid in expired:
                del self.tracks[tid]
            return []

        # 限制最大跟踪数
        if len(self.tracks) > 30:
            old = sorted(self.tracks, key=lambda tid: self.tracks[tid]['age'], reverse=True)
            for tid in old[20:]:
                del self.tracks[tid]

        # 简单IoU匹配
        matched_tracks = set()
        matched_dets = set()

        for tid, track in list(self.tracks.items()):
            best_iou = 0.3
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self._iou(track['box'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                det = detections[best_idx]
                track['box'] = det['bbox']
                track['age'] = 0
                track['color_history'].append(det['color'])
                if len(track['color_history']) > self.stability_frames:
                    track['color_history'].pop(0)

                # 投票决定稳定颜色
                from collections import Counter
                votes = Counter(track['color_history'])
                track['stable_color'] = votes.most_common(1)[0][0]
                track['confidence'] = det['confidence']

                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        # 新检测创建新跟踪
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_id] = {
                    'box': det['bbox'],
                    'color_history': [det['color']],
                    'stable_color': det['color'],
                    'confidence': det['confidence'],
                    'age': 0,
                }
                self.next_id += 1

        # 更新全局灯态
        colors = [t['stable_color'] for t in self.tracks.values() if t['age'] == 0]
        if colors:
            from collections import Counter
            self._current_state = Counter(colors).most_common(1)[0][0]

        return [{
            'bbox': t['box'],
            'color': t['stable_color'],
            'confidence': t['confidence'],
            'track_id': tid,
        } for tid, t in self.tracks.items() if t['age'] <= 1]

    @property
    def current_state(self) -> str:
        """当前路口灯态: red/yellow/green/unknown"""
        return self._current_state

    def _iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
