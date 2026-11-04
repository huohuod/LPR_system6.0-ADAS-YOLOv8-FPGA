"""
轨迹跟踪与运动分析模块
- 保留前N帧位置，计算位移向量
- 判断行驶方向、速度估算
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class TrajectoryTracker:
    """通用轨迹跟踪器 - 用于车辆和行人的运动分析"""

    def __init__(self, history_len: int = 8, max_age: int = 20, iou_thresh: float = 0.25):
        self.history_len = history_len
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks = {}  # id -> {box, positions, velocities, age, class_id}
        self.next_id = 0

    def reset(self):
        """清空跟踪状态 (四路分屏每路独立检测用)"""
        self.tracks = {}
        self.next_id = 0

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        更新跟踪，返回 {track_id: track_info}
        每个 track_info 包含:
            box: 当前框
            positions: deque of (cx, cy) 最近N帧
            velocity: (vx, vy) 最近两帧位移
            displacement: (dx, dy) 最近N帧总位移
            direction_angle: 运动方向角度 (度, 0=右, 90=下)
            speed_px: 像素速度 (帧)
            age: 未匹配帧数
            class_id: 类别ID
        """
        det_boxes = [d['bbox'] for d in detections]
        matched_tracks = set()
        matched_dets = set()

        # 匹配
        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_thresh
            best_idx = -1
            for i, box in enumerate(det_boxes):
                if i in matched_dets:
                    continue
                iou = self._iou(track['box'], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                det = detections[best_idx]
                new_box = det['bbox']
                new_cx = (new_box[0] + new_box[2]) / 2
                new_cy = (new_box[1] + new_box[3]) / 2

                track['box'] = new_box
                track['age'] = 0
                track['class_id'] = det.get('class_id', -1)
                track['confidence'] = det.get('confidence', 0)

                old_cx, old_cy = track['positions'][-1] if track['positions'] else (new_cx, new_cy)
                track['positions'].append((new_cx, new_cy))
                if len(track['positions']) > self.history_len:
                    track['positions'].popleft()

                # 速度 = 最近两帧位移
                track['velocity'] = (new_cx - old_cx, new_cy - old_cy)

                # 总位移 = 第一帧到当前帧
                if len(track['positions']) >= 2:
                    first = track['positions'][0]
                    track['displacement'] = (new_cx - first[0], new_cy - first[1])
                else:
                    track['displacement'] = (0, 0)

                # 方向角度
                vx, vy = track['velocity']
                track['direction_angle'] = np.degrees(np.arctan2(vy, vx)) if (vx or vy) else 0

                # 像素速度
                track['speed_px'] = np.sqrt(vx**2 + vy**2)

                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        # 新目标
        for i, det in enumerate(detections):
            if i not in matched_dets:
                box = det['bbox']
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                self.tracks[self.next_id] = {
                    'box': box,
                    'positions': deque([(cx, cy)], maxlen=self.history_len),
                    'velocity': (0, 0),
                    'displacement': (0, 0),
                    'direction_angle': 0,
                    'speed_px': 0,
                    'age': 0,
                    'class_id': det.get('class_id', -1),
                    'confidence': det.get('confidence', 0),
                }
                self.next_id += 1

        # 老化
        expired = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in expired:
            del self.tracks[tid]
        for t in self.tracks.values():
            t['age'] += 1

        return {tid: t for tid, t in self.tracks.items() if t['age'] <= 2}

    def get_moving_toward(self, target_point: Tuple[float, float],
                          min_speed: float = 3.0) -> List[Tuple[int, Dict]]:
        """获取正在朝目标点移动的跟踪目标"""
        results = []
        for tid, t in self.tracks.items():
            if t['age'] > 1 or t['speed_px'] < min_speed:
                continue
            cx, cy = t['positions'][-1] if t['positions'] else (0, 0)
            vx, vy = t['velocity']
            # 目标方向向量
            dx = target_point[0] - cx
            dy = target_point[1] - cy
            # 速度方向与目标方向的点积 > 0 表示朝目标移动
            dot = vx * dx + vy * dy
            if dot > 0:
                results.append((tid, t))
        return results

    def _iou(self, b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0


class TrackAdapter:
    """基于外部跟踪器(已赋值 track_id)的轨迹适配层
    信任传入的 track_id (来自 ultralytics BoT-SORT), 只负责重建下游所需的运动量。
    输出格式与 TrajectoryTracker 完全一致:
        box / positions / velocity / displacement / direction_angle / speed_px / age / class_id / confidence
    额外附加的键 (如 distance) 会被原样保留在 track 中。
    """

    def __init__(self, history_len: int = 8, max_age: int = 20):
        self.history_len = history_len
        self.max_age = max_age
        self.tracks = {}  # track_id -> {box, positions, velocity, ...}

    def reset(self):
        """清空跟踪状态 (四路分屏每路独立检测用)"""
        self.tracks = {}

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """detections: List[{bbox, track_id, class_id?, confidence?}]
        返回 {track_id: track_info} (与 TrajectoryTracker 同格式)
        """
        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                continue  # 未跟踪到的目标跳过 (BoT-SORT 首帧一般都会给ID)
            box = det['bbox']
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            if tid not in self.tracks:
                self.tracks[tid] = {
                    'box': box,
                    'positions': deque([(cx, cy)], maxlen=self.history_len),
                    'velocity': (0, 0),
                    'displacement': (0, 0),
                    'direction_angle': 0,
                    'speed_px': 0,
                    'age': 0,
                    'class_id': det.get('class_id', -1),
                    'confidence': det.get('confidence', 0),
                }
            else:
                t = self.tracks[tid]
                old_cx, old_cy = t['positions'][-1]
                t['box'] = box
                t['age'] = 0
                t['class_id'] = det.get('class_id', t.get('class_id', -1))
                t['confidence'] = det.get('confidence', t.get('confidence', 0))
                t['positions'].append((cx, cy))
                if len(t['positions']) > self.history_len:
                    t['positions'].popleft()

                t['velocity'] = (cx - old_cx, cy - old_cy)
                if len(t['positions']) >= 2:
                    fx, fy = t['positions'][0]
                    t['displacement'] = (cx - fx, cy - fy)
                vx, vy = t['velocity']
                t['direction_angle'] = np.degrees(np.arctan2(vy, vx)) if (vx or vy) else 0
                t['speed_px'] = np.sqrt(vx**2 + vy**2)

        # 老化 (与 TrajectoryTracker 语义一致)
        expired = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in expired:
            del self.tracks[tid]
        for t in self.tracks.values():
            t['age'] += 1

        return {tid: t for tid, t in self.tracks.items() if t['age'] <= 2}
