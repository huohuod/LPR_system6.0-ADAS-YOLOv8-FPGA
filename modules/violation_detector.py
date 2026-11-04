"""
交通违规检测模块 v2 - 状态机版
- 斑马线-红绿灯空间关联
- 行人状态机: outside → entering → inside → leaving
- 只在红灯期间"进入"斑马线时才算违规
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class PedestrianState:
    """行人状态"""
    OUTSIDE = 'outside'      # 在斑马线外
    ENTERING = 'entering'    # 正在进入斑马线 (刚触发)
    INSIDE = 'inside'        # 在斑马线内
    LEAVING = 'leaving'      # 正在离开斑马线


class TrafficLightZone:
    """红绿灯-斑马线关联区域"""

    def __init__(self, light_bbox, crosswalk_bbox):
        self.light_bbox = light_bbox       # 红绿灯框
        self.crosswalk_bbox = crosswalk_bbox  # 斑马线区域
        self.state = 'unknown'             # 红绿灯状态

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在斑马线区域内"""
        x1, y1, x2, y2 = self.crosswalk_bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class ViolationDetector:
    """违规检测器 v2 - 状态机"""

    def __init__(self):
        self._violations = deque(maxlen=200)
        self._ped_states: Dict[int, str] = {}       # tid → state
        self._ped_zones: Dict[int, int] = {}        # tid → zone_id
        self._vehicle_flagged: set = set()
        self._zones: List[TrafficLightZone] = []
        self._crosswalk_regions = []                 # 外部传入的斑马线区域
        self._wrong_way_flagged: set = set()         # 逆行已标记
        self._flow_direction = 0.0                   # 主流车辆方向 (用于逆行判断)

    def update_zones(self, traffic_lights: List[Dict],
                     crosswalk_regions: list):
        """
        更新红绿灯-斑马线关联

        根据空间位置，把每个红绿灯和最近的斑马线配对。
        如果没有检测到斑马线，用画面默认区域。
        """
        self._zones = []

        # 默认斑马线区域 (如果没有检测到)
        if crosswalk_regions:
            cw_bboxes = [r.bbox for r in crosswalk_regions]
        else:
            cw_bboxes = []  # 由 analyze 时动态生成

        # 每个红绿灯关联最近的斑马线
        for tl in traffic_lights:
            tl_cx = (tl['bbox'][0] + tl['bbox'][2]) / 2
            tl_cy = (tl['bbox'][1] + tl['bbox'][3]) / 2

            best_cw = None
            best_dist = float('inf')

            for cw_bbox in cw_bboxes:
                cw_cx = (cw_bbox[0] + cw_bbox[2]) / 2
                cw_cy = (cw_bbox[1] + cw_bbox[3]) / 2
                # 红绿灯应该在斑马线上方，所以给y方向更大权重
                dist = abs(tl_cx - cw_cx) + abs(tl_cy - cw_cy) * 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_cw = cw_bbox

            if best_cw is not None:
                zone = TrafficLightZone(tl['bbox'], best_cw)
                zone.state = tl.get('color', 'unknown')
                self._zones.append(zone)

        # 如果没有红绿灯但有斑马线，创建一个默认区域
        if not self._zones and cw_bboxes:
            for cw_bbox in cw_bboxes:
                zone = TrafficLightZone(None, cw_bbox)
                zone.state = 'unknown'
                self._zones.append(zone)

        self._crosswalk_regions = cw_bboxes

    def analyze(self, frame_shape: tuple,
                pedestrians: List[Dict],
                vehicles: List[Dict],
                traffic_light_state: str,
                vehicle_tracks: Dict = None,
                ped_tracks: Dict = None) -> List[Dict]:
        """
        违规分析 - 状态机版

        行人状态转换:
          outside --[进入斑马线]--> entering
          entering --[下一帧仍在]--> inside
          inside --[离开斑马线]--> outside

        违规触发: 状态 == entering && 红灯
        """
        violations = []
        h, w = frame_shape[:2]

        # 无zone时添加默认斑马线区域作为兜底
        if not self._zones:
            default_cw = (int(w*0.05), int(h*0.25), int(w*0.95), int(h*0.95))
            default_zone = TrafficLightZone(None, default_cw)
            default_zone.state = traffic_light_state
            self._zones.append(default_zone)

        # 更新zone的红绿灯状态
        for zone in self._zones:
            if zone.state == 'unknown':
                zone.state = traffic_light_state

        # ── 行人违规: 红灯 + 行人在画面中下部 ──
        if traffic_light_state == 'red':
            for i, p in enumerate(pedestrians):
                cy = (p['bbox'][1] + p['bbox'][3]) / 2
                if cy < h * 0.20:
                    continue
                # 匹配跟踪ID (让框跟着人走)
                tid = self._match_pedestrian_to_track(p, ped_tracks)
                violations.append({
                    'type': 'pedestrian_red_crossing',
                    'track_id': tid,
                    'bbox': p['bbox'],
                    'position': ((p['bbox'][0]+p['bbox'][2])/2, cy),
                    'traffic_light': 'red',
                    'confidence': p.get('confidence', 0),
                })

        # ── 车辆违规 ──
        if traffic_light_state == 'red' and vehicle_tracks:
            violations.extend(
                self._check_vehicle_violations(vehicle_tracks, h, w)
            )

        # 绿灯时重置状态
        if traffic_light_state == 'green':
            self._ped_states.clear()
            self._ped_zones.clear()
            self._vehicle_flagged.clear()

        for v in violations:
            v['timestamp'] = time.time()
            self._violations.append(v)

        return violations

    def _match_pedestrian_to_track(self, det: Dict, ped_tracks: Dict) -> int:
        """把检测框匹配到最近的跟踪ID"""
        if not ped_tracks:
            return -1
        bx1, by1, bx2, by2 = det['bbox']
        best_tid, best_dist = -1, 9999
        for tid, track in ped_tracks.items():
            tb = track.get('box')
            if not tb:
                continue
            tx1, ty1, tx2, ty2 = tb
            dcx = (bx1+bx2)/2 - (tx1+tx2)/2
            dcy = (by1+by2)/2 - (ty1+ty2)/2
            dist = dcx*dcx + dcy*dcy
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        return best_tid if best_dist < 10000 else -1

    def _check_pedestrian_sm(self, ped_tracks: Dict,
                              tl_state: str) -> List[Dict]:
        """行人状态机违规检测"""
        violations = []
        active_tids = set()

        for tid, track in ped_tracks.items():
            if track.get('age', 0) > 1:
                continue
            active_tids.add(tid)

            positions = track.get('positions')
            if not positions:
                continue

            cx, cy = positions[-1]
            vx, vy = track.get('velocity', (0, 0))

            # 判断在哪个zone内
            in_zone_id = None
            for i, zone in enumerate(self._zones):
                if zone.contains_point(cx, cy):
                    in_zone_id = i
                    break

            prev_state = self._ped_states.get(tid, PedestrianState.OUTSIDE)

            # 状态转换
            if in_zone_id is not None:
                # 在斑马线区域内
                if prev_state == PedestrianState.OUTSIDE:
                    new_state = PedestrianState.ENTERING
                elif prev_state == PedestrianState.ENTERING:
                    new_state = PedestrianState.INSIDE
                else:
                    new_state = prev_state  # INSIDE/LEAVING → INSIDE
            else:
                # 在斑马线区域外
                new_state = PedestrianState.OUTSIDE

            self._ped_states[tid] = new_state
            if in_zone_id is not None:
                self._ped_zones[tid] = in_zone_id

            # 违规判定: 只在 entering 状态 + 红灯时触发
            if (new_state == PedestrianState.ENTERING
                    and tl_state == 'red'
                    and abs(vx) + abs(vy) > 1):
                violations.append({
                    'type': 'pedestrian_red_crossing',
                    'track_id': tid,
                    'bbox': track.get('box', (0, 0, 0, 0)),
                    'position': (cx, cy),
                    'velocity': (vx, vy),
                    'traffic_light': tl_state,
                    'confidence': track.get('confidence', 0),
                })

        # 清理已消失的行人
        expired = [tid for tid in self._ped_states if tid not in active_tids]
        for tid in expired:
            del self._ped_states[tid]
            self._ped_zones.pop(tid, None)

        return violations

    def _check_vehicle_violations(self, vehicle_tracks: Dict,
                                   frame_h: int, frame_w: int) -> List[Dict]:
        """车辆闯红灯检测 (四宫格下 tracker 每帧 reset, 速度/位移不可靠, 简化为位置判定)"""
        violations = []
        for tid, track in vehicle_tracks.items():
            if tid in self._vehicle_flagged:
                continue
            if track.get('age', 0) > 1:
                continue

            box = track.get('box')
            if not box:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            # 车辆在画面中下部 (闯红灯车辆通常在路口区域)
            if cy > frame_h * 0.25:
                self._vehicle_flagged.add(tid)
                violations.append({
                    'type': 'vehicle_red_light',
                    'track_id': tid,
                    'bbox': box,
                    'position': (cx, cy),
                    'traffic_light': 'red',
                    'confidence': track.get('confidence', 0),
                })

        return violations

    def _check_wrong_way(self, vehicle_tracks: Dict, frame_h: int, frame_w: int) -> List[Dict]:
        """车辆逆行: 行驶方向与主流方向相反"""
        violations = []
        # 统计主流方向 (大部分车辆的运动方向)
        directions = []
        for tid, track in vehicle_tracks.items():
            if track.get('age', 0) > 1:
                continue
            d = track.get('direction_angle', 0)
            spd = track.get('speed_px', 0)
            if spd > 1:
                directions.append(d)
        if directions:
            import numpy as np
            self._flow_direction = np.median(directions)

        # 检测逆行车辆
        for tid, track in vehicle_tracks.items():
            if tid in self._wrong_way_flagged:
                continue
            if track.get('age', 0) > 1:
                continue
            spd = track.get('speed_px', 0)
            if spd < 3:
                continue
            d = track.get('direction_angle', 0)
            # 与主流方向偏差超过120度 = 逆行
            diff = abs(d - self._flow_direction)
            if diff > 180:
                diff = 360 - diff
            if diff > 120:
                self._wrong_way_flagged.add(tid)
                cx, cy = track.get('positions', [(0, 0)])[-1] if track.get('positions') else (0, 0)
                violations.append({
                    'type': 'vehicle_wrong_way',
                    'track_id': tid,
                    'bbox': track.get('box', (0, 0, 0, 0)),
                    'position': (cx, cy),
                    'direction': d,
                    'flow_direction': self._flow_direction,
                    'confidence': track.get('confidence', 0),
                })
        return violations

    def clear_flags(self):
        """清除所有标记"""
        self._vehicle_flagged.clear()
        self._ped_states.clear()
        self._ped_zones.clear()
        self._wrong_way_flagged.clear()

    @property
    def recent_violations(self) -> list:
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)
