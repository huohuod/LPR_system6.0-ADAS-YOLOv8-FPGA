"""
前车碰撞预警 (FCW) 模块 — 基于测距 + 跟踪的 TTC 计算
TTC (Time To Collision) = 距离 / 相对速度
TTC 低于阈值时报警 (前车过近 / 接近过快)
"""
import time
from collections import defaultdict, deque
from typing import Dict, Optional


class CollisionWarning:
    """前车碰撞预警: 维护每辆车的距离历史, 算相对速度 + TTC"""

    def __init__(self, ttc_threshold: float = 3.0, history_len: int = 6,
                 min_rel_speed: float = 0.5,
                 ttc_prompt: float = 5.0, ttc_warning: float = 2.0):
        self.ttc_threshold = ttc_threshold    # 保留兼容
        self.history_len = history_len
        self.min_rel_speed = min_rel_speed    # 最小接近速度 (m/s), 低于此不算接近
        self.ttc_prompt = ttc_prompt          # TTC < 此值 → Prompt Risk (低风险)
        self.ttc_warning = ttc_warning        # TTC < 此值 → Warning Risk (高风险)
        # track_id -> deque[(时间, 距离)]
        self._history = defaultdict(lambda: deque(maxlen=history_len))

    def update(self, tracks: Dict, t: Optional[float] = None) -> Dict:
        """输入 {track_id: {'box':..., 'distance': float|None}}
        返回 {track_id: {'ttc':..., 'rel_speed':..., 'warning': bool}}"""
        t = time.time() if t is None else t
        results = {}
        for tid, track in tracks.items():
            dist = track.get('distance')
            if dist is None:
                continue
            hist = self._history[tid]
            # 距离实际变化才记录 (测距降频时避免重复值)
            if not hist or abs(dist - hist[-1][1]) > 0.05:
                hist.append((t, dist))

            if len(hist) >= 2:
                t0, d0 = hist[0]
                t1, d1 = hist[-1]
                dt = t1 - t0
                if dt > 0.1:
                    rel_speed = (d0 - d1) / dt        # >0 表示接近
                    if rel_speed > self.min_rel_speed:
                        ttc = d1 / rel_speed
                        # 三级风险: Normal(绿) / Prompt(橙红) / Warning(红)
                        if ttc < self.ttc_warning:
                            risk = 'Warning Risk'
                        elif ttc < self.ttc_prompt:
                            risk = 'Prompt Risk'
                        else:
                            risk = 'Normal Risk'
                        results[tid] = {
                            'ttc': ttc,
                            'rel_speed': rel_speed,
                            'distance': d1,
                            'risk': risk,
                            'warning': risk != 'Normal Risk',
                        }

        # 清理消失的 track
        active = set(tracks.keys())
        for tid in list(self._history.keys()):
            if tid not in active:
                del self._history[tid]
        return results
