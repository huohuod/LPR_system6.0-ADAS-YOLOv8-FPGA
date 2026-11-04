"""
违规抓拍保存模块
自动保存含违规行为的帧图片到本地目录
"""
import os
import cv2
import time
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class ViolationSaver:
    """违规抓拍保存器"""

    def __init__(self, output_dir: str = None, max_per_minute: int = 10):
        """
        Args:
            output_dir: 保存目录，默认为项目根目录/output/violations/
            max_per_minute: 每分钟最大保存数，防止刷屏
        """
        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base, 'output', 'violations')

        self.output_dir = output_dir
        self.max_per_minute = max_per_minute
        self._save_timestamps: List[float] = []
        self._total_saved = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, frame: np.ndarray, violations: List[Dict],
             extra_info: Optional[Dict] = None) -> List[str]:
        """
        保存违规帧

        Args:
            frame: 原始帧 (BGR)
            violations: 违规事件列表
            extra_info: 额外信息 (车牌号等)

        Returns:
            保存的文件路径列表
        """
        if not violations:
            return []

        # 频率限制
        now = time.time()
        self._save_timestamps = [t for t in self._save_timestamps if now - t < 60]
        if len(self._save_timestamps) >= self.max_per_minute:
            return []

        saved_paths = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        for i, v in enumerate(violations):
            vtype = v.get('type', 'unknown')
            filename = f"{ts}_{vtype}_{i}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            # 在帧上画违规标注
            annotated = frame.copy()
            bbox = v.get('bbox')
            if bbox:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, vtype, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 保存图片
            cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 保存元数据 (numpy 类型转 Python 原生类型)
            meta = {
                'type': str(vtype),
                'timestamp': datetime.now().isoformat(),
                'bbox': [int(c) for c in bbox] if bbox else None,
                'confidence': float(v.get('confidence', 0)),
                'traffic_light': str(v.get('traffic_light', '')),
                'plate_no': str(v.get('plate_no', '')),
            }
            if extra_info:
                meta.update(extra_info)

            meta_path = filepath.replace('.jpg', '.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            saved_paths.append(filepath)
            self._save_timestamps.append(now)
            self._total_saved += 1

        return saved_paths

    @property
    def total_saved(self) -> int:
        return self._total_saved
