"""
斑马线检测模块
基于边缘检测 + 梯度一致性分析

原理:
  斑马线 = 多根平行白色条纹，具有以下特征:
  1. 梯度方向一致 (所有条纹的边缘方向相同)
  2. 等间隔分布
  3. 条纹宽度大于车道线

实现:
  1. 取画面下半部分作为ROI
  2. Canny边缘检测 + Sobel梯度
  3. 在滑动窗口内统计梯度方向直方图
  4. 峰值方向点数超过阈值 → 判定为斑马线
  5. 输出斑马线的包围矩形区域
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


class CrosswalkRegion:
    """检测到的斑马线区域"""

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class CrosswalkDetector:
    """斑马线检测器"""

    def __init__(self,
                 roi_top_ratio: float = 0.4,
                 window_width: int = 120,
                 window_height: int = 60,
                 stride: int = 40,
                 gradient_thresh: int = 30,
                 peak_ratio_thresh: float = 0.3):
        """
        Args:
            roi_top_ratio: ROI顶部占画面比例 (只检测下半部分)
            window_width: 滑动窗口宽度
            window_height: 滑动窗口高度
            stride: 滑动步长
            gradient_thresh: 梯度模值阈值
            peak_ratio_thresh: 峰值方向占比阈值
        """
        self.roi_top_ratio = roi_top_ratio
        self.win_w = window_width
        self.win_h = window_height
        self.stride = stride
        self.grad_thresh = gradient_thresh
        self.peak_ratio_thresh = peak_ratio_thresh
        self._last_regions: List[CrosswalkRegion] = []
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> List[CrosswalkRegion]:
        """
        检测斑马线

        Returns:
            检测到的斑马线区域列表
        """
        self._frame_count += 1

        # 每帧检测 (之前每3帧跳一次太慢)
        h, w = frame.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = frame[roi_top:h, :]

        # 预处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny边缘 (恢复原始阈值, 减少误检)
        edges = cv2.Canny(gray, 50, 150)

        # Sobel梯度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        direction = np.abs(np.arctan2(gy, gx))  # 0~pi
        direction_deg = np.degrees(direction)     # 0~180

        # 归一化到 0~90 度 (不区分正反)
        direction_deg = np.where(direction_deg > 90, 180 - direction_deg, direction_deg)

        # 滑动窗口检测
        hit_windows = []
        roi_h, roi_w = roi.shape[:2]

        for y in range(0, roi_h - self.win_h, self.stride):
            for x in range(0, roi_w - self.win_w, self.stride):
                # 窗口内的梯度
                win_mag = mag[y:y+self.win_h, x:x+self.win_w]
                win_dir = direction_deg[y:y+self.win_h, x:x+self.win_w]

                # 只看梯度足够大的点
                mask = win_mag > self.grad_thresh
                if np.sum(mask) < 20:
                    continue

                # 梯度方向直方图 (0-90度, 每5度一个bin)
                dirs = win_dir[mask]
                hist, _ = np.histogram(dirs, bins=18, range=(0, 90))

                # 峰值方向占比
                peak_idx = np.argmax(hist)
                peak_ratio = hist[peak_idx] / np.sum(hist)

                if peak_ratio > self.peak_ratio_thresh:
                    hit_windows.append((x, y, x + self.win_w, y + self.win_h, peak_ratio))

        # 合并重叠窗口
        regions = self._merge_windows(hit_windows, roi_top, w)

        # 如果没检测到，用保守方案 (颜色+形态学)
        if not regions:
            regions = self._fallback_detect(roi, roi_top, w)

        self._last_regions = regions
        return regions

    def _merge_windows(self, hits: List, roi_top: int, frame_w: int) -> List[CrosswalkRegion]:
        """合并重叠的检测窗口为斑马线区域"""
        if not hits:
            return []

        # 转为numpy数组
        boxes = np.array([(h[0], h[1], h[2], h[3]) for h in hits])
        scores = np.array([h[4] for h in hits])

        # 简单NMS
        keep = []
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
            area_j = (boxes[order[1:], 2]-boxes[order[1:], 0]) * (boxes[order[1:], 3]-boxes[order[1:], 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            inds = np.where(iou < 0.3)[0]
            order = order[inds + 1]

        regions = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # 转换回全图坐标
            bbox = (int(x1), int(y1 + roi_top), int(x2), int(y2 + roi_top))
            conf = float(scores[i])
            regions.append(CrosswalkRegion(bbox, conf))

        # 如果有多个区域，合并为一个大区域
        if len(regions) > 1:
            all_x1 = min(r.bbox[0] for r in regions)
            all_y1 = min(r.bbox[1] for r in regions)
            all_x2 = max(r.bbox[2] for r in regions)
            all_y2 = max(r.bbox[3] for r in regions)
            avg_conf = sum(r.confidence for r in regions) / len(regions)
            return [CrosswalkRegion((all_x1, all_y1, all_x2, all_y2), avg_conf)]

        return regions

    def _fallback_detect(self, roi: np.ndarray, roi_top: int,
                         frame_w: int) -> List[CrosswalkRegion]:
        """保守方案: 白色区域+形态学检测"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 白色条纹
        white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        # 形态学: 先开运算去噪，再闭运算连接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 斑马线特征: 宽度远大于高度，且面积足够大
            if bw > bh * 3 and area > 5000:
                bbox = (x, y + roi_top, x + bw, y + roi_top + bh)
                conf = min(area / 5000, 1.0)
                regions.append(CrosswalkRegion(bbox, conf))

        # 合并
        if len(regions) > 1:
            all_x1 = min(r.bbox[0] for r in regions)
            all_y1 = min(r.bbox[1] for r in regions)
            all_x2 = max(r.bbox[2] for r in regions)
            all_y2 = max(r.bbox[3] for r in regions)
            avg_conf = sum(r.confidence for r in regions) / len(regions)
            return [CrosswalkRegion((all_x1, all_y1, all_x2, all_y2), avg_conf)]

        return regions

    def draw(self, frame: np.ndarray, regions: List[CrosswalkRegion] = None) -> np.ndarray:
        """绘制检测到的斑马线区域"""
        if regions is None:
            regions = self._last_regions
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"Crosswalk {r.confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        return frame
