"""LPR系统核心引擎 v6.0 - 交通视觉分析版
模块化检测流水线:
  车辆 / 车牌 / 红绿灯 / 行人 / 违规 / 车流量 / OCR
"""
import time
import os
import sys
import threading
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.draw_text import draw_chinese_text, TextBatchRenderer
from src.utils.accel_wrapper import ImageProcessor
from src.modules.traffic_signal import TrafficLightClassifier


# ── 车牌跟踪器 ────────────────────────────────────────────

class PlateTracker:
    """车牌跟踪器 - IoU匹配 + 投票稳定"""

    def __init__(self, max_age: int = 15, iou_threshold: float = 0.35):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
        self.smooth_factor = 0.4

    def _iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def _smooth_box(self, old_box, new_box) -> tuple:
        if old_box is None:
            return new_box
        a = self.smooth_factor
        return tuple(int(old_box[i]*(1-a) + new_box[i]*a) for i in range(4))

    def update(self, detections: List[Dict]) -> List[Dict]:
        det_boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in detections]
        matched_tracks = set()
        matched_dets = set()

        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_idx = -1
            for i, db in enumerate(det_boxes):
                if i in matched_dets:
                    continue
                iou = self._iou(track['box'], db)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                det = detections[best_idx]
                new_box = (det['x1'], det['y1'], det['x2'], det['y2'])
                track['box'] = self._smooth_box(track['box'], new_box)
                track['age'] = 0
                track['hits'] += 1

                if det.get('plate_no'):
                    track['plate_history'].append(det['plate_no'])
                    if len(track['plate_history']) > 5:
                        track['plate_history'].pop(0)
                    votes = defaultdict(int)
                    for p in track['plate_history']:
                        if p:
                            votes[p] += 1
                    if votes:
                        track['stable_plate'] = max(votes, key=votes.get)
                    track['plate_no'] = track['stable_plate']

                track['confidence'] = det.get('confidence', track['confidence'])
                track['color'] = det.get('color', track.get('color', '未知'))
                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'box': (det['x1'], det['y1'], det['x2'], det['y2']),
                    'age': 0, 'hits': 1,
                    'confidence': det.get('confidence', 0),
                    'plate_no': det.get('plate_no', ''),
                    'color': det.get('color', '未知'),
                    'plate_history': [det['plate_no']] if det.get('plate_no') else [],
                    'stable_plate': det.get('plate_no', ''),
                }
                self.next_id += 1

        expired = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in expired:
            del self.tracks[tid]
        for t in self.tracks.values():
            t['age'] += 1

        results = []
        for tid, t in self.tracks.items():
            if t['hits'] >= 2 or t['age'] <= 1:
                x1, y1, x2, y2 = t['box']
                results.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': t['confidence'],
                    'plate_no': t['stable_plate'] or t['plate_no'],
                    'color': t.get('color', '未知'),
                    'track_id': tid,
                    'is_stable': t['hits'] >= 3,
                })
        return results


# ── 检测结果容器 ──────────────────────────────────────────

class DetectionResult:
    """单帧检测结果"""
    __slots__ = ('frame', 'vehicles', 'plates', 'traffic_lights',
                 'pedestrians', 'violations', 'flow_stats', 'latency')

    def __init__(self, frame):
        self.frame = frame
        self.vehicles: List[Dict] = []
        self.plates: List[Dict] = []
        self.traffic_lights: List[Dict] = []
        self.pedestrians: List[Dict] = []
        self.violations: List[Dict] = []
        self.flow_stats: Dict = {}
        self.latency: float = 0.0


# ── COCO 类别 ─────────────────────────────────────────────

COCO_VEHICLE_CLASSES = [1, 2, 3, 5, 7]
COCO_PERSON_CLASS = [0]
COCO_TRAFFIC_LIGHT_CLASS = [9]
COCO_ALL_INTEREST = [0, 1, 2, 3, 5, 7, 9]

COCO_NAMES = {
    0: '行人', 1: '自行车', 2: '轿车', 3: '摩托车',
    5: '客车', 7: '卡车', 9: '红绿灯',
}

# 红绿灯专用模型类别映射 (traffic_light.pt)
TL_CLASS_NAMES = {
    0: '红绿灯(红)', 1: '红绿灯(绿)', 2: '红绿灯(黄)',
    3: '行人', 4: '斑马线', 5: 'w', 6: 'light',
}
TL_COLOR_MAP = {
    0: 'red', 1: 'green', 2: 'yellow',
    6: 'unknown',  # light 类不确定颜色
}
TL_CROSSWALK_CLASS = 4  # 斑马线


# ── 主引擎 ────────────────────────────────────────────────

class LPREngine:

    def __init__(self, config, enable_tilt_correction: bool = False):
        self.config = config
        self.model_config = config.model
        self.enable_tilt_correction = enable_tilt_correction

        # 模型
        self.plate_model = None
        self.vehicle_model = None
        self.traffic_light_model = None  # 红绿灯专用检测模型
        self.pedestrian_model = None     # 行人专用检测模型
        self.crosswalk_model = None      # 斑马线专用检测模型
        self.ocr_engine = None
        self.depth_estimator = None      # Depth Anything V2 深度估计器
        self._last_depth_map = None      # 最近一帧深度图 (供标定读取)
        self._depth_frame_counter = 0    # 深度降频计数
        self.collision_warning = None    # 前车碰撞预警 (FCW)
        self.fcw_results = {}            # {track_id: {'ttc':..., 'warning':...}}
        self.collision_log = []          # 碰撞预警记录 [{seq, time, distance, confidence}], 最多50条
        self._collision_seq = 0          # 碰撞记录单调递增序号 (供 UI 增量同步)
        self._collision_recorded = {}    # lane_slot → bool 上升沿去重
        self.lane_detector = None        # UFLD 车道检测器 (上半 Q1/Q2 / 单路)
        self.lane_result = None          # 车道检测结果 (LaneResult)
        self.lane_detector_cv = None     # 传统 CV 车道检测器 (下半 Q3/Q4 二次采集)
        self._lane_slot = 0              # 车道检测槽位 (四路分屏 0-3, 单路 0)

        # 交通分析模块
        self.traffic_classifier = None   # HSV 或 CNN
        self.traffic_tracker = None
        self.violation_detector = None
        self.violation_saver = None
        self.flow_counter = None
        self.crosswalk_detector = None   # 斑马线检测
        self.vehicle_tracker = None      # 车辆轨迹
        self.ped_tracker = None          # 行人轨迹

        self.is_ready = False
        self._warmup_done = False

        self.stats = {
            'total_frames': 0,
            'avg_latency': 0.0,
            'latency_history': deque(maxlen=60),
            'fps_history': deque(maxlen=60),
            'detection_count': 0,
            'violation_count': 0,
        }

        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.tracker = PlateTracker(max_age=15, iou_threshold=0.20)

        # 批量文字渲染器
        self._text_renderer = TextBatchRenderer()

        # 图像颜色校正器 (OV5640 发绿修复)
        self._img_processor = ImageProcessor()
        self._sync_color_config()

        # 车流量检测线动态设置标记
        self._flow_line_set = False
        self._yolo_crosswalk_regions = []  # YOLO 斑马线检测区域 (不画框，供违规检测用)
        self._violators = {}  # track_id → {type, frame} 违规车辆/行人跟踪

        # 后台保存线程
        self._save_thread = None

        # 检测开关
        self.enable_vehicle = True
        self.enable_plate = False
        self.enable_traffic_light = True
        self.enable_pedestrian = True
        self.enable_violation = True
        self.enable_multiscale = True
        self.enable_flow_count = False
        self.enable_crosswalk = True
        self.enable_depth = True
        self.enable_lane = True

    def _sync_color_config(self):
        """同步颜色校正参数到 ImageProcessor"""
        c = self.config.color
        self._img_processor.set_brightness(c.brightness)
        self._img_processor.set_contrast(c.contrast)
        self._img_processor.set_saturation(c.saturation)
        self._img_processor.set_white_balance(c.wb_r, c.wb_g, c.wb_b)

    def _detect_gpu(self) -> dict:
        """检测 GPU 能力, 返回设备信息字典"""
        info = {
            'device': 'cpu',
            'device_str': 'cpu',
            'name': 'CPU',
            'vram_gb': 0,
            'fp16_available': False,
            'compute_capability': (0, 0),
        }

        if not torch.cuda.is_available():
            print("[WARN] CUDA 不可用 (可能安装了 CPU 版 PyTorch), 使用 CPU 模式")
            return info

        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        cc = (props.major, props.minor)

        # CUDA 计算能力 >= 7.0 支持高效 fp16
        # RTX 3070 = CC 8.6, 完全支持 fp16 tensor cores
        fp16_ok = cc[0] >= 7

        info.update({
            'device': self.model_config.device,
            'device_str': f'cuda:{self.model_config.device}' if self.model_config.device != 'cpu' else 'cpu',
            'name': gpu_name,
            'vram_gb': vram,
            'fp16_available': fp16_ok,
            'compute_capability': cc,
        })

        print(f"[INFO] GPU: {gpu_name} | VRAM: {vram:.1f} GB | CC: {cc[0]}.{cc[1]} | fp16: {'OK' if fp16_ok else 'NO'}")

        # 自动启用/禁用 fp16
        if fp16_ok and not self.model_config.fp16:
            print("[INFO] GPU 支持 fp16, 自动启用半精度加速")
            self.model_config.fp16 = True
        elif not fp16_ok and self.model_config.fp16:
            print("[WARN] GPU 不支持高效 fp16, 禁用半精度")
            self.model_config.fp16 = False

        return info

    @property
    def _device_str(self) -> str:
        """获取当前设备字符串, 用于 ultralytics 推理"""
        if self.model_config.device == 'cpu':
            return 'cpu'
        return f'cuda:{self.model_config.device}'

    def initialize(self) -> bool:
        print(f"\n{'='*60}")
        print(f"LPR System v6.0 - Traffic Vision (GPU Accelerated)")
        print(f"{'='*60}")

        self._gpu_info = self._detect_gpu()

        if self._gpu_info['device_str'] == 'cpu':
            self.model_config.device = 'cpu'
            self.model_config.fp16 = False

        self._load_yolo_detector()
        self._load_traffic_light_model()
        self._load_pedestrian_model()
        self._load_crosswalk_model()
        self._load_traffic_modules()
        self._load_depth_estimator()
        self._warmup()

        self.is_ready = True
        print("\n[INFO] 引擎初始化完成\n")
        return True

    def _load_plate_detector(self) -> bool:
        print("[1/5] 加载车牌检测器...")
        try:
            from ultralytics import YOLO
            model_path = os.path.join(self.project_root, 'models', 'plate_detector.pt')
            if not os.path.exists(model_path):
                print(f"[WARN] 车牌模型不存在: {model_path}")
                return False
            self.plate_model = YOLO(model_path)
            self.plate_model.to(self._device_str)
            print("[INFO] 车牌检测器加载成功")
            return True
        except Exception as e:
            print(f"[ERROR] 车牌检测器加载失败: {e}")
            return False

    def _load_yolo_detector(self):
        print("[2/5] 加载YOLOv8通用检测器...")
        try:
            from ultralytics import YOLO
            model_path = os.path.join(self.project_root, 'models', self.model_config.vehicle_detector)
            if os.path.exists(model_path):
                self.vehicle_model = YOLO(model_path)
                self.vehicle_model.to(self._device_str)
                print("[INFO] YOLOv8加载成功")
            else:
                print(f"[WARN] YOLOv8不存在: {model_path}")
                self.vehicle_model = None
        except Exception as e:
            print(f"[WARN] YOLOv8加载失败: {e}")
            self.vehicle_model = None

    def _load_traffic_light_model(self):
        """加载红绿灯专用检测模型 (检测+颜色分类一步到位)"""
        try:
            from ultralytics import YOLO
            model_path = os.path.join(self.project_root, 'models', self.model_config.traffic_light_detector)
            if os.path.exists(model_path):
                self.traffic_light_model = YOLO(model_path)
                self.traffic_light_model.to(self._device_str)
                print(f"[INFO] 红绿灯检测模型加载成功 (类别: {self.traffic_light_model.names})")
            else:
                print(f"[WARN] 红绿灯模型不存在: {model_path}，将使用 HSV 回退")
                self.traffic_light_model = None
        except Exception as e:
            print(f"[WARN] 红绿灯模型加载失败: {e}，将使用 HSV 回退")
            self.traffic_light_model = None

    def _load_pedestrian_model(self):
        """加载行人专用检测模型"""
        try:
            from ultralytics import YOLO
            model_path = os.path.join(self.project_root, 'models', self.model_config.pedestrian_detector)
            if os.path.exists(model_path):
                self.pedestrian_model = YOLO(model_path)
                self.pedestrian_model.to(self._device_str)
                print(f"[INFO] 行人检测模型加载成功 (类别: {self.pedestrian_model.names})")
            else:
                print(f"[WARN] 行人模型不存在: {model_path}")
                self.pedestrian_model = None
        except Exception as e:
            print(f"[WARN] 行人模型加载失败: {e}")
            self.pedestrian_model = None

    def _load_crosswalk_model(self):
        """加载斑马线专用检测模型"""
        try:
            from ultralytics import YOLO
            model_path = os.path.join(self.project_root, 'models', self.model_config.crosswalk_detector)
            if os.path.exists(model_path):
                self.crosswalk_model = YOLO(model_path)
                self.crosswalk_model.to(self._device_str)
                print(f"[INFO] 斑马线检测模型加载成功 (类别: {self.crosswalk_model.names})")
            else:
                print(f"[WARN] 斑马线模型不存在: {model_path}，将使用梯度检测器回退")
                self.crosswalk_model = None
        except Exception as e:
            print(f"[WARN] 斑马线模型加载失败: {e}")
            self.crosswalk_model = None

    def _load_ocr(self):
        print("[3/5] 加载OCR引擎...")
        try:
            from src.modules.plate_ocr import PlateOCR
            self.ocr_engine = PlateOCR()
            print(f"[INFO] OCR加载成功: {self.ocr_engine.engine_name}")
        except Exception as e:
            print(f"[WARN] OCR加载失败: {e}")
            self.ocr_engine = None

    def _load_traffic_modules(self):
        print("[4/5] 加载交通分析模块...")

        # 红绿灯分类器 (优先级: 专用YOLO模型 > CNN > HSV)
        if self.traffic_light_model:
            print("[INFO] 红绿灯检测: YOLO专用模型 (颜色直接检测)")
        else:
            try:
                from src.modules.traffic_light_cnn import TrafficLightCNNClassifier
                cnn_clf = TrafficLightCNNClassifier(device=self._device_str)
                if cnn_clf.available:
                    self.traffic_classifier = cnn_clf
                    print("[INFO] 红绿灯分类器: CNN")
                else:
                    from src.modules.traffic_signal import TrafficLightClassifier
                    self.traffic_classifier = TrafficLightClassifier
                    print("[INFO] 红绿灯分类器: HSV回退")
            except Exception:
                from src.modules.traffic_signal import TrafficLightClassifier
                self.traffic_classifier = TrafficLightClassifier
                print("[INFO] 红绿灯分类器: HSV回退")

        # 红绿灯状态跟踪
        try:
            from src.modules.traffic_signal import TrafficLightTracker
            self.traffic_tracker = TrafficLightTracker(stability_frames=3)
        except Exception:
            pass

        # 违规检测
        try:
            from src.modules.violation_detector import ViolationDetector
            self.violation_detector = ViolationDetector()
        except Exception:
            pass

        # 违规抓拍保存
        try:
            from src.modules.violation_saver import ViolationSaver
            self.violation_saver = ViolationSaver()
        except Exception:
            pass

        # 车流量统计
        try:
            from src.modules.flow_counter import FlowCounter
            self.flow_counter = FlowCounter()
        except Exception:
            pass

        # 斑马线检测
        try:
            from src.modules.crosswalk_detector import CrosswalkDetector
            self.crosswalk_detector = CrosswalkDetector(roi_top_ratio=0.35, peak_ratio_thresh=0.30, gradient_thresh=30)
        except Exception:
            pass

        # 轨迹跟踪器
        try:
            from src.modules.trajectory import TrajectoryTracker, TrackAdapter
            # 车辆: BoT-SORT 提供 ID, 适配层重建运动量 (更稳, 抗遮挡)
            self.vehicle_tracker = TrajectoryTracker(history_len=8)
            # 行人: 保留原 IoU 跟踪器
            self.ped_tracker = TrajectoryTracker(history_len=8)
        except Exception:
            pass

        # 前车碰撞预警 (FCW)
        try:
            from src.modules.collision_warning import CollisionWarning
            self.collision_warning = CollisionWarning(ttc_threshold=3.0)
        except Exception as e:
            print(f"[WARN] 碰撞预警模块加载失败: {e}")
            self.collision_warning = None

        # 车道检测 (UFLD)
        try:
            from src.modules.ufld_lane_detector import UFLDLaneDetector
            model_path = os.path.join(self.project_root, 'models', 'ufld', 'culane_18.onnx')
            if os.path.exists(model_path):
                self.lane_detector = UFLDLaneDetector(model_path)
            else:
                print(f"[WARN] 车道模型不存在: {model_path}")
                self.lane_detector = None
        except Exception as e:
            print(f"[WARN] 车道检测加载失败: {e}")
            self.lane_detector = None

        # 传统 CV 车道检测 (用于下半象限 Q3/Q4 的二次采集劣化画面)
        try:
            from src.modules.lane_detector import LaneDetector
            self.lane_detector_cv = LaneDetector()
            print("[INFO] 传统 CV 车道检测器加载成功")
        except Exception as e:
            print(f"[WARN] 传统车道检测加载失败: {e}")
            self.lane_detector_cv = None

        print("[INFO] 交通分析模块加载完成")

    def _load_depth_estimator(self):
        """加载深度估计器 (Depth Anything V2 Small)"""
        try:
            from src.modules.depth_estimator import DepthEstimator
            self.depth_estimator = DepthEstimator(
                project_root=self.project_root,
                device=self._device_str,
                interval=3,
            )
            self.enable_depth = self.depth_estimator.enabled
            # 预热 (避免首帧卡顿)
            if self.depth_estimator.available:
                dummy = np.zeros((518, 518, 3), dtype=np.uint8)
                self.depth_estimator.estimate(dummy)
        except Exception as e:
            print(f"[WARN] 深度估计器初始化失败: {e}")
            self.depth_estimator = None

    def _warmup(self):
        print("[5/5] GPU预热中...")
        dummy = np.zeros((self.model_config.img_size, self.model_config.img_size, 3), dtype=np.uint8)
        if self.plate_model:
            try:
                self.plate_model.predict(dummy, conf=0.25, verbose=False, device=self._device_str)
            except Exception:
                pass
        if self.vehicle_model:
            try:
                self.vehicle_model.predict(dummy, conf=0.25, verbose=False, device=self._device_str)
            except Exception:
                pass
        if self.traffic_light_model:
            try:
                self.traffic_light_model.predict(dummy, conf=0.25, verbose=False, device=self._device_str)
            except Exception:
                pass
        if self.pedestrian_model:
            try:
                self.pedestrian_model.predict(dummy, conf=0.25, verbose=False, device=self._device_str)
            except Exception:
                pass
        if self.crosswalk_model:
            try:
                self.crosswalk_model.predict(dummy, conf=0.25, verbose=False, device=self._device_str)
            except Exception:
                pass
        self._warmup_done = True
        print("[INFO] 预热完成")

    # ── 统一检测入口 ──────────────────────────────────────

    def detect(self, frame: np.ndarray,
               plate_enabled: bool = True,
               vehicle_enabled: bool = True) -> Tuple:
        """兼容旧接口"""
        result = self.detect_full(frame)
        return result.frame, result.plates, result.vehicles

    def detect_full(self, frame: np.ndarray) -> DetectionResult:
        """完整检测流水线"""
        if not self.is_ready or frame is None:
            return DetectionResult(frame if frame is not None else np.zeros((480, 640, 3), np.uint8))

        start = time.time()
        # 颜色校正 (白平衡/亮度/对比度/饱和度)
        frame = self._img_processor.process(frame)
        result = DetectionResult(frame)
        self._yolo_crosswalk_regions = []

        # 1. YOLO 通用检测 (车辆+行人+红绿灯)
        if self.vehicle_model and self.enable_vehicle:
            self._detect_yolo_all(result)

        # 1.5 行人专用检测 (先跑, 红绿灯重叠过滤需要行人数据)
        if self.pedestrian_model and self.enable_pedestrian:
            self._detect_pedestrians(result)

        # 1.6 红绿灯专用检测 (用行人+车辆结果过滤误检)
        if self.traffic_light_model and self.enable_traffic_light:
            self._detect_traffic_lights(result)

        # 1.7 斑马线专用检测
        if self.crosswalk_model and self.enable_crosswalk:
            self._detect_crosswalks(result)

        # 2. 红绿灯状态稳定
        if self.traffic_tracker and result.traffic_lights:
            result.traffic_lights = self.traffic_tracker.update(result.traffic_lights)

        tl_state = self.traffic_tracker.current_state if self.traffic_tracker else 'unknown'

        # 同一个路口所有灯颜色一致
        if tl_state in ('red', 'green', 'yellow'):
            for tl in result.traffic_lights:
                tl['color'] = tl_state

        # 3. 轨迹跟踪
        vehicle_tracks = {}
        ped_tracks = {}
        if self.vehicle_tracker:
            v_dets = [{'bbox': v['bbox'], 'class_id': v['class_id'], 'confidence': v['confidence'],
                       'track_id': v.get('track_id')}
                      for v in result.vehicles]
            vehicle_tracks = self.vehicle_tracker.update(v_dets)

        if self.ped_tracker:
            p_dets = [{'bbox': p['bbox'], 'class_id': 0, 'confidence': p['confidence']}
                      for p in result.pedestrians]
            ped_tracks = self.ped_tracker.update(p_dets)

        # 3.5 车辆测距 (Depth Anything V2, 降频)
        if self.depth_estimator and self.enable_depth and vehicle_tracks:
            self._estimate_vehicle_distances(result.frame, vehicle_tracks)

        # 3.6 前车碰撞预警 (FCW)
        if self.collision_warning and vehicle_tracks:
            self.fcw_results = self.collision_warning.update(vehicle_tracks)

        # 3.6.5 斑马线检测 (提前到车道检测前, 供 HUD 减速慢行提示复用)
        crosswalk_regions = []
        if self.enable_crosswalk:
            from src.modules.crosswalk_detector import CrosswalkRegion
            for cw in self._yolo_crosswalk_regions:
                crosswalk_regions.append(CrosswalkRegion(cw['bbox'], cw['confidence']))
            if not crosswalk_regions and self.crosswalk_detector:
                crosswalk_regions = self.crosswalk_detector.detect(frame)

        # 3.7 车道检测 + 截断到前车距离 + 停止线/路口隐藏 (下半象限用传统 CV)
        if (self.lane_detector or self.lane_detector_cv) and self.enable_lane:
            front_y = None
            nearest_dist = float('inf')
            if vehicle_tracks:
                for tid, t in vehicle_tracks.items():
                    d = t.get('distance')
                    if d is not None and d < nearest_dist:
                        nearest_dist = d
                        front_y = int(t['box'][3])   # 最近前车底边 y

            # 车距告警等级: <2m 红(触发减速慢行), <5m 橙(仅车道颜色预警), 其余绿
            warn_level = 'green'
            if nearest_dist < 2.0:
                warn_level = 'red'
            elif nearest_dist < 5.0:
                warn_level = 'orange'

            # 停止线检测 + 距离 < 5m → 隐藏车道辅助线
            hide_lane = False
            stop_y = None  # 临时禁用停止线检测 (测试车道线)
            if stop_y is not None and self._last_depth_map is not None and self.depth_estimator:
                cx = result.frame.shape[1] // 2
                rel = self.depth_estimator.depth_at(self._last_depth_map, cx, int(stop_y))
                dist = self.depth_estimator.to_metric(rel)
                if dist is not None and dist < 5.0:
                    hide_lane = True

            crosswalk_detected = bool(crosswalk_regions)
            try:
                # 下半象限(Q3/Q4)用传统 CV(对二次采集劣化更鲁棒), 上半(Q1/Q2)/单路用 UFLD
                if self._lane_slot >= 2 and self.lane_detector_cv is not None:
                    self.lane_result = self.lane_detector_cv.detect(result.frame, lane_slot=self._lane_slot)
                elif self.lane_detector is not None:
                    self.lane_result = self.lane_detector.detect(
                        result.frame, clip_y=None, hide_lane=hide_lane,
                        warn_level=warn_level, crosswalk_detected=crosswalk_detected,
                        lane_slot=self._lane_slot)
                else:
                    self.lane_result = None
                if self.lane_result is not None:
                    result.frame = self.lane_result.frame
                    if self.stats['total_frames'] % 30 == 0:
                        print(f"[LANE] detected={self.lane_result.detected} "
                              f"L={'Y' if self.lane_result.left else 'N'} "
                              f"R={'Y' if self.lane_result.right else 'N'} warn={warn_level}")
            except Exception as e:
                print(f"[WARN] 车道检测失败: {e}")

        # 4. 车牌检测
        if self.enable_plate:
            self._detect_plates(result)

        # 5. 车流量统计
        if self.enable_flow_count and self.flow_counter and vehicle_tracks:
            # 首帧根据实际画面高度动态设检测线 (40% 位置, 适配更多场景)
            if not self._flow_line_set:
                line_y = int(frame.shape[0] * 0.40)
                self.flow_counter.set_lines([line_y])
                self._flow_line_set = True
            result.flow_stats = self.flow_counter.update(vehicle_tracks)

        # 5.5 红绿灯-斑马线区域关联 (斑马线已在 3.6.5 提前算好)
        if self.violation_detector:
            self.violation_detector.update_zones(result.traffic_lights, crosswalk_regions)

        # 6. 违规分析 (用当前帧检测的灯色, 比tracker更实时)
        cur_is_red = any(tl.get('color') == 'red' for tl in result.traffic_lights)
        cur_is_green = any(tl.get('color') == 'green' for tl in result.traffic_lights)
        viol_tl_state = 'red' if (cur_is_red and not cur_is_green) else ('green' if cur_is_green else tl_state)

        if self.enable_violation and self.violation_detector:
            result.violations = self.violation_detector.analyze(
                frame.shape,
                result.pedestrians,
                result.vehicles,
                viol_tl_state,
                vehicle_tracks,
                ped_tracks,
            )
            if viol_tl_state == 'green' or (not cur_is_red and not cur_is_green):
                self.violation_detector.clear_flags()
                self._violators.clear()
                viol_tl_state = 'green'

            self.stats['violation_count'] += len(result.violations) if result.violations else 0

            # 记录违规, 每帧更新bbox (同一track_id会覆盖)
            for v in result.violations:
                tid = v.get('track_id', -1)
                self._violators[tid] = {
                    'type': v.get('type', 'unknown'),
                    'bbox': v.get('bbox', (0,0,0,0)),
                    'frame': self.stats['total_frames'],
                }

            # 绿灯时清空
            if tl_state == 'green':
                self._violators.clear()

            # 7. 违规抓拍保存 (后台线程，不阻塞主帧)
            if result.violations and self.violation_saver:
                snap = frame.copy()
                viols = list(result.violations)
                self._save_thread = threading.Thread(
                    target=self.violation_saver.save,
                    args=(snap, viols), daemon=True
                )
                self._save_thread.start()

        # 8. 绘制结果
        self._draw_results(result, tl_state, vehicle_tracks, ped_tracks)

        result.latency = (time.time() - start) * 1000
        self._update_stats(result.latency, len(result.plates))

        return result

    # ── YOLO 通用检测 ─────────────────────────────────────

    def _detect_yolo_all(self, result: DetectionResult):
        try:
            dets = self.vehicle_model.predict(
                result.frame, conf=0.4, iou=0.5,
                classes=COCO_ALL_INTEREST,
                verbose=False, device=self._device_str
            )

            for r in dets:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    tid = int(box.id[0]) if box.id is not None else None

                    det = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': COCO_NAMES.get(cls, '未知'),
                        'track_id': tid,
                    }

                    if cls in COCO_VEHICLE_CLASSES:
                        result.vehicles.append(det)
                    elif cls in COCO_PERSON_CLASS:
                        # 行人专用模型可用时跳过 YOLOv8n class 0
                        if self.pedestrian_model:
                            continue
                        result.pedestrians.append(det)
                    elif cls in COCO_TRAFFIC_LIGHT_CLASS:
                        # 如果红绿灯专用模型可用，跳过 YOLOv8n 的 class 9
                        if self.traffic_light_model:
                            continue
                        crop = result.frame[max(0,y1):y2, max(0,x1):x2]
                        color, c_conf = self.traffic_classifier.classify(crop)
                        det['color'] = color
                        det['color_confidence'] = c_conf
                        result.traffic_lights.append(det)
        except Exception as e:
            print(f"[WARN] YOLO检测失败: {e}")

    # ── 红绿灯专用检测 ─────────────────────────────────────

    def _detect_traffic_lights(self, result: DetectionResult):
        """模型找位置, HSV判定颜色 (模型分类不准, 不依赖class_id)"""
        if not self.traffic_light_model:
            return

        try:
            dets = self.traffic_light_model.predict(
                result.frame, conf=0.2, iou=0.4,
                classes=[0, 1, 2],
                verbose=False, device=self._device_str
            )

            h, w = result.frame.shape[:2]
            frame_area = h * w

            raw = []
            for r in dets:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    bw, bh = x2 - x1, y2 - y1
                    box_area = bw * bh

                    # 几何过滤
                    if box_area < 15 or box_area > frame_area * 0.015:
                        continue
                    if bh > 0 and not (0.35 < bw / bh < 2.5):
                        continue
                    if y1 > h * 0.65:
                        continue

                    # HSV判定实际颜色
                    crop = result.frame[max(0,y1):y2, max(0,x1):x2]
                    real_color, _ = TrafficLightClassifier.classify(crop)

                    raw.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'color': real_color,
                    })

            filtered = self._dedup_traffic_lights(raw)

            for d in filtered:
                result.traffic_lights.append({
                    'bbox': d['bbox'],
                    'confidence': d['confidence'],
                    'class_id': -1,
                    'class_name': f"红绿灯({d['color']})",
                    'color': d['color'],
                    'color_confidence': d['confidence'],
                })

        except Exception as e:
            print(f"[WARN] 红绿灯模型检测失败: {e}")

    def _dedup_traffic_lights(self, detections: list) -> list:
        """红绿灯去重: 重叠区域保留置信度最高的"""
        if len(detections) <= 1:
            return detections
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        keep = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            suppressed = False
            for k in keep:
                kx1, ky1, kx2, ky2 = k['bbox']
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                karea = (kx2 - kx1) * (ky2 - ky1)
                union = area + karea - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.20:
                    suppressed = True; break
            if not suppressed:
                keep.append(d)
        return keep

    # ── 行人专用检测 ─────────────────────────────────────

    def _detect_pedestrians(self, result: DetectionResult):
        """使用专用模型检测行人"""
        if not self.pedestrian_model:
            return

        try:
            dets = self.pedestrian_model.predict(
                result.frame, conf=0.4, iou=0.5,
                verbose=False, device=self._device_str
            )

            for r in dets:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    result.pedestrians.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': 0,
                        'class_name': '行人',
                    })

        except Exception as e:
            print(f"[WARN] 行人模型检测失败: {e}")

    # ── 斑马线专用检测 ─────────────────────────────────────

    def _detect_crosswalks(self, result: DetectionResult):
        """使用专用模型检测斑马线"""
        if not self.crosswalk_model:
            return

        try:
            dets = self.crosswalk_model.predict(
                result.frame, conf=0.25, iou=0.5,
                verbose=False, device=self._device_str
            )

            for r in dets:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    # 画框
                    cv2.rectangle(result.frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    ty = max(y1 - 8, 15)
                    cv2.putText(result.frame, f"Crosswalk {conf:.2f}", (x1, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                    # 存储区域供违规检测使用
                    self._yolo_crosswalk_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                    })

        except Exception as e:
            print(f"[WARN] 斑马线模型检测失败: {e}")

    # ── 车牌检测 ──────────────────────────────────────────

    def _detect_plates(self, result: DetectionResult):
        all_plates = []

        plates = self._run_plate_detect(result.frame, self.model_config.img_size)
        all_plates.extend(plates)

        # 多尺度: 仅当主尺度未检出或检出极少时才跑第二次 (省一半开销)
        if (self.enable_multiscale
                and len(plates) < 2
                and max(result.frame.shape[:2]) > 800):
            large_size = int(min(self.model_config.img_size * 1.5, 1280) // 32) * 32
            if large_size != self.model_config.img_size:
                plates2 = self._run_plate_detect(result.frame, large_size)
                all_plates.extend(plates2)

        all_plates = self._nms_plates(all_plates, iou_thresh=0.5)

        for p in all_plates:
            x1, y1, x2, y2 = int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])
            bw, bh = x2-x1, y2-y1

            # 跳过太小的检测框 (可能是误检)
            if bw < 15 or bh < 10:
                p['plate_no'] = ''
                p['color'] = '未知'
                continue

            # 扩展裁剪区域: 水平 20%, 垂直 15% (使馆车牌末尾"使"字需要更多右边距)
            px, py = max(int(bw*0.20), 5), max(int(bh*0.15), 3)
            cx1 = max(0, x1-px); cy1 = max(0, y1-py)
            cx2 = min(result.frame.shape[1], x2+px)
            cy2 = min(result.frame.shape[0], y2+py)
            plate_img = result.frame[cy1:cy2, cx1:cx2]

            plate_no, color = '', '未知'
            if self.ocr_engine and plate_img.size > 0 and plate_img.shape[0] > 5 and plate_img.shape[1] > 10:
                # 小图放大到 OCR 模型能识别的最小尺寸
                h, w = plate_img.shape[:2]
                if w < 80:
                    scale = 80 / w
                    plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

                # 倾斜校正: 摆正车牌再OCR (防8→3等倾斜误判)
                plate_img = self._deskew_plate(plate_img)

                # 黑底白字车牌(使馆/领馆)反色
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                mean_v = np.mean(gray)
                if mean_v < 100:
                    plate_img = cv2.bitwise_not(plate_img)

                plate_no, color = self.ocr_engine.recognize(plate_img)
                # 使馆/领馆→黑牌, 警用/军用→白牌
                if any(c in plate_no for c in '使领'):
                    color = '黑牌'
                elif '警' in plate_no or 'WJ' in plate_no or '军' in plate_no:
                    color = '白牌'
            p['plate_no'] = plate_no
            p['color'] = color


        result.plates = self.tracker.update(all_plates)

    def _deskew_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """车牌倾斜校正: 找出旋转角并摆正"""
        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) < 20:
                return plate_img
            # 最小外接矩形求角度
            rect = cv2.minAreaRect(coords.astype(np.float32))
            angle = rect[2]
            # 角度归一化: OpenCV返回(-90,0], 转为实际偏角
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 2:  # 小于2度不处理
                return plate_img
            h, w = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(plate_img, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return plate_img

    def _run_plate_detect(self, frame, imgsz) -> List[Dict]:
        results = []
        try:
            # 在小画面(四路分割)中降低置信度阈值
            conf = 0.25 if max(frame.shape[:2]) < 600 else self.model_config.conf_threshold
            dets = self.plate_model.predict(
                frame, conf=conf,
                iou=self.model_config.iou_threshold,
                imgsz=imgsz, verbose=False, device=self._device_str
            )
            for r in dets:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    results.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'confidence': conf})
        except Exception as e:
            print(f"[ERROR] 车牌检测异常: {e}")
        return results

    def _nms_plates(self, plates: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        if len(plates) <= 1:
            return plates
        boxes = np.array([[p['x1'], p['y1'], p['x2'], p['y2']] for p in plates])
        scores = np.array([p['confidence'] for p in plates])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2-x1) * (y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return [plates[i] for i in keep]

    # ── 绘制 ──────────────────────────────────────────────

    # ── 车辆测距 ──────────────────────────────────────────

    def _estimate_vehicle_distances(self, frame, vehicle_tracks):
        """对每个车辆 track 计算底边中心距离 (降频, 非估计帧复用上一帧距离)"""
        if not self.depth_estimator:
            return
        self._depth_frame_counter += 1
        if (self._depth_frame_counter % self.depth_estimator.interval) != 0:
            return
        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return
        self._last_depth_map = depth_map
        for tid, t in vehicle_tracks.items():
            box = t['box']
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])  # 底边中心 ≈ 车辆接地处
            rel = self.depth_estimator.depth_at(depth_map, cx, cy)
            t['distance'] = self.depth_estimator.to_metric(rel)

    def _detect_stop_line(self, frame):
        """检测水平停止线, 返回其 y 位置 (或 None)"""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 60]))
        lines = cv2.HoughLinesP(white, 1, np.pi / 180, 50,
                                minLineLength=800, maxLineGap=50)
        if lines is None:
            return None
        h = frame.shape[0]
        best = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            if abs((y2 - y1) / dx) < 0.1:   # 近水平
                y = (y1 + y2) / 2
                if y > 0.4 * h:             # 画面下半部分
                    if best is None or y > best:  # 取最低(离车最近)的
                        best = y
        return best

    def calibrate_depth(self, frame, px: int, py: int, meters: float):
        """标定: 用画面中一个已知距离的地面点反算尺度 α, 返回 (成功, 消息)"""
        if not self.depth_estimator:
            return False, "深度估计器未加载"
        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return False, "深度估计失败"
        rel = self.depth_estimator.depth_at(depth_map, px, py)
        if rel is None or rel <= 1e-6:
            return False, "该点深度无效, 请点地面位置"
        alpha = meters * rel
        self.depth_estimator.set_alpha(alpha)
        self._last_depth_map = depth_map
        return True, f"标定成功: α = {alpha:.3f}"

    def _draw_results(self, result: DetectionResult, tl_state: str,
                      vehicle_tracks: Dict, ped_tracks: Dict):
        frame = result.frame
        r = self._text_renderer  # 批量渲染器

        # 车辆: 跟踪框 + 底边中心标记点(浅粉) + 最近5帧轨迹 (其他模型框不受影响)
        PINK = (193, 182, 255)  # 浅粉色 BGR

        if not hasattr(self, '_vehicle_trail'):
            self._vehicle_trail = {}  # (lane_slot, track_id) → deque[(cx, cy)] 底边中心轨迹

        # 四路分屏按象限隔离轨迹, 避免不同象限同名 track_id 串扰 (轨迹乱飞根因)
        slot = getattr(self, '_lane_slot', 0)

        # 更新底边中心轨迹 (仅存活 track)
        for tid, t in vehicle_tracks.items():
            bx, by = (t['box'][0] + t['box'][2]) / 2, t['box'][3]  # 底边中心
            key = (slot, tid)
            if key not in self._vehicle_trail:
                self._vehicle_trail[key] = deque(maxlen=5)
            self._vehicle_trail[key].append((bx, by))
        for key in list(self._vehicle_trail.keys()):
            if key[0] != slot or key[1] not in vehicle_tracks:
                del self._vehicle_trail[key]

        # 绘制车辆框 + 底边中心点 + 轨迹
        for tid, t in vehicle_tracks.items():
            x1, y1, x2, y2 = [int(c) for c in t['box']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), PINK, 1)  # 车辆框
            label_y = y1 - 5 if y1 - 5 > 12 else y1 + 14
            cv2.putText(frame, f"ID:{tid}", (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, PINK, 1)  # 车辆编号
            cx, cy = (x1 + x2) // 2, y2  # 底边中心
            cv2.circle(frame, (cx, cy), 4, PINK, -1)  # 底边中心点
            dist = t.get('distance')
            if dist is not None:
                # 距离标注在底边中心点正下方 (深绿色)
                text = f"{dist:.1f}m"
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.putText(frame, text, (cx - tw // 2, cy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 0), 1)
            # FCW 碰撞预警已移除 (改用车道辅助线颜色体现: 绿/橙/红)
            # 轨迹 (过去5帧底边中心位置)
            trail = self._vehicle_trail.get((slot, tid))
            if trail and len(trail) >= 2:
                pts = [(int(px), int(py)) for px, py in trail]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], PINK, 2)
                for px, py in pts:
                    cv2.circle(frame, (px, py), 2, PINK, -1)

        # 红绿灯 (只画框不标文字, 避免闪烁)
        color_map = {
            'red': (0, 0, 255), 'yellow': (0, 200, 255),
            'green': (0, 255, 80), 'unknown': (128, 128, 128),
        }
        for tl in result.traffic_lights:
            c = color_map.get(tl.get('color', 'unknown'))
            if c == (128, 128, 128):  # unknown不画
                continue
            x1, y1, x2, y2 = tl['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        # 行人: 当前帧有红灯→红色"行人闯红灯", 否则→青色"person"
        has_red = any(tl.get('color') == 'red' for tl in result.traffic_lights)
        has_green = any(tl.get('color') == 'green' for tl in result.traffic_lights)
        is_red = has_red and not has_green  # 有红无绿才判定为红灯

        # ── 行人绿灯漏检修正 ──
        # 画面中大量行人(≥3)正常通行 → 人行道是绿灯, 不标违规
        if is_red and len(result.pedestrians) >= 3:
            h = result.frame.shape[0]
            # 统计下半画面行人(正在过马路)
            crossing = sum(1 for p in result.pedestrians
                          if (p['bbox'][1] + p['bbox'][3]) / 2 > h * 0.30)
            if crossing >= 3:
                is_red = False  # 多人同时过马路 → 人行绿灯
        for p in result.pedestrians:
            x1, y1, x2, y2 = p['bbox']
            if is_red:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                r.add("闯红灯", (x1, max(y1-14, 0)), (0, 0, 255), 18)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                r.add("person", (x1, max(y1-14, 0)), (255, 200, 0), 14)

        # 车辆违规框: 位置匹配跟踪 (四宫格下 track_id 每帧重置, 用中心距离匹配让框跟随车辆)
        expired = []
        for tid, vinfo in list(self._violators.items()):
            # 非车辆违规 或 过期 → 清除
            if 'vehicle' not in vinfo.get('type', ''):
                if self.stats['total_frames'] - vinfo['frame'] > 60:
                    expired.append(tid)
                continue
            bbox = vinfo.get('bbox')
            if bbox is None:
                if self.stats['total_frames'] - vinfo['frame'] > 60:
                    expired.append(tid)
                continue
            # 用中心距离匹配当前帧车辆, 让框跟随车辆移动
            if vehicle_tracks:
                bx, by = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                best_box, best_d = None, 150 * 150
                for t in vehicle_tracks.values():
                    tb = t.get('box')
                    if not tb:
                        continue
                    tx, ty = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                    d = (bx - tx) ** 2 + (by - ty) ** 2
                    if d < best_d:
                        best_d = d
                        best_box = tb
                if best_box is not None:
                    bbox = best_box
                    vinfo['bbox'] = bbox
                    vinfo['frame'] = self.stats['total_frames']
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            r.add("违规闯红灯", (x1, max(y1-14, 0)), (0, 0, 255), 18)
        for tid in expired:
            del self._violators[tid]

        # 车牌 (只标号码)
        for det in result.plates:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            stable = det.get('is_stable', False)
            c = (0, 255, 80) if stable else (0, 160, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2 if stable else 1)
            label = det.get('plate_no', '')
            if label:
                r.add(label, (x1, max(y1-14, 0)), (255, 255, 0), 18)

        # 车流量检测线
        if self.flow_counter:
            frame = self.flow_counter.draw_lines(frame)

        # 斑马线区域 (由 _detect_crosswalks 用 YOLO 模型画框)

        # 状态栏 — 简洁信号灯指示
        y_offset = 25
        short = {'red': 'R', 'yellow': 'Y', 'green': 'G', 'unknown': '-'}
        state_c = color_map.get(tl_state, (128, 128, 128))
        cv2.circle(frame, (18, y_offset - 4), 6, state_c, -1)
        cv2.putText(frame, short.get(tl_state, '-'), (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if result.flow_stats:
            for line_name, stats in result.flow_stats.items():
                y_offset += 22
                cv2.putText(frame, f"Flow: In={stats['in_count']} Out={stats['out_count']}",
                            (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if result.violations:
            y_offset += 22
            cv2.putText(frame, f"VIOLATIONS: {len(self._violators)}",
                        (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 碰撞提示: 仅在 Q3/Q4(下半象限) 显示, 基于最近前车距离 (< 2m 触发, 检测出 >2m 才解除)
        if self._lane_slot >= 2:
            if not hasattr(self, '_collision_latch'):
                self._collision_latch = {}        # lane_slot → bool 锁存
            if not hasattr(self, '_collision_dist_cache'):
                self._collision_dist_cache = {}   # lane_slot → float 最近一次检测距离

            slot = self._lane_slot
            nearest = None
            nearest_conf = 0.0
            for t in vehicle_tracks.values():
                d = t.get('distance')
                if d is not None and (nearest is None or d < nearest):
                    nearest = d
                    nearest_conf = t.get('confidence', 0.0)

            # 深度估计每3帧才测一次; 只有检测帧(nearest有值)才更新状态, 非检测帧复用上次检测值保持 → 不闪
            if vehicle_tracks:
                if nearest is not None:
                    self._collision_dist_cache[slot] = nearest
                cur = self._collision_dist_cache.get(slot)
            else:
                self._collision_dist_cache.pop(slot, None)  # 无车 → 清缓存
                cur = None

            # 锁存: 检测出 <2m → 触发并保持; 检测出 >2m → 解除
            if cur is not None and cur < 2.0:
                self._collision_latch[slot] = True
            elif cur is None or cur > 2.0:
                self._collision_latch[slot] = False

            if self._collision_latch.get(slot, False) and cur is not None:
                self._draw_collision_alert(frame, cur)

            # 碰撞记录: 检测出 <2m 且为上升沿 → 记一条 (供记录页打印, 最多50条)
            if cur is not None and cur < 2.0:
                if not self._collision_recorded.get(slot, False):
                    self._collision_recorded[slot] = True
                    self._collision_seq += 1
                    self.collision_log.append({
                        'seq': self._collision_seq,
                        'time': time.strftime('%H:%M:%S'),
                        'distance': cur,
                        'confidence': nearest_conf,
                    })
                    if len(self.collision_log) > 50:
                        self.collision_log.pop(0)
            elif cur is None or cur >= 2.0:
                self._collision_recorded[slot] = False

        # 一次性渲染所有中文文字 (只做一次 PIL 转换)
        frame = r.render(frame)

        result.frame = frame

    def _get_collision_icon(self):
        """加载碰撞提示图标: 优先 models/ 下的 png/jpg, 否则桌面测试图"""
        if not hasattr(self, '_collision_icon'):
            self._collision_icon = None
            candidates = [
                os.path.join(self.project_root, 'models', 'collision_icon.png'),
                os.path.join(self.project_root, 'models', 'collision_icon.jpg'),
                r'C:/Users/Moxiaojiu/Desktop/3df6712b3989ea0a34165950af815a03.png',
            ]
            for p in candidates:
                if os.path.exists(p):
                    self._collision_icon = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    break
        return self._collision_icon

    def _draw_collision_alert(self, frame, dist):
        """碰撞提示: 完全模仿减速慢行 HUD 格式(图标 + 红色文字 + 红色方框)"""
        from PIL import Image, ImageDraw, ImageFont
        icon = self._get_collision_icon()
        h, w = frame.shape[:2]

        icon_h = 80  # 图标高度(与 HUD 一致, 调大)
        x0, y0 = 12, 12  # Q3/Q4 无减速慢行三角, 碰撞图标放左上角

        # 1. 碰撞图标
        if icon is not None:
            ih, iw = icon.shape[:2]
            scale = icon_h / ih
            ic = cv2.resize(icon, (int(iw * scale), int(ih * scale)))
            ih, iw = ic.shape[:2]
            x, y = min(x0, w - ih), min(y0, h - ih)
            if ic.shape[2] == 4:  # BGRA
                alpha = ic[..., 3:4].astype(np.float32) / 255.0
                roi = frame[y:y + ih, x:x + iw].astype(np.float32)
                frame[y:y + ih, x:x + iw] = (ic[..., :3].astype(np.float32) * alpha + roi * (1 - alpha)).astype(np.uint8)
            else:
                frame[y:y + ih, x:x + iw] = ic
        else:
            cx, cy = x0 + icon_h // 2, y0 + icon_h // 2
            r = icon_h // 2
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 0, 255))

        # 2. 文字 + 方框(完全模仿减速慢行 _draw_hud: 微软雅黑 10 号, 红色, 红色细边框)
        text = f"碰撞 {dist:.1f}m"
        cn_size = 10
        pad, line_h = 10, 13
        if not hasattr(self, '_cn_fonts'):
            self._cn_fonts = {}
        if cn_size not in self._cn_fonts:
            font = None
            for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, cn_size)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
            self._cn_fonts[cn_size] = font
        font = self._cn_fonts[cn_size]

        bb = font.getbbox(text)
        box_w = (bb[2] - bb[0]) + pad * 2
        box_h = pad * 2 + line_h

        yt = min(y0 + icon_h + 6, h)
        box_x = int(x0 + icon_h / 2 - box_w / 2)
        box_x = max(0, min(box_x, w - box_w))
        x1 = min(box_x + box_w, w)
        y1 = min(yt + box_h, h)
        if y1 > yt and x1 > box_x:
            crop = frame[yt:y1, box_x:x1]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, crop.shape[1] - 1, crop.shape[0] - 1], outline=(0, 0, 255), width=1)
            draw.text((pad, pad), text, font=font, fill=(0, 0, 255))
            frame[yt:y1, box_x:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── 统计 ──────────────────────────────────────────────

    def _update_stats(self, latency: float, plate_count: int):
        self.stats['total_frames'] += 1
        self.stats['latency_history'].append(latency)
        self.stats['avg_latency'] = sum(self.stats['latency_history']) / len(self.stats['latency_history'])
        fps = 1000 / latency if latency > 0 else 0
        self.stats['fps_history'].append(fps)
        self.stats['detection_count'] += plate_count

    def get_fps(self) -> float:
        if not self.stats['fps_history']:
            return 0.0
        return sum(self.stats['fps_history'][-30:]) / min(len(self.stats['fps_history']), 30)

    def reset_tracker(self):
        self.tracker = PlateTracker(max_age=15, iou_threshold=0.20)
