"""
深度估计模块 — Depth Anything V2 (Small)
- 加载 DAv2 Small 权重, 输出相对逆深度图 (值越大越近)
- 车辆框底边中心查深度 → 乘尺度 α 转真实米数
- 标定: 点一个已知距离的地面点 → α = Z * rel

依赖: pip install depth-anything-v2 (自带 dpt.py / dinov2.py 模型代码, 仅需 torch/torchvision/numpy)
权重: models/depth/depth_anything_v2_vits.pth (需从 HuggingFace 下载, 见 README 说明)
"""
import os
import json
from typing import Optional, Dict

import numpy as np
import cv2


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class DepthEstimator:
    """Depth Anything V2 相对深度估计器 + 尺度标定"""

    def __init__(self, project_root: str, device: str = 'cuda', interval: int = 3):
        self.project_root = project_root
        self.device = 'cuda' if (device != 'cpu' and _cuda_available()) else 'cpu'
        self.interval = max(1, interval)
        self.model = None
        self.available = False
        self.input_size = 518
        self.alpha = 1.0          # 尺度因子: Z_meter = alpha / rel_depth
        self.model_path = None

        cfg = self._load_config()
        self.enabled = bool(cfg.get('enabled', True))
        self.input_size = int(cfg.get('input_size', 518))
        self.interval = max(1, int(cfg.get('interval', 3)))
        self.alpha = float(cfg.get('alpha', 0.0))
        self.model_path = os.path.join(
            project_root, cfg.get('model_path', 'models/depth/depth_anything_v2_vits.pth')
        )

        if not self.enabled:
            print("[INFO] 深度估计已禁用 (depth.json enabled=false)")
            return
        self._load_model()

    # ── 模型加载 ────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[WARN] 深度模型不存在: {self.model_path} (跳过测距, 请先下载权重)")
            return
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
            # Small (ViT-S): encoder=vits, features=64, out_channels=[48,96,192,384]
            self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            if self.device != 'cpu':
                self.model = self.model.to(self.device)
            self.available = True
            print(f"[INFO] 深度估计器加载成功 (DAv2-Small, {self.device})")
        except ImportError:
            print("[WARN] 深度估计器加载失败: 未安装 depth-anything-v2, 请运行 pip install depth-anything-v2")
        except Exception as e:
            print(f"[WARN] 深度估计器加载失败: {e}")

    # ── 配置读写 ────────────────────────────────────────
    def _config_path(self) -> str:
        return os.path.join(self.project_root, 'configs', 'depth.json')

    def _load_config(self) -> Dict:
        p = self._config_path()
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_config(self):
        p = self._config_path()
        cfg = {
            'enabled': self.enabled,
            'model_path': 'models/depth/depth_anything_v2_vits.pth',
            'input_size': self.input_size,
            'interval': self.interval,
            'alpha': self.alpha,
        }
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── 推理 ────────────────────────────────────────────
    def estimate(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """返回相对逆深度图 (HxW float32, 值越大越近)"""
        if not self.available or frame_bgr is None:
            return None
        try:
            # 该包 infer_image 接收 numpy RGB 数组 (非 PIL, 见报错 "'Image' object has no attribute 'shape'")
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth = self.model.infer_image(rgb, input_size=self.input_size)
            return np.asarray(depth, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] 深度估计失败: {e}")
            return None

    def depth_at(self, depth_map, x: int, y: int) -> Optional[float]:
        """读取深度图 (x, y) 处的相对逆深度"""
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        return float(depth_map[y, x])

    def to_metric(self, rel: Optional[float]) -> Optional[float]:
        """相对逆深度 → 米 (Z = alpha / rel); alpha<=0 表示未标定, 返回 None"""
        if rel is None or rel <= 1e-6 or self.alpha <= 0:
            return None
        return self.alpha / rel

    def set_alpha(self, alpha: float):
        """标定: 更新尺度 α 并写回 depth.json"""
        self.alpha = float(alpha)
        self._save_config()
        print(f"[INFO] 深度标定完成: α = {self.alpha:.3f}")
