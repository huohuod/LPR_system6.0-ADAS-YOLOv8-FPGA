"""配置管理系统"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """模型配置"""
    plate_detector: str = "plate_detector.pt"
    vehicle_detector: str = "yolov8n.pt"
    traffic_light_detector: str = "traffic_light.pt"
    pedestrian_detector: str = "pedestrian.pt"
    crosswalk_detector: str = "crosswalk.pt"
    hyperlpr3_detect: str = "detect.onnx"
    hyperlpr3_rec: str = "rec.onnx"
    device: str = "0"
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    img_size: int = 640
    fp16: bool = True


@dataclass
class PipelineConfig:
    """流水线配置"""
    max_fps: int = 30
    frame_queue_size: int = 3
    worker_threads: int = 2
    enable_preprocess: bool = True
    enable_tracking: bool = True
    enable_traffic_light: bool = True
    enable_pedestrian: bool = True
    enable_violation: bool = True
    enable_multiscale: bool = True


@dataclass
class VLMConfig:
    """VLM兜底识别配置"""
    api_url: str = "http://localhost:11434/api/generate"
    model_name: str = "qwen2.5-vl:3b"
    enabled: bool = False
    confidence_threshold: float = 0.5


@dataclass
class ColorConfig:
    """颜色配置"""
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    wb_r: int = 0
    wb_g: int = 0
    wb_b: int = 0
    enable_histogram_match: bool = False


@dataclass
class SystemConfig:
    """系统总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    source: int = 0
    gui_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            color=ColorConfig(**data.get("color", {})),
            vlm=VLMConfig(**data.get("vlm", {})),
            source=data.get("source", 0),
            gui_enabled=data.get("gui_enabled", True),
        )


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent.parent / "configs" / "system.json"
        self.config = SystemConfig()
        self._load()

    def _load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.config = SystemConfig.from_dict(data)
            except Exception as e:
                print(f"[CONFIG] 加载失败: {e}")

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
