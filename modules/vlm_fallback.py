"""
VLM (Vision Language Model) 兜底识别模块
当OCR置信度低时，将车牌图片发送给本地VLM进行二次识别

支持:
- 本地部署的 Qwen2.5-VL / InternVL 等模型
- OpenAI 兼容 API (vLLM, Ollama, LMStudio 等)

使用方式:
1. 启动本地VLM服务，例如: ollama run qwen2.5-vl:3b
2. 在 config 中启用 VLM fallback
3. OCR 识别结果置信度低于阈值时自动调用
"""
import base64
import json
import os
import urllib.request
from typing import Tuple, Optional
import numpy as np
import cv2


class VLMFallback:
    """VLM 兜底识别器"""

    def __init__(self,
                 api_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "qwen2.5-vl:3b",
                 enabled: bool = False,
                 confidence_threshold: float = 0.5):
        """
        Args:
            api_url: VLM API 地址
            model_name: 模型名称
            enabled: 是否启用
            confidence_threshold: OCR置信度低于此值时调用VLM
        """
        self.api_url = api_url
        self.model_name = model_name
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self._available = False

        if enabled:
            self._check_availability()

    def _check_availability(self):
        """检查VLM服务是否可用"""
        try:
            req = urllib.request.Request(
                self.api_url.replace('/api/generate', '/api/tags'),
                method='GET'
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._available = True
                print(f"[VLM] 服务可用: {self.api_url}")
        except Exception as e:
            self._available = False
            print(f"[VLM] 服务不可用: {e}")

    def recognize_plate(self, plate_img: np.ndarray,
                        ocr_result: str = "",
                        ocr_confidence: float = 0.0) -> Tuple[str, float]:
        """
        VLM 识别车牌

        Args:
            plate_img: 车牌裁剪图 (BGR)
            ocr_result: OCR已有结果 (作为参考)
            ocr_confidence: OCR置信度

        Returns:
            (识别结果, 置信度)
        """
        if not self.enabled or not self._available:
            return ocr_result, ocr_confidence

        if ocr_confidence >= self.confidence_threshold:
            return ocr_result, ocr_confidence

        try:
            # 编码图片为base64
            _, buf = cv2.imencode('.jpg', plate_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(buf).decode('utf-8')

            prompt = (
                "这是中国车牌图片。请只输出车牌号码，不要输出其他任何内容。"
                "格式示例: 京A12345, 粤B88888, 沪C00001"
            )

            # Ollama API 格式
            payload = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50}
            }).encode('utf-8')

            req = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                vlm_text = data.get('response', '').strip()

            # 清理结果：只保留车牌号格式的字符
            vlm_result = self._clean_plate_text(vlm_text)

            if vlm_result and len(vlm_result) >= 5:
                return vlm_result, 0.7  # VLM给一个中等置信度

            return ocr_result, ocr_confidence

        except Exception as e:
            print(f"[VLM] 识别失败: {e}")
            return ocr_result, ocr_confidence

    def _clean_plate_text(self, text: str) -> str:
        """清理VLM输出，提取车牌号"""
        import re
        # 匹配中国车牌格式
        pattern = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-Z0-9]{4,6}'
        match = re.search(pattern, text)
        if match:
            return match.group(0)
        # 回退：去掉非字母数字中文的字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text)
        return cleaned[:10] if len(cleaned) >= 5 else ""


# 全局实例
_vlm_instance: Optional[VLMFallback] = None


def get_vlm_instance(config=None) -> VLMFallback:
    """获取VLM单例"""
    global _vlm_instance
    if _vlm_instance is None:
        api_url = "http://localhost:11434/api/generate"
        model_name = "qwen2.5-vl:3b"
        enabled = False
        threshold = 0.5

        if config and hasattr(config, 'vlm'):
            api_url = getattr(config.vlm, 'api_url', api_url)
            model_name = getattr(config.vlm, 'model_name', model_name)
            enabled = getattr(config.vlm, 'enabled', enabled)
            threshold = getattr(config.vlm, 'confidence_threshold', threshold)

        _vlm_instance = VLMFallback(
            api_url=api_url,
            model_name=model_name,
            enabled=enabled,
            confidence_threshold=threshold,
        )
    return _vlm_instance
