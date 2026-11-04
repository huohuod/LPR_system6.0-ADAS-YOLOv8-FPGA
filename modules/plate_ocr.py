"""车牌OCR识别模块 v4.2 - 78字符集完整映射(正确顺序)"""
import os
import cv2
import numpy as np
from typing import Tuple, List


# HyperLPR3 rec模型完整78字符集 (按模型训练顺序)
# 来源: LPR_初赛提交/modules/plate_ocr.py 原始项目验证过的字符集
# 注意: 原始项目只有77个字符，但模型输出78类，补充第78个字符
CHAR_SET = [
    "blank", "'",  # 0-1
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # 2-11
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",  # 12-23
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",  # 24-36
    # 省份和特殊字符 (37-77)
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋",  # 37-46
    "桂", "民", "沪", "津", "浙", "渝", "港", "湘", "琼", "甘",  # 47-56
    "皖", "粤", "航", "苏", "蒙", "藏", "警", "豫", "贵", "赣",  # 57-66
    "辽", "鄂", "闽", "陕", "青", "鲁", "黑", "领", "使", "澳",  # 67-76
    "危",  # 77 - 危险品车牌
]  # 共78个字符

# 省份字符集合 (不含使/领/澳 — 它们在末尾不应被前置)
PROVINCE_CHARS = set([
    "云", "京", "冀", "吉", "宁", "川", "新", "晋", "桂",
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤",
    "苏", "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽",
    "陕", "青", "鲁", "黑",
])

# 外交/特殊标记字符 (放在末尾, 不参与省份前置)
DIPLOMATIC_CHARS = set(["领", "使", "澳"])


class PlateOCR:
    """车牌OCR识别器 v4.3 - GPU 加速版"""

    def __init__(self, model_path: str = None, enable_gpu: bool = True):
        self.session = None
        self.engine_name = "None"
        self._device = 'cpu'
        self._enable_gpu = enable_gpu
        self._load_model(model_path)

    @staticmethod
    def is_gpu_available() -> bool:
        """检测 CUDAExecutionProvider 是否可用"""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            return 'CUDAExecutionProvider' in available
        except Exception:
            return False

    def _load_model(self, model_path: str = None):
        """加载ONNX模型 (优先GPU, 支持回退)"""
        try:
            import onnxruntime as ort

            if model_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(base_dir, 'models', 'hyperlpr3', 'rec.onnx')

            if not os.path.exists(model_path):
                print(f"[OCR] 模型不存在: {model_path}")
                return

            # ── 配置 Provider 选项 ──
            available_providers = ort.get_available_providers()
            print(f"[OCR] 可用执行提供器: {available_providers}")

            session_opts = ort.SessionOptions()
            session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # 限制单次推理线程数, 避免多路场景下 GPU 争抢
            session_opts.intra_op_num_threads = 2
            session_opts.inter_op_num_threads = 2
            # 抑制 ONNX Runtime 模型格式警告 (不影响推理)
            session_opts.log_severity_level = 3  # 3 = ERROR only

            if self._enable_gpu and 'CUDAExecutionProvider' in available_providers:
                # 先尝试 GPU 模式 (laptop Optimus 可能失败)
                try:
                    cuda_opts = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 512 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'HEURISTIC',
                        'do_copy_in_default_stream': True,
                    }
                    self.session = ort.InferenceSession(
                        model_path,
                        providers=[('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider'],
                        sess_options=session_opts,
                    )
                    if 'CUDAExecutionProvider' in self.session.get_providers():
                        self._device = 'cuda'
                        print(f"[OCR] 使用 CUDA 加速推理 (显存限制: 512MB)")
                    else:
                        self._device = 'cpu'
                        print(f"[OCR] CUDA 未激活, 使用 CPU 推理")
                except Exception as e:
                    print(f"[OCR] CUDA 初始化失败 ({e}), 回退 CPU...")
                    self.session = ort.InferenceSession(
                        model_path,
                        providers=['CPUExecutionProvider'],
                        sess_options=session_opts,
                    )
                    self._device = 'cpu'
                    print(f"[OCR] 使用 CPU 推理")
            else:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=session_opts,
                )
                self._device = 'cpu'
                if self._enable_gpu and 'CUDAExecutionProvider' not in available_providers:
                    print("[OCR] CUDA 不可用 (请安装 onnxruntime-gpu), 回退到 CPU 模式")
                else:
                    print("[OCR] 使用 CPU 推理")

            # 验证实际使用的 provider
            actual_providers = self.session.get_providers()
            print(f"[OCR] 实际执行提供器: {actual_providers}")

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]

            # 获取模型输出维度
            output_shape = self.session.get_outputs()[0].shape
            self.model_char_count = output_shape[-1]
            self.model_seq_len = output_shape[-2]

            # 验证字符集大小
            actual_charset_size = len(CHAR_SET)
            if actual_charset_size != self.model_char_count:
                print(f"[OCR] 警告: 字符集({actual_charset_size})与模型输出({self.model_char_count})不匹配")
            else:
                print(f"[OCR] 字符集验证通过: {actual_charset_size}类")

            # 测试推理
            test_input = np.zeros((1, 3, 48, 160), dtype=np.float32)
            _ = self.session.run(None, {self.input_name: test_input})

            self.engine_name = f"HyperLPR3-OCR ({self._device})"
            print(f"[OCR] 模型加载成功 (设备: {self._device}, 序列: {self.model_seq_len}, 字符: {self.model_char_count})")

        except Exception as e:
            print(f"[OCR] 加载失败: {e}")
            self.session = None
            self.input_name = 'images'
            self._device = 'cpu'

    def _preprocess_plate(self, plate_img: np.ndarray, enhance: bool = False) -> np.ndarray:
        """车牌图像预处理

        HyperLPR3 标准流程: resize → normalize (no CLAHE).
        CLAHE 增强可选 (默认关闭, 因为模型通常用 BGR 原图训练).

        Args:
            plate_img: 车牌裁剪图 (BGR)
            enhance: 是否启用 CLAHE 对比度增强
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # 确保是 3 通道 BGR
        if len(plate_img.shape) == 2:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(plate_img, (160, 48))

        if enhance:
            # 可选 CLAHE 增强 (仅当确定模型训练时使用了此预处理时启用)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return img

    def recognize(self, plate_img: np.ndarray, debug: bool = False) -> Tuple[str, str]:
        """识别车牌

        Args:
            plate_img: 车牌裁剪图 (BGR)
            debug: 是否打印调试信息

        Returns:
            (车牌号, 颜色名)
        """
        if plate_img is None or plate_img.size == 0:
            return '', '未知'

        if self.session is None:
            if debug:
                print("[OCR] Session 未加载, 使用回退方案")
            return self._fallback_recognize(plate_img)

        try:
            plate_number = ''

            # 两轮: 标准 → CLAHE增强, 取更好的结果
            best_plate = ''
            for enhance in (False, True):
                processed = self._preprocess_plate(plate_img, enhance=enhance)
                if processed is None:
                    continue

                plate_normalized = processed.astype(np.float32) / 255.0
                plate_chw = plate_normalized.transpose(2, 0, 1)
                input_data = np.expand_dims(plate_chw, axis=0).astype(np.float32)

                outputs = self.session.run(None, {self.input_name: input_data})
                output = outputs[0]

                plate_number = self._ctc_decode(output, min_prob=0.008)
                plate_number = self._post_process(plate_number)
                # 取更长或包含使/领的结果
                if len(plate_number) > len(best_plate):
                    best_plate = plate_number
                if any(c in plate_number for c in '使领港澳'):
                    best_plate = plate_number
                    break  # 检出外交/特殊字符, 不再跑增强轮
            plate_number = best_plate

            color = self._detect_color(plate_img)
            return plate_number, color

        except Exception as e:
            print(f"[OCR] 识别失败: {e}")
            import traceback
            traceback.print_exc()
            return '', '未知'
    
    def _ctc_decode(self, output: np.ndarray, min_prob: float = 0.012) -> str:
        """CTC 贪婪解码 — 数值稳定版

        Args:
            output: 模型原始输出 (1, seq_len, num_classes)
            min_prob: 最小字符概率阈值 (78类softmax中, 建议 0.05~0.15)

        Returns:
            解码后的车牌号字符串
        """
        logits = output[0]  # (seq_len, num_classes)

        # 数值稳定的 softmax: subtract max before exp
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        plate_number = ""
        prev_idx = -1
        char_confidences = []  # 记录每个字符的置信度, 用于诊断

        for i in range(logits.shape[0]):
            char_idx = int(np.argmax(probs[i]))  # argmax on probs (same as logits)
            char_prob = float(probs[i, char_idx])

            # blank=0 跳过; 重复字符跳过 (CTC 折叠)
            if char_idx != 0 and char_idx != prev_idx:
                if char_idx < len(CHAR_SET) and char_prob > min_prob:
                    plate_number += CHAR_SET[char_idx]
                    char_confidences.append((CHAR_SET[char_idx], char_prob))

            prev_idx = char_idx

        # 诊断: 如果完全没解码出字符, 打印 top-3 概率帮助调试
        if not plate_number:
            top_probs = probs.max(axis=-1)
            top_indices = probs.argmax(axis=-1)
            non_blank_mask = top_indices != 0
            if non_blank_mask.any():
                best_t = int(np.argmax(top_probs * non_blank_mask))
                best_idx = int(top_indices[best_t])
                best_char = CHAR_SET[best_idx] if best_idx < len(CHAR_SET) else '?'
                best_prob = float(top_probs[best_t])
                # 静默: 只在引擎打开 verbose 时打印
                # print(f"[OCR] 解码为空, 最高非blank: '{best_char}' (idx={best_idx}, prob={best_prob:.3f})")

        return plate_number
    
    def _post_process(self, plate_number: str) -> str:
        """车牌后处理 — 清洗 + 外交车牌修正"""
        if not plate_number:
            return ""

        plate_number = plate_number.replace("blank", "")
        if not plate_number:
            return ""

        # 过滤非法字符
        allowed = set(
            "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
            "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
            "领使学警挂港澳航危民"
        )
        plate_number = ''.join(c for c in plate_number if c in allowed)
        if not plate_number:
            return ""

        # ── 判断车牌类型 ──
        # 规则: 开头不是省份汉字 + 6位以上数字/字母 → 使馆/领馆黑牌
        # 开头是省份汉字 → 普通车牌
        starts_with_province = plate_number[0] in PROVINCE_CHARS if plate_number else False
        digit_like = set('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
        stripped = ''.join(c for c in plate_number if c in digit_like)
        is_diplomatic = (not starts_with_province and len(stripped) >= 6)

        if is_diplomatic:
            # 使馆/领馆车牌: 6位数字 + 使/领
            digits = ''.join(c for c in plate_number if c in '0123456789')
            digits = digits[-6:]  # 只保留最后6位

            # 检测末尾标记
            if '领' in plate_number:
                marker = '领'
            elif '澳' in plate_number and not any(
                plate_number.startswith(f'澳{c}') for c in 'ABCDEFGHJKLMNPQRSTUVWXYZ'
            ):
                marker = '使'  # "澳"被误认
            else:
                marker = '使'  # 默认使馆

            # 字母→数字还原
            char_map = {'D':'0','O':'0','Q':'0','I':'1','L':'1','Z':'2','B':'8','S':'5','G':'6','T':'7','A':'4'}
            digits = ''.join(char_map.get(c, c) for c in digits)
            plate_number = digits + marker

        else:
            # 普通车牌: 保持 OCR 原始输出, 不做省份移位
            pass

        # 连续省份字去重: "粤津A12345" → "津A12345"
        if len(plate_number) >= 2 and plate_number[0] in PROVINCE_CHARS and plate_number[1] in PROVINCE_CHARS:
            plate_number = plate_number[1:]

        # 长度过滤: <4 或 >15 明显不对
        if len(plate_number) < 4 or len(plate_number) > 15:
            return ""

        return plate_number

    def _fallback_recognize(self, plate_img: np.ndarray) -> Tuple[str, str]:
        """回退识别"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_count = len([c for c in contours if cv2.contourArea(c) > 30])
        return f'PLATE-{char_count}', self._detect_color(plate_img)

    def _detect_color(self, plate_img: np.ndarray) -> str:
        """检测车牌颜色

        直接用 HSV 色相占比投票, 不做白平衡(会洗掉黄/绿色)。
        宽色域容忍摄像头偏色。
        """
        if plate_img is None or plate_img.size == 0:
            return '未知'

        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        total = h * w

        # ── 综合五色车牌判定: 绿→黄→蓝→白→黑 ──
        v_channel = hsv[:, :, 2].astype(np.float32)
        s_channel = hsv[:, :, 1].astype(np.float32)
        h_channel = hsv[:, :, 0].astype(np.float32)

        # 绿牌优先: 渐变绿半绿半白也判绿
        green_px = np.sum((h_channel >= 35) & (h_channel <= 90) &
                          (s_channel > 20) & (v_channel > 35))
        green_r = green_px / total
        if green_r > 0.06:
            return '绿牌'

        # 蓝牌: 最常见的车牌, 宽松匹配
        blue_px = np.sum((h_channel >= 80) & (h_channel <= 155) &
                         (s_channel > 20) & (v_channel > 30))
        blue_r = blue_px / total
        if blue_r > 0.06:
            return '蓝牌'

        # 黄牌: 极严格, 必须压倒性黄色才判 (大车/挂车)
        yellow_px = np.sum((h_channel >= 15) & (h_channel <= 40) &
                           (s_channel > 70) & (v_channel > 100))
        yellow_r = yellow_px / total
        if yellow_r > 0.25:
            return '黄牌'

        # 白牌: 彩色都不中才判白
        white_px = np.sum((v_channel > 160) & (s_channel < 55))
        if white_px / total > 0.30:
            return '白牌'

        # 收尾
        dark_ratio = np.sum(v_channel < 70) / total
        if dark_ratio > 0.40:
            return '黑牌'
        scores = {'蓝牌': blue_r, '黄牌': yellow_r, '绿牌': green_r}
        best = max(scores, key=scores.get)
        if scores[best] > 0.04:
            return best
        return '黑牌'
