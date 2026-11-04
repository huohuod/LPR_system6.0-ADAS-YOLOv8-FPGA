"""
轻量级红绿灯CNN分类器
输入: 60x32x3 BGR图像
输出: 3类 (red/yellow/green)

架构参考经典LeNet风格，极轻量，推理<1ms
"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── 模型架构 ──────────────────────────────────────────────

if HAS_TORCH:
    class TrafficLightNet(nn.Module):
        """轻量级红绿灯分类网络
        输入: (B, 3, 60, 32)
        输出: (B, 3) logits
        """
        def __init__(self):
            super().__init__()
            # 3x60x32 -> 6x28x14 -> 16x12x5
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            # 16*12*5 = 960
            self.fc1 = nn.Linear(16 * 12 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 3)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # -> 6x28x14
            x = self.pool(F.relu(self.conv2(x)))  # -> 16x12x5
            x = x.view(-1, 16 * 12 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


# ── 分类器封装 ────────────────────────────────────────────

class TrafficLightCNNClassifier:
    """CNN红绿灯分类器"""

    LABELS = ['red', 'yellow', 'green']
    INPUT_SIZE = (32, 60)  # (w, h)

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.available = False

        if not HAS_TORCH:
            print("[TL-CNN] PyTorch 不可用，CNN 分类器未加载")
            return

        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base, 'models', 'traffic_light_cnn.pth')

        if not os.path.exists(model_path):
            print(f"[TL-CNN] 模型不存在: {model_path}，使用 HSV 回退")
            return

        try:
            self.model = TrafficLightNet()
            state = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()

            # GPU 时启用 cuDNN 自动调优
            if device != 'cpu':
                torch.backends.cudnn.benchmark = True

            self.available = True
            dev_name = device.upper() if device == 'cpu' else f"GPU ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else device})"
            print(f"[TL-CNN] CNN 分类器加载成功 → {dev_name}")
        except Exception as e:
            print(f"[TL-CNN] 加载失败: {e}")
            self.model = None

    def classify(self, img: np.ndarray) -> Tuple[str, float]:
        """
        分类交通灯颜色

        Args:
            img: 交通灯区域裁剪图 (BGR)

        Returns:
            (颜色名, 置信度)
        """
        if img is None or img.size == 0:
            return 'unknown', 0.0

        if not self.available or self.model is None:
            return self._hsv_fallback(img)

        try:
            # 预处理
            resized = cv2.resize(img, self.INPUT_SIZE)
            blob = resized.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)  # HWC -> CHW
            tensor = torch.from_numpy(blob).unsqueeze(0).to(self.device)

            # 推理 (inference_mode 比 no_grad 更快, PyTorch >= 1.9)
            with torch.inference_mode():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, idx = probs.max(1)

            label = self.LABELS[idx.item()]
            confidence = conf.item()
            return label, confidence

        except Exception as e:
            print(f"[TL-CNN] 推理失败: {e}")
            return self._hsv_fallback(img)

    def _hsv_fallback(self, img: np.ndarray) -> Tuple[str, float]:
        """HSV回退方案"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total = h * w

        ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            'yellow': [([15, 100, 100], [35, 255, 255])],
            'green': [([35, 80, 80], [85, 255, 255])],
        }

        scores = {}
        for color, rgs in ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in rgs:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            scores[color] = np.sum(mask > 0) / total

        if max(scores.values()) < 0.05:
            return 'unknown', 0.0

        best = max(scores, key=scores.get)
        return best, min(scores[best] * 5, 1.0)


# ── 训练脚本入口 ──────────────────────────────────────────

def train_model(data_dir: str, epochs: int = 30, batch_size: int = 32, lr: float = 0.001):
    """
    训练红绿灯分类器

    数据目录结构:
        data_dir/
            red/    # 红灯图片
            yellow/ # 黄灯图片
            green/  # 绿灯图片

    每个文件夹中放对应颜色的交通灯裁剪图即可
    """
    if not HAS_TORCH:
        print("需要 PyTorch")
        return

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((60, 32)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[TRAIN] 数据集: {len(dataset)} 张, 类别: {dataset.classes}")

    model = TrafficLightNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        avg_loss = total_loss / len(loader)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")

    # 保存
    save_path = os.path.join(os.path.dirname(data_dir), 'traffic_light_cnn.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[TRAIN] 模型已保存: {save_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        train_model(sys.argv[1])
    else:
        print("用法: python traffic_light_cnn.py <数据目录>")
        print("数据目录结构: red/ yellow/ green/ 各放对应图片")
