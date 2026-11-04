"""UFLD 车道检测测试 — 免标定深度学习模型
用法: python run_ufld.py [视频路径]
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.modules.ufld_lane_detector import UFLDLaneDetector


def make_process_view(frame):
    """生成图像处理三视图: 灰度化 | 二值化 | 边缘检测"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return np.hstack([gray_bgr, binary_bgr, edges_bgr])

MODEL = "models/ufld/culane_18.onnx"
VIDEO = "C:/Users/Moxiaojiu/Desktop/ceshi43/video_20260929_112507.mp4"
SHOW_PROCESS = False  # 设为 True 显示灰度/二值/边缘三视图 (较慢, 演示用)


def main():
    if not os.path.exists(MODEL):
        print(f"[ERROR] 模型不存在: {MODEL}")
        print('请先下载: curl -L -o models/ufld/culane_18.onnx '
              '"https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/culane_18.onnx"')
        return
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    det = UFLDLaneDetector(MODEL)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video}")
        return
    print("[INFO] 按 q 退出")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = det.detect(frame)
        cv2.imshow("UFLD Lane Detection", result.frame)
        if SHOW_PROCESS:
            proc = make_process_view(frame)
            cv2.imshow("Image Process (灰度|二值|边缘)", proc)
        idx += 1
        if idx % 15 == 0:
            print(f"[frame {idx}] offset={result.vehicle_offset:+.0f}px "
                  f"R={result.curvature_radius:.0f} LDWS={result.ldws} LKAS={result.lkas} "
                  f"L={'Y' if result.left else '-'} R={'Y' if result.right else '-'}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
