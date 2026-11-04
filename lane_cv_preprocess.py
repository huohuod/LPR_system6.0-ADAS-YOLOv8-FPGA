# -*- coding: utf-8 -*-
"""传统 CV 车道检测 + 预处理测试: 只测 Q3/Q4, 看去摩尔纹能否让检测更稳定
用法: python lane_cv_preprocess.py
"""
import sys
import cv2

sys.path.insert(0, '.')
from src.utils.quad_splitter import QuadSplitter
from src.modules.lane_detector import LaneDetector

VIDEO = "C:/Users/Moxiaojiu/Desktop/Desktop 2026.11.01 - 18.17.20.02.mp4"


def p_orig(f):
    return f

def p_gauss5(f):
    return cv2.GaussianBlur(f, (5, 5), 0)

def p_gauss9(f):
    return cv2.GaussianBlur(f, (9, 9), 0)

def p_median(f):
    return cv2.medianBlur(f, 5)

def p_downup(f):
    h, w = f.shape[:2]
    return cv2.resize(cv2.resize(f, (w // 2, h // 2)), (w, h))

def p_gauss_downup(f):
    return p_downup(p_gauss5(f))


PREPROC = {
    '1原始': p_orig,
    '2高斯5': p_gauss5,
    '3高斯9': p_gauss9,
    '4中值5': p_median,
    '5下采样2x': p_downup,
    '6高斯+下采样': p_gauss_downup,
}


def main():
    det = LaneDetector()
    splitter = QuadSplitter()
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"读不到视频: {VIDEO}")
        return

    N = 30
    frames = []
    for _ in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"共 {len(frames)} 帧, 每帧测 Q3(左下)+Q4(右下) 两个象限\n")

    print(f"{'预处理':<12} {'检测到(任一)':>10} {'左线':>6} {'右线':>6}")
    print('-' * 38)
    for name, fn in PREPROC.items():
        n_det = n_l = n_r = 0
        for frame in frames:
            quads = splitter.split(frame)
            for i in (2, 3):  # 左下=Q3, 右下=Q4
                det._prev_left = None
                det._prev_right = None
                r = det.detect(fn(quads[i]))
                if r.detected:
                    n_det += 1
                if r.left:
                    n_l += 1
                if r.right:
                    n_r += 1
        total = len(frames) * 2
        print(f"{name:<12} {n_det:>7}/{total} {n_l:>6} {n_r:>6}")


if __name__ == '__main__':
    main()
