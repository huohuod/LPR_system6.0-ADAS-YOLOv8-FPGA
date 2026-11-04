"""多帧车道标记 — 在多个帧上标记车道线4角, 取中位数得到稳健透视标定

用法:
    python mark_lanes.py <视频路径>

每帧点选顺序(逆时针):
    1. 左车道线 近处(画面下方)
    2. 右车道线 近处(画面下方)
    3. 右车道线 远处(消失点附近)
    4. 左车道线 远处(消失点附近)

操作:
    左键 = 点一个点
    右键 / u = 撤销上一个点
    r = 清空当前帧的点
    n = 前进10帧   p = 后退10帧   (找路直、线清楚、没车挡的帧)
    s = 保存当前帧的4个点, 并前进10帧
    d = 全部完成: 保存所有标记 + 取中位数写入 lane.json
    Esc = 退出
"""
import sys
import os
import json
import cv2
import numpy as np

points = []
all_marks = []
SCALE = 1.0


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        ox, oy = int(x / SCALE), int(y / SCALE)
        points.append([ox, oy])
        print(f"  已点 {len(points)}/4: ({ox}, {oy})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            print(f"  撤销, 剩 {len(points)} 个点")


def main():
    global SCALE
    if len(sys.argv) < 2:
        print("用法: python mark_lanes.py <视频路径>")
        return
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        print("[ERROR] 读取失败")
        return
    frame_idx = 0
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 读取帧失败")
        return

    def make_disp(f):
        global SCALE
        h, w = f.shape[:2]
        SCALE = 1280 / w if w > 1280 else 1.0
        return cv2.resize(f, (int(w * SCALE), int(h * SCALE)))

    disp = make_disp(frame)
    print("左键点4个点(左线近→右线近→右线远→左线远)")
    print("n/p=翻帧  s=保存本帧并前进10帧  d=完成  r=清空  Esc=退出")

    cv2.namedWindow("mark")
    cv2.setMouseCallback("mark", on_mouse)

    while True:
        img = disp.copy()
        for i, p in enumerate(points):
            pp = (int(p[0] * SCALE), int(p[1] * SCALE))
            cv2.circle(img, pp, 5, (0, 255, 0), -1)
            cv2.putText(img, str(i + 1), (pp[0] + 8, pp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                a = (int(points[i][0] * SCALE), int(points[i][1] * SCALE))
                b = (int(points[i + 1][0] * SCALE), int(points[i + 1][1] * SCALE))
                cv2.line(img, a, b, (255, 255, 0), 1)
        cv2.putText(img, f"frame {frame_idx}  已存 {len(all_marks)} 帧",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("mark", img)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            return
        if k == ord('r'):
            points.clear()
            print("  已清空当前帧")
        if k == ord('u'):
            if points:
                points.pop()
                print(f"  撤销, 剩 {len(points)} 个点")
        if k in (ord('n'), ord('p')):
            frame_idx += 10 if k == ord('n') else -10
            frame_idx = max(0, frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                points.clear()
                disp = make_disp(frame)
                print(f"  跳到第 {frame_idx} 帧")
        if k == ord('s') and len(points) == 4:
            all_marks.append({'frame': frame_idx, 'pts': [p[:] for p in points]})
            print(f"  已保存第 {frame_idx} 帧 (共 {len(all_marks)} 帧)")
            points.clear()
            frame_idx += 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                disp = make_disp(frame)
                print(f"  → 跳到第 {frame_idx} 帧")
        if k == ord('d'):
            break

    if len(all_marks) < 3:
        print(f"[WARN] 只标了 {len(all_marks)} 帧, 建议至少 5 帧, 未保存")
        return

    # 取中位数
    arr = np.array([m['pts'] for m in all_marks], dtype=np.float32)  # (N,4,2)
    median = np.median(arr, axis=0)  # (4,2)

    out_marks = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'lane_marks.json')
    out_lane = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'lane.json')
    os.makedirs(os.path.dirname(out_marks), exist_ok=True)
    with open(out_marks, 'w', encoding='utf-8') as f:
        json.dump({'marks': all_marks}, f, indent=2)
    with open(out_lane, 'w', encoding='utf-8') as f:
        json.dump({'src_pts': median.tolist()}, f, indent=2)
    print(f"\n已保存 {len(all_marks)} 帧标记 → {out_marks}")
    print(f"中位数透视点 → {out_lane}")
    print(json.dumps({'src_pts': median.tolist()}, indent=2))


if __name__ == '__main__':
    main()
