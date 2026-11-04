"""
LPR System v6.0 - 统一深色科技 UI

运行方式:
    python main.py                # 默认启动 v6 竞赛版 UI
    python main.py --v5           # 经典 PySide6 UI (v5.3)
    python main.py --source 1     # 指定摄像头
    python main.py --device 0     # 指定GPU
    python main.py --no-gui       # 无GUI模式
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import ConfigManager
from src.core.engine import LPREngine


def run_gui(engine: LPREngine, source: int = 0, config_manager=None, use_v5: bool = False) -> None:
    try:
        from PySide6.QtWidgets import QApplication, QSplashScreen
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap, QFont, QColor

        app = QApplication(sys.argv)

        splash_pix = QPixmap(400, 200)
        splash_pix.fill(QColor(15, 17, 23))
        splash = QSplashScreen(splash_pix)
        splash.setFont(QFont("Microsoft YaHei", 12))
        splash.showMessage("LPR System v6.0 加载中...", Qt.AlignCenter, QColor(0, 212, 170))
        splash.show()
        app.processEvents()

        from src.pipeline.async_pipeline import AsyncPipeline

        pipeline_cfg = engine.config.pipeline if hasattr(engine.config, 'pipeline') else None
        pipeline = AsyncPipeline(engine, pipeline_cfg) if pipeline_cfg else None

        if use_v5:
            from src.ui.main_window import MainWindow
        else:
            from src.ui.main_window_v6 import MainWindow

        window = MainWindow(engine=engine, pipeline=pipeline,
                           config_manager=config_manager, source=source)

        splash.finish(window)
        window.show()

        sys.exit(app.exec())

    except ImportError as e:
        print(f"[ERROR] PySide6未安装: {e}")
        print("[TIP] 请运行: pip install PySide6")
        sys.exit(1)


def run_flet_ui(engine: LPREngine, config_manager=None) -> None:
    from src.ui.app_flet import run_flet_ui as _run
    _run(engine=engine, config_manager=config_manager)


def run_headless(engine: LPREngine, source: int = 0) -> None:
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头: {source}")
        return

    print(f"[INFO] 按Q键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, detections, _ = engine.detect(frame)

        fps = engine.get_fps()
        cv2.putText(result, f"FPS: {fps:.1f} | Latency: {engine.stats['avg_latency']:.1f}ms",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("LPR System v5.2", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="LPR System v5.2 - 性能优化版")
    parser.add_argument('--source', type=int, default=0, help='摄像头索引')
    parser.add_argument('--device', type=str, default='0', help='GPU设备号')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--no-gui', action='store_true', help='无GUI模式')
    parser.add_argument('--v5', action='store_true', help='使用 v5.3 经典 UI')
    parser.add_argument('--enable-tilt', action='store_true', default=False,
                       help='启用倾斜校正 (降低帧率但提高倾斜车牌识别率)')
    args = parser.parse_args()

    config = ConfigManager()
    config.config.model.device = args.device
    config.config.model.conf_threshold = args.conf

    engine = LPREngine(config.config, enable_tilt_correction=args.enable_tilt)

    if not engine.initialize():
        print("[FATAL] 引擎初始化失败!")
        sys.exit(1)

    if args.no_gui:
        run_headless(engine, args.source)
    else:
        run_gui(engine, args.source, config_manager=config, use_v5=args.v5)


if __name__ == '__main__':
    main()
