# ADAS 开发进度状态

> 更新时间：2026-10-25（下次接着从这里做）

## 一、当前状态：车道检测已换成 UFLD 深度学习（CULane）

**结论**：CV 方案（Hough/透视+滑窗）全部失败，已换 **UFLD ONNX 模型**，用 **CULane**（城市道路）版本，效果比 TuSimple 好很多。

- 模型文件：`models/ufld/culane_18.onnx`（已下载）
- 检测器：`src/modules/ufld_lane_detector.py`
- 测试脚本：`run_ufld.py`（视频路径已内置，直接 `python run_ufld.py`）
- 旧 CV 检测器：`src/modules/lane_detector.py`（已废弃，保留参考）
- FCWS 碰撞预警：`src/modules/collision_warning.py`（三级 Normal/Prompt/Warning）
- 诊断脚本：`debug_ufld.py`

## 二、已实现的功能

- **LDWS**：offset（偏右负/偏左正）、曲率 R、Good Lane Keeping / Please Keep Left·Right
- **LKAS**：Keep Straight / Gentle·Hard Left·Right Curve Ahead（基于远近车道中心偏移）
- **FCWS**（在主 LPR 引擎里）：三级风险 Normal/Prompt/Warning Risk + TTC
- 时序平滑：中值滤波（最近 5 帧取中位数）
- 截断：只画检测到的部分（不往远处外推）
- 交叉截断：两线交叉就砍掉交叉点以上
- 曲率限制：二次项过大退回线性（避免发散交叉）
- 转弯不识别：急弯（Hard Curve）时不画
- CLAHE 对比度增强 + 图像处理三视图（灰度/二值/边缘，`SHOW_PROCESS` 开关）

## 三、待办 / 未解决的问题

1. **模型跑在 CPU**：`实际执行提供器: ['CPUExecutionProvider']`，CUDA 没生效（opset 11 的 ONNX 不支持 GPU），导致卡顿。需要换 GPU 能跑的 ONNX（更高 opset）或转模型格式。
2. **显示截止到前车距离**：需要「前车检测+测距」（在 LPR 引擎里有），standalone 车道检测里没有。集成到主系统时做。
3. 车道线「平行不交叉、转弯不识别」等规则已加，**待用户实测确认**。

## 四、下一步（按顺序）

1. 实测当前 UFLD 效果（平行/交叉/转弯/稳定性），微调参数
2. 把 UFLD 车道检测**集成进 LPR 引擎**（main.py），和 FCW 测距合起来 → 实现「显示截止到前车距离」
3. 解决 GPU 加速（换模型或转格式）
4. 补 LKA（车道保持辅助，复用车道检测 + 曲率）

## 五、测试视频

- 路径：`C:\Users\Moxiaojiu\Desktop\ceshi43\video_20260929_112507.mp4`
- 类型：行车记录仪第一视角 1920x1080，路口+斑马线

## 六、环境注意

- **本环境无法运行 shell / 联网**（`.claude/session-env` EPERM），只能读写文件；命令需用户在 Git Bash 自己跑。
- WebSearch 可用，WebFetch 被墙，图片读取（Read 图片）返回 Unsupported Image（看不到画面，靠用户文字反馈 + 数据诊断）。
- 车辆测距已完成（Depth Anything V2，α 已标定，见 configs/depth.json）。
