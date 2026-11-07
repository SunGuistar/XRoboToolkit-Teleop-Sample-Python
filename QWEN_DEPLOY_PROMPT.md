# 任务：在本机部署 XRoboToolkit-Teleop-Sample-Python（FR3 遥操作 + VLA/RL 数据采集项目）

请严格按照以下步骤在本机完成项目部署，不要跳步。每完成一步做对应验证，全部完成后给出验证结果汇总。

## 1. 项目背景

本项目是一套 XR 遥操作框架：通过 PICO VR 手柄遥操作 FAIRINO FR3 协作机械臂，支持 BC（行为克隆）数据采集、VLA 模型推理（ZMQ 远程服务）、episode 回放等。原开发机为 Ubuntu 22.04，conda 环境名 `xrobot`，Python 3.10.20。

核心代码链路：
```
FR3/teleop_single_fr3_hand.py（遥操作入口）
  → xrobotoolkit_teleop/hardware/single_arm_fr3_controller.py（FR3 控制器：placo IK + ServoJ）
      → FR3/fairino-python-sdk/linux/fairino（法奥 SDK，运行时 sys.path 动态插入）
  → xrobotoolkit_teleop/common/xr_client.py（VR 手柄客户端，依赖 XRoboToolkit PC Service）
```

## 2. 系统要求

- Ubuntu 22.04（其他版本需自行适配，未测试）
- Miniconda 或 Anaconda
- 网卡：需要一块网卡与 FR3 控制箱同网段（192.168.58.x，控制箱固定 IP 192.168.58.2）
- 可选：Intel RealSense 相机（D435/D435I），遥操作录制需要；无相机可用 `--no-camera` 仅遥操作

## 3. 环境搭建（在项目根目录执行）

```bash
# 方式 A（推荐）：pip 精确复现
# 1. 创建 conda 环境（必须 Python 3.10）
conda create -n xrobot python=3.10 -y
conda activate xrobot

# 2. 安装依赖（项目自带精确版本清单，共 247 个包）
pip install -r requirements_xrobot.txt

# 方式 B（备选）：conda 规格文件整环境重建（含 conda 层包，较慢）
# conda env create -f environment_xrobot.yml

# 3. 以可编辑模式安装本包（注册 xrobotoolkit_teleop 包）
pip install -e .
```

关键包版本（requirements_xrobot.txt 已含，列出来供核对）：

| 包 | 版本 | 用途 |
|---|---|---|
| placo | 0.9.23 | 逆运动学求解 |
| mujoco | 3.10.0 | 仿真 |
| torch | 2.7.1 | BC 训练/推理 |
| tyro | 1.0.13 | CLI 参数解析（teleop 脚本必需） |
| pyrealsense2 | 2.58.1.10581 | RealSense 相机 |
| pyzmq | 27.1.0 | evoRl.py ZMQ 推理通信 |
| opencv-python | 4.13.0.92 | 图像处理 |
| meshcat | 0.3.2 | IK 可视化（可选） |
| dex_retargeting | 0.5.0 | 灵巧手重定向 |
| numpy | 2.2.6 | — |
| scipy | 1.15.3 | — |

注意：
- `requirements_xrobot.txt` 中不含 fairino SDK —— 它内嵌在仓库 `FR3/fairino-python-sdk/linux/` 中，脚本运行时自动加入 `sys.path`，**不要 pip 安装 fairino**。
- 若新 PC 无法联网，按仓库根目录 `SETUP_NEW_PC.md` 第 3 节的离线安装说明（pip download 预下载 wheels 后拷贝）。
- 详细人工操作步骤见仓库根目录 `SETUP_NEW_PC.md`。

## 4. 安装后验证

```bash
conda activate xrobot
cd <项目根目录>

# 验证核心依赖
python -c "import tyro, placo, zmq, cv2, numpy; print('core deps OK')"
python -c "import pyrealsense2; print('realsense OK')"   # 无相机也能 import
python -c "import torch; print('torch', torch.__version__)"

# 验证 fairino SDK（不连接机器人，只验证可导入）
python -c "
import sys
sys.path.insert(0, 'FR3/fairino-python-sdk/linux')
from fairino import Robot
print('fairino SDK OK')
"

# 验证项目包
python -c "from xrobotoolkit_teleop.hardware.single_arm_fr3_controller import SingleArmFR3Controller; print('package OK')"
```

## 5. VR 手柄服务（XRoboToolkit PC Service，遥操作必需）

遥操作依赖 `dependencies/XRoboToolkit-PC-Service-Pybind/` 中的 PC Service（VR 手柄数据源）。按其目录内 README.md / setup_orin.sh 编译安装（需要 CMake + C++ 编译器：`sudo apt install cmake build-essential`）。仅做离线代码阅读/训练则可跳过此步。

## 6. 网络配置（连接 FR3 实机时）

```bash
# 给与控制箱相连的网卡配置静态 IP（网卡名按实际替换，如 enp2s0）
sudo ip addr add 192.168.58.100/24 dev <网卡名>
ping 192.168.58.2   # 必须通
```

## 7. 主要运行命令

```bash
conda activate xrobot

# FR3 遥操作（VR 手柄 + 双 RealSense，BC 录制 10Hz）
python FR3/teleop_single_fr3_hand.py

# 仅复位机械臂到 Home
python FR3/teleop_single_fr3_hand.py --reset

# 保存当前机械臂姿态为新 Home
python FR3/teleop_single_fr3_hand.py --save-home

# 无相机遥操作
python FR3/teleop_single_fr3_hand.py --no-camera

# VLA 推理 + RL 数据录制（ZMQ 连推理服务器 192.168.0.104:5555）
# 录制中按 T=成功保存 / F=失败保存
python FR3/evoRl.py --record --task "pick up the object" --yes

# episode 回放（默认 MoveJ 15°/s，无安全确认）
python FR3/play_episode.py

# episode 合并处理（fail+raw 拼接 → Rl_correction → 合并到 RL 目录）
python FR3/merge_rl_correction.py

# episode 末尾补 10 帧静止帧
python FR3/pad_episode_frames.py
```

Home 姿态存于 `FR3/teleop_single_fr3_hand.home.json`（仅 `--save-home` 时更新）。

## 8. 已知坑位（必读）

1. **必须用 xrobot 环境**：系统 Python 缺 tyro 等包，直接 `python FR3/teleop_single_fr3_hand.py` 会报 `ModuleNotFoundError: No module named 'tyro'`。
2. **报警 1-29（关节反馈速度超限）为不可复位故障**：`ResetAllError()` 清不掉，所有运动指令返回错误码 14（接口执行失败）。唯一解法：控制箱断电 30 秒后重启。控制器代码已在连接阶段提前拦截并提示。
3. **MoveJ 错误码 14** 通用含义是"接口执行失败——web 界面有故障"，先查 `GetRobotErrorCode()`。
4. `FR3/imges/` 目录名就是这个拼写（不是 images），脚本内写死，勿改名。
5. logs/（13G 录制数据）与 datasets/ 默认不在代码包内，如需训练数据另行拷贝。
6. evoRl.py 的推理服务器地址默认 `192.168.0.104:5555`（`--server-ip/--server-port` 可改），服务器端不在本仓库内。

## 9. 完成标准

依次输出以下验证结果：
1. `conda activate xrobot` 后所有第 4 节 import 测试通过
2. `python -c "from xrobotoolkit_teleop.hardware.fr3_config import DEFAULT_ROBOT_IP; print(DEFAULT_ROBOT_IP)"` 输出 192.168.58.2
3. `python FR3/play_episode.py --dry-run --episode <任意存在的pkl>` 能正常加载（无实机时用 --dry-run 验证到加载层即可；若 logs 未拷贝则跳过）
4. 汇总：环境名、Python 版本、关键包版本表、未完成项及原因
