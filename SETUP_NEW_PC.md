# 新 PC 部署快速指南（人工版）

> 本文件与 `QWEN_DEPLOY_PROMPT.md`（给 AI 助手的详细版提示词）配套。
> 本文件是简明人工版，照命令逐行执行即可。

## 0. 你拿到的东西

```
xrobotoolkit_code.tar.gz   # 代码 + 依赖清单 + 部署文档（约 536MB）
```

内含：
- 全部代码（遥操作框架 + FR3 脚本 + fairino SDK linux 版）
- `requirements_xrobot.txt`  ← 原开发机 conda 环境 `xrobot` 导出的精确依赖（247 个包）
- `QWEN_DEPLOY_PROMPT.md`    ← 给 qwen-code 的自动部署提示词
- 本文件

## 1. 解压

```bash
tar -xzf xrobotoolkit_code.tar.gz -C ~/
cd ~/XRoboToolkit-Teleop-Sample-Python
```

## 2. 创建 conda 环境（必须 Python 3.10）

```bash
# 没有 conda 先装 Miniconda：
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh && source ~/.bashrc

conda create -n xrobot python=3.10 -y
conda activate xrobot
```

## 3. 安装依赖（联网，约 5-10 分钟）

```bash
cd ~/XRoboToolkit-Teleop-Sample-Python
pip install -r requirements_xrobot.txt
pip install -e .
```

> 注意：fairino SDK 已内嵌在 `FR3/fairino-python-sdk/linux/`，脚本运行时自动加载，
> **不要** `pip install fairino`。

### 无法联网时的离线安装（可选）

在**原开发机**（或任何能联网的同架构 Linux + Python 3.10 机器）上：

```bash
mkdir wheels
pip download -r requirements_xrobot.txt -d wheels/
# 体积约 3-4GB（含 torch CUDA 轮子），拷贝整个 wheels/ 目录到新 PC 后：
pip install --no-index --find-links=wheels -r requirements_xrobot.txt
```

## 4. 安装后验证（逐行执行，每行应输出 OK）

```bash
conda activate xrobot
cd ~/XRoboToolkit-Teleop-Sample-Python

python -c "import tyro, placo, zmq, cv2, numpy; print('core deps OK')"
python -c "import torch; print('torch', torch.__version__, 'OK')"
python -c "import pyrealsense2; print('realsense OK')"
python -c "
import sys; sys.path.insert(0, 'FR3/fairino-python-sdk/linux')
from fairino import Robot; print('fairino SDK OK')
"
python -c "from xrobotoolkit_teleop.hardware.single_arm_fr3_controller import SingleArmFR3Controller; print('package OK')"
```

任何一行报 ImportError → 对应包没装上，重跑第 3 步看报错。

## 5. VR 手柄服务（需要遥操作实机时才装）

```bash
sudo apt install -y cmake build-essential
cd ~/XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind
# 按该目录 README.md 编译（qwen-code 也会做这一步）
```

只做训练/数据处理可跳过。

## 6. 连接 FR3 实机（可选）

```bash
# 网卡配到 192.168.58.x 网段（网卡名用 ip addr 查）
sudo ip addr add 192.168.58.100/24 dev <网卡名>
ping 192.168.58.2    # 必须通
```

## 7. 常用命令速查

```bash
conda activate xrobot
cd ~/XRoboToolkit-Teleop-Sample-Python

python FR3/teleop_single_fr3_hand.py            # 遥操作 + BC 录制
python FR3/teleop_single_fr3_hand.py --reset    # 复位到 Home
python FR3/teleop_single_fr3_hand.py --save-home  # 保存当前姿态为 Home
python FR3/evoRl.py --record --task "..." --yes # VLA 推理 + RL 录制（T/F 标记成败）
python FR3/play_episode.py                      # episode 回放（MoveJ 15°/s）
python FR3/merge_rl_correction.py               # episode 拼接合并
python FR3/pad_episode_frames.py                # 末尾补 10 静止帧
```

## 8. 用 qwen-code 自动部署（可选，替代 2-5 步）

```bash
cd ~/XRoboToolkit-Teleop-Sample-Python
qwen
# 输入：请阅读项目根目录的 QWEN_DEPLOY_PROMPT.md，严格按其中步骤完成部署，
#       每步执行对应验证，最后按第 9 节完成标准输出汇总。
```

## 9. 已知坑位

1. 必须 `conda activate xrobot`，系统 Python 缺包（tyro 等）。
2. 机器人报警 主码1/子码29（关节反馈速度超限）不可软件复位，运动指令报错误码 14，
   唯一解法：控制箱断电 30 秒重启。
3. `FR3/imges/` 目录名就是拼写 imges（历史遗留），代码内写死，勿改名。
4. evoRl.py 推理服务器默认 192.168.0.104:5555，按需 `--server-ip` 修改。
5. logs/（13G 录制数据）不在本压缩包内，需要训练数据另行拷贝。
