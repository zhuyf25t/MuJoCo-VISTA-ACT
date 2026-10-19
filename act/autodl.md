# AutoDL：安装、看训练进度、用 checkpoint 录像

以下命令在实例的 **Linux 终端**、项目根目录 `/root/autodl-tmp/MuJoCo_Brain` 执行。使用 Python 3.12 的兼容镜像，通过 `python -m act.train --help` 查看训练参数。数据划分、训练及评估流程见 [ACT README](README.md)。

## 1. 系统依赖和 Python 依赖分开安装

Ubuntu 镜像准备无窗口 MuJoCo 渲染和可断线的终端会话：

```bash
apt-get update
apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri tmux
```

`apt-get update` 只更新软件包索引，不会升级整个系统；`apt-get install` 才安装软件。这组系统依赖不能写入 pip 的 requirements.txt。已安装时无需每次重复，纯训练读取 HDF5 不依赖这些 EGL 渲染库。`tmux` 是可选的会话管理器，下面会说明。

新建隔离的 Python 环境，然后先安装匹配主机驱动的 CUDA wheel，再安装整个项目的 Python 依赖：

```bash
cd /root/autodl-tmp/MuJoCo_Brain
python --version
nvidia-smi
python -m venv /root/autodl-tmp/act-env
source /root/autodl-tmp/act-env/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

上述 cu128 wheel 要求主机驱动兼容 CUDA 12.8；其它组合参见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。不需要另外安装 torchaudio，也不要在容器里重装 NVIDIA 驱动。若 torch 已装成同版本 CPU wheel，显式卸载 torch/torchvision 后重装上述 CUDA 组合；只看包名或版本主号不能确认 GPU 可用。

如果 `python -m venv` 提示缺少 ensurepip/venv，在系统 Python 3.12 对应的 Ubuntu 仓库中安装 `python3.12-venv`；若使用镜像自带 Conda Python，先确认 `which python`，不要盲装另一个 Python 版本来混用。正常情况下无需额外执行这一条。

requirements 已包含当前源码直接使用的 ACT 依赖 torch、torchvision、PyYAML、tqdm，以及 MuJoCo、NumPy、SciPy、h5py、Pillow、imageio、imageio-ffmpeg 和采集侧依赖。glfw/PyOpenGL 等由 MuJoCo 依赖自动安装，标准库不写入 requirements。固定直接依赖不等于锁定所有传递依赖；新环境安装后仍应 `pip check` 并保存 `pip freeze`。

标准平台的 imageio-ffmpeg wheel 自带 FFmpeg 程序，因此这条 MP4 生成链路通常不需要再 apt 安装 ffmpeg。可确认：

```bash
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
python -m pip freeze > /root/autodl-tmp/act-environment.txt
```

依据：[imageio-ffmpeg 官方安装说明](https://github.com/imageio/imageio-ffmpeg)。驱动、网络和系统库需要在使用的实例上检查。

## 2. train 的输出怎样看

数据应先由 `act.prepare` 冻结到新的实验目录。以下假设数据已经上传到 `act-upload-002/dataset`，且准备元数据位于 `output/act/exp_002`：

```bash
python -u -m act.train \
  --dataset-root /root/autodl-tmp/act-upload-002/dataset \
  --experiment output/act/exp_002 \
  --config act/configs/baseline.yaml \
  --epochs 2 --device cuda
```

终端底部会动态更新，显示该阶段 batch 进度、速度、耗时、ETA 和 loss。每个阶段完成后保留最终一行，然后显示中文每轮摘要。示意数值如下，不是新的训练结果：

```text
Train 01/2:  50%|...| 472/943 [..., train_loss=0.4321, train_L1=0.3012, val_loss=pending]
Valid 01/2:  40%|...| 113/283 [..., train_loss=0.4123, val_loss=0.2789]
第 1/2 轮验证完成，正在保存 last.pt 和 best.pt…
第 1/2 轮 | ... 秒 | 累计更新 ... 次
  训练总 loss=... | 训练动作 L1=... | 验证混合损失=... | 验证动作 L1=...
```

`train_loss` 与 `val_loss` 均为四头混合损失（肩肘 L1、夹爪 CE、自动类别加权底盘 CE、真实转向位置幅度 L1），当前版本无 KL。best.pt 按 val_loss 选择；五维 L1 和逐类指标用于诊断。本分支只支持新混合模型 checkpoint，旧模型权重不可用于训练或评估；完整说明见 [ACT README](README.md)。

`--progress auto` 默认：交互终端用动态条；重定向文件、`tee` 或非交互终端中，约每 10 秒在下一个 batch 完成时追加一行进展，阶段开始/结束也会报告。它不是独立后台心跳，若单个 batch 或磁盘读取阻塞，仍需等当前 batch 返回。

`--progress always` 强制动态条，适合不正确识别终端的界面；`--progress never` 只显示阶段提示和每轮摘要。文件 `metrics.jsonl` 保留完整数值，新增 `train_loss` 字段，旧字段与从 0 开始的 epoch 编号保持不变。数据校验、网络初始化和 checkpoint 保存也会先打印提示。

## 3. 一个 checkpoint、一个 seed、一段 MP4

工具已经存在：`python -m act.rollout --video`。MuJoCo 负责推进仿真与渲染，rollout 负责加载策略、按 seed reset、执行动作，再用 imageio/FFmpeg 写成 MP4。不需要录屏或云端桌面。

在同一个已激活 Python 环境的 Linux 终端执行：

```bash
export MUJOCO_GL=egl
python -m act.rollout \
  --checkpoint output/act/exp_002/best.pt \
  --seeds 7 \
  --output output/act/exp_002/video_seed7_run1 \
  --device cuda --max-steps 600 --video --label validation
```

替换 checkpoint、seed 和输出路径。输出目录必须尚不存在；再次运行改为 run2。seed 7 仅为格式示例，正式验证从自己的 val 列表选，`--label` 不会自动核查数据归属。

输出目录含 `seed-7.mp4`（10 FPS 俯视录像）、`seed-7.json`（动作轨迹）、`summary.json`（成功率等）。默认最多 600 拍=60 秒控制，正常结束后再静置 3 秒；提前成功时更短。直接下载 MP4 用普通视频播放器播放。视频是策略在仿真中的新执行，不是专家示范回放。

需要时间集成可增加 `--ensemble-coefficient 0.01`；不传就是关闭。云端无窗口时不加 `--gui`。本地 Windows 可以用 `--device cpu --gui --video`，不设置 Linux 的 MUJOCO_GL=egl；CPU 推理可能慢于仿真时钟。

若 EGL 报缺少系统库，先确认第 1 节 Mesa/EGL 包安装成功。如果报 Conda libstdc++ 缺 GLIBCXX_3.4.30，且系统库文件存在，可对当前终端设置：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

这个设置只针对上述动态库冲突，没有该错误无需设置。只安装 pip 包不能保证所有云端镜像的渲染环境可用。依据：[MuJoCo 渲染上下文](https://mujoco.readthedocs.io/en/stable/python.html#rendering)。

## 4. tmux 与 act-env 各自是什么

**tmux 保留远程终端会话。**你关闭 VS Code、SSH 断线后，会话中的训练仍可继续；重新连接再回到同一终端。它不管理 Python 包，也不会自动关机或省掉运行费用。

```bash
tmux new -s act002          # 创建远程终端会话
# 进入后激活 Python 环境，再运行训练
source /root/autodl-tmp/act-env/bin/activate
# 分离：先 Ctrl+B，松开后按 D；不会停止训练
# 重新连接后执行：
tmux attach -t act002
```

**act-env 是我们给 Python 虚拟环境目录取的名字。**`python -m venv /root/autodl-tmp/act-env` 与 `python -m venv .venv` 做的是同一件事，区别只是位置和名称。它里面有独立 Python 入口和安装的包，不是模型、数据或 tmux 会话。

`act-env`、`.venv`、`.venv-gui` 都可以作为虚拟环境的目录名。Windows 虚拟环境不能直接复制到 Linux 使用，需要在目标机器重新创建并安装依赖。`source .../bin/activate` 只让当前终端优先使用该环境；新终端通常需要重新激活。用 `which python` 可核对实际解释器。

依据：[Python venv](https://docs.python.org/3.12/library/venv.html)、[tmux 入门](https://github.com/tmux/tmux/wiki/Getting-Started)。
