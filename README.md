# MuJoCo·VISTA·ACT

在 MuJoCo 中，让移动机械臂通过视觉完成捡球、搬运和投放，并将成功示范用于 ACT 模仿学习。

项目连接三个部分：**MuJoCo** 提供机器人、物理仿真和数据记录；**VISTA** 用视觉模型、工具和记忆生成示范；**ACT** 从示范中学习连续控制策略，再回到同一仿真环境中评估。策略输入为车头 RGB 图像和五维机器人实测状态，输出为肩、肘、夹爪及底盘的控制目标。

```text
MuJoCo 场景 → VISTA 示范 → 成功轨迹 HDF5 + manifest
                                      ↓
MuJoCo 策略评估 ← ACT checkpoint ← 数据划分与训练
```

## 项目布局

```text
MuJoCo_Brain/
├── brains/
│   └── vista/
├── act/
├── control/
├── sim/
├── models/
├── tests/
├── run_collect.py
├── recorder.py
├── playback.py
├── replay_gui.py
├── rerender.py
├── inspect_data.py
├── compose_model.py
├── hal.py
├── config.py
├── viz.py
├── requirements.txt
├── .env.example
├── data/                  # 运行时生成，Git 忽略
└── output/                # 运行时生成，Git 忽略
```

| 路径 | 职责 |
| --- | --- |
| `brains/vista/` | 视觉决策、工具调用、图片档案与笔记；见下文 VISTA |
| `act/` | 数据准备、网络、训练、策略推理与评估；见下文 ACT |
| `control/` | 底盘、两自由度机械臂、夹爪的控制器与动作原语 |
| `sim/` / `models/` | 仿真环境、机器人封装与 MuJoCo XML 模型 |
| `hal.py` / `config.py` | 硬件抽象接口，以及物理参数、相机、频率和路径 |
| `run_collect.py` / `recorder.py` | 采集循环、原始记录、成功集导出和中断恢复 |
| `playback.py` / `replay_gui.py` | 历史状态回放、回放窗口与轨迹校验 |
| `rerender.py` / `inspect_data.py` | 重渲染录像、查看元数据和决策记录 |
| `compose_model.py` / `viz.py` | 模型组合与相机、运行窗口 |
| `tests/` | 数据、控制、VISTA 协议及 ACT 训练推理测试 |
| `data/` / `output/` | 采集数据，以及 VISTA 记忆、ACT 实验和评估产物 |

以下命令均在项目根目录执行。数据、模型权重、运行产物和 `.env` 不随 Git 提交；换机器时按需单独传输。

## 环境配置

#### 本地环境

使用 Python 3.12，建议创建独立虚拟环境。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

上述 CPU wheel 适合无 NVIDIA 显卡的电脑，用于采集、数据准备、调试和 CPU 推理。Linux 激活环境使用 `source .venv/bin/activate`。ACT 正式训练建议使用 CUDA GPU。

#### AutoDL / Ubuntu GPU 环境

选择 Python 3.12 和兼容 NVIDIA 驱动的镜像，克隆项目后，在实例终端执行：

```bash
# 系统库：MuJoCo 无窗口渲染；tmux 用于保留断线后的终端会话。
apt-get update
apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri tmux

cd /root/autodl-tmp/MuJoCo_Brain
python --version
nvidia-smi
python -m venv /root/autodl-tmp/act-env
source /root/autodl-tmp/act-env/bin/activate

# CUDA 12.8 wheel 示例，要求宿主机驱动兼容。
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 后续在这个终端中运行 MuJoCo 离屏采集或策略录像。
export MUJOCO_GL=egl
```

`apt-get update` 更新系统软件包索引，`pip install` 安装虚拟环境里的 Python 包。EGL 库用于渲染，读取 HDF5 训练本身不需要图形窗口；标准平台的 `imageio-ffmpeg` wheel 自带录像所需的 FFmpeg。其他 CUDA 组合见 [PyTorch 版本表](https://pytorch.org/get-started/previous-versions/)。

`act-env` 是 `python -m venv` 创建的环境目录，名称可以自定，新终端要重新激活。需要断线后继续训练时，可先 `tmux new -s act`，再激活环境并运行训练；按 `Ctrl+B`、松开后按 `D` 离开会话，使用 `tmux attach -t act` 返回。tmux 不负责关闭计费实例。数据迁移和训练命令见 [ACT 使用说明](act/README.md)。

## MuJoCo

#### 场景与控制

机器人由移动底盘、两自由度机械臂和夹爪组成，任务是将球放入收纳箱。车头相机提供 **320×240 RGB**，控制频率为 **10 Hz**。五维状态依次是肩角、肘角、夹爪连续开度、车体前向速度和转向角速度；动作对应关节目标、夹爪开闭目标和底盘目标速度。

VISTA 与 ACT 共用环境及底层执行接口。策略看图和本体状态，物体真值用于环境评分。俯视相机用于观察与录像，不输入策略。正常结束后停止底盘并继续仿真 3 秒，再判断最终任务结果。

#### 数据采集与格式

采集入口为 `run_collect.py`，配置和启动命令见下一节。每个控制拍先采样图像与实测状态，再执行 0.1 秒动作；完整执行后写入原始轨迹。一集正常结束且环境判定成功后，整集导出 HDF5，包括纠错、等待和收尾控制拍。

```text
data/
├── episodes/
│   └── ep_0000/            # 原始轨迹、图片、决策及结束观测
└── act_v1/
    ├── manifest.json
    └── episodes/
        └── episode-0000.h5
```

| HDF5 字段 | 类型与形状 | 含义 |
| --- | --- | --- |
| `/observations/images/front` | `uint8 [T,240,320,3]` | 执行前的 RGB 像素 |
| `/observations/state` | `float32 [T,5]` | 同一时刻的实测状态 |
| `/action` | `float32 [T,5]` | 随后施加的控制目标 |

状态和动作保留物理单位，归一化在 ACT 数据准备与读取阶段完成。`manifest.json` 保存 schema、episode 身份、seed、来源、状态、相对 HDF5 路径及控制拍数。采集阶段只判断是否为合格示范，训练/验证划分由 `act.prepare` 完成。

同一个 `--dataset-out` 持续追加数据。失败和中断集保留原始记录，但不进入训练候选；`--no-images` 用于诊断，不导出 HDF5。原始记录逐拍刷盘，HDF5 和 manifest 原子发布。下次采集自动检查未结算记录，也可运行 `python recorder.py --dataset-out data/act_v1` 恢复发布已有的完整成功集；不完整的 episode 标记为中断。

#### 回放与检查

```bash
# 回放专家轨迹；--check 只校验、不打开窗口。
python replay_gui.py data/episodes/ep_0000
python replay_gui.py data/episodes/ep_0000 --check

# 查看元数据；从历史物理状态重新渲染俯视录像。
python inspect_data.py --meta ep_0000
python rerender.py ep_0000 --cams overhead

python -m pytest -q
```

协议测试使用模拟模型，不调用付费 API；仿真测试及重渲染需要可用的 MuJoCo 图形后端。历史轨迹回放重现保存的状态，ACT 策略评估则让 checkpoint 在环境中重新执行任务。

## VISTA

#### 视觉决策与记忆

VISTA 将视觉模型的工具调用组织成一个 LangGraph 循环。模型可以查看原图或局部、读取像素、查询动作历史、维护跨集指南和本集草稿，再用 `play` 提交动作批次。采集器执行后返回新观测和回执；模型继续检查结果，直至调用 `finish` 或达到决策上限。

#### 配置与运行

首次配置时，将 `.env.example` 复制为 `.env`，填写支持**图片输入和工具调用**的 OpenAI 兼容 Chat Completions 端点、模型和密钥。设置 `stream=false` 和正整数 `max_model_calls`；进程环境变量优先于 `.env`。仅使用 ACT 训练或推理不需要这些 API 配置。

```bash
python run_collect.py --brain vista --gui --episodes 5 --seed 0 --max-decisions 64 --out data/episodes/run_001 --brain-memory output/vista/run_001 --dataset-out data/act_v1
```

云端无窗口采集去掉 `--gui`，并使用前述 EGL 配置。`--seed` 为首集场景，后续逐次递增；再次采集新场景时相应调整起始 seed。`--max-decisions` 限制每集外层决策次数，`max_model_calls` 限制每次决策内部的模型请求次数。

无窗口并发使用 `--workers 4`（也支持10等正整数，默认1）。`--episodes`始终是总集数，不是每个进程的集数。每个进程独立创建仿真、相机和模型客户端；`--workers > 1`不能搭配`--gui`。例如：

```bash
python run_collect.py --workers 4 --episodes 10 --seed 0 --max-decisions 32 --out data/episodes/run_002 --brain-memory output/vista/run_002 --dataset-out data/act_v1
```

原始数据与VISTA目录都按seed命名：seed 7对应两处的`ep_0007`，与启动/完成顺序无关；已有目录报错，不覆盖或偷偷改号。ACT HDF5继续使用manifest管理唯一编号，`manifest.episodes[].seed`与`source_dir`记录对应关系，数据格式不变。并行子进程日志在原始episode目录的`collector.log`，结果和实际耗时在`worker_result.json`。第一次Ctrl+C停止派发，等待当前各集完成；再次Ctrl+C中断。也可传`--stop-file tmp/STOP`，该文件出现后不再启动新集（串行也支持）。

所有进程共享`--brain-memory/guide.md`。短文件锁串行化读写，模型调用不持锁；`write_guide(content, base_revision)`按版本检查，冲突返回`GUIDE_CONFLICT`，由VISTA重读最新版、合并经验后重写。每次模型请求前发现guide变化会注入新版内容。`guide_access.jsonl`记录进程、episode、时间及读写版本，`guide_versions/`保留写入前后全文。`working.md`、帧与history仍按episode隔离。独立启动的采集命令使用同一记忆根目录时，也遵循同一套锁与版本协议。

文件布局、LangGraph 状态与图、九个工具及控制约定见 [VISTA README](brains/vista/README.md)。

## ACT

#### 数据准备与训练

ACT 使用 ImageNet 预训练 ResNet-18 提取视觉特征，通过 CVAE 和 Transformer 预测未来动作块。默认 **K=32、batch size=8、单车头相机、五维状态与动作**。一个样本对应一个控制时刻；一个 episode 可以提供多个样本。

```bash
# 选择一个尚不存在的实验目录；按场景 seed 分组划分训练/验证集。
python -m act.prepare --dataset-root data/act_v1 --experiment output/act/exp_001 --val-fraction 0.2 --seed 42

# epochs 表示遍历训练集的总轮数，可按训练预算调整。
python -m act.train --dataset-root data/act_v1 --experiment output/act/exp_001 --config act/configs/baseline.yaml --epochs 100 --device cuda
```

`prepare` 冻结数据清单、划分和训练集统计；`train` 显示每轮训练/验证进度与 loss，保存指标和 `best.pt` / `last.pt`。`exp_001` 是一次实验的目录，可以包含多个 checkpoint；推理直接加载 checkpoint，无需重新训练。

上传云端时保留数据目录中的 `manifest.json` 和 `episodes/*.h5` 相对结构。若已在本地执行 `prepare`，同时上传对应实验目录，再通过 `--dataset-root` 指向云端数据位置。来源记录中的本地绝对路径不参与训练读取。

#### 策略评估与视频

云端设置 `MUJOCO_GL=egl` 后，指定 checkpoint 和场景 seed 即可生成新策略执行的视频：

```bash
python -m act.rollout --checkpoint output/act/exp_001/best.pt --seeds 1000 --output output/act/eval_seed1000 --device cuda --video --label validation
```

输出目录须为新目录，包含 `seed-1000.mp4`、动作轨迹 JSON 和 `summary.json`。离线验证动作误差用于观察拟合情况；MuJoCo 成功率和视频用于检查实际行为。用于选模型或调参数的 seed 属于验证，最终测试应预留独立 seed。

数据契约、模型设置、恢复训练、时间集成及评估说明见 [ACT README](act/README.md)；逐模块实现逻辑见 [ACT 模块详解](act/README_log.md)。
