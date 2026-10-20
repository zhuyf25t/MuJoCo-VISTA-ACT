# MuJoCo · VISTA · ACT

在 MuJoCo 中，让移动机械臂通过视觉完成捡球、搬运和投放。VISTA 调用视觉模型生成专家示范，ACT 从成功示范中学习，再回到同一环境进行闭环评估。

完整流程：**`run_collect.py` 采集 → `act.prepare` 准备数据 → `act.train` 训练 → `act.rollout` 批量执行与录像**。

## 环境安装

以下命令适用于 **AutoDL / Ubuntu、Python 3.12、NVIDIA GPU**，均在项目根目录执行。项目路径按实际位置修改。

```bash
cd /root/autodl-tmp/MuJoCo-VISTA-ACT

# 无窗口渲染所需系统库；tmux 用于断开 SSH 后继续运行。
apt-get update
apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri tmux

python --version
nvidia-smi
python -m venv /root/autodl-tmp/act-env
source /root/autodl-tmp/act-env/bin/activate

python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export MUJOCO_GL=egl
export TORCH_HOME=/root/autodl-tmp/torch-cache
# 首次下载 ACT 使用的 ImageNet ResNet-18 权重，后续训练读取缓存。
python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)"
```

这里使用项目固定版本的 CUDA 12.8 wheel，要求宿主机驱动兼容；版本组合见 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)。EGL 用于采集、每轮训练后的闭环评估及 rollout 录像，因此这三个阶段都需要 `MUJOCO_GL=egl`。

新终端须重新激活环境，并设置上述两个环境变量。长任务可在 `tmux new -s act` 中运行；按 `Ctrl+B` 后按 `D` 离开，`tmux attach -t act` 返回。

## 从采集到批量 rollout

### 1. VISTA 采集

首次执行 `cp -n .env.example .env`，编辑 `.env`，填写支持**图片输入和工具调用**的 OpenAI 兼容端点、模型和密钥。保持 `stream=false`，设置正整数 `max_model_calls`。仅训练 ACT 或运行 ACT 不需要 API 配置。

```bash
python -u run_collect.py \
  --brain vista --episodes 100 --seed 0 --workers 4 --max-decisions 64 \
  --out data/episodes --brain-memory output/vista --dataset-out data/act_v1
```

这条命令采集 seed 0–99，共 100 集，最多 4 集并行；成功集数由实际任务结果决定。每集原始记录和 VISTA 记忆分别保存到各自根目录的 `ep_NNNN/`，编号取场景 seed。再次追加时改用新的起始 seed；已有 episode 目录不会覆盖。

所有采集进程共享 `output/vista/guide.md`，图片、草稿和动作日志按 episode 隔离。`--max-decisions` 限制每集外层决策次数，`.env` 的 `max_model_calls` 限制单次决策内部的模型请求数。

### 2. 准备训练数据

```bash
python -m act.prepare \
  --dataset-root data/act_v1 --experiment output/act_run01
```

实验目录必须尚不存在。程序校验并冻结全部 `ready + selected_for_dataset + success` 示范，计算归一化统计和文件哈希。全部示范用于训练，`splits.json` 的 `val` 为空；后续追加的数据不会自动进入这个实验。

### 3. 训练 ACT

```bash
python -u -m act.train \
  --dataset-root data/act_v1 --experiment output/act_run01 \
  --config act/configs/baseline.yaml --epochs 100 --device cuda --workers 10
```

默认网络为 **ResNet-18 + Transformer + 四个动作头**，输入车头 RGB 和五维实测状态，不使用 CVAE 或 KL。默认预测长度 **K=32**、batch size **8**；`--epochs 100` 表示总共训练 100 轮。

每轮先保存 `last.pt`，再在固定 **520–539，共 20 个场景**上闭环评估。训练示范不能使用这些 seed。策略每次预测 32 拍，依次执行前 8 拍后重新观察，即 **K32/H8**；不使用 temporal ensemble。每场景最多执行 600 拍，正常结束后停止底盘、保持臂爪目标，继续仿真 3 秒再判定最终结果。

`best.pt` 按 **最终成功集数 → 验证投放集数 → 确认抓取集数**依次比较，平分保留更早轮次。投放需释放后在箱内稳定至少 0.5 秒；抓取需后端确认稳定接触。训练 loss 只用于观察优化，不参与选优或早停。

`--workers 10` 控制闭环评估进程数；YAML 中的 `training.num_workers` 控制 DataLoader 读取进程数，两者独立。

### 4. 批量 rollout 与录像

```bash
python -u -m act.rollout \
  --checkpoint output/act_run01/best.pt --seeds $(seq 1000 1099) \
  --output output/rollout_1000_1099 --device cuda --workers 10 \
  --max-steps 600 --video
```

这条命令在 100 个新场景中运行策略，最多 10 个进程并行。输出目录必须尚不存在；重跑时换一个目录名。最终测试应选择未参与训练和模型选优的 seed。

每集输出 `seed-N.json`（动作轨迹、任务结果与 checkpoint 身份）和 `seed-N.mp4`（俯视录像）；`summary.json` 汇总成功率、抓取、投放及失败信息。策略只看车头图和机器人实测状态，不调用 VISTA，也不把预测动作追加到专家数据集。

## 数据采集与格式

机器人由移动底盘、两自由度机械臂和夹爪组成，控制频率为 **10 Hz**，车头图像为 **320×240 RGB**。五维状态依次为肩角、肘角、夹爪连续开度、底盘前向速度和转向角速度；动作对应肩肘目标角、夹爪开闭目标及底盘目标速度。

每个控制拍先采样图像和实测状态，再执行 0.1 秒动作，完整执行后写入原始轨迹。一集正常结束且环境判定成功后，整集导出 HDF5，包括纠错、等待和收尾控制拍。

```text
data/
├── episodes/
│   └── ep_0000/                 # 原始轨迹、图片、决策及结束观测
└── act_v1/
    ├── manifest.json           # 可用示范及其数据契约
    └── episodes/
        └── episode-0000.h5     # manifest 分配的编号，不等同于场景 seed
```

| HDF5 字段 | 类型与形状 | 含义 |
| --- | --- | --- |
| `/observations/images/front` | `uint8 [T,240,320,3]` | 执行前的 RGB 像素 |
| `/observations/state` | `float32 [T,5]` | 同一时刻的实测状态 |
| `/action` | `float32 [T,5]` | 随后施加的控制目标 |

状态和动作保留物理单位，归一化统计在 `act.prepare` 中计算，读取训练样本时应用。`manifest.json` 保存 schema、episode 身份、seed、来源、状态、相对 HDF5 路径及控制拍数。

同一个 `--dataset-out` 持续追加数据。失败和中断集保留原始记录，但不进入训练候选；`--no-images` 不导出 HDF5。原始记录逐拍刷盘，HDF5 和 manifest 原子发布。下次采集自动检查未结算记录，也可运行 `python recorder.py --dataset-out data/act_v1` 恢复发布已有的完整成功集；不完整的 episode 标记为中断。

**v1 和 v2 在数据采集层面的格式完全一致，只是项目期间重构过一次控制层和神经网络架构。**

## 实验产物与续训

```text
output/act_run01/
├── manifest.json / splits.json  # 冻结的训练清单，train 包含全部示范，val=[]
├── stats.json / prepared.json   # 归一化统计、数据哈希和样本数
├── config.yaml                 # 本次网络与训练配置
├── class_balance.json          # 类别与切换样本权重
├── runtime.json                # 环境版本、设备和运行设置
├── metrics.jsonl               # 每轮训练损失、动作 L1 和耗时
├── task_eval_protocol.json     # 固定评估场景、执行方式及评分协议
├── task_metrics.jsonl          # 各轮闭环任务成绩
├── checkpoints/                # 每轮不可覆盖的推理快照
├── eval_epoch_001/             # 该轮评估轨迹和 task_eval.json
├── source_runs/                # 每次启动的源码归档，仅用于追溯
├── best.pt / best.json         # 最佳推理模型及选优依据
└── last.pt                     # 最近完整轮次的续训状态
```

续训仍使用同一个实验目录，`--epochs` 指定包含已完成部分的总轮数：

```bash
python -u -m act.train \
  --dataset-root data/act_v1 --experiment output/act_run01 \
  --resume --epochs 150 --device cuda --workers 10
```

`last.pt` 已保存模型、训练配置、优化器、AMP、随机数状态、epoch 和累计更新次数，不必再传 YAML 或骨干权重。epoch 从 0 开始：`epoch=99` 表示第 100 轮已完成，续训从第 101 轮开始；`--epochs 150` 表示训练到第 150 轮。

原训练 HDF5 放在 `--dataset-root` 指定的目录；以下文件一起放在 `--experiment` 指定的实验目录：

```text
output/act_run01/
├── last.pt                  # 恢复模型、优化器、随机状态和轮次
├── manifest.json            # 冻结的训练示范清单
├── splits.json              # 训练样本分配，全部示范在 train 中
├── stats.json               # 归一化统计
├── prepared.json            # 准备格式、数据哈希和样本数
├── best.pt                  # 历史最佳模型（可选，加载历史 best）
├── best.json                # 最佳轮次、成绩和模型哈希（可选，加载历史 best）
└── task_eval_protocol.json  # 评估场景与评分规则（可选，加载历史 best）
```

加载历史 best 时需一起保留标为可选的三份文件，新轮次成绩超过历史最佳才替换 `best.pt`，同分保留原 best。不保留 `best.pt` 和 `best.json` 时，从续训后的轮次重新选优；评估协议缺失时自动生成，已有协议继续沿用。best 文件缺失一部分或相互不匹配时，会明确报错。

训练始终从 `last.pt` 恢复，`best.pt` 不含优化器等续训状态。原地续训无需移动文件；换机器时，将上述文件复制到同一个实验目录，保持训练数据中 manifest 与 HDF5 的相对结构，并修改 `--dataset-root`。来源记录中的旧绝对路径不参与数据读取。

**`--resume` 直接训练下一轮，不补评历史轮次，即使上轮评估中断也一样。** 新轮次结束后仍照常评估并选择 best。旧逐轮快照、评估目录和日志都不是续训必需文件。源码变化允许续训；数据、训练配置及保留的评估协议仍需匹配。只恢复已保存的完整 epoch，未保存的半轮需要重跑。

## 项目架构与文件职责

下面列出计划发布到 GitHub 的运行源码、模型、配置和使用说明。数据、权重、日志、环境目录及本地 review 清单不在此树中。`output/vista/guide.md` 是保留的跨集经验文件。

```text
MuJoCo-VISTA-ACT/
├── README.md                       # 安装、完整流程、数据格式和项目导航
├── .env.example                    # VISTA 模型端点与密钥的配置模板
├── .gitignore                      # 排除凭据、数据、运行产物和本地清单
├── .gitattributes                  # Git 文本换行规则
├── requirements.txt                # Python 依赖及固定版本
├── config.py                       # 仿真、控制、相机、频率与数据路径常量
├── run_collect.py                  # VISTA 数据采集入口、并行调度和结果结算
├── recorder.py                     # 逐拍记录、成功集 HDF5 导出和中断恢复
├── observations.py                 # 帧号、RGB、实测状态和连续观测窗口
├── process_lock.py                 # 数据发布与共享记忆使用的跨进程文件锁
├── hal.py                          # 机器人传感、执行器与世界接口约定
├── compose_model.py                # 组合底盘和机械臂 XML，生成并自检机器人模型
├── viz.py                          # 相机渲染、采集窗口及录像视角
├── playback.py                     # 读取历史轨迹，恢复已记录的完整物理姿态
├── replay_gui.py                   # 历史轨迹回放界面与无窗口轨迹校验
├── models/
│   ├── base_chassis.xml            # 移动底盘源模型
│   ├── reference_arm.xml           # 两自由度机械臂与夹爪源模型
│   ├── mobile_manip.generated.xml  # 组合后的机器人模型，场景实际引用
│   └── scene_pickball.xml          # 球、箱子、地面、机器人和相机的任务场景
├── sim/
│   ├── __init__.py                 # 仿真包入口
│   ├── backend.py                  # MuJoCo 硬件实现、渲染和稳定抓取辅助
│   └── env.py                      # 场景重置、seed、观测流和任务成功判定
├── control/
│   ├── README.md                   # 稳定判定、等待优化与 VISTA 可见信息
│   ├── __init__.py                 # 控制模块导出
│   ├── drivers.py                  # 底盘、机械臂、夹爪动作及执行终止条件
│   ├── kinematics.py               # 两关节机械臂正逆运动学
│   ├── stability.py                # 图像与实测状态的稳定性判断
│   ├── tools.py                    # 机器人动作原语、参数校验及实际执行
│   └── perception.py               # 颜色分割与投影测距；保留，当前主流程不调用
├── brains/vista/
│   ├── README.md                   # VISTA 使用说明与模块介绍
│   ├── __init__.py                 # 导出 VistaBrain，供采集器加载
│   ├── base.py                     # Brain 接口与 Decision 数据结构
│   ├── brain.py                    # 决策入口、LangGraph 调用与动作批次回执
│   ├── graph.py                    # 模型节点与工具节点流转，play/finish 交接
│   ├── model.py                    # 模型 HTTP 请求、图像消息和工具回复解析
│   ├── settings.py                 # 环境配置、模型参数及工具和存储限制
│   ├── state.py                    # 消息、动作回执及批次状态规则
│   ├── storage.py                  # 帧、动作日志、本集笔记与共享指南存储
│   ├── tools.py                    # 模型可调用的查看、查询、笔记及控制工具
│   └── prompts/system.txt          # 当前实际加载的系统提示词
├── act/
│   ├── README.md                   # ACT 数据、模型、训练和部署说明
│   ├── __init__.py                 # ACT 包标识
│   ├── config.py                   # 网络与训练配置的数据结构、读取和校验
│   ├── configs/baseline.yaml       # 默认网络、优化器和 DataLoader 参数
│   ├── common.py                   # 数据契约、路径、哈希及准备文件校验
│   ├── prepare.py                  # 冻结全部有效示范，计算归一化统计
│   ├── dataset.py                  # 按帧取样、未来动作块、归一化和 padding
│   ├── action_codec.py             # 五维物理动作与四个输出头的编码、解码
│   ├── model.py                    # ResNet-18、Transformer 和四头网络
│   ├── losses.py                   # 混合损失、类别与切换权重、训练统计
│   ├── train.py                    # 优化更新、保存 last、每轮评估与续训
│   ├── experiment.py               # 固定评估协议、源码归档、各轮快照和 best 选优
│   ├── task_eval.py                # 固定场景闭环评估与三项任务评分
│   ├── checkpoint.py               # checkpoint 读写、模型重建和部署契约校验
│   ├── policy.py                   # 图像和状态推理、动作解码及 H8 缓存执行
│   └── rollout.py                  # MuJoCo 并行策略执行、结果汇总、轨迹和录像
└── output/vista/guide.md            # 随项目保留的 VISTA 跨集经验
```

根目录 `config.py` 管理仿真与控制；`act/config.py` 定义网络和训练配置，读取 `act/configs/baseline.yaml`。它们职责不同，不存在两个配置文件互相覆盖的问题。

VISTA 通过工具调用执行专家动作；ACT 直接输出五维执行器目标。两者共用 MuJoCo 环境和底层接口，模型不接收物体真值坐标；VISTA 的闭爪回执会反馈后端是否确认夹持。模块细节见 [Control README](control/README.md)、[VISTA README](brains/vista/README.md) 和 [ACT README](act/README.md)。
