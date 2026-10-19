# ACT：从示范到控制策略

本目录实现 Action Chunking with Transformers：使用成功示范训练策略，根据车头图像和机器人实测状态预测未来动作块。包含数据准备、训练、离线验证、checkpoint 恢复、独立推理和 MuJoCo 策略评估。

默认 **K=32、10 Hz、单张 320×240 RGB、五维实测状态、五维目标动作、batch size=8**。

使用顺序为 `prepare → train → rollout`。安装与云端环境配置见[项目 README](../README.md#环境配置)，逐模块实现逻辑见 [ACT 模块详解](README_log.md)。

## 各文件的责任

| 文件 | 责任 |
| --- | --- |
| `config.py` / `configs/baseline.yaml` | 可校验、可保存的网络与训练参数 |
| `common.py` | 数据契约、相对路径、文件校验与原子保存 |
| `prepare.py` | 冻结清单、显式划分、检查 HDF5、计算训练集统计量 |
| `dataset.py` | 每个控制时刻构成一个样本，读取并标准化当前观测与未来动作块 |
| `model.py` | ResNet-18、观测 Encoder、动作 Decoder、四个动作头 |
| `train.py` | 混合损失训练与验证、日志、best/last checkpoint、恢复训练 |
| `metrics.py` | 转向召回、静止启动、误转向率、零转向基线与闭爪机会诊断 |
| `policy.py` | 只凭 checkpoint 推理；可选同一执行时刻的时间集成 |
| `rollout.py` | 五维命令逐拍执行、指定 seed 的 MuJoCo 评估 |
| `replay.py` | 专家命令回放，区分学习误差与执行契约不一致 |
| `eval_turn.py` | 不重新训练，独立评估 checkpoint 的左转、右转、停止和不转向 |
| `action_codec.py` / `losses.py` | 物理单位与类别映射、自动类别权重和四头损失 |
| `checkpoint.py` | 混合模型 checkpoint 保存、加载、格式与动作映射校验 |

## episode、样本、batch、epoch

一个 episode 是从 reset 到结束的一整次任务。一集有 T 个控制拍，就提供 T 个训练样本。
样本 t 的输入是当前图片和状态，标签是从 t 开始最多 K 个动作。相邻样本的动作块重叠。
末尾不足 K 时补零，并用 `is_pad=True` 排除补齐位置；不跨集，不丢弃末尾控制拍。

假设训练集有 16 集，每集 300 拍：共有 4,800 个样本。batch size=8 意味着每次拿 8 个
控制时刻的样本，一次前向预测的动作张量是 `[8,32,5]`。一个 epoch 遍历全部 4,800 个样本，
需要 `ceil(4800/8)=600` 次参数更新。下一 epoch 重新打乱样本顺序，再学习一遍；数据不会消耗掉。
这里没有梯度累积，最后一个不足 8 的 batch 也保留。

所有帧按同样概率采样，所以长 episode 提供更多样本。K=32 覆盖约 3.2 秒，
但默认部署每拍重规划，只执行预测块第一拍；不会盲走 3.2 秒。

## 当前网络和损失（v2）

每个 Decoder 向量分别经过四个头：肩肘两个标准化角度、夹爪两个 logits、底盘五个 logits、
一个线性转向幅度（训练不裁剪，执行解码时限幅到 0–1）。CNN、Encoder/Decoder 层数和 K 不变。训练与推理共用一个可学习的
常量 latent token；不读取专家未来动作作为网络输入，不使用 CVAE 后验或 KL。

底盘类别顺序固定为 stop、forward、back、left、right。forward/back 为 ±0.3 m/s，
left/right 的角速度为 ±r×1.0 rad/s；其余对应速度为零。夹爪类别为 closed=0、open=1。
肩肘反归一化后保持连续。最终仍输出五维目标，每个动作持续一拍，不调用 VISTA tools。
训练前检查命令满足这些条件；不支持同时前进和转向的标签，出现此类数据会明确报错。

```text
loss = mean(肩肘标准化 L1) + mean(w_grip[class,group] × 夹爪 CE)
     + mean(w_base[class,group] × 底盘 CE) + mean(真实转向位置的原始幅度 MSE)
```

四项系数固定为 1。幅度是角速度除以 1 rad/s 后的无量纲数，绝不与 m/s 共用幅度。
幅度损失仅按真值转向位置计算，与模型是否选对类别无关；没有转向标签的 batch 此项为零。
所有项排除 padding，交叉熵直接接 logits；argmax 只用于解码和评估。

每轮均匀打乱全部训练起点，不再提供重采样/Huber 开关。类别权重自动用训练集统计：
`w_c=N/(C_present*N_c)`。N_c 统计的是有效 chunk 位置：同一集第 t 帧的标签出现
`min(K,t+1)` 次。验证集和 padding 不参与权重。缺失类权重为零并在统计中列出。
CE 用有效位置数作分母，不用每个 batch 的权重和，避免稀有类权重在单类 batch 中被抵消。
随后在各类别内部区分 ordinary/switch：保持该类别的原总权重，让存在的两个子组各占一半。
子组权重为 `w_c * N_c / (G_c * N_cg)`，G_c 是该类别实际存在的子组数。夹爪原始 w_c=1，
保留开/闭各自原有总权重。训练中没有出现的子组沿用该类别原权重，避免验证时被忽略。
停止切换要求实测 |v|>0.05 m/s 或 |w|>0.2 rad/s；启动/换向要求目标方向的实测速度小于
0.05，或另一运动分量仍在运动（直行检测 |w|>0.2、转向检测 |v|>0.05）。开爪切换要求
当前开度<0.4，闭爪切换要求开度>0.6；其余位置归入 ordinary。
Dataset 返回动作块对应的物理状态 `action_states`，仅供计算这些监督分组，绝不送入模型。
网络输入仍然是当前图片和当前状态；不修改 HDF5、控制器或采集流程。
统计写入 `class_balance.json` 并存进 checkpoint，恢复时重新核对。

`best.pt` 和早停使用最小 `val_loss`（同一混合目标，复用训练权重），L1 只作诊断。
验证损失逐项累计：幅度项按真实转向数平均，其余按有效位置数平均，不受验证 batch 切分影响。
日志报告底盘五类/夹爪两类混淆矩阵与召回、左右转启动、停止、误转向、前后行启动和开闭爪机会。
新增全 chunk 各类别的切换/其余计数与召回，以及减速标签位置的条件幅度 MSE、限幅后 MAE
和恒最大转速基线。条件幅度指标不含分类错误；闭环表现还需要结合方向和停止指标。
转向正确要求方向正确且 |w|>0.2 rad/s；停止转向要求 |w|<0.05 rad/s。
高离线指标不能替代闭环捡球、搬运、投放成功率。

本分支只支持当前混合动作头 checkpoint（action contract version=2）。旧五维回归和
旧 sigmoid 幅度头 checkpoint 均不能训练、恢复或评估，避免权重被按不同语义解释。
请建立新实验目录；数据格式不变，可继续使用原来的 HDF5。latent_dim、posterior_layers、
kl_weight、重采样和 Huber 配置项已删除。

已有 `output/act` 的实验应保留，更新代码后使用新目录、原来的划分重新训练：

```bash
python -m act.prepare --dataset-root data/act_v1 --experiment output/act_switch_1019 --splits output/act/splits.json
python -m act.train --dataset-root data/act_v1 --experiment output/act_switch_1019 --config act/configs/baseline.yaml --epochs 10 --device cuda
```

这次首次训练不加 `--resume`；新实验后续中断恢复才使用该参数。

### 不重新训练，检查一个 checkpoint 的转向

```bash
python -m act.eval_turn --checkpoint path/to/best.pt --dataset-root data/act_v1 --output tmp/turn_eval_01
```

默认使用 CPU，并在原训练/验证集分别按每集每类最多一帧进行分层抽查。六类是启动左转、
启动右转、继续左转、继续右转、停止转向、不应转向。`--per-category 0` 可检查所有符合
这些条件的帧；抽查结果不能解释成全数据精度。工具验证原始 HDF5 哈希及划分哈希，不会
把后来采集的数据混进评估。若自动重建划分不匹配，传入原来的 `--splits path/to/splits.json`。

报告分别显示两组数据的各类样本数、正确率、角速度 MAE 与平均预测幅度；不把六类平均成
一个掩盖问题的分数。它检查 z=0 时 chunk 第一拍，未使用 temporal ensembling。
预测方向正确且幅度超过 0.2 rad/s 才算转向正确；停止要求幅度小于 0.05 rad/s。
“正确”表示符合这个专家观测上的动作标签，不能代替真实任务成功率。

输出 `report.json` 和逐帧 `frames.csv`。每次使用新输出目录。零转向策略可能停止项满分，
但左右转项为零；持续转圈策略可能转向项不错，却停止项失败。两者都不能通过导航评估。

## 数据和跨机器迁移

数据使用采集器的 `mobile_act_v1` 契约：

```text
data/act_v1/
  manifest.json
  episodes/episode-XXXX.h5

HDF5:
  observations/images/front  uint8   [T,240,320,3]
  observations/state         float32 [T,5]
  action                     float32 [T,5]
```

只有 `status=ready, selected_for_dataset=true, success=true` 的完整集可被准备程序选中。
图像、状态和命令使用相同下标；图像与状态在执行前采样。夹爪状态连续，目标 0 闭、1 开。
五维依次为肩、肘、夹爪、车体前向速度、转向角速度。实际值不能用命令替代。

加载只使用 `--dataset-root` 加相对 `h5` 字段，例如 `episodes/episode-0000.h5`。
`source_dir` 即使仍是 Windows 的绝对路径也不参与训练读取，保留它用于追溯。
上传 AutoDL 时复制 HDF5 目录与实验目录；不需要复制原始 PNG、VISTA 日志或 `.env`。
HDF5 内已有真实像素。换机器后只改 `--dataset-root`；推理只需 checkpoint。
训练启动/恢复会核对准备时保存的文件 SHA-256，防止同名数据被偷偷替换。

`prepare.py` 读取一次完整 manifest 快照，检查全部入选 HDF5 的元数据、数组、像素可读性；
原始采集的逐拍时间连续性由 recorder 在发布 HDF5 时检查。准备程序不会重新解释原始日志。
继续追加采集不会改变已准备实验。需要新数据时创建新的实验目录。

## 安装和预训练 ResNet

训练依赖统一位于根目录 `requirements.txt`：固定 `torch==2.10.0` 与
`torchvision==0.25.0` 的兼容组合，并使用 PyYAML 读取配置。根据机器驱动选择
官方对应的 CPU/CUDA wheel；这不是要求 AutoDL 使用 CPU wheel。

```bash
# 本地无显卡时可以先安装这对 CPU wheel；再安装项目其余依赖。
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

ResNet-18 原版含分类头约 1,169 万参数；本项目移除分类头，只使用空间特征图，
并采用 FrozenBatchNorm，骨干约 1,117 万可训练参数。FrozenBatchNorm 固定的是 BN
统计量与仿射量，卷积权重仍参与微调；CNN 学习率为 `1e-5`，其他网络为 `1e-4`。
ACT 的 Transformer、投影和动作头从随机初始化开始；启动训练时打印实际参数数量。

默认调用 torchvision 的 `ResNet18_Weights.IMAGENET1K_V1`，首次初始化会从
[PyTorch 官方地址](https://download.pytorch.org/models/resnet18-f37072fd.pth)下载 ImageNet 权重，
通常约 45 MB，不依赖 Hugging Face。可设置 `TORCH_HOME` 控制缓存目录。
无法联网时可以提前下载该文件，训练传 `--backbone-weights /path/resnet18-f37072fd.pth`。
`pretrained_backbone: false` 供合成数据测试或从零训练对照使用；下载失败不会静默改用随机权重。
恢复训练和 checkpoint 推理不再下载骨干，因为完整骨干权重已经包含在 checkpoint 中。

图像在 Dataset/policy 中转为 `[camera,3,H,W]` 并除以 255，进入 model 后只做一次
ImageNet 通道标准化；保持 240×320，不做 torchvision 分类模型默认的中心裁剪。

## 数据划分

选择一个尚不存在的实验目录。以下两种方式任选其一：按场景 seed 随机分组，或提供显式 episode 清单。

```bash
# 按场景 seed 分组，验证比例为 20%；同一场景不会按帧拆开。
python -m act.prepare --dataset-root data/act_v1 --experiment output/act/exp_001 --val-fraction 0.2 --seed 42

# 或明确列出要使用的 episode。没有列出的成功集不参与本次实验。
python -m act.prepare --dataset-root data/act_v1 --experiment output/act/exp_001 --splits my_splits.json
```

`my_splits.json` 示例：

```json
{"train": ["episode-0000", "episode-0001"], "val": ["episode-0002"]}
```

必须有非空训练与验证集；同一 seed 不得跨组。比例按 seed 组计算，集数可能不精确等于比例。
归一化统计只用训练集，按每维保存均值和标准差（下限 0.01）；验证与推理复用。
先用少量轨迹做过拟合排错时可指定很小的 train 列表，并留出独立 val；训练误差下降证明
优化链路工作，验证误差和闭环表现才帮助判断策略是否能推广。

## 训练、恢复、停止

```bash
# 训练上限示例：遍历训练集 100 轮。
python -m act.train --dataset-root data/act_v1 --experiment output/act/exp_001 --config act/configs/baseline.yaml --epochs 100 --device cuda

# 从最后一个已保存的完整 epoch 继续，到总计 150 个 epoch。
python -m act.train --dataset-root /root/autodl-tmp/act_v1 --experiment output/act/exp_001 --resume --epochs 150 --device cuda
```

`--epochs` 必须显式提供，表示总训练遍数。baseline.yaml 使用 8 个数据加载 worker、FP32；可在配置中修改
`num_workers` 或 CUDA `amp`。这些设置会保存在实验中，恢复时不允许默默修改。
模型、优化器、梯度缩放器、随机数及采样生成器状态均保存。按完整 epoch 恢复；
中途被杀会重跑尚未保存的 epoch。相同软件/硬件上的 CPU 恢复可验证一致性，
跨 GPU、驱动或平台不承诺逐位相同。新实验不允许覆盖已开始训练的目录。

训练终端依次显示本轮训练与验证进度条：

```text
Train 01/100: ... train_loss=..., train_L1=..., val_loss=pending
Valid 01/100: ... train_loss=..., val_loss=...
```

进度条显示 batch 完成比例、耗时和 ETA。train_loss/val_loss 均报告四头混合损失，
训练期间验证值为 pending。`--progress auto` 在终端显示动态条，重定向时约每 10 秒在
batch 完成后输出进展；always 强制条，never 只输出阶段和每轮摘要。

`metrics.jsonl` 保存 train/val 的 arm_l1、gripper_ce、base_ce、turn_mse、总 loss；
`train_l1` 和 `val_l1` 为解码后五维标准化 L1。
还有 `val_mae_physical`、`val_first_mae_physical`、底盘/夹爪逐类指标和动作切换指标。
训练有 dropout 且参数持续更新，验证为 eval 模式，不能单凭一个小差值断言过拟合。

停止有两种机制：显式 epoch 上限；或配置 `patience>0` 后，连续该数量的 epoch
没有超过 `min_delta` 的验证改善时提前停止。默认 `patience=0`，即关闭早停。
`best.pt` 总是跟踪最小 `val_loss`，不受 `min_delta` 阈值影响。已经早停的实验
不能通过恢复偷偷重置耐心值；更改停止策略应建立新实验。

先观察小数据能否拟合、各动作维度是否正常，再在独立 MuJoCo 验证 seed 上比较少量候选。
最终测试使用未用于选模型/改参数的新 seed；反复用于调参的 seed 应称为验证集。

## 一个实验不等于一个 checkpoint

```text
output/act/exp_001/
  manifest.json   本次数据快照，包含相对 HDF5 路径和原始来源
  splits.json     固定的训练/验证 episode id
  stats.json      训练集归一化统计
  prepared.json   清单、统计、划分和 HDF5 的哈希及样本数量
  config.yaml    实际训练配置
  class_balance.json  仅训练集有效 chunk 位置的类别数与自动权重
  runtime.json   运行设备、依赖/Git 版本、目标 epoch、每 epoch 更新次数
  metrics.jsonl  各 epoch 的指标（epoch 字段从 0 开始）
  best.pt        离线验证指标最好的完整 checkpoint
  last.pt        最近一个完整 epoch 的 checkpoint
```

新实验对应新的目录，恢复同一个实验不会创建新编号。推理直接加载 checkpoint 即可，
无需重新训练；checkpoint 已包含结构配置、schema、统计和完整权重。
checkpoint 只加载自己或可信来源产生的文件，使用 PyTorch `weights_only=True`。

## 在 MuJoCo 中运行策略并生成视频

离线验证比较模型动作与专家动作；仿真评估则让 checkpoint 根据新的观察持续控制机器人，检查能否完成捡球和投放。`rollout.py` 已包含策略执行和 MP4 导出，不需要额外录屏。

### 运行一个 seed

AutoDL / Linux 无窗口模式，在已安装 EGL 依赖的环境中运行：

```bash
export MUJOCO_GL=egl
python -m act.rollout \
  --checkpoint output/act/exp_001/best.pt \
  --seeds 1000 \
  --output output/act/eval_seed1000 \
  --device cuda --max-steps 600 --video --label validation
```

本地使用 CPU 并打开窗口，可在 Windows PowerShell 执行：

```powershell
python -m act.rollout --checkpoint output/act/exp_001/best.pt --seeds 1000 --output output/act/eval_seed1000_local --device cpu --gui --video
```

把 checkpoint、seed 和输出目录换成自己的路径。输出目录必须尚不存在，重跑时使用新名称。批量场景可写 `--seeds 1000 1001 1002`；可用 `last.pt` 检查最后一轮，也可用 `best.pt` 检查离线验证误差最低的一轮。

### 输出与评分

```text
output/act/eval_seed1000/
├── seed-1000.mp4
├── seed-1000.json
└── summary.json
```

| 文件 | 查看什么 |
| --- | --- |
| `seed-1000.mp4` | 10 FPS 俯视录像，观察接近、抓取、搬运和释放过程 |
| `seed-1000.json` | 单场景结果、停止原因、请求动作与实际施加动作 |
| `summary.json` | 各 seed 的结果、成功率、控制拍数、推理延迟、checkpoint 哈希和评估设置 |

默认最多执行 600 拍，即 60 秒控制；检测到成功可提前结束。正常结束后停底盘并继续仿真 3 秒，再判最终成功，视频也包含该段。手动关闭窗口记为未成功。仿真会等待推理，因此 10 Hz 控制与 10 FPS 视频不代表实际计算达到实时速度。

策略只看车头图和实测状态，俯视图仅用于录像，物体真值仅用于评分。评估不会调用 VISTA，也不会将策略自己的动作导出为专家示范。GPU 训练和 MuJoCo 渲染需要分别检查；EGL 系统库安装见[项目环境配置](../README.md#环境配置)。

`--label validation|test` 是报告标签，不会自动核查 seed 是否已用于训练。挑模型或调参数时使用固定验证场景，最终测试预留独立 seed；不要只凭一次成功视频或较低的动作误差判断泛化能力。

### 时间集成超参数

时间集成在**推理/评估命令**中设置，改变它不需要重新训练：

```bash
python -m act.rollout --checkpoint output/act/exp_001/best.pt --seeds 1000 --output output/act/eval_seed1000_m001 --device cuda --video --ensemble-coefficient 0.01
```

| 设置 | 执行方式 |
| --- | --- |
| 不传 `--ensemble-coefficient` | 关闭集成，每拍重新预测，执行最新动作块的第一拍 |
| `--ensemble-coefficient 0` | 对覆盖同一执行时刻的候选预测等权融合 |
| `--ensemble-coefficient 0.01` | 按旧到新顺序 `i=0,1,...` 使用 `exp(-m*i)` 权重，正系数偏向较旧预测 |

v2 集成先平均分类概率，再 argmax；肩肘平均连续目标，转向幅度按选中方向的概率加权平均。
不会直接平均左右转的有符号速度。
集成组合的是不同预测块中指向同一时刻的预测，最多涉及 K 个块。它可能平滑动作，也可能延迟状态切换；比较时固定 checkpoint 和验证 seeds，并使用不同输出目录。

### 动作适配约定

预测反归一化后，由 `SimActionAdapter` 执行关节范围限制、夹爪 0.4/0.6 滞回以及底盘执行器限幅。
关节标签已经是专家每拍施加的目标，不再对目标变化重复限速，也不提供额外限速或开环执行跨度参数。
每拍重新推理；关节范围和底盘执行器限幅仍有效。此处是仿真执行约定，不是实机安全控制器。
夹爪从开到闭的命令变化开启一次 1.2 秒抓取判定，持续闭合不重新计时；重新触发需要先开再闭。

五维动作无法表示专家连续两次 `close_gripper` 的高层调用边界。因此，适配器与 VISTA 原语的重复抓取语义存在差异。分析抓取失败时，应同时检查预测动作、阈值转换和执行回执，不能仅用动作 L1 判断控制语义是否一致。

### 训练前先检查专家动作能否回放

```bash
python -m act.replay --seeds 0 1 3 5 13 17 25 --modes direct recorded-events --output tmp/expert_replay.json
```

加 `--dataset-root data/act_v1` 可直接回放训练 HDF5 的实际精度；工具会先核对 seed、
source_uid 和逐帧内容，防止误把并发采集时的 episode 编号当成 seed。

输出路径必须尚不存在。`direct` 直接施加每拍关节目标；
`recorded-events` 在 direct 基础上恢复 VISTA 日志里的开闭爪调用与结束边界。后一种模式使用专家
未来信息，只用于定位缺失的控制语义，绝不是策略成功率。报告逐维状态/动作误差与最终成功，
并记录来源文件哈希。原始数据包含程序收尾，回放不会重复追加收尾。

若 direct 回放不能复现成功，但 recorded-events 可以，应先修复采集与部署的控制契约，
而不是归咎于模型或继续扩充同一种数据。旧五维文件没有编码重复闭爪事件；补回事件需要新的
明确版本与策略输出，或者把专家与学生统一到只由五维目标触发的同一个控制器。不要只延长
学生的抓取窗口就宣称契约已一致，也不要在训练时把删除的帧拼接成仍然等间隔的序列。

## 测试与参考

```bash
python -m pytest tests/test_act.py tests/test_act_hybrid.py tests/test_act_motion_metrics.py tests/test_act_eval_turn.py -q
```

测试覆盖样本对齐、padding 与损失屏蔽、场景隔离、训练集统计、路径迁移、文件哈希、四头梯度、类别权重、确定性推理、checkpoint 格式校验、训练恢复、早停、时间集成及 MuJoCo 录像。合成数据测试用于检查实现行为，任务表现由独立场景评估。

网络保留 ACT 的动作分块和视觉 Transformer，改用分类与连续回归混合头；当前版本关闭 CVAE。结构与数据契约以本项目源码为准。

参考：[ACT 原实现](https://github.com/tonyzhaozh/act)、[torchvision ResNet-18](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)、[PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)。
