# ACT：先把训练过程看清楚

当前实现支持数据准备、训练、离线验证、checkpoint 恢复、独立推理和 MuJoCo 闭环评估。
默认 **K=32、10 Hz、单张 320×240 RGB、五维实测状态、五维目标动作、batch size=8**。
这里只提供代码和命令；不会在导入模块时划分数据、启动训练或修改采集目录。

## 各文件的责任

| 文件 | 责任 |
| --- | --- |
| `config.py` / `configs/baseline.yaml` | 可校验、可保存的网络与训练参数 |
| `common.py` | 数据契约、相对路径、文件校验与原子保存 |
| `prepare.py` | 冻结清单、显式划分、检查 HDF5、计算训练集统计量 |
| `dataset.py` | 每个控制时刻构成一个样本，读取并标准化当前观测与未来动作块 |
| `model.py` | ResNet-18、CVAE 后验、观测 Encoder、动作 Decoder |
| `train.py` | 训练、`z=0` 验证、日志、best/last checkpoint、恢复训练 |
| `policy.py` | 只凭 checkpoint 推理；可选同一执行时刻的时间集成 |
| `rollout.py` | 五维命令逐拍执行、指定 seed 的 MuJoCo 评估 |

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

## 数据和跨机器迁移

沿用采集器的 `mobile_act_v1`，不修改正在采集的 manifest：

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
ACT 的 Transformer、投影和动作头从随机初始化开始。当前默认模型实测共有
59,853,573 个可训练参数，其中骨干 11,166,912 个；启动训练时会再次打印实际数量。

默认调用 torchvision 的 `ResNet18_Weights.IMAGENET1K_V1`，首次初始化会从
[PyTorch 官方地址](https://download.pytorch.org/models/resnet18-f37072fd.pth)下载 ImageNet 权重，
通常约 45 MB，不依赖 Hugging Face。可设置 `TORCH_HOME` 控制缓存目录。
无法联网时可以提前下载该文件，训练传 `--backbone-weights /path/resnet18-f37072fd.pth`。
`pretrained_backbone: false` 供合成数据测试或从零训练对照使用；下载失败不会静默改用随机权重。
恢复训练和 checkpoint 推理不再下载骨干，因为完整骨干权重已经包含在 checkpoint 中。

图像在 Dataset/policy 中转为 `[camera,3,H,W]` 并除以 255，进入 model 后只做一次
ImageNet 通道标准化；保持 240×320，不做 torchvision 分类模型默认的中心裁剪。

## 数据划分：由你决定后显式运行

以下均为命令示例，没有实际划分当前数据。两种方式任选其一。

```bash
# 比例由用户指定；按场景 seed 分组，而非按帧随机切分。
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
# epochs=100 仅展示命令格式，不表示本任务已确定训练预算。
python -m act.train --dataset-root data/act_v1 --experiment output/act/exp_001 --config act/configs/baseline.yaml --epochs 100 --device cuda

# 从最后一个已保存的完整 epoch 继续，到总计 150 个 epoch。
python -m act.train --dataset-root /root/autodl-tmp/act_v1 --experiment output/act/exp_001 --resume --epochs 150 --device cuda
```

`--epochs` 必须显式提供，表示总训练遍数。默认单进程读取、FP32；需要时在配置中开启
`num_workers` 或 CUDA `amp`。这些设置会保存在实验中，恢复时不允许默默修改。
模型、优化器、梯度缩放器、随机数及采样生成器状态均保存。按完整 epoch 恢复；
中途被杀会重跑尚未保存的 epoch。相同软件/硬件上的 CPU 恢复可验证一致性，
跨 GPU、驱动或平台不承诺逐位相同。新实验不允许覆盖已开始训练的目录。

每个 epoch 后验证并记录：

| 指标 | 含义 |
| --- | --- |
| `train_l1` | 训练时使用后验 z 的动作误差，只统计有效动作元素 |
| `train_kl` | CVAE 后验与标准正态先验的 KL |
| `val_l1_z0` | 验证时不读取专家动作，用 z=0 预测的标准化动作误差，越低越好 |
| `val_mae_physical` | 五维动作的物理单位 MAE，顺序与 schema 一致 |
| `val_first_mae_physical` | 只检查每次预测第一拍，更直接对应逐拍重规划 |
| `val_gripper_accuracy` | 原始物理标度下按 0.5 阈值的开闭准确率 |

训练与验证使用不同 z，不能直接用两者差值诊断过拟合。动作块误差也不等于任务成功率。
连续帧高度相关；验证只有少数 episode 时，结果波动不能当作稳定的泛化结论。

停止有两种机制：显式 epoch 上限；或配置 `patience>0` 后，连续该数量的 epoch
没有超过 `min_delta` 的验证改善时提前停止。默认 `patience=0`，不替用户决定耐心值。
`best.pt` 总是跟踪最小 `val_l1_z0`，不受 `min_delta` 阈值影响。已经早停的实验
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
  runtime.json   运行设备、依赖/Git 版本、目标 epoch、每 epoch 更新次数
  metrics.jsonl  各 epoch 的指标（epoch 字段从 0 开始）
  best.pt        离线验证指标最好的完整 checkpoint
  last.pt        最近一个完整 epoch 的 checkpoint
```

新实验对应新的目录，恢复同一个实验不会创建新编号。推理直接加载 checkpoint 即可，
无需重新训练；checkpoint 已包含结构配置、schema、统计和完整权重。
checkpoint 只加载自己或可信来源产生的文件，使用 PyTorch `weights_only=True`。

## MuJoCo 闭环评估

```bash
python -m act.rollout --checkpoint output/act/exp_001/best.pt --seeds 1000 1001 --output output/act/eval_001 --gui
```

加上 `--video` 会在评估目录保存各 seed 的俯视 MP4，包含最后 3 秒的静置判定，
按控制频率 10 FPS 播放。云端无窗口评估可设置 `MUJOCO_GL=egl`，使用 `--device cuda --video`；
俯视相机只用于回放，不输入策略网络。

AutoDL 的精简 Ubuntu 镜像可能缺少离屏渲染依赖。若出现 EGL 加载错误，可先安装
`libegl1 libegl-mesa0 libgl1-mesa-dri`。若随后提示 Conda 的 `libstdc++.so.6`
缺少 `GLIBCXX_3.4.30`，本项目在该镜像上验证过以下进程级设置：

```bash
export MUJOCO_GL=egl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

这两个环境变量用于云端 Linux；本地 Windows 不需要。GPU 训练和仿真渲染是两项
独立能力，应先确认 `torch.cuda.is_available()`，再单独测试实际 MuJoCo 录像。

seed 只是命令示例，正式验证/测试名单仍由用户决定。默认 CPU 推理，每拍只执行一个
0.1 秒区间，可选 `--device cuda`。MuJoCo 可以等待模型，因此仿真 10 Hz 不代表实际推理达到 10 Hz。
报告包含最终环境成功率、仿真时间、控制拍数、推理延迟、请求与实际施加动作；
每集正常结束后停底盘并继续仿真 3 秒，再判成功。模型不会获得物体真值。
评估不调用 VISTA、不向采集 manifest 注册 episode、不把预测当作新的专家示范。

默认关闭时间集成。传 `--ensemble-coefficient 0.01` 后，对指向同一执行时刻的预测加权；
沿用文档中旧到新 `exp(-m*i)` 的方向，正系数偏旧，需要单独评估切换延迟。
动作反归一化后执行关节范围/目标速率限制、夹爪 0.4/0.6 滞回、底盘执行器限幅。

**当前仿真夹爪有一个明确限制。** VISTA 的 `close_gripper` 是一个 1.2 秒的有状态原语；
同样的闭合目标可以连续触发两次原语，采集的五维动作却不能标识第二次调用的边界。
ACT 适配器采用“开到闭的命令变化开启一次 1.2 秒判定，保持闭合不反复重置”的基线。
它不会自动重试或在超时后延迟吸附；重新开启需要开再闭。因此不能宣称已经完全复现
专家连续两次闭爪的隐藏状态。若失败集中于该处，应单独讨论统一底层抓取语义或增加可观测
的重试信号，而不是仅增大模型。现有采集器和 VISTA 控制逻辑保持原样。

训练网络保留 `act.md` 的核心结构，包含教学设计的后验层数、相机/类型嵌入和有效元素
损失归一化选择，不宣称与原论文的某个上游提交逐行相同。

参考：[ACT 原实现](https://github.com/tonyzhaozh/act)、
[torchvision ResNet-18](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)、
[PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)。

## 当前验证范围

`tests/test_act.py` 使用临时合成 HDF5，覆盖执行拍样本对齐、补齐和损失屏蔽、场景隔离、
训练集统计、Windows 来源路径迁移、文件哈希校验、CVAE 梯度、确定性 z=0 推理、
epoch 恢复一致性、早停和 best/last 区分、独立 checkpoint 推理、同时间集成，以及
两拍真实 MuJoCo 执行和收尾。另在 CPU 上用官方 ImageNet 权重验证了默认规模网络的
单样本前向/反向。RTX 3080 Ti 上已验证默认完整模型的 8 次 FP32/AdamW 更新和
z=0 推理：batch=8、K=32、单相机 240×320，PyTorch 峰值分配约 1.12 GiB、
峰值保留约 1.34 GiB；这是合成输入短测，不能等同于所有配置的显存保证。
同一云端环境的训练/恢复/独立策略/真实 MuJoCo 录像集成测试通过。AMP 尚未实测。
正式实验指标与任务成功率以对应 `output/act/exp_*/` 的日志和评估摘要为准。
