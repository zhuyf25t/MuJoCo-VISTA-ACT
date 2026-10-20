# ACT：动作块预测与闭环执行

ACT 用当前车头 RGB 和五维机器人实测状态，预测未来 32 拍动作。训练使用全部合格专家示范，每轮在固定场景中闭环评估，按任务表现选择 `best.pt`。

当前网络是 **ResNet-18 + Transformer Encoder/Decoder + 四个动作头**。执行采用 **K32/H8**：预测 32 拍，执行前 8 拍，再观察和预测。**不使用 temporal ensemble，也没有 CVAE 或 KL 损失。**

## 准备、训练与运行

先按[主 README](../README.md#环境安装)安装 AutoDL/Linux 环境并缓存 ImageNet 权重。以下命令在项目根目录执行。

```bash
export MUJOCO_GL=egl
export TORCH_HOME=/root/autodl-tmp/torch-cache

python -m act.prepare \
  --dataset-root data/act_v1 --experiment output/act_run01

python -u -m act.train \
  --dataset-root data/act_v1 --experiment output/act_run01 \
  --config act/configs/baseline.yaml --epochs 100 --device cuda --workers 10

python -u -m act.rollout \
  --checkpoint output/act_run01/best.pt --seeds $(seq 1000 1099) \
  --output output/rollout_1000_1099 --device cuda --workers 10 \
  --max-steps 600 --video
```

`prepare` 和 rollout 的输出目录须尚不存在。`--epochs` 是总轮数，`--workers` 是闭环评估进程数；YAML 中 `training.num_workers` 是训练数据读取进程数，两者独立。

## 一个训练样本是什么

对专家轨迹中的时刻 `t`，模型只读取当前观察，学习从这一拍开始的未来动作：

```text
输入：图像 I_t + 实测状态 s_t
标签：动作 a_t, a_(t+1), …, a_(t+31)
```

图像为 `240×320 RGB`；状态依次是肩角、肘角、夹爪连续开度、底盘线速度和角速度。动作对应肩肘目标角、夹爪开闭目标及底盘目标速度。

每集 T 拍提供 T 个训练起点，相邻样本的动作块重叠；不足 32 拍的末尾补齐并屏蔽损失，不跨 episode。图像除以 255 后再做 ImageNet 标准化，状态使用训练集均值和标准差归一化。未来动作及未来状态仅用于监督，不输入网络。

`prepare` 冻结全部合格示范及统计量，`splits.json` 的 `val=[]`。没有离线验证集，完整 HDF5 格式见[主 README](../README.md#数据采集与格式)。

## 神经网络架构

以下为 `configs/baseline.yaml` 对应的默认结构，B 表示 batch size。

### 1. 图像与状态编码

```text
车头 RGB [B,3,240,320]
  → ImageNet ResNet-18（去掉全局池化和原分类层）
  → 特征图 [B,512,8,10]
  → 1×1 卷积投影、展开 → 80 个图像 token

实测状态 [B,5] → Linear(5,512) → 1 个状态 token
可学习的共享常量                  → 1 个 latent token

合并：82 个 token，每个 512 维
```

图像 token 使用二维正弦位置编码，另加相机和类型编码；状态与 latent 使用类型编码。ResNet 使用 FrozenBatchNorm，卷积权重仍参与微调。

latent token 是所有样本共享的可学习向量，训练和推理使用同一个值；没有随机采样、未来动作编码器或后验网络。

### 2. Transformer 预测动作块

```text
82 个观测 token
  → 4 层 Encoder → 观测 memory
                       ↓
32 个可学习动作 query → 7 层 Decoder → [B,32,512]
```

Encoder 融合图像与状态。Decoder 的内容向量从零开始，用 32 个可学习 query 区分未来拍次；每层先做 query 间的 self-attention，再用 cross-attention 读取观测 memory，随后经过前馈网络。

32 拍一次并行预测，self-attention 不使用因果掩码。默认隐藏维度 512、8 个注意力头、前馈维度 2048、dropout 0.1。

### 3. 四个并行的动作头

每个未来拍的 512 维特征同时送入四个线性头：

| 动作头 | 输出形状 | 任务与训练损失 | 执行时的解码 |
| --- | --- | --- | --- |
| 肩肘 `Linear(512,2)` | `[B,32,2]` | 回归标准化目标角；L1 | 反归一化为肩、肘目标角 |
| 夹爪 `Linear(512,2)` | `[B,32,2]` | closed/open 二分类；加权交叉熵 | 选最大分数类别，得到开度目标 0/1 |
| 底盘 `Linear(512,5)` | `[B,32,5]` | stop/forward/back/left/right 五分类；加权交叉熵 | 停止、前后 ±0.30 m/s，或确定左右转方向 |
| 转向幅度 `Linear(512,1)` | `[B,32,1]` | 回归转速幅度比例；MSE | 限幅到 0–1，乘 1 rad/s，再按左右类别赋符号 |

**夹爪头和底盘头本身就是分类器。** 四个头直接接在 Decoder 后，各自输出回归值或类别分数；随后由动作解码器组合成五维物理动作：

```text
[肩目标角, 肘目标角, 夹爪开闭目标, 底盘线速度, 底盘角速度]
```

底盘选择前进或后退时角速度为 0，选择转向时线速度为 0；选择停止时两者都为 0。ACT 输出的是逐拍控制目标，不是 VISTA 的 `play` 工具调用。

## 训练与模型选择

总损失为四项之和，各项系数均为 1：

```text
L = 肩肘 L1 + 夹爪加权 CE + 底盘加权 CE + 转向幅度 MSE
```

转向幅度只在**专家标签为转向**的位置监督，训练时使用线性原始输出，执行时才限幅。分类权重由训练标签的类别频次和“切换 / 延续”分组统计，保存到 `class_balance.json`；padding 不参与损失，数据不因类别加权而重采样。

默认 batch size 为 8，优化器为 AdamW：ResNet 学习率 `1e-5`，其余网络 `1e-4`，weight decay `1e-4`。训练 loss 和解码动作 L1 用于观察优化，不触发早停或选择 best。

每轮先保存完整续训状态 `last.pt` 和该轮推理快照，再用 **520–539 的固定 20 个场景**闭环评估。训练示范不得使用这些 seed。每场景最多执行 600 拍，正常结束后停止底盘、保持臂爪目标继续 3 秒，再判断最终结果。

`best.pt` 按下面的顺序逐项比较，数值越大越好，完全相同保留更早轮次：

```text
(最终成功集数, 曾完成验证投放的集数, 曾确认稳定抓取的集数)
```

每集每项最多计 1 分。投放要求释放后在箱内稳定至少 0.5 秒；抓取要求后端确认稳定接触。只有完整且身份匹配的评估结果参与选优；最终测试另选未参与训练和选优的 seed。

## Rollout：K32/H8，无 ensemble

策略以 10 Hz 执行，一拍是 0.1 秒：

```text
第 0 拍：读取 I_0、s_0 → 预测 0–31 拍 → 执行 0–7 拍
第 8 拍：读取 I_8、s_8 → 预测 8–39 拍 → 执行 8–15 拍
第16拍：读取 I_16、s_16 → 重新预测……
```

一次预测覆盖 3.2 秒，实际只执行前 0.8 秒；剩余 24 拍丢弃。缓存的 8 拍执行期间不重新调用网络，到下一次规划时才使用新图像和状态。

**没有 temporal ensemble：不会将多个历史预测对同一拍的动作加权平均。** 训练时的任务评估和独立 rollout 都使用相同的 K32/H8。checkpoint 中的 `ensemble_coefficient: null` 仅记录“未启用集成”，没有对应的可调参数。

`policy.py` 负责缓存与解码，`rollout.py` 将五维目标交给仿真执行器。每集保存 `seed-N.json`，启用 `--video` 时另存俯视 MP4；`summary.json` 汇总完整结果。

## 续训

```bash
python -u -m act.train \
  --dataset-root data/act_v1 --experiment output/act_run01 \
  --resume --epochs 150 --device cuda --workers 10
```

`last.pt` 包含权重、训练配置、优化器、AMP、随机状态、epoch 和累计更新次数，无需再传 YAML。epoch 从 0 开始：`epoch=99` 表示已完成第 100 轮，接下来训练第 101 轮；`--epochs 150` 指总共训练到第 150 轮。

原训练 HDF5 由 `--dataset-root` 指定，以下文件一起放在 `--experiment` 指定的目录：

```text
实验目录/
├── last.pt                  # 恢复模型、优化器、随机状态和轮次
├── manifest.json            # 冻结的训练示范清单
├── splits.json              # 训练样本分配
├── stats.json               # 归一化统计
├── prepared.json            # 准备格式、数据哈希和样本数
├── best.pt                  # 历史最佳模型（可选，加载历史 best）
├── best.json                # 最佳轮次、成绩和模型哈希（可选，加载历史 best）
└── task_eval_protocol.json  # 评估场景与评分规则（可选，加载历史 best）
```

三份可选文件一起保留即可继承历史最佳成绩；新成绩更好才替换 `best.pt`，同分保留原 best。不保留 `best.pt` 和 `best.json` 时，从续训后的轮次重新选优；评估协议缺失时自动生成，已有协议继续沿用。best 文件缺失一部分或相互不匹配时，会明确报错。训练始终恢复 `last.pt`。

**`--resume` 直接进入下一轮，不补评历史轮次，即使上轮评估中断也一样。** 新轮次结束后照常评估。旧逐轮快照、评估目录和日志都不是续训必需文件。只从完整 epoch 恢复；源码变化允许续训，数据、训练配置及保留的评估协议仍需匹配。

## 文件布局

```text
act/
├── README.md               # 数据、网络、训练和执行说明
├── __init__.py             # ACT 包标识
├── config.py               # 网络和训练配置的数据结构、加载与校验
├── configs/baseline.yaml   # 默认网络、优化器和 DataLoader 设置
├── common.py               # 数据契约、路径、哈希和准备文件校验
├── prepare.py              # 冻结有效示范并计算归一化统计
├── dataset.py              # 当前观察、未来动作块和 padding
├── model.py                # ResNet、Transformer 和四个动作头
├── action_codec.py         # 训练标签编码与五维物理动作解码
├── losses.py               # 混合损失、类别与切换权重、训练统计
├── train.py                # 参数更新、保存 last、每轮评估与续训
├── experiment.py           # 固定评估协议、源码归档、各轮快照和 best 选优
├── task_eval.py            # 固定场景评估与三项任务评分
├── checkpoint.py           # 模型保存、加载和部署契约校验
├── policy.py               # 当前观察推理与 H8 动作缓存
└── rollout.py              # 并行仿真执行、轨迹、视频与结果汇总
```

数据格式、实验产物目录及完整使用流程见[主 README](../README.md)。设计参考 [ACT 原仓库](https://github.com/tonyzhaozh/act)，网络与部署方式以本项目实现为准。
