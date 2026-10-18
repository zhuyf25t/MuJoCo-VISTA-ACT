# 从数据到机器人：读懂本项目的 ACT

本文解释 `act/` 的模块职责、数据流和实现逻辑，面向第一次亲手训练模型的读者。使用入口见 [ACT README](README.md)，云端安装见[项目环境配置](../README.md#环境配置)。

这个项目用成功的专家轨迹做模仿学习：给定当前前视图和机器人实测状态，预测未来 32 拍动作。训练时用专家动作计算误差并更新网络；部署时网络自己预测，MuJoCo 执行动作后产生下一张图。当前实现参考 ACT 的 CVAE、动作分块和 Transformer 设计，包含本项目自己的数据契约与控制适配，不是上游代码的逐行复刻。

## 1. 先认识文件之间的关系

```text
采集侧（act 目录外）
run_collect.py → VISTA / 控制原语 → recorder.py
                                    ↓
                         data/act_v1/manifest.json + episodes/*.h5
                                    ↓
prepare.py ── common.py 检查 ──→ 实验目录中的清单、划分、统计、哈希
                                    ↓
train.py → dataset.py → model.py → 损失、反向传播、AdamW 更新
    │                         ↓
    └── validate(z=0) → metrics.jsonl / best.pt / last.pt
                                    ↓
policy.py → 当前图像和实测状态 → 五维动作
                                    ↓
rollout.py → SimActionAdapter → MuJoCo → 新观测 / 成功率 / 视频
```

| 文件 | 要读懂的核心内容 |
| --- | --- |
| `__init__.py` | 标识 Python 包；导入包不会开始训练或划分数据 |
| `config.py` | 数据类、参数检查、配置的读入与序列化 |
| `configs/baseline.yaml` | 默认超参数；不是已经冻结的数据划分 |
| `common.py` | 数据契约、可迁移路径、内容哈希、原子写入 |
| `prepare.py` | 选择合格轨迹、按场景划分、统计与冻结 |
| `dataset.py` | 将轨迹中的每个时刻变成一个训练样本 |
| `model.py` | 图像骨干、训练用后验、观测编码器、动作解码器 |
| `train.py` | 优化、离线验证、进度显示、模型保存与恢复 |
| `policy.py` | 从 checkpoint 重建模型、预处理、时间集成、反归一化 |
| `rollout.py` | 预测动作转成仿真命令，记录闭环结果 |
| `README.md` / `README_log.md` | 简要使用说明 / 模块实现的详细解释 |

`__pycache__/` 是 Python 缓存，不是模型。运行行为以实际 Python 实现为准。

## 2. 训练到底读取什么

### 2.1 一个 episode 文件

```text
data/act_v1/
  manifest.json
  episodes/
    episode-0000.h5
    episode-0001.h5

每个 HDF5：
  observations/images/front   uint8   [T, 240, 320, 3]
  observations/state          float32 [T, 5]
  action                      float32 [T, 5]
```

`T` 是这一条轨迹的控制拍数，控制频率为 10 Hz。同一个下标 `t` 的图片和 state 在动作执行前记录，action 是随后 0.1 秒内保持的实际施加目标。训练使用 RGB 像素，不读取原始 PNG 文件名，也不使用 VISTA 的思考文字。

| 维度 | state：机器人测到了什么 | action：希望执行什么 | 单位 |
| --- | --- | --- | --- |
| 0 | `shoulder_actual` | `shoulder_target` | rad |
| 1 | `elbow_actual` | `elbow_target` | rad |
| 2 | `gripper_actual` | `gripper_target` | 开度比例，0 闭、1 开 |
| 3 | `base_v_measured` | `base_v_cmd` | m/s |
| 4 | `base_w_measured` | `base_w_cmd` | rad/s |

夹爪实测开度可以连续变化，专家目标必须近似为 0 或 1。两个关节的 action 是目标角度，不是角度增量；底盘 action 是速度命令，不是位置。训练目标不能用实测 state 替代。

### 2.2 manifest 和 schema

manifest 顶层为 `schema`、`episodes`。schema 名为 `mobile_act_v1`，固定了上述名称、单位、相机、RGB 顺序、形状、10 Hz、`before_action` 和动作语义。代码采用严格相等检查，因此随意改一个字段也会被拒绝，避免同样的五个数在训练与执行时含义不同。

每个 episode 条目包含 `episode_id`、`source_uid`、`seed`、`source_dir`、`status`、`success`、`selected_for_dataset`、`num_steps`、`h5` 等信息。身份与来源用于追溯及去重；scene seed 决定场景划分；长度用于定位样本。

读取路径是 `dataset_root / entry['h5']`。例如 `h5='episodes/episode-0000.h5'` 可以直接从 Windows 迁到 Linux。`source_dir` 即使保留 Windows 绝对路径也不参与训练读取。上传必须保留 manifest 与 HDF5 的相对目录结构；只 `git clone` 不会带上被 `.gitignore` 排除的 data 和 output。

### 2.3 episode、sample、batch、epoch

一条 300 拍轨迹提供 300 个 sample。sample `t` 的输入仅为当前图片与状态，标签为 `action[t:t+K]`。相邻标签块会重叠，但不会跨越 episode。

例如训练集有 16 条轨迹、每条 300 拍，共 4,800 个 sample。batch=8，一轮需要 `ceil(4800/8)=600` 次更新；若最后一批不足 8 个样本，也会保留。数据每轮重新打乱后再读一遍，不会因为用过而耗尽。验证集同样按控制拍组成 batch，但不做参数更新。

`K=32` 在 10 Hz 下覆盖 3.2 秒。一次网络输出 `[B,32,5]`；默认执行时每拍重新预测，只取第一拍，不一次开环执行整块。

## 3. config.py 与 baseline.yaml：配置什么、检查什么

`ModelConfig`、`TrainConfig` 是 dataclass；`Config` 把两者组合起来。`to_dict()` 将元组转换成 YAML/JSON 可表示的列表；`config_from_dict()` 从保存的配置重建对象，并拒绝未知顶层配置；`load_config()` 使用 `yaml.safe_load`，不传文件时生成默认配置。

| 模型参数 | 默认值 | 实际作用 |
| --- | --- | --- |
| state_dim / action_dim | 5 / 5 | 状态投影输入维数 / 动作头输出维数 |
| cameras / image_hw | front / 240×320 | 数据契约；当前端到端管线只接受这个组合 |
| chunk_size | 32 | 动作查询数量与标签块长度 |
| hidden_dim / nheads | 512 / 8 | token 宽度 / 注意力头数，每头 64 维 |
| latent_dim | 32 | CVAE 潜变量维数 |
| posterior_layers | 2 | 用专家动作编码训练后验的层数 |
| encoder_layers / decoder_layers | 4 / 7 | 观测编码器 / 动作解码器深度 |
| dim_feedforward / dropout | 2048 / 0.1 | 各 Transformer 层的 FFN 中间宽度 / 丢弃比例 |
| pretrained_backbone | true | 新训练时载入 ImageNet ResNet-18 权重 |

| 训练参数 | 默认值 | 实际作用 |
| --- | --- | --- |
| batch_size | 8 | 每次更新的 sample 数量 |
| learning_rate | 0.0001 | 除图像骨干外的 AdamW 学习率 |
| backbone_learning_rate | 0.00001 | ResNet 卷积微调学习率 |
| weight_decay | 0.0001 | AdamW 权重衰减 |
| kl_weight | 10 | 损失中 KL 项系数 |
| grad_clip | 1 | 参数梯度总范数上限 |
| seed | 42 | 训练随机种子；与 prepare 的 split seed 分开设置 |
| num_workers | 0 | DataLoader 子进程数，0 最易排错 |
| amp | false | 是否启用 CUDA FP16 混合精度 |
| patience / min_delta | 0 / 0 | 0 表示关闭早停；否则按改善幅度和等待轮数停止 |

`__post_init__()` 检查整数范围、有限数、正学习率和标准差相关设置等；hidden_dim 必须同时被 nheads 和 4 整除，后者来自二维位置编码。配置类允许表达多相机，但当前 schema 检查限制为单 front 相机，改 YAML 本身不足以扩展输入。

没有学习率调度器、梯度累积、多卡 DDP 或数据增强。默认 YAML 也没有 epoch 数；训练必须显式传 `--epochs`。数据读取进程数、早停和最小改善幅度由每个实验的配置决定。

## 4. common.py：让一份实验可以可信地重现

`read_json()` 用 UTF-8 读文件。`atomic_json()` 先写同目录临时文件，flush、fsync，再 `os.replace` 发布；不接受 NaN。这样正常读取者看到的是旧版本或完整新版本，而不是半份 JSON。

`digest()` 对排序、紧凑编码后的 JSON 算 SHA-256，因此与普通文件逐字节哈希不同；`file_digest()` 分块读取文件计算 SHA-256。前者用于 metadata，后者用于 HDF5 和 checkpoint 标识。

`h5_path()` 同时按 POSIX/Windows 语法检查路径，拒绝绝对路径、盘符、反斜杠、`..`；解析后的文件还必须存在并留在 dataset root 内。它不会拿 `source_dir` 找数据。

`validate_schema()` 核对完整数据语义与模型输入；`validate_stats()` 检查均值、标准差长度为 5、数值有限、标准差为正。`load_prepared()` 读冻结的四份 JSON，检查 metadata 哈希、唯一 episode、非空且不重叠的划分、seed 不交叉、所有条目合格，并默认重新核对每个 HDF5 的文件哈希。`verify_hashes=False` 是内部可选参数，训练入口没有关闭检查。

哈希校验说明“这次读取的文件与准备时相同”，不是外部真实性认证。训练启动时校验后，仍应把数据视为只读，不能边训练边替换已有 HDF5。

## 5. prepare.py：建立一次固定实验

调用顺序是 `main → prepare → choose_splits / inspect_episode → 写 metadata`。导入模块不会运行准备；prepare 不修改实时采集 manifest。

### 5.1 选中和划分

`prepare()` 一次读取 manifest，仅保留同时满足 `status=='ready'`、`success is True`、`selected_for_dataset is True` 的条目。`choose_splits()` 支持两种互斥输入：

```bash
# 在项目根目录运行；exp_002 必须尚不存在。
python -m act.prepare --dataset-root data/act_v1 --experiment output/act/exp_002 --val-fraction 0.2 --seed 42

# 或使用自己的 JSON 清单，替代上面这一条命令。
python -m act.prepare --dataset-root data/act_v1 --experiment output/act/exp_002 --splits my_splits.json
```

```json
{"train": ["episode-0000", "episode-0001"], "val": ["episode-0002"]}
```

这些 ID 是格式示例，需要换成真实合格条目。手工清单可以只选择部分合格轨迹；未列出的不参与本次实验。重复 ID、未知 ID、空集合、同 seed 跨组都会报错。

自动划分先排序唯一 scene seeds，再用 NumPy 独立 RNG 打乱；验证组数为 `round(seed组数×比例)`，至少 1 组且至少留 1 组训练。同一个 seed 下的所有轨迹进同一组。因此 20% 指 seed 组比例，不保证帧数或 episode 数恰好 20%。不能把相邻帧随机分到训练和验证来制造“没见过的样本”。

### 5.2 检查数据和计算统计

`inspect_episode()` 检查 HDF5 属性与清单一致，包括 episode/source 身份、T、fps、schema。随后检查 state/action 的形状与 float32 类型、图像形状与 uint8 类型、有限值、夹爪范围；分块读图像以触发解压，发现损坏块。连续时间戳检查由 recorder 发布数据时负责，prepare 不重新解析原始日志。

`prepare()` 拒绝重复 source_uid；对入选文件算哈希。只有训练 episode 的 state/action 被连接后计算统计：

`mean[d] = 所有训练拍第 d 维的均值`

`std[d] = max(总体标准差(ddof=0), 0.01)`

计算用 float64，防止累计误差；Dataset 读取统计后用 float32。长轨迹包含更多拍，因此占统计量和采样的权重更大。验证集完全不参与均值与标准差估计。

### 5.3 产生的文件

| 文件 | 内容 |
| --- | --- |
| manifest.json | 仅包含本次选中 episode 的冻结清单与 schema |
| splits.json | train / val 的 episode ID 列表 |
| stats.json | state_mean、state_std、action_mean、action_std，各 5 个数 |
| prepared.json | 格式版本、内容/文件哈希、split seed、两组 sample 数量 |

准备只保存这些小文件，**不会复制 HDF5**。训练时仍需提供实际 dataset root。实验目录必须不存在；准备异常留下的不完整目录也不能直接继续，检查后用新编号重做。数据增加以后新建 exp_003 等，不修改旧实验里的 JSON。

## 6. dataset.py：一个下标怎样变成输入与标签

`EpisodeDataset.__init__()` 接收 root、某一 split 的 entries、stats 和 ModelConfig。先检查每个 HDF5 的长度与形状，把每集路径保存到 `paths`，把累计长度保存到 `ends`。没有把所有图片一次加载进内存。

例如两集长度为 3、2，`ends=[3,5]`，总长度为 5。`__getitem__(3)` 用 `bisect_right` 找到第二集，再减去前一集累计长度得到 `t=0`。每次 sample 打开自己的 HDF5、读取后关闭，避免 DataLoader 子进程共享 h5py 句柄；代价是较频繁的文件打开。

读取当前各相机图片、state[t]、action[t:t+K] 后：

1. `image_tensor()` 检查 RGB uint8，把 `[C,H,W,3]` 转成连续的 `[C,3,H,W]` float32，再除以 255。
2. state 与 action 各自减训练均值、除训练标准差。
3. 动作不足 K 时，在**标准化空间**补零。
4. `is_pad[k] = (k >= 有效动作长度)`；True 表示补齐，应排除。

单 sample 返回 `images [1,3,240,320]`、`state [5]`、`actions [32,5]`、`is_pad [32]`。DataLoader 再增加 batch 维。

假设 T=3、K=32，sample t=2 只含最后一个真实动作，后面 31 拍全为 padding。补齐零不代表物理上的“停住”；它根本不进入动作误差，也不作为后验注意力的有效 key/value。

## 7. model.py：一次前向传播

### 7.1 ResNet-18 保留空间信息

`ACT.__init__()` 创建 torchvision ResNet-18，使用 FrozenBatchNorm2d，删去全局平均池化和分类头。FrozenBatchNorm 的统计量与仿射量固定；卷积参数仍然训练。标准 ResNet-18 含分类头约 1,169 万参数；这里骨干可训练参数 11,166,912，整个默认 ACT 为 59,853,573。

新训练默认加载 `ResNet18_Weights.IMAGENET1K_V1`。它从 PyTorch 官方下载约 45 MB，不需要 Hugging Face；`--backbone-weights` 可传本地官方 state_dict。恢复/推理设置 `initialize_backbone=False`，直接恢复完整 ACT 权重，不重复下载。

图像进入 forward 前已在 [0,1]。模型用缓冲区保存的 ImageNet mean `[.485,.456,.406]` 和 std `[.229,.224,.225]` 做一次通道标准化，不做中心裁剪或改变分辨率。

| 默认 batch=8 的计算 | 张量形状 |
| --- | --- |
| 输入图片 / 状态 | `[8,1,3,240,320]` / `[8,5]` |
| 合并 B、相机维后经过 ResNet | `[8,512,8,10]` |
| 1×1 卷积投影并展平 | `[8,80,512]` |
| 加 state、z 两个 token | `[8,82,512]` |
| 32 个动作查询 | `[8,32,512]` |
| 动作输出 | `[8,32,5]` |

没有把图片压成单个向量，80 个位置仍分别进入注意力。

### 7.2 位置信息加在哪里

`sine_1d()` 使用按维度递减的频率生成 sin/cos 对。`sine_2d()` 将网格中心坐标缩放到 0～2π，纵向和横向各占一半通道，以行优先顺序排列，与 feature.flatten(2) 对齐。图像位置还加相机 embedding 和 image 类型 embedding；state、z 使用各自类型 embedding。

`EncoderLayer` 为 post-norm 结构：令 Q=K=x+pos、V=x，注意力更新经 dropout 与残差相加，再 LayerNorm；随后 Linear→ReLU→Dropout→Linear 的 FFN，再残差、LayerNorm。位置作用于注意力寻址，不混进 V。后验 encoder 可以额外传 padding mask。

### 7.3 训练后验：为什么训练会读取答案

训练时 `encode_posterior()` 把 `[CLS]`、投影后的当前 state、K 个投影后的专家动作接成 `[B,K+2,512]`，加入一维位置编码，经两层后验 encoder。mask 的前两位 False，随后直接接动作 padding mask。

CLS 输出投影成 `2×latent_dim`，分成 μ、logvar；logvar 表示方差的对数。用重参数化得到：

`z = μ + exp(0.5×logvar) × ε，ε ~ N(0,I)`。

这一路让训练时的潜变量概括专家未来动作，但 KL 项约束后验不要任意偏离标准正态。梯度通过重参数化流向后验网络，不更新数据标签。后验不输入图片；预测网络同时读取图片、state 和 z。

验证/部署没有未来专家动作可供读取。只要 `actions is None`，forward 就创建全零 z，并令 μ/logvar=None；此处取先验均值，不随机抽先验噪声。`model.eval()` 自身不会禁止后验，真正控制后验的是有没有传 actions。

### 7.4 观测 encoder 和动作 decoder

图片 token、state token、z token 经过 4 层观测 encoder 得到 memory。decoder 初始化 K 个全零内容向量 h，另有 K 个可学习的 `action_queries` 作为位置，代表未来第 0～31 拍。

每个 `DecoderLayer` 先在 K 个动作位置之间自注意力，再以动作表示为 query、观测 memory 为 key/value 做交叉注意力，最后走 FFN；每段均有 dropout、残差、LayerNorm。自注意力没有 causal mask：32 拍同时预测，不是依次生成并把上一拍输出喂回去。

7 层后经 `decoder_norm` 和 `Linear(512,5)` 输出标准化连续动作。头部没有 sigmoid/softmax，也没有额外 padding 分类头。即使夹爪目标为二值，训练时仍将它当连续数做 L1；执行阈值在 rollout 适配器处理。

## 8. train.py：参数怎样更新，结果怎样被保存

### 8.1 启动与 DataLoader

入口 `main()` 解析命令，`train()` 检查 epoch、progress 模式、prepared 是否存在，再取得 `.training.lock` 防止同目录并发训练；`_train()` 完成实际工作。开始新训练时若已经有 last/best/config/metrics 文件则拒绝覆盖。

加载和校验冻结数据后，显式 cuda 不可用会报错；auto 才会回落 CPU。训练集 shuffle=True，验证集 False，均 drop_last=False；CUDA 使用 pin_memory。Python、NumPy、PyTorch 及 DataLoader generator 都设置种子。

AdamW 分为 backbone 与其余参数两组学习率。AMP 仅在配置开启且设备为 CUDA 时生效；FP32 是默认。

### 8.2 一次更新与损失

循环依次执行：搬 batch 到设备 → 清零梯度 → 带专家动作的 forward → 算损失 → backward → unscale → 梯度裁剪 → optimizer.step → scaler.update。

设有效动作位置 mask 为 M，D=5，标准化预测为 p，标签为 a：

`L1 = sum(M×abs(p-a)) / (sum(M)×D)`

`KL = mean_batch[-0.5×sum_latent(1+logvar-μ²-exp(logvar))]`

`loss = L1 + kl_weight×KL`。

masked_l1 用 `torch.where` 排除补齐位置，而不是让补齐零参与平均。如果一批两条 sample 分别剩 2、1 个动作，分母为 `(2+1)×5=15`；15 个有效误差都为 0.2 时 L1 就为 0.2，无论后面补多少零。若 KL=0.01、系数=10，总损失为 0.3。AdamW 根据这个总损失的梯度更新模型；一次下降不保证下一轮验证改善。

非有限 loss 或梯度会报错。epoch 的 train_l1 按有效动作元素数量加权，train_kl 按 sample 数加权，不是随意平均各 batch 的均值。重叠动作块使某一真实动作可能在多个预测起点下重复计入；指标衡量的是“各起点的未来块预测误差”。

### 8.3 validate：不做反向传播

`@torch.no_grad()` 加 `model.eval()` 禁止梯度、关闭 dropout；forward 不传专家动作，用 z=0 预测。专家动作只在预测之后用于评分。它们不会泄漏进输入。

| 文件里的指标名 | 终端/数学含义 | 应怎样看 |
| --- | --- | --- |
| train_loss | train_l1 + kl_weight × train_kl | 本轮累计训练总损失；与纯动作验证 L1 的定义不同 |
| train_l1 | 训练动作误差，训练后验 z | 是否能拟合示范；与验证的 z 条件不同 |
| train_kl | 每 sample 的后验 KL | 观察潜变量正则化；不是任务分数 |
| val_l1_z0 | 五维标准化绝对误差平均 | 越低越好；选择 best.pt 的依据 |
| val_mae_physical | 每维标准化 MAE 乘训练 std | 分别是 rad、rad、ratio、m/s、rad/s |
| val_first_mae_physical | 每个 sample 只统计第一个动作 | 更直接对应默认每拍执行第一动作 |
| val_gripper_accuracy | 物理开度按 0.5 判断开/闭 | 不是抓取成功率；也不同于执行的 0.4/0.6 滞回 |
| elapsed_seconds | 本轮训练和验证耗时 | 当前计时不含后续保存 checkpoint 的耗时 |

val_l1_z0=0.253 不等于 25.3% 错误，也不是 25.3% 成功率。不同实验若更换了训练统计，标准化 L1 的尺度也可能变化；比较时同时看物理 MAE、固定验证场景成功率与视频。

### 8.4 进度条与每轮摘要

训练使用 tqdm，训练阶段 `Train 01/20` 和验证阶段 `Valid 01/20` 在底部依次动态更新。显示 batch 完成比例、耗时、预计剩余时间和当前累计 loss；它们都是**该阶段**的进度，不是整个训练的进度。训练条显示 train_loss、train_L1 和 val_loss=pending；验证条保留本轮 train_loss 并更新 val_loss（即 val_l1_z0）。阶段结束保留最终进度行，再打印中文摘要，格式如下：

```text
第 10/20 轮 | 73.0 秒 | 累计更新 9,430 次
  训练总 loss=... | 训练动作 L1=... | KL=... | 验证动作 L1(z=0)=0.2531（越低越好）
  验证 MAE：肩 ... rad | 肘 ... rad | 夹爪 ... | 前进 ... m/s | 转向 ... rad/s
  夹爪开闭准确率=87.0% | 最佳验证 L1=0.2531（第 10 轮） | 已更新 best.pt；已保存 last.pt
  早停计数=0/5（有效改善需超过 0.001）
```

上面是输出格式示意，不是重新训练的结果。`--progress auto` 默认在终端显示动态条；重定向文件或经 `tee` 管道后，约每 10 秒在下一个 batch 结束时写一行进展，阶段开始/结束也会报告。`always` 强制动态条，`never` 仅阶段提示与摘要。它不能在阻塞的 batch 内报告进度。`BatchProgress` 管理这些显示，`print_epoch_summary` 负责中文摘要；不改变优化计算。使用 tmux 可以保留交互会话。`metrics.jsonl` 新增 train_loss，原有字段保持；文件 epoch 从 0 开始，显示从 1 开始。视频命令见 [ACT README](README.md#在-mujoco-中运行策略并生成视频)。

### 8.5 best、last 与早停

每轮验证后：只要 score 严格小于历史最小值就更新 best；另用 `score < early_best - min_delta` 判断“足够改善”，否则 stale 加一。patience>0 且 stale 达到 patience 时正常早停。

例如 best 从 0.2530 降至 0.2528，best.pt 会更新；当 min_delta=0.001 时，这次改善未必重置早停计数，因为 early_best 是另一个参考值。因此“更新 best”和“早停计数归零”不是同一个条件。

`save_checkpoint()` 使用临时文件、fsync、替换；每轮先写 last，若有新最佳再写 best，最后追加指标。checkpoint 包含 format_version、模型、优化器、GradScaler、结构配置、schema、归一化统计、prepared 元数据、epoch/global_step、best 与早停状态、随机状态、当轮指标、版本来源。模型文件约 719 MB，推理参数本身约 240 MB，其余主要包括 AdamW 状态。

### 8.6 resume 的边界

`--resume` 固定读取实验目录的 last.pt，恢复完整状态，不重新载入 ImageNet。配置和冻结数据必须一致，epochs 表示总轮数；已训练 2 轮后 `--epochs 20` 是再做最多 18 轮。

随机状态包含 Python、NumPy、torch、DataLoader generator 和可用 CUDA RNG。恢复还会补写因中断未发布的 best、修复 metrics 到 checkpoint 的轮次。相同环境 CPU 测试验证了完整两轮与一轮后恢复的权重逐位相同；不同驱动/设备不保证逐位一致。

只在完整 epoch 保存，不会保存半轮。Ctrl+C 或关机会丢失当前未保存轮次，重启从最近完整轮重新做；第一轮尚未保存时还没有可恢复的 last.pt。已经达到 epoch 上限需要提高总上限；已经触发早停的 run 则拒绝直接 resume。当前没有“加载 best 再换数据微调”的 CLI，不能把旧 checkpoint 随便塞进新实验伪装成 resume。

## 9. policy.py：checkpoint 怎样变成可调用策略

`ACTPolicy` 读取 checkpoint，校验 config/schema/stats，构造不下载权重的 ACT 并严格加载 state_dict，设置 eval。推理只依赖这个文件和程序环境，不需要原始 HDF5。

`predict_chunk(images_rgb, measured_state)` 检查图片与 state 形状，按训练统计标准化 state，图片转 [0,1]，加 batch 维并前向。返回 `[K,5]` **标准化**动作块。`@torch.inference_mode()` 进一步关闭推理中的梯度相关开销。

`act()` 默认取 chunk[0]，然后 `action = normalized×action_std + action_mean` 返回物理单位。每调用一次 step 加一。每个新场景必须 `reset()`，清除 step 与时间集成历史。

`TemporalEnsembler.select(step, chunk)` 保存仍能覆盖当前时刻的旧预测。旧块起于 s，则取 `chunk[step-s]`，因此所有候选都指向同一个执行时刻。按旧到新顺序 i=0,1,... 使用 `exp(-m×i)` 并归一化：正 m **偏向较旧预测**，m=0 为平均。不是对同一块不同未来时刻求平均，也不会把有效零动作误当 padding。

比如第 0 拍预测第 1 拍动作是 0，第 1 拍新预测当前动作是 2；m=0 时输出 1。时间集成可能平滑动作，也可能延迟切换，默认关闭。它在标准化空间融合后再反归一化。

## 10. rollout.py：从动作到闭环任务结果

`evaluate()` 要求显式且不重复的非负 seeds、新的输出目录和正 max_steps。默认 CPU；云端可 `--device cuda`。先检查控制频率和相机分辨率与训练契约一致，再创建策略和 MobileManipEnv。

每个 scene seed：env.reset → policy.reset → 新建 SimActionAdapter。循环读取实测底盘速度、关节角、夹爪开度与前视图；计时模型推理；adapter.apply 施加动作；记录 requested/applied；仿真前进一步；处理夹爪计时；更新视频/GUI；调用 env.success 检查是否结束。

env.success 所用物体真值仅评分，不作为策略输入。录像使用俯视相机，模型输入仍是 front。没有调用 VISTA，也不会把 ACT 自己的动作写回训练数据。

### SimActionAdapter 的实际规则

关节目标先限于机械范围，再限制每拍目标变化不超过 `3 rad/s × 0.1 s = 0.3 rad`。夹爪输出先裁剪到 [0,1]，≥0.6 为开、≤0.4 为闭，中间保持原状态。只有开到闭的转换才开启一次 1.2 秒抓取判定；持续闭合不会每拍重新计时。底盘 v/w 经 HAL 的 command_drive 执行和限幅。stop 会停底盘并结束抓取尝试。

这里有尚未解决的表达限制：专家可以连续调用两次高层 close_gripper，而五维目标里第二次调用仍然是相同的“闭合”，无法表示新的调用边界。当前适配器是明确的基线，不能宣称完全复现专家内部抓取状态。

达到成功或最大步数后正常停底盘，再静置 3 秒，最后重新判断 success；手动关 GUI 判为未成功。每集保存轨迹 JSON；有 --video 则保存 10 FPS MP4。summary.json 包含 seed、checkpoint 哈希、控制上限、集成系数、控制适配设置、成功率和延迟。运行中 summary 只覆盖已结束的 episodes，需对照 requested_seeds 看是否全部完成。

`--label validation/test` 只是结果标签，不会自动检查 seed 是否见过。挑模型、调参数用过的场景应算验证；最终测试必须预留未参与训练和调参的 seeds。仿真失败要结合轨迹与视频判断，不能只看动作 L1。

## 11. 实验组织与问题排查

### 固定数据和评估条件

一次实验对应一个新目录，保存数据身份、划分、配置、日志和 checkpoint。先用 `prepare` 冻结合格示范，再训练；更换数据或配置时建立新实验。恢复同一实验时，`--epochs` 表示新的总轮数上限。

用固定验证场景比较候选模型和推理参数。最终测试可以直接采用独立 MuJoCo seeds，不要求额外存一份 test HDF5；但未参与训练和调参的场景需要提前预留。训练结束后直接加载 checkpoint 运行仿真，无需重新训练。

### 结合误差和行为诊断

如果 loss 下降而机器人未能完成任务，按以下顺序检查：

1. **样本与标签。** 确认图像和实测状态对应动作执行前的同一时刻，五维动作顺序、单位、归一化及 padding 与 schema 一致。
2. **各动作维度。** 检查物理单位 MAE、首拍误差，以及转向、闭爪等关键动作的预测。大量等待或直行帧可能使总体平均误差掩盖关键阶段失败。
3. **预测到执行。** 对照轨迹中的 requested / applied，检查关节限幅、夹爪阈值转换和抓取判定是否符合预期。
4. **闭环偏离。** 视频按接近、抓取、抬起、搬运、释放拆解；策略在自己造成的新观察上执行，离线动作拟合不能单独保证这些阶段都成功。
5. **控制变量。** 固定数据和验证 seeds，分别比较训练时长、时间集成或数据覆盖，避免同时修改多个因素后无法解释效果。

数据量没有保证成功的统一门槛。示范应覆盖场景变化和任务关键阶段；增加高度相似的等待画面不能替代转向、抓取与恢复行为。固定 batch、K、图像尺寸和模型时，增加轨迹主要影响存储、读取量及每轮时长，数据不会整份进入显存；改变上述网络输入设置后，应重新测量资源使用。

### 区分自动产物与自定义报告

`prepare.py` 生成数据快照、划分、统计和校验信息；`train.py` 自动生成 `config.yaml`、`runtime.json`、`metrics.jsonl`、`best.pt` 和 `last.pt`；`rollout.py` 生成评估摘要、轨迹及可选视频。曲线、额外汇总和精简权重文件需要另行生成，训练程序也不负责关闭云端实例。

参考：[ACT 原实现](https://github.com/tonyzhaozh/act)、[torchvision ResNet-18](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)、[PyTorch 对应版本安装](https://pytorch.org/get-started/previous-versions/)。模块行为以本项目源码为准。
