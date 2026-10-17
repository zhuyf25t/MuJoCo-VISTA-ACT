# MuJoCo · VISTA

让视觉模型通过车头相机控制移动机械臂，在 MuJoCo 中完成捡球、搬运和投放。采集器记录执行前图像与实测状态、随后施加的动作，完整成功集自动导出 ACT HDF5，并持续追加 manifest。当前只维护 VISTA 这一条示范路线；ACT 网络、训练和实机接入尚未实现。

## 当前结构

```text
brains/vista/       VISTA 决策、工具、图片与笔记管理
control/           底盘、两自由度机械臂、夹爪控制
sim/               MuJoCo 场景与机器人封装
models/            当前使用的四个 XML 模型文件
tests/             VISTA、控制与回放测试
act.md             已实现的数据接口，以及待实现的 ACT 模型与训练设计
assets/act/        ACT 设计配图
run_collect.py     采集入口，默认 VISTA、1 个 episode
hal.py             仿真硬件接口
config.py          物理参数、相机、频率与路径
recorder.py         原始记录、成功集 HDF5、manifest 与崩溃恢复
playback.py         历史状态回放
replay_gui.py       回放窗口与轨迹校验
rerender.py         从轨迹重新生成录像
inspect_data.py     采集数据检查
compose_model.py    组合底盘与机械臂模型
viz.py             相机与运行窗口
_archive/          本地冻结归档，不参与运行、测试或 Git 提交
```

完整设计与工具协议见 [VISTA 说明](brains/vista/readme.md)。VISTA 仍使用 LangGraph 库；旧的 `brains/langgraph/` 策略实现已归档。

## 运行

使用 Python 3.12+，安装唯一一份依赖：

```powershell
python -m pip install -r requirements.txt
```

首次配置可复制 `.env.example` 为 `.env`，填入支持**图片输入和工具调用**的 OpenAI 兼容端点、模型与密钥。保留已有 `.env`，不要覆盖其中的本机配置。VISTA 当前要求 `stream=false`，并且必须提供正整数 `max_model_calls`；进程环境变量优先于 `.env`。

当前电脑可直接使用已有 Windows 环境：

```powershell
.\.venv-gui\Scripts\python.exe run_collect.py --gui
```

Linux / WSL 可使用 `./run`；第一次运行会通过 `uv` 创建环境并安装依赖。已有环境若缺少 VISTA 依赖，请先安装 `requirements.txt`。

常用参数：

- `--episodes 1`：采集集数，默认 1。
- `--max-decisions 64`：外层决策轮数上限；每轮内部的模型请求上限由 `max_model_calls` 单独控制。
- `--seed 0`：初始场景随机种子，后续 episode 逐次递增。
- `--keep-open`：配合 `--gui`，结束后保留窗口。
- `--no-images`：关闭通用采集图像；VISTA 仍保存必需的车头 PNG。
- `--out <目录>`：通用采集记录目录，默认 `data/episodes/`。
- `--brain-memory <目录>`：VISTA 图片、笔记和调用日志目录，默认 `output/vista/`。
- `--dataset-out <目录>`：HDF5 和单个 `manifest.json` 的根目录，默认 `data/act_v1/`；多次运行追加，不重置。采集阶段不划分训练集/验证集。

原来的 `--brain=vista` 仍可使用，其余 brain 选项已移除。

## 闭环与结果

VISTA 用 `play` 交出一批动作；采集器按顺序执行后，将批次回执、最终车头图和同一时刻的五维实测状态交回模型。状态包括肩角、肘角、夹爪连续开度、车体前向速度和左转角速度。上一批结束后的观测直接作为下一批的执行前观测，不重复采样。模型可以查旧图及其状态、检查局部、读像素、维护笔记，最后调用 `finish`。动作执行进度不确定或关键记录失败时，程序停止，避免重复执行。

肩、肘实测角可见，支持 `arm_pose` 四种预设及 `shoulder` / `elbow` 单关节微调。微调参数 `delta` 是相对当前实测角的增量（rad），范围为 ±0.5；整批动作仍须通过参数校验后执行。

`close_gripper` 固定执行 **1.2 秒（12 个控制拍）**，停止底盘、保持臂目标，一次连续完成闭合与稳定夹持判定；不提前返回、不自动延长。判定仍要求同一球连续 0.1 秒接触两侧指垫内面，开度在 0.05–0.45 且波动不超过 0.02。成功回执表示结束时仍确认夹持，抬臂后仍须直接看图验球；未确认则返回 `GRASP_NOT_CONFIRMED`。抓取时在同一 `play` 连续提交两次闭爪：第一次未确认仅允许紧邻第二次继续，第二次仍未确认则停止同批后续动作并交回新观察。两次均逐拍记录，共 2.4 秒、24 拍。首条失败、次条成功时保留首条 `failed` 和批次 `partial`，以第二条结果判断当前夹持。结束或异常均关闭新抓取判定，后续 observe/抬臂不会延迟吸附；已确认夹持仍会持续监测接触丢失。开爪仍为 0.8 秒。所有闭爪控制拍照常记录，ACT 动作字段仍是相同的五维目标。

四个预设在 `config.py` 标定，角度为 `[肩, 肘]`（rad）：`stow=[0.00, 2.20]` 空爪收纳看路，`reach=[1.13, 1.47]` 地面抓取，`carry=[0.95, 1.10]` 抬高验球及携带，`drop=[1.10, 0.79]` 平顺前伸投放。正常流程为 stow 看图对齐 → reach 闭爪 → carry 看图确认离地并运输 → 切 drop → 用底盘对准箱口、停稳再开爪。当前相机的抓取对位参考为 320×240 原图中的地面球心约 `(160,110)`、直径约 `44px`；它是固定机器人与相机的近似标定，不保证抓取，也不是持球证据。预设不负责球/箱位置对齐，也不提供持球真值。reach、carry、drop 的 TCP 前伸量约 0.54 / 0.67 / 0.72 m（相对底盘中心），高度约 0.10 / 0.31 / 0.31 m（相对地面）；这些数值不是球心位置或到相机的距离。carry/drop 下球顶部可能部分出画，仍须直接检查夹持关系与离地。`tests/test_arm_presets.py` 用真实物理步进检查抓取偏差、投影可见范围、触地、运输和姿态过渡的越沿余量及入箱；seed 1、2 各测五种抓取偏差：三种在一次 1.2 秒内确认夹持，运输与 carry→drop 期间检查约 4 cm 的球底越沿余量；另两种按未确认夹持返回失败。测试不调用模型，这些标定结果不等于自主任务成功率。

正常结束后继续仿真并采集 3 秒，再进行环境判定。模型自评与环境成功标志分别保存。`data/episodes/` 保存通用轨迹和录像图片，`output/vista/` 保存图片证据、消息、调用日志、每集草稿及跨集指南；下次运行时会自动创建这些目录。VISTA 的 `frame_index.jsonl` 为每张原图保存采样时间 `timestamp_s` 和实测状态 `observation`。`actions.jsonl` 在批次结算时追加一条记录，`observation` 只存该批执行前的五维实测状态，与 `before_frame_id` 对应；保留前后帧编号，不展开第二组执行后状态。`history` 查询已结算批次；`inspect` 按 `frame_id` 查询已归档图片，因此当前图即使没有对应的动作记录也可查看。

## 采集、HDF5 与 manifest

新控制录像直接渲染 320×240 RGB 并保存无损 PNG（车头与辅助俯视图均使用此尺寸）。每拍在物理推进前采样，再执行 0.1 秒，成功执行完这一拍才追加并刷盘 `trajectory.jsonl`。每行 `state` 是实测肩、肘、连续夹爪开度、前向速度、转向速度；`action` 是肩肘目标、二值夹爪目标、限幅后的底盘目标速度。原始 `t/t_end` 仅用于检查执行区间，`qpos/qvel/ctrl` 留作仿真诊断。最终观测单独保存在 `terminal.json`，回放会读取它，训练不会给它伪造动作。

一集正常结束且最终环境判定成功，即整集入选；保留全部控制拍，包括静止、纠错和收尾等待，不做行为质量打分或片段裁剪。仍会检查图片可读、形状正确、数值有限和控制拍连续，防止损坏的数据被当成训练数据。失败或中断仅排除出训练候选集，原始记录不删除；`--no-images` 是诊断模式，即使任务成功也不导出 HDF5。

普通任务失败（包括模型主动结束但环境未成功、耗尽 `--max-decisions`）不会终止多集采集；结算失败集后继续下一个 seed。已知的机械臂到位超时、卡住或闭爪未确认稳定夹持会返回 `failed, action_applied=true`：停止当前批次的剩余动作，接回新观察，让 VISTA 继续决策；仅连续两次闭爪中的第一次未确认允许第二次继续。全部请求的集数完成后，只要有失败，命令最终返回退出码 1，这不表示在首个失败处提前停止。模型请求异常、文件写入异常、控制器执行进度未知等运行错误仍会停止；关闭 GUI 也会提前结束。

每个入选 episode 在 `--dataset-out` 下得到一个 `episodes/episode-0000.h5`，含三个数组：

```text
/observations/images/front  uint8   [T, 240, 320, 3]
/observations/state         float32 [T, 5]
/action                     float32 [T, 5]
```

图像像素直接打包在 HDF5 中，使用无损分块压缩；没有 `/timestamp`、`/source_step`，状态和动作保留物理单位、不提前归一化。`T = action.shape[0]`，也写入文件属性和 manifest 的 `num_steps`。这是本项目面向 ACT 的格式，不是原版 ACT loader 的直接替代。以后 Dataset 负责通道转换、归一化和未来动作块抽取。

每个数据集目录只有一个 manifest，顶层为 `schema` 和 `episodes`。每个 episode 记录独立身份、源目录、seed、状态、是否入选、HDF5 路径、T 和原因。原始目录/VISTA 目录的局部编号无需相同，manifest 保存原始路径，原始 meta 保存 VISTA 路径；不同采集目录中的同名 `ep_0000` 不会覆盖同一个 HDF5。`selected_for_dataset=true` 表示合格候选；**现在不保存 train/val 归属、不自动划分**，准备训练时冻结清单并按 episode/场景分组划分。

```powershell
.\.venv-gui\Scripts\python.exe run_collect.py --gui --episodes 5 --seed 0 --out data/episodes/run_001 --brain-memory output/vista/run_001 --dataset-out data/act_v1
```

再次使用相同 `--dataset-out` 会继续追加清单和 HDF5 编号；指定新目录才是新数据集。默认 seed 每次从 0 开始，若要新场景，后续命令应显式改变 `--seed`。原始目录预留和 manifest 更新采用本地文件锁，避免多个进程覆盖或丢失条目；并发 VISTA 任务建议使用各自的 `--brain-memory`，避免共享并改写同一份 guide。

## 中断与恢复

在首次动作前，先在 manifest 登记 `status=recording, selected_for_dataset=false, h5=null`。原始 JSONL 每拍追加并 flush/fsync；每集结束才读改写 manifest，使用临时文件、fsync、原子替换，保持旧条目。成功 HDF5 也先写临时文件，完整关闭并发布后才将 manifest 改成 `ready` 和入选。

强制杀进程、崩溃或断电时，不会保证调用 Python 收尾代码。半集从开始就处于排除状态，不能依靠 finally 才排除。下次采集自动检查尚未结算的条目：正在运行的源目录仍有进程锁，因此跳过；已停止且没有完整结果的标为 `interrupted`；已经写下完整成功结果、只差导出/更新清单的则恢复发布，不重复创建条目。可单独恢复，不调用模型：

```powershell
.\.venv-gui\Scripts\python.exe recorder.py --dataset-out data/act_v1
```

此前完成的成功集不受后一集崩溃影响；半集保留已完成且刷盘的控制拍，正在执行或尚未刷完的一拍不保证完整，整个半集不会导出为训练数据。断电后的存储介质可靠性仍取决于操作系统与磁盘，不能承诺零字节丢失。

当前已实现采集与回放闭环；真实模型的视觉判断、抓球稳定性和成功率仍需单独验证。

## 验证

```powershell
.\.venv-gui\Scripts\python.exe -m pytest -q
.\.venv-gui\Scripts\python.exe replay_gui.py <episode目录或trajectory.jsonl> --check
```

测试使用模拟模型，不调用付费 API；其中仿真测试需要可用的 MuJoCo DLL 和图形环境。`pytest.ini` 将默认测试范围限定在 `tests/`，不会收集归档中的旧测试。

## 历史归档

2026-10-13 之前的旧 brain、运行结果、日志、研究资料、旧测试和压缩包已集中放入 `_archive/20261013-pre-vista/`。其中 `files/` 保持原相对路径，`before-cleanup/` 保存本次修改前的文件，`manifest.json` 记录移动清单。归档用于需要时找回资料，不作为当前代码的一部分。

临时的 64 轮实验启动脚本已放入 `_archive/20261014-vista-baseline/files/`；继续使用 `run_collect.py` 和 `--max-decisions` 参数运行实验。
