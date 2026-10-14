# MuJoCo · VISTA

让视觉模型通过车头相机控制移动机械臂，在 MuJoCo 中完成捡球、搬运和投放，并记录可回放的操作轨迹。当前只维护 VISTA 这一条决策路线；长远目标是把成功示范整理成模仿学习数据，再验证学习策略和实机迁移。训练、数据集导出和实机接入尚未实现。

## 当前结构

```text
brains/vista/       VISTA 决策、工具、图片与笔记管理
control/           底盘、两自由度机械臂、夹爪控制
sim/               MuJoCo 场景与机器人封装
models/            当前使用的四个 XML 模型文件
tests/             VISTA、控制与回放测试
act.md             ACT 数据接口、模型与训练设计（尚未实现）
assets/act/        ACT 设计配图
run_collect.py     采集入口，默认 VISTA、1 个 episode
hal.py             仿真硬件接口
config.py          物理参数、相机、频率与路径
recorder.py         决策日志、控制轨迹和图像记录
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

原来的 `--brain=vista` 仍可使用，其余 brain 选项已移除。

## 闭环与结果

VISTA 用 `play` 交出一批动作；采集器按顺序执行后，将批次回执、最终车头图和同一时刻的五维实测状态交回模型。状态包括肩角、肘角、夹爪连续开度、车体前向速度和左转角速度。上一批结束后的观测直接作为下一批的执行前观测，不重复采样。模型可以查旧图及其状态、检查局部、读像素、维护笔记，最后调用 `finish`。动作执行进度不确定或关键记录失败时，程序停止，避免重复执行。

肩、肘实测角可见，但单关节 `shoulder` / `elbow` 微调暂时禁止；手臂只通过 `arm_pose` 的四种预设运动。提示词、模型动作 Schema 和批次校验共同执行这项限制，底层控制能力仍保留。

正常结束后继续仿真并采集 3 秒，再进行环境判定。模型自评与环境成功标志分别保存。`data/episodes/` 保存通用轨迹和录像图片，`output/vista/` 保存图片证据、消息、调用日志、每集草稿及跨集指南；下次运行时会自动创建这些目录。VISTA 的 `frame_index.jsonl` 为每张原图保存采样时间 `timestamp_s` 和实测状态 `observation`。`actions.jsonl` 在批次结算时追加一条记录，`observation` 只存该批执行前的五维实测状态，与 `before_frame_id` 对应；保留前后帧编号，不展开第二组执行后状态。`history` 查询已结算批次；`inspect` 按 `frame_id` 查询已归档图片，因此当前图即使没有对应的动作记录也可查看。控制拍级 ACT 数据的时间对齐与导出仍待实现。

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
