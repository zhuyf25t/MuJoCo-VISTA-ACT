# VISTA Brain：用基础 LangGraph 串起视觉、动作和记忆

实现日期：2026-10-08。**本目录已实现第一版 VISTA Brain，并接入 `run_collect.py --brain=vista`。** 旧论文、译本和官方源码快照已归档到仓库根目录 `_archive/20261013-pre-vista/files/brains/`。当前 `brains/` 仅保留 `vista/`。下文保留架构和工具约定，代码片段用于解释流程，完整实现以同目录的 Python 文件为准。

第一版的目标很具体：模型收到车头图，可以一次请求多个工具来查旧图、裁图、读像素、读写笔记；用一次 `play` 提交一批 MuJoCo 动作；外层按顺序执行整批，再返回最终图片和各子动作结果；模型沿用原来的消息历史继续。先证明这条链能闭合。暂不做上下文压缩、语义图片搜索、子图或多个模型分工。多个工具请求先顺序执行、统一反馈；不要求引入并发编程。

**State 只保留 `messages`、`model_calls` 两个字段。** 有效 play 经条件边到 END，外层执行动作；无法继续的异常直接 raise，最外层打印说明并以退出码 1 结束整个程序。动作和预期已在 AIMessage 中，工具结果另建 ToolMessage；只核对当前轮的配对，不另存动作副本或停止原因。模型调用上限从 `max_model_calls` 配置读取。

阅读顺序：先看文件布局和整张图，再看状态如何延续，最后逐个看工具协议。第 8 节用同一段抓球经历把它们串起来。

运行前安装项目依赖，在项目 `.env` 或进程环境中配置 `max_model_calls`，以及支持**图片输入和工具调用**的 OpenAI 兼容 Chat Completions 端点、模型和密钥。密钥配置沿用 `LLM_*` / `OPENAI_*` 名称，示例见项目 `.env.example`；不要把凭据写进指南、草稿或代码。

```bash
python -m pip install -r requirements.txt
python run_collect.py --brain=vista --episodes 1 --gui --max-decisions 8
```

当前电脑已经准备好的 Windows 环境也可以直接运行：

```powershell
.\.venv-gui\Scripts\python.exe run_collect.py --brain=vista --episodes 1 --gui --max-decisions 8
```

`--max-decisions` 限制外层决策轮数；`.env` 中的 `max_model_calls` 限制每轮内部的模型请求次数，两者独立。`--brain-memory` 可指定图片与笔记根目录，默认 `output/vista/`；`--out` 仍指定通用采集记录目录。两个根目录分别分配 episode 编号，不要求编号相同，VISTA 的 `episode_start` 日志记录对应采集目录。

第一版使用完整 JSON 回复，要求 `stream=false`。每次模型调用只发送一次 POST，不自动重试、不探测其他 URL。`OPENAI_BASE_URL` 应包含服务所需的完整 API 前缀（例如 `/v1`），也可直接配置到 `/chat/completions`。保留服务要求回传的 reasoning 字段和原始 tool_calls；不自动切换到其他模型或 API 协议。工具协议参考 [Chat Completions 接口](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

本地验证使用模拟模型和真实 MuJoCo 控制器，不会调用付费模型：

```bash
python -m pytest tests/test_vista.py -q
```

这里验证了消息、工具、动作与图片的闭环；真实模型的视觉判断和抓球成功率仍需另行实测。

## 1. 文件怎么摆，谁负责什么

下面是**已实现的布局**。`output/vista/` 内的图片、笔记和日志在运行时创建。**`*` 表示包含可复用的 VISTA 框架机制，不表示整个文件已经与 MuJoCo 解耦；星号不是文件名的一部分。**

```text
mujoco/
├── run_collect.py                  已有：外层采集循环，执行动作、推进仿真
├── control/
│   └── tools.py                    已有：底盘、机械臂、夹爪的真实执行函数
├── brains/
│   └── vista/                     新 brain 的源代码目录
│       ├── readme.md               本设计稿
│       ├── base.py                 Brain 和 Decision 接口
│       ├── __init__.py             导出 VistaBrain
│       ├── * brain.py              接收新观察，保存 state，交出动作，接回结果
│       ├── * graph.py              两个节点及其条件边
│       ├── * state.py              两个 State 字段、动作批次和回执的格式约定
│       ├── * model.py              模型接口、工具定义绑定、图片消息的转换
│       ├── * tools.py              九个工具的定义、参数检查和分发
│       ├── * storage.py            原图索引、PNG、笔记和日志读写
│       ├── * settings.py           读取 max_model_calls，管理尺寸和字符数等配置
│       └── prompts/
│           └── system.txt          当前 MuJoCo 任务、车头相机和工具使用说明
├── tests/
│   └── test_vista.py               消息、工具边界、故障退出及真实 MuJoCo 验证
└── output/
    └── vista/
        ├── guide.md               同一环境设定下，可跨 episode 保留的指南
        └── ep_0000/                一次独立仿真任务
            ├── working.md         本 episode 的当前计划、假设和未决问题
            ├── frames/
            │   ├── f000000.png    初始车头原图
            │   ├── f000001.png    第一批动作结束后的车头原图
            │   └── ...
            ├── frame_index.jsonl  图片编号 → 相对文件路径、尺寸、来源批次
            ├── actions.jsonl      每次 play 一条：整批请求、子动作结果、预期、前后帧
            ├── calls.jsonl        所有工具的请求与结果，包括失败的检查
            ├── messages.jsonl     消息轨迹，供人检查；不是第一版的自动恢复入口
            └── outcome.json       本集结局与外层评分，仅供人检查，不作为模型输入
```

这里的“可复用”是指这份实现换一个仿真或游戏时可以沿用的机制，不代表原论文规定必须有这些文件。当前版本仍是 MuJoCo 项目中的实现，尚不是拿到任何环境就能直接运行的通用库。

| 文件 | 可以复用什么 | 换环境时需要检查或替换什么 |
| --- | --- | --- |
| `graph.py`、`state.py` | model/tools 循环、消息累计、调用预算、工具请求与结果配对 | 新环境是否继续采用 play 提交批次、外层回传结果的约定 |
| `model.py` | 文本/图片/工具消息转换和模型请求 | 当前仅适配 OpenAI 兼容 Chat Completions；换 API 协议需要另写适配 |
| `brain.py` | 同一 episode 的上下文延续、动作交接和结果接回 | 本项目的 Brain/Decision 接口、车头观察格式和结束判定 |
| `tools.py` | 查图、读像素、history、笔记读写及参数检查 | `action_schemas()` 中固定的十个 MuJoCo 动作、姿态、单位和范围 |
| `storage.py`、`settings.py` | PNG 编号与归档、笔记作用域、JSONL 日志、预算和长度配置 | `images.front` 观察入口、车头相关初始笔记，以及项目 config 和默认路径 |

`prompts/system.txt`、外层 `run_collect.py` 和 `control/tools.py` 当前承担具体任务和环境的适配，因此没有把它们整体标为通用。提示词中的查图、记笔记、提交批次等协议可以沿用，但车头相机、抓球任务和机器人动作说明需要随环境调整。迁移时主要替换**观察入口、动作定义与执行器、任务提示词**，通常不必重写 LangGraph 的两个节点和转移边。

这里先把图片和笔记的普通文件操作集中在 `storage.py`，不必为每一项功能再建一层框架。等它真的变大时再拆。裁图是由原图生成的派生图片，可以直接进入消息，也可以放临时缓存；**不把裁图冒充新的环境帧**。

`output/` 已被本项目 Git 忽略。源代码与运行时生成的记忆分开，便于清楚地知道“改了程序”还是“模型又记了一条经验”。环境模型、相机或控制参数有变化时，应选择新的记忆目录或人工检查指南，不能默认旧经验仍然成立。第一版先由运行配置指定目录。

### 1.1 分清三层

```text
run_collect.py：拿图 → 调 brain → 执行整批动作 → 拿批次结束后的图
                         │                 │
                         ▼                 │
VistaBrain：维护同一个 state、图片档案和笔记 ◀┘
                         │
                         ▼
LangGraph：model → tools → model → ... → 交出一个动作批次
```

`inspect` 等工具在 brain 内部完成。真正推进 MuJoCo 的动作仍由外层 `ToolLayer.execute()` 完成。这样可以沿用项目已有的控制器、仿真 tick 和录像逻辑。

图不是整个世界的容器。模型客户端、文件目录和图片索引可以由 `VistaBrain` 持有，节点通过普通 Python 对象使用它们；没必要把网络连接、MuJoCo 对象或整本图片库塞进 State。

## 2. 先看整张 LangGraph

### 2.1 三种“轮”

| 名称 | 本文含义 | 举例 |
| --- | --- | --- |
| episode | 从仿真初始化到任务结束的一整次实验 | 捡球并放入箱子 |
| 外层决策轮 | 一次 `brain.decide(...)`；内部可请求模型很多次 | 查图、记计划，再交出一批动作 |
| 模型调用 | 一次向模型 API 发请求 | 模型在一条回复中提出两个 `read_pixels` |

另有“批次序号 `batch_id`”“批内子动作序号 `index`”和“图片编号 `frame_id`”。一次 `play` 对应一个批次，批内可以有多个子动作；`inspect` 只增加工具调用记录，不增加批次序号，也不推进仿真。不要把这些计数都命名成 `step`。

### 2.2 只有两个业务节点

| 节点 | 它具体做什么 | 返回给 State 的更新 |
| --- | --- | --- |
| `model` | 检查本轮预算，向模型发送消息和工具定义，取得一条 AIMessage | 新模型消息、递增的 `model_calls` |
| `tools` | 处理本条回复中的工具请求；执行本地工具，或校验待外层执行的批次 | 本地调用返回 ToolMessage；有效 play/finish 返回空更新 `{}` |

`START`、`END` 是入口和出口标记。inspect、像素读取、笔记读写等是 `tools` 调用的普通函数，不必各建一个节点。没有额外的计划、预期或反思节点，这些内容由模型在同一对话中表达。

```text
START → model → tools
          ▲       │
          │       ├─ 本地工具结果或可修正错误 ──┘
          │       │
          │       └─ play/finish 校验通过 → END → 外层处理

节点中出现无法继续的异常：raise → 退出本次 invoke
                                   ↓
                         最外层打印说明，以退出码 1 结束整个程序
```

`END` 是正常返回：这时消息里留着一个经过 tools 校验、等待外层结果的 play 或 finish。异常从节点向外传播，不假装成正常 END，也不需要在 State 中保存停止原因。

### 2.3 所有边的具体判断

| 从哪里 | 条件 | 去哪里 |
| --- | --- | --- |
| `START` | 一次 decide 开始 | `model` |
| `model` | 得到合法的、含工具请求的 AIMessage | `tools` |
| `tools` | 本地结果或可修正错误已经追加 | `model` |
| `tools` | 整批 play 或 finish 校验通过，尚无外层结果 | `END` |
| 任一节点 | 发生无法继续的异常 | 抛出，不再走图的下一条边 |

普通节点返回状态更新；条件边返回下一个节点名或 END。这里通过消息的明确形状判断，无需新增路由字段：有效 play/finish 的 tools 返回 `{}`，末尾仍是原 AIMessage；本地结果或参数错误会追加 ToolMessage（必要时再追加图像消息），末尾就不再是 AIMessage。

```python
def after_tools(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        calls = last.tool_calls
        if len(calls) != 1 or calls[0]["name"] not in ("play", "finish"):
            raise RuntimeError("工具节点未给本地调用返回结果，程序终止。")
        # tools 只有在交接调用校验成功时，才会保留这个消息尾部形状。
        return END
    return "model"

builder = StateGraph(VistaState)
builder.add_node("model", model_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "model")
builder.add_edge("model", "tools")
builder.add_conditional_edges("tools", after_tools,
                              {"model": "model", END: END})
graph = builder.compile()
```

模型没有发出工具请求、调用 ID 缺失或重复时，model/tools 校验抛出带说明的异常；不会把普通正文当成动作或成功证明。API 故障和预算耗尽也走外层异常处理。程序记录仍可说明具体原因，只是不把原因复制进 State。

允许一条模型回复提出多个本地工具调用，例如两个 read_pixels，或 inspect 加 history。按顺序执行、为每项返回配对结果，收齐后再调用模型。依赖结果才能决定参数的调用，需要等下一次模型请求。独立读取以后可以并行；同一文件的写入保持顺序。

含 play 的回复只包含这一个工具调用，但 `actions` 内可包含多个子动作；finish 也独占一条回复。先查证，再提交整批。这是本版的反馈顺序约定，不是所有工具一次只能调用一个。

### 2.4 为什么有效 play 后直接 END

当前项目由 brain 返回 `list[Decision]`，外层 `run_collect.py` 再调用控制器。tools 校验 play.actions 成功后返回空更新 `{}`，条件边去 END。批次和预期已经在 AIMessage 的工具参数中，不必复制到别的字段。

此时控制器还没有执行，不能构造“动作已成功”的 ToolMessage。外层执行整批后，再将真实结果和最终图作为该 play 的响应追加。收到结果之前，不调用模型。

异常处理也很直接：节点 `raise RuntimeError("具体说明")`，最外层打印并退出。普通 Python 异常退出与 LangGraph 正常 END 分开处理，不使用 interrupt、Command 或数据库恢复。

## 3. State 是什么，需要哪些字段

State 是节点之间传递的 Python 字典；模型不会自动看到整份字典。第一版只保留两个字段：

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph.message import add_messages

class VistaState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
```

| 字段 | 作用 | 更新方式 |
| --- | --- | --- |
| `messages` | 用户任务、模型请求、工具结果、模型看过的图片等连续上下文 | 用 add_messages 合并新增消息 |
| `model_calls` | 当前一次 decide 已调用模型多少次，限制内部循环 | 普通替换；下一次正常 decide 开始时归零 |

`max_model_calls` 是运行配置，读一次放在 VistaBrain/settings 中，不放 State，也不发给模型。批次动作、预期和检查目的已经在 messages 的工具请求参数中，不保存副本。当前图片编号、存储目录等由图外对象管理。

### 3.1 `add_messages` 合并消息，不把请求变成结果

`add_messages` 合并各种消息；新消息 ID 不存在时追加，存在时更新对应消息。它不压缩对话，也不自动写磁盘。节点只返回增量，不直接 append 到传入的 State：

```python
return {"messages": [reply], "model_calls": state["model_calls"] + 1}
```

这里有三个不同的东西：

- `AIMessage.id`：这一整条模型消息的 ID。
- `AIMessage.tool_calls[i].id`：这条模型消息中某一次工具请求的 ID。
- `ToolMessage.tool_call_id`：说明这个结果是在回复哪一次工具请求。

**工具请求在 AIMessage.tool_calls 中；工具返回值在新建的 ToolMessage.content 中。** 不是先为请求建一条 ToolMessage，再为结果建第二条。一条 AIMessage 可发出多个请求，各请求分别对应一条结果消息。消息合并语义见 [LangGraph 消息实现](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py)。

### 3.2 跨两次 decide 延续，检查当前轮即可

同一个 VistaBrain 实例保存 `self.state`，图不配置 checkpointer。以下示意省略外层执行和异常处理，完整顺序见第 7.4 节：

```python
# 新 episode 初始化一次。
self.state = {"messages": initial_messages, "model_calls": 0}

# 本轮开始前，上一轮的 play/finish 结果必须已配对接回。
check_current_round_results(self.state["messages"])
self.state["model_calls"] = 0
self.state = self.graph.invoke(
    self.state, {"recursion_limit": 2 * self.max_model_calls + 4}
)
# 正常返回：末尾是 tools 已校验的 play/finish AIMessage。
# 从它读取请求，交给外层；此时还没有 ToolMessage 结果。

# 外层稍后提交对应结果。
self.state["messages"] = add_messages(self.state["messages"], result_messages)
```

`check_current_round_results` 是 `state.py` 中的普通协议检查函数，不是 LangGraph 内置函数，也不是测试。只看最近一条 AIMessage 以及它后面的结果，检查各 ID 有且只有一份匹配结果；初始时没有模型请求则无需检查。不扫描每轮以前的整段历史，不维护全局未完成请求列表。

外层回执也与当前这条经过校验的请求配对；两次 decide 不并发。任何错配都抛异常，不猜测、不继续请求模型。保留当前轮 ID 检查，是为了避免把甲工具的结果回复给乙工具；不需要名为 unresolved_calls 的全历史扫描函数。

同一进程中，图返回的完整状态成为下一次输入，节点在图内仍只交增量。进程关闭后，内存 State 消失；PNG、笔记和日志仍在，但第一版不从崩溃处恢复仿真。异常终止时也不承诺 graph.invoke 会返回最后的 State，错误现场以已保存的工具/执行日志为准。

未来换成 checkpointer 时，需要改用其会话标识和输入增量规则，不能两套保存方式重复叠加；图存档本身也不保存外部 MuJoCo 世界。参考 [LangGraph 持久化说明](https://docs.langchain.com/oss/python/langgraph/persistence)。

### 3.3 图外维护的内容

| 内容 | 放在哪里 | 模型怎样取得 |
| --- | --- | --- |
| 当前 episode ID、图片根目录、当前帧 ID、批次序号 | VistaBrain 的运行对象 | 当前图和批次结果中的身份元数据 |
| 模型客户端、稳定提示、工具定义、max_model_calls | 普通对象和配置 | 提示和工具定义随请求发送；预算配置不发给模型 |
| 全部归档图片和索引 | PNG + frame_index.jsonl | 当前图直接提供，旧图通过 inspect |
| 指南、草稿 | guide.md、working.md | 初始化和对应读写工具 |
| 客观批次历史 | actions.jsonl | history 返回选定范围 |
| 仿真坐标、关节回读、成功真值 | 现有环境/记录器 | 第一版视觉 brain 不作为判断依据读取 |

State 不必装下文件本体、连接和整个仿真世界。节点通过普通对象访问所需资源即可。

### 3.4 max_model_calls 从哪里来，如何限制

配置名称使用用户指定的 **`max_model_calls`**。它是与 API key 并列的独立配置项，不是 API key 字符串的一部分，也不是模型 API 的请求参数。例如，在保存 API 配置的项目 `.env` 中添加：

```dotenv
max_model_calls=12
```

12 只是配置示例，不是代码里固定的上限。settings 在启动时先读取进程环境变量 `max_model_calls`；若未设置，再从项目现有 `.env` 文件的同名项读取。这里使用 dotenv_values 读取文件，不会自动把文件中的键导出到进程环境，因此不能只调用 os.getenv 就认为读到了 `.env`。VISTA 模型连接配置也采用进程环境优先、项目 `.env` 其次的顺序。

该项必填，去掉首尾空白后必须是正整数字符串。缺失、空值、0、负数或非整数时，在任何 API/动作执行之前抛出 `RuntimeError("max_model_calls 必须配置为正整数，例如 12")`。最高优先级来源存在但无效时直接报错，不悄悄改用另一个来源。用户自行填写配置；实现只更新 `.env.example`，不改凭据文件。

每次 decide 开始将 model_calls 归零；一次 decide 内可能有很多次查询和模型请求。model 节点在每次 API 请求前检查：

```python
if state["model_calls"] >= self.max_model_calls:
    raise RuntimeError(
        f"本轮模型调用已达到 max_model_calls={self.max_model_calls}，程序终止。"
    )
reply = request_model(state["messages"])
return {"messages": [reply], "model_calls": state["model_calls"] + 1}
```

一次模型回复中有多个 tool_call，仍只计一次模型请求。可修正参数错误后的下一次请求继续计数。第 N 次请求产生有效 play 可以正常交接；若它还需要继续查询，第 N+1 次请求前就抛错。调用失败直接终止，日志记录失败的请求序号，不做隐藏 API 重试；接入时需要关闭旧适配器或 SDK 的自动重试。

本版两节点图将 recursion_limit 设置为 `2 * max_model_calls + 4`，使图步数保护不会先于配置预算误停。它是图调度上限，不是另一个模型次数配置；以后增加节点时再相应调整。

### 3.5 无法继续的异常：外层打印后退出整个程序

使用普通 `RuntimeError` 和人能看懂的说明即可，不建立十几种自定义异常类。例如：

```python
raise RuntimeError("批次 6 的第 2 个动作执行进度未知，剩余动作未执行，程序终止。")
```

API 失败、上下文超限、存储故障、调用 ID 错配、无工具调用、预算耗尽、无法取得最终图等不可继续的情况，都抛出异常。已知错误需要转述时用清楚的固定说明；不要把含密钥的请求头或原始请求全文拼进异常文本。

最外层负责捕获一次、打印说明、退出码 1。异常应离开整个 episode 循环，不是仅 break 当前一集然后运行下一集：

```python
import sys

if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    raise SystemExit(exit_code)
```

上面是入口的简化示意，实际由 `run_collect.cli()` 捕获并返回退出码 1。资源通过 with/finally 释放；能保存的实际执行记录先保存，再向外抛出，保存失败也不能被当成正常完成。终止后不发新 API、不执行剩余子动作、不进入下一个 episode、不进入 keep-open 等待，也不自动重试。

普通模型参数错误仍属于可修正的工具返回，例如裁剪越界：返回一条 ToolMessage，让模型自己决定下一步。它不需要 raise。区分的是“请求可以改正”和“宿主已经无法可靠继续”，而不是把所有 ok=false 都当成崩溃。正常 finish 走 END，外层处理任务结束及评分，不抛异常。

## 4. 模型每次实际收到什么

### 4.1 初始请求

```text
SystemMessage：行为说明、工具使用规则、图像坐标约定
HumanMessage：任务 + 初始帧身份/尺寸 + 初始图
              + guide.md 当前内容 + working.md 当前内容
tools 参数：九个工具的结构化名称、描述、参数 Schema
```

两份笔记在初始化主动提供，是本设计的简化选择。原项目主要提供读写接口，不要求每次新动作后重新发送两份笔记。后续只将新结果追加到消息末尾；旧内容仍留在对话中。

三类说明分开提供：system.txt 写稳定的行为规则；本次任务目标来自初始 HumanMessage 的 task_text；工具名、作用、参数类型/单位/范围写在 tools.py 的工具描述和 Schema 中，经 model.py 的 API tools 参数发送。不能只给 action 名称枚举，让模型猜各动作的含义。

尤其是 play 的 actions，每种子动作都有说明：forward/back 是前进/后退，turn_left/right 是原地转向，四个 arm_pose 预设分别做什么，以及 seconds、delta 的含义和约束。`tools.action_schemas()` 从外层提供的控制定义生成 Schema，并补充严格的参数范围、姿态枚举和视觉反馈说明。`prompts/system.txt` 说明稳定规则，`model.ChatModel` 将这些工具定义随每次请求发出。

稳定 system 文本可以保存在 `VistaBrain` 中，每次请求临时拼在 `state["messages"]` 前面，不必重复追加进历史。工具清单也是模型接口的独立参数，不能只把 Python 函数名写在用户文本里就期待它自动调用。

### 4.2 一次普通工具调用

先区分请求和返回：模型输出的是 AIMessage，工具运行后程序创建新的 ToolMessage。下面是消息对象的简化示例：

```python
request = AIMessage(
    content="",
    tool_calls=[{"name": "read_guide", "args": {}, "id": "c2"}],
)
result = ToolMessage(
    tool_call_id="c2",
    content='{"ok": true, "data": {"content": "暂无经过验证的经验"}, "error": null}',
)
```

request 和 result 按顺序加入 messages。ToolMessage.content 就是返回内容；不再另起第二条 ToolMessage 放相同结果。失败时也可在这条结果里放结构化错误；遇到致命异常则停止运行，不伪造成功结果。

```text
AIMessage：tool_calls=[
  {id: "c2", name: "inspect",
   args: {question: "球是否位于两指之间？", views: [...]}}
]
ToolMessage：tool_call_id="c2"，返回元数据和检查得到的图
下一次模型请求：此前所有消息 + 以上两条新消息
```

一次请求多个工具时，对话顺序是：

```text
AIMessage：tool_calls=[read_pixels(id=c2, ...), read_pixels(id=c3, ...)]
ToolMessage：tool_call_id=c2，第一张图的采样结果
ToolMessage：tool_call_id=c3，第二张图的采样结果
下一次 model：同时看到两项结果，再判断颜色是否接近
```

一条工具请求对应一条结果；一个 `play` 批次也只对应一条汇总结果，子动作结果放在其内部。多工具请求没有全部配对之前，不继续请求模型。

`question` 留在模型自己发出的参数里，下一次自然还能看见。它是可见的检查目的，不是隐藏思维链。工具只按参数读图，不再调用另一个模型回答问题。

保存的是 API 提供且可继续使用的消息，并不是模型内部的隐藏状态或 KV cache。若模型接口要求回传特定响应字段，由 `model.py` 按该接口协议保留；不能只保留 `content` 而丢掉 `tool_calls`，也不要求模型公开内部推理。

### 4.3 图片真的进入请求，而不是只给文件名

`frame_id="f000001"` 只是检索编号。模型要看图，`model.py` 还必须读取图片字节并转换成目标 API 支持的图像内容块。

逻辑上的工具结果可以同时包含文字和图像。本版统一采用便于 OpenAI 兼容端点接收的方式：先发本条 AIMessage 对应的**全部文字 ToolMessage**，再发带调用 ID 和图片身份的图片 HumanMessage。不会把 user 图片插进尚未配齐的多个工具结果中。检查工具的图片标明“环境未变化”；play 的图片标明“该批次结束后的观察”。图片以真实 PNG 内容块发送，不是只给模型一个本地路径。

第一版需要选一个实际支持**图像输入和工具调用**的模型及端点。现有 OpenAI 兼容接口形式可以复用，但“兼容接口”并不证明任何模型都支持这些能力；接入时必须用实际端点验证。本文不默认现有 `.env` 中的模型自动满足全部条件。

本版不压缩消息。历史图片进入过消息后，也会继续占用请求上下文；外部图片库不会自动降低 token 用量。接口报告上下文超限时抛出带清楚说明的异常，由外层打印并退出，不悄悄删旧图。压缩策略留到闭环成功之后。

## 5. 工具清单和所有工具共用的规则

第一版向模型注册 **9 个工具**：

| 工具 | 作用 | 是否交出本轮控制权 |
| --- | --- | --- |
| `inspect` | 取出一张或几张历史图，可裁剪放大 | 否 |
| `read_pixels` | 从原图指定区域读取 RGB 采样值 | 否 |
| `history` | 按 play 批次查客观动作记录和对应帧编号 | 否 |
| `read_guide` | 读相对稳定的环境经验 | 否 |
| `write_guide` | 完整替换环境经验 | 否 |
| `read_working` | 读当前草稿 | 否 |
| `write_working` | 完整替换当前草稿 | 否 |
| `play` | 提交一个包含多个子动作的 MuJoCo 批次 | 整批校验通过后，是 |
| `finish` | 提出结束当前 episode，并给出模型自评 | 校验通过后，是 |

前八项对应原版常用工具的功能；`finish` 是为本项目的 `done(success)` 接口增加的入口。原版另有用于压缩交接的 `save_compact_checkpoint`，本版不注册。它不是 LangGraph 的 checkpointer，两者不要混淆。

下面所有具体阈值和返回格式都是**我们第一版的约定**，不是声称官方仓库逐字段如此实现。

### 5.1 参数检查

Schema 告诉模型怎么填参数，Python 校验才决定是否接受。`TypedDict` 本身不会在运行时替我们检查。所有工具统一采用：

- 拒绝未知工具名和未知参数，不静默删除模型填错的字段。
- 字符串有长度限制，要求非空的字段还要检查 `text.strip()`。
- 整数不接受 `True` / `False`；坐标不接受浮点数。数值参数拒绝 NaN 和无穷大。
- 不接受模型给出的任意磁盘路径。图片通过已登记的 `frame_id` 查找，笔记路径由程序固定。
- 参数校验完再执行。多张 `views` 先全部检查；有一张无效，整个请求返回错误，不混合返回部分成功。

### 5.2 结果和错误格式

非动作工具的成功结果统一形如：

```json
{"ok": true, "data": {"...": "该工具的数据"}, "error": null}
```

可修正错误统一形如：

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "REGION_OUT_OF_BOUNDS",
    "message": "区域右边界 720 超过原图宽度 640",
    "can_retry": true,
    "hint": "使用该帧的原图坐标，并令 x + width <= 640"
  }
}
```

`can_retry=true` 的意思是模型可以修改请求后再调用，不是程序偷偷重复执行原请求。错误也必须写成匹配原 `tool_call_id` 的 ToolMessage；保留原来的错误调用，模型才能知道哪里错了。动作执行结果另有 `action_applied` 字段，见第 6.8 节。

例如原图损坏、存储不可用或物理动作进度未知，重复同一请求并不能可靠继续，诊断记录可标为 `can_retry=false`。这类情况保存能保存的事实后抛异常，不再把重试决定交给模型。can_retry 是结果含义的说明，异常退出由 Python 流程决定，不增加控制字段。对可修正错误只返回结果即可，不需要额外追加“请重试”的消息。

每次 decide 的调用上限由 `max_model_calls` 决定，参数错误后的下一次模型调用同样计数。达到上限后若还需请求模型，直接 raise 说明原因；不在 State 里登记停止状态。配置读取、计数和图步数上限见第 3.4 节。

每条模型回复可以有 **1–8 个本地工具调用**；这个数量上限只限制一次反馈的大小，不禁止多工具。每项各自校验、顺序执行并返回结果；某项参数错误不影响其他独立请求得到结果。模型必须等全部结果返回，才能据此发出依赖这些结果的新请求。第一版没有同时执行多个写文件操作。

以下两种情况整组不执行，并为每个有效唯一 ID 返回错误，让模型修正：超过 8 项 → `TOOL_CALL_LIMIT`；将 `play`/`finish` 与其他工具混合，或同时提交多个交接工具 → `HANDOFF_MUST_BE_ALONE`。`play.actions` 是一次调用的批内动作列表，不受“本地工具调用个数”的规则替代。

调用 ID 缺失、当前回复中重复，或回执无法匹配当前请求时，直接抛出说明配对错误的异常。若本地工具执行到一半发生不可继续的故障，在可用日志中记录已完成项、故障项和后续未执行项；不重试整组、不撤回已写入的笔记，然后向外 raise。既然已经终止，不必为了再次调用模型而拼凑未执行工具的结果；不能把这段未完成对话继续发给模型。

## 6. 逐个工具的完整约定

### 6.1 `inspect(question, views)`：带着问题回看图片

输入示例：

```json
{
  "question": "整批抓取动作结束后，球与两指的相对位置改变了吗？",
  "views": [
    {"label": "批次前", "frame_id": "f000004",
     "region": {"x": 220, "y": 260, "width": 160, "height": 120}},
    {"label": "批次后", "frame_id": "f000005",
     "region": {"x": 220, "y": 260, "width": 160, "height": 120}}
  ]
}
```

示例假设两幅原图都足够大，实际必须以索引中的宽高为准。

| 参数 | 必填 | 内容与限制 |
| --- | --- | --- |
| `question` | 是 | 这次想通过图片确认什么；1–1024 字符，不能全为空白 |
| `views` | 是 | 按希望查看的顺序排列，1–4 项 |
| `views[i].label` | 是 | 这张图在比较中的角色，如“批次前”；1–128 字符 |
| `views[i].frame_id` | 是 | 当前 episode 已归档的原图编号 |
| `views[i].region` | 否 | 省略表示整张原图；提供时必须含整数 `x,y,width,height` |

区域采用原图坐标：左上角 `(0,0)`，x 向右，y 向下。范围是 `[x, x+width)`、`[y, y+height)`；要求 `x,y>=0`、`width,height>=1`、`x+width<=W`、`y+height<=H`。显式传 `null` 不等于省略。

上述坐标约定、inspect 的回看/裁剪用途及“不推进环境”都必须写进模型实际收到的工具 description 与 region 参数说明；只写在 README 不够。尺寸由图片元数据提供，Python 负责实际越界检查。

返回内容：每张图的 `label`、`source_frame_id`、原图尺寸、实际选取区域、输出尺寸、真实图片内容；另有 `current_state_unchanged=true`。结果顺序与 `views` 一致。

裁剪后按比例将长边调整为 1024 像素，用最近邻插值；短边至少为 1 像素。第一版采用固定规则，不提供额外的缩放参数。返回 `rendered_size`，让模型知道展示图和原图不是同一坐标系。原始 PNG 不变，工具也不会生成“球抓住了”的判断。

错误边界：未知帧 → `FRAME_NOT_FOUND`；缺参数或类型不对 → `INVALID_ARGUMENT`；越界 → `REGION_OUT_OF_BOUNDS`；超过四幅图 → `TOO_MANY_VIEWS`。这些都可修正后重试。索引有记录但文件损坏/丢失 → `FRAME_UNAVAILABLE`，第一版按存储故障停止，不能改用另一张图冒充。

### 6.2 `read_pixels(question, views)`：读取像素，不解释像素

名称使用原项目的复数 **`read_pixels`**。它与 `inspect` 分开：前者返回数字，后者返回图片。

```json
{
  "question": "这一小块区域的采样像素分别是什么颜色？",
  "views": [
    {"label": "疑似球的区域", "frame_id": "f000005",
     "region": {"x": 300, "y": 320, "width": 20, "height": 10},
     "rows": 2, "columns": 4}
  ]
}
```

| 参数 | 必填 | 内容与限制 |
| --- | --- | --- |
| `question` | 是 | 检查目的，1–1024 字符 |
| `views` | 是 | 1–4 个采样区域 |
| `label`、`frame_id` | 是 | 与 `inspect` 相同 |
| `region` | 是 | 原图上的矩形，四项整数，范围检查与 `inspect` 相同 |
| `rows` | 是 | 采样行数，整数，`1 <= rows <= region.height` |
| `columns` | 是 | 采样列数，整数，`1 <= columns <= region.width` |

一次请求所有区域的 `rows * columns` 总和不能超过 **1024**。这限制返回长度，不是把整张图的每个像素都发给模型。

采样规则很直接：把矩形等分成若干行、若干列，在每个小格中心取一个原图像素。第 `r` 行、第 `c` 列从 0 开始：

```text
sample_x = x + floor((2*c + 1) * width  / (2*columns))
sample_y = y + floor((2*r + 1) * height / (2*rows))
```

工具从归档的 RGB PNG 读取 `(sample_x, sample_y)` 的 `[R,G,B]`，每个通道是 0–255 整数。第一版直接返回坐标与 RGB 数组，按行从上到下、按列从左到右排列。例如：

```json
{
  "ok": true,
  "data": {
    "current_state_unchanged": true,
    "sample_count": 1,
    "views": [{"label": "单点", "source_frame_id": "f000005",
               "samples": [[{"x": 300, "y": 320, "rgb": [120, 180, 40]}]]}]
  },
  "error": null
}
```

上面的单点返回对应 `region={x:300,y:320,width:1,height:1}`、`rows=columns=1`；不是前面 2×4 请求的完整返回。颜色数值仅作格式示意。

原版将颜色编码成调色板和符号行，适合离散游戏图；MuJoCo 图像颜色更连续，本版先直接返回 RGB，避免额外维护颜色符号表。工具不算物体距离，不做目标识别，不把相似颜色自动合并。

错误边界：帧和区域错误同 `inspect`；行列数非正整数或超出区域尺寸 → `INVALID_GRID`；总采样数过多 → `SAMPLE_LIMIT`。这些返回错误后让模型改参数。读取的是原图像素，不能用裁剪放大后的展示坐标，也不能声称插值后的像素是新的环境证据。

### 6.3 `history(start_batch, end_batch, limit)`：按 play 批次找事实和图片编号

历史查询不调用模型，直接读本 episode 的 `actions.jsonl`。**一条历史记录对应一次交给外层执行的 `play` 批次**，内部列出所有子动作和各自结果。它不是所有 tool_call 的列表，`inspect`、像素查询和笔记读写只进入 `calls.jsonl`。

| 参数 | 必填 | 内容与默认值 |
| --- | --- | --- |
| `start_batch` | 否 | 起始批次序号，整数，至少 1，包含此序号 |
| `end_batch` | 否 | 结束批次序号，整数，至少 1，包含此序号 |
| `limit` | 否 | 最多返回多少个批次，默认 10，范围 1–50；不是动作执行上限或图片数 |

无范围时返回最近 `limit` 个批次，返回数组仍按 `batch_id` 从小到大排列。有范围时先筛选；若超限，保留范围内最后 `limit` 条，并返回 `truncated=true`、`returned_range`，便于继续查更早的范围。`start_batch > end_batch` 是参数错误；未来范围或没有匹配项返回成功的空列表。每条记录包含完整的子动作结果，不把一个批次截成半条。

limit 是可选的输出量控制，不是主流程必需字段，也不会进入 State。早期历史很短时可忽略它；历史变长后用它避免一次取回过多内容。模型无需每次填写，有需要再调小或调大，结果明确告知是否截断。

每条公开记录返回：`batch_id`、`tool_call_id`、请求的 `actions`、批次 `expectation`、`before_frame_id`、`after_frame_id`、整体 `execution_status` 和 `action_results`。子动作在批内按 `index=1,2,...` 编号，其状态区分完成、失败、进度未知和未执行；具体结构见第 6.8 节。

批次编号由外层在执行交接时分配，执行请求先进入 `calls.jsonl`，结算后将一条批次记录写入 `actions.jsonl`。部分执行或已交接后取消的批次也保留，不伪装成全部成功；参数校验阶段被拒的 `play` 不占批次编号。`finish` 是结束请求，也不占批次编号。进程中途崩溃而未结算的请求只能在调用日志中看到，第一版不自动补造完整历史。

例如第 6 批包含 `reach → close_gripper → carry` 三个动作，`history(start_batch=6, end_batch=6, limit=10)` 返回一条批次记录和三个子动作结果，而不是三条 history。随后模型可以凭这条记录的前后帧编号去看图。

**history 不自动返回图片。** 模型先找到 `after_frame_id`，再通过 `inspect` 查看。第一版不提供自然语言搜索，不自动生成图片语义标签，也不查询其他 episode。

返回的记录不含底层持球真值、坐标或调试提示。文件解析损坏 → `HISTORY_UNAVAILABLE` 并停止，不把损坏误报成“历史为空”。

### 6.4 `read_guide()`：读取稳定经验

参数为空对象 `{}`。返回 `content`、`char_count` 和 `scope="environment"`。初始化时若文件不存在，由程序创建“暂无经过验证的经验”的模板，再正常返回。运行中原有文件突然消失或不能读取则报 `MEMORY_IO_ERROR`，不静默重建。

工具描述必须解释：**`scope="environment"` 表示这份记忆属于同一套环境设定，可以跨多个 episode 保留。** 它不是“本次模型调用”或“当前画面”。这里的环境设定包括仿真场景、相机和控制约定；设定变化时应换记忆目录或重新核对经验。`scope` 是程序固定返回的说明字段，模型不能通过传入 scope 选择任意文件，也不需要把它加入 State。

`guide.md` 保存能复用的环境理解，例如动作的大致视觉后果、已验证的抓取线索、哪些判断容易出错。它是可修订的经验，不是不可质疑的规则；应注明适用条件和证据。

### 6.5 `write_guide(content)`：完整替换稳定经验

唯一参数 `content`，必填字符串，1–8000 字符且不能全为空白。它表示**整个新版文件**，不是要追加的一句话。

返回 `scope`、实际写入的 `content`、`char_count` 和 `written=true`，让模型后续能看到文件当前版本。使用临时文件写入后替换目标文件，避免写到一半留下残缺正文；只在替换成功后返回成功。

本工具的描述同样必须写明：固定返回 `scope="environment"`，写入的指南供同一环境设定下的后续 episode 复用，不在新 episode 开始时自动清空。

字符超限或空白 → `INVALID_CONTENT`，原文件保持原样。写入失败 → `MEMORY_IO_ERROR` 并停止，不能向模型声称已经保存。工具验证文件操作与长度，不保证经验在语义上正确；把“模型推测”写成“已证实”仍可能发生，需要模型保留证据与不确定性。

第一版仅允许一个运行进程写这个指南，不设计多个 episode 同时写入的冲突处理。第二次覆写应包含需要保留的旧内容。

### 6.6 `read_working()`：读取本次任务的草稿

参数为空对象 `{}`。返回 `content`、`char_count` 和 `scope="episode"`。每个新 episode 创建自己的模板，不继承上一局的当前计划。该 episode 内的工具调用和外层决策轮都不会自动清空它。

工具描述必须解释：**`scope="episode"` 表示这份记忆只描述当前一次完整实验，从环境初始化到任务结束。** 一个 episode 包含很多外层决策轮和模型调用；所以执行完一批动作不会清空 working。开始新 episode 时创建新文件，旧 episode 的草稿仍作为历史文件保留。该值由程序固定返回，模型无需传它。

初始化模板可以包含“当前目标、当前假设、下一步、预期、等待检查的问题、相关帧编号”。读错误的处理与 `read_guide` 一致。

### 6.7 `write_working(content)`：保存当前计划和未决问题

唯一参数 `content`，必填字符串，1–4000 字符，不能全为空白；完整替换文件。返回字段与 `write_guide` 相同，`scope="episode"`。超限、空白和写入错误的处理也相同。需要清空时写明确内容，例如“当前无待办”，不传空字符串。

本工具的描述也必须解释固定的 `scope="episode"`：只替换本次完整实验的草稿，跨工具调用和动作批次保留；新 episode 使用另一份文件，不覆盖旧实验的草稿。

建议内容示意：

```markdown
当前目标：确认球是否随夹爪抬起。
当前假设：f000005 中球可能已进入两指之间，但遮挡较多。
下一步：执行 carry，然后对比动作前后的图。
预期：球应与夹爪一起上升，相对位置大致保持。
待核对：如果球留在地面，本次抓取假设失败。
参考：闭爪前 f000004；闭爪后 f000005。
```

这份草稿是当前可编辑的理解；每一个动作批次的原始预期已经保存在消息中的 `play.expectation`，也会进入批次日志。因此修改草稿不会抹掉“当时究竟预期什么”。无需为了预期再设一个专门的 State 字段或工具。

模型不必每次动作前都重写草稿。计划变化、出现矛盾、需要跨多轮保留疑问时再写；简单的下一步预期直接随 `play` 提交即可。

### 6.8 `play(actions, expectation, basis_frame_id)`：交出一批连续动作

```json
{
  "actions": [
    {"action": "arm_pose", "args": {"pose": "reach"}},
    {"action": "close_gripper", "args": {}},
    {"action": "arm_pose", "args": {"pose": "carry"}}
  ],
  "expectation": "整批动作结束后，球随夹爪抬起，仍位于两指附近。",
  "basis_frame_id": "f000005"
}
```

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `actions` | 是 | 按执行顺序排列的非空动作数组；第一版默认 1–16 项，上限集中配置 |
| `actions[i].action` | 是 | 下表中的一个 MuJoCo 动作名 |
| `actions[i].args` | 是 | 对应子动作的参数对象；没有参数也传 `{}` |
| `expectation` | 是 | **整批结束后**预期看到的变化，1–1024 字符；是预测，不是执行成功证明 |
| `basis_frame_id` | 是 | 决定本批次时依据的当前原图，必须等于外层登记的当前帧编号 |

`expectation` 是本版新增的明确参数。原版 ARC3 提示要求动作前陈述预期，但其 `play` Schema 没有这个字段。显式放进参数以后，即使模型回复的普通正文为空，预期也会随工具调用进入消息和动作日志。

`basis_frame_id` 避免模型回看旧图后把旧情境当成当前情境；它只验证图片身份，不证明模型真的理解了图片。

本批次只核对一次动作前图；批内不重新调用模型或要求每个子动作提供新图。第一版不要求每个子动作另写预期，因为默认没有中间图用来逐项验证。`expectation` 的拼写表示“预期”，与程序异常 `exception` 不同。

**这里的组合是一次提交、按数组顺序连续执行、结束后统一反馈。** 当前采集器也是逐项调用执行器。若要底盘和机械臂在同一仿真时间区间真正同时运动，需要进一步定义并行组、持续时间及冲突处理并改执行器；本版不把顺序执行声称为物理并行。

第一版复用当前 `control/tools.py` 的十个非结束动作：

| `action` | `args` | 作用及边界 |
| --- | --- | --- |
| `forward` | `seconds` 可省略，默认 1.0；0.1–3.0 秒 | 底盘前进 |
| `back` | 同上 | 底盘后退 |
| `turn_left` | 同上 | 底盘原地左转 |
| `turn_right` | 同上 | 底盘原地右转 |
| `arm_pose` | 必填 `pose`，仅 `stow/reach/carry/drop` | 移动至对应机械臂预设 |
| `shoulder` | 必填 `delta`，-0.5–0.5 弧度 | 肩关节增量控制 |
| `elbow` | 必填 `delta`，-0.5–0.5 弧度 | 肘关节增量控制 |
| `open_gripper` | `{}` | 张开夹爪 |
| `close_gripper` | `{}` | 闭合夹爪；是否抓到球由后续图像判断 |
| `observe` | `{}` | 发送底盘停止指令并推进两个控制拍，再取得新图 |

数值上限对应当前 `config.PRIM_MAX_S=3.0`、`config.ARM_NUDGE_MAX=0.5`；实施时从同一配置生成说明和校验，避免出现两套上限。`arm_pose` 的枚举也应明确写入工具 Schema。当前底层 `ToolLayer.execute` 并不完整执行 JSON Schema 校验，还会忽略未知参数，所以 VISTA 入口需要先严格检查，不能仅依赖现有执行器。

`observe` 是一个真实环境动作：它会停止底盘并推进时间。它和只读取归档文件的 `inspect` 不同，不应在查资料时隐式调用。

**seconds 的实际含义也要告诉模型。** 当前前进/后退主要按固定指令速度和控制拍数运行，随后停止并稳定；turn_left/right 则将 seconds 乘标称角速度换算为目标转角，再由底层控制器减速到位。转向实际仿真耗时不保证等于参数秒数；左右动作是原地转向，不是横向平移。说明依据当前 control/drivers.py，不能给模型错误的动作定义。

模型可在不熟悉时用只含一个短动作的批次对比前后图，记录带条件的局部经验；熟悉的流程再组合执行。不期待一个通用的“每秒变化多少像素”常数，因为距离、视角和接触会影响画面变化。批内只有最终图时，不能可靠归因每个子动作的视觉效果。VISTA 的记忆和回看机制支持这样积累经验，但控制准确性需要运行实验验证。

执行前先校验整个动作数组；后面的子动作参数错误时，前面的子动作也不能提前执行。全部合法后，tools 返回空更新，末尾仍是模型发出的 play AIMessage，条件边据此去 END。**此刻没有 ToolMessage**：

```python
return {}  # 保留当前 AIMessage，由 after_tools 正常路由到 END。
```

模型刚输出的完整 AIMessage 已经在 messages。brain.py 从本次图正常返回时的这条消息读取 play，把 args.actions 转成现有的 list[Decision]；每个 Decision 的 thought 可引用批次预期，日志明确它是整批预期。原工具调用 ID 标识整批，不塞进各控制函数参数，也不替每个子动作伪造模型调用 ID。

外层记录请求后执行整批，结束时生成该 `play` 的**一条汇总结果**。以下对应上面的三动作请求：

```json
{
  "ok": true,
  "data": {
    "batch_id": 6,
    "tool_call_id": "c8",
    "before_frame_id": "f000005",
    "after_frame_id": "f000006",
    "execution_status": "completed",
    "action_results": [
      {"index": 1, "action": "arm_pose", "args": {"pose": "reach"},
       "status": "completed", "action_applied": true, "error": null},
      {"index": 2, "action": "close_gripper", "args": {},
       "status": "completed", "action_applied": true, "error": null},
      {"index": 3, "action": "arm_pose", "args": {"pose": "carry"},
       "status": "completed", "action_applied": true, "error": null}
    ]
  },
  "error": null
}
```

同一结果还提供 `f000006` 的尺寸和真实图片，文本示例省略图像内容块。`ok=true` 表示整批控制调用正常完成，子项 `action_applied=true` 表示该控制指令已经施加；**都不保证抓到球或完成任务**。模型根据最终图判断物理效果。

批次失败时返回 `ok=false` 和具体 `error`，但 `data` 仍保留实际子动作结果和可用的前后帧信息，不能像纯参数错误一样将它清为 null。历史记录从原请求补入 `actions` 和 `expectation`，与回执一起保存。

批次 `execution_status` 可为 `completed`（全部完成）、`partial`（部分完成后遇到进度明确的失败）、`failed`（没有完成项且失败情况明确）、`unknown`（存在进度未知项）、`cancelled`（主动取消）。子项 `status` 可为 `completed/failed/unknown/not_executed`。子项 `action_applied` 分别记录已知已施加 `true`、已知未施加 `false`、无法确认 `null`；不要用一个批次级布尔值抹掉部分执行的事实。

遇到第一个执行失败或中断就停止后续子动作，逐项列出结果。例如第二项异常时，第一项可以是 `completed`，第二项为 `unknown`，第三项为 `not_executed`，绝不能把前三项都说成“未执行”。能取到图就保存批次停止时的最终图，不能取到时 `after_frame_id=null`，也不把动作前图冒充动作后图。

错误与处理：

| 情况 | 判定 | 后续 |
| --- | --- | --- |
| 数组为空、超过上限，任一子动作或参数不合法 | 整批执行前验证失败 | 返回 `INVALID_ACTION` / `INVALID_ARGUMENT` / `ACTION_LIMIT`；整批不执行，图内让模型修正 |
| `basis_frame_id` 不是当前图 | 与外层当前帧 ID 比较 | `STALE_OBSERVATION`，不执行，告诉模型当前 ID |
| 上一批次还没有收到结果，就再次请求模型 | 当前轮配对检查失败 | raise 说明调用顺序错误；不发第二批动作 |
| 子动作失败，但每项施加情况都明确 | 外层结构化回执 | 停止批内剩余动作，记录实际结果；最终图可用时允许下一轮模型重新选择 |
| 子动作异常，可能已动了一部分 | 外层不能确定其进度 | 记录该项 action_applied=null、批次状态 unknown；raise 说明，退出整个程序，不重放 |
| 动作执行后取图/保存失败 | 有执行记录但无可靠新图 | 保留能保存的真实执行事实；raise 说明，退出整个程序，不重执行动作补图 |

现有控制层只返回 `(ok, result)`，不能仅从 `ok=false` 推出“机器人没动”。第一版保守地把进入执行器后的失败视为施加情况未知，除非执行器提供可核实的进度信息。不能从中文错误文本里猜这件事。参数修正不施加动作；真实失败后是否重新尝试，必须由看过批次结果的模型提出一个新请求。

### 6.9 `finish(success, reason, evidence_frame_ids)`：结束请求

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `success` | 是 | 严格布尔值，模型自评是否完成任务 |
| `reason` | 是 | 结束原因，1–1024 字符 |
| `evidence_frame_ids` | 否 | 最多 4 个已归档原图编号；默认空列表，用于说明视觉证据 |

finish 校验通过后 tools 返回 `{}`，条件边正常到 END。外层从当前 finish 请求读取参数，映射为现有 `Decision(tool="done", args={"success":...})`，接受请求后追加它的 ToolMessage，结束当前 episode，不再调用本集模型。结果返回 accepted=true、model_success 和结束原因。它不推进物理仿真，不新增环境帧，也不生成 history 批次；done 不允许塞进 play.actions。正常结束不等于发生程序异常，实验评分仍由外层记录。

模型的 `success=true` 不覆盖环境评估。`env.success()` 仍负责实验评分，保存在最终报告中；第一版不把这个真值作为模型学习抓取的工具反馈。参数非法或证据帧不存在时返回可修正错误；episode 已结束后调用是程序生命周期错误，停止。

## 7. 图外需要维护什么

### 7.1 图片库就是文件加索引

第一版只保存模型可使用的车头图：初始图一张，每个 `play` 批次结束后的图一张；若批次中途停止，保存停止时能取得的图并标明实际完成情况。直接从原始 RGB 数组保存 PNG，保留原始宽高，不先经过 JPEG 再转 PNG。**PNG 只能无损保存拿到的像素，不能还原此前已经丢失的信息。**

```json
{
  "episode_id": "ep_0000",
  "frame_id": "f000006",
  "camera": "front",
  "path": "frames/f000006.png",
  "width": 640,
  "height": 480,
  "source_batch_id": 6
}
```

上面的宽高是示意，必须从实际图片采集，不写死 640×480。path 相对于当前 episode 目录，模型不可以修改；完整身份是 (episode_id, frame_id)。工具只访问当前 episode，因此模型填一个 frame_id 就够了。初始帧的 source_batch_id=null。图片索引不保存仿真时间；身份与顺序由图片编号、批次编号确定。现有采集器自己的时间日志保持独立。

**保存到哪里与消息是否在内存里是两回事。** `self.state` 保持进程内的上下文；磁盘 PNG 和 JSONL 独立存在，不使用 SQLite。messages 中可以有模型看过的图像内容块，但整个图片库不进入 State。按文件编号查到图片后，发送给模型时仍须读取实际像素内容。

存储地址由外层程序管理，具体约定如下：

1. `output/vista/` 是默认输出根，可以在启动前配置其他位置；每个 episode 分配新目录，初始化时解析并固定其绝对路径 `episode_dir`。后续不依赖进程当前工作目录去重新猜路径。
2. 索引只保存 `frames/f000006.png` 这种相对路径；读取时用 `episode_dir / index.path` 定位。检查解析后的文件仍在该 episode 的 frames 目录内。
3. `frame_id` 一旦公布，就永久对应同一张原图；原图和已有索引记录不重命名、不覆盖、不删除。`inspect`、`read_pixels` 只读，裁剪放大在内存或另外的缓存中进行，不改原文件，不修改输出根。
4. 同一运行期间外部程序也不能随意移动或清理图片目录；缺文件时明确报错。若停机后需要归档搬迁，可以整体移动 episode 目录并在读取入口配置新的根；相对索引仍有效。只移动其中一个文件就会破坏映射。整体搬迁方便离线查看，并不增加仿真断点恢复能力。

所以需要保证的是“编号到原图的映射稳定”，不是要求电脑上的绝对输出地址永远不能改。目录选择、编号分配和路径校验都由程序控制，工具参数不提供改址入口。

保存流程：分配编号 → 写完整 PNG → 成功后追加索引 → 把图片和编号送入消息。如果保存失败，不对外宣称该帧已经可检索。内存中可用一个 `dict` 按编号查索引，暂不需要数据库、embedding 或图片说明生成模型。

当前帧 ID 是外层运行对象里的一个指针。旧图不会因为 `inspect` 而变成当前帧；裁图也不会抢占这个指针。第 6.8 节的 `basis_frame_id` 正是和它核对。

这是对第一版采集范围的明确限制：**批内子动作之间、子动作执行过程中的中间画面暂不进入 VISTA 图片库**。一次三动作 play 默认仍只有一张最终图。原版可以归档环境一次动作返回的多帧；MuJoCo 以后也可以采集批内边界或 tick 帧并附上子动作编号，但现在不把录像目录里碰巧存在的画面当成模型已经能查询的图片。

现有录像可以继续保存其他视角，但 VISTA 的 `inspect`、`history` 不暴露俯视图、侧视图或真值记录。`--no-images` 只关闭通用录像图片，VISTA 仍会保存必需的初始图和每批最终车头 PNG。

### 7.2 裁剪图与原图坐标的关系

假设原图裁出 `(x=100,y=80,width=200,height=100)`，展示尺寸变成 `1000×500`。展示图上的 `(u=500,v=250)` 对应原图约 `(200,130)`：

```text
original_x = x + u * width  / rendered_width
original_y = y + v * height / rendered_height
```

这是连续位置的比例换算。用于单个像素查询时，需要取整并再次检查原图边界；最直接的方法是始终让 `read_pixels` 使用工具返回的原图区域。当前 MuJoCo 动作是时间/关节参数，不把这个坐标换算成“点击动作”。

编号、来源和区域写在图片旁边的文字元数据中，**不画进原图像素**，否则 `read_pixels` 可能读到标记而非环境。

### 7.3 两份笔记与三类记录各管一件事

| 保存位置 | 回答的问题 | 谁写 |
| --- | --- | --- |
| `guide.md` | 对这个环境已经形成哪些可复用认识？ | 模型经 `write_guide` |
| `working.md` | 现在准备做什么、哪些假设还没验证？ | 模型经 `write_working` |
| `actions.jsonl` | 每次 play 提交了什么批次，各子动作发生了什么，前后是哪两张图？ | 外层结算后每批追加一条 |
| `calls.jsonl` | 哪个工具被怎样调用，返回了什么或哪里出错？ | 程序自动追加请求/结果事件 |
| `messages.jsonl` | 模型当时收到和产生了哪些消息？ | 程序记录消息；图片可用稳定引用表示 |

笔记是可变的总结，日志是发生过的记录。模型不能通过 `write_working` 修改历史动作事实。工具记录保存 `question`、`label`、`expectation` 等原始参数，不靠事后从自然语言回复里猜测。

`messages.jsonl` 用引用表示图片时，应同时记录真实发送的图像来源和裁剪变换。它不等同于完整 HTTP 请求备份，也不是一键恢复 state 的保证。记录 API 请求元信息时排除密钥与认证头。

### 7.4 与当前采集器的衔接：正常交接与异常退出分开

Brain.decide 返回 list[Decision]，采集器顺序执行，将整个列表对应到一次 play。传入 brain 的 history 是逐动作的原始记录，不等于按批次 history 工具；VISTA 不把这份含底层结果文字的 history 发给模型，公开历史由自己的批次日志提供。

`run_collect._run_vista_rounds()` 单独处理 VISTA 批次。它不走其他 brain 的“捕获决策异常后仅结束当前集”的路径：记录完当前集的错误后继续 raise，让异常离开全部 episode 循环，由第 3.5 节的最外层入口打印、以退出码 1 终止整个程序。其他 brain 的决策流程保留。

已实现的正常流程：

1. 启动时读取并校验 max_model_calls，创建 VistaBrain 和模型接口；每集 begin_episode 固定目录，初始化两个 State 字段、笔记和空日志。初始 decide 归档初始图，后续不重复追加同一观察。
2. decide 运行图。只有 tools 验证成功的 play/finish 会正常到 END；外层读取返回状态末尾的这条 AIMessage。play 转成一组 Decision，finish 转成一个 done Decision。不保存等待动作或停止原因的副本。
3. 外层核对 Decision 列表与本轮原请求一致，保存请求并分配批次号，再逐项执行。遇到第一项失败或中断就停止后续子动作，逐项记录 completed/failed/unknown/not_executed；不为每个子动作单独回复一次 play。
4. 批次结束后取得最终图，调用 accept_batch_result(receipt, obs_after)：保存 PNG、索引和一条 history 批次记录，向 messages 追加匹配当前 play 的一条 ToolMessage 及实际图像。结果可靠时正常返回；不可继续时记录能确定的事实并 raise。
5. finish 用 accept_finish_result(receipt) 追加结束确认，不新增图片或批次；随后结束当前 episode。两个 accept 方法都是普通 Python 方法，不是模型工具或图节点。
6. 真实回执接回后，再判断环境成功、外层决策轮数耗尽等结束条件。继续时沿用刚取得的 obs_after；结束时保存 outcome。最后一批也要记结果，不能依赖下一次 decide 才补齐。

正常路径的简化示意；回执构造和执行循环实际位于 `run_collect._run_vista_rounds()`，`accept_*` 方法位于 `brain.py`：

```python
brain.begin_episode(ep_dir, task_text, tool_schemas)
obs = env.get_obs()

while episode_can_continue:
    decisions = brain.decide(obs, task_text, tool_schemas, history)
    # invoke 正常返回，末尾必为 tools 已校验的交接 AIMessage。
    request_message = brain.state["messages"][-1]
    call = request_message.tool_calls[0]

    if call["name"] == "finish":
        receipt = accept_done_and_make_receipt(call, decisions)
        brain.accept_finish_result(receipt)
        break

    receipt = execute_batch_and_make_receipt(call, decisions)
    obs_after = env.get_obs()
    brain.accept_batch_result(receipt, obs_after)
    # 之后检查环境成功或外层轮数等正常结束条件。
    obs = obs_after

brain.end_episode(outcome)
```

异常不在这个循环里 catch 后继续。底层有些函数会将执行异常转成 (ok=false, result) 返回；因此 VISTA 适配层还必须把无法确认进度等不可继续的失败提升为 RuntimeError，不能认为“底层没抛”就可以继续。在 raise 前记录已执行和未执行部分，不把已发生的动作当成未发生。

对于执行、取图或存储异常，错误处理只保存能够可靠取得的诊断，不新增物理动作或 API 调用。图不可用时 after_frame_id=null；存储本身失效时也不承诺错误记录一定能落盘，最外层打印的原始错误仍必须保留。底层异常和保存诊断的二次异常不能相互掩盖。

接回结果时只检查当前这轮：回执 ID、动作顺序和参数必须与当前 AIMessage 匹配。相同回执在本轮重复到达时与已接回的 ToolMessage 核对并去重；同 ID 不同结果、未知 ID 或漏项都 raise，不重执行、不追加第二条批次记录。下一次请求模型前确认本轮每个调用已配对，不扫描整段历史查找未完成请求。

主动取消、未知执行进度等异常路径不得执行剩余动作或下一集；异常以字符串说明到达最外层。用户正常 finish 和正常任务评分走返回流程，不能与程序故障混在一起。第一版不支持异常后恢复同一仿真，也不自动重放失败批次。

### 7.5 哪些反馈允许进入模型

当前 `env.get_obs()` 除图片外还有关节角、TCP、底盘位置；采集器还附加 `holding`。原控制工具的结果文本可能包含抓取真值和夹爪提示。**不能把整个 obs、history 或 result 原样拼进消息。**

本版公开给模型的只有：任务、车头图片和身份元数据、它自己请求的动作及参数、施加情况/技术执行状态、它自己写的笔记与预期、上述公开工具结果。底层详细反馈继续留在本地采集记录，用于实验分析。

未知执行错误用固定的技术错误码解释，不复述可能包含真值的底层文本。API 网络故障、鉴权失败、上下文过长等由宿主记录并停止；第一版不自动重试 API，也不会为修复 API 错误重放 MuJoCo 动作。

对参数错误的修正、对物理动作效果的重试，是两件事：前者在图内即可；后者必须看过新图，由模型提出一个新的 `play`。

### 7.6 批次执行以后，working 和 guide 怎样更新

程序自动完成的是“保存真实结果和图片，接回原 messages”。**笔记内容由模型看过结果后，通过写笔记工具更新**，不是执行器猜出一段总结。批次预期已在原 AIMessage 中，不需要从别处再注入一遍。

```text
play(actions=[...], expectation=整批预期)
    → 外层执行整批
    → 一条 play 结果 + 最终图进入 messages
    → 下一次 model 比较预期与观察
    → 需要时再 inspect / read_pixels / history
    → write_working：当前进展、失败假设、下一步
    → 形成可复用认识时 write_guide
    → 下一次 play 或 finish
```

working 用来保持最新计划和不确定处；guide 只在形成或修正跨实验可用的认识时写。不要求每批动作都重写两份文件，也不增加专门的反思节点。模型可以在同一条回复里发出两个笔记写入，工具顺序执行，各自返回结果后再让模型提交 play。

例如 `reach → close_gripper → carry` 执行完，最终图中球仍在地面：模型可将 working 改成“整批抓取没有达到预期，下一步检查对齐”。没有中间图时，不应仅凭最终图就断言到底在哪个子动作失败。若以后观察支持某种稳定规律，再更新 guide，并注明条件和证据。

如果外层因环境成功、预算耗尽或取消直接结束，仍须保存最后一批真实回执，但此时没有再请求模型，就不能声称笔记已根据最终图自动更新。需要模型整理的内容应在正常继续轮中写入，并在 finish 之前完成；第一版不隐式增加一次结束后的模型调用。

## 8. 跟着一个批次走完两次外层调用

假设本 episode 已运行一段时间，当前图为 `f000005`，旧图 `f000004` 也可查询。下面的图像描述是示意，不是对实际场景的预判。

### 第一次 decide：多工具查证，再提交三动作批次

1. `model` 在一条 AIMessage 中发出两个 `read_pixels`，分别采样 `f000004`、`f000005` 中疑似球的区域，调用 ID 为 `c10` 和 `c11`。
2. `tools` 顺序完成两个请求，生成两条 ToolMessage，各自匹配原 ID。消息收齐后才回 model，模型现在能同时比较两个结果。
3. 模型调用 `write_working`，保存当前假设：“目标可能仍是同一个球，准备尝试抓取；随后看是否抬起”。工具保存并返回新版内容。
4. 模型发出一次 `play`，`actions=[reach, close_gripper, carry]`（实际传完整参数），`basis_frame_id=f000005`，预期是“最终球随夹爪抬起”，调用 ID 为 `c13`。
5. tools 校验整个数组后返回 `{}`，消息末尾仍是原 play AIMessage，条件边到 END。这次 decide 用了三次模型调用，返回三个 Decision；MuJoCo 尚未执行这一批。

### 外层执行：三个子动作，一条反馈

6. 外层直接读取本轮正常交接的 c13 请求，记录第 6 批，依次执行 reach、close_gripper、carry。中间不请求模型，也不补三个独立的 play 结果。
7. 批次结束后取图，`accept_batch_result` 将其存为 `f000006`，写一条含三个子动作结果的 history，追加匹配 `c13` 的 ToolMessage 和实际图像。
8. 本轮结果和请求 ID 配对完成后，允许下一次模型调用。无需清除任何等待状态；若有无法继续的异常，则 raise 到最外层，打印说明并退出整个程序。

### 第二次 decide：比较预期，更新笔记，再选择下一批

```text
messages 中已经有此前的全部上下文，以及：
  assistant: read_pixels(id=c10, f000004), read_pixels(id=c11, f000005)
  tool c10:  第一组像素
  tool c11:  第二组像素
  assistant: write_working(id=c12, content=...)
  tool c12:  已写入的 working 全文
  assistant: play(id=c13, actions=[reach, close_gripper, carry], expectation=...)
  tool c13:  第 6 批的三个子动作结果 + f000006 原图
```

第二次 decide 把本轮模型调用预算归零，沿用完整 messages。模型可以先在一个 inspect 请求中比较 `f000005` 与 `f000006`，再更新 working；形成值得复用的认识时更新 guide。两份笔记不由执行器自动改，也不需要复制某个 pending 字段来提醒模型曾经预期什么。

之后模型可以提交另一个多动作批次，或调用 finish。history 中一个 play 仍是一条记录；三个子动作仅是这条记录里的三个条目。查询旧图不会使当前帧从 `f000006` 退回 `f000005`。

## 9. 第一版怎样算跑通

这里的“跑通一轮”至少覆盖一个多动作批次的结果进入上下文，并让**下一次模型调用**确实读到结果。只收到一个 `play` 就结束，只能说明提出动作的部分通了。

实施时按以下顺序验证：

1. 用固定的模拟模型输出串起“两项像素查询 → write_working → 三动作 play”。检查请求位于 AIMessage，两个查询各有一条配对 ToolMessage，图内不推进 MuJoCo，State 只有 messages 和 model_calls。
2. 外层执行整批，中途不调用模型，最后只追加一条 play 汇总结果和最终图；下一次模型调用能看到原批次预期与真实结果，并能更新 working/guide。
3. 验证 history 一条对应一个 play，`limit` 按批次数计算；三个子动作不会被当成三条历史。图片查询只读，改变进程工作目录不影响已固定的图片根路径；原 PNG 不被裁图覆盖。
4. 检查关键边界：多个合法查询正常返回；混合交接调用和超限请求被拒；整批参数先校验；第二个子动作异常时第三项不执行，下一 episode 也不启动。致命错误抛到最外层，只打印清楚说明并以退出码 1 结束；部分结果照实记录，不重放。正常最终一轮和 finish 均有配对结果。
5. 在真正支持视觉和工具的模型端点上验证上述闭环；检查笔记工具的描述包含 scope 两种值的含义。区分模型自评与环境评分，确认模型输入没有混入持球真值或隐藏位置。
6. 设置不同 max_model_calls 验证预算确实来自配置；缺失和非法值在启动时失败，第 N 次请求仍可交出动作，第 N+1 次不发送。图步数上限随之调整，SDK 无隐藏重试；图片索引不需要仿真时间。

上述消息配对、多工具、笔记、预算、批次、异常退出和输入隔离已纳入 `tests/test_vista.py`。测试也使用真实 MuJoCo 执行 `observe → forward`，验证批次结果与新图进入下一次模拟模型调用。**本次验证没有调用付费模型，也不声称真实模型已能成功抓球。**

第一版仍有明确限制：消息会持续增长；进程崩溃后不自动恢复仿真；只采初始图和批次结束图；批内顺序执行，不支持物理并行；图片按 ID 和批次记录检索；笔记可能包含模型误判；控制成功不保证任务成功。它们不会阻止先验证上述基础闭环。

## 10. 原版依据与本设计的改动

核对的官方代码快照：`c97c354f0a8948256545a3e8850a1848f8e5b8c1`。不同 benchmark 的工具并非完全相同，本文主要参考 ARC3 的工具及通用视觉工具实现。

| 项目 | 原版参考 | 本设计 |
| --- | --- | --- |
| 会话运行 | Codex / Claude 运行后端 | 普通模型接口 + 两节点 LangGraph |
| 工具集合 | play、inspect、read_pixels、history、四个笔记接口；可选压缩交接 | 保留八项功能，新增 finish，省略压缩交接 |
| 查图身份 | `turn` + 可选 `frame` | 当前 episode 内唯一的 `frame_id` |
| 检查目的 | inspect/read_pixels 的 `question`、每幅图的 `label` | 保留其作用，不让工具代答 |
| 动作预期 | 提示词要求动作前陈述 | 必填 `play.expectation` 描述整批结束后的可见预期 |
| 像素结果 | 调色板 + 符号行 | 直接返回 RGB 采样和坐标 |
| 笔记初始加载 | 提供工具供 agent 读取 | 初始化主动读入一次，后续按需读写 |
| 图片采集 | 可保存一次环境响应中的多帧 | 第一版只采初始帧与批次结束帧，PNG + 相对路径索引 |
| 执行动作 | ARC3 的 play 执行一个游戏动作 | 一个 play 提交多个顺序子动作，外层完成后一次反馈 |
| 历史粒度 | 依 benchmark 的动作/事件接口 | 每个 play 批次一条，内部保留各子动作结果 |
| 本版状态 | 不照搬运行后端的会话管理 | 仅 messages、model_calls；当前轮请求结果按 ID 配对，致命异常直接退出 |

主要阅读位置：

- [官方 VISTA 仓库](https://github.com/joshhhhhan/VISTA)；[原论文](https://arxiv.org/abs/2610.02200)。
- [本地 ARC3 工具定义](../VISTA-docs/upstream/src/vista/benchmarks/arc3/codex/tools.py)：play、history 和笔记工具的实际参数。
- [本地共用视觉工具定义](../VISTA-docs/upstream/src/vista/core/tools.py)：question、views、region、rows、columns 及原版限额。
- [本地视觉工具执行](../VISTA-docs/upstream/src/vista/core/session.py)：裁剪、取原图像素、错误返回和笔记读写。
- [本地 ARC3 行为提示](../VISTA-docs/upstream/src/vista/benchmarks/arc3/codex/prompt.md)：动作前预期、动作后观察与两份笔记的分工。
- [当前 Brain 接口](../base.py)、[采集循环](../../run_collect.py)、[控制工具](../../control/tools.py)：决定本设计的动作交接方式。
- [LangGraph 图接口](https://docs.langchain.com/oss/python/langgraph/graph-api)：基础节点、边与状态更新；本文只用其中的基础串行结构。

你的《9.18 LangGraph》笔记第 5–7 页的 `StateGraph + model/tools + add_messages` 已足以理解这里的主循环。本文选择手动保留 State，所以暂不需要后面关于数据库、并行、流式或中断恢复的内容。

旧讲义中写作 `brains/VISTA/...` 的本地路径现在统一对应 `brains/VISTA-docs/...`。原论文、译本和源码快照保留；新 brain 已在 `brains/vista/` 独立实现，原论文在新目录也保留一份副本。
