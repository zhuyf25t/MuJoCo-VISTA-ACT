"""Small, explicit settings; credentials never enter the graph or its logs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

import config


def local_values(path: Path | None = None) -> dict:
    return dict(dotenv_values(path or config.LOCAL_LLM_FILE,
                              encoding="utf-8-sig", interpolate=False))


def setting(names: tuple[str, ...], values: Mapping, environ: Mapping, default=None):
    """An explicitly present but invalid high-priority value must not fall back."""
    for source in (environ, values):
        for name in names:
            if name in source:
                return source[name]
    return default


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value).strip()):
        raise RuntimeError(f"{name} 必须配置为正整数，例如 12")
    number = int(str(value).strip())
    if number < 1:
        raise RuntimeError(f"{name} 必须配置为正整数，例如 12")
    return number


@dataclass(frozen=True)
class VistaSettings:
    """VISTA 本地限制；注释说明当前行为，不代表模型已收到这些注释。

    tools.py 将部分限制写入工具 Schema/description；模型只能从这些说明、
    提示词及工具结果中了解限制，不会读取此 Python 文件。Schema 也不保证
    模型一定遵守，所以本地仍会校验：参数超限返回错误 ToolMessage，
    不静默截字、不只执行前几个动作；模型可在剩余调用预算内修改请求。

    *_chars 用 Python len(参数字符串) 计数，包含空格/换行，不是 token 数，
    不计 JSON 转义后的长度，也不计独立的 reasoning_content；若模型把
    推理文字写进被检查的参数，则这些文字同样计数。本文件不裁剪 messages。

    模型输出 token 上限另在 model.ChatModel 中读取 OPENAI_MAX_TOKENS /
    max_tokens，默认 config.LLM_MAX_TOKENS=65536。它由 API 服务端执行，
    当前未作为提示词告知模型。DeepSeek 思考模式的生成预算包含 reasoning
    和最终输出；工具调用也是模型输出的一部分。换提供商需核对其计量规则。
    若 finish_reason=length，model.parse_reply 会 raise，外层打印并以
    退出码 1 结束，不执行被截断回复中的工具调用，也不自动补全或重试。
    DeepSeek 规则：https://api-docs.deepseek.com/api/create-chat-completion/
    """

    # 每次 brain.decide() 最多请求模型多少次；从 .env/进程环境读取，无默认值。
    # 下一外层决策轮重置计数；工具错误后的再请求也计次，reasoning 不另计次。
    # 第 N 次仍可正常交出 play/finish；若还需第 N+1 次请求则先 raise，程序结束。
    # 当前没有把上限或剩余次数注入模型上下文，模型无法据此主动安排调用。
    max_model_calls: int

    # 图片、指南和日志的根目录；默认 output/vista，可由 --brain-memory 覆盖。
    # guide.md 跨 episode 共用；每个 ep_XXXX 保存独立 working.md、frames 和日志。
    # 不是容量限制；不向模型开放任意磁盘路径，模型用 frame_id 和笔记工具访问。
    # 无法可靠读写所需文件时 raise 并终止，不通过截断文件继续运行。
    output_root: Path = field(default_factory=lambda: config.ROOT / "output" / "vista")

    # 一条 AIMessage 中的工具调用数上限；play/finish 另要求各自独占一条回复。
    # 超限时整组都不执行，为每个调用返回 TOOL_CALL_LIMIT，再交回模型。
    # 当前 Schema/system prompt 未提前公布此数值，模型在超限错误中才看到它。
    max_tool_calls: int = 8

    # 一次 play.actions 最多包含的子动作数，不是模型调用次数或 history 条数。
    # 通过 Schema.maxItems 告知模型；超限返回 ACTION_LIMIT，整批不交给执行器。
    max_actions: int = 16

    # 每次 inspect/read_pixels 的 views 数量上限，也限制 finish 的证据帧数量。
    # 通过 Schema.maxItems 告知；超限返回工具错误，不仅处理前几张。
    # 多个合法工具调用分别计算此限制，不是整条 AIMessage 的图片总数上限。
    max_views: int = 4

    # inspect 派生展示图的目标长边（像素）：请求区域等比例缩放至此长度，
    # 使用最近邻采样，小图放大、大图缩小；缩小时会损失展示细节。
    # 不是区域越界阈值或原图尺寸限制；不修改归档 PNG，read_pixels 仍读原图。
    # 当前 Schema 未提前写明固定长边；结果用 original_size/region/rendered_size
    # 告知原图、选区和实际展示尺寸，查询坐标仍须用原图坐标。
    inspect_edge: int = 1024

    # 一次 read_pixels 所有 views 的 sum(rows * columns) 上限，单位为采样点。
    # 工具 description 已告知总点数限制；超限返回 SAMPLE_LIMIT，不只返回前 N 点。
    # 各次 read_pixels 分别计数；正常调用按请求网格采样，本来就不是返回全部像素。
    max_samples: int = 1024

    # 每个 question、play.expectation、finish.reason 字符串的字符数上限。
    # 通过 Schema.maxLength 告知模型；超长/空白返回 INVALID_ARGUMENT，不截字。
    max_question_chars: int = 1024

    # inspect/read_pixels 中每个 view.label 的字符数上限，用于标识比较角色。
    # 通过 Schema.maxLength 告知；超长/空白返回 INVALID_ARGUMENT，不截字。
    max_label_chars: int = 128

    # guide.md 完整内容的字符数上限，不是每次新增内容的上限（写入是完整替换）。
    # write_guide 的 Schema.maxLength 已告知；超限返回 INVALID_CONTENT，旧文件不变。
    # 读到人工改成超长/空白的文件则 raise MEMORY_IO_ERROR，终止程序，不裁剪读取。
    max_guide_chars: int = 8000

    # 当前 episode 的 working.md 完整内容上限；Schema.maxLength 已告知模型。
    # write_working 超限返回 INVALID_CONTENT，旧文件不变；读取超长/空白文件则
    # raise MEMORY_IO_ERROR 并终止。是否精简笔记由模型显式重写决定，不自动截字。
    max_working_chars: int = 4000

    # history 每页默认条数：batch 为决策记录数，frame 为抽样后的帧数。
    # 有 start 从前向后分页；没有 start 取最近一页。始终升序返回，
    # truncated 与 next_query 明示尚未返回的范围，next_query 可直接重调。
    history_default_limit: int = 10

    # history.limit 参数允许的最大值；Schema.maximum 已告知模型。
    # 请求超过上限返回 INVALID_ARGUMENT，不把用户传入的 limit 悄悄改成 50。
    # 只限制单页返回条数，不限制 episode 保存量。
    history_max_limit: int = 50

    # batch 返回完整记录；其 observations 总数超过此上限则给出可修正错误，
    # 让模型缩小范围或改用 frame 分页，不静默截断批次内部的观测。
    history_max_observations: int = 500

    def __post_init__(self):
        # 配置值本身无效属于启动错误，直接 raise；不走模型的可修正工具错误流程。
        for name, value in vars(self).items():
            if name != "output_root" and (type(value) is not int or value < 1):
                raise RuntimeError(f"VISTA 配置 {name} 必须是正整数")
        if self.history_default_limit > self.history_max_limit:
            raise RuntimeError("history 默认条数不能超过上限")
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser().resolve())

    @classmethod
    def from_env(cls, *, memory_dir=None, env_file=None, environ=None):
        env = os.environ if environ is None else environ
        values = local_values(env_file)
        budget = positive_integer(setting(("max_model_calls",), values, env), "max_model_calls")
        root = memory_dir if memory_dir is not None else config.ROOT / "output" / "vista"
        return cls(max_model_calls=budget, output_root=Path(root))
