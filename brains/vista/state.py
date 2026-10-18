"""The two graph fields, plus ordinary message/receipt helpers (not State)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages


class VistaState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int


class ActionResult(TypedDict):
    index: int
    action: str
    args: dict
    status: str
    action_applied: bool | None
    error: dict | None
    frame_range: list[int] | None


class BatchReceipt(TypedDict):
    batch_id: int
    tool_call_id: str
    execution_status: str
    action_results: list[ActionResult]


def calls_of(message: AIMessage) -> list[dict[str, Any]]:
    """Include malformed JSON calls so a usable ID can still get an error result."""
    calls = [dict(call) for call in message.tool_calls]
    calls += [dict(call) for call in message.invalid_tool_calls]
    raw = message.additional_kwargs.get("provider_message", {}).get("tool_calls", [])
    if raw:
        order = {call.get("id"): index for index, call in enumerate(raw)}
        calls.sort(key=lambda call: order.get(call.get("id"), len(order)))
    return calls


def validate_reply(message: AIMessage) -> list[dict]:
    if not isinstance(message, AIMessage):
        raise RuntimeError("VISTA 模型没有返回 AIMessage")
    calls = calls_of(message)
    if not calls:
        raise RuntimeError("VISTA 模型未返回工具调用，程序终止。")
    ids = [call.get("id") for call in calls]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise RuntimeError("VISTA 工具调用缺少有效 ID，无法配对结果。")
    if len(set(ids)) != len(ids):
        raise RuntimeError("VISTA 当前回复中的工具调用 ID 重复，无法配对结果。")
    return calls


def check_current_round_results(messages: list[AnyMessage]) -> None:
    """Check only the latest assistant turn; no historical unresolved queue."""
    index = next((i for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], AIMessage)), None)
    if index is None:
        if any(isinstance(message, ToolMessage) for message in messages):
            raise RuntimeError("VISTA 初始消息中存在没有请求的工具回执")
        return
    expected = {call["id"] for call in validate_reply(messages[index])}
    received = set()
    images_started = False
    for message in messages[index + 1:]:
        if isinstance(message, ToolMessage):
            if images_started or message.tool_call_id not in expected or message.tool_call_id in received:
                raise RuntimeError("VISTA 当前轮工具回执 ID 错配、重复或顺序错误")
            received.add(message.tool_call_id)
        elif isinstance(message, HumanMessage) and received == expected:
            images_started = True
        else:
            raise RuntimeError("VISTA 必须先配齐当前轮工具结果，再追加观察图片")
    if received != expected:
        raise RuntimeError("VISTA 上一轮工具尚未收到全部结果，不能再次请求模型。")


def action_result(index: int, action: dict, status: str) -> ActionResult:
    applied = {"completed": True, "failed": False, "unknown": None, "not_executed": False}[status]
    error = None
    if status in ("failed", "unknown"):
        error = {"code": "ACTION_FAILED" if status == "failed" else "ACTION_PROGRESS_UNKNOWN",
                 "message": "控制执行失败" if status == "failed" else "控制调用未正常完成，无法确认施加进度",
                 "can_retry": status == "failed"}
    return {"index": index, "action": action["action"], "args": dict(action["args"]),
            "status": status, "action_applied": applied, "error": error, "frame_range": None}


def can_continue_close_pair(actions: list[dict], result: ActionResult, index: int) -> bool:
    """Only the first close may recover through its requested adjacent second close.

    index is zero-based. Never invent an action or continue past the second
    unconfirmed close; other failures retain the normal stop-on-failure rule.
    """
    return (result["action"] == "close_gripper" and result["status"] == "failed"
            and result["action_applied"] is True and isinstance(result["error"], dict)
            and result["error"].get("code") == "GRASP_NOT_CONFIRMED"
            and index + 1 < len(actions) and actions[index + 1]["action"] == "close_gripper"
            and (index == 0 or actions[index - 1]["action"] != "close_gripper"))


def batch_status(results: list[ActionResult], *, cancelled: bool = False) -> str:
    statuses = [entry["status"] for entry in results]
    if "unknown" in statuses:
        return "unknown"
    if all(status == "completed" for status in statuses):
        return "completed"
    if cancelled:
        return "cancelled"
    if "failed" in statuses:
        return "partial" if "completed" in statuses else "failed"
    return "cancelled"
