"""Bridge the two-node graph to the collector's synchronous batch execution."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.message import add_messages

from .base import Brain, Decision
from .graph import build_graph
from .model import ChatModel, image_message
from .settings import VistaSettings
from .state import (VistaState, batch_status, check_current_round_results, validate_reply)
from .storage import VistaStorage
from .tools import VistaTools


class VistaBrain(Brain):
    name = "vista"
    gui_realtime = True

    def __init__(self, *, settings: VistaSettings | None = None, memory_dir=None,
                 model=None, model_options=None):
        if settings is not None and memory_dir is not None:
            raise RuntimeError("VISTA settings.output_root 与 memory_dir 只能指定一个")
        self.settings = settings or VistaSettings.from_env(memory_dir=memory_dir)
        self.max_model_calls = self.settings.max_model_calls
        self.storage = VistaStorage(self.settings)
        try:
            self.system_prompt = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
        except OSError:
            raise RuntimeError("VISTA 无法读取 prompts/system.txt") from None
        self.model = model if model is not None else ChatModel(**(model_options or {}))
        self.state: VistaState = {"messages": [], "model_calls": 0}
        self.graph = None
        self.tools = None
        self._active = False
        self._finished = False
        self._failed = False
        self._closed = False
        self.decision_round = 0

    def begin_episode(self, ep_dir, task_text, tool_schemas):
        if self._active or self._closed:
            raise RuntimeError("VISTA 必须结束上一 episode，且不能复用已关闭的 brain")
        if not isinstance(task_text, str) or not task_text.strip():
            raise RuntimeError("VISTA 任务描述不能为空")
        self.tools = VistaTools(self.storage, self.settings, tool_schemas)
        self.storage.begin_episode(Path(ep_dir), task_text)
        self.state = {"messages": [], "model_calls": 0}
        self.task_text = task_text
        self._control_schemas = deepcopy(tool_schemas)
        self.graph = build_graph(self._model_node, self.tools.node)
        self._active, self._finished, self._failed = True, False, False
        self.decision_round = 0

    def _require_active(self):
        if not self._active or self._closed or self._failed:
            raise RuntimeError("VISTA episode 未开始、已结束或已失败，不能继续调用")

    def _append(self, messages):
        for message in messages:
            self.storage.log_message(message)
        self.state["messages"] = add_messages(self.state["messages"], messages)

    def _model_node(self, state: VistaState):
        check_current_round_results(state["messages"])
        attempt = state["model_calls"] + 1
        if attempt > self.max_model_calls:
            raise RuntimeError(f"本轮模型调用已达到 max_model_calls={self.max_model_calls}，程序终止。")
        self.storage.event("model_request", decision_round=self.decision_round, model_call=attempt)
        try:
            reply = self.model.invoke(state["messages"], tools=self.tools.schemas, system=self.system_prompt)
            validate_reply(reply)
            self.storage.log_message(reply)
        except Exception as error:
            self.storage.diagnostic("model_error", error, decision_round=self.decision_round, model_call=attempt)
            raise
        return {"messages": [reply], "model_calls": attempt}

    def decide(self, obs, task_text, tool_schemas, history) -> list[Decision]:
        self._require_active()
        if self._finished:
            raise RuntimeError("VISTA 已接受 finish，不能再次请求本集模型")
        if task_text != self.task_text or tool_schemas != self._control_schemas:
            raise RuntimeError("VISTA 任务或动作定义在 episode 中途发生变化")
        try:
            check_current_round_results(self.state["messages"])
            if self.storage.current_frame_id is None:
                info = self.storage.archive(obs, source_batch_id=None)
                context = {"task": task_text, "initial_frame": info,
                           "guide": self.storage.read_memory("guide"),
                           "working": self.storage.read_memory("working")}
                message = image_message([(info, self.storage.load_image(info["frame_id"]))],
                                        "本次任务、初始观察和初始笔记：\n" + json.dumps(context, ensure_ascii=False))
                self._append([message])
            else:
                self.storage.check_observation(obs)
            # Only the budget is reset. All previous requests, results and images remain.
            self.state["model_calls"] = 0
            self.decision_round += 1
            self.state = self.graph.invoke(self.state, {"recursion_limit": 2 * self.max_model_calls + 4})
            call = self.handoff()
            if call["name"] == "finish":
                return [Decision(call["args"]["reason"], "done", {"success": call["args"]["success"]})]
            return [Decision("整批预期：" + call["args"]["expectation"], action["action"], deepcopy(action["args"]))
                    for action in call["args"]["actions"]]
        except Exception as error:
            self._failed = True
            self.storage.diagnostic("decision_fatal", error, decision_round=self.decision_round)
            raise

    def handoff(self) -> dict:
        self._require_active()
        messages = self.state["messages"]
        if not messages or not isinstance(messages[-1], AIMessage):
            raise RuntimeError("VISTA 没有等待结果的交接调用，或已接收过该回执")
        calls = validate_reply(messages[-1])
        if len(calls) != 1 or calls[0]["name"] not in ("play", "finish"):
            raise RuntimeError("VISTA 图未返回独占的 play/finish 交接调用")
        return deepcopy(calls[0])

    @property
    def next_batch_id(self):
        return self.storage.batch_count + 1

    def record_batch_start(self, batch_id):
        call = self.handoff()
        if call["name"] != "play" or type(batch_id) is not int or batch_id != self.next_batch_id:
            raise RuntimeError("VISTA 批次交接编号或工具类型错误")
        self.storage.event("batch_start", batch_id=batch_id, tool_call_id=call["id"], args=call["args"])

    def _validate_receipt(self, receipt, call):
        if (not isinstance(receipt, dict) or set(receipt) != {
                "batch_id", "tool_call_id", "execution_status", "action_results"}
                or type(receipt["batch_id"]) is not int or receipt["batch_id"] != self.next_batch_id
                or receipt["tool_call_id"] != call["id"]):
            raise RuntimeError("VISTA 批次回执编号或工具调用 ID 错配")
        actions, results = call["args"]["actions"], receipt["action_results"]
        if not isinstance(results, list) or len(results) != len(actions):
            raise RuntimeError("VISTA 必须接回整批每个子动作的结果，包括未执行项")
        stopped = False
        for index, (action, result) in enumerate(zip(actions, results), 1):
            if (not isinstance(result, dict) or set(result) != {
                    "index", "action", "args", "status", "action_applied", "error"}
                    or type(result["index"]) is not int or result["index"] != index
                    or result["action"] != action["action"] or result["args"] != action["args"]):
                raise RuntimeError("VISTA 子动作回执与原请求不匹配")
            status, applied = result["status"], result["action_applied"]
            if status not in ("completed", "failed", "unknown", "not_executed"):
                raise RuntimeError("VISTA 子动作回执状态无效")
            if (status == "completed" and (applied is not True or result["error"] is not None)
                    or status == "not_executed" and (applied is not False or result["error"] is not None)
                    or status == "unknown" and applied is not None
                    or status == "failed" and type(applied) is not bool):
                raise RuntimeError("VISTA 子动作施加情况与回执状态矛盾")
            if stopped and status != "not_executed":
                raise RuntimeError("VISTA 第一项失败或取消后不能继续执行批内动作")
            stopped = status != "completed"
        expected = batch_status(results, cancelled=receipt["execution_status"] == "cancelled")
        if receipt["execution_status"] != expected:
            raise RuntimeError("VISTA 批次状态与各子动作结果矛盾")

    def accept_batch_result(self, receipt, obs_after):
        call = self.handoff()
        if call["name"] != "play":
            raise RuntimeError("VISTA 当前等待的是 finish，不能接收动作回执")
        self._validate_receipt(receipt, call)
        before = self.storage.current_frame_id
        final_info, fatal = None, None
        try:
            if obs_after is None:
                raise RuntimeError("FRAME_UNAVAILABLE: 动作批次结束后未取得可靠车头图，程序终止。")
            final_info = self.storage.archive(obs_after, receipt["batch_id"])
        except Exception as error:
            fatal = error
        # Whitelist public facts. Raw controller text can contain simulator truth.
        results = []
        for item in receipt["action_results"]:
            public = {key: deepcopy(item[key]) for key in ("index", "action", "args", "status", "action_applied")}
            public["error"] = None
            if item["status"] in ("failed", "unknown"):
                public["error"] = {"code": "ACTION_PROGRESS_UNKNOWN" if item["status"] == "unknown" else "ACTION_FAILED",
                                   "message": "控制调用未正常完成，物理效果需由图像判断",
                                   "can_retry": item["status"] == "failed"}
            results.append(public)
        status = receipt["execution_status"]
        error = None if status == "completed" else {
            "code": "ACTION_PROGRESS_UNKNOWN" if status == "unknown" else "BATCH_" + status.upper(),
            "message": "批次未全部完成，未执行项没有施加控制指令",
            "can_retry": status in ("partial", "failed")}
        if status == "unknown" and fatal is None:
            fatal = RuntimeError(f"批次 {receipt['batch_id']} 的动作执行进度未知，剩余动作未执行，程序终止。")
        if final_info is None:
            error = {"code": "FRAME_UNAVAILABLE", "message": "没有取得或保存可靠的批次最终图", "can_retry": False}
        data = {"batch_id": receipt["batch_id"], "tool_call_id": call["id"], "before_frame_id": before,
                "after_frame_id": final_info["frame_id"] if final_info else None,
                "execution_status": status, "action_results": results, "frame": final_info}
        record = {key: deepcopy(data[key]) for key in (
            "batch_id", "tool_call_id", "before_frame_id", "after_frame_id", "execution_status", "action_results")}
        record.update(actions=deepcopy(call["args"]["actions"]), expectation=call["args"]["expectation"])
        result = {"ok": status == "completed" and final_info is not None, "data": data, "error": error}
        try:
            self.storage.append_batch(record)
            self.storage.event("tool_result", tool_call_id=call["id"], result=result)
            messages = [ToolMessage(tool_call_id=call["id"], content=json.dumps(result, ensure_ascii=False))]
            if final_info:
                metadata = {**final_info, "tool_call_id": call["id"], "batch_id": receipt["batch_id"]}
                messages.append(image_message([(metadata, self.storage.load_image(final_info["frame_id"]))],
                                              "这是该 play 批次结束或停止后的真实车头观察。控制完成不保证任务成功。"))
            self._append(messages)
        except Exception as error:
            if fatal is None:
                fatal = error
            else:
                fatal.add_note(f"批次诊断保存也失败：{type(error).__name__}")
        if fatal is not None:
            self._failed = True
            self.storage.diagnostic("batch_fatal", fatal, batch_id=receipt["batch_id"])
            raise fatal

    def accept_finish_result(self, receipt):
        call = self.handoff()
        if (call["name"] != "finish" or not isinstance(receipt, dict)
                or set(receipt) != {"tool_call_id", "accepted"}
                or receipt["tool_call_id"] != call["id"] or receipt["accepted"] is not True):
            raise RuntimeError("VISTA finish 回执 ID 或接受状态错误")
        result = {"ok": True, "data": {"accepted": True, "model_success": call["args"]["success"],
                                      "reason": call["args"]["reason"]}, "error": None}
        self.storage.event("tool_result", tool_call_id=call["id"], result=result)
        self._append([ToolMessage(tool_call_id=call["id"], content=json.dumps(result, ensure_ascii=False))])
        self._finished = True

    def end_episode(self, outcome):
        self._require_active()
        check_current_round_results(self.state["messages"])
        self.storage.outcome(outcome)
        self._active = False

    def abort_episode(self, error, outcome):
        self._failed, self._active = True, False
        if self.storage.episode_dir is not None:
            self.storage.diagnostic("episode_fatal", error)
            try:
                self.storage.outcome({**outcome, "success": False, "error": str(error)})
            except Exception as secondary:
                error.add_note(f"VISTA 结局保存也失败：{type(secondary).__name__}")

    def close(self):
        if not self._closed:
            self._closed = True
            close = getattr(self.model, "close", None)
            if close is not None:
                close()
