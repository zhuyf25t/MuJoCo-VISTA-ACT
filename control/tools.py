"""工具层: LLM 可见的低层动作原语 + 视觉 look.

风格: turn_left/turn_right/forward/back(秒) + 臂预设/微调 + 手指开合.
没有坐标式 move-to; 大脑看图 → 原语动作 → 再看图, 闭环控制.
只依赖 hal 接口 —— 换真机后端本类不动.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import config
from hal import RobotInterface
from .drivers import ActionFailed, ArmDriver, BaseDriver
from . import perception


class Tool:
    def __init__(self, name: str, desc: str, params: dict, required: list[str],
                 fn: Callable[..., tuple[bool, str]]):
        self.name = name
        self.desc = desc
        self.params = params
        self.required = required
        self.fn = fn

    def schema(self) -> dict:
        return {"type": "object", "properties": self.params, "required": self.required}


def _p(desc: str, ptype: str = "number", mn: float | None = None, mx: float | None = None):
    s: dict[str, Any] = {"type": ptype, "description": desc}
    if mn is not None:
        s["minimum"] = mn
    if mx is not None:
        s["maximum"] = mx
    return s


def _s(desc: str, mn: float = 0.1, mx: float = None):
    return _p(f"{desc} (秒, 默认1)", mn=mn, mx=mx or config.PRIM_MAX_S)


class ToolLayer:
    """注册并执行工具; on_tick 挂记录器实现 10Hz 采样."""

    def __init__(self, base: BaseDriver, arm: ArmDriver, robot: RobotInterface,
                 on_tick: Callable[[], None] | None = None):
        self.base = base
        self.arm = arm
        self.robot = robot
        self.on_tick = on_tick
        self.tools: dict[str, Tool] = {}
        self._register()

    def _register(self) -> None:
        def forward(seconds: float = 1.0) -> tuple[bool, str]:
            return self.base.forward(seconds, on_tick=self.on_tick)

        def back(seconds: float = 1.0) -> tuple[bool, str]:
            return self.base.back(seconds, on_tick=self.on_tick)

        def turn_left(seconds: float = 1.0) -> tuple[bool, str]:
            return self.base.turn_left(seconds, on_tick=self.on_tick)

        def turn_right(seconds: float = 1.0) -> tuple[bool, str]:
            return self.base.turn_right(seconds, on_tick=self.on_tick)

        def arm_pose(pose: str) -> tuple[bool, str]:
            return self.arm.move_pose(pose, on_tick=self.on_tick)

        def shoulder(delta: float) -> tuple[bool, str]:
            return self.arm.nudge("shoulder", delta, on_tick=self.on_tick)

        def elbow(delta: float) -> tuple[bool, str]:
            return self.arm.nudge("elbow", delta, on_tick=self.on_tick)

        def open_gripper() -> tuple[bool, str]:
            return self.arm.open_gripper(on_tick=self.on_tick)

        def close_gripper() -> tuple[bool, str]:
            ok, msg = self.arm.close_gripper(on_tick=self.on_tick)
            if "✗" in msg:
                # 失败瞬间臂仍在 reach 位(指尖贴地): 立即拍一帧做近距分析,
                # 把"球相对夹爪的前后/左右偏差"直接写进结果 —— 模型的抓取批次
                # 通常以 carry 结尾, reach 位的观测轮到它看时已不存在
                try:
                    hint = perception.gripper_hint(perception.sense(self.robot))
                except Exception:
                    hint = ""
                if hint:
                    msg += f" | 夹爪视角: {hint}"
            return ok, msg

        def done(success: bool) -> tuple[bool, str]:            # noqa: ARG001
            return True, "任务结束"

        def observe() -> tuple[bool, str]:
            # No pose/holding feedback: simply stop drive and allow a new frame.
            self.robot.command_drive(0.0, 0.0)
            self.robot.step_ticks(2, self.on_tick)
            return True, "已发送底盘停止指令并等待两个控制拍，可重新观察"

        add = self._add
        add(Tool("observe", "停止底盘并短暂等待，下一轮取得新图；保持机械臂和夹爪目标", {}, [], observe))
        add(Tool("forward", "底盘前进", {"seconds": _s("前进时长")}, [], forward))
        add(Tool("back", "底盘后退", {"seconds": _s("后退时长")}, [], back))
        add(Tool("turn_left", "底盘原地左转(逆时针)", {"seconds": _s("左转时长")}, [], turn_left))
        add(Tool("turn_right", "底盘原地右转(顺时针)", {"seconds": _s("右转时长")}, [], turn_right))
        add(Tool("arm_pose",
                 "臂到预设姿态: stow=行驶收纳 / carry=持球携带 / reach=前伸到地面抓取位 / drop=高位投放位",
                 {"pose": _p("stow|carry|reach|drop", ptype="string")}, ["pose"], arm_pose))
        add(Tool("shoulder",
                 "肩关节角度微调：零位时上臂向上，从零位向车头倾转为正向。夹爪升降取决于当前肩、肘姿态。",
                 {"delta": _p("相对当前肩角的增量（弧度；正值增加、负值减少）",
                              mn=-config.ARM_NUDGE_MAX, mx=config.ARM_NUDGE_MAX)},
                 ["delta"], shoulder))
        add(Tool("elbow",
                 "肘关节角度微调：改变前臂相对上臂的角度，零位时两臂同向伸直，正向与肩关节相同。"
                 "夹爪升降取决于当前肩、肘姿态。",
                 {"delta": _p("相对当前肘角的增量（弧度；正值增加、负值减少）",
                              mn=-config.ARM_NUDGE_MAX, mx=config.ARM_NUDGE_MAX)},
                 ["delta"], elbow))
        add(Tool("open_gripper", "张开手指(释放球)", {}, [], open_gripper))
        add(Tool("close_gripper", "闭合手指(抓球; 球在夹爪内则持有)", {}, [], close_gripper))
        add(Tool("done", "宣告任务结束",
                 {"success": _p("自评是否完成", ptype="boolean")}, ["success"], done))

    def _add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    # ---------------- 对外接口 ----------------
    def schemas_openai(self) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.desc,
                              "parameters": t.schema()}}
                for t in self.tools.values()]

    def schemas_anthropic(self) -> list[dict]:
        return [{"name": t.name, "description": t.desc, "input_schema": t.schema()}
                for t in self.tools.values()]

    def execute(self, name: str, args: dict) -> tuple[bool, str]:
        tool = self.tools.get(name)
        if tool is None:
            return False, f"未知工具: {name} (可用: {', '.join(self.tools)})"
        try:
            args = {k: v for k, v in (args or {}).items() if k in tool.params}
            missing = [k for k in tool.required if k not in args]
            if missing:
                return False, f"工具 {name} 缺少参数: {missing}"
            return tool.fn(**args)
        except ActionFailed:
            raise  # Known applied motion failure; preserve its recoverable type.
        except TypeError as e:
            return False, f"工具 {name} 参数错误: {e}"
        except Exception as e:                       # noqa: BLE001
            return False, f"工具 {name} 执行异常: {type(e).__name__}: {e}"
