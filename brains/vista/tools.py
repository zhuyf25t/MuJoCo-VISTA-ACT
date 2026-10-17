"""Nine model-facing tools. Only play/finish hand control back to the collector."""

from __future__ import annotations

import json
import math
from copy import deepcopy

from langchain_core.messages import ToolMessage
from PIL import Image

import config
from .model import image_message
from .settings import VistaSettings
from .state import validate_reply
from .storage import VistaStorage


class _Arguments(ValueError):
    def __init__(self, code, message, hint="请根据工具说明修改参数"):
        super().__init__(message)
        self.code, self.hint = code, hint


def success(data):
    return {"ok": True, "data": data, "error": None}


def failure(code, message, hint="请根据工具说明修改参数"):
    return {"ok": False, "data": None, "error": {
        "code": code, "message": message, "can_retry": True, "hint": hint}}


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _text(description, maximum=1024):
    return {"type": "string", "description": description, "minLength": 1, "maxLength": maximum}


def _integer(description, minimum=None, maximum=None):
    schema = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _check(schema, value, path="参数"):
    """Validate the small JSON Schema vocabulary used below, without coercion."""
    if "oneOf" in schema:
        name = value.get("action") if isinstance(value, dict) else None
        branch = next((item for item in schema["oneOf"]
                       if name in item["properties"]["action"]["enum"]), None)
        if branch is None:
            raise _Arguments("INVALID_ACTION", f"{path} 中的动作不受支持，done 只能通过 finish 调用")
        return _check(branch, value, path)
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "integer": type(value) is int,
             "number": type(value) is int or type(value) is float and math.isfinite(value)}.get(kind, False)
    if not valid:
        raise _Arguments("INVALID_ARGUMENT", f"{path} 必须是 {kind} 类型；不接受 null 或隐式类型转换")
    if "enum" in schema and value not in schema["enum"]:
        raise _Arguments("INVALID_ARGUMENT", f"{path} 必须取工具说明中的枚举值")
    if kind == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - props.keys():
            raise _Arguments("INVALID_ARGUMENT", f"{path} 含未知参数：{', '.join(map(str, set(value) - props.keys()))}")
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise _Arguments("INVALID_ARGUMENT", f"{path} 缺少参数：{', '.join(sorted(missing))}")
        for key, item in value.items():
            if key in props:
                _check(props[key], item, f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise _Arguments("INVALID_ARGUMENT", f"{path} 的条目数量不符合限制")
        for index, item in enumerate(value):
            _check(schema["items"], item, f"{path}[{index}]")
    elif kind == "string":
        if (not value.strip() or not schema.get("minLength", 0) <= len(value)
                <= schema.get("maxLength", math.inf)):
            raise _Arguments("INVALID_ARGUMENT", f"{path} 不能为空白，且必须符合字符数限制")
    elif kind in ("number", "integer"):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise _Arguments("INVALID_ARGUMENT", f"{path} 超出允许范围")


def action_schemas(tool_schemas: list[dict]) -> dict[str, dict]:
    """Use the real controller's definitions and limits; tighten missing enums."""
    names = {"forward", "back", "turn_left", "turn_right", "arm_pose", "shoulder", "elbow",
             "open_gripper", "close_gripper", "observe"}
    definitions = {}
    for original in tool_schemas:
        tool = original.get("function", original)
        name = tool.get("name")
        if name not in names:
            continue
        schema = deepcopy(tool.get("input_schema", tool.get("parameters")))
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise RuntimeError("VISTA 接收到不完整的控制动作定义")
        schema["additionalProperties"] = False
        description = tool.get("description", "")
        if name in ("forward", "back", "turn_left", "turn_right"):
            schema["properties"]["seconds"] = {
                "type": "number", "minimum": 0.1, "maximum": config.PRIM_MAX_S, "default": 1.0,
                "description": (f"名义秒数，默认 1.0，范围 0.1–{config.PRIM_MAX_S}。"
                                "转向以该值换算目标转角，实际仿真耗时可能不同。")}
            description = ({"forward": "底盘前进", "back": "底盘后退", "turn_left": "底盘原地左转（逆时针）",
                            "turn_right": "底盘原地右转（顺时针）"}[name]
                           + "；左右转不是横向平移。执行完会停止并稳定，用后续图片判断视觉效果。")
        elif name == "arm_pose":
            schema["properties"]["pose"]["enum"] = ["stow", "reach", "carry", "drop"]
            description += "；只改变机械臂关节目标，不主动改变夹爪开合目标；姿态名称不证明当前是否持有物体。"
        elif name in ("shoulder", "elbow"):
            schema["properties"]["delta"].update(
                minimum=-config.ARM_NUDGE_MAX, maximum=config.ARM_NUDGE_MAX)
        definitions[name] = {"description": description, "parameters": schema}
    if set(definitions) != names:
        raise RuntimeError("VISTA 需要完整的十个控制动作定义，请传入 ToolLayer 的工具 Schema")
    return definitions


def tool_schemas(settings: VistaSettings, actions: dict) -> list[dict]:
    coordinates = ("原图像素坐标：左上角 (0,0)，x 向右，y 向下；"
                   "区域为 [x,x+width) × [y,y+height)，必须在原图宽高内；不是展示裁图坐标。")
    region = _object({name: _integer(coordinates if name in ("x", "y") else "原图区域的正整数宽/高")
                      for name in ("x", "y", "width", "height")}, ("x", "y", "width", "height"))
    region["description"] = coordinates
    view = {"label": _text("这张图在比较中的角色，例如批次前", settings.max_label_chars),
            "frame_id": _text("当前 episode 内已归档的原图编号，例如 f000000；不是文件名或路径。当前图也可查询，无需先存在 history 批次记录", 64), "region": region}
    inspect_schema = _object({"question": _text("此次查图要直接看清的状态、关系或待解决疑点；原文保留在上下文中", settings.max_question_chars),
                              "views": {"type": "array", "minItems": 1, "maxItems": settings.max_views,
                                        "items": _object(view, ("label", "frame_id"))}}, ("question", "views"))
    pixel_view = {**deepcopy(view), "rows": _integer("均匀采样行数；至少 1，不能超过区域高度"),
                  "columns": _integer("均匀采样列数；至少 1，不能超过区域宽度")}
    pixels_schema = _object({"question": _text("此次取像素的检查目的", settings.max_question_chars),
                             "views": {"type": "array", "minItems": 1, "maxItems": settings.max_views,
                                       "items": _object(pixel_view, pixel_view.keys())}}, ("question", "views"))
    action_variants = []
    for name, definition in actions.items():
        action_variants.append(_object({"action": {"type": "string", "enum": [name],
                                                   "description": definition["description"]},
                                         "args": definition["parameters"]}, ("action", "args")))
    guide_scope = "scope=environment：同一套场景、相机和控制设定的经验，跨多个 episode 保留。"
    working_scope = "scope=episode：从环境初始化到任务结束的一次完整实验，跨工具调用和动作批次保留，新实验另建文件。"
    definitions = [
        ("inspect", "按 frame_id 读取当前或历史归档原图/裁剪图；图旁 observation 和 timestamp_s 属于该原图采样时刻。"
         "通过图片索引查找，不依赖 history 中已有动作记录；不推进环境、不回答 question、不改变当前观测。" + coordinates,
         inspect_schema),
        ("read_pixels", f"从原图区域的均匀网格中心读取 RGB 和坐标，不识别物体；所有区域合计最多 {settings.max_samples} 点。" + coordinates,
         pixels_schema),
        ("history", "查询本 episode 已结束或中断并结算的 play 批次；尚未提交或正在执行的批次不在历史中，初始为空。"
         "每批结束时只追加一条记录，包含全部子动作 actions、expectation、execution_status、action_results 及前后帧编号。"
         "observation 只保存本批执行前的五维实测状态，严格对应 before_frame_id；不是目标命令或执行后状态。"
         "字段为 shoulder_rad、elbow_rad（实际角度，rad），gripper_opening（实际连续开度，0 全闭、1 全开），"
         "base_v_mps（车体前向实际速度，m/s，正前进），base_w_radps（绕车体竖轴实际角速度，rad/s，正左转）。"
         "after_frame_id 仅关联执行后观测，不额外展开一组执行后状态；它也是下一批的 before_frame_id。"
         "当前观测已随图片提供，即使尚无对应 play 记录也可用 inspect 查看。返回帧编号，不自动返回图片；用 inspect 按编号查看。"
         "无范围时取最近 limit 批，始终按编号升序。",
         _object({"start_batch": _integer("包含的起始批次", 1), "end_batch": _integer("包含的结束批次", 1),
                  "limit": {**_integer("最多返回的批次数，不是动作数", 1, settings.history_max_limit),
                            "default": settings.history_default_limit}})),
        ("read_guide", "读取稳定指南。" + guide_scope, _object({})),
        ("write_guide", "完整替换指南，不是追加；保留需要的旧内容。" + guide_scope,
         _object({"content": _text("完整新版指南", settings.max_guide_chars)}, ("content",))),
        ("read_working", "读取当前计划、假设、预期和未决问题。" + working_scope, _object({})),
        ("write_working", "完整替换本次实验草稿，不会修改历史事实。" + working_scope,
         _object({"content": _text("完整新版草稿", settings.max_working_chars)}, ("content",))),
        ("play", "提交一批按数组顺序连续执行的动作，支持 arm_pose 预设和 shoulder/elbow 单关节微调。"
         "批内不再调用模型，结束后反馈最终图及同一时刻的 observation 实测状态。"
         "history 本批 observation 记录的是执行前状态。必须独占本条模型回复；expectation 是整批预期，不是成功证明。",
         _object({"actions": {"type": "array", "minItems": 1, "maxItems": settings.max_actions,
                               "items": {"oneOf": action_variants}},
                  "expectation": _text("整批结束后预期看见的变化；若用于验证，说明要直接看清什么、支持或否定假设的结果，以及何时仍无法判断", settings.max_question_chars),
                  "basis_frame_id": _text("必须等于当前观察的 frame_id，不能使用回看的旧图", 64)},
                 ("actions", "expectation", "basis_frame_id"))),
        ("finish", "提出结束当前 episode，success 仅为模型自评。独占一条回复，不执行动作、不新增图片或 play 批次。",
         _object({"success": {"type": "boolean", "description": "模型根据视觉自评任务是否完成"},
                  "reason": _text("结束原因及不确定性", settings.max_question_chars),
                  "evidence_frame_ids": {"type": "array", "maxItems": settings.max_views,
                                         "items": _text("本 episode 已归档的证据帧编号", 64)}}, ("success", "reason"))),
    ]
    return [{"type": "function", "function": {"name": name, "description": description,
                                               "parameters": schema}}
            for name, description, schema in definitions]


class VistaTools:
    def __init__(self, storage: VistaStorage, settings: VistaSettings, control_schemas: list[dict]):
        self.storage, self.settings = storage, settings
        self.actions = action_schemas(control_schemas)
        self.schemas = tool_schemas(settings, self.actions)
        self.parameters = {item["function"]["name"]: item["function"]["parameters"] for item in self.schemas}

    def _frame(self, frame_id):
        try:
            return self.storage.frame_info(frame_id)
        except KeyError:
            raise _Arguments("FRAME_NOT_FOUND", f"当前 episode 不存在图片 {frame_id}",
                             "使用已公布的 frame_id，或通过 history 查询") from None

    def _views(self, args, pixels=False):
        prepared, total = [], 0
        for view in args["views"]:
            info = self._frame(view["frame_id"])
            region = view.get("region", {"x": 0, "y": 0, "width": info["width"], "height": info["height"]})
            x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
            if x < 0 or y < 0 or width < 1 or height < 1 or x + width > info["width"] or y + height > info["height"]:
                raise _Arguments("REGION_OUT_OF_BOUNDS", f"区域超出原图 {info['width']}×{info['height']}",
                                 f"原图左上角 (0,0)；x+width<={info['width']}，y+height<={info['height']}")
            if pixels:
                if not (1 <= view["rows"] <= height and 1 <= view["columns"] <= width):
                    raise _Arguments("INVALID_GRID", "rows/columns 必须是正整数，且不能超过区域高/宽")
                total += view["rows"] * view["columns"]
            prepared.append((view, info, region))
        if total > self.settings.max_samples:
            raise _Arguments("SAMPLE_LIMIT", f"总采样点数不能超过 {self.settings.max_samples}")
        return prepared

    def invoke(self, name: str, args) -> tuple[dict | None, list]:
        """None means a validated handoff; it does NOT mean physical success."""
        try:
            if name not in self.parameters:
                raise _Arguments("UNKNOWN_TOOL", "未知工具，请使用提供的九个工具名")
            if isinstance(args, dict):
                views = args.get("views")
                if name in ("inspect", "read_pixels") and isinstance(views, list) and len(views) > self.settings.max_views:
                    raise _Arguments("TOO_MANY_VIEWS", f"一次最多 {self.settings.max_views} 幅图")
                actions = args.get("actions")
                if name == "play" and isinstance(actions, list):
                    if not actions:
                        raise _Arguments("INVALID_ACTION", "play.actions 不能为空")
                    if len(actions) > self.settings.max_actions:
                        raise _Arguments("ACTION_LIMIT", f"每批最多 {self.settings.max_actions} 个动作")
            try:
                _check(self.parameters[name], args)
            except _Arguments as error:
                if name in ("write_guide", "write_working") and isinstance(args, dict) and set(args) == {"content"}:
                    error.code = "INVALID_CONTENT"
                if name == "read_pixels" and (".rows" in str(error) or ".columns" in str(error)):
                    error.code = "INVALID_GRID"
                raise
            if name == "play":
                if args["basis_frame_id"] != self.storage.current_frame_id:
                    raise _Arguments("STALE_OBSERVATION", "动作依据不是当前观察",
                                     f"当前 frame_id={self.storage.current_frame_id}")
                return None, []
            if name == "finish":
                for frame_id in args.get("evidence_frame_ids", []):
                    self._frame(frame_id)
                return None, []
            if name in ("read_guide", "read_working"):
                return success(self.storage.read_memory(name.removeprefix("read_"))), []
            if name in ("write_guide", "write_working"):
                return success(self.storage.write_memory(name.removeprefix("write_"), args["content"])), []
            if name == "history":
                if args.get("start_batch", 1) > args.get("end_batch", math.inf):
                    raise _Arguments("INVALID_ARGUMENT", "start_batch 不能大于 end_batch")
                return success(self.storage.history(args.get("start_batch"), args.get("end_batch"),
                                                     args.get("limit", self.settings.history_default_limit))), []
            prepared = self._views(args, pixels=name == "read_pixels")
            images, output, total = [], [], 0
            for view, info, region in prepared:
                original = self.storage.load_image(view["frame_id"])
                x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
                meta = {"label": view["label"], "source_frame_id": info["frame_id"],
                        "episode_id": info["episode_id"], "original_size": [info["width"], info["height"]],
                        "region": dict(region)}
                if name == "inspect":
                    scale = self.settings.inspect_edge / max(width, height)
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    rendered = original.crop((x, y, x + width, y + height)).resize(size, Image.Resampling.NEAREST)
                    meta["rendered_size"] = list(size)
                    images.append(({**meta, "timestamp_s": info["timestamp_s"],
                                    "observation": info["observation"]}, rendered))
                else:
                    rows, columns = view["rows"], view["columns"]
                    samples = []
                    for row in range(rows):
                        sample_y = y + (2 * row + 1) * height // (2 * rows)
                        line = []
                        for column in range(columns):
                            sample_x = x + (2 * column + 1) * width // (2 * columns)
                            line.append({"x": sample_x, "y": sample_y,
                                         "rgb": list(original.getpixel((sample_x, sample_y)))})
                        samples.append(line)
                    meta["samples"] = samples
                    total += rows * columns
                output.append(meta)
            data = {"current_state_unchanged": True, "views": output}
            if name == "read_pixels":
                data["sample_count"] = total
            return success(data), images
        except _Arguments as error:
            return failure(error.code, str(error), error.hint), []

    def node(self, state):
        calls = validate_reply(state["messages"][-1])
        messages, images = [], []
        for call in calls:
            self.storage.event("tool_request", tool_call_id=call["id"], name=call["name"], args=call["args"])
        group_error = None
        if len(calls) > self.settings.max_tool_calls:
            group_error = failure("TOOL_CALL_LIMIT", f"每条回复最多 {self.settings.max_tool_calls} 个本地工具调用")
        elif len(calls) > 1 and any(call["name"] in ("play", "finish") for call in calls):
            group_error = failure("HANDOFF_MUST_BE_ALONE", "play/finish 必须独占一条回复；多个动作请放在 play.actions 内")
        for index, call in enumerate(calls):
            try:
                result, views = (deepcopy(group_error), []) if group_error else self.invoke(call["name"], call["args"])
                if result is None:
                    self.storage.event("handoff_validated", tool_call_id=call["id"], name=call["name"])
                    return {}
                self.storage.event("tool_result", tool_call_id=call["id"], result=result)
                message = ToolMessage(tool_call_id=call["id"], content=json.dumps(result, ensure_ascii=False))
                self.storage.log_message(message)
                messages.append(message)
                images.extend(({**meta, "tool_call_id": call["id"], "current_state_unchanged": True}, image)
                              for meta, image in views)
            except Exception as error:
                self.storage.diagnostic("tool_fatal", error, tool_call_id=call["id"],
                                        not_executed_ids=[item["id"] for item in calls[index + 1:]])
                raise
        # All text tool results precede any user image block, including multi-tool replies.
        if images:
            message = image_message(images, "以下是本轮 inspect 请求的归档图片；环境未变化，当前帧编号保持不变。")
            self.storage.log_message(message)
            messages.append(message)
        return {"messages": messages}
