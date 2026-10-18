"""VISTA protocol, tool boundaries, persistence and collector integration.

All model responses are scripted or served by httpx.MockTransport. No paid API.
The final smoke test uses the actual MuJoCo environment and control layer.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from brains.vista import VistaBrain
from brains.vista.model import ChatModel, api_messages, parse_reply
from brains.vista.settings import VistaSettings
from brains.vista.state import action_result, check_current_round_results
from control.tools import ToolLayer
from observations import FrameObservation, ObservationWindow, STATE_FIELDS


SCHEMAS = ToolLayer(None, None, None).schemas_anthropic()
TASK = "看图找到球并放入箱子"


def reply(*calls):
    return AIMessage(content="", tool_calls=[{"id": identifier, "name": name, "args": args}
                                             for identifier, name, args in calls])


def finish(identifier="finish"):
    return reply((identifier, "finish", {"success": False, "reason": "本次测试结束"}))


def play(identifier="play", frame="f000000", actions=None):
    return reply((identifier, "play", {"actions": actions or [{"action": "observe", "args": {}}],
                                       "expectation": "取得新的车头观察", "basis_frame_id": frame}))


def observation(value=0, size=(8, 12)):
    return {"frame_no": int(value), "images": {"front": np.full((*size, 3), value, dtype=np.uint8),
                       "overhead": np.full((*size, 3), 255, dtype=np.uint8)},
            "timestamp_s": float(value),
            "observation": {"shoulder_rad": 0.1, "elbow_rad": -0.2,
                               "gripper_opening": 0.6, "base_v_mps": 0.05, "base_w_radps": -0.1},
            "holding": "PRIVATE_HOLDING", "base_pose": "PRIVATE_POSE", "tcp_pos": "PRIVATE_TCP"}


class QueueModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.closed = False

    def invoke(self, messages, *, tools, system):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system})
        if not self.replies:
            raise AssertionError("unexpected model request")
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value(self.requests[-1]) if callable(value) else value

    def close(self):
        self.closed = True


@pytest.fixture
def brain_factory(tmp_path):
    created = []

    def create(replies, budget=8, begin=True, **settings):
        configuration = VistaSettings(max_model_calls=budget, output_root=tmp_path / "vista", **settings)
        brain = VistaBrain(settings=configuration, model=QueueModel(replies))
        created.append(brain)
        if begin:
            brain.begin_episode(tmp_path / "collector" / "ep_0000", TASK, SCHEMAS)
        return brain

    yield create
    for brain in created:
        brain.close()


def decide(brain, obs=None):
    return brain.decide(as_frame(observation() if obs is None else obs), TASK, SCHEMAS,
                        [{"result": "PRIVATE_RAW_CONTROL_RESULT", "holding": True}])


def as_frame(obs):
    if isinstance(obs, FrameObservation):
        return obs
    return FrameObservation(obs["frame_no"], obs["images"]["front"],
                            tuple(obs["observation"][key] for key in STATE_FIELDS))


def receipt(brain, statuses=None):
    call = brain.handoff()
    actions = call["args"]["actions"]
    cursor = int(brain.storage.current_frame_id[1:])
    items = []
    for i, (action, status) in enumerate(zip(actions, statuses or ["completed"] * len(actions)), 1):
        item = action_result(i, action, status)
        item["frame_range"] = None if status == "not_executed" else [cursor, cursor + 1]
        if status != "not_executed":
            cursor += 1
        items.append(item)
    return {"batch_id": brain.active_batch_id, "tool_call_id": call["id"],
            "execution_status": "completed", "action_results": items}


def accept(brain, value, obs):
    if obs is None:
        return brain.accept_batch_result(value, None)
    start = int(brain.storage.current_frame_id[1:])
    initial = brain.storage.frame_info(brain.storage.current_frame_id)
    frames = [FrameObservation(start, np.asarray(brain.storage.load_image(brain.storage.current_frame_id)),
                               tuple(initial["state"]))]
    last = as_frame(obs)
    frames.extend(as_frame(observation(i)) for i in range(start + 1, last.frame_no))
    if last.frame_no != start:
        frames.append(last)
    return brain.accept_batch_result(value, ObservationWindow(tuple(frames)))


def current_frame(request):
    for message in reversed(request["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, list):
            for block in reversed(message.content):
                if block.get("type") != "text":
                    continue
                try:
                    metadata = json.loads(block["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(metadata, dict) and "frame_id" in metadata:
                    return metadata["frame_id"]
    raise AssertionError("missing current observation")


def results(brain):
    return {message.tool_call_id: json.loads(message.content)
            for message in brain.state["messages"] if isinstance(message, ToolMessage)}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_batch_boundary_has_one_shared_observation(brain_factory):
    brain = brain_factory([play(), play("next", "f000001"), finish()])
    initial, middle, final = observation(), observation(1), observation(2)
    middle["observation"]["shoulder_rad"] = 0.4
    final["observation"]["shoulder_rad"] = 0.8
    initial["observation"]["private"] = "PRIVATE_EXTRA"
    decide(brain, initial)
    accept(brain, receipt(brain), middle)
    assert len(brain.model.requests) == 1
    decide(brain, middle)
    request = api_messages(brain.model.requests[-1]["messages"], "system")
    assert [item["role"] for item in request] == ["system", "user", "assistant", "tool", "user"]
    snapshots = [json.loads(block["text"]) for item in request if item["role"] == "user"
                 for block in item["content"] if block["type"] == "text" and '"observation"' in block["text"]]
    assert [info["frame_id"] for info in snapshots] == ["f000000", "f000001"]
    assert [info["observation"]["shoulder_rad"] for info in snapshots] == [0.1, 0.4]
    assert sum(block["type"] == "image_url" for item in request if item["role"] == "user"
               for block in item["content"]) == 2
    assert "observation" not in request[3]["content"]
    assert "PRIVATE_" not in json.dumps(request)
    accept(brain, receipt(brain), final)
    records = brain.storage.history()["batches"]
    assert [row["frame_range"] for row in records] == [[0, 1], [1, 2]]
    assert [row["observations"][0]["state"][0] for row in records] == [0.1, 0.4]
    assert lines(brain.storage.episode_dir / "actions.jsonl") == records
    assert len(brain.storage.frames) == 3
    assert brain.storage.frame_info("f000002")["state"] == list(as_frame(final).state)
    assert not (brain.storage.episode_dir / "frame_index.jsonl").exists()
    decide(brain, final)
    brain.accept_finish_result({"tool_call_id": "finish", "accepted": True})
    owned = [o["frame_id"] for row in lines(brain.storage.episode_dir / "actions.jsonl")
             for o in row["observations"]]
    assert owned == ["f000000", "f000001", "f000002"]

def test_history_includes_pending_observation_and_inspect_can_read_current_frame(brain_factory):
    view = lambda frame: {"question": "当前图", "views": [{"label": "当前", "frame_id": frame}]}
    brain = brain_factory([
        reply(("initial_history", "history", {}), ("initial", "inspect", view("f000000"))), play(),
        reply(("past", "history", {}), ("current", "inspect", view("f000001"))),
        play("next", "f000001"),
    ])
    before, after = observation(), observation(1)
    after["observation"]["elbow_rad"] = 0.45
    decide(brain, before)
    pending, = results(brain)["initial_history"]["data"]["batches"]
    assert pending["execution_status"] == "pending" and pending["actions"] == []
    assert pending["observations"] == [as_frame(before).record()]
    assert results(brain)["initial"]["ok"] is True
    path = brain.storage.episode_dir / "actions.jsonl"
    assert lines(path)[0]["execution_status"] == "pending"
    accept(brain, receipt(brain), after)
    saved = lines(path)[0]
    decide(brain, after)
    old, current = results(brain)["past"]["data"]["batches"]
    assert old == saved
    assert current["observations"] == [as_frame(after).record()]
    assert results(brain)["current"]["ok"] is True
    accept(brain, receipt(brain), observation(2))
    rows = lines(path)
    assert rows[0] == saved and rows[1]["observations"] == [as_frame(after).record()]
    old["observations"][0]["state"][0] = 99
    assert brain.storage.history()["batches"][0] == saved

def test_history_rejects_state_replaced_with_post_action_observation(brain_factory):
    brain = brain_factory([play()])
    decide(brain)
    after = observation(1)
    after["observation"]["gripper_opening"] = 0.1
    accept(brain, receipt(brain), after)
    path = brain.storage.episode_dir / "actions.jsonl"
    row, = lines(path)
    row["observations"][0]["state"] = list(as_frame(after).state)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_UNAVAILABLE"):
        brain.tools.invoke("history", {})


def test_archived_state_is_detached_and_inspect_uses_historical_state(brain_factory):
    brain = brain_factory([play()])
    obs = observation()
    original = deepcopy(obs["observation"])
    decide(brain, obs)
    obs["observation"]["shoulder_rad"] = 9.0
    copy = brain.storage.frame_info("f000000")
    copy["observation"]["elbow_rad"] = 9.0
    after = observation(1)
    after["observation"]["shoulder_rad"] = 0.7
    accept(brain, receipt(brain), after)
    result, images = brain.tools.invoke("inspect", {"question": "旧状态", "views": [
        {"label": "初始图", "frame_id": "f000000"}]})
    assert images[0][0]["observation"] == original
    assert "timestamp_s" not in images[0][0]
    assert "observation" not in result["data"]["views"][0]
    assert brain.storage.current_frame_id == "f000001"
    assert lines(brain.storage.episode_dir / "actions.jsonl")[0]["observations"][0]["state"] == [original[key] for key in STATE_FIELDS]


@pytest.mark.parametrize("field,value", [("shoulder_rad", float("nan")), ("elbow_rad", True),
                                         ("base_v_mps", float("inf")), ("base_w_radps", "0"),
                                         ("gripper_opening", 1.1)])
def test_invalid_measurements_are_not_published(brain_factory, field, value):
    brain = brain_factory([])
    obs = observation()
    obs["observation"][field] = value
    with pytest.raises(ValueError, match="OBSERVATION_UNAVAILABLE"):
        brain.storage.ensure_frame(as_frame(obs))
    assert brain.storage.current_frame_id is None
    assert not list((brain.storage.episode_dir / "frames").iterdir())


@pytest.mark.parametrize("change", ["state", "frame_no", "missing"])
def test_same_pixels_cannot_hide_changed_or_missing_measurements(brain_factory, change):
    brain = brain_factory([])
    obs = observation()
    brain.storage.ensure_frame(as_frame(obs))
    if change == "state":
        obs["observation"]["shoulder_rad"] += 0.1
    elif change == "frame_no":
        obs["frame_no"] += 1
    else:
        del obs["observation"]
    with pytest.raises((RuntimeError, KeyError)):
        brain.storage.check_observation(as_frame(obs))


def test_multitool_play_feedback_memory_and_second_decision(brain_factory):
    view = {"label": "初始图", "frame_id": "f000000", "region": {"x": 0, "y": 0, "width": 2, "height": 2},
            "rows": 1, "columns": 1}
    actions = [{"action": "arm_pose", "args": {"pose": "reach"}},
               {"action": "close_gripper", "args": {}}, {"action": "arm_pose", "args": {"pose": "carry"}}]
    brain = brain_factory([
        reply(("p1", "read_pixels", {"question": "比较颜色", "views": [view]}),
              ("p2", "read_pixels", {"question": "另一处颜色", "views": [view]})),
        reply(("notes", "write_working", {"content": "计划：抓取后检查球是否抬起。"})),
        play(actions=actions),
        reply(("hist", "history", {}), ("work", "read_working", {})),
        reply(("guide", "write_guide", {"content": "观察到 f000001 的颜色变化，物理效果仍待确认。"})),
        finish(),
    ])
    decisions = decide(brain)
    assert [item.tool for item in decisions] == [item["action"] for item in actions]
    assert set(brain.state) == {"messages", "model_calls"}
    assert brain.state["model_calls"] == 3
    assert isinstance(brain.state["messages"][-1], AIMessage)
    assert "play" not in results(brain)
    brain.record_batch_start(1)
    accept(brain, receipt(brain), observation(3))
    assert results(brain)["play"]["data"]["after_frame_id"] == "f000003"
    assert len(results(brain)["play"]["data"]["action_results"]) == 3
    first_size = len(brain.state["messages"])
    decide(brain, observation(3))
    assert brain.state["model_calls"] == 3  # Reset for the second decide.
    assert len(brain.model.requests[3]["messages"]) == first_size
    assert len(results(brain)["hist"]["data"]["batches"]) == 2
    assert results(brain)["work"]["data"]["scope"] == "episode"
    assert results(brain)["guide"]["data"]["scope"] == "environment"
    brain.accept_finish_result({"tool_call_id": "finish", "accepted": True})
    check_current_round_results(brain.state["messages"])
    brain.end_episode({"success": False})
    assert len(list((brain.storage.episode_dir / "frames").glob("*.png"))) == 4
    assert len(lines(brain.storage.episode_dir / "actions.jsonl")) == 2
    serialized = json.dumps(api_messages(brain.state["messages"], brain.system_prompt), ensure_ascii=False)
    assert "PRIVATE_" not in serialized
    assert "timestamp_s" not in (brain.storage.episode_dir / "actions.jsonl").read_text()


def test_inspect_results_before_all_images_and_immutable_original(brain_factory):
    views = [{"label": "局部", "frame_id": "f000000", "region": {"x": 1, "y": 1, "width": 3, "height": 2}}]
    brain = brain_factory([reply(("a", "inspect", {"question": "这里是什么", "views": views}),
                                 ("b", "inspect", {"question": "比较", "views": views})), finish()])
    obs = observation()
    obs["images"]["front"] = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    decide(brain, obs)
    messages = brain.model.requests[-1]["messages"]
    assert [type(item) for item in messages[1:]] == [AIMessage, ToolMessage, ToolMessage, HumanMessage]
    metadata = results(brain)["a"]["data"]["views"][0]
    assert metadata["rendered_size"] == [1024, 683]
    assert metadata["original_size"] == [12, 8]
    assert brain.storage.current_frame_id == "f000000"
    assert np.array_equal(np.array(brain.storage.load_image("f000000")), obs["images"]["front"])
    trace = (brain.storage.episode_dir / "messages.jsonl").read_text(encoding="utf-8")
    assert "image_reference" in trace and "base64," not in trace
    assert '"tool_call_id": "a"' in trace


def test_pixel_centres_read_original_rgb(brain_factory):
    brain = brain_factory([])
    obs = observation(size=(4, 6))
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    obs["images"]["front"] = image
    brain.storage.ensure_frame(as_frame(obs))
    value, views = brain.tools.invoke("read_pixels", {"question": "采样", "views": [
        {"label": "原图", "frame_id": "f000000", "region": {"x": 0, "y": 0, "width": 6, "height": 4},
         "rows": 2, "columns": 3}]})
    samples = value["data"]["views"][0]["samples"]
    assert [[(item["x"], item["y"]) for item in row] for row in samples] == [
        [(1, 1), (3, 1), (5, 1)], [(1, 3), (3, 3), (5, 3)]]
    assert samples[1][2]["rgb"] == image[3, 5].tolist()
    assert value["data"]["sample_count"] == 6 and views == []


@pytest.mark.parametrize("name,args,code", [
    ("unknown", {}, "UNKNOWN_TOOL"),
    ("read_guide", {"scope": "episode"}, "INVALID_ARGUMENT"),
    ("write_working", {"content": "  "}, "INVALID_CONTENT"),
    ("write_working", {"content": "x" * 4001}, "INVALID_CONTENT"),
    ("write_guide", {"content": "x" * 8001}, "INVALID_CONTENT"),
    ("history", {"limit": True}, "INVALID_ARGUMENT"),
    ("history", {"start": 3, "end": 2}, "INVALID_ARGUMENT"),
    ("history", {"mode": "frame", "step": 0}, "INVALID_ARGUMENT"),
    ("history", {"mode": "frame", "start": -1}, "INVALID_ARGUMENT"),
    ("history", {"mode": "frame", "start": True}, "INVALID_ARGUMENT"),
    ("history", {"mode": "frame", "step": 1.5}, "INVALID_ARGUMENT"),
    ("history", {"mode": "batch", "start": 0}, "INVALID_ARGUMENT"),
    ("history", {"mode": "batch", "step": 2}, "INVALID_ARGUMENT"),
    ("history", {"mode": "invalid"}, "INVALID_ARGUMENT"),
    ("finish", {"success": 1, "reason": "done"}, "INVALID_ARGUMENT"),
    ("finish", {"success": True, "reason": "done", "evidence_frame_ids": ["missing"]}, "FRAME_NOT_FOUND"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "missing"}]}, "FRAME_NOT_FOUND"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": None}]}, "INVALID_ARGUMENT"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "f000000"}] * 5}, "TOO_MANY_VIEWS"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": 1.0, "y": 0, "width": 1, "height": 1}}]}, "INVALID_ARGUMENT"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": -1, "y": 0, "width": 1, "height": 1}}]}, "REGION_OUT_OF_BOUNDS"),
    ("inspect", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": 39, "y": 0, "width": 2, "height": 1}}]}, "REGION_OUT_OF_BOUNDS"),
    ("read_pixels", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": 0, "y": 0, "width": 1, "height": 1}, "rows": True, "columns": 1}]}, "INVALID_GRID"),
    ("read_pixels", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": 0, "y": 0, "width": 1, "height": 1}, "rows": 2, "columns": 1}]}, "INVALID_GRID"),
    ("read_pixels", {"question": "?", "views": [{"label": "x", "frame_id": "f000000", "region": {
        "x": 0, "y": 0, "width": 33, "height": 33}, "rows": 33, "columns": 33}]}, "SAMPLE_LIMIT"),
])
def test_recoverable_tool_boundaries(brain_factory, name, args, code):
    brain = brain_factory([])
    brain.storage.ensure_frame(as_frame(observation(size=(40, 40))))
    value, images = brain.tools.invoke(name, args)
    assert value["ok"] is False and value["error"]["code"] == code
    assert value["error"]["can_retry"] is True and images == []
    assert brain.storage.current_frame_id == "f000000"


@pytest.mark.parametrize("action", [
    {"action": "done", "args": {"success": True}},
    {"action": "forward", "args": {"seconds": True}},
    {"action": "forward", "args": {"seconds": 0.0}},
    {"action": "forward", "args": {"seconds": float("nan")}},
    {"action": "forward", "args": {"seconds": float("inf")}},
    {"action": "forward", "args": {"seconds": 10 ** 1000}},
    {"action": "forward", "args": {"seconds": 4}},
    {"action": "observe", "args": {"seconds": 3.1}},
    {"action": "arm_pose", "args": {"pose": "typo"}},
    {"action": "elbow", "args": {"delta": -0.6}},
])
def test_entire_action_batch_is_validated_before_handoff(brain_factory, action):
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}}, action]), finish()])
    decisions = decide(brain)
    assert [decision.tool for decision in decisions] == ["done"]
    assert results(brain)["play"]["ok"] is False
    assert brain.active_batch_id == 1 and brain.storage.history()["batches"][0]["actions"] == []


@pytest.mark.parametrize("joint", ["shoulder", "elbow"])
@pytest.mark.parametrize("delta", [-0.5, 0.1, 0.5])
def test_joint_nudges_are_exposed_and_handed_off_with_their_delta(brain_factory, joint, delta):
    actions = [{"action": "forward", "args": {"seconds": 0.1}},
               {"action": joint, "args": {"delta": delta}}]
    brain = brain_factory([play(actions=actions)])
    decisions = decide(brain)
    assert [{"action": decision.tool, "args": decision.args} for decision in decisions] == actions
    assert brain.handoff()["args"]["actions"] == actions
    schema = brain.tools.parameters["play"]["properties"]["actions"]["items"]["oneOf"]
    available = {item["properties"]["action"]["enum"][0] for item in schema}
    assert {"shoulder", "elbow", "arm_pose"} <= available
    assert brain.storage.history()["batches"][0]["execution_status"] == "pending"


def test_handoff_mix_and_call_limit_never_execute_group(brain_factory):
    mixed = reply(("write", "write_working", {"content": "MUST_NOT_WRITE"}),
                  ("play", "play", play().tool_calls[0]["args"]))
    too_many = reply(*[(f"many{i}", "write_working", {"content": "MUST_NOT_WRITE"}) for i in range(9)])
    brain = brain_factory([mixed, too_many, finish()])
    decide(brain)
    assert results(brain)["write"]["error"]["code"] == "HANDOFF_MUST_BE_ALONE"
    assert all(results(brain)[f"many{i}"]["error"]["code"] == "TOOL_CALL_LIMIT" for i in range(9))
    assert "MUST_NOT_WRITE" not in brain.storage.read_memory("working")["content"]


def test_one_bad_local_call_does_not_hide_independent_results(brain_factory):
    brain = brain_factory([reply(("bad", "history", {"limit": 0}), ("good", "read_guide", {})), finish()])
    decide(brain)
    assert results(brain)["bad"]["ok"] is False and results(brain)["good"]["ok"] is True


@pytest.mark.parametrize("bad_reply", [AIMessage(content="no tools"),
    reply(("same", "read_guide", {}), ("same", "read_working", {})),
    AIMessage(content="", tool_calls=[{"id": "", "name": "read_guide", "args": {}}])])
def test_protocol_errors_stop_before_any_local_tool(brain_factory, bad_reply):
    brain = brain_factory([bad_reply])
    with pytest.raises(RuntimeError):
        decide(brain)
    assert len(brain.model.requests) == 1
    assert not any(item["event"] == "tool_request" for item in lines(brain.storage.episode_dir / "calls.jsonl"))


def test_budget_allows_nth_handoff_but_never_n_plus_one_request(brain_factory):
    brain = brain_factory([reply(("a", "read_guide", {})), play()], budget=2)
    assert decide(brain)[0].tool == "observe" and len(brain.model.requests) == 2
    other = brain_factory([reply(("a", "read_guide", {})), reply(("b", "read_guide", {}))], budget=2)
    with pytest.raises(RuntimeError, match="max_model_calls=2"):
        decide(other)
    assert len(other.model.requests) == 2


def test_no_second_decision_before_receipt_and_no_duplicate_receipt(brain_factory):
    brain = brain_factory([play()])
    decide(brain)
    value = receipt(brain)
    value["tool_call_id"] = "wrong"
    with pytest.raises(RuntimeError, match="错配"):
        accept(brain, value, observation(1))
    assert brain.storage.current_frame_id == "f000000"
    accept(brain, receipt(brain), observation(1))
    with pytest.raises(RuntimeError, match="已接收"):
        accept(brain, value, observation(1))
    other = brain_factory([play()])
    decide(other)
    with pytest.raises(RuntimeError, match="尚未收到"):
        decide(other)
    assert len(other.model.requests) == 1


def test_history_limits_ranges_and_missing_frames(brain_factory):
    brain = brain_factory([play("a"), play("b", "f000001"), finish()])
    decide(brain)
    accept(brain, receipt(brain), observation(1))
    decide(brain, observation(1))
    accept(brain, receipt(brain), observation(2))
    value = brain.storage.history(limit=1)
    assert value["truncated"] is True and value["returned_range"] == [2, 2]
    assert brain.storage.history(start=1, end=1)["batches"][0]["tool_call_id"] == "a"
    assert brain.storage.history(start=99)["batches"] == []
    (brain.storage.episode_dir / "actions.jsonl").write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_UNAVAILABLE"):
        brain.tools.invoke("history", {})


def test_frame_history_pages_cross_batch_boundaries_without_images_or_state_changes(brain_factory):
    brain = brain_factory([play("a"), play("b", "f000003"), finish()])
    decide(brain)
    first = receipt(brain)
    first["action_results"][0]["frame_range"] = [0, 3]
    accept(brain, first, observation(3))
    decide(brain, observation(3))
    second = receipt(brain)
    second["action_results"][0]["frame_range"] = [3, 6]
    accept(brain, second, observation(6))
    assert brain.storage.history(mode="frame", start=6)["frames"][0]["batch_id"] is None
    decide(brain, observation(6))
    before = (brain.storage.episode_dir / "actions.jsonl").read_bytes()
    query = {"mode": "frame", "start": 0, "end": 6, "step": 2, "limit": 2}
    pages = []
    while query:
        result, images = brain.tools.invoke("history", query)
        assert result["ok"] and images == []
        data = result["data"]
        assert data["current_frame_no"] == 6
        assert data["available_frame_range"] == [0, 6]
        assert data["available_batch_range"] == [1, 3]
        assert data["state_fields"] == list(STATE_FIELDS)
        assert data["truncated"] == (data["next_query"] is not None)
        pages.extend(data["frames"])
        query = data["next_query"]
    assert [item["frame_no"] for item in pages] == [0, 2, 4, 6]
    assert [item["batch_id"] for item in pages] == [1, 1, 2, 3]
    for item in pages:
        assert item["state"] == list(as_frame(observation(item["frame_no"])).state)
        assert set(item) == {"frame_no", "frame_id", "state", "batch_id"}
        image_result, images = brain.tools.invoke("inspect", {"question": "核对过程帧", "views": [
            {"label": "过程", "frame_id": item["frame_id"]}]})
        assert image_result["ok"] and images[0][0]["frame_no"] == item["frame_no"]
    # Boundary f000003 belongs to the next decision, not the batch producing it.
    assert brain.storage.history(mode="frame", start=3, end=3)["frames"][0]["batch_id"] == 2
    assert brain.storage.current_frame_id == "f000006"
    assert (brain.storage.episode_dir / "actions.jsonl").read_bytes() == before
    pages[0]["state"][0] = 99
    assert brain.storage.history(mode="frame", start=0, end=0)["frames"][0]["state"][0] == 0.1


def test_history_latest_and_explicit_pages_keep_stride_and_complete_batches(brain_factory):
    brain = brain_factory([play(), finish()])
    decide(brain)
    value = receipt(brain)
    value["action_results"][0]["frame_range"] = [0, 7]
    accept(brain, value, observation(7))
    decide(brain, observation(7))
    latest = brain.storage.history(mode="frame", step=2, limit=2)
    assert [row["frame_no"] for row in latest["frames"]] == [4, 6]
    older = brain.storage.history(**latest["next_query"])
    assert [row["frame_no"] for row in older["frames"]] == [0, 2]
    assert older["next_query"] is None
    anchored = brain.storage.history(mode="frame", start=1, end=7, step=3, limit=2)
    assert [row["frame_no"] for row in anchored["frames"]] == [1, 4]
    assert brain.storage.history(**anchored["next_query"])["returned_range"] == [7, 7]
    assert brain.storage.history(mode="frame", start=8)["frames"] == []
    assert brain.storage.history(mode="frame", start=7, end=999)["returned_range"] == [7, 7]
    assert brain.storage.history(mode="frame", start=0, end=7)["truncated"] is False
    batches = brain.storage.history(start=1, end=2, limit=1)
    assert batches["returned_range"] == [1, 1] and batches["truncated"] is True
    assert len(batches["batches"][0]["observations"]) == 7
    assert brain.storage.history(**batches["next_query"])["returned_range"] == [2, 2]
    latest_batch = brain.storage.history(limit=1)
    assert latest_batch["returned_range"] == [2, 2]
    assert brain.storage.history(**latest_batch["next_query"])["returned_range"] == [1, 1]


def test_history_oversized_batch_is_recoverable_and_frame_query_still_works(brain_factory):
    brain = brain_factory([play()], history_max_observations=2)
    decide(brain)
    value = receipt(brain)
    value["action_results"][0]["frame_range"] = [0, 4]
    accept(brain, value, observation(4))
    result, images = brain.tools.invoke("history", {"mode": "batch", "start": 1, "end": 1})
    assert result["error"]["code"] == "HISTORY_LIMIT" and images == []
    assert result["error"]["can_retry"] is True
    result, _ = brain.tools.invoke("history", {"mode": "frame", "start": 1, "end": 3})
    assert result["ok"] and [row["frame_no"] for row in result["data"]["frames"]] == [1, 2, 3]
    path = brain.storage.episode_dir / "actions.jsonl"
    row, = lines(path)
    assert len(row["observations"]) == 4
    row["observations"][0]["state"][0] = 99
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_UNAVAILABLE"):
        brain.tools.invoke("history", {"mode": "frame"})


def test_frame_history_current_pending_and_zero_tick_recovery_have_one_owner(brain_factory):
    brain = brain_factory([
        reply(("history0", "history", {"mode": "frame", "start": 0, "end": 0})), play(),
        reply(("history1", "history", {"mode": "frame"})), finish(),
    ])
    assert brain.storage.history(mode="frame")["available_frame_range"] is None
    decide(brain)
    initial = results(brain)["history0"]["data"]
    assert initial["frames"][0]["batch_id"] == 1 and initial["current_frame_no"] == 0
    failed = receipt(brain, ["failed"])
    failed["execution_status"] = "failed"
    failed["action_results"][0]["frame_range"] = [0, 0]
    accept(brain, failed, observation())
    decide(brain)
    current = results(brain)["history1"]["data"]
    assert len(current["frames"]) == 1 and current["frames"][0]["batch_id"] == 2
    batches = brain.storage.history()["batches"]
    assert batches[0]["observations"] == []
    assert batches[1]["observations"] == [as_frame(observation()).record()]


def test_scope_new_episode_preserves_guide_and_resets_working(brain_factory, tmp_path):
    brain = brain_factory([finish()])
    brain.tools.invoke("write_guide", {"content": "可复用经验"})
    brain.tools.invoke("write_working", {"content": "旧实验计划"})
    decide(brain)
    brain.accept_finish_result({"tool_call_id": "finish", "accepted": True})
    old = brain.storage.episode_dir
    brain.end_episode({"success": False})
    brain.begin_episode(tmp_path / "other" / "ep_0000", TASK, SCHEMAS)
    assert brain.storage.episode_dir != old
    assert brain.storage.read_memory("guide")["content"] == "可复用经验"
    assert brain.storage.read_memory("working")["content"] != "旧实验计划"
    assert (old / "working.md").read_text(encoding="utf-8") == "旧实验计划"
    assert brain.state == {"messages": [], "model_calls": 0}


def test_archive_is_independent_of_cwd_and_rejects_tampering(brain_factory, monkeypatch, tmp_path):
    brain = brain_factory([])
    brain.storage.ensure_frame(as_frame(observation()))
    monkeypatch.chdir(tmp_path)
    assert brain.storage.load_image("f000000").size == (12, 8)
    brain.storage.frames["f000000"]["path"] = "../guide.md"
    with pytest.raises(RuntimeError, match="FRAME_UNAVAILABLE"):
        brain.storage.load_image("f000000")
    brain.storage.frames["f000000"]["path"] = "frames/f000000.png"
    (brain.storage.episode_dir / "frames/f000000.png").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="FRAME_UNAVAILABLE"):
        brain.storage.load_image("f000000")


def test_note_write_failure_does_not_claim_success_or_erase_previous(brain_factory, monkeypatch):
    import brains.vista.storage as storage
    brain = brain_factory([])
    before = brain.storage.read_memory("guide")

    def fail(*args):
        raise OSError("write failed")

    monkeypatch.setattr(storage, "_atomic_text", fail)
    with pytest.raises(RuntimeError, match="MEMORY_IO_ERROR"):
        brain.tools.invoke("write_guide", {"content": "new"})
    assert brain.storage.read_memory("guide") == before


def test_failed_final_image_records_actions_without_inventing_a_frame(brain_factory, monkeypatch):
    brain = brain_factory([play()])
    decide(brain)
    with pytest.raises(RuntimeError, match="FRAME_UNAVAILABLE"):
        accept(brain, receipt(brain), None)
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["frame_range"] == [0, 1]
    assert record["observation_error"]
    assert record["actions"][0]["status"] == "completed"
    assert record["observations"] == [as_frame(observation()).record()]
    assert brain.storage.current_frame_id == "f000000"


@pytest.mark.parametrize("bad", [None, "", " ", "0", "-1", "1.5", "true"])
def test_max_model_calls_is_required_and_strict(tmp_path, bad):
    env_file = tmp_path / ".env"
    env_file.write_text("max_model_calls=7\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="max_model_calls"):
        VistaSettings.from_env(env_file=env_file, environ={"max_model_calls": bad})
    assert VistaSettings.from_env(env_file=env_file, environ={}).max_model_calls == 7
    assert VistaSettings.from_env(env_file=env_file, environ={"max_model_calls": " 3 "}).max_model_calls == 3
    with pytest.raises(RuntimeError, match="max_model_calls"):
        VistaSettings.from_env(env_file=tmp_path / "absent", environ={})


def http_model(tmp_path, handler):
    return ChatModel(base_url="https://example.invalid/v1", api_key="TEST_SECRET", model="mock-vision",
                     transport=httpx.MockTransport(handler), request_options={},
                     env_file=tmp_path / "empty-env", environ={})


def test_http_adapter_preserves_original_calls_reasoning_and_image_order(tmp_path):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "reasoning_content": "provider continuity",
                       "tool_calls": [{"id": "a", "type": "function", "function": {"name": "inspect", "arguments": json.dumps({
                           "question": "查看原图", "views": [{"label": "初始图", "frame_id": "f000000"}]})}},
                                      {"id": "b", "type": "function", "function": {"name": "read_guide", "arguments": "{}"}}]}
        else:
            message = {"role": "assistant", "tool_calls": [{"id": "done", "type": "function", "function": {
                "name": "finish", "arguments": '{"success": false, "reason": "test"}'}}]}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "tool_calls"}]})

    backend = http_model(tmp_path, handler)
    brain = VistaBrain(settings=VistaSettings(3, tmp_path / "memory"), model=backend)
    try:
        brain.begin_episode(tmp_path / "collector", TASK, SCHEMAS)
        decide(brain)
        assert len(requests) == 2 and len(requests[0]["tools"]) == 9
        messages = requests[1]["messages"]
        assert [item["role"] for item in messages] == ["system", "user", "assistant", "tool", "tool", "user"]
        assert messages[2]["reasoning_content"] == "provider continuity"
        assert messages[2]["tool_calls"][0]["function"]["name"] == "inspect"
        assert any(block["type"] == "image_url" for block in messages[-1]["content"])
        assert "PRIVATE_" not in json.dumps(requests)
        assert "TEST_SECRET" not in (brain.storage.episode_dir / "messages.jsonl").read_text(encoding="utf-8")
    finally:
        brain.close()


@pytest.mark.parametrize("status", [400, 401, 404, 413, 429, 500])
def test_http_errors_are_one_request_and_do_not_leak_response(tmp_path, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "SECRET_RESPONSE_AND_CREDENTIALS"})

    backend = http_model(tmp_path, handler)
    try:
        with pytest.raises(RuntimeError) as error:
            backend.invoke([HumanMessage(content="test")], tools=[], system="test")
        assert len(requests) == 1 and str(status) in str(error.value)
        assert "SECRET" not in str(error.value)
    finally:
        backend.close()


def test_invalid_argument_json_can_receive_a_matched_tool_error(brain_factory):
    invalid = parse_reply({"choices": [{"message": {"tool_calls": [
        {"id": "bad-json", "type": "function", "function": {"name": "history", "arguments": "[broken"}}]}}]})
    brain = brain_factory([invalid, finish()])
    decide(brain)
    assert results(brain)["bad-json"]["error"]["code"] == "INVALID_ARGUMENT"
    assert api_messages(brain.state["messages"], "test")[2]["tool_calls"][0]["function"]["arguments"] == "[broken"


def fake_environment():
    state = {"renders": 0, "resets": 0, "closed": False, "ticks": 0, "drive_commands": []}

    class Environment:
        model = None

        def __init__(self):
            self.data = SimpleNamespace(time=0, qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(5))
            self.observation_stream_active = False
            self._frames = []
            self.robot = SimpleNamespace(base=None, arm=None, base_pose=np.zeros(3),
                                         arm_q=np.zeros(2), finger_opening=1.0, base_velocity=np.zeros(2),
                                         hal=SimpleNamespace(tick=self.tick, command_drive=self.command_drive,
                                                             applied_action=np.array([0, 0, 1, 0, 0])))

        def command_drive(self, v, w):
            state["drive_commands"].append((v, w))

        def tick(self, on_tick=None):
            import config
            if getattr(self.robot.hal, "before_tick", None):
                self.robot.hal.before_tick()
            state["ticks"] += 1
            self.data.time += config.CTRL_DT
            if on_tick:
                on_tick()

        @property
        def current_frame_no(self):
            return self._frames[-1].frame_no

        def start_observation_stream(self):
            self.observation_stream_active = True
            self._frames = [as_frame(observation())]
            state["renders"] += 1

        def capture_observation_tick(self):
            self._frames.append(as_frame(observation(self.current_frame_no + 1)))
            state["renders"] += 1

        def get_obs(self):
            return ObservationWindow(tuple(self._frames))

        def ack_observations(self, frame_no):
            assert self.current_frame_no == frame_no
            self._frames = [self._frames[-1]]

        def stop_observation_stream(self):
            self.observation_stream_active = False

        def success(self):
            return False

        def reset(self, **kwargs):
            state["resets"] += 1
            self.data.time = 0.0
            self._frames = []
            self.observation_stream_active = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

    return Environment(), state


def test_collector_settles_last_batch_even_at_decision_limit(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}}])], begin=False)
    brain.max_decisions = 1
    env, state = fake_environment()
    executed = []

    def execute(name, args):
        executed.append(name)
        return True, "PRIVATE_POSE_AND_HOLDING"

    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=execute))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert executed == ["observe", "forward"] and state["renders"] == 1
    assert outcome["stop_code"] == "decision_limit"
    assert state["ticks"] == 30 and outcome["settling"]["completed"] is True
    assert results(brain)["play"]["ok"] is True
    assert len(list((brain.storage.episode_dir / "frames").glob("*.png"))) == 1
    assert "PRIVATE_" not in (brain.storage.episode_dir / "actions.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE_" in (tmp_path / "collector/decisions.jsonl").read_text(encoding="utf-8")


def test_collector_reuses_final_observation_and_finishes_without_new_frame(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play(), finish()], begin=False)
    env, state = fake_environment()
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=lambda *a: (True, "ok")))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert state["renders"] == 1 and outcome["decision_rounds"] == 2
    assert outcome["brain_success"] is False and outcome["success"] is False
    assert results(brain)["finish"]["data"]["accepted"] is True
    records = lines(brain.storage.episode_dir / "actions.jsonl")
    assert len(records) == 2 and records[-1]["execution_status"] == "finished"
    assert [obs["frame_id"] for row in records for obs in row["observations"]] == ["f000000"]
    check_current_round_results(brain.state["messages"])


@pytest.mark.parametrize("stop", ["finish", "decision_limit"])
def test_unsuccessful_episodes_continue_with_next_seeds(brain_factory, monkeypatch, tmp_path, capsys, stop):
    import run_collect
    responses = ([finish(f"finish-{i}") for i in range(3)] if stop == "finish"
                 else [play(f"play-{i}") for i in range(3)])
    brain = brain_factory(responses, begin=False)
    env, state = fake_environment()
    seeds = []
    original_reset = env.reset

    def reset(**kwargs):
        seeds.append(kwargs["seed"])
        original_reset(**kwargs)

    env.reset = reset
    monkeypatch.setattr(run_collect, "MobileManipEnv", lambda: env)
    monkeypatch.setattr(run_collect, "build_brain", lambda *a: brain)
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(
        execute=lambda *a: (True, "ok"), schemas_anthropic=lambda: SCHEMAS))
    code = run_collect.cli(["--episodes", "3", "--seed", "7", "--max-decisions", "1",
                            "--no-images", "--out", str(tmp_path / "collector"),
                            "--dataset-out", str(tmp_path / "dataset")])
    assert code == 1  # Report unsuccessful collection only after all requested episodes.
    assert state["resets"] == 3 and seeds == [7, 8, 9]
    assert state["closed"] and brain.model.closed and len(brain.model.requests) == 3
    manifest = json.loads((tmp_path / "dataset/manifest.json").read_text(encoding="utf-8"))
    assert [entry["seed"] for entry in manifest["episodes"]] == seeds
    assert all(entry["status"] == "rejected" and entry["reason"] == "task_failed"
               and not entry["selected_for_dataset"] and entry["h5"] is None
               for entry in manifest["episodes"])
    assert not list((tmp_path / "dataset/episodes").glob("*.h5"))
    assert capsys.readouterr().out.count("继续下一集") == 2


def test_blocked_arm_recovers_without_stopping_collection(brain_factory, monkeypatch, tmp_path):
    import run_collect
    from control.drivers import ArmDriver
    brain = brain_factory([play(actions=[{"action": "arm_pose", "args": {"pose": "drop"}},
                                         {"action": "forward", "args": {"seconds": 1}}]),
                           finish("finish-0"), finish("finish-1")], begin=False)
    env, state = fake_environment()
    hal = env.robot.hal
    hal.arm_q = np.zeros(2)  # Simulate a mechanically blocked arm despite valid commands.
    hal.arm_geometry = lambda: (1, 1, np.full(2, -3.0), np.full(2, 3.0))
    hal.command_arm = lambda q: hal.applied_action.__setitem__(slice(0, 2), q)
    hal.now = lambda: env.data.time
    hal.step_ticks = lambda n, on_tick=None: [env.tick(on_tick) for _ in range(n)]
    env.robot.arm = ArmDriver(hal)
    monkeypatch.setattr(run_collect, "MobileManipEnv", lambda: env)
    monkeypatch.setattr(run_collect, "build_brain", lambda *a: brain)
    code = run_collect.cli(["--episodes", "2", "--no-images", "--out", str(tmp_path / "collector"),
                            "--dataset-out", str(tmp_path / "dataset")])
    assert code == 1 and state["resets"] == 2 and len(brain.model.requests) == 3
    record = lines(tmp_path / "vista/ep_0000/actions.jsonl")[0]
    assert record["execution_status"] == "failed"
    assert [r["status"] for r in record["actions"]] == ["failed", "not_executed"]
    assert [r["action_applied"] for r in record["actions"]] == [True, False]
    assert "残差" not in json.dumps(record, ensure_ascii=False)
    assert all(command == (0.0, 0.0) for command in state["drive_commands"])
    manifest = json.loads((tmp_path / "dataset/manifest.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in manifest["episodes"]] == ["rejected", "rejected"]


@pytest.mark.parametrize("seed,confirmed", [(1, True), (2, False)])
@pytest.mark.parametrize("closes", [1, 2])
def test_recorded_close_respects_cap_and_blocks_carry_when_unconfirmed(
        brain_factory, tmp_path, monkeypatch, seed, confirmed, closes):
    import mujoco
    import config
    from run_collect import run_episode
    from sim.env import MobileManipEnv

    states = json.loads((Path(__file__).parent / "fixtures/grasp_settling_before_close.json")
                        .read_text(encoding="utf-8"))["states"]
    state = next(row for row in states if row["seed"] == seed)
    first_confirmed = confirmed
    confirmed = confirmed or closes == 2
    actions = [{"action": "close_gripper", "args": {}} for _ in range(closes)]
    actions.append({"action": "arm_pose", "args": {"pose": "carry"}})
    brain = brain_factory([play(actions=actions),
                           finish()], begin=False)
    monkeypatch.setattr(config, "EPISODE_SETTLE_SECONDS", 0)
    source = tmp_path / "collector"
    with MobileManipEnv() as env:
        env.reset(seed=seed)
        env.data.time = state["t"]
        for field in ("qpos", "qvel", "ctrl"):
            getattr(env.data, field)[:] = state[field]
        mujoco.mj_forward(env.model, env.data)
        initial_arm_target = env.data.ctrl[env.hal.i_act_arm].copy()
        run_episode(env, brain, SCHEMAS, TASK, source, save_images=False, verbose=False)
        assert (env.hal.attached_object() is not None) is confirmed
        assert not env.hal._grasp_attempt_active
        if not confirmed:
            np.testing.assert_array_equal(env.data.ctrl[env.hal.i_act_arm], initial_arm_target)
        env.hal.step_ticks(20)
        assert (env.hal.attached_object() is not None) is confirmed

    rows = lines(source / "trajectory.jsonl")
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    close_ticks = record["actions"][closes - 1]["frame_range"][1]
    close_rows = rows[:close_ticks]
    for action in record["actions"][:closes]:
        start, end = action["frame_range"]
        assert 1 <= end - start <= 12
        if action["status"] == "failed":
            assert end - start == 12  # Static images cannot substitute for confirmation.
    np.testing.assert_allclose([row["t"] - state["t"] for row in close_rows], np.arange(close_ticks) * 0.1,
                               atol=1e-8)
    np.testing.assert_allclose([row["t_end"] - row["t"] for row in close_rows], 0.1)
    np.testing.assert_allclose([row["action"] for row in close_rows],
                               np.tile([*initial_arm_target, 0, 0, 0], (close_ticks, 1)))
    assert [row["status"] for row in record["actions"]] == (
        ["completed" if first_confirmed else "failed"] + ["completed"] * (closes - 1)
        + ["completed" if confirmed else "not_executed"])
    assert record["actions"][0]["action_applied"] is True
    assert len(brain.model.requests) == 2  # Recoverable failure returns to the model.
    if not confirmed:
        assert len(rows) == 12  # No carry ticks were executed.
        error = record["actions"][0]["error"]
        assert error["code"] == "GRASP_NOT_CONFIRMED"
        assert "未确认稳定夹持" in error["message"]
        feedback = json.dumps(api_messages(brain.model.requests[1]["messages"], "test"), ensure_ascii=False)
        assert "GRASP_NOT_CONFIRMED" in feedback and "未确认稳定夹持" in feedback
    if not first_confirmed and closes == 2:
        assert record["execution_status"] == "partial"  # Preserve the first failure honestly.
        error = record["actions"][0]["error"]
        assert error["code"] == "GRASP_NOT_CONFIRMED"
        assert "第二次回执" in error["message"] and "后续动作未执行" not in error["message"]


@pytest.mark.parametrize("requested_closes", [2, 3])
def test_empty_close_pair_stops_after_two_and_never_lifts(brain_factory, tmp_path, monkeypatch, requested_closes):
    import config
    from run_collect import run_episode
    from sim.env import MobileManipEnv
    actions = [{"action": "close_gripper", "args": {}} for _ in range(requested_closes)]
    actions.append({"action": "arm_pose", "args": {"pose": "carry"}})
    brain = brain_factory([play(actions=actions), finish()], begin=False)
    monkeypatch.setattr(config, "EPISODE_SETTLE_SECONDS", 0)
    with MobileManipEnv() as env:
        env.reset(seed=1)
        env.robot.arm.move_pose("reach")  # All balls are far away from the initial robot.
        start = env.hal.now()
        target = env.data.ctrl[env.hal.i_act_arm].copy()
        run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False, verbose=False)
        assert env.hal.now() - start == pytest.approx(2.4)
        assert env.hal.attached_object() is None and not env.hal._grasp_attempt_active
        np.testing.assert_array_equal(env.data.ctrl[env.hal.i_act_arm], target)
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert [r["status"] for r in record["actions"]] == ["failed", "failed"] + ["not_executed"] * (requested_closes - 1)
    assert len(lines(tmp_path / "collector/trajectory.jsonl")) == 24
    assert len(brain.model.requests) == 2


@pytest.mark.parametrize("failure_code,second_action", [
    ("ACTION_FAILED", "close_gripper"), ("GRASP_NOT_CONFIRMED", "observe")])
def test_close_pair_exception_does_not_allow_other_failure_continuations(brain_factory, failure_code, second_action):
    brain = brain_factory([play(actions=[{"action": "close_gripper", "args": {}},
                                         {"action": second_action, "args": {}}])])
    decide(brain)
    value = receipt(brain, ["failed", "completed"])
    value["execution_status"] = "partial"
    value["action_results"][0]["action_applied"] = True
    value["action_results"][0]["error"]["code"] = failure_code
    with pytest.raises(RuntimeError, match="失败或取消"):
        accept(brain, value, observation(2))


def test_arm_timeout_is_known_failure_but_tick_exception_stays_fatal():
    from control.drivers import ActionFailed, ArmDriver
    robot = SimpleNamespace(arm_q=np.zeros(2),
                            arm_geometry=lambda: (1, 1, np.full(2, -3.0), np.full(2, 3.0)),
                            now=lambda: 0.0, command_arm=lambda q: None)
    arm = ArmDriver(robot)
    with pytest.raises(ActionFailed, match="超时"):
        arm.move_arm_q([1, 1], timeout=-1)

    def broken_tick(on_tick=None):
        raise RuntimeError("physics failed")

    robot.tick = broken_tick
    with pytest.raises(RuntimeError, match="physics failed"):
        arm.move_arm_q([1, 1])
    assert ToolLayer(None, arm, robot).execute("arm_pose", {"pose": "drop"}) == (
        False, "工具 arm_pose 执行异常: RuntimeError: physics failed")


def test_execution_failure_stops_remaining_actions_and_entire_cli(brain_factory, monkeypatch, tmp_path, capsys):
    import run_collect
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}},
                                         {"action": "back", "args": {"seconds": 0.1}}])], begin=False)
    env, state = fake_environment()
    executed = []

    def execute(name, args):
        executed.append(name)
        return len(executed) == 1, "PRIVATE_CONTROLLER_DIAGNOSTIC"

    monkeypatch.setattr(run_collect, "MobileManipEnv", lambda: env)
    monkeypatch.setattr(run_collect, "build_brain", lambda *a: brain)
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(
        execute=execute, schemas_anthropic=lambda: SCHEMAS))
    code = run_collect.cli(["--brain=vista", "--episodes", "3", "--no-images", "--out", str(tmp_path / "collector"),
                            "--dataset-out", str(tmp_path / "dataset")])
    assert code == 1 and state["resets"] == 1 and state["closed"] and brain.model.closed
    assert executed == ["observe", "forward"] and len(brain.model.requests) == 1
    assert state["ticks"] == 0
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert [entry["status"] for entry in record["actions"]] == ["completed", "unknown", "not_executed"]
    assert [entry["action_applied"] for entry in record["actions"]] == [True, None, False]
    assert record["execution_status"] == "unknown" and record["frame_range"] == [0, 0]
    stderr = capsys.readouterr().err
    assert "ERROR:" in stderr and "进度未知" in stderr and "Traceback" not in stderr


def test_actual_mujoco_vista_batch_and_next_model_call(brain_factory, tmp_path):
    from run_collect import run_episode
    from sim.env import MobileManipEnv

    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}}]), finish()], begin=False)
    with MobileManipEnv() as env:
        env.reset(seed=73)
        before = env.data.time
        outcome = run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False, verbose=False)
        assert env.data.time > before
    assert outcome["decision_rounds"] == 2
    assert len(brain.model.requests) == 2
    assert results(brain)["play"]["data"]["execution_status"] == "completed"
    assert results(brain)["finish"]["data"]["accepted"] is True
    records = lines(brain.storage.episode_dir / "actions.jsonl")
    end = records[0]["frame_range"][1]
    assert records[0]["actions"][0]["frame_range"] == [0, 1]  # Default observe is one tick.
    assert 3 <= end <= 4  # Then forward: one motion tick plus one or two settling ticks.
    assert len(brain.storage.frames) == end + 1
    assert [obs["frame_id"] for row in records for obs in row["observations"]] == [f"f{i:06d}" for i in range(end + 1)]


def test_gui_close_cancels_rest_of_batch_and_still_settles_it(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}}])], begin=False)
    env, state = fake_environment()
    window = {"running": True}
    gui = SimpleNamespace(is_running=lambda: window["running"], sync=lambda: None,
                          set_status=lambda value: None)

    def execute(*args):
        window["running"] = False
        return True, "ok"

    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=execute))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", gui=gui,
                                      save_images=False, verbose=False)
    assert outcome["stop_code"] == "window_closed" and state["renders"] == 1
    assert state["ticks"] == 0 and "settling" not in outcome
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["execution_status"] == "cancelled"
    assert [item["status"] for item in record["actions"]] == ["completed", "not_executed"]
    check_current_round_results(brain.state["messages"])


def test_environment_success_keeps_final_receipt_even_without_finish(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play()], begin=False)
    env, state = fake_environment()
    env.success = lambda: True
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=lambda *a: (True, "ok")))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert outcome["success"] is True and outcome["brain_success"] is None
    assert outcome["stop_code"] == "environment_success"
    assert results(brain)["play"]["data"]["after_frame_id"] == "f000000"


def test_finish_self_assessment_does_not_override_environment(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([reply(("claimed", "finish", {"success": True, "reason": "看起来完成了"}))], begin=False)
    env, state = fake_environment()
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace())
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert outcome["brain_success"] is True and outcome["success"] is False
    assert state["renders"] == 1 and brain.storage.history()["batches"][0]["execution_status"] == "finished"
    assert "environment_success" not in json.dumps(results(brain))


def test_finish_settles_and_records_without_more_model_calls(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([reply(("finish", "finish", {"success": True, "reason": "球已入箱"}))], begin=False)
    env, state = fake_environment()
    env.success = lambda: state["ticks"] >= 18
    monkeypatch.setattr(run_collect.config, "EPISODE_SETTLE_SECONDS", 3.0)
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector",
                                      save_images=False, verbose=False)
    assert outcome["brain_success"] is True and outcome["environment_success"] is True
    assert state["ticks"] == 30 and state["drive_commands"] == [(0.0, 0.0)]
    assert len(brain.model.requests) == outcome["decision_rounds"] == outcome["decisions"] == 1
    assert state["renders"] == 1 and brain.storage.history()["batches"][0]["execution_status"] == "finished"
    settling = outcome["settling"]
    assert settling == {"requested_seconds": 3.0, "start_t_sim": 0.0, "end_t_sim": 3.0,
                        "elapsed_seconds": 3.0, "completed": True,
                        "environment_success_before": False, "environment_success_after": True}
    meta = json.loads((tmp_path / "collector/meta.json").read_text(encoding="utf-8"))
    vista_outcome = json.loads((brain.storage.episode_dir / "outcome.json").read_text(encoding="utf-8"))
    assert meta["settling"] == vista_outcome["settling"] == settling
    assert meta["frames"] == len(lines(tmp_path / "collector/trajectory.jsonl")) == 30
    assert meta["success"] is True and meta["t_sim"] == 3.0
    assert "settling" not in json.dumps(api_messages(brain.state["messages"], brain.system_prompt))


def test_transient_success_is_rechecked_after_full_settling(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play()], begin=False)
    env, state = fake_environment()
    env.success = lambda: state["ticks"] < 15
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=lambda *a: (True, "ok")))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector",
                                      save_images=False, verbose=False)
    assert state["ticks"] == 30 and len(brain.model.requests) == 1
    assert outcome["settling"]["environment_success_before"] is True
    assert outcome["settling"]["environment_success_after"] is False
    assert outcome["success"] is False and "收尾后环境判定未通过" in outcome["reason"]


@pytest.mark.parametrize("seconds,ticks", [(0.0, 0), (0.25, 3)])
def test_settle_duration_can_be_disabled_or_rounded_to_control_ticks(brain_factory, monkeypatch, tmp_path, seconds, ticks):
    import run_collect
    brain = brain_factory([finish()], begin=False)
    env, state = fake_environment()
    monkeypatch.setattr(run_collect.config, "EPISODE_SETTLE_SECONDS", seconds)
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector",
                                      save_images=False, verbose=False)
    assert state["ticks"] == ticks
    assert outcome["settling"]["elapsed_seconds"] == pytest.approx(ticks * run_collect.config.CTRL_DT)
    assert state["drive_commands"] == ([(0.0, 0.0)] if ticks else [])


def test_closing_gui_during_settling_stops_physics_and_keeps_partial_record(brain_factory, tmp_path):
    import run_collect
    brain = brain_factory([finish()], begin=False)
    env, state = fake_environment()
    gui = SimpleNamespace(is_running=lambda: state["ticks"] < 3, sync=lambda: None,
                          set_status=lambda value: None)
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", gui=gui,
                                      save_images=False, verbose=False)
    assert outcome["stop_code"] == "window_closed" and state["ticks"] == 3
    assert outcome["settling"]["completed"] is False
    assert outcome["settling"]["elapsed_seconds"] == 0.3
    assert len(lines(tmp_path / "collector/trajectory.jsonl")) == 3


def test_recorded_drop_passes_after_settling_without_extra_model_actions(brain_factory, tmp_path, monkeypatch):
    """Reproduce ep_0018's released-but-still-rolling ball using its action sequence."""
    import config
    import h5py
    from PIL import Image
    from recorder import DatasetStore
    from run_collect import run_episode
    from sim.env import MobileManipEnv
    from control.drivers import ArmDriver

    # This historical action sequence depends on the original high carry/drop.
    # Freeze its calibration so it keeps testing the unsettled -> settled result;
    # current presets have separate real-physics coverage in test_arm_presets.py.
    recorded_presets = {"stow": [0.15, -0.6], "reach": [1.15, 1.46],
                        "carry": [0.55, -1.1], "drop": [0.48503, 1.44]}
    monkeypatch.setattr(config, "ARM_STOW", recorded_presets["stow"])
    monkeypatch.setattr(ArmDriver, "_PRESETS", recorded_presets)

    actions = [
        ("forward", {"seconds": 1.0}), ("forward", {"seconds": 1.5}),
        ("turn_left", {"seconds": 0.3}), ("forward", {"seconds": 1.5}),
        ("turn_left", {"seconds": 0.12}), ("forward", {"seconds": 1.0}),
        ("forward", {"seconds": 0.6}), ("arm_pose", {"pose": "reach"}),
        ("close_gripper", {}), ("arm_pose", {"pose": "carry"}),
        ("turn_right", {"seconds": 0.7}), ("arm_pose", {"pose": "drop"}),
        ("turn_right", {"seconds": 0.35}), ("open_gripper", {}),
    ]
    replies = [lambda request, i=i, name=name, args=args: play(
                   identifier=f"play-{i}", frame=current_frame(request), actions=[{"action": name, "args": args}])
               for i, (name, args) in enumerate(actions)]
    replies.append(reply(("finish", "finish", {"success": True, "reason": "球已入箱"})))
    brain = brain_factory(replies, begin=False)
    dataset = DatasetStore(tmp_path / "dataset")
    with MobileManipEnv() as env:
        env.reset(seed=0)
        outcome = run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector",
                              save_images=True, verbose=False, dataset=dataset)
        np.testing.assert_allclose(env.data.ctrl[env.hal.i_act_arm], recorded_presets["drop"])
        assert env.data.ctrl[env.hal.i_act_fi] == config.FINGER_OPEN_CTRL
        assert env.success() is True
    assert outcome["settling"]["environment_success_before"] is False
    assert outcome["settling"]["environment_success_after"] is True
    assert outcome["settling"]["elapsed_seconds"] == 3.0 and outcome["success"] is True
    assert len(brain.model.requests) == outcome["decision_rounds"] == 15
    records = lines(brain.storage.episode_dir / "actions.jsonl")
    assert len(records) == 15 and records[-1]["execution_status"] == "finished"
    tail = [row for row in lines(tmp_path / "collector/trajectory.jsonl")
            if row["t_end"] > outcome["settling"]["start_t_sim"] + 1e-6]
    assert len(tail) == 30
    publication = outcome["dataset"]
    rows = lines(tmp_path / "collector/trajectory.jsonl")
    assert len(brain.storage.frames) == len(rows) - len(tail) + 1
    assert [obs["frame_id"] for row in records for obs in row["observations"]] == [
        f"f{i:06d}" for i in range(len(rows) - len(tail) + 1)]
    assert publication["status"] == "ready" and publication["num_steps"] == len(rows)
    assert "split" not in publication
    with h5py.File(dataset.root / publication["h5"], "r") as exported:
        assert exported["observations/images/front"].shape == (len(rows), 240, 320, 3)
        for index in (0, len(rows) // 2, len(rows) - 1):
            row = rows[index]
            with Image.open(tmp_path / "collector" / row["imgs"][config.CAM_FRONT]) as image:
                np.testing.assert_array_equal(exported["observations/images/front"][index], np.asarray(image))
            np.testing.assert_allclose(exported["observations/state"][index], row["state"], atol=1e-7)
            np.testing.assert_allclose(exported["action"][index], row["action"], atol=1e-7)


def test_after_action_storage_failure_never_replays_and_retains_actual_results(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play()], begin=False)
    env, state = fake_environment()
    execute_count = []
    original_archive = brain.storage.ensure_frame

    def archive(obs):
        if obs.frame_no > 0:
            raise RuntimeError("FRAME_UNAVAILABLE: test disk failure")
        return original_archive(obs)

    def tool_factory(*args, **kwargs):
        def execute(*args):
            execute_count.append(1)
            env.tick(kwargs["on_tick"])
            return True, "control completed"
        return SimpleNamespace(execute=execute)

    monkeypatch.setattr(brain.storage, "ensure_frame", archive)
    monkeypatch.setattr(run_collect, "ToolLayer", tool_factory)
    with pytest.raises(RuntimeError, match="test disk failure"):
        run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert len(execute_count) == 1 and len(brain.model.requests) == 1
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["frame_range"] == [0, 1] and record["actions"][0]["action_applied"] is True
    assert record["actions"][0]["status"] == "completed" and record["observation_error"]
    assert list(brain.storage.frames) == ["f000000"]
    assert len(lines(tmp_path / "collector/trajectory.jsonl")) == 1


def test_known_partial_failure_is_returned_to_model_without_replaying_completed_actions(brain_factory):
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}},
                                         {"action": "back", "args": {"seconds": 0.1}}]), finish()])
    decide(brain)
    value = receipt(brain, ["completed", "failed", "not_executed"])
    value["execution_status"] = "partial"
    value["action_results"][1]["error"]["message"] = "PRIVATE_CONTROLLER_TRUTH"
    accept(brain, value, observation(2))
    decide(brain, observation(2))
    result = results(brain)["play"]
    assert result["ok"] is False and result["error"]["can_retry"] is True
    assert "PRIVATE_CONTROLLER_TRUTH" not in json.dumps(result)
    assert len(brain.model.requests) == 2


def test_transport_timeout_is_sanitized_and_never_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("TEST_SECRET_PRIVATE_URL", request=request)

    model = http_model(tmp_path, handler)
    try:
        with pytest.raises(RuntimeError, match="ReadTimeout") as error:
            model.invoke([], tools=[], system="test")
        assert len(calls) == 1 and "TEST_SECRET" not in str(error.value)
    finally:
        model.close()


@pytest.mark.parametrize("response", [
    {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]},
    {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {"name": "play", "arguments": "{}"}}]}}]},
    {"choices": [{"message": {"tool_calls": [{"id": "a", "type": "function", "function": {"name": "play", "arguments": 3}}]}}]},
])
def test_invalid_response_envelope_cannot_become_an_action(response):
    with pytest.raises(RuntimeError):
        parse_reply(response)
