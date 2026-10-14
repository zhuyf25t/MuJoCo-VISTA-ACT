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
    return {"images": {"front": np.full((*size, 3), value, dtype=np.uint8),
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
    return brain.decide(observation() if obs is None else obs, TASK, SCHEMAS,
                        [{"result": "PRIVATE_RAW_CONTROL_RESULT", "holding": True}])


def receipt(brain, statuses=None):
    call = brain.handoff()
    actions = call["args"]["actions"]
    return {"batch_id": brain.next_batch_id, "tool_call_id": call["id"],
            "execution_status": "completed",
            "action_results": [action_result(i, action, status) for i, (action, status)
                               in enumerate(zip(actions, statuses or ["completed"] * len(actions)), 1)]}


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
    brain.accept_batch_result(receipt(brain), middle)
    assert len(brain.model.requests) == 1  # Appending feedback does not invoke the model.
    decide(brain, middle)
    request = api_messages(brain.model.requests[-1]["messages"], "system")
    assert [item["role"] for item in request] == ["system", "user", "assistant", "tool", "user"]
    snapshots = [json.loads(block["text"]) for item in request if item["role"] == "user"
                 for block in item["content"] if block["type"] == "text" and '"observation"' in block["text"]]
    assert [info["frame_id"] for info in snapshots] == ["f000000", "f000001"]
    assert [info["observation"]["shoulder_rad"] for info in snapshots] == [0.1, 0.4]
    assert sum(block["type"] == "image_url" for item in request if item["role"] == "user"
               for block in item["content"]) == 2
    assert "observation" not in request[3]["content"]  # Not also repeated in the tool receipt.
    assert "PRIVATE_" not in json.dumps(request)
    brain.accept_batch_result(receipt(brain), final)
    assert len(brain.model.requests) == 2
    records = brain.storage.history()["batches"]
    assert records[0]["after_frame_id"] == records[1]["before_frame_id"] == "f000001"
    assert [record["observation"]["shoulder_rad"] for record in records] == [0.1, 0.4]
    assert all(set(record["observation"]) == set(middle["observation"]) for record in records)
    assert lines(brain.storage.episode_dir / "actions.jsonl") == records
    frames = lines(brain.storage.episode_dir / "frame_index.jsonl")
    assert len(frames) == 3  # Terminal observation saved even without another decide().
    assert frames[-1]["observation"] == final["observation"]
    assert frames[-1]["timestamp_s"] == final["timestamp_s"]


def test_history_is_settled_only_but_inspect_can_read_current_frame(brain_factory):
    view = lambda frame: {"question": "当前图", "views": [{"label": "当前", "frame_id": frame}]}
    brain = brain_factory([
        reply(("empty", "history", {}), ("initial", "inspect", view("f000000"))), play(),
        reply(("past", "history", {}), ("current", "inspect", view("f000001"))),
        play("next", "f000001"),
    ])
    before, after = observation(), observation(1)
    after["observation"]["elbow_rad"] = 0.45
    decide(brain, before)
    assert results(brain)["empty"]["data"]["batches"] == []
    assert results(brain)["initial"]["ok"] is True
    path = brain.storage.episode_dir / "actions.jsonl"
    assert path.read_bytes() == b""  # Submitted play has no history row until settled.
    brain.accept_batch_result(receipt(brain), after)
    saved = path.read_bytes()
    decide(brain, after)
    row, = results(brain)["past"]["data"]["batches"]
    assert row["observation"] == before["observation"] != after["observation"]
    assert results(brain)["current"]["ok"] is True
    assert path.read_bytes() == saved  # Starting the next play does not rewrite history.
    brain.accept_batch_result(receipt(brain), observation(2))
    rows = lines(path)
    assert rows[0] == row and rows[1]["observation"] == after["observation"]
    assert path.read_bytes().startswith(saved)
    # Returned history objects cannot mutate the saved state or subsequent queries.
    row["observation"]["shoulder_rad"] = 99
    assert brain.storage.history()["batches"][0]["observation"] == before["observation"]


def test_history_rejects_state_replaced_with_post_action_observation(brain_factory):
    brain = brain_factory([play()])
    decide(brain)
    after = observation(1)
    after["observation"]["gripper_opening"] = 0.1
    brain.accept_batch_result(receipt(brain), after)
    path = brain.storage.episode_dir / "actions.jsonl"
    row, = lines(path)
    row["observation"] = after["observation"]
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
    brain.accept_batch_result(receipt(brain), after)
    result, images = brain.tools.invoke("inspect", {"question": "旧状态", "views": [
        {"label": "初始图", "frame_id": "f000000"}]})
    assert images[0][0]["observation"] == original
    assert images[0][0]["timestamp_s"] == 0.0
    assert "observation" not in result["data"]["views"][0]
    assert brain.storage.current_frame_id == "f000001"
    assert lines(brain.storage.episode_dir / "frame_index.jsonl")[0]["observation"] == original


@pytest.mark.parametrize("field,value", [("shoulder_rad", float("nan")), ("elbow_rad", True),
                                         ("base_v_mps", float("inf")), ("base_w_radps", "0"),
                                         ("gripper_opening", 1.1)])
def test_invalid_measurements_are_not_published(brain_factory, field, value):
    brain = brain_factory([])
    obs = observation()
    obs["observation"][field] = value
    with pytest.raises(RuntimeError, match="OBSERVATION_UNAVAILABLE"):
        brain.storage.archive(obs, None)
    assert brain.storage.current_frame_id is None
    assert not list((brain.storage.episode_dir / "frames").iterdir())


@pytest.mark.parametrize("change", ["state", "timestamp", "missing"])
def test_same_pixels_cannot_hide_changed_or_missing_measurements(brain_factory, change):
    brain = brain_factory([])
    obs = observation()
    brain.storage.archive(obs, None)
    if change == "state":
        obs["observation"]["shoulder_rad"] += 0.1
    elif change == "timestamp":
        obs["timestamp_s"] += 0.1
    else:
        del obs["observation"]
    with pytest.raises(RuntimeError):
        brain.storage.check_observation(obs)


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
    brain.accept_batch_result(receipt(brain), observation(30))
    assert results(brain)["play"]["data"]["after_frame_id"] == "f000001"
    assert len(results(brain)["play"]["data"]["action_results"]) == 3
    first_size = len(brain.state["messages"])
    decide(brain, observation(30))
    assert brain.state["model_calls"] == 3  # Reset for the second decide.
    assert len(brain.model.requests[3]["messages"]) == first_size
    assert len(results(brain)["hist"]["data"]["batches"]) == 1
    assert results(brain)["work"]["data"]["scope"] == "episode"
    assert results(brain)["guide"]["data"]["scope"] == "environment"
    brain.accept_finish_result({"tool_call_id": "finish", "accepted": True})
    check_current_round_results(brain.state["messages"])
    brain.end_episode({"success": False})
    assert len(list((brain.storage.episode_dir / "frames").glob("*.png"))) == 2
    assert len(lines(brain.storage.episode_dir / "actions.jsonl")) == 1
    serialized = json.dumps(api_messages(brain.state["messages"], brain.system_prompt), ensure_ascii=False)
    assert "PRIVATE_" not in serialized
    assert "sim_time" not in (brain.storage.episode_dir / "frame_index.jsonl").read_text()


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
    brain.storage.archive(obs, None)
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
    ("history", {"start_batch": 3, "end_batch": 2}, "INVALID_ARGUMENT"),
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
    brain.storage.archive(observation(size=(40, 40)), None)
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
    {"action": "observe", "args": {"seconds": 1}},
    {"action": "arm_pose", "args": {"pose": "typo"}},
    {"action": "elbow", "args": {"delta": -0.6}},
])
def test_entire_action_batch_is_validated_before_handoff(brain_factory, action):
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}}, action]), finish()])
    decisions = decide(brain)
    assert [decision.tool for decision in decisions] == ["done"]
    assert results(brain)["play"]["ok"] is False
    assert brain.next_batch_id == 1 and brain.storage.history()["batches"] == []


@pytest.mark.parametrize("joint", ["shoulder", "elbow"])
def test_joint_nudges_are_hidden_and_rejected_before_any_batch_handoff(brain_factory, joint):
    brain = brain_factory([play(actions=[{"action": "forward", "args": {"seconds": 0.1}},
                                        {"action": joint, "args": {"delta": 0.1}}]), finish()])
    decisions = decide(brain)
    assert [decision.tool for decision in decisions] == ["done"]
    assert results(brain)["play"]["error"]["code"] == "ACTION_DISABLED"
    schema = brain.tools.parameters["play"]["properties"]["actions"]["items"]["oneOf"]
    available = {item["properties"]["action"]["enum"][0] for item in schema}
    assert not {"shoulder", "elbow"} & available
    assert "arm_pose" in available
    assert brain.storage.history()["batches"] == []


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
        brain.accept_batch_result(value, observation(1))
    assert brain.storage.current_frame_id == "f000000"
    brain.accept_batch_result(receipt(brain), observation(1))
    with pytest.raises(RuntimeError, match="已接收"):
        brain.accept_batch_result(value, observation(1))
    other = brain_factory([play()])
    decide(other)
    with pytest.raises(RuntimeError, match="尚未收到"):
        decide(other)
    assert len(other.model.requests) == 1


def test_history_limits_ranges_and_missing_frames(brain_factory):
    brain = brain_factory([play("a"), play("b", "f000001"), finish()])
    decide(brain)
    brain.accept_batch_result(receipt(brain), observation(1))
    decide(brain, observation(1))
    brain.accept_batch_result(receipt(brain), observation(2))
    value = brain.storage.history(limit=1)
    assert value["truncated"] is True and value["returned_range"] == [2, 2]
    assert brain.storage.history(start_batch=1, end_batch=1)["batches"][0]["tool_call_id"] == "a"
    assert brain.storage.history(start_batch=99)["batches"] == []
    (brain.storage.episode_dir / "actions.jsonl").write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_UNAVAILABLE"):
        brain.tools.invoke("history", {})


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
    brain.storage.archive(observation(), None)
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
        brain.accept_batch_result(receipt(brain), None)
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["after_frame_id"] is None
    assert record["action_results"][0]["status"] == "completed"
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
            self.data = SimpleNamespace(time=0, qpos=np.zeros(1), ctrl=np.zeros(4))
            self.robot = SimpleNamespace(base=None, arm=None, base_pose=np.zeros(3),
                                         arm_q=np.zeros(2), finger_opening=1.0,
                                         hal=SimpleNamespace(tick=self.tick, command_drive=self.command_drive))

        def command_drive(self, v, w):
            state["drive_commands"].append((v, w))

        def tick(self, on_tick=None):
            import config
            state["ticks"] += 1
            self.data.time += config.CTRL_DT
            if on_tick:
                on_tick()

        def get_obs(self):
            state["renders"] += 1
            return observation(state["renders"])

        def success(self):
            return False

        def reset(self, **kwargs):
            state["resets"] += 1

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
    assert executed == ["observe", "forward"] and state["renders"] == 2
    assert outcome["stop_code"] == "decision_limit"
    assert state["ticks"] == 30 and outcome["settling"]["completed"] is True
    assert results(brain)["play"]["ok"] is True
    assert len(list((brain.storage.episode_dir / "frames").glob("*.png"))) == 2
    assert "PRIVATE_" not in (brain.storage.episode_dir / "actions.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE_" in (tmp_path / "collector/decisions.jsonl").read_text(encoding="utf-8")


def test_collector_reuses_final_observation_and_finishes_without_new_frame(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play(), finish()], begin=False)
    env, state = fake_environment()
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=lambda *a: (True, "ok")))
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert state["renders"] == 2 and outcome["decision_rounds"] == 2
    assert outcome["brain_success"] is False and outcome["success"] is False
    assert results(brain)["finish"]["data"]["accepted"] is True
    assert len(lines(brain.storage.episode_dir / "actions.jsonl")) == 1
    check_current_round_results(brain.state["messages"])


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
    code = run_collect.cli(["--brain=vista", "--episodes", "3", "--no-images", "--out", str(tmp_path / "collector")])
    assert code == 1 and state["resets"] == 1 and state["closed"] and brain.model.closed
    assert executed == ["observe", "forward"] and len(brain.model.requests) == 1
    assert state["ticks"] == 0
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert [entry["status"] for entry in record["action_results"]] == ["completed", "unknown", "not_executed"]
    assert [entry["action_applied"] for entry in record["action_results"]] == [True, None, False]
    assert record["execution_status"] == "unknown" and record["after_frame_id"] == "f000001"
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
    assert len(lines(brain.storage.episode_dir / "frame_index.jsonl")) == 2


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
    assert outcome["stop_code"] == "window_closed" and state["renders"] == 2
    assert state["ticks"] == 0 and "settling" not in outcome
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["execution_status"] == "cancelled"
    assert [item["status"] for item in record["action_results"]] == ["completed", "not_executed"]
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
    assert results(brain)["play"]["data"]["after_frame_id"] == "f000001"


def test_finish_self_assessment_does_not_override_environment(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([reply(("claimed", "finish", {"success": True, "reason": "看起来完成了"}))], begin=False)
    env, state = fake_environment()
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace())
    outcome = run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert outcome["brain_success"] is True and outcome["success"] is False
    assert state["renders"] == 1 and brain.storage.history()["batches"] == []
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
    assert state["renders"] == 1 and brain.storage.history()["batches"] == []
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


def test_recorded_drop_passes_after_settling_without_extra_model_actions(brain_factory, tmp_path):
    """Reproduce ep_0018's released-but-still-rolling ball using its action sequence."""
    import config
    from run_collect import run_episode
    from sim.env import MobileManipEnv

    actions = [
        ("forward", {"seconds": 1.0}), ("forward", {"seconds": 1.5}),
        ("turn_left", {"seconds": 0.3}), ("forward", {"seconds": 1.5}),
        ("turn_left", {"seconds": 0.12}), ("forward", {"seconds": 1.0}),
        ("forward", {"seconds": 0.6}), ("arm_pose", {"pose": "reach"}),
        ("close_gripper", {}), ("arm_pose", {"pose": "carry"}),
        ("turn_right", {"seconds": 0.7}), ("arm_pose", {"pose": "drop"}),
        ("turn_right", {"seconds": 0.35}), ("open_gripper", {}),
    ]
    replies = [play(identifier=f"play-{i}", frame=f"f{i:06d}", actions=[{"action": name, "args": args}])
               for i, (name, args) in enumerate(actions)]
    replies.append(reply(("finish", "finish", {"success": True, "reason": "球已入箱"})))
    brain = brain_factory(replies, begin=False)
    with MobileManipEnv() as env:
        env.reset(seed=0)
        outcome = run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector",
                              save_images=False, verbose=False)
        np.testing.assert_allclose(env.data.ctrl[env.hal.i_act_arm], config.ARM_DROP)
        assert env.data.ctrl[env.hal.i_act_fi] == config.FINGER_OPEN_CTRL
        assert env.success() is True
    assert outcome["settling"]["environment_success_before"] is False
    assert outcome["settling"]["environment_success_after"] is True
    assert outcome["settling"]["elapsed_seconds"] == 3.0 and outcome["success"] is True
    assert len(brain.model.requests) == outcome["decision_rounds"] == 15
    assert len(lines(brain.storage.episode_dir / "actions.jsonl")) == 14
    assert len(lines(brain.storage.episode_dir / "frame_index.jsonl")) == 15
    tail = [row for row in lines(tmp_path / "collector/trajectory.jsonl")
            if row["t"] > outcome["settling"]["start_t_sim"]]
    assert len(tail) == 30


def test_after_action_storage_failure_never_replays_and_retains_actual_results(brain_factory, monkeypatch, tmp_path):
    import run_collect
    brain = brain_factory([play()], begin=False)
    env, state = fake_environment()
    execute_count = []
    original_archive = brain.storage.archive

    def archive(obs, source_batch_id):
        if source_batch_id is not None:
            raise RuntimeError("FRAME_UNAVAILABLE: test disk failure")
        return original_archive(obs, source_batch_id)

    def execute(*args):
        execute_count.append(1)
        return True, "control completed"

    monkeypatch.setattr(brain.storage, "archive", archive)
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=execute))
    with pytest.raises(RuntimeError, match="test disk failure"):
        run_collect.run_episode(env, brain, SCHEMAS, TASK, tmp_path / "collector", save_images=False)
    assert len(execute_count) == 1 and len(brain.model.requests) == 1
    record = lines(brain.storage.episode_dir / "actions.jsonl")[0]
    assert record["after_frame_id"] is None and record["action_results"][0]["action_applied"] is True


def test_known_partial_failure_is_returned_to_model_without_replaying_completed_actions(brain_factory):
    brain = brain_factory([play(actions=[{"action": "observe", "args": {}},
                                         {"action": "forward", "args": {"seconds": 0.1}},
                                         {"action": "back", "args": {"seconds": 0.1}}]), finish()])
    decide(brain)
    value = receipt(brain, ["completed", "failed", "not_executed"])
    value["execution_status"] = "partial"
    value["action_results"][1]["error"]["message"] = "PRIVATE_CONTROLLER_TRUTH"
    brain.accept_batch_result(value, observation(2))
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
