#!/usr/bin/env python3
"""VISTA 数据采集入口：视觉决策、动作批次、仿真与记录。

用法：python run_collect.py --episodes 1 --gui --max-decisions 64
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import signal
import sys
import time
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" or "--gui" in sys.argv
                      else "egl")  # 必须在导入仿真模块之前设置。

import config  # noqa: E402
from brains.vista.base import Decision  # noqa: E402
from control.drivers import ActionFailed, GraspNotConfirmed  # noqa: E402
from control.tools import ToolLayer  # noqa: E402
from recorder import DatasetStore, EpisodeRecorder, _atomic_json  # noqa: E402
from sim.env import MobileManipEnv, parse_task  # noqa: E402
from viz import CameraRig, GuiViewer  # noqa: E402


def build_brain(memory_dir=None):
    from brains.vista import VistaBrain
    return VistaBrain(memory_dir=memory_dir)


def _run_vista_rounds(env, brain, tools, recorder, task_text, tool_schemas,
                      history, outcome, gui, verbose):
    """Execute each accepted play exactly once and settle it before any next model call."""
    from brains.vista.state import action_result, batch_status, can_continue_close_pair

    for index in range(getattr(brain, "max_decisions", config.MAX_DECISIONS)):
        if gui is not None and not gui.is_running():
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
            break
        outcome["decision_rounds"] = index + 1
        if gui is not None:
            gui.set_status(f"VISTA | round {index + 1}\nThinking...")
        decisions = brain.decide(env.get_obs().current, task_text, tool_schemas, history)
        call = brain.handoff()
        if call["name"] == "finish":
            expected = [Decision(call["args"]["reason"], "done", {"success": call["args"]["success"]})]
            if decisions != expected:
                raise RuntimeError("VISTA finish 与返回的 Decision 不一致")
            brain.accept_finish_result({"tool_call_id": call["id"], "accepted": True})
            decision = decisions[0]
            recorder.log_decision(t=env.data.time, thought=decision.thought, tool="done",
                                  args=decision.args, ok=True, result="已接受模型结束请求",
                                  extra={"decision_round": index + 1, "tool_call_id": call["id"]})
            history.append({"tool": "done", "args": decision.args, "ok": True, "result": "已接受结束请求"})
            outcome.update(brain_success=call["args"]["success"], reason="brain 请求结束", stop_code="brain_done")
            break

        actions = call["args"]["actions"]
        if [{"action": decision.tool, "args": decision.args} for decision in decisions] != actions:
            raise RuntimeError("VISTA play 与返回的 Decision 批次不一致")
        batch_id = brain.active_batch_id
        brain.record_batch_start(batch_id)
        results = [action_result(i, action, "not_executed") for i, action in enumerate(actions, 1)]
        fatal = None
        cancelled = False
        for action_index, decision in enumerate(decisions):
            known_failure = False
            grasp_not_confirmed = False
            if gui is not None and not gui.is_running():
                cancelled = True
                break
            if gui is not None:
                gui.set_status(f"VISTA | batch {batch_id} | action {action_index + 1}/{len(actions)}\n{decision.tool}")
            start_frame = env.current_frame_no
            try:
                ok, raw_result = tools.execute(decision.tool, decision.args)
                if type(ok) is not bool or not isinstance(raw_result, str):
                    raise RuntimeError("控制器返回了无效结果")
            except ActionFailed as error:
                ok, raw_result, known_failure = False, str(error), True
                grasp_not_confirmed = isinstance(error, GraspNotConfirmed)
            except BaseException as error:
                ok, raw_result = False, f"控制执行中断: {type(error).__name__}"
                fatal = error
            # A false boolean from this controller does not prove the robot never moved.
            results[action_index] = action_result(action_index + 1, actions[action_index],
                                                   "completed" if ok else "failed" if known_failure else "unknown")
            results[action_index]["frame_range"] = [start_frame, env.current_frame_no]
            if known_failure:
                results[action_index]["action_applied"] = True
            if grasp_not_confirmed:
                results[action_index]["error"]["code"] = "GRASP_NOT_CONFIRMED"
            if not ok and not known_failure and fatal is None:
                fatal = RuntimeError(f"VISTA 批次 {batch_id} 第 {action_index + 1} 个动作执行进度未知；"
                                     "剩余动作未执行，程序终止。")
            history.append({"tool": decision.tool, "args": dict(decision.args), "ok": ok, "result": raw_result})
            outcome["decisions"] = len(history)
            try:
                # Raw controller diagnostics stay in human-facing collector logs only.
                recorder.log_decision(t=env.data.time, thought=decision.thought, tool=decision.tool,
                                      args=decision.args, ok=ok, result=raw_result,
                                      extra={"decision_round": index + 1, "batch_id": batch_id,
                                             "action_index": action_index + 1, "tool_call_id": call["id"]})
            except Exception as error:
                if fatal is None:
                    fatal = RuntimeError(f"VISTA 批次 {batch_id} 的采集记录写入失败（{type(error).__name__}）")
                else:
                    fatal.add_note("采集器未能保存该子动作的日志")
            if verbose:
                print(f"  [批次{batch_id:02d}/动作{action_index+1:02d}/轮{index+1:02d}] "
                      f"{decision.tool}({decision.args}) "
                      f"[{'OK' if ok else 'FAILED' if known_failure else 'UNKNOWN'}]", flush=True)
            if fatal is not None:
                cancelled = ok  # A recorder failure can cancel the remainder after a known completion.
                break
            if known_failure and not can_continue_close_pair(actions, results[action_index], action_index):
                break  # Skip the rest of this batch, then let VISTA inspect and recover.

        receipt = {"batch_id": batch_id, "tool_call_id": call["id"],
                   "execution_status": batch_status(results, cancelled=cancelled), "action_results": results}
        obs_after = None
        try:
            obs_after = env.get_obs()
        except Exception as error:
            if fatal is None:
                fatal = RuntimeError(f"FRAME_UNAVAILABLE: VISTA 批次 {batch_id} 结束后取图失败（{type(error).__name__}）")
            else:
                fatal.add_note("批次停止后也未能取得最终车头图")
        try:
            brain.accept_batch_result(receipt, obs_after)
            env.ack_observations(obs_after.current.frame_no)
        except Exception as error:
            if fatal is None:
                raise
            if str(error) != str(fatal):
                fatal.add_note(str(error))
        if fatal is not None:
            raise fatal
        # Even the last batch has already received its ToolMessage and final PNG.
        if cancelled:
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
            break
        if env.success():
            outcome.update(reason="环境判定成功", stop_code="environment_success")
            break


def _settle_episode(env, outcome, on_tick, gui=None, verbose=True):
    """Advance a fixed physical tail, without new decisions or arm/gripper commands."""
    if gui is not None and not gui.is_running():
        outcome.update(reason="用户关闭窗口", stop_code="window_closed")
        return
    seconds = float(config.EPISODE_SETTLE_SECONDS)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("EPISODE_SETTLE_SECONDS 必须是非负有限数")
    ticks = math.ceil(seconds / config.CTRL_DT)
    start = float(env.data.time)
    settling = {"requested_seconds": seconds, "start_t_sim": round(start, 4),
                "environment_success_before": bool(env.success()), "completed": False}
    outcome["settling"] = settling
    try:
        if ticks:
            if verbose:
                print(f"  [收尾] 继续仿真 {ticks * config.CTRL_DT:g} 秒并采集，再进行最终判定", flush=True)
            if gui is not None:
                gui.set_status(f"Settling physics | {ticks * config.CTRL_DT:g}s")
            env.robot.hal.command_drive(0.0, 0.0)
        for _ in range(ticks):
            if gui is not None and not gui.is_running():
                outcome.update(reason="用户关闭窗口", stop_code="window_closed")
                break
            env.robot.hal.tick(on_tick)
        else:
            settling["completed"] = True
        if gui is not None and not gui.is_running():
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
    finally:
        settling.update(end_t_sim=round(float(env.data.time), 4),
                        elapsed_seconds=round(float(env.data.time) - start, 4))


def run_episode(env, brain, tool_schemas, task_text, ep_dir, gui=None,
                save_images: bool = True, verbose: bool = True,
                seed: int | None = None, dataset: DatasetStore | None = None,
                episode_name: str | None = None) -> dict:
    # LIFO 释放：记录文件 → 录像相机；环境相机由 env.close() 释放。
    with ExitStack() as resources:
        width, height = config.CAM_FRONT_RES
        rig = (resources.enter_context(CameraRig(env.model, cams=config.REC_CAMS,
                                                width=width, height=height))
               if save_images else None)
        gui_realtime = getattr(brain, "gui_realtime", False)
        recorder = resources.enter_context(EpisodeRecorder(ep_dir, {
            "task": task_text, "brain": brain.name, "task_kind": "pickball", "seed": seed,
            "recording_images": save_images,
        }, capture=rig.render if rig else None, dataset=dataset))
        hal = env.robot.hal
        previous_hook = getattr(hal, "before_tick", None)
        if previous_hook is not None:
            raise RuntimeError("机器人已绑定另一个执行前采集器")
        hal.before_tick = lambda: recorder.before_tick(env.model, env.data, env.robot)
        resources.callback(setattr, hal, "before_tick", previous_hook)
        last_gui_tick = time.monotonic()

        def on_tick():
            nonlocal last_gui_tick
            recording_error = None
            try:
                recorder.on_tick(env.model, env.data, env.robot)
            except BaseException as error:
                recording_error = error
            try:
                if env.observation_stream_active:
                    env.capture_observation_tick()
            except BaseException as error:
                if recording_error is None:
                    raise
                recording_error.add_note(f"VISTA 同拍采样也失败：{type(error).__name__}: {error}")
            if recording_error is not None:
                raise recording_error
            if gui is not None:
                if gui_realtime:
                    time.sleep(max(0.0, config.CTRL_DT - (time.monotonic() - last_gui_tick)))
                gui.sync()
                last_gui_tick = time.monotonic()

        tools = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal, on_tick=on_tick,
                          read_observation=lambda: env.get_obs().current)
        history: list[dict] = []
        outcome = {"success": False, "brain_success": None, "decisions": 0,
                   "decision_rounds": 0, "reason": "决策轮数耗尽", "stop_code": "decision_limit"}
        try:
            brain.begin_episode(ep_dir, task_text, tool_schemas,
                                **({"episode_name": episode_name} if episode_name is not None else {}))
            env.start_observation_stream()
            try:
                _run_vista_rounds(env, brain, tools, recorder, task_text, tool_schemas,
                                  history, outcome, gui, verbose)
                brain.finalize_collection(env.get_obs(), outcome["stop_code"])
                env.ack_observations(env.current_frame_no)
            finally:
                env.stop_observation_stream()
            if outcome["stop_code"] in ("brain_done", "decision_limit", "environment_success"):
                _settle_episode(env, outcome, on_tick, gui, verbose)
            outcome["success"] = env.success()
            if "settling" in outcome:
                outcome["settling"]["environment_success_after"] = bool(outcome["success"])
                if outcome["stop_code"] == "environment_success" and not outcome["success"]:
                    outcome["reason"] = "曾检测到成功，收尾后环境判定未通过"
            outcome["decisions"] = len(history)
            outcome["environment_success"] = outcome["success"]
            outcome["vista_episode"] = str(brain.storage.episode_dir)
            recorder.record_terminal(env.data, env.robot)
            brain.end_episode(outcome)
        except BaseException as error:
            # Interrupted/unknown execution is never eligible for training.
            window = None
            try:
                window = env.get_obs()
            except BaseException as secondary:
                # Even a second interrupt during cleanup must not skip failed
                # episode finalization or replace the original interruption.
                error.add_note(f"VISTA 中断时无法读取观测缓存：{type(secondary).__name__}")
            env.stop_observation_stream()
            brain.abort_episode(error, outcome, window)
            try:
                recorder.finish(False, {"reason": f"采集中断: {type(error).__name__}", "error": str(error),
                                        "stop_code": "interrupted",
                                        **({"settling": outcome["settling"]} if "settling" in outcome else {}),
                                        "t_sim": round(float(env.data.time), 2)})
            except Exception as secondary:
                error.add_note(f"采集中断记录也未能保存：{type(secondary).__name__}")
            raise
        else:
            # Export failure must leave the already finalized task result intact.
            published = recorder.finish(outcome["success"], {
                "reason": outcome["reason"], "brain_success": outcome["brain_success"],
                "stop_code": outcome["stop_code"], "error": outcome.get("error"),
                "environment_success": outcome["environment_success"],
                **({"settling": outcome["settling"]} if "settling" in outcome else {}),
                "decision_rounds": outcome["decision_rounds"],
                "vista_episode": outcome["vista_episode"],
                "t_sim": round(float(env.data.time), 2),
            })
            if published is not None:
                outcome["dataset"] = published
                if verbose:
                    print(f"  [数据] {published['episode_id']}: {published['status']} "
                          f"({published['h5'] or published['reason']})", flush=True)
        return outcome


def episode_path(root, seed):
    """Names are determined by scene seed, never by completion order."""
    return Path(root).resolve() / f"ep_{seed:04d}"


def check_episode_paths(args):
    memory = args.brain_memory or config.ROOT / "output" / "vista"
    for seed in range(args.seed, args.seed + args.episodes):
        for root in (args.out, memory):
            path = episode_path(root, seed)
            if path.exists():
                raise RuntimeError(f"seed={seed} 对应目录已存在：{path}；请换起始seed或输出根目录，不覆盖旧记录。")


def _collect_worker(args, seed, ep_dir):
    """Spawned process owns its renderer, physics, model client and episode files."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Parent handles graceful drain.
    started = time.monotonic()
    result_path = ep_dir / "worker_result.json"
    with (ep_dir / "collector.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                dataset = DatasetStore(args.dataset_out)
                task = parse_task(args.task)
                with ExitStack() as resources:
                    env = resources.enter_context(MobileManipEnv())
                    brain = build_brain(args.brain_memory)
                    resources.callback(brain.close)
                    if args.max_decisions is not None:
                        brain.max_decisions = args.max_decisions
                    schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()
                    env.reset(seed=seed, task=task)
                    result = run_episode(env, brain, schemas, args.task or task.text(), ep_dir,
                                         save_images=not args.no_images, verbose=not args.quiet,
                                         seed=seed, dataset=dataset, episode_name=ep_dir.name)
            except BaseException as error:
                traceback.print_exc()
                result = {"success": False, "reason": str(error), "stop_code": "worker_error",
                          "error_type": type(error).__name__, "decision_rounds": 0, "decisions": 0}
            result.update(seed=seed, wall_seconds=time.monotonic() - started, pid=os.getpid())
            _atomic_json(result_path, result)


def run_parallel(args, dataset):
    context = multiprocessing.get_context("spawn")
    pending = iter(range(args.seed, args.seed + args.episodes))
    active, results = {}, []
    exhausted = stopping = False
    started = time.monotonic()
    def request_stop(signum, frame):
        nonlocal stopping
        if stopping:
            raise KeyboardInterrupt
        stopping = True
        print("停止派发新episode；等待当前episode完成并保存。再次Ctrl+C立即中断。", flush=True)
    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        while active or not exhausted and not stopping:
            if args.stop_file and Path(args.stop_file).exists() and not stopping:
                request_stop(None, None)
            while not exhausted and not stopping and len(active) < args.workers:
                seed = next(pending, None)
                if seed is None:
                    exhausted = True
                    break
                ep_dir = episode_path(args.out, seed)
                ep_dir.parent.mkdir(parents=True, exist_ok=True)
                ep_dir.mkdir(exist_ok=False)
                process = context.Process(target=_collect_worker, args=(args, seed, ep_dir),
                                          name=f"vista-seed-{seed}")
                process.start()
                active[seed] = (process, ep_dir)
                print(f"[启动] seed={seed} -> {ep_dir.name}, pid={process.pid}", flush=True)
            for seed, (process, ep_dir) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                path = ep_dir / "worker_result.json"
                result = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else
                          {"seed": seed, "success": False, "stop_code": "worker_crash",
                           "reason": f"worker退出码{process.exitcode}，未写完成结果"})
                results.append(result)
                print(f"[完成] seed={seed} {'PASS' if result['success'] else 'FAIL'} "
                      f"({result.get('decision_rounds', 0)}轮, {result.get('wall_seconds', 0):.1f}s)", flush=True)
                process.close()
                del active[seed]
            if active:
                time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, previous)
        for process, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        dataset.recover()
    elapsed = time.monotonic() - started
    success = sum(bool(r["success"]) for r in results)
    print(f"===== 汇总: {success}/{len(results)}成功；总耗时{elapsed:.1f}s；"
          f"完成吞吐{len(results)*3600/max(elapsed, .001):.2f}集/小时 =====", flush=True)
    return 0 if len(results) == args.episodes and success == args.episodes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MuJoCo VISTA 数据采集")
    ap.add_argument("--brain", default="vista", choices=["vista"], help="当前仅保留 VISTA")
    ap.add_argument("--brain-memory", help="VISTA 图片、笔记与调用记录根目录，默认 output/vista")
    ap.add_argument("--max-decisions", type=int, help="最大外层决策轮数")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="并行episode进程数；大于1时使用无GUI采集")
    ap.add_argument("--stop-file", help="该文件出现后停止派发新episode，当前episode完整结束")
    ap.add_argument("--task", default=None, help="任务文本，默认把一个网球放进收纳箱")
    ap.add_argument("--gui", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--keep-open", action="store_true", help="采集结束后保持最后画面，关闭窗口退出")
    ap.add_argument("--no-images", action="store_true", help="关闭通用录像图片；VISTA 必需的车头 PNG 仍保存")
    ap.add_argument("--out", default=str(config.EPISODES_DIR))
    ap.add_argument("--dataset-out", default=str(config.ACT_DATA_DIR),
                    help="HDF5 和 manifest 根目录；多次运行始终追加，不重置")
    ap.add_argument("--seed", type=int, default=0, help="起始seed；seed i固定保存到ep_i，已有目录不覆盖")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes 必须至少为 1")
    if args.workers < 1 or args.seed < 0:
        ap.error("--workers 必须至少为1，--seed必须非负")
    if args.workers > 1 and args.gui:
        ap.error("--workers > 1 时请去掉 --gui；各进程仍独立渲染采集图像")
    if args.keep_open and not args.gui:
        ap.error("--keep-open 需要配合 --gui")
    if args.max_decisions is not None and args.max_decisions < 1:
        ap.error("--max-decisions 必须至少为 1")

    dataset = DatasetStore(args.dataset_out)
    for recovered in dataset.recover():
        print(f"[恢复] {recovered['episode_id']}: {recovered['status']}", flush=True)
    check_episode_paths(args)
    if args.workers > 1:
        return run_parallel(args, dataset)
    task = parse_task(args.task)
    task_text = args.task or task.text()
    results = []
    with ExitStack() as resources:
        env = resources.enter_context(MobileManipEnv())
        brain = build_brain(args.brain_memory)
        resources.callback(brain.close)
        if args.max_decisions is not None:
            brain.max_decisions = args.max_decisions
        schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()
        print(f"brain={brain.name} task='{task_text}' episodes={args.episodes} out={args.out}")
        gui = None
        if args.gui:
            print("正在创建 GUI 窗口。", flush=True)
            gui = resources.enter_context(GuiViewer(env.model, env.data))
            print("GUI 已打开，开始采集。", flush=True)
        for ep in range(args.episodes):
            if args.stop_file and Path(args.stop_file).exists():
                print("停止文件已存在；不再开始新episode。", flush=True)
                break
            env.reset(seed=args.seed + ep, task=task)
            if gui:
                gui.sync()
            ep_dir = episode_path(args.out, args.seed + ep)
            ep_dir.parent.mkdir(parents=True, exist_ok=True)
            ep_dir.mkdir(exist_ok=False)
            print(f"\n== episode {ep + 1}/{args.episodes} (seed={args.seed + ep}) -> {ep_dir.name}")
            result = run_episode(env, brain, schemas, task_text, ep_dir, gui=gui,
                                 save_images=not args.no_images, verbose=not args.quiet,
                                 seed=args.seed + ep, dataset=dataset, episode_name=ep_dir.name)
            results.append(result)
            print(f"   环境判定: {'PASS' if result['success'] else 'FAIL'} "
                  f"({result['reason']}, 决策 {result['decision_rounds']} 轮 / 工具调用 {result['decisions']} 次)")
            if gui is not None and not gui.is_running():
                break
            if not result["success"] and ep + 1 < args.episodes:
                print("   本集任务失败，已排除；继续下一集。", flush=True)
        n_ok = sum(1 for result in results if result["success"])
        print(f"\n===== 汇总: {n_ok}/{len(results)} 成功 =====")
        if args.keep_open and gui.is_running():
            gui.set_status("VISTA collection finished | Close window to exit")
            print("采集已结束，窗口保留最后画面；关闭窗口退出。", flush=True)
            while gui.is_running():
                gui.sync()
                time.sleep(0.02)
    return 0 if n_ok == args.episodes else 1


def cli(argv=None) -> int:
    try:
        return main(argv)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        for note in getattr(error, "__notes__", []):
            print(f"  {note}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
