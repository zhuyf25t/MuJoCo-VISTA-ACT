"""Explicit-seed MuJoCo closed-loop evaluation; never calls VISTA or exports training data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .common import atomic_json, file_digest, read_json
from .checkpoint import load_checkpoint, deployment_metadata
from .policy import ACTPolicy


class SimActionAdapter:
    """Five physical-unit targets, applied for one control tick.

    Closed finger targets remain eligible for the backend's confirmed-contact
    grasp rule. There is no additional timer or hidden grasp-attempt state.
    """
    def __init__(self, robot):
        self.robot = robot
        current = np.asarray(robot.applied_action, dtype=float)
        if current.shape != (5,) or not np.isfinite(current).all():
            raise ValueError("Need the known current actuator targets")
        self.gripper = float(current[2] >= .5)
        self.arm = current[:2].copy()

    def apply(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected five finite physical-unit targets")
        _, _, low, high = self.robot.arm_geometry()
        arm = np.clip(action[:2], low, high)
        # Execute per-tick targets directly; expert labels already include its arm ramp.
        g = float(np.clip(action[2], 0, 1))
        target = 1.0 if g >= .6 else 0.0 if g <= .4 else self.gripper
        if target != self.gripper:
            lo, hi = self.robot.finger_ctrl_range()
            self.robot.command_finger(hi - target * (hi - lo))
            self.gripper = target
        self.robot.command_arm(arm)
        self.robot.command_drive(float(action[3]), float(action[4]))
        self.arm = arm
        return np.asarray(self.robot.applied_action).copy()

    def stop(self):
        self.robot.command_drive(0, 0)


class TaskProgress:
    """Evaluation-only truth, sampled after physics and never passed to a policy.

    A grasp means the backend confirmed stable bilateral inner-pad contact, not
    a touch or a close command. Delivery additionally requires a released ball
    to remain slow and inside the bin for a continuous verification window.
    Object identities are deduplicated, so repeated grasps cannot inflate credit.
    """
    def __init__(self, object_names, delivery_stable_seconds=0.5):
        if not np.isfinite(delivery_stable_seconds) or delivery_stable_seconds <= 0:
            raise ValueError("Delivery verification window must be positive")
        self.names = tuple(object_names)
        self.delivery_stable_seconds = float(delivery_stable_seconds)
        self.grasped, self.delivered, self.final_delivered = set(), set(), set()
        self.in_bin_since = {}
        self.first_grasp = self.first_delivery = None
        self.last_attached = None
        self.grasp_events = []

    def update(self, env, elapsed):
        attached = env.hal.attached_object()
        if attached in self.names:
            self.grasped.add(attached)
            if self.first_grasp is None:
                self.first_grasp = float(elapsed)
            if attached != self.last_attached:
                self.grasp_events.append({"object": attached, "seconds": float(elapsed)})
        self.last_attached = attached
        self.final_delivered = set()
        for name in self.names:
            if name != attached and env.world.object_in_bin(name):
                since = self.in_bin_since.setdefault(name, float(elapsed))
                if elapsed - since >= self.delivery_stable_seconds - 1e-9:
                    self.delivered.add(name)
                    self.final_delivered.add(name)
                    if self.first_delivery is None:
                        self.first_delivery = float(elapsed)
            else:
                self.in_bin_since.pop(name, None)

    def result(self):
        return {"ever_confirmed_grasp": bool(self.grasped),
                "confirmed_grasp_count": len(self.grasped),
                "grasped_objects": sorted(self.grasped),
                "deliveries": len(self.delivered), "delivered_objects": sorted(self.delivered),
                "final_delivered_objects": sorted(self.final_delivered),
                "time_to_first_grasp_seconds": self.first_grasp,
                "time_to_first_delivery_seconds": self.first_delivery,
                "grasp_events": list(self.grasp_events)}


def checkpoint_metadata(checkpoint):
    saved = load_checkpoint(checkpoint)
    epoch = saved.get("epoch")
    if epoch is not None and (type(epoch) is not int or epoch < 0):
        raise ValueError("Invalid checkpoint epoch")
    return {"checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_digest(checkpoint),
            "checkpoint_epoch": epoch, "checkpoint_epoch_number": None if epoch is None else epoch + 1,
            "checkpoint_global_step": saved.get("global_step"),
            "checkpoint_format_version": saved["format_version"],
            "execution_config": deployment_metadata(saved)}


def _evaluate_serial(checkpoint, seeds, output, *, metadata, device="cpu", max_steps=600, gui=False,
                     video=False, on_result=None):
    import config
    from sim.env import MobileManipEnv
    from sim.backend import FRONT_RENDER_PROTOCOL
    from viz import CameraRig, GuiViewer
    if config.CTRL_HZ != 10 or config.CAM_FRONT_RES != (320, 240):
        raise ValueError("Simulation observation/control settings differ from the trained schema")
    output = Path(output)
    if file_digest(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before policy loading")
    policy = ACTPolicy(checkpoint, device)
    metadata["render_protocol"] = FRONT_RENDER_PROTOCOL
    results = []
    with MobileManipEnv() as env:
        with ExitStack() as resources:
            viewer = resources.enter_context(GuiViewer(env.model, env.data)) if gui else None
            camera = resources.enter_context(CameraRig(env.model, [config.CAM_OVERHEAD])) if video else None
            for seed in seeds:
                env.reset(seed=seed)
                policy.reset()
                adapter = SimActionAdapter(env.hal)
                writer = None
                if video:
                    import imageio.v2 as imageio
                    writer = resources.enter_context(imageio.get_writer(
                        str(output / f"seed-{seed}.mp4"), fps=config.CTRL_HZ,
                        codec="libx264", quality=8, macro_block_size=None))
                    writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                latencies, trajectory, stopped = [], [], "horizon"
                start = env.hal.now()
                progress = TaskProgress(config.BALL_NAMES)
                progress.update(env, 0.0)
                try:
                    for step in range(max_steps):
                        if viewer and not viewer.is_running():
                            stopped = "window_closed"
                            break
                        # Reading feedback refreshes MuJoCo derived state without stepping.
                        velocity = env.hal.base_velocity
                        state = np.array([*env.hal.arm_q, env.hal.finger_opening, *velocity])
                        image = env.hal.front_image()
                        started = time.perf_counter()
                        requested = policy.act(image[None], state)
                        latencies.append(time.perf_counter() - started)
                        applied = adapter.apply(requested)
                        trajectory.append({"step": step, "t": env.hal.now(), "state": state.tolist(),
                                           "requested": requested.tolist(), "applied": applied.tolist()})
                        env.hal.tick()
                        progress.update(env, env.hal.now() - start)
                        trajectory[-1]["task_progress"] = progress.result()
                        if writer:
                            writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                        if viewer:
                            viewer.set_status(f"ACT | seed {seed} | step {step + 1}")
                            viewer.sync()
                        if progress.final_delivered and env.success():
                            stopped = "success_detected"
                            break
                finally:
                    adapter.stop()
                if stopped != "window_closed":
                    for _ in range(math.ceil(config.EPISODE_SETTLE_SECONDS / config.CTRL_DT)):
                        if viewer and not viewer.is_running():
                            stopped = "window_closed"
                            break
                        env.hal.tick()
                        progress.update(env, env.hal.now() - start)
                        if writer:
                            writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                        if viewer:
                            viewer.sync()
                if writer:
                    writer.close()
                success = stopped != "window_closed" and bool(progress.final_delivered) and bool(env.success())
                result = {"seed": seed, "success": success, "final_success": success,
                          **metadata, **progress.result(),
                          "stop_reason": stopped, "control_steps": len(trajectory),
                          "sim_seconds": env.hal.now() - start,
                          "mean_policy_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
                          "p95_policy_latency_ms": float(np.quantile(latencies, .95) * 1000) if latencies else None,
                          "video": f"seed-{seed}.mp4" if video else None}
                atomic_json(output / f"seed-{seed}.json", {**result, "trajectory": trajectory})
                results.append(result)
                if on_result is not None:
                    on_result(result)
                print(result, flush=True)
                if stopped == "window_closed":
                    break
    return results


class RolloutEvaluationError(RuntimeError):
    """Incomplete evaluation; completed artifacts and explicit errors are preserved."""
    def __init__(self, output, summary):
        self.output, self.summary = str(output), summary
        self.partial_results = summary["episodes"]
        details = "; ".join(error["message"] for error in summary["errors"])
        super().__init__(f"Incomplete rollout evaluation at {output}: {details or 'missing/interrupted episodes'}")


def _error(scope, exc, **extra):
    return {"scope": scope, "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), **extra}


def _run_worker(checkpoint, seeds, output, metadata, settings, worker_id):
    """Spawn entry point: no simulator, renderer or policy objects cross processes."""
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = {"worker_id": worker_id, "worker_pid": os.getpid(), "worker_threads": torch.get_num_threads()}
    metadata = {**metadata, **identity}
    errors = []
    try:
        _evaluate_serial(checkpoint, seeds, output, metadata=metadata, **settings)
        if file_digest(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during worker evaluation")
    except Exception as exc:
        errors.append(_error("worker", exc, worker_id=worker_id, seeds=seeds))
    return {**identity, "requested_seeds": seeds, "checkpoint_sha256": metadata["checkpoint_sha256"],
            "errors": errors}


def _collect_worker(output, worker_output, seeds, metadata, report):
    """Validate each artifact before moving it into the parent-owned output."""
    if (report["requested_seeds"] != seeds or report["checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or len(set(report["requested_seeds"])) != len(seeds)):
        raise ValueError("Worker seed/checkpoint identity does not match its assignment")
    results, errors = [], list(report["errors"])
    expected_files = {f"seed-{seed}.json" for seed in seeds}
    if any(path.name not in expected_files for path in worker_output.glob("seed-*.json")):
        raise ValueError("Worker produced an unassigned seed artifact")
    for seed in seeds:
        source = worker_output / f"seed-{seed}.json"
        if not source.exists():
            continue
        try:
            saved = read_json(source)
            if (type(saved.get("seed")) is not int or saved["seed"] != seed
                    or any(saved.get(key) != metadata[key] for key in (
                        "checkpoint_sha256", "checkpoint_epoch", "execution_config", "render_protocol"))
                    or len(saved["trajectory"]) != saved["control_steps"]):
                raise ValueError(f"Worker artifact identity/trajectory differs for seed {seed}")
            target = output / source.name
            if target.exists():
                raise ValueError(f"Duplicate output for seed {seed}")
            if saved["video"] is not None:
                if saved["video"] != f"seed-{seed}.mp4":
                    raise ValueError("Unexpected worker video path")
                video = worker_output / saved["video"]
                if not video.is_file() or (output / video.name).exists():
                    raise ValueError(f"Missing or duplicate video for seed {seed}")
                video.rename(output / video.name)
            source.rename(target)
            results.append({key: value for key, value in saved.items() if key != "trajectory"})
        except Exception as exc:
            errors.append(_error("artifact", exc, seed=seed, worker_id=report.get("worker_id")))
    return results, errors


def _summary(metadata, seeds, results, errors, *, settings, workers, active_workers, finished):
    import config
    by_seed = {result["seed"]: result for result in results}
    ordered = [by_seed[seed] for seed in seeds if seed in by_seed]
    completed = [result["seed"] for result in ordered if result["stop_reason"] != "window_closed"]
    missing = [seed for seed in seeds if seed not in by_seed]
    interrupted = [result["seed"] for result in ordered if result["stop_reason"] == "window_closed"]
    complete = finished and not errors and not missing and not interrupted and len(results) == len(seeds)
    successes = sum(bool(result["success"]) for result in ordered)
    return {**metadata, "requested_seeds": seeds,
            "max_steps": settings["max_steps"],
            "grasp_trigger": "closed_finger_target_with_backend_confirmed_stable_contacts",
            "delivery_stable_seconds": .5, "task_truth_is_policy_input": False,
            "video_camera": config.CAM_OVERHEAD if settings["video"] else None,
            "video_fps": config.CTRL_HZ if settings["video"] else None,
            "arm_target_semantics": "direct_per_tick_target", "workers": workers,
            "active_workers": active_workers, "process_start_method": "spawn" if workers > 1 else None,
            "complete": bool(complete), "errors": errors, "missing_seeds": missing,
            "interrupted_seeds": interrupted, "completed_seeds": completed,
            "requested_episode_count": len(seeds), "completed_episode_count": len(completed),
            "success_count": successes, "success_rate": successes / len(seeds) if complete else None,
            "episodes": ordered}


def evaluate(checkpoint, seeds, output, *, device="cpu", max_steps=600, gui=False,
             video=False, workers=1):
    """Evaluate in requested seed order; failures save partial artifacts then raise.

    Parallel workers use spawn and independent simulator/renderer/policy state.
    Only the parent writes summary.json; incomplete runs never publish a rate.
    """
    from sim.backend import FRONT_RENDER_PROTOCOL
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Supply distinct nonnegative evaluation seeds")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("Invalid evaluation settings")
    if type(workers) is not int or workers < 1 or (gui and workers != 1):
        raise ValueError("Workers must be positive; GUI requires workers=1")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    checkpoint = str(Path(checkpoint).resolve())
    metadata = {**checkpoint_metadata(checkpoint), "render_protocol": FRONT_RENDER_PROTOCOL}
    metadata["render_warmup_frames"] = 1
    settings = {"device": device, "max_steps": max_steps, "gui": gui, "video": video}
    output.mkdir(parents=True)
    results, errors = [], []
    active_workers = min(workers, len(seeds))
    def write_summary(finished=False):
        summary = _summary(metadata, seeds, results, errors, settings=settings,
                           workers=workers, active_workers=active_workers, finished=finished)
        atomic_json(output / "summary.json", summary)
        return summary
    def receive(result):
        if result["seed"] not in seeds or any(row["seed"] == result["seed"] for row in results):
            raise ValueError("Duplicate or unassigned episode result")
        results.append(result)
        write_summary()
    write_summary()
    try:
        if workers == 1:
            _evaluate_serial(checkpoint, seeds, output, metadata=metadata, on_result=receive, **settings)
        else:
            worker_root = output / "_workers"
            worker_root.mkdir()
            assignments = [(index, seeds[index::active_workers], worker_root / f"worker-{index:03d}")
                           for index in range(active_workers)]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=active_workers, mp_context=context) as pool:
                futures = {pool.submit(_run_worker, checkpoint, shard, str(path), metadata, settings, index):
                           (index, shard, path) for index, shard, path in assignments}
                for future in as_completed(futures):
                    index, shard, path = futures[future]
                    try:
                        report = future.result()
                    except Exception as exc:
                        report = {"worker_id": index, "requested_seeds": shard,
                                  "checkpoint_sha256": metadata["checkpoint_sha256"],
                                  "errors": [_error("process", exc, worker_id=index, seeds=shard)]}
                    try:
                        rows, failures = _collect_worker(output, path, shard, metadata, report)
                    except Exception as exc:
                        errors.append(_error("collection", exc, worker_id=index, seeds=shard))
                        write_summary()
                        continue
                    errors.extend(failures)
                    for result in rows:
                        receive(result)
        if file_digest(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during evaluation")
    except Exception as exc:
        errors.append(_error("evaluation", exc))
    summary = write_summary(finished=True)
    if summary["missing_seeds"]:
        errors.append({"scope": "missing_episodes", "type": "IncompleteEvaluation",
                       "message": f"No validated result for seeds {summary['missing_seeds']}",
                       "seeds": summary["missing_seeds"]})
        summary = write_summary(finished=True)
    if summary["errors"] or summary["missing_seeds"]:
        raise RolloutEvaluationError(output, summary)
    return summary["episodes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--workers", type=int, default=10, help="Spawn workers; GUI requires 1")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record each rollout as an overhead MP4 at control FPS")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.seeds, args.output, device=args.device, max_steps=args.max_steps,
             gui=args.gui, video=args.video,
             workers=args.workers)


if __name__ == "__main__":
    main()
