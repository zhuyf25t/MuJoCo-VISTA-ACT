"""Explicit-seed MuJoCo closed-loop evaluation; never calls VISTA or exports training data."""
import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import time

import numpy as np

from .common import atomic_json, file_digest
from .policy import ACTPolicy


class SimActionAdapter:
    """Five physical-unit targets, applied for one control tick.

    A close transition starts one bounded grasp attempt. Holding a closed target
    does not restart it. The five-dimensional dataset cannot encode a repeated
    high-level close command with an unchanged target; see README before judging
    rollout differences from the VISTA expert.
    """
    def __init__(self, robot, dt=0.1, close_seconds=1.2):
        if not all(np.isfinite(x) and x > 0 for x in (dt, close_seconds)):
            raise ValueError("Adapter timing/rate limits must be positive")
        self.robot, self.dt = robot, dt
        current = np.asarray(robot.applied_action, dtype=float)
        if current.shape != (5,) or not np.isfinite(current).all():
            raise ValueError("Need the known current actuator targets")
        self.gripper = float(current[2] >= .5)
        self.arm = current[:2].copy()
        self.close_ticks = math.ceil(close_seconds / dt)
        self.remaining = 0

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
            self.remaining = self.close_ticks if target == 0 else 0
            self.gripper = target
        self.robot.command_arm(arm)
        self.robot.command_drive(float(action[3]), float(action[4]))
        self.arm = arm
        return np.asarray(self.robot.applied_action).copy()

    def after_tick(self):
        if self.remaining:
            self.remaining -= 1
            if self.remaining == 0:
                self.robot.finish_grasp_attempt()

    def stop(self):
        self.robot.command_drive(0, 0)
        self.robot.finish_grasp_attempt()


def evaluate(checkpoint, seeds, output, *, device="cpu", max_steps=600, gui=False,
             ensemble_coefficient=None, label="validation", video=False):
    import config
    from sim.env import MobileManipEnv
    from viz import CameraRig, GuiViewer
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Supply distinct nonnegative evaluation seeds")
    if max_steps < 1 or label not in ("validation", "test"):
        raise ValueError("Invalid evaluation settings")
    if config.CTRL_HZ != 10 or config.CAM_FRONT_RES != (320, 240):
        raise ValueError("Simulation observation/control settings differ from the trained schema")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    policy = ACTPolicy(checkpoint, device, ensemble_coefficient)
    output.mkdir(parents=True)
    results = []
    with MobileManipEnv() as env:
        with ExitStack() as resources:
            viewer = resources.enter_context(GuiViewer(env.model, env.data)) if gui else None
            camera = resources.enter_context(CameraRig(env.model, [config.CAM_OVERHEAD])) if video else None
            for seed in seeds:
                env.reset(seed=seed)
                policy.reset()
                adapter = SimActionAdapter(env.hal, config.CTRL_DT, config.GRIPPER_CLOSE_SECONDS)
                writer = None
                if video:
                    import imageio.v2 as imageio
                    writer = resources.enter_context(imageio.get_writer(
                        str(output / f"seed-{seed}.mp4"), fps=config.CTRL_HZ,
                        codec="libx264", quality=8, macro_block_size=None))
                    writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                latencies, trajectory, stopped = [], [], "horizon"
                start = env.hal.now()
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
                        adapter.after_tick()
                        if writer:
                            writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                        if viewer:
                            viewer.set_status(f"ACT {label} | seed {seed} | step {step + 1}")
                            viewer.sync()
                        if env.success():  # Evaluation-only ground truth, never a policy input.
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
                        if writer:
                            writer.append_data(camera.render(env.data)[config.CAM_OVERHEAD])
                        if viewer:
                            viewer.sync()
                if writer:
                    writer.close()
                result = {"seed": seed, "success": stopped != "window_closed" and bool(env.success()),
                          "stop_reason": stopped, "control_steps": len(trajectory),
                          "sim_seconds": env.hal.now() - start,
                          "mean_policy_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
                          "p95_policy_latency_ms": float(np.quantile(latencies, .95) * 1000) if latencies else None,
                          "video": f"seed-{seed}.mp4" if video else None}
                atomic_json(output / f"seed-{seed}.json", {**result, "trajectory": trajectory})
                results.append(result)
                atomic_json(output / "summary.json", {
                    "label": label, "checkpoint_sha256": file_digest(checkpoint), "requested_seeds": seeds,
                    "max_steps": max_steps, "ensemble_coefficient": ensemble_coefficient,
                    "grasp_attempt_seconds": config.GRIPPER_CLOSE_SECONDS, "grasp_trigger": "open_to_closed_transition",
                    "arm_target_semantics": "direct_per_tick_target",
                    "success_rate": sum(r["success"] for r in results) / len(results), "episodes": results})
                print(result, flush=True)
                if stopped == "window_closed":
                    break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record each rollout as an overhead MP4 at control FPS")
    parser.add_argument("--ensemble-coefficient", type=float)
    parser.add_argument("--label", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.seeds, args.output, device=args.device, max_steps=args.max_steps,
             gui=args.gui, ensemble_coefficient=args.ensemble_coefficient, label=args.label, video=args.video)


if __name__ == "__main__":
    main()
