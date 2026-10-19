"""Audit whether recorded expert targets work through the ACT executor.

This is an oracle replay, not a learned policy evaluation. recorded-events reads
expert tool boundaries for diagnosis only; a deployed policy has no such input.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import atomic_json, file_digest, h5_path
from .rollout import SimActionAdapter


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gripper_events(rows, frame_count):
    starts, ends = {}, set()
    for row in rows:
        for action in row.get("actions", []):
            name = action.get("action")
            if not action.get("action_applied") or name not in ("open_gripper", "close_gripper"):
                continue
            start, end = action["frame_range"]
            if not (0 <= start < end <= frame_count) or start in starts:
                raise ValueError("Invalid or duplicate recorded gripper interval")
            starts[start] = name
            ends.add(end)
    return starts, ends


def replay(raw_root, vista_root, seeds, output, modes=("direct",), dataset_root=None):
    import config
    from sim.env import MobileManipEnv

    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Supply distinct nonnegative seeds")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"direct", "recorded-events"}:
        raise ValueError("Unknown or duplicate replay mode")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new replay report path")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = None
    if dataset_root is not None:
        manifest = json.loads((Path(dataset_root) / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for seed in seeds:
        episode = Path(raw_root) / f"ep_{seed:04d}"
        meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
        if meta["seed"] != seed or meta["schema"]["observation_timing"] != "before_action":
            raise ValueError("Raw episode seed or timing mismatch")
        if meta["schema"]["fps"] != config.CTRL_HZ:
            raise ValueError("Control frequency mismatch")
        trajectory_path = episode / "trajectory.jsonl"
        frames = read_jsonl(trajectory_path)
        actions = np.asarray([f["action"] for f in frames], dtype=float)
        states = np.asarray([f["state"] for f in frames], dtype=float)
        if not len(frames) or actions.shape != (len(frames), 5) or states.shape != actions.shape:
            raise ValueError("Expected nonempty five-dimensional trajectory")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("Nonfinite trajectory")
        h5_hash = None
        if manifest is not None:
            import h5py
            entries = [e for e in manifest["episodes"] if e["seed"] == seed and e["status"] == "ready"]
            if len(entries) != 1 or entries[0]["source_uid"] != meta["source_uid"]:
                raise ValueError("HDF5 source does not uniquely match the raw episode")
            path = h5_path(dataset_root, entries[0]["h5"])
            with h5py.File(path, "r") as f:
                h5_actions, h5_states = f["action"][:], f["observations/state"][:]
                if f.attrs["source_uid"] != meta["source_uid"]:
                    raise ValueError("HDF5 source uid mismatch")
            if (not np.array_equal(h5_actions, actions.astype(h5_actions.dtype))
                    or not np.array_equal(h5_states, states.astype(h5_states.dtype))):
                raise ValueError("HDF5 frames differ from the raw episode")
            actions, states = h5_actions, h5_states
            h5_hash = file_digest(path)
        starts, ends, events_hash = {}, set(), None
        if "recorded-events" in modes:
            events_path = Path(vista_root) / f"ep_{seed:04d}" / "actions.jsonl"
            starts, ends = gripper_events(read_jsonl(events_path), len(frames))
            events_hash = file_digest(events_path)
        for mode in modes:
            with MobileManipEnv() as env:
                env.reset(seed=seed)
                adapter = SimActionAdapter(env.hal, config.CTRL_DT, config.GRIPPER_CLOSE_SECONDS)
                state_error, action_error = np.zeros(5), np.zeros(5)
                for t, action in enumerate(actions):
                    velocity = env.hal.base_velocity
                    measured = np.array([*env.hal.arm_q, env.hal.finger_opening, *velocity])
                    state_error = np.maximum(state_error, abs(measured - states[t]))
                    applied = adapter.apply(action)
                    action_error = np.maximum(action_error, abs(applied - action))
                    if mode == "recorded-events" and t in starts:
                        closing = starts[t] == "close_gripper"
                        env.hal.command_finger(config.FINGER_CLOSE_CTRL if closing else config.FINGER_OPEN_CTRL)
                        adapter.remaining = adapter.close_ticks if closing else 0
                    env.hal.tick()
                    adapter.after_tick()
                    if mode == "recorded-events" and t + 1 in ends:
                        env.hal.finish_grasp_attempt()
                        adapter.remaining = 0
                success = bool(env.success())
                adapter.stop()
            result = {"seed": seed, "mode": mode, "success": success,
                      "expert_success": bool(meta.get("success")), "control_steps": len(frames),
                      "max_state_error_per_dimension": state_error.tolist(),
                      "max_action_error_per_dimension": action_error.tolist(),
                      "trajectory_sha256": file_digest(trajectory_path),
                      "source": "h5" if manifest is not None else "raw",
                      "h5_sha256": h5_hash,
                      "expert_events_sha256": events_hash if mode == "recorded-events" else None}
            results.append(result)
            atomic_json(output, {"kind": "expert_oracle_replay", "learned_policy": False,
                                 "grasp_attempt_seconds": config.GRIPPER_CLOSE_SECONDS,
                                 "results": results})
            print(f"seed={seed} mode={mode} success={success}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/episodes")
    parser.add_argument("--vista-root", default="output/vista")
    parser.add_argument("--dataset-root", help="Replay the matching actual HDF5 labels, including their precision")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("direct", "recorded-events"), default=["direct"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(args.raw_root, args.vista_root, args.seeds, args.output, args.modes, args.dataset_root)


if __name__ == "__main__":
    main()
