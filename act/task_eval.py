"""Evaluate saved checkpoints on fixed scenes for real-task model selection.

The score is lexicographic, with larger values preferred: final successful
episodes, episodes with a verified delivery, episodes with a confirmed grasp.
Each episode contributes at most one point to each component. Touches, repeated
grasps and validation imitation loss never increase this score.
"""
import argparse
from pathlib import Path

from .common import atomic_json, read_json
from .checkpoint import load_checkpoint, deployment_metadata, DEFAULT_DEPLOYMENT
from .rollout import evaluate, RolloutEvaluationError
from sim.backend import FRONT_RENDER_PROTOCOL


VALIDATION_SEEDS = tuple(range(520, 540))
SCORE_NAMES = ("final_success_episodes", "verified_delivery_episodes", "confirmed_grasp_episodes")
TASK_EXECUTION = DEFAULT_DEPLOYMENT
SCORE_VERSION = 2


def task_score(episodes):
    """Return a comparable, bounded progress tuple (higher is better)."""
    episodes = tuple(episodes)
    return (sum(bool(e.get("final_success", e.get("success", False))) for e in episodes),
            sum(bool(e.get("deliveries", 0)) for e in episodes),
            sum(bool(e.get("ever_confirmed_grasp", False)) for e in episodes))


def evaluate_checkpoint(checkpoint, output, *, seeds=VALIDATION_SEEDS, device="cpu",
                        max_steps=600, video=False, gui=False, workers=10):
    """Save rollout artifacts and task_eval.json; return its JSON-safe summary.

    The frozen selection protocol uses K32/H8 execution, 600 control steps,
    seeds 520..539, ten spawn workers and no temporal ensemble. The checkpoint
    must already declare H8 deployment; this evaluator never rewrites it.
    Loading from disk cannot change a
    trainer's live model or its train/eval mode. ``output`` must be a fresh path.
    Zero-based checkpoint_epoch and one-based checkpoint_epoch_number are both
    recorded in every episode and summary. Videos are optional overhead MP4s.
    """
    seeds = list(seeds)
    saved = load_checkpoint(checkpoint)
    if ("deployment" not in saved or deployment_metadata(saved) != TASK_EXECUTION
            or saved.get("config", {}).get("model", {}).get("chunk_size") != 32):
        raise ValueError("Task checkpoint selection requires saved K32/H8 deployment with ensemble off")
    del saved
    failure = None
    try:
        evaluate(checkpoint, seeds, output, device=device, max_steps=max_steps,
                 video=video, gui=gui, workers=workers)
        summary = read_json(Path(output) / "summary.json")
    except RolloutEvaluationError as exc:
        summary, failure = exc.summary, exc
    episodes = summary["episodes"]
    protocol_valid = (summary.get("execution_config") == TASK_EXECUTION
                      and summary.get("render_protocol") == FRONT_RENDER_PROTOCOL
                      and all(e.get("execution_config") == TASK_EXECUTION
                              and e.get("render_protocol") == FRONT_RENDER_PROTOCOL for e in episodes))
    unique_seeds = [e["seed"] for e in episodes]
    complete = (protocol_valid and summary.get("complete") is True and not summary.get("errors")
                and unique_seeds == seeds and len(set(unique_seeds)) == len(seeds)
                and summary.get("completed_seeds") == seeds)
    if not protocol_valid:
        summary.setdefault("errors", []).append({"scope": "task_protocol", "type": "ValueError",
                                                  "message": "Task evaluation execution/render protocol differs"})
    if not complete and failure is None:
        summary.setdefault("errors", []).append({"scope": "task_completeness", "type": "ValueError",
                                                  "message": "Task evaluation did not complete every requested seed exactly once",
                                                  "requested_seeds": seeds, "observed_seeds": unique_seeds})
    summary.update({"kind": "closed_loop_task_evaluation", "score_version": SCORE_VERSION,
                    "score_names": list(SCORE_NAMES), "score": list(task_score(episodes)) if complete else None,
                    "complete": complete, "workers": workers,
                    "success_rate": summary.get("success_rate") if complete else None,
                    "confirmed_grasp_rate": sum(bool(e["ever_confirmed_grasp"]) for e in episodes) / len(seeds) if complete else None,
                    "verified_delivery_rate": sum(bool(e["deliveries"]) for e in episodes) / len(seeds) if complete else None})
    atomic_json(Path(output) / "task_eval.json", summary)
    if not complete:
        if failure is None:
            failure = RolloutEvaluationError(output, summary)
        failure.task_summary = summary
        raise failure
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(VALIDATION_SEEDS))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    summary = evaluate_checkpoint(args.checkpoint, args.output, seeds=args.seeds, device=args.device,
                                  max_steps=args.max_steps, video=args.video, gui=args.gui, workers=args.workers)
    print({"checkpoint_epoch": summary["checkpoint_epoch"], "score": summary["score"],
           "complete": summary["complete"], "output": str(Path(args.output).resolve())}, flush=True)


if __name__ == "__main__":
    main()
