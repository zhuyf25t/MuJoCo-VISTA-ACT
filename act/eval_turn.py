"""Score a saved ACT's turn decisions on its original expert observations.

No training, simulator or API call. Default: deterministic stratified spot check;
--per-category 0 checks every eligible frame. Reports train and val separately.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .common import atomic_json, digest, file_digest, h5_path
from .dataset import EpisodeDataset
from .prepare import choose_splits, inspect_episode
from .checkpoint import load_checkpoint, model_from_checkpoint, normalized_prediction


LABELS = {
    "start_left": "启动左转", "start_right": "启动右转",
    "continue_left": "继续左转", "continue_right": "继续右转",
    "stop": "停止转向", "nonturn": "不应转向",
}


def select_frames(states, actions, per_category):
    """Disjoint categories, defined using physical units before any normalization."""
    if type(per_category) is not int or per_category < 0:
        raise ValueError("per_category must be a nonnegative integer")
    current, wanted = states[:, 4], actions[:, 4]
    stationary, rotating = abs(current) < .05, abs(current) > .2
    masks = {
        "start_left": stationary & (wanted > .2),
        "start_right": stationary & (wanted < -.2),
        "continue_left": rotating & (wanted > .2),
        "continue_right": rotating & (wanted < -.2),
        "stop": rotating & (abs(wanted) < .05),
        "nonturn": stationary & (abs(wanted) < .05),
    }
    selected = []
    for group, mask in masks.items():
        ids = np.flatnonzero(mask)
        if per_category and len(ids) > per_category:
            ids = ids[np.linspace(0, len(ids) - 1, per_category, dtype=int)]
        selected.extend((int(t), group) for t in ids)
    return selected


def correct_decision(group, predicted):
    if group.endswith("left"):
        return bool(predicted > .2)
    if group.endswith("right"):
        return bool(predicted < -.2)
    return bool(abs(predicted) < .05)


def summarize(rows):
    result = {}
    for split in ("train", "val"):
        result[split] = {}
        for group in LABELS:
            chosen = [r for r in rows if r["split"] == split and r["group"] == group]
            n = len(chosen)
            result[split][group] = {
                "count": n,
                "correct_rate": sum(r["correct"] for r in chosen) / n if n else None,
                "mean_abs_prediction": float(np.mean([abs(r["predicted_w"]) for r in chosen])) if n else None,
                "mae": float(np.mean([abs(r["predicted_w"] - r["target_w"]) for r in chosen])) if n else None,
            }
    return result


def evaluate(checkpoint_path, dataset_root, output, *, splits_path=None,
             split_seed=42, val_fraction=.2, per_category=1, batch_size=8, device="cpu"):
    output, root = Path(output), Path(dataset_root)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if per_category < 0 or batch_size < 1:
        raise ValueError("Invalid sampling or batch size")
    ck = load_checkpoint(checkpoint_path)
    ck.pop("optimizer", None)
    model = model_from_checkpoint(ck, device)
    cfg = model.cfg
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != ck["schema"]:
        raise ValueError("Dataset schema differs from checkpoint")
    hashes = ck["prepared"]["h5_sha256"]
    entries = [e for e in manifest["episodes"] if e["episode_id"] in hashes]
    if len(entries) != len(hashes) or {e["episode_id"] for e in entries} != set(hashes):
        raise ValueError("Original checkpoint episodes missing or duplicated")
    explicit = json.loads(Path(splits_path).read_text(encoding="utf-8")) if splits_path else None
    splits = choose_splits(entries, val_fraction, split_seed, explicit)
    if digest(splits) != ck["prepared"]["splits_sha256"]:
        raise ValueError("Split differs from training; supply the original --splits file")
    selected, rows, offset = [], [], 0
    print("核对 checkpoint 原始数据、哈希及训练/验证划分…", flush=True)
    for e in entries:
        if file_digest(h5_path(root, e["h5"])) != hashes[e["episode_id"]]:
            raise ValueError(f"Training data changed: {e['episode_id']}")
        state, action = inspect_episode(root, e, ck["schema"], check_pixels=False)
        for t, group in select_frames(state, action, per_category):
            selected.append(offset + t)
            rows.append({"seed": e["seed"], "episode_id": e["episode_id"], "frame": t,
                         "split": "train" if e["episode_id"] in splits["train"] else "val",
                         "group": group, "state_w": float(state[t, 4]), "target_w": float(action[t, 4])})
        offset += e["num_steps"]
    if not rows:
        raise ValueError("No eligible turn diagnostic frames")
    stats, epoch = ck["stats"], ck["epoch"]
    del ck
    dataset = EpisodeDataset(root, entries, stats, cfg)
    loader = DataLoader(Subset(dataset, selected), batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"评估 {len(rows)} 帧，只运行推理，不更新权重。", flush=True)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            model_output = model(batch["images"].to(device), batch["state"].to(device))
            prediction = normalized_prediction(model_output, stats)[:, 0, 4]
            physical = prediction.cpu().numpy() * stats["action_std"][4] + stats["action_mean"][4]
            if not np.isfinite(physical).all():
                raise FloatingPointError("Nonfinite turn prediction")
            for value in physical:
                row = rows[offset]
                row["predicted_w"] = float(value)
                row["correct"] = correct_decision(row["group"], float(value))
                offset += 1
            if offset % (batch_size * 8) == 0 or offset == len(rows):
                print(f"  {offset}/{len(rows)}", flush=True)
    report = {"checkpoint_sha256": file_digest(checkpoint_path), "checkpoint_epoch_one_based": epoch + 1,
              "splits_sha256": digest(splits), "original_data_hashes_verified": True,
              "sampling": "all_eligible_frames" if per_category == 0 else "stratified_spot_check",
              "per_category_per_episode": per_category,
              "note": "First predicted action, deterministic learned token, without temporal ensembling; not closed-loop success.",
              "thresholds_rad_s": {"turn": .2, "stop": .05}, "scores": summarize(rows)}
    output.mkdir(parents=True)
    atomic_json(output / "report.json", report)
    with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split, title in (("train", "训练集"), ("val", "验证集")):
        print(f"\n{title}（{'全量符合条件帧' if per_category == 0 else '分层抽查'}）", flush=True)
        for group, score in report["scores"][split].items():
            rate = "无样本" if score["correct_rate"] is None else f"{score['correct_rate']:.1%}"
            print(f"  {LABELS[group]}：正确率 {rate}，样本数 {score['count']}", flush=True)
    print(f"\n结果：{output / 'report.json'}；逐帧明细：{output / 'frames.csv'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="data/act_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", help="Original splits.json; otherwise reconstruct and verify its hash")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--per-category", type=int, default=1, help="Frames per category per episode; 0 for all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    evaluate(args.checkpoint, args.dataset_root, args.output, splits_path=args.splits,
             split_seed=args.split_seed, val_fraction=args.val_fraction, per_category=args.per_category,
             batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
