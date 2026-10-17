"""Explicit preparation only; never edits the live collection manifest."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from .common import atomic_json, digest, file_digest, h5_path, read_json, validate_schema


def inspect_episode(root, entry, schema, check_pixels=True):
    path = h5_path(root, entry["h5"])
    n = entry["num_steps"]
    if type(n) is not int or n < 1:
        raise ValueError("num_steps must be positive")
    with h5py.File(path, "r") as f:
        import json
        if (f.attrs.get("episode_id") != entry["episode_id"] or f.attrs.get("source_uid") != entry["source_uid"]
                or f.attrs.get("num_steps") != n or f.attrs.get("fps") != schema["fps"]
                or f.attrs.get("schema_version") != schema["version"]
                or json.loads(f.attrs["schema_json"]) != schema):
            raise ValueError(f"HDF5 metadata mismatch: {path}")
        for key in ("observations/state", "action"):
            if f[key].shape != (n, 5) or f[key].dtype != np.dtype("float32"):
                raise ValueError(f"Invalid {key}: {path}")
        image = f["observations/images/front"]
        if image.shape != (n, 240, 320, 3) or image.dtype != np.uint8:
            raise ValueError(f"Invalid images: {path}")
        states, actions = f["observations/state"][:], f["action"][:]
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite state/action: {path}")
        if np.any((states[:, 2] < 0) | (states[:, 2] > 1)):
            raise ValueError(f"Invalid measured gripper opening: {path}")
        if not np.all(np.isclose(actions[:, 2], 0) | np.isclose(actions[:, 2], 1)):
            raise ValueError(f"Expected binary gripper targets: {path}")
        if check_pixels:
            for start in range(0, n, 32):
                image[start:start + 32]  # Force decompression, detect corrupt chunks.
    return states, actions


def choose_splits(entries, val_fraction=None, seed=42, explicit=None):
    by_id = {e["episode_id"]: e for e in entries}
    if len(by_id) != len(entries) or any(type(e.get("seed")) is not int for e in entries):
        raise ValueError("Need unique episode ids and explicit integer scene seeds")
    if explicit is not None:
        if set(explicit) != {"train", "val"}:
            raise ValueError("Split file must have train and val lists")
        result = explicit
    else:
        if val_fraction is None or not 0 < val_fraction < 1:
            raise ValueError("Specify --val-fraction or --splits explicitly")
        seeds = sorted({e["seed"] for e in entries})
        if len(seeds) < 2:
            raise ValueError("Need at least two scene seeds")
        np.random.default_rng(seed).shuffle(seeds)
        n = min(len(seeds) - 1, max(1, round(len(seeds) * val_fraction)))
        val_seeds = set(seeds[:n])
        result = {part: [e["episode_id"] for e in entries if (e["seed"] in val_seeds) == (part == "val")]
                  for part in ("train", "val")}
    if not all(isinstance(result[p], list) and result[p] for p in ("train", "val")):
        raise ValueError("Need nonempty train and val lists")
    ids = result["train"] + result["val"]
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError("Duplicate, overlapping or unknown episode ids")
    train_seeds = {by_id[i]["seed"] for i in result["train"]}
    if train_seeds & {by_id[i]["seed"] for i in result["val"]}:
        raise ValueError("Same scene seed cannot occur in both splits")
    return result


def prepare(root, experiment, *, val_fraction=None, seed=42, explicit=None):
    root, experiment = Path(root), Path(experiment)
    manifest = read_json(root / "manifest.json")  # Atomic collector publication gives a coherent snapshot.
    validate_schema(manifest["schema"])
    entries = [e for e in manifest["episodes"] if e.get("status") == "ready"
               and e.get("selected_for_dataset") is True and e.get("success") is True]
    splits = choose_splits(entries, val_fraction, seed, explicit)
    ids = set(splits["train"] + splits["val"])
    entries = [e for e in entries if e["episode_id"] in ids]
    if len({e["source_uid"] for e in entries}) != len(entries):
        raise ValueError("Duplicate source episode")
    train_states, train_actions, hashes = [], [], {}
    for entry in entries:
        states, actions = inspect_episode(root, entry, manifest["schema"])
        hashes[entry["episode_id"]] = file_digest(h5_path(root, entry["h5"]))
        if entry["episode_id"] in splits["train"]:
            train_states.append(states)
            train_actions.append(actions)
    stats = {}
    for name, arrays in (("state", train_states), ("action", train_actions)):
        data = np.concatenate(arrays).astype(np.float64)
        stats[f"{name}_mean"] = data.mean(0).tolist()
        stats[f"{name}_std"] = np.maximum(data.std(0), 1e-2).tolist()
    frozen = {"schema": manifest["schema"], "episodes": entries}
    metadata = {"format_version": 1, "manifest_sha256": digest(frozen), "splits_sha256": digest(splits),
                "stats_sha256": digest(stats), "h5_sha256": hashes, "split_seed": seed,
                "sample_counts": {part: sum(e["num_steps"] for e in entries if e["episode_id"] in splits[part])
                                  for part in ("train", "val")}}
    experiment.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest", frozen), ("splits", splits), ("stats", stats), ("prepared", metadata)):
        atomic_json(experiment / f"{name}.json", value)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--experiment", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--val-fraction", type=float)
    group.add_argument("--splits", help="JSON with explicit train and val episode-id lists")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.dataset_root, args.experiment, val_fraction=args.val_fraction, seed=args.seed,
                     explicit=read_json(args.splits) if args.splits else None)
    print(result["sample_counts"])


if __name__ == "__main__":
    main()
