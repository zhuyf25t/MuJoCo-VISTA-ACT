"""Train ACT on an explicitly prepared experiment; resume at epoch boundaries."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from recorder import FileLock
from .common import atomic_json, load_prepared
from .config import config_from_dict, load_config
from .dataset import EpisodeDataset
from .model import ACT


def masked_l1(prediction, target, is_pad):
    valid = (~is_pad).unsqueeze(-1)
    count = valid.sum() * prediction.shape[-1]
    if count.item() == 0:
        raise ValueError("A batch must contain valid actions")
    # torch.where also prevents arbitrary padded values from entering the loss.
    return torch.where(valid, (prediction - target).abs(), 0).sum() / count


def act_loss(prediction, target, is_pad, mu, logvar, beta):
    l1 = masked_l1(prediction, target, is_pad)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
    return l1 + beta * kl, l1, kl


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model, loader, device, stats):
    model.eval()
    error = torch.zeros(model.cfg.action_dim, device=device, dtype=torch.float64)
    first_error = torch.zeros_like(error)
    count = first_count = correct_gripper = 0
    std = torch.as_tensor(stats["action_std"], device=device)
    mean = torch.as_tensor(stats["action_mean"], device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        prediction, _, _ = model(batch["images"], batch["state"])  # Prior mean, no expert actions.
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite validation prediction")
        valid = ~batch["is_pad"]
        difference = (prediction - batch["actions"]).abs()
        error += torch.where(valid.unsqueeze(-1), difference, 0).double().sum((0, 1))
        first_error += difference[:, 0].double().sum(0)
        count += int(valid.sum())
        first_count += prediction.shape[0]
        pred_g = prediction[:, :, 2] * std[2] + mean[2]
        target_g = batch["actions"][:, :, 2] * std[2] + mean[2]
        correct_gripper += int((((pred_g >= .5) == (target_g >= .5)) & valid).sum())
    if count == 0:
        raise ValueError("Empty validation loader")
    return {"val_l1_z0": float((error / count).mean()),
            "val_mae_physical": (error / count * std).cpu().tolist(),
            "val_first_mae_physical": (first_error / first_count * std).cpu().tolist(),
            "val_gripper_accuracy": correct_gripper / count}


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0],
            torch.tensor(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def provenance():
    try:
        repo = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    import torchvision
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "git_revision": revision, "git_dirty": dirty}


def train(root, experiment, *, epochs, cfg=None, device="auto", resume=False, backbone_weights=None):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive total epoch count")
    experiment = Path(experiment)
    if not (experiment / "prepared.json").is_file():
        raise ValueError("Run act.prepare with an explicit split first")
    with FileLock(experiment / ".training.lock", timeout=0.2):
        return _train(root, experiment, epochs, cfg, device, resume, backbone_weights)


def _train(root, experiment, epochs, cfg, device, resume, backbone_weights):
    last = experiment / "last.pt"
    checkpoint = load_checkpoint(last) if resume else None
    if resume:
        if cfg is not None and cfg.to_dict() != checkpoint["config"]:
            raise ValueError("Resume configuration differs from checkpoint; start a new experiment")
        if backbone_weights:
            raise ValueError("Resume restores all weights; do not pass ImageNet weights")
        cfg = config_from_dict(checkpoint["config"])
    else:
        if any((experiment / name).exists() for name in ("last.pt", "best.pt", "config.yaml", "metrics.jsonl")):
            raise FileExistsError("Experiment already started; use --resume or a new experiment directory")
        cfg = cfg or load_config()
    manifest, splits, stats, prepared = load_prepared(experiment, root, cfg.model)
    if resume and checkpoint["prepared"] != prepared:
        raise ValueError("Resume dataset differs from checkpoint")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    datasets = {part: EpisodeDataset(root, [e for e in manifest["episodes"] if e["episode_id"] in splits[part]], stats, cfg.model)
                for part in ("train", "val")}
    loaders = {part: DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=part == "train",
                               num_workers=cfg.training.num_workers, drop_last=False,
                               pin_memory=device.type == "cuda", generator=generator)
               for part, dataset in datasets.items()}
    model = ACT(cfg.model, initialize_backbone=not resume, backbone_weights=backbone_weights).to(device)
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") else other).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": cfg.training.backbone_learning_rate},
                                   {"params": other, "lr": cfg.training.learning_rate}], weight_decay=cfg.training.weight_decay)
    amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, step, best, best_epoch, early_best, stale = 0, 0, math.inf, -1, math.inf, 0
    if resume:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best, best_epoch = checkpoint["best_val_l1"], checkpoint["best_epoch"]
        early_best, stale = checkpoint["early_best"], checkpoint["stale_epochs"]
        restore_rng(checkpoint["rng"], generator)
        # Recover best.pt if publication was interrupted after last.pt committed.
        if best_epoch == checkpoint["epoch"]:
            save_checkpoint(experiment / "best.pt", checkpoint)
        if start_epoch >= epochs:
            raise ValueError(f"Already completed {start_epoch} epochs; --epochs is a total, not additional count")
        if cfg.training.patience and stale >= cfg.training.patience:
            raise ValueError("This run has already early-stopped; start a new experiment to change its policy")
        log_path = experiment / "metrics.jsonl"
        rows = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A crash may leave a partial final log line.
                if row["epoch"] < start_epoch:
                    rows.append(row)
        if not rows or rows[-1]["epoch"] != checkpoint["epoch"]:
            rows.append(checkpoint["metrics"])
        log_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:
        (experiment / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    versions = provenance()
    atomic_json(experiment / "runtime.json", {**versions, "device": str(device), "amp_enabled": amp,
                "target_epochs": epochs, "resumed": resume, "sample_counts": prepared["sample_counts"],
                "updates_per_epoch": len(loaders["train"])})
    print(f"device={device}; train samples={len(datasets['train'])}; val samples={len(datasets['val'])}; "
          f"updates/epoch={len(loaders['train'])}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(start_epoch, epochs):
        started = time.monotonic()
        model.train()
        action_total = action_count = kl_total = sample_count = 0.0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                prediction, mu, logvar = model(batch["images"], batch["state"], batch["actions"], batch["is_pad"])
                loss, l1, kl = act_loss(prediction.float(), batch["actions"], batch["is_pad"], mu.float(), logvar.float(), cfg.training.kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            valid_elements = int((~batch["is_pad"]).sum()) * cfg.model.action_dim
            n = batch["state"].shape[0]
            action_total += float(l1.detach()) * valid_elements
            action_count += valid_elements
            kl_total += float(kl.detach()) * n
            sample_count += n
        metrics = {"epoch": epoch, "global_step": step, "train_l1": action_total / action_count,
                   "train_kl": kl_total / sample_count, **validate(model, loaders["val"], device, stats),
                   "elapsed_seconds": time.monotonic() - started}
        score = metrics["val_l1_z0"]
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        if score < early_best - cfg.training.min_delta:
            early_best, stale = score, 0
        else:
            stale += 1
        checkpoint = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "config": cfg.to_dict(), "schema": manifest["schema"],
                      "stats": stats, "prepared": prepared, "epoch": epoch, "global_step": step,
                      "best_val_l1": best, "best_epoch": best_epoch, "early_best": early_best,
                      "stale_epochs": stale, "rng": rng_state(generator), "metrics": metrics, "versions": versions}
        save_checkpoint(last, checkpoint)
        if improved:
            save_checkpoint(experiment / "best.pt", checkpoint)
        with (experiment / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(metrics), flush=True)
        if cfg.training.patience and stale >= cfg.training.patience:
            print(f"Early stopping after {stale} validation checks without sufficient improvement.", flush=True)
            break
    return experiment / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs, including previously completed epochs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Restore last.pt; no ImageNet download")
    parser.add_argument("--backbone-weights", help="Optional local official torchvision ResNet-18 state_dict")
    args = parser.parse_args()
    train(args.dataset_root, args.experiment, epochs=args.epochs, cfg=load_config(args.config) if args.config else None,
          device=args.device, resume=args.resume, backbone_weights=args.backbone_weights)


if __name__ == "__main__":
    main()
