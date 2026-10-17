"""Synthetic, isolated ACT integration tests. Never split or train on live data."""
import json
import shutil

import h5py
import numpy as np
import pytest
import torch

from act.common import h5_path, load_prepared
from act.config import Config, ModelConfig, TrainConfig
from act.dataset import EpisodeDataset
from act.model import ACT
from act.policy import ACTPolicy, TemporalEnsembler
from act.prepare import choose_splits, prepare
from act.rollout import SimActionAdapter
from act.train import act_loss, load_checkpoint, masked_l1, save_checkpoint, train
from recorder import dataset_schema


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    (root / "episodes").mkdir(parents=True)
    entries = []
    schema = dataset_schema()
    for ep, n in enumerate((3, 2)):
        entry = {"episode_id": f"episode-{ep:04d}", "source_uid": str(ep), "seed": ep,
                 "source_dir": f"F:\\old-computer\\ep_{ep}", "status": "ready", "success": True,
                 "selected_for_dataset": True, "num_steps": n, "h5": f"episodes/episode-{ep:04d}.h5"}
        with h5py.File(root / entry["h5"], "w") as f:
            f.attrs.update(episode_id=entry["episode_id"], source_uid=str(ep), num_steps=n,
                           fps=10, schema_version=schema["version"], schema_json=json.dumps(schema))
            states = np.zeros((n, 5), np.float32)
            states[:, 0] = np.arange(n) + ep * 100  # Detect accidental val-stat leakage.
            states[:, 2] = .4
            actions = states.copy()
            actions[:, 2] = np.arange(n) % 2
            f.create_dataset("observations/state", data=states)
            f.create_dataset("action", data=actions)
            pixels = np.zeros((n, 240, 320, 3), np.uint8)
            pixels[..., 0] = 255
            f.create_dataset("observations/images/front", data=pixels, compression="gzip")
        entries.append(entry)
    (root / "manifest.json").write_text(json.dumps({"schema": schema, "episodes": entries}), encoding="utf-8")
    return root


def tiny_config():
    return Config(ModelConfig(hidden_dim=16, nheads=4, latent_dim=4, dim_feedforward=32,
                              encoder_layers=1, decoder_layers=1, posterior_layers=1,
                              dropout=0, pretrained_backbone=False), TrainConfig(batch_size=8))


def prepared(root, out):
    return prepare(root, out, explicit={"train": ["episode-0000"], "val": ["episode-0001"]})


def test_prepare_portability_stats_and_immutability(dataset_root, tmp_path):
    original = (dataset_root / "manifest.json").read_bytes()
    experiment = tmp_path / "exp"
    result = prepared(dataset_root, experiment)
    assert result["sample_counts"] == {"train": 3, "val": 2}
    moved = tmp_path / "autodl-data"
    shutil.copytree(dataset_root, moved)
    manifest, splits, stats, _ = load_prepared(experiment, moved, ModelConfig())
    assert stats["state_mean"][0] == 1.0
    assert (dataset_root / "manifest.json").read_bytes() == original
    assert manifest["episodes"][0]["source_dir"].startswith("F:")
    with pytest.raises(FileExistsError):
        prepared(dataset_root, experiment)
    with h5py.File(moved / manifest["episodes"][0]["h5"], "r+") as f:
        f["action"][0, 0] = 9
    with pytest.raises(ValueError, match="changed"):
        load_prepared(experiment, moved)


def test_scene_leakage_paths_and_explicit_split(dataset_root, tmp_path):
    with pytest.raises(ValueError, match="Specify"):
        prepare(dataset_root, tmp_path / "no-split")
    entries = [{"episode_id": "a", "seed": 3}, {"episode_id": "b", "seed": 3}]
    with pytest.raises(ValueError, match="Same scene"):
        choose_splits(entries, explicit={"train": ["a"], "val": ["b"]})
    for path in ("/tmp/a.h5", "F:/data/a.h5", "..\\a.h5", "../a.h5", "C:a.h5"):
        with pytest.raises(ValueError):
            h5_path(dataset_root, path)


def test_sample_alignment_and_padding(dataset_root, tmp_path):
    experiment = tmp_path / "exp"
    prepared(dataset_root, experiment)
    manifest, _, stats, _ = load_prepared(experiment, dataset_root)
    data = EpisodeDataset(dataset_root, manifest["episodes"], stats, ModelConfig())
    assert len(data) == 5
    first, last = data[0], data[2]
    assert first["images"].shape == (1, 3, 240, 320)
    assert torch.all(first["images"][0, 0] == 1)
    assert torch.all(first["images"][0, 1:] == 0)
    assert last["actions"].shape == (32, 5)
    assert last["is_pad"].tolist() == [False] + [True] * 31
    physical = first["actions"][:3].numpy() * stats["action_std"] + stats["action_mean"]
    np.testing.assert_allclose(physical[:, 0], [0, 1, 2], atol=1e-6)
    assert torch.all(last["actions"][1:] == 0)
    assert data[3]["state"][0] > 100


def test_network_backward_mask_and_prior():
    cfg = tiny_config().model
    model = ACT(cfg)
    pixels = torch.rand(2, 1, 3, 240, 320)
    state = torch.zeros(2, 5)
    actions = torch.randn(2, 32, 5)
    pad = torch.arange(32).expand(2, -1) >= 3
    model.eval()
    _, mu, logvar = model(pixels, state, actions, pad)
    changed = actions.clone()
    changed[pad] = 10000
    _, mu2, logvar2 = model(pixels, state, changed, pad)
    torch.testing.assert_close(mu, mu2)
    torch.testing.assert_close(logvar, logvar2)
    prior, _, _ = model(pixels, state)
    torch.testing.assert_close(prior, model(pixels, state)[0])
    prediction, mu, logvar = model(pixels, state, actions, pad)
    loss, _, _ = act_loss(prediction, actions, pad, mu, logvar, 10)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_head.weight.grad.abs().sum() > 0
    assert model.backbone[0].weight.grad.abs().sum() > 0
    assert model.posterior_head.weight.grad.abs().sum() > 0
    # One valid action with unit errors stays L1=1 regardless of padding length.
    assert masked_l1(torch.ones(1, 32, 5), torch.zeros(1, 32, 5),
                     torch.arange(32)[None] > 0).item() == 1


def test_train_resume_and_standalone_policy(dataset_root, tmp_path, monkeypatch):
    cfg = tiny_config()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    prepared(dataset_root, full)
    prepared(dataset_root, resumed)
    train(dataset_root, full, epochs=2, cfg=cfg, device="cpu")
    train(dataset_root, resumed, epochs=1, cfg=cfg, device="cpu")
    train(dataset_root, resumed, epochs=2, device="cpu", resume=True)
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert a["global_step"] == b["global_step"] == 2
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1]
    assert len(rows[0]["val_mae_physical"]) == 5
    with pytest.raises(FileExistsError):
        train(dataset_root, resumed, epochs=2, cfg=cfg)
    # Simulate deployment with just a single checkpoint, no source dataset or files.
    checkpoint_path = tmp_path / "standalone.pt"
    deployment = load_checkpoint(resumed / "best.pt")
    deployment["config"]["model"]["pretrained_backbone"] = True
    save_checkpoint(checkpoint_path, deployment)
    def no_download(*args, **kwargs):
        raise AssertionError("Checkpoint inference must not download ImageNet weights")
    from torchvision.models import ResNet18_Weights
    monkeypatch.setattr(ResNet18_Weights, "get_state_dict", no_download)
    policy = ACTPolicy(checkpoint_path)
    action = policy.act(np.zeros((1, 240, 320, 3), np.uint8), np.zeros(5, np.float32))
    assert action.shape == (5,) and np.isfinite(action).all()
    from act.rollout import evaluate
    results = evaluate(checkpoint_path, [987654], tmp_path / "rollout", max_steps=2, video=True)
    assert results[0]["control_steps"] == 2
    assert results[0]["sim_seconds"] == pytest.approx(3.2)
    trace = json.loads((tmp_path / "rollout" / "seed-987654.json").read_text())
    assert len(trace["trajectory"]) == 2
    assert np.isfinite(trace["trajectory"][0]["applied"]).all()
    import imageio.v2 as imageio
    with imageio.get_reader(tmp_path / "rollout" / results[0]["video"]) as reader:
        assert reader.get_meta_data()["fps"] == 10
        assert reader.count_frames() == 33  # Initial state, two control ticks, 30 settle ticks.
        assert reader.get_data(0).shape == (480, 640, 3)


def test_early_stopping_preserves_best_checkpoint(dataset_root, tmp_path, monkeypatch):
    import importlib
    training = importlib.import_module("act.train")
    cfg = tiny_config()
    cfg.training.patience = 1
    experiment = tmp_path / "early-stop"
    prepared(dataset_root, experiment)
    scores = iter([.2, .3])
    monkeypatch.setattr(training, "validate", lambda *args: {"val_l1_z0": next(scores)})
    train(dataset_root, experiment, epochs=9, cfg=cfg, device="cpu")
    assert load_checkpoint(experiment / "best.pt")["epoch"] == 0
    assert load_checkpoint(experiment / "last.pt")["epoch"] == 1
    with pytest.raises(ValueError, match="early-stopped"):
        train(dataset_root, experiment, epochs=10, resume=True, device="cpu")


def test_temporal_ensemble_keeps_valid_zero_actions():
    ensemble = TemporalEnsembler(0)
    np.testing.assert_array_equal(ensemble.select(0, np.array([[9.], [0.]])), [9])
    np.testing.assert_array_equal(ensemble.select(1, np.array([[2.], [4.]])), [1])
    np.testing.assert_array_equal(ensemble.select(2, np.array([[8.], [0.]])), [6])
    ensemble.reset()
    np.testing.assert_array_equal(ensemble.select(0, np.zeros((2, 5))), np.zeros(5))


def test_adapter_does_not_restart_grasp_every_tick():
    class Robot:
        applied_action = np.array([0., 0., 1., 0., 0.])
        calls = 0
        finished = 0
        def arm_geometry(self):
            return 1, 1, np.array([-2., -2.]), np.array([2., 2.])
        def finger_ctrl_range(self):
            return 0., .05
        def command_finger(self, value):
            self.calls += 1
            self.applied_action[2] = 1 - value / .05
        def command_arm(self, value):
            self.applied_action[:2] = value
        def command_drive(self, v, w):
            self.applied_action[3:] = v, w
        def finish_grasp_attempt(self):
            self.finished += 1
    robot = Robot()
    adapter = SimActionAdapter(robot)
    for _ in range(15):
        adapter.apply([3, 3, 0, 0, 0])
        adapter.after_tick()
    assert robot.calls == 1 and robot.finished == 1
    np.testing.assert_array_equal(robot.applied_action[:2], [2, 2])
    adapter.apply([0, 0, .5, 0, 0])
    assert robot.calls == 1
    adapter.apply([0, 0, 1, 0, 0])
    adapter.apply([0, 0, 0, 0, 0])
    assert robot.calls == 3
