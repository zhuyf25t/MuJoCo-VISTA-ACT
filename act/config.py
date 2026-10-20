"""Explicit, serializable experiment settings. No data split is chosen here."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import yaml


@dataclass
class ModelConfig:
    state_dim: int = 5
    action_dim: int = 5
    cameras: tuple = ("front",)
    image_hw: tuple = (240, 320)
    chunk_size: int = 32
    hidden_dim: int = 512
    nheads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 7
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pretrained_backbone: bool = True
    # Optional denser visual grid; default keeps existing checkpoint behavior.
    visual_stride: int = 32

    def __post_init__(self):
        self.cameras, self.image_hw = tuple(self.cameras), tuple(self.image_hw)
        for name in ("state_dim", "action_dim", "chunk_size", "hidden_dim",
                     "nheads", "encoder_layers", "decoder_layers", "dim_feedforward"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.nheads or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by nheads and by four")
        if self.state_dim != 5 or self.action_dim != 5:
            raise ValueError("Hybrid ACT requires five-dimensional state and action")
        if not 0 <= self.dropout < 1 or type(self.pretrained_backbone) is not bool:
            raise ValueError("Invalid dropout / pretrained_backbone")
        if type(self.visual_stride) is not int or self.visual_stride not in (16, 32):
            raise ValueError("visual_stride must be 16 or 32")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Need unique camera names")
        if len(self.image_hw) != 2 or any(type(x) is not int or x < 32 for x in self.image_hw):
            raise ValueError("image_hw must contain two integers >= 32")


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    amp: bool = False
    # Legacy checkpoint compatibility only; all-demonstration training requires zero.
    patience: int = 0
    min_delta: float = 0.0
    # 1 preserves full inverse-frequency balancing; 0.5 softens rare-group weights.
    balance_power: float = 1.0
    # Position mass: (1-alpha) + alpha*K on query zero; normalized over valid positions.
    first_query_weight: float = 0.0

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in ("seed", "num_workers", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "backbone_learning_rate", "weight_decay", "grad_clip", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if min(self.learning_rate, self.backbone_learning_rate, self.grad_clip) <= 0:
            raise ValueError("Learning rates and grad_clip must be positive")
        if type(self.amp) is not bool:
            raise ValueError("amp must be boolean")
        if not math.isfinite(self.balance_power) or not 0 <= self.balance_power <= 1:
            raise ValueError("balance_power must be in [0, 1]")
        if not math.isfinite(self.first_query_weight) or not 0 <= self.first_query_weight <= 1:
            raise ValueError("first_query_weight must be in [0, 1]")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        value = asdict(self)
        value["model"]["cameras"] = list(self.model.cameras)
        value["model"]["image_hw"] = list(self.model.image_hw)
        return value


def config_from_dict(value):
    if set(value) - {"model", "training"}:
        raise ValueError("Unknown configuration section")
    return Config(ModelConfig(**value.get("model", {})), TrainConfig(**value.get("training", {})))


def load_config(path=None):
    if path is None:
        return Config()
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
