"""ACT v2: unchanged visual/Transformer trunk, four supervised action heads.

The latent token is deterministic in training and inference; no CVAE posterior.
"""
import math
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

from .config import ModelConfig

def sine_1d(position, dim):
    if dim % 2:
        raise ValueError("Sine embedding dimension must be even")
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(0, dim, 2, device=position.device) / dim
    )
    angle = position.float().unsqueeze(-1) * frequency
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)


def sine_2d(height, width, dim, device):
    # 与特征 flatten(2) 一致：行优先；两个坐标各占一半通道。
    if dim % 4:
        raise ValueError("2D sine embedding dimension must be divisible by 4")
    y = (torch.arange(height, device=device) + 0.5) / height * 2 * math.pi
    x = (torch.arange(width, device=device) + 0.5) / width * 2 * math.pi
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.cat((sine_1d(yy, dim // 2), sine_1d(xx, dim // 2)), -1).reshape(-1, dim)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.dim_feedforward), nn.ReLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.dim_feedforward, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pos, is_pad=None):
        qk = x + pos
        update = self.attn(qk, qk, x, key_padding_mask=is_pad, need_weights=False)[0]
        x = self.norm1(x + self.drop(update))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.self_attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.dim_feedforward), nn.ReLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.dim_feedforward, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, query_pos, memory, memory_pos):
        qk = h + query_pos
        update = self.self_attn(qk, qk, h, need_weights=False)[0]  # 没有 causal mask
        h = self.norms[0](h + self.drop(update))
        update = self.cross_attn(h + query_pos, memory + memory_pos, memory,
                                 need_weights=False)[0]
        h = self.norms[1](h + self.drop(update))
        return self.norms[2](h + self.drop(self.ffn(h)))


class ACT(nn.Module):
    def __init__(self, cfg: ModelConfig, *, initialize_backbone=True, backbone_weights=None):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        if d % cfg.nheads or d % 4:
            raise ValueError("Invalid hidden_dim / nheads")
        weights = (ResNet18_Weights.IMAGENET1K_V1
                   if cfg.pretrained_backbone and initialize_backbone and backbone_weights is None else None)
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        if initialize_backbone and backbone_weights is not None:
            # Offline ImageNet weights, downloaded from the official torchvision URL.
            backbone.load_state_dict(torch.load(backbone_weights, map_location="cpu", weights_only=True))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.visual_projection = nn.Conv2d(512, d, kernel_size=1)
        self.state_projection = nn.Linear(cfg.state_dim, d)
        self.latent_token = nn.Parameter(torch.zeros(1, 1, d))
        self.type_pos = nn.Embedding(3, d)  # state / z / image
        self.camera_pos = nn.Embedding(len(cfg.cameras), d)
        self.action_queries = nn.Embedding(cfg.chunk_size, d)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.decoder_norm = nn.LayerNorm(d)
        self.arm_head = nn.Linear(d, 2)
        self.gripper_head = nn.Linear(d, 2)
        self.base_head = nn.Linear(d, 5)
        self.turn_head = nn.Linear(d, 1)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1))

    def forward(self, images, state):
        # images 已在 [0,1]；这里仅做 ImageNet 标准化。
        if images.ndim != 5 or tuple(images.shape[1:]) != (len(self.cfg.cameras), 3, *self.cfg.image_hw):
            raise ValueError("Expected images [B,cameras,3,H,W] matching configuration")
        B, C, _, H, W = images.shape
        if state.shape != (B, self.cfg.state_dim):
            raise ValueError("State shape mismatch")
        pixels = (images - self.rgb_mean) / self.rgb_std
        feature = self.visual_projection(self.backbone(pixels.reshape(B * C, 3, H, W)))
        _, d, h, w = feature.shape
        visual = feature.flatten(2).transpose(1, 2).reshape(B, C * h * w, d)
        spatial_pos = sine_2d(h, w, d, feature.device).view(1, h * w, d)
        image_pos = (spatial_pos + self.camera_pos.weight[:, None, :]
                     + self.type_pos.weight[2]).reshape(1, C * h * w, d)
        state_token = self.state_projection(state).unsqueeze(1)
        z_token = self.latent_token.expand(B, -1, -1)
        memory = torch.cat((state_token, z_token, visual), dim=1)
        pos = torch.cat((self.type_pos.weight[:2].unsqueeze(0), image_pos), dim=1)
        for layer in self.encoder:
            memory = layer(memory, pos)

        query_pos = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        h = torch.zeros_like(query_pos)
        for layer in self.decoder:
            h = layer(h, query_pos, memory, pos)
        h = self.decoder_norm(h)
        return {"arm": self.arm_head(h), "gripper_logits": self.gripper_head(h),
                "base_logits": self.base_head(h), "turn": self.turn_head(h)}
