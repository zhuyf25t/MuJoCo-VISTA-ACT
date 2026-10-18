"""Synchronized public observations at the 10 Hz control boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


STATE_FIELDS = ("shoulder_rad", "elbow_rad", "gripper_opening", "base_v_mps", "base_w_radps")
OBSERVATION_FPS = 10


@dataclass(frozen=True, eq=False)
class FrameObservation:
    frame_no: int
    image: np.ndarray
    state: tuple[float, float, float, float, float]

    def __post_init__(self):
        if type(self.frame_no) is not int or self.frame_no < 0:
            raise ValueError("OBSERVATION_UNAVAILABLE: 帧号必须是非负整数")
        rgb = self.image
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
            raise ValueError("FRAME_UNAVAILABLE: 需要非空 uint8 RGB 原图")
        if (len(self.state) != len(STATE_FIELDS)
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not np.isfinite(value) for value in self.state)
                or not 0 <= self.state[2] <= 1):
            raise ValueError("OBSERVATION_UNAVAILABLE: 需要五维有限实测状态，夹爪开度须在 [0,1]")
        # A renderer may reuse its array. Own immutable bytes, not a mutable view.
        image = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "state", tuple(float(value) for value in self.state))

    @property
    def frame_id(self) -> str:
        return f"f{self.frame_no:06d}"

    def record(self) -> dict:
        return {"frame_id": self.frame_id, "state": list(self.state)}

    def measurements(self) -> dict:
        return dict(zip(STATE_FIELDS, self.state))


@dataclass(frozen=True)
class ObservationWindow:
    """Inclusive boundary observations o_k ... o_m; no inferred or missing ticks."""

    frames: tuple[FrameObservation, ...]

    def __post_init__(self):
        frames = tuple(self.frames)
        if not frames or any(not isinstance(frame, FrameObservation) for frame in frames):
            raise ValueError("OBSERVATION_UNAVAILABLE: 观测窗口不能为空")
        if any(right.frame_no != left.frame_no + 1 for left, right in zip(frames, frames[1:])):
            raise ValueError("OBSERVATION_UNAVAILABLE: 观测窗口帧号不连续")
        object.__setattr__(self, "frames", frames)

    @property
    def current(self) -> FrameObservation:
        return self.frames[-1]
