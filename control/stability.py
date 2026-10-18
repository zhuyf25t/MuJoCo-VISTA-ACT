"""Compare public observations and shorten bounded waits without extra sampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Callable

import numpy as np

import config
from observations import FrameObservation


ObservationReader = Callable[[], FrameObservation]


@dataclass(frozen=True)
class ImageComparison:
    similarity: float
    max_local_changed_ratio: float
    changed_blocks: int
    total_blocks: int
    stable: bool


@dataclass(frozen=True)
class ObservationComparison:
    image: ImageComparison
    state_delta: tuple[float, ...]
    state_stable: bool
    base_stopped: bool

    @property
    def stable(self) -> bool:
        return self.image.stable and self.state_stable and self.base_stopped


@dataclass(frozen=True)
class WaitResult:
    ticks: int
    reason: str
    comparison: ObservationComparison | None = None


def compare_images(previous: np.ndarray, current: np.ndarray) -> ImageComparison:
    """Mean absolute RGB differences in 3x3 pixel blocks, including partial edges.

    Local windows overlap at every block, so motion cannot hide at tile boundaries.
    A high global similarity alone does not certify a stable image.
    """
    if (not isinstance(previous, np.ndarray) or not isinstance(current, np.ndarray)
            or previous.dtype != np.uint8 or current.dtype != np.uint8
            or previous.ndim != 3 or previous.shape != current.shape
            or previous.shape[2] != 3 or min(previous.shape[:2]) < 1):
        raise ValueError("Image comparison requires equal-sized nonempty uint8 RGB images")
    block = config.STABILITY_BLOCK_PIXELS
    height, width = previous.shape[:2]
    ys, xs = np.arange(0, height, block), np.arange(0, width, block)
    # Subtract before pooling: opposite changes inside one block cannot cancel.
    difference = np.abs(current.astype(np.float32) - previous.astype(np.float32))
    sums = np.add.reduceat(np.add.reduceat(difference, ys, axis=0), xs, axis=1)
    area = np.minimum(block, height - ys)[:, None] * np.minimum(block, width - xs)[None, :]
    changed = (sums / area[..., None]).max(axis=2) > config.STABILITY_IMAGE_EPS
    count = int(changed.sum())
    similarity = 1.0 - count / changed.size
    side = max(1, config.STABILITY_LOCAL_PIXELS // block)
    rows, columns = min(side, changed.shape[0]), min(side, changed.shape[1])
    integral = np.pad(changed.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    local_counts = (integral[rows:, columns:] - integral[:-rows, columns:]
                    - integral[rows:, :-columns] + integral[:-rows, :-columns])
    local_ratio = float(local_counts.max() / (rows * columns))
    return ImageComparison(similarity, local_ratio, count, changed.size,
                           similarity >= config.STABILITY_IMAGE_SIMILARITY
                           and local_ratio <= config.STABILITY_LOCAL_CHANGED_RATIO)


def compare_observations(previous: FrameObservation, current: FrameObservation) -> ObservationComparison:
    if current.frame_no != previous.frame_no + 1:
        raise ValueError("Stability requires consecutive newly sampled control frames")
    delta = np.abs(np.asarray(current.state) - np.asarray(previous.state))
    state_stable = bool(np.all(delta <= np.asarray(config.STABILITY_STATE_EPS)))
    # Equal velocities can mean uniform motion. Require a stopped base as well.
    base_stopped = bool(np.all(np.abs(np.asarray(current.state)[3:])
                               <= np.asarray(config.STABILITY_BASE_SPEED_EPS)))
    return ObservationComparison(compare_images(previous.image, current.image),
                                 tuple(float(value) for value in delta), state_stable, base_stopped)


def wait_until_stable(robot, seconds: float, on_tick=None,
                      read_observation: ObservationReader | None = None,
                      ready: Callable[[], bool] | None = None) -> WaitResult:
    """Execute at least one tick; never exceed the budget or bypass readiness.

    The collector owns sampling. Without a reader (standalone control clients),
    retain the full bounded wait instead of inventing observations or stability.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(seconds):
        raise ValueError("Wait duration must be finite seconds")
    ticks = floor(seconds / config.CTRL_DT + 1e-9)
    if ticks < 1:
        raise ValueError("Wait duration must allow at least one control tick")
    previous = read_observation() if read_observation is not None else None
    comparison = None
    for count in range(1, ticks + 1):
        robot.tick(on_tick)
        if read_observation is not None:
            current = read_observation()
            comparison = compare_observations(previous, current)
            if comparison.stable and (ready is None or ready()):
                return WaitResult(count, "stable", comparison)
            previous = current
    return WaitResult(ticks, "time_limit", comparison)
