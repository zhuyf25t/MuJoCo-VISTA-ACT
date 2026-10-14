"""Brain 抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Decision:
    thought: str
    tool: str
    args: dict = field(default_factory=dict)


class Brain(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, obs: dict, task_text: str, tool_schemas: list[dict],
               history: list[dict]) -> list[Decision]:
        """根据观测(含图像)/任务/工具列表/历史, 产生一批工具调用(按序执行)."""
