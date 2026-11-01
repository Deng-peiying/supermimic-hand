"""Differentiable reward aggregation; parameter semantics belong to each term."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


class RewardContext(Protocol):
    @property
    def batch_shape(self) -> tuple[int, ...]: ...


@dataclass
class TensorRewardTermCfg:
    func: Callable[..., torch.Tensor]
    weight: float
    params: dict[str, Any] = field(default_factory=dict)
    # Stable identity survives module relocation in checkpoint descriptions.
    identifier: str | None = None

    def describe(self):
        return {
            "func": self.identifier or f"{self.func.__module__}.{self.func.__name__}",
            "weight": self.weight,
            "params": dict(self.params),
        }


@dataclass
class RewardResult:
    total: torch.Tensor
    raw: dict[str, torch.Tensor]
    weighted: dict[str, torch.Tensor]


class TensorRewardManager:
    def __init__(self, terms: Mapping[str, TensorRewardTermCfg], *, dt: float, scale_by_dt: bool):
        if not terms or dt <= 0 or not math.isfinite(dt):
            raise ValueError("Rewards must be nonempty and dt positive and finite")
        for name, term in terms.items():
            if not callable(term.func) or not math.isfinite(term.weight):
                raise ValueError(f"Invalid reward term: {name}")
        self.terms = dict(terms)
        self.scale = dt if scale_by_dt else 1.0

    def compute(self, data: RewardContext) -> RewardResult:
        raw, weighted = {}, {}
        for name, term in self.terms.items():
            value = term.func(data, **term.params)
            if value.shape != data.batch_shape:
                raise ValueError(f"Reward {name} must return one scalar per sample")
            raw[name] = value
            weighted[name] = value * term.weight * self.scale
        return RewardResult(torch.stack(list(weighted.values())).sum(0), raw, weighted)
