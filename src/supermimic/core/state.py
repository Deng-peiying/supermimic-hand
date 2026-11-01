"""Named views of packed model tensors, independent of robot and simulator."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateLayout:
    fields: tuple[tuple[str, int], ...]

    def __post_init__(self):
        names = [name for name, _ in self.fields]
        if not names or len(set(names)) != len(names):
            raise ValueError("State fields must be nonempty and unique")
        if any(size < 1 for _, size in self.fields):
            raise ValueError("State field sizes must be positive")

    @property
    def size(self):
        return sum(size for _, size in self.fields)

    def index(self, name):
        offset = 0
        for field, size in self.fields:
            if field == name:
                return slice(offset, offset + size)
            offset += size
        raise KeyError(name)

    def unpack(self, tensor):
        if tensor.shape[-1] != self.size:
            raise ValueError(f"Expected state width {self.size}, got {tensor.shape[-1]}")
        return {name: tensor[..., self.index(name)] for name, _ in self.fields}


@dataclass
class RewardInputs:
    state: torch.Tensor
    reference: torch.Tensor
    target: torch.Tensor
    previous_target: torch.Tensor
    layout: StateLayout

    @property
    def batch_shape(self):
        return self.state.shape[:-1]

    def current(self, name):
        return self.state[..., self.layout.index(name)]

    def desired(self, name):
        return self.reference[..., self.layout.index(name)]

    def __post_init__(self):
        if self.state.shape != self.reference.shape or self.state.shape[-1] != self.layout.size:
            raise ValueError("State and reference must match the state layout")
        if self.target.shape != self.previous_target.shape:
            raise ValueError("Current and previous actions must have matching shapes")
        if self.target.shape[:-1] != self.batch_shape:
            raise ValueError("Action and state batch dimensions must match")
