"""Reusable models with explicit dimensions; no robot, task or simulator imports."""

import torch
import torch.nn.functional as F
from torch import nn

from supermimic.core.state import StateLayout


class ResidualActor(nn.Module):
    """Joint history + reference joints + phase -> bounded joint targets."""

    def __init__(self, limits, *, history_steps, residual_limit, max_target_delta):
        super().__init__()
        limits = torch.as_tensor(limits, dtype=torch.float32)
        if limits.ndim != 2 or limits.shape[1] != 2 or limits.shape[0] < 1:
            raise ValueError("Expected one lower/upper limit pair per joint")
        if not torch.isfinite(limits).all() or (limits[:, 0] > limits[:, 1]).any():
            raise ValueError("Joint limits must be finite and ordered")
        self.num_joints = limits.shape[0]
        self.residual_limit = residual_limit
        self.max_target_delta = max_target_delta
        self.register_buffer("limits", limits)
        self.net = nn.Sequential(
            nn.Linear(history_steps * 3 * self.num_joints + self.num_joints + 1, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.num_joints),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history, reference_q, phase):
        residual = self.residual_limit * torch.tanh(
            self.net(torch.cat((history.flatten(1), reference_q, phase), -1))
        )
        previous = history[:, -1, 2 * self.num_joints : 3 * self.num_joints]
        target = torch.clamp(
            reference_q + residual,
            previous - self.max_target_delta,
            previous + self.max_target_delta,
        )
        return torch.clamp(target, self.limits[:, 0], self.limits[:, 1])


class ResidualWorldModel(nn.Module):
    """Normalized state increments with named quaternion normalization."""

    def __init__(self, layout: StateLayout, action_dim, scales, quaternion_field="object_quat"):
        super().__init__()
        scale = torch.as_tensor(scales, dtype=torch.float32)
        if scale.shape != (layout.size,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Each state coordinate needs a positive finite scale")
        self.quaternion = layout.index(quaternion_field)
        if self.quaternion.stop - self.quaternion.start != 4:
            raise ValueError("Quaternion field must have four coordinates")
        self.register_buffer("scale", scale)
        self.net = nn.Sequential(
            nn.Linear(layout.size + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, layout.size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _replace_quaternion(self, state, quaternion):
        return torch.cat(
            (state[..., : self.quaternion.start], quaternion, state[..., self.quaternion.stop :]),
            -1,
        )

    def forward(self, state, action):
        result = state + self.scale * self.net(torch.cat((state / self.scale, action), -1))
        quat = F.normalize(result[..., self.quaternion], dim=-1, eps=1e-6)
        return self._replace_quaternion(result, quat)

    def loss(self, state, action, next_state):
        prediction = self(state, action)
        sign = torch.where(
            (prediction[..., self.quaternion] * next_state[..., self.quaternion]).sum(
                -1, keepdim=True
            )
            < 0,
            -1.0,
            1.0,
        )
        target = self._replace_quaternion(next_state, next_state[..., self.quaternion] * sign)
        return ((prediction - target) / self.scale).square().mean()
