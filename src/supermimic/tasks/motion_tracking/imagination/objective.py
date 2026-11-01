"""Policy objective (Eq. 2) and critic GAE targets (Eq. 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from supermimic.algorithms.objective import discounted_return, gae_targets  # noqa: F401

if TYPE_CHECKING:
    import torch
    from ..imagination.rollout import ImagRollout


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    return discounted_return(ro.rewards, ro.survival, gamma, terminal_values).mean()


