"""Shared finite-horizon BPTT return and optional value bootstrap."""

import torch


def discounted_return(rewards, survival, gamma, terminal_values=None):
    """Time-major rewards (H, ...), survival (H+1, ...); retain all gradients.

    A terminal transition earns its reward; survival masks subsequent rewards.
    Task adapters handle reference timing and episode truncation before this call.
    """
    horizon = rewards.shape[0]
    if horizon < 1 or survival.shape != (horizon + 1, *rewards.shape[1:]):
        raise ValueError("Expected H rewards and H+1 survival weights")
    discount = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discount = discount.reshape(horizon, *([1] * (rewards.ndim - 1)))
    result = (discount * survival[:-1] * rewards).sum(0)
    if terminal_values is not None:
        result = result + gamma**horizon * survival[-1] * terminal_values
    return result


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out
