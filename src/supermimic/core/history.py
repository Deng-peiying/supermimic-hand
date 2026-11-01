"""Proprioceptive history shared by simulation and deployment."""

import torch


def initial_history(q, steps):
    # Stationary start; never fabricate velocities from future reference frames.
    return torch.cat((q, torch.zeros_like(q), q), -1)[:, None].repeat(1, steps, 1)


def append_history(history, q, previous_q, executed_target, dt, velocity_scale=10.0):
    if dt <= 0:
        raise ValueError("History sample interval must be positive")
    velocity = ((q - previous_q) / dt / velocity_scale).clamp(-5, 5)
    sample = torch.cat((q, velocity, executed_target), -1)
    return torch.cat((history[:, 1:], sample[:, None]), 1)
