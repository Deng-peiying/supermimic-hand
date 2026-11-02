"""公式与边界审计：不改正在训练的算法。"""

import numpy as np
import pytest
import torch

from supermimic.algorithms.objective import discounted_return, gae_targets
from supermimic.core.rewards import TensorRewardManager
from supermimic.tasks.pen_spin.envs.mdp.rewards import RewardInputs
from supermimic.tasks.pen_spin.envs.pen_spin_env_cfg import make_env_cfg


def test_reward_matches_independent_scalar_formula_and_gradcheck():
    cfg = make_env_cfg("full")
    manager = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=cfg.scale_rewards_by_dt)
    ref = torch.zeros(1, 57, dtype=torch.double)
    ref[:, 47] = 1
    state = ref.clone()
    state[:, :22] = 0.04
    state[:, 22:44] = 0.3
    state[:, 44:47] = torch.tensor([0.01, -0.02, 0.005])
    angle = 0.4
    state[:, 47:51] = torch.tensor([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    action = torch.full((1, 22), 0.02, dtype=torch.double)
    previous = torch.zeros_like(action)
    expected = 0.02 * (
        np.exp(-(0.04**2) / 0.15**2)
        + 0.2 * np.exp(-(0.3**2) / 3**2)
        + 3 * np.exp(-0.000525 / 0.025**2)
        + 2 * np.exp(-(np.sin(angle) ** 2) / 0.2)
        - 20 * 0.000525
        - 0.02 * (0.02 / 0.12) ** 2
    )
    result = manager.compute(RewardInputs(state, ref, action, previous))
    assert float(result.total) == pytest.approx(expected + 0.02 * 0.5, abs=1e-8)
    assert torch.autograd.gradcheck(
        lambda s, a: manager.compute(RewardInputs(s, ref, a, previous)).total,
        (state.requires_grad_(), action.requires_grad_()),
        atol=1e-5,
    )


def test_terminal_reward_and_bootstrap_exact_values():
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    done = torch.tensor([[0.0], [1.0], [1.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [1000.0]])
    # lambda=1：第二步终止，其奖励2仍保留，后续100与1000自举不进入首步目标。
    target = gae_targets(rewards, done, values, 0.9, 1.0)
    torch.testing.assert_close(target[:2], torch.tensor([[2.8], [2.0]]))
    survival = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    torch.testing.assert_close(
        discounted_return(rewards, survival, 0.9, values[-1]), torch.tensor([2.8])
    )


def test_axis_reward_has_known_stationary_point_at_ninety_degrees():
    cfg = make_env_cfg()
    ref = torch.zeros(1, 57, dtype=torch.double)
    ref[:, 47] = 1
    theta = torch.tensor(np.pi / 2, dtype=torch.double, requires_grad=True)
    quat = torch.stack((torch.cos(theta / 2), theta * 0, theta * 0, torch.sin(theta / 2)))
    state = torch.cat((ref[:, :47], quat[None], ref[:, 51:]), -1)
    zero = torch.zeros(1, 22, dtype=torch.double)
    reward = (
        TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=True)
        .compute(RewardInputs(state, ref, zero, zero))
        .weighted["pen_axis"]
    )
    (gradient,) = torch.autograd.grad(reward.sum(), theta)
    assert abs(float(gradient)) < 1e-10  # 公式本身的局限，不能误报为有强方向纠正梯度。
