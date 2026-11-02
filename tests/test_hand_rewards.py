"""重构回归：旧目标的数值/梯度、圆柱对称性、分项日志与配置身份。"""

import numpy as np
import pytest
import torch

from supermimic.core.rewards import TensorRewardManager
from supermimic.tasks.pen_spin.envs.mdp.rewards import RewardInputs, pen_axis
from supermimic.tasks.pen_spin.envs.pen_spin_env_cfg import make_env_cfg
from supermimic.tasks.pen_spin.eval.benchmark import evaluate
from supermimic.tasks.pen_spin.policy.nets import Actor
from supermimic.tasks.pen_spin.training.checkpoint import check_task_config


def original_reward(s, ref, target, previous):
    """冻结重构前的六项公式作为回归基准；不调用新的 term 函数。"""
    q = (s[..., :22] - ref[..., :22]).square().mean(-1)
    qd = (s[..., 22:44] - ref[..., 22:44]).square().mean(-1)
    pos = (s[..., 44:47] - ref[..., 44:47]).square().sum(-1)
    dot = (pen_axis(s[..., 47:51]) * pen_axis(ref[..., 47:51])).sum(-1)
    axis = (1 - dot.square()).clamp(0, 1)
    return (
        torch.exp(-q / 0.15**2)
        + 0.2 * torch.exp(-qd / 3**2)
        + 3 * torch.exp(-pos / 0.025**2)
        + 2 * torch.exp(-axis / 0.2)
        - 20 * pos
        - 0.02 * ((target - previous) / 0.12).square().mean(-1)
    )


def test_refactor_preserves_reward_and_bptt_gradients():
    torch.manual_seed(8)
    ref = torch.randn(4, 57, dtype=torch.float64)
    state = (ref + 0.02 * torch.randn_like(ref)).requires_grad_()
    target = torch.randn(4, 22, dtype=torch.float64, requires_grad=True)
    previous = target.detach() + 0.03
    cfg = make_env_cfg()
    result = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=False).compute(
        RewardInputs(state, ref, target, previous)
    )
    # 六项基线公式保持不变；第七项单独验证。
    result = TensorRewardManager(
        {k: v for k, v in cfg.rewards.items() if k != "pen_flip_velocity"},
        dt=0.02,
        scale_by_dt=False,
    ).compute(RewardInputs(state, ref, target, previous))
    expected = original_reward(state, ref, target, previous)
    torch.testing.assert_close(result.total, expected)
    grads = torch.autograd.grad(result.total.sum(), (state, target), retain_graph=True)
    old_grads = torch.autograd.grad(expected.sum(), (state, target))
    for actual, old in zip(grads, old_grads):
        torch.testing.assert_close(actual, old)
    assert result.weighted["pen_distance"].max() <= 0
    assert result.weighted["action_delta"].max() <= 0


def test_explicit_dt_scaling_and_config_changes():
    cfg = make_env_cfg()
    assert cfg.scale_rewards_by_dt is True
    state = torch.zeros(2, 57)
    state[:, 47] = 1
    zero = torch.zeros(2, 22)
    data = RewardInputs(state, state, zero, zero)
    plain = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=False).compute(data)
    scaled = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=True).compute(data)
    torch.testing.assert_close(plain.total, torch.full((2,), 6.7))
    torch.testing.assert_close(scaled.total, plain.total * 0.02)
    # 每个配置单独构造；改权重不能污染其他实验。
    cfg.rewards["pen_position"].weight = 0
    changed = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=False).compute(data)
    torch.testing.assert_close(changed.total, plain.total - 3)
    assert make_env_cfg().rewards["pen_position"].weight == 3


def test_no_reward_inheritance_and_checkpoint_identity():
    from types import SimpleNamespace

    cfg = make_env_cfg()
    assert cfg.describe()["inherits_rewards_from"] == []
    env = SimpleNamespace(cfg=cfg, stage="short")
    ckpt = {"task_config": cfg.describe()}
    check_task_config(ckpt, env)
    cfg.rewards["pen_position"].weight = 4
    with pytest.raises(ValueError, match="配置不同"):
        check_task_config(ckpt, env)
    with pytest.raises(ValueError, match="旧检查点"):
        check_task_config({}, env)


def test_physical_evaluation_term_totals(tmp_path):
    from supermimic.tasks.pen_spin.data import DEFAULT_CLIP, Clip
    from supermimic.tasks.pen_spin.envs.env import HandEnv

    if not (DEFAULT_CLIP / "candidate.npz").exists():
        pytest.skip("需要本地人工验收的 Prima1 数据和资产")
    env = HandEnv(Clip(), "short")
    metrics = evaluate(env, Actor(env.limits), "cpu", tmp_path / "baseline", baseline=True)
    assert len(metrics["reward_raw_mean"]) == 7
    assert np.isclose(sum(metrics["reward_weighted_mean"].values()), metrics["mean_reward"])
    assert np.isclose(sum(metrics["reward_term_return"].values()), metrics["return"])
    assert (tmp_path / "task_config.json").exists()
