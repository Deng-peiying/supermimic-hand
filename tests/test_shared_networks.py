"""Shared networks accept task-provided dimensions and preserve gradient flow."""

import ast
from pathlib import Path

import torch

from supermimic.algorithms.networks import DynamicsMLP, GoalConditionedPolicy, ValueNetwork


def test_policy_and_critic_use_explicit_dimensions():
    policy = GoalConditionedPolicy(
        goal_scale=torch.ones(5),
        obs_scale=torch.ones(12),
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        latent_dim=4,
        action_dim=3,
        last_gain=0.01,
        interface="grasp",
    )
    obs = torch.randn(2, 12, requires_grad=True)
    goal = torch.randn(2, 5, requires_grad=True)
    action = policy(obs, goal)
    assert action.shape == (2, 3)
    action.sum().backward()
    assert obs.grad.abs().sum() > 0 and goal.grad.abs().sum() > 0
    critic = ValueNetwork(
        feat_dim=7,
        goal_scale=torch.ones(5),
        context_dim=2,
        error_dim=1,
        hidden=(8,),
    )
    assert critic(torch.randn(2, 7), goal, torch.randn(2, 2), torch.randn(2, 1)).shape == (2,)


def test_dynamics_supports_different_state_encodings():
    model = DynamicsMLP(feat_dim=7, act_dim=3, xi_dim=2, hidden=(8,), out_dim=6)
    features, action, context = torch.randn(2, 7), torch.randn(2, 3), torch.randn(2, 2)
    torch.testing.assert_close(model(features, action, context), torch.zeros(2, 6))
    torch.nn.init.normal_(model.mlp.net[-1].weight, std=0.01)
    action.requires_grad_()
    model(features, action, context).sum().backward()
    assert action.grad.abs().sum() > 0


def test_tasks_do_not_import_each_other_and_robots_do_not_define_networks():
    root = Path(__file__).resolve().parents[1] / "src/supermimic"
    for name in ("pen_spin", "motion_tracking"):
        other = "motion_tracking" if name == "pen_spin" else "pen_spin"
        for path in (root / "tasks" / name).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                assert not any(m.startswith(f"supermimic.tasks.{other}") for m in modules), path
    for path in (root / "robots").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("torch.nn", "torch.nn.functional"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("torch.nn") for alias in node.names), path
