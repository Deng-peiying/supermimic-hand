"""Task isolation, alternate embodiments, backward compatibility and BPTT semantics."""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.algorithms.models import ResidualActor, ResidualWorldModel
from supermimic.algorithms.objective import discounted_return
from supermimic.core.history import initial_history
from supermimic.core.rewards import TensorRewardManager, TensorRewardTermCfg
from supermimic.core.state import RewardInputs, StateLayout
from supermimic.tasks.pen_spin.envs.mdp.rewards import (
    joint_position_tracking_exp,
    pen_position_error_l2,
)


def test_rewards_and_models_support_alternate_layout():
    # Put the object BEFORE three joints: no Prima1 offsets are usable here.
    layout = StateLayout(
        (
            ("object_quat", 4),
            ("object_pos", 3),
            ("joint_pos", 3),
            ("joint_vel", 3),
        )
    )
    state = torch.zeros(2, layout.size, dtype=torch.float64)
    state[:, 0] = 1
    reference = state.clone()
    state[:, layout.index("object_pos")] = 0.1
    state[:, layout.index("joint_pos")] = 0.03
    state.requires_grad_()
    action = torch.zeros(2, 3, dtype=torch.float64)
    data = RewardInputs(state, reference, action, action, layout)
    manager = TensorRewardManager(
        {
            "joints": TensorRewardTermCfg(joint_position_tracking_exp, 1, {"std": 0.15}),
            "object": TensorRewardTermCfg(pen_position_error_l2, -20),
        },
        dt=0.02,
        scale_by_dt=False,
    )
    result = manager.compute(data)
    expected = torch.exp(torch.tensor(-(0.03**2) / 0.15**2, dtype=torch.float64)) - 20 * 0.03
    torch.testing.assert_close(result.total, expected.expand(2))
    result.total.sum().backward()
    assert state.grad[:, layout.index("object_pos")].abs().sum() > 0
    wm = ResidualWorldModel(layout, 3, [1.0] * layout.size).double()
    prediction = wm(state.detach(), action)
    torch.testing.assert_close(prediction, state.detach())
    actor = ResidualActor(
        [[-1, 1]] * 3, history_steps=4, residual_limit=0.15, max_target_delta=0.12
    )
    q = torch.zeros(2, 3)
    torch.testing.assert_close(actor(initial_history(q, 4), q, torch.zeros(2, 1)), q)


def test_manager_allows_term_specific_parameters_and_validates_shape():
    def custom(data, field, offset, enabled):
        return data.values[field] + offset if enabled else data.values[field] * 0

    data = SimpleNamespace(batch_shape=(2,), values={"score": torch.ones(2)})
    manager = TensorRewardManager(
        {
            "custom": TensorRewardTermCfg(
                custom, 2, {"field": "score", "offset": -1, "enabled": True}
            ),
        },
        dt=0.02,
        scale_by_dt=False,
    )
    torch.testing.assert_close(manager.compute(data).total, torch.zeros(2))
    data.batch_shape = (3,)
    with pytest.raises(ValueError, match="one scalar"):
        manager.compute(data)


def test_shared_return_preserves_terminal_transition_and_gradients():
    rewards = torch.tensor([[1.0], [2.0], [100.0]], requires_grad=True)
    survival = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=True)
    terminal = torch.tensor([50.0], requires_grad=True)
    result = discounted_return(rewards, survival, 0.9, terminal)
    torch.testing.assert_close(result, torch.tensor([2.8]))
    result.sum().backward()
    torch.testing.assert_close(rewards.grad[:, 0], torch.tensor([1.0, 0.9, 0.0]))
    assert terminal.grad.item() == 0
    assert survival.grad[1].item() > 0


@pytest.mark.parametrize(
    "module_name",
    [
        "supermimic.tasks.pen_spin.policy.nets",
        "supermimic.tasks.pen_spin.train",
        "supermimic.tasks.motion_tracking.world_model.state_spec",
        "supermimic.algorithms.replay_buffer",
    ],
)
def test_canonical_modules_have_their_own_source(module_name):
    module = importlib.import_module(module_name)
    expected = Path(__file__).resolve().parents[1] / "src" / (module_name.replace(".", "/") + ".py")
    assert Path(module.__file__).resolve() == expected.resolve()
    assert "sys.modules[__name__]" not in expected.read_text(encoding="utf-8")


def test_obsolete_sources_and_entry_points_removed():
    import tomllib

    repo = Path(__file__).resolve().parents[1]
    root = repo / "src/supermimic"
    for folder in ("hand", "envs", "eval", "imagination", "policy", "world_model"):
        assert not list((root / folder).rglob("*.py"))
    for file in ("config.py", "scripts/train.py", "training/loop.py", "kinematics/g1_fk.py"):
        assert not (root / file).exists()
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"sm-run": "supermimic.scripts.run:main"}


def test_shared_layers_do_not_import_task_or_robot_or_simulator():
    root = Path(__file__).resolve().parents[1] / "src/supermimic"
    forbidden = (
        "supermimic.tasks",
        "supermimic.tasks.pen_spin",
        "supermimic.robots",
        "mjlab",
        "mujoco",
    )
    for folder in ("core", "algorithms"):
        for path in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                assert not any(name.startswith(forbidden) for name in modules), str(path)


def test_g1_simulation_and_imagination_share_reward_configuration():
    from supermimic.tasks.motion_tracking.envs.tracking_env_cfg import (
        SuperMimicEnvCfgArgs,
        make_env_cfg,
        resolved_reward_cfg,
    )
    from supermimic.tasks.motion_tracking.reward_cfg import TrackingRewardCfg

    reward = TrackingRewardCfg(w_root_pos=0.7, std_root_pos=0.45, w_joint_limit=-3)
    args = SuperMimicEnvCfgArgs(npz_files=[], reward_cfg=reward)
    cfg = make_env_cfg(args)
    assert cfg.rewards["motion_global_root_pos"].weight == reward.w_root_pos
    assert cfg.rewards["motion_global_root_pos"].params["std"] == reward.std_root_pos
    assert cfg.rewards["joint_limit"].weight == reward.w_joint_limit
    args.joint_limit_weight = -4
    assert resolved_reward_cfg(args).w_joint_limit == -4
    assert reward.w_joint_limit == -3


def test_task_registry_is_lazy_and_rejects_duplicate_registration():
    from supermimic.tasks.registry import get_task, list_tasks, register_task

    assert {task.name for task in list_tasks()} >= {"g1_tracking", "prima1_pen_spin"}
    with pytest.raises(ValueError, match="already registered"):
        register_task(get_task("prima1_pen_spin"))
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("unknown")
