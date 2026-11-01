"""关注无信息泄漏、自由物体动力学、训练梯度和数据身份，不测试实现细枝末节。"""

import numpy as np
import pytest
import torch

from supermimic.hand.data import DEFAULT_CLIP, JOINTS, Clip
from supermimic.hand.deployment import OfflineController
from supermimic.hand.env import HandEnv
from supermimic.hand.networks import (
    Actor,
    WorldModel,
    initial_history,
    mimic_reward,
    survival_probability,
)
from supermimic.hand.train import collect, export


@pytest.fixture
def env():
    if not (DEFAULT_CLIP / "candidate.npz").exists():
        pytest.skip("需要本地单片段与 Prima1 资产；见 docs/HANDMIMIC.md")
    return HandEnv(Clip())


def test_actor_no_privileged_inputs_and_bounded():
    actor = Actor(np.tile([-0.5, 0.5], (22, 1)))
    torch.nn.init.normal_(actor.net[-1].weight, std=0.05)
    state = torch.zeros(2, 57)
    state[1, 44:] = 100  # 笔的位置、速度、姿态全部不同。
    history = initial_history(state[:, :22])
    output = actor(history, torch.zeros(2, 22), torch.zeros(2, 1))
    assert torch.equal(output[0], output[1])
    assert output.abs().max() <= 0.12


def test_imagination_survival_has_distance_gradient():
    state = torch.zeros(3, 57)
    state[:, 44] = torch.tensor([0.0, 0.1, 0.2])
    state.requires_grad_(True)
    alive = survival_probability(state, torch.zeros_like(state))
    assert alive[0] > 0.999 and alive[2] < 0.001
    assert torch.isclose(alive[1], torch.tensor(0.5))
    alive.sum().backward()
    assert state.grad[1, 44] < 0 and torch.isfinite(state.grad).all()


def test_world_model_action_gradient_and_quaternion():
    torch.manual_seed(3)
    wm = WorldModel()
    torch.nn.init.normal_(wm.net[-1].weight, std=0.001)
    wm.requires_grad_(False)
    state = torch.zeros(3, 57)
    state[:, 47] = 1
    action = torch.zeros(3, 22, requires_grad=True)
    for _ in range(4):
        state = wm(state, action)
    assert torch.allclose(state[:, 47:51].norm(dim=-1), torch.ones(3), atol=1e-5)
    state[:, 44:47].sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in wm.parameters())


def test_reward_pen_symmetry_and_position():
    state = torch.zeros(1, 57)
    state[:, 47] = 1
    ref = state.clone()
    zero = torch.zeros(1, 22)
    good = mimic_reward(state, ref, zero, zero)
    state[:, 47] = -1
    assert torch.allclose(good, mimic_reward(state, ref, zero, zero))
    state[:, 44] += 0.1
    assert mimic_reward(state, ref, zero, zero) < good


def test_free_pen_and_terminal_reset_separation(env):
    state = env.reset()
    # 将笔移远以排除接触：重力应改变它，step 不得写回参考轨迹。
    env.data.qpos[env.pq : env.pq + 3] += 1.0
    before = env.state()[44:47].copy()
    after, done, info = env.step(state[:22])
    assert np.dot(after[44:47] - before, env.clip.gravity) > 0
    assert np.linalg.norm(after[44:47] - env.clip.pose[1, :3]) > 1
    assert done and info["dropped"]
    assert np.isclose(env.data.time, 0.02)
    actor = Actor(env.limits)
    records = collect(env, actor, 80, "cpu", np.random.default_rng(5), noise=0)
    terminals = [r for r in records if r[5]]
    assert terminals
    assert all(np.linalg.norm(r[2][44:47] - env.clip.pose[0, :3]) > 0.01 for r in terminals)
    reset_indices = [i + 1 for i, r in enumerate(records[:-1]) if r[5]]
    assert all(records[i][4] == 0 for i in reset_indices)


def test_export_runtime_and_watchdog(env, tmp_path):
    actor = Actor(env.limits)
    export(actor, env.clip, env, tmp_path)
    mapping = [
        {"urdf_joint": n, "sdk_name": n, "scale_to_rad": 1.0, "offset_rad": 0.0} for n in JOINTS
    ]
    controller = OfflineController(tmp_path, mapping)
    q = env.clip.q[0]
    values = dict(zip(JOINTS, q))
    result = controller.step(values, 0.0)
    expected = actor(
        initial_history(torch.tensor(q[None])), torch.tensor(env.clip.q[None, 1]), torch.zeros(1, 1)
    )
    np.testing.assert_allclose(list(result.values()), expected.detach().numpy()[0], atol=1e-6)
    with pytest.raises(TimeoutError):
        controller.step(values, 0.2)
    with pytest.raises(RuntimeError):
        controller.step(values, 0.22)


def test_reset_stage_boundary(env):
    env.reset(49)
    _, done, info = env.step(env.clip.q[50])
    assert done and info["truncated"] and env.frame == 50
    with pytest.raises(ValueError):
        env.reset(50)
