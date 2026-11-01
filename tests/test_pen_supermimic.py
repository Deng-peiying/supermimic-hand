"""完整转笔机制的边界：积分约定、终止标签、末帧自举、真实GPU接触与恢复。"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.algorithms.replay_buffer import StartBatch
from supermimic.core.rewards import TensorRewardManager
from supermimic.tasks.pen_spin.data import DEFAULT_CLIP, Clip
from supermimic.tasks.pen_spin.data.replay import HandReplay, history_from_start
from supermimic.tasks.pen_spin.envs.pen_spin_env_cfg import make_env_cfg
from supermimic.tasks.pen_spin.imagination.rollout import imagine
from supermimic.tasks.pen_spin.policy.nets import Actor, Critic
from supermimic.tasks.pen_spin.world_model.state_spec import error_vector, integrate
from supermimic.tasks.pen_spin.world_model.world_model import WorldModel


def state(n=2):
    s = torch.zeros(n, 57)
    s[:, 47] = 1
    return s


def test_future_reference_boundary_hold_and_actor_uses_later_frames():
    from supermimic.tasks.pen_spin.policy.reference import future_joint_reference

    reference = torch.arange(20).float()[:, None].expand(-1, 22)
    future = future_joint_reference(reference, torch.tensor([0, 17]), 19, "full")
    torch.testing.assert_close(future[0, :, 0], torch.arange(1, 11).float())
    torch.testing.assert_close(future[1, :, 0], torch.tensor([18.0] + [19.0] * 9))
    short = future_joint_reference(reference, torch.tensor([4]), 5, "short")
    assert torch.all(short == 5)
    hold = future_joint_reference(reference, torch.tensor([3]), 19, "hold")
    assert torch.all(hold == 3)
    torch.manual_seed(2)
    actor = Actor([[-1, 1]] * 22)
    torch.nn.init.normal_(actor.net[-1].weight, std=0.01)
    goals = torch.zeros(1, 10, 22, requires_grad=True)
    actor(torch.zeros(1, 30, 66), goals, torch.zeros(1, 1)).sum().backward()
    assert goals.grad[:, 1:].abs().sum() > 0


def test_target_critic_updates_slowly_and_retains_input_gradient():
    from supermimic.tasks.pen_spin.config import TrainConfig
    from supermimic.tasks.pen_spin.imagination.policy_trainer import PolicyTrainer

    torch.manual_seed(7)
    cfg = TrainConfig(horizon=4, policy_batch_size=2, critic_epochs=1, critic_minibatches=1)
    ref = state(20)
    env = SimpleNamespace(
        ref=ref,
        stage="full",
        length=19,
        clip=SimpleNamespace(dt=0.02),
        reward_manager=TensorRewardManager(make_env_cfg("full").rewards, dt=0.02, scale_by_dt=True),
    )
    s = state()
    packed = torch.cat((s, s[:, :22], s[:, :22]), -1)
    start = StartBatch(
        packed,
        torch.ones(2, 2),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, 22),
        packed[None].expand(30, -1, -1),
        torch.zeros(30, 2, 22),
    )
    replay = SimpleNamespace(sample_starts=lambda *args: start)
    actor, wm, critic = Actor([[-1, 1]] * 22), WorldModel((16, 16)), Critic((16, 16))
    trainer = PolicyTrainer(wm, actor, critic, env, cfg)
    before = [p.clone() for p in trainer.target_critic.parameters()]
    metrics = trainer.update(replay)
    assert all(np.isfinite(v) for v in metrics.values())
    for old, target, online in zip(before, trainer.target_critic.parameters(), critic.parameters()):
        torch.testing.assert_close(target, old.lerp(online, cfg.critic_target_tau))
        assert not target.requires_grad and target.grad is None
    differentiable = s.clone().requires_grad_()
    trainer.target_critic(
        differentiable, torch.zeros(2, 30, 66), s, torch.zeros(2, 1), torch.ones(2, 2)
    ).sum().backward()
    assert differentiable.grad is not None and differentiable.grad.abs().sum() > 0


def test_short_final_bin_does_not_concentrate_samples_at_last_frame():
    from supermimic.tasks.pen_spin.data.sampling import ReturnSampler

    torch.manual_seed(123)
    sampler = ReturnSampler(2, 368, "cpu")
    frames = sampler.sample(20000)
    assert frames.min() >= 0 and frames.max() < 368
    # 8个区间，最后区间18帧；clamp错误会使末帧占比约6.6%。
    assert float((frames == 367).float().mean()) < 0.02
    assert 0.18 < float((frames == 0).float().mean()) < 0.23


def test_sampler_is_invariant_to_reward_time_scaling():
    from supermimic.tasks.pen_spin.data.sampling import ReturnSampler

    plain = ReturnSampler(2, 368, "cpu", max_step_reward=6.2)
    scaled = ReturnSampler(2, 368, "cpu", max_step_reward=6.2 * 0.02)
    plain.value.copy_(torch.arange(8) * 10.0)
    scaled.value.copy_(plain.value * 0.02)
    torch.manual_seed(9)
    expected = plain.sample(10000)
    torch.manual_seed(9)
    actual = scaled.sample(10000)
    torch.testing.assert_close(actual, expected)


def test_explicit_integration_matches_mujoco_free_joint():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>'
    )
    s = state(1)
    s[:, 47:51] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    s[:, 51:57] = torch.tensor([0.1, 0.2, -0.3, 1.0, -2.0, 0.5])
    s[:, 22:44] = 0.2
    result = integrate(s, torch.zeros(1, 56))
    qpos = s[0, 44:51].double().numpy().copy()
    mujoco.mj_integratePos(model, qpos, s[0, 51:57].double().numpy(), 0.02)
    np.testing.assert_allclose(result[0, 44:51], qpos, atol=1e-6)
    torch.testing.assert_close(result[:, :22], torch.full((1, 22), 0.004))
    flipped = result.clone()
    flipped[:, 47:51] *= -1
    torch.testing.assert_close(error_vector(result, flipped), torch.zeros(1, 56), atol=1e-6, rtol=0)


def test_replay_terminal_labels_history_and_episode_separation():
    replay = HandReplay(2, 64, "cpu")
    ids = torch.arange(2)
    s = state()
    packed = torch.cat((s, s[:, :22], s[:, :22]), -1)
    for i in range(3):
        replay.record(
            ids,
            packed,
            torch.zeros(2, 22),
            torch.zeros(2, 22),
            torch.ones(2, 2),
            torch.ones(2),
            torch.full((2,), i),
            packed,
            torch.full((2,), i + 1),
            torch.tensor([i == 2, False]),
            torch.tensor([False, i == 2]),
        )
    w = replay.sample_windows(200, 3)
    assert (w.done[-1] == 1).any() and (w.done[-1] == 0).any()
    assert not w.done[:-1].any()
    start = replay.sample_starts(8, 30)
    h = history_from_start(start)
    torch.testing.assert_close(h, torch.zeros(8, 30, 66))
    changed = start.hist_states.clone()
    changed[..., 44:57] = 100
    start.hist_states = changed
    torch.testing.assert_close(history_from_start(start), h)


def test_reference_end_stops_rewards_bootstrap_and_preserves_action_gradient():
    cfg = make_env_cfg("full")
    ref = state(4)
    env = SimpleNamespace(
        ref=ref,
        stage="full",
        length=3,
        clip=SimpleNamespace(dt=0.02),
        reward_manager=TensorRewardManager(
            cfg.rewards, dt=0.02, scale_by_dt=cfg.scale_rewards_by_dt
        ),
    )
    s = state()
    packed = torch.cat((s, s[:, :22], s[:, :22]), -1)
    start = StartBatch(
        packed,
        torch.ones(2, 2),
        torch.full((2,), 2),
        torch.zeros(2, 22),
        packed[None].expand(30, -1, -1),
        torch.zeros(30, 2, 22),
    )
    actor, wm, critic = Actor([[-1, 1]] * 22), WorldModel((32, 32)), Critic((32, 32))
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=0.01)
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    ro = imagine(wm, actor, env, start, 4)
    assert torch.all(ro.rewards[1:] == 0) and torch.all(ro.survival[1:] == 0)
    assert torch.all(ro.dhat[0] == 1)
    for i in range(2, len(ro.states)):
        torch.testing.assert_close(ro.states[i], ro.states[1])
        torch.testing.assert_close(ro.histories[i], ro.histories[1])
    (-ro.rewards[0].mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in wm.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (DEFAULT_CLIP / "candidate.npz").exists(),
    reason="requires GPU and reviewed assets",
)
def test_warp_cpu_parity_and_selective_reset():
    from supermimic.tasks.pen_spin.envs.env import HandEnv
    from supermimic.tasks.pen_spin.envs.parallel_env import ParallelHandEnv

    clip = Clip()
    gpu = ParallelHandEnv(clip, 2, action_noise=0, domain_randomization=False)
    cpu = HandEnv(clip, "full")
    cpu.reset()
    for i in range(5):
        gs, _, _, _ = gpu.step(gpu.ref[gpu.frame + 1, :22])
        cs, _, _ = cpu.step(clip.q[i + 1])
        if i == 0:
            np.testing.assert_allclose(gs[0, :22].cpu(), cs[:22], atol=1e-3, rtol=0)
    # CPU double与GPU float32接触求解允许小数值差异，不能错坐标/错关节顺序。
    # 接触算法不同，五步后某关节约相差0.017rad；不宣称长轨迹后端等价。
    np.testing.assert_allclose(gs[0, 44:47].cpu(), cs[44:47], atol=1e-3, rtol=0)
    old = gpu.qpos[0].clone()
    gpu.reset(torch.tensor([1], device="cuda"), torch.tensor([0], device="cuda"))
    torch.testing.assert_close(old, gpu.qpos[0], rtol=0, atol=0)
    assert gpu.qvel[1].count_nonzero() == 0
    gpu.check_capacity()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (DEFAULT_CLIP / "candidate.npz").exists(),
    reason="requires GPU and reviewed assets",
)
def test_warp_cpu_without_contact_agree(monkeypatch):
    from supermimic.tasks.pen_spin.envs.env import HandEnv
    from supermimic.tasks.pen_spin.envs.parallel_env import ParallelHandEnv

    clip = Clip()
    original = clip.model

    def no_contact():
        model = original()
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        return model

    monkeypatch.setattr(clip, "model", no_contact)
    gpu = ParallelHandEnv(clip, 2, action_noise=0)
    cpu = HandEnv(clip, "full")
    cpu.reset()
    for i in range(10):
        gs, _, _, _ = gpu.step(gpu.ref[gpu.frame + 1, :22])
        cs, _, _ = cpu.step(clip.q[i + 1])
        np.testing.assert_allclose(gs[0].cpu(), cs, atol=1e-4, rtol=0)
