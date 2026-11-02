import torch

from supermimic.kinematics.rotations import quat_from_rotvec
from supermimic.tasks.pen_spin.data.reference import reference_flip_velocity
from supermimic.tasks.pen_spin.envs.mdp.rewards import (
    RewardInputs,
    flip_angular_velocity,
    pen_flip_velocity_tracking,
)


def test_reference_constant_flip_and_quaternion_sign():
    t = torch.arange(12, dtype=torch.double) * 0.02
    q = quat_from_rotvec(torch.stack((t * 0, t * 0, 2 * t), -1))
    velocity = reference_flip_velocity(q, 0.02)
    torch.testing.assert_close(
        velocity, torch.tensor([0.0, 0.0, 2.0], dtype=torch.double).expand(12, 3)
    )
    q[::2] *= -1
    torch.testing.assert_close(reference_flip_velocity(q, 0.02), velocity)
    spin = quat_from_rotvec(torch.stack((3 * t, t * 0, t * 0), -1))
    torch.testing.assert_close(reference_flip_velocity(spin, 0.02), torch.zeros_like(velocity))


def test_flip_reward_direction_roll_invariance_and_gradient():
    ref = torch.zeros(1, 57, dtype=torch.double)
    ref[:, 47] = 1
    ref[:, 56] = 2
    actual = ref.clone()
    actual[:, 54] = 50  # 长轴自旋不影响翻转奖励。
    zero = torch.zeros(1, 22, dtype=torch.double)
    reward = lambda s: pen_flip_velocity_tracking(RewardInputs(s, ref, zero, zero), 3.0)
    torch.testing.assert_close(reward(actual), torch.ones(1, dtype=torch.double))
    reverse = actual.clone()
    reverse[:, 56] = -2
    assert reward(reverse) < reward(actual)
    actual[:, 47:51] = quat_from_rotvec(
        torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=torch.double)
    )
    actual[:, 54:57] = 0
    actual.requires_grad_()
    reward(actual).sum().backward()
    assert actual.grad[0, 56] > 0  # 轴恰好90°，仍能提供翻转速度纠正信号。
    assert torch.autograd.gradcheck(reward, (actual.detach().requires_grad_(),))


def test_body_to_world_velocity_convention():
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=torch.double))
    body = torch.tensor([[7.0, 2.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(
        flip_angular_velocity(q, body),
        torch.tensor([[-2.0, 0.0, 0.0]], dtype=torch.double),
        atol=1e-12,
        rtol=0,
    )
