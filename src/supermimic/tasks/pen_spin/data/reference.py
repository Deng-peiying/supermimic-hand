"""参考数据的张量转换；笔参考仅供奖励与 WM 训练使用。"""

import numpy as np
import torch

from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.robots.prima1.config import STATE_LAYOUT

from ..envs.mdp.rewards import pen_axis


def reference_flip_velocity(quat, dt):
    """相邻笔轴最短弧的角速度，内点平均两侧区间；末端单侧。

    连续有向轴保留真实翻转方向，禁止为追求两端等价逐帧翻转轴符号。
    返回物体系分量以匹配物理状态布局；不包含绕笔轴自旋。
    """
    if dt <= 0 or len(quat) < 2:
        raise ValueError("角速度参考需要正dt及至少两帧")
    axis = pen_axis(quat)
    cross = torch.linalg.cross(axis[:-1], axis[1:])
    sine = cross.norm(dim=-1, keepdim=True)
    cosine = (axis[:-1] * axis[1:]).sum(-1, keepdim=True).clamp(-1, 1)
    if (cosine < -0.999).any():
        raise ValueError("相邻笔轴接近反向，翻转速度不可可靠重建；需要重新检查轨迹")
    interval = cross * (torch.atan2(sine, cosine) / sine.clamp_min(1e-12)) / dt
    world = torch.cat((interval[:1], (interval[:-1] + interval[1:]) / 2, interval[-1:]))
    world = world - (world * axis).sum(-1, keepdim=True) * axis
    return quat_apply_inverse(torch.nn.functional.normalize(quat, dim=-1), world)


def tensor(value, device):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def references(clip, stage, device):
    if hasattr(clip, 'clips'):
        if stage != 'full':
            raise ValueError('多片段目前仅支持 full；不得截断或拼接动作')
        # 分片计算速度，避免把两段之间的位姿跳变当作角速度。
        return torch.cat([references(c, stage, device) for c in clip.clips])
    ref = np.zeros((len(clip.q), STATE_LAYOUT.size), dtype=np.float32)
    ref[:, STATE_LAYOUT.index("joint_pos")] = clip.q
    ref[:, STATE_LAYOUT.index("joint_vel")] = clip.qd
    ref[:, STATE_LAYOUT.index("object_pos")] = clip.pose[:, :3]
    ref[:, STATE_LAYOUT.index("object_quat")] = clip.pose[:, 3:]
    if stage == "hold":
        ref[:, STATE_LAYOUT.index("joint_vel")] = 0
    result = tensor(ref, device)
    if stage != "hold":
        result[:, STATE_LAYOUT.index("object_ang_vel")] = reference_flip_velocity(
            result[:, STATE_LAYOUT.index("object_quat")], clip.dt
        )
    return result
