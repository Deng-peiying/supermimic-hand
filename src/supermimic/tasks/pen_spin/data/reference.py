"""参考数据的张量转换；笔参考仅供奖励与 WM 训练使用。"""

import numpy as np
import torch

from supermimic.robots.prima1.config import STATE_LAYOUT


def tensor(value, device):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def references(clip, stage, device):
    ref = np.zeros((len(clip.q), STATE_LAYOUT.size), dtype=np.float32)
    ref[:, STATE_LAYOUT.index("joint_pos")] = clip.q
    ref[:, STATE_LAYOUT.index("joint_vel")] = clip.qd
    ref[:, STATE_LAYOUT.index("object_pos")] = clip.pose[:, :3]
    ref[:, STATE_LAYOUT.index("object_quat")] = clip.pose[:, 3:]
    if stage == "hold":
        ref[:, STATE_LAYOUT.index("joint_vel")] = 0
    return tensor(ref, device)
