"""想象中的可微存活权重；与真实环境共用配置中的距离阈值。"""

import torch

from supermimic.robots.prima1.config import STATE_LAYOUT


def survival_probability(
    state, reference, distance_threshold=0.10, softness=0.005, *, layout=STATE_LAYOUT
):
    distance = (
        state[..., layout.index("object_pos")] - reference[..., layout.index("object_pos")]
    ).norm(dim=-1)
    return torch.sigmoid((distance_threshold - distance) / softness)
