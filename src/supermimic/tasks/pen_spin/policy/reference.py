"""已知手关节未来参考；不读取笔状态。末端重复末帧，不循环拼接。"""

import torch

FUTURE_FRAMES = 10


def future_joint_reference(reference_q, frame, length, stage):
    """返回 (..., 10, 22)，从当前帧的下一帧开始；hold 固定当前帧。"""
    offsets = torch.arange(1, FUTURE_FRAMES + 1, device=frame.device)
    indices = frame[..., None] + (0 * offsets if stage == "hold" else offsets)
    return reference_q[indices.clamp(max=min(length, len(reference_q) - 1)), :22]
