"""固定掌坐标：q22, qd22, pen xyz3, wxyz4, v_world3, omega_body3。

预测速度增量和显式积分后的位姿修正，零网络输出对应恒速运动。
"""

import torch

from supermimic.kinematics.rotations import (
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

STATE_DIM, FEAT_DIM, DELTA_DIM, CONTEXT_DIM = 57, 59, 56, 2


def canonical(s):
    return torch.cat(
        (
            s[..., :22],
            s[..., 22:44] / 10,
            s[..., 44:47] / 0.1,
            rot6d_from_quat(quat_normalize(s[..., 47:51])),
            s[..., 51:54],
            s[..., 54:57] / 10,
        ),
        -1,
    )


def integrate(s, delta, dt=0.02):
    qd = s[..., 22:44] + delta[..., :22] * 2
    v = s[..., 51:54] + delta[..., 22:25] * 0.3
    w = s[..., 54:57] + delta[..., 25:28] * 2
    q = s[..., :22] + dt * qd + delta[..., 28:50] * 0.02
    p = s[..., 44:47] + dt * v + delta[..., 50:53] * 0.005
    ori = quat_normalize(
        quat_mul(quat_normalize(s[..., 47:51]), quat_from_rotvec(dt * w + delta[..., 53:56] * 0.02))
    )
    return torch.cat((q, qd, p, ori, v, w), -1)


def error_vector(pred, true):
    q = quat_normalize(true[..., 47:51])
    inverse = torch.cat((q[..., :1], -q[..., 1:]), -1)
    rotation = rotvec_from_quat(quat_mul(inverse, quat_normalize(pred[..., 47:51])))
    return torch.cat(
        (
            (pred[..., :22] - true[..., :22]) / 0.1,
            (pred[..., 22:44] - true[..., 22:44]) / 2,
            (pred[..., 44:47] - true[..., 44:47]) / 0.025,
            rotation / 0.1,
            (pred[..., 51:54] - true[..., 51:54]) / 0.5,
            (pred[..., 54:57] - true[..., 54:57]) / 2,
        ),
        -1,
    )
