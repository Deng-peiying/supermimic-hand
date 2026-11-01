"""State representation for the world model (plan D1–D3).

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

from supermimic.robots.g1.state import NUM_JOINTS, STATE_LAYOUT, split_state

STATE_DIM = STATE_LAYOUT.size
FEAT_DIM = 71
DELTA_DIM = 70
NJ = NUM_JOINTS


@dataclass # 规定世界模型如何理解、处理和预测机器人的物理状态
class StateSpecCfg:
    dt: float = 0.02  # 时间步长（秒）
    # 状态误差尺度（D3）：[根位置(m)、根旋转向量(rad)、根线速度(m/s)、根角速度(rad/s)、关节位置(rad)、关节速度(rad/s)]
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # 动力学预测输出尺度(将网络预测量还原为物理增量)：[水平线速度增量、机体系角速度增量、关节速度增量、水平位置增量、机体系旋转增量、关节位置增量]
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # 输入特征缩放除数(对输入特征进行数值缩放)：[根高度、6D旋转、水平线速度、机体系角速度、关节位置、关节速度]
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True  # 启用显式积分
    # 物理参数归一化：[摩擦系数、编码器偏差、质心偏移]
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)  # 归一化偏移量
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)  # 归一化尺度

class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return split_state(s)

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 71)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        dv_h, dw_b, dqd, dp_h, dth_b, dq = d[..., 0:3], d[..., 3:6], d[..., 6:35], d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
