"""真实 MuJoCo 接触数据采集。只有 reset 可以设置自由笔位姿。"""

import numpy as np

from supermimic.backends.mujoco_cpu import MujocoCPU
from supermimic.core.rewards import TensorRewardManager
from supermimic.robots.prima1.config import MAX_TARGET_DELTA, NUM_JOINTS, STATE_LAYOUT

from ..data import JOINTS
from .pen_spin_env_cfg import PenSpinEnvCfg, make_env_cfg


class HandEnv:
    def __init__(self, clip, stage="short", cfg: PenSpinEnvCfg | None = None):
        self.cfg = cfg if cfg is not None else make_env_cfg(stage)
        if self.cfg.stage != stage:
            raise ValueError("stage 与显式任务配置不一致")
        self.clip, self.stage = clip, stage
        self.reward_manager = TensorRewardManager(
            self.cfg.rewards, dt=clip.dt, scale_by_dt=self.cfg.scale_rewards_by_dt
        )
        self.sim = MujocoCPU(clip.model(), clip.dt)
        self.model, self.data = self.sim.model, self.sim.data
        ids = [self.model.joint(n).id for n in JOINTS]
        self.qa = self.model.jnt_qposadr[ids]
        self.va = self.model.jnt_dofadr[ids]
        self.limits = self.model.jnt_range[ids].copy()
        if list(self.model.actuator_trnid[:, 0]) != ids:
            raise ValueError("执行器顺序与数据不一致")
        pen = self.model.joint("pen_free").id
        self.pq = self.model.jnt_qposadr[pen]
        self.pv = self.model.jnt_dofadr[pen]
        self.substeps = self.sim.substeps
        self.length = {
            "hold": self.cfg.hold_steps,
            "short": self.cfg.short_steps,
            "full": len(clip.q) - 1,
        }[stage]
        if self.length < 1 or (stage != "hold" and self.length >= len(clip.q)):
            raise ValueError("阶段长度超出参考片段")
        self.frame, self.elapsed = 0, 0

    def state(self):
        # 特权状态 57 维：q22, qd22, pen xyz3, quat4, v3, omega3。
        d = self.data
        return np.concatenate(
            [
                d.qpos[self.qa],
                d.qvel[self.va],
                d.qpos[self.pq : self.pq + 7],
                d.qvel[self.pv : self.pv + 6],
            ]
        ).astype(np.float32)

    def reset(self, frame=0):
        if not 0 <= frame < (1 if self.stage == "hold" else self.length):
            raise ValueError("reset 帧超出当前阶段")
        self.sim.reset()
        self.frame, self.elapsed = frame, 0
        self.data.qpos[self.qa] = self.clip.q[frame]
        self.data.qpos[self.pq : self.pq + 7] = self.clip.pose[frame]
        # 人工放笔启动：初始速度为零；不使用未来笔速度来启动策略。
        self.data.ctrl[:] = self.clip.q[frame]
        self.previous = self.clip.q[frame].copy()
        self.sim.forward()
        return self.state()

    def step(self, target):
        target = np.asarray(target)
        if target.shape != (NUM_JOINTS,) or not np.isfinite(target).all():
            raise ValueError("非法关节命令")
        # 与策略输出相同的限位、每步最大目标变化。保存实际执行动作。
        target = np.clip(target, self.previous - MAX_TARGET_DELTA, self.previous + MAX_TARGET_DELTA)
        target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = target
        self.previous = target.astype(np.float32)
        self.sim.step(target)
        self.elapsed += 1
        if self.stage != "hold":
            self.frame += 1
        state = self.state()
        if not np.isfinite(state).all():
            raise FloatingPointError("MuJoCo 状态非有限，停止收集")
        error = float(
            np.linalg.norm(state[STATE_LAYOUT.index("object_pos")] - self.clip.pose[self.frame, :3])
        )
        dropped = error > self.cfg.drop_distance_m
        truncated = self.elapsed >= self.length or (
            self.stage != "hold" and self.frame >= self.length
        )
        return (
            state,
            dropped or truncated,
            {
                "dropped": dropped,  # Legacy name: this is a tracking-distance failure.
                "tracking_failed": dropped,
                "truncated": truncated,
                "position_error_m": error,
            },
        )
