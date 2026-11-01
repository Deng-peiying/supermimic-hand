"""真实 MuJoCo 接触数据采集。只有 reset 可以设置自由笔位姿。"""

import mujoco
import numpy as np

from .data import JOINTS


class HandEnv:
    def __init__(self, clip, stage="short"):
        if stage not in ("hold", "short", "full"):
            raise ValueError(stage)
        self.clip, self.stage = clip, stage
        self.model = clip.model()
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(n).id for n in JOINTS]
        self.qa = self.model.jnt_qposadr[ids]
        self.va = self.model.jnt_dofadr[ids]
        self.limits = self.model.jnt_range[ids].copy()
        if list(self.model.actuator_trnid[:, 0]) != ids:
            raise ValueError("执行器顺序与数据不一致")
        pen = self.model.joint("pen_free").id
        self.pq = self.model.jnt_qposadr[pen]
        self.pv = self.model.jnt_dofadr[pen]
        self.substeps = round(clip.dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, clip.dt):
            raise ValueError("仿真步长无法整除控制周期")
        self.length = {"hold": 100, "short": 50, "full": len(clip.q) - 1}[stage]
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
        mujoco.mj_resetData(self.model, self.data)
        self.frame, self.elapsed = frame, 0
        self.data.qpos[self.qa] = self.clip.q[frame]
        self.data.qpos[self.pq : self.pq + 7] = self.clip.pose[frame]
        # 人工放笔启动：初始速度为零；不使用未来笔速度来启动策略。
        self.data.ctrl[:] = self.clip.q[frame]
        self.previous = self.clip.q[frame].copy()
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def step(self, target):
        target = np.asarray(target)
        if target.shape != (22,) or not np.isfinite(target).all():
            raise ValueError("非法关节命令")
        # 与策略输出相同的限位、每步最大目标变化。保存实际执行动作。
        target = np.clip(target, self.previous - 0.12, self.previous + 0.12)
        target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = target
        self.previous = target.astype(np.float32)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.elapsed += 1
        if self.stage != "hold":
            self.frame += 1
        state = self.state()
        if not np.isfinite(state).all():
            raise FloatingPointError("MuJoCo 状态非有限，停止收集")
        error = float(np.linalg.norm(state[44:47] - self.clip.pose[self.frame, :3]))
        dropped = error > 0.10
        truncated = self.elapsed >= self.length or (
            self.stage != "hold" and self.frame >= self.length
        )
        return (
            state,
            dropped or truncated,
            {"dropped": dropped, "truncated": truncated, "position_error_m": error},
        )
