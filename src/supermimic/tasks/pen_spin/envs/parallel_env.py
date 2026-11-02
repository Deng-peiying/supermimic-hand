"""批量 MuJoCo Warp 自由笔环境；只有 reset 写笔位姿，step 只写手指控制。"""

from contextlib import contextmanager

import mujoco_warp as mjw
import torch
import warp as wp

from supermimic.backends.warp_capacity import configure_epa_horizon
from supermimic.core.rewards import TensorRewardManager
from supermimic.robots.prima1.config import JOINTS, MAX_TARGET_DELTA

from ..data.reference import references
from .mdp.observations import append_history, initial_history
from .mdp.rewards import RewardInputs
from .pen_spin_env_cfg import make_env_cfg


class ParallelHandEnv:
    def __init__(
        self,
        clip,
        num_envs,
        device="cuda:0",
        *,
        stage="full",
        action_noise=0.02,
        nconmax=64,
        njmax=256,
        capture=True,
        domain_randomization=False,
    ):
        self.clip, self.num_envs, self.device = clip, num_envs, device
        self.cfg, self.stage = make_env_cfg(stage), stage
        self.length = {
            "hold": self.cfg.hold_steps,
            "short": self.cfg.short_steps,
            "full": len(clip.q) - 1,
        }[stage]
        self.action_noise = action_noise
        self.domain_randomization = domain_randomization
        self.ref = references(clip, stage, device)
        if hasattr(clip, 'clips'):
            self.reference_starts = torch.empty(len(clip.q), dtype=torch.long, device=device)
            self.reference_ends = torch.empty_like(self.reference_starts)
            for start, end in zip(clip.starts, clip.ends):
                self.reference_starts[start:end+1] = int(start)
                self.reference_ends[start:end+1] = int(end)
        self.model = clip.model()
        self.substeps = round(clip.dt / self.model.opt.timestep)
        if abs(self.substeps * self.model.opt.timestep - clip.dt) > 1e-8:
            raise ValueError("控制周期必须是仿真步长整数倍")
        ids = [self.model.joint(n).id for n in JOINTS]
        if list(self.model.actuator_trnid[:, 0]) != ids:
            raise ValueError("执行器顺序与已验收数据不匹配")
        self.qa = torch.tensor(self.model.jnt_qposadr[ids], device=device)
        self.va = torch.tensor(self.model.jnt_dofadr[ids], device=device)
        self.limits = torch.tensor(self.model.jnt_range[ids], device=device, dtype=torch.float32)
        pen = self.model.joint("pen_free").id
        self.pq, self.pv = int(self.model.jnt_qposadr[pen]), int(self.model.jnt_dofadr[pen])
        self.reward_manager = TensorRewardManager(
            self.cfg.rewards, dt=clip.dt, scale_by_dt=self.cfg.scale_rewards_by_dt
        )
        wp.init()
        self.epa_horizon = configure_epa_horizon()
        self.wp_device = wp.get_device(device)
        # 独立可捕获stream；physics_stream在Torch读写和Warp计算之间建立事件依赖。
        self.torch_stream = torch.cuda.Stream(device=device)
        self.stream = wp.stream_from_torch(self.torch_stream)
        with self.physics_stream():
            self.m = mjw.put_model(
                self.model,
                batch_sizes={
                    "geom_friction": num_envs,
                    "actuator_gainprm": num_envs,
                    "actuator_biasprm": num_envs,
                },
            )
            self.d = mjw.make_data(self.model, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            mjw.reset_data(self.m, self.d)
        self.qpos, self.qvel = wp.to_torch(self.d.qpos), wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xi = torch.ones(num_envs, 2, device=device)
        self.friction = wp.to_torch(self.m.geom_friction)
        self.gain = wp.to_torch(self.m.actuator_gainprm)
        self.bias = wp.to_torch(self.m.actuator_biasprm)
        self.base_friction = self.friction[0].clone()
        self.base_gain, self.base_bias = self.gain[0].clone(), self.bias[0].clone()
        self.frame = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros_like(self.frame)
        self.previous = torch.zeros(num_envs, 22, device=device)
        self.previous_q = self.previous.clone()
        self.history = initial_history(self.previous)
        self._reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._wp_mask = wp.from_torch(self._reset_mask)
        self._step_graph = self._reset_graph = None
        self.reset(torch.arange(num_envs, device=device), torch.zeros_like(self.frame))
        if capture:
            with self.physics_stream():
                with wp.ScopedCapture() as cap:
                    for _ in range(self.substeps):
                        mjw.step(self.m, self.d)
                self._step_graph = cap.graph
                with wp.ScopedCapture() as cap:
                    mjw.reset_data(self.m, self.d, self._wp_mask)
                self._reset_graph = cap.graph

    @contextmanager
    def physics_stream(self):
        current = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(current)
        with wp.ScopedDevice(self.wp_device), wp.ScopedStream(self.stream):
            yield
        current.wait_stream(self.torch_stream)

    def state(self):
        return torch.cat(
            (
                self.qpos[:, self.qa],
                self.qvel[:, self.va],
                self.qpos[:, self.pq : self.pq + 7],
                self.qvel[:, self.pv : self.pv + 6],
            ),
            -1,
        )

    def packed_state(self):
        # 后44维仅供重建传感器历史：上次位置与上次实际执行目标。
        return torch.cat((self.state(), self.previous_q, self.previous), -1)

    @torch.no_grad()
    def reset(self, ids, frames):
        if not len(ids):
            return
        self._reset_mask.zero_()
        self._reset_mask[ids] = True
        with self.physics_stream():
            if self._reset_graph is None:
                mjw.reset_data(self.m, self.d, self._wp_mask)
            else:
                wp.capture_launch(self._reset_graph)
        self.frame[ids], self.elapsed[ids] = frames, 0
        q = self.ref[frames, :22]
        self.qpos[ids[:, None], self.qa] = q
        self.qpos[ids, self.pq : self.pq + 7] = self.ref[frames, 44:51]
        self.ctrl[ids] = q
        self.previous[ids], self.previous_q[ids] = q, q
        self.history[ids] = initial_history(q)
        if self.domain_randomization:
            self.xi[ids, 0] = 0.8 + 0.4 * torch.rand(len(ids), device=self.device)
            self.xi[ids, 1] = 0.9 + 0.2 * torch.rand(len(ids), device=self.device)
        self.friction[ids] = self.base_friction * self.xi[ids, 0, None, None]
        self.gain[ids] = self.base_gain * self.xi[ids, 1, None, None]
        self.bias[ids] = self.base_bias * self.xi[ids, 1, None, None]
        # 人工放笔启动：reset_data 已将所有速度清零；不注入参考笔速度。

    @torch.no_grad()
    def step(self, target):
        old = self.state()
        prev = self.previous.clone()
        target = target + self.action_noise * torch.randn_like(target)
        target = torch.maximum(
            torch.minimum(target, prev + MAX_TARGET_DELTA), prev - MAX_TARGET_DELTA
        )
        target = torch.maximum(torch.minimum(target, self.limits[:, 1]), self.limits[:, 0])
        self.ctrl.copy_(target)
        with self.physics_stream():
            if self._step_graph is not None:
                wp.capture_launch(self._step_graph)
            else:
                for _ in range(self.substeps):
                    mjw.step(self.m, self.d)
        self.elapsed += 1
        if self.stage != "hold":
            self.frame += 1
        state = self.state()
        self.check_capacity()
        if not torch.isfinite(state).all():
            raise FloatingPointError("Warp 状态非有限，停止实验")
        self.previous_q.copy_(old[:, :22])
        self.previous.copy_(target)
        self.history = append_history(
            self.history, state[:, :22], old[:, :22], target, self.clip.dt
        )
        ref = self.ref[self.frame]
        result = self.reward_manager.compute(RewardInputs(state, ref, target, prev))
        failed = (state[:, 44:47] - ref[:, 44:47]).norm(dim=-1) > self.cfg.drop_distance_m
        end = self.reference_ends[self.frame] if hasattr(self, 'reference_ends') else self.length
        ended = self.elapsed >= self.length if self.stage == "hold" else self.frame >= end
        return state, result, failed, ended

    def check_capacity(self):
        """容量不足会丢约束，必须报错而非静默训练。"""
        ncon = int(wp.to_torch(self.d.nacon).max())
        nefc = int(wp.to_torch(self.d.nefc).max())
        overflow = wp.to_torch(self.d.overflow)
        if (overflow != 0).any():
            raise RuntimeError(
                f"Warp碰撞/约束溢出，停止训练: flags={torch.unique(overflow).tolist()}"
            )
        if ncon >= self.d.naconmax or nefc >= self.d.njmax:
            raise RuntimeError(f"接触/约束容量不足: contacts={ncon}, constraints={nefc}")
        return {"contacts": ncon, "max_constraints": nefc}

    def close(self):
        pass
