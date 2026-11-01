"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.algorithms.replay_buffer import ReplayBuffer, WindowBatch
from ..envs.mdp.motion_library import MotionLibrary
from ..world_model.heads import ReferenceFrame
from supermimic.training.distributed import allreduce_grads
from ..world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8  # 训练序列长度（8 个时间步）
    batches_per_update: int = 10  # 每轮更新处理的批次数
    batch_size: int = 2048  # 每批训练样本数
    lr: float = 2e-4  # 学习率（0.0002）
    grad_clip: float = 1.0  # 梯度裁剪阈值
    weight_decay: float = 0.0  # 权重衰减系数，0 表示不启用
    compile: str | None = "default"  # torch.compile 编译模式，None 表示不编译


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm  # 世界模型，需要训练的神经网络
        self.lib = lib  # 参考动作库
        self.cfg = cfg  # 训练配置，决定批次大小、学习率等参数
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)  # 创建Adam优化器
        wm.enable_compile(cfg.compile)  # 配置模型编译

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:  # 遍历优化器参数组
            g["lr"] = lr  # 更新学习率--用于按照训练调度动态调整学习率，对应Trainer.run()看到的self.wm_trainer.set_lr(lr)

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape # 获取序列长度和批次数量
        # 终止预测特征只需要参考运动的根部位姿
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)] # 按时间步构建参考帧
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs) # 计算序列预测损失

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}  # 累计训练指标
        for _ in range(self.cfg.batches_per_update):  # 遍历训练批次
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)  # 采样时间窗口
            loss, m = self.loss_on(w)  # 计算世界模型损失
            self.opt.zero_grad(set_to_none=True)  # 清空历史梯度
            loss.backward()  # 反向传播计算梯度
            allreduce_grads(self.wm)  # 跨进程平均梯度
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)  # 裁剪梯度范数
            self.opt.step()  # 更新世界模型参数
            m["loss"] = loss.detach()  # 记录损失，分离计算图
            m["grad_norm"] = gn.detach()  # 记录裁剪前梯度范数
            for k, v in m.items():  # 遍历训练指标
                agg[k] = agg.get(k, 0.0) + v  # 累计指标
        # 每轮更新仅进行一次主机同步，避免CPU在各批次间等待GPU
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}  # 返回平均训练指标

    @torch.no_grad()  # 禁用梯度计算 在线评估预测误差
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """在近期经验窗口上计算k步开环预测误差，返回形状为(k,)的张量。"""
        w = buffer.sample_windows(n, k)  # 采样n条长度为k的序列
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)  # 计算多步开环预测误差
