"""冻结 WM/Critic 的 BPTT Actor 更新 + 固定 GAE 目标的 Critic 更新。"""

import copy

import torch

from supermimic.algorithms.objective import discounted_return, gae_targets

from ..envs.mdp.observations import HISTORY
from .rollout import imagine


class PolicyTrainer:
    def __init__(self, wm, actor, critic, env, cfg):
        self.wm, self.actor, self.critic, self.env, self.cfg = wm, actor, critic, env, cfg
        self.opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
        if not 0 < cfg.critic_target_tau <= 1:
            raise ValueError("critic_target_tau 必须在 (0, 1] 内")
        # 目标网络参数固定，但保留对输入的梯度，Actor 仍通过 WM 做完整 BPTT。
        self.target_critic = copy.deepcopy(critic).requires_grad_(False)

    def update(self, replay):
        cfg = self.cfg
        start = replay.sample_starts(cfg.policy_batch_size, HISTORY)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            ro = imagine(self.wm, self.actor, self.env, start, cfg.horizon)
            last = ro.frames[-1]
            bootstrap = self.target_critic(
                ro.states[-1],
                ro.histories[-1],
                self.env.ref[last],
                last[:, None].float() / self.env.length,
                start.xi,
            )
            objective = discounted_return(ro.rewards, ro.survival, cfg.gamma, bootstrap).mean()
            loss = -objective / cfg.horizon
            if not torch.isfinite(loss):
                raise FloatingPointError("Actor loss 非有限")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            if not torch.isfinite(grad):
                raise FloatingPointError("Actor gradient 非有限")
            self.opt.step()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        with torch.no_grad():
            h, n = ro.frames.shape
            s, history, frame = ro.states.detach(), ro.histories.detach(), ro.frames
            xi = start.xi[None].expand(h, n, -1)
            inputs = (
                s.reshape(-1, 57),
                history.reshape(-1, HISTORY, 66),
                self.env.ref[frame.reshape(-1)],
                frame.reshape(-1, 1).float() / self.env.length,
                xi.reshape(-1, 2),
            )
            values = self.target_critic(*inputs).reshape(h, n)
            targets = gae_targets(
                ro.rewards.detach(), ro.dhat.detach(), values, cfg.gamma, cfg.gae_lambda
            ).reshape(-1)
            inputs = tuple(x[: (h - 1) * n] for x in inputs)
            valid = ro.active.reshape(-1).bool().nonzero().flatten()
            weights = ro.survival[:-1].detach().reshape(-1)
        for _ in range(cfg.critic_epochs):
            perm = valid[torch.randperm(len(valid), device=valid.device)]
            for idx in torch.tensor_split(perm, cfg.critic_minibatches):
                if not len(idx):
                    continue
                pred = self.critic(*(x[idx] for x in inputs))
                # 已终止后的虚构状态不应与真实可达状态等权拟合。
                weight = weights[idx]
                closs = ((pred - targets[idx]).square() * weight).sum() / weight.sum().clamp_min(1)
                if not torch.isfinite(closs):
                    raise FloatingPointError("Critic loss 非有限")
                self.critic_opt.zero_grad(set_to_none=True)
                closs.backward()
                cgrad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if not torch.isfinite(cgrad):
                    raise FloatingPointError("Critic gradient 非有限")
                self.critic_opt.step()
        with torch.no_grad():
            for target, online in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(online, cfg.critic_target_tau)
        return {
            "objective": float(objective.detach()),
            "policy_grad_norm": float(grad),
            "imag_return": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "survival": float(ro.survival[-1].mean().detach()),
            "critic_loss": float(closs.detach()),
            "critic_value": float(values.mean()),
            "critic_target_mean": float(targets.mean()),
            "critic_target_abs_max": float(targets.abs().max()),
        }
