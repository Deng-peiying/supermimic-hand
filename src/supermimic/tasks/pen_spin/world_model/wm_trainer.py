"""k步自回归 WM 监督训练，包含学习终止标签；窗口不跨 reset。"""

import torch


class WorldModelTrainer:
    def __init__(self, wm, ref, cfg):
        self.wm, self.ref, self.cfg = wm, ref, cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
        self.loss_fn = (
            torch.compile(wm.window_loss, dynamic=False) if cfg.compile else wm.window_loss
        )

    def update(self, replay):
        self.wm.requires_grad_(True)
        values = {}
        for _ in range(self.cfg.wm_steps):
            w = replay.sample_windows(self.cfg.batch_size, self.cfg.wm_window)
            next_frames = w.frame if self.cfg.stage == "hold" else w.frame + 1
            loss, metrics = self.loss_fn(
                w.states[..., :57], w.a_exec, w.xi, w.done, self.ref[next_frames]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("WM loss 非有限")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("WM gradient 非有限")
            self.opt.step()
            metrics["loss"], metrics["grad_norm"] = loss.detach(), grad.detach()
            for name, value in metrics.items():
                values[name] = values.get(name, 0.0) + value
        return {name: float(value / self.cfg.wm_steps) for name, value in values.items()}
