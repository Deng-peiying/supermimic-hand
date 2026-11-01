"""通用序列回放适配：额外保存历史所需的前一步位置与实际执行目标。"""

import torch

from supermimic.algorithms.replay_buffer import ReplayBuffer


def history_from_start(start, dt=0.02):
    s = start.hist_states
    q, prev_q, target = s[..., :22], s[..., 57:79], s[..., 79:101]
    return torch.cat((q, ((q - prev_q) / dt / 10).clamp(-5, 5), target), -1).transpose(0, 1)


class HandReplay(ReplayBuffer):
    def __init__(self, num_envs, capacity, device):
        super().__init__(num_envs, capacity, 101, 22, 2, device)
        self.is_start = torch.ones(num_envs, device=device, dtype=torch.bool)

    @torch.no_grad()
    def record(
        self,
        ids,
        packed,
        action,
        executed,
        xi,
        reward,
        frame,
        next_packed,
        next_frame,
        failed,
        ended,
    ):
        self.add(
            ids,
            packed,
            action,
            executed,
            xi,
            reward,
            torch.zeros_like(reward),
            frame,
            torch.zeros_like(frame),
            torch.zeros_like(failed),
            self.is_start,
        )
        self.is_start.zero_()
        terminal_ids = ids[failed | ended]
        if len(terminal_ids):
            count = len(terminal_ids)
            zero_a = torch.zeros(count, 22, device=self.device)
            self.add(
                terminal_ids,
                next_packed[terminal_ids],
                zero_a,
                zero_a,
                xi[terminal_ids],
                torch.zeros(count, device=self.device),
                failed[terminal_ids].float(),
                next_frame[terminal_ids],
                torch.zeros(count, dtype=torch.long, device=self.device),
                torch.ones(count, dtype=torch.bool, device=self.device),
                torch.zeros(count, dtype=torch.bool, device=self.device),
            )
            self.end_episode(terminal_ids)
            self.is_start[terminal_ids] = True
