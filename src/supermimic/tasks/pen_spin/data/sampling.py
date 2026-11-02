"""按真实回报更新的片段区间采样；保留首帧采样以验证人工放笔启动。"""

import torch


class ReturnSampler:
    def __init__(self, nenv, length, device, gamma=0.99, mode="return", max_step_reward=0.124):
        self.length, self.gamma, self.mode = length, gamma, mode
        if not 0 < gamma < 1 or max_step_reward <= 0:
            raise ValueError("采样器需要正的单步奖励上界和0<gamma<1")
        # 与真实奖励采用相同的dt缩放；不得保留旧的6.2硬编码。
        self.return_scale = max_step_reward / (1 - gamma)
        self.bins = (length + 49) // 50
        self.value = torch.zeros(self.bins, device=device)
        self.memory = torch.ones_like(self.value) / self.bins
        self.rewards = torch.zeros(nenv, length, device=device)
        self.frames = torch.zeros(nenv, length, device=device, dtype=torch.long)
        self.count = torch.zeros(nenv, device=device, dtype=torch.long)

    def sample(self, n):
        if n == 0:
            return torch.empty(0, device=self.value.device, dtype=torch.long)
        if self.mode == "start":
            return torch.zeros(n, device=self.value.device, dtype=torch.long)
        if self.mode == "uniform":
            return torch.randint(self.length, (n,), device=self.value.device)
        hard = torch.softmax(-25 * self.value / self.return_scale, 0)
        self.memory.lerp_(hard, 0.01)
        prob = 0.5 / self.bins + 0.4 * hard + 0.1 * self.memory / self.memory.sum()
        bins = torch.multinomial(prob, n, replacement=True)
        # 末区间不足50帧时按真实长度采样；不能clamp，否则概率会堆积在末帧。
        span = (self.length - bins * 50).clamp(max=50)
        offset = (torch.rand(n, device=self.value.device) * span).long()
        frames = bins * 50 + offset
        # 至少约20%回合从人工放笔的首帧启动。
        return torch.where(torch.rand(n, device=self.value.device) < 0.2, 0, frames)

    @torch.no_grad()
    def record(self, reward, frame, done):
        ids = torch.arange(len(reward), device=reward.device)
        self.rewards[ids, self.count] = reward
        self.frames[ids, self.count] = frame
        self.count += 1
        ended = ids[done]
        if not len(ended):
            return None
        size = int(self.count[ended].max())
        steps = torch.arange(size, device=reward.device)
        valid = steps[None] < self.count[ended, None]
        rewards = self.rewards[ended, :size] * valid
        powers = self.gamma ** steps.double()
        discounted = rewards.double() * powers
        returns = torch.flip(torch.cumsum(torch.flip(discounted, [1]), 1), [1]) / powers
        bins = (self.frames[ended, :size] // 50)[valid]
        total = torch.zeros_like(self.value).index_add_(0, bins, returns[valid].float())
        count = torch.zeros_like(self.value).index_add_(0, bins, torch.ones_like(bins).float())
        mask = count > 0
        self.value[mask] = 0.95 * self.value[mask] + 0.05 * total[mask] / count[mask]
        metrics = {
            "episodes": len(ended),
            "mean_reward": float((rewards.sum(1) / self.count[ended]).mean()),
            "episode_steps": float(self.count[ended].float().mean()),
        }
        self.count[ended] = 0
        return metrics


class MultiClipSampler:
    """先等概率选片段，再按片段内回报采样；全局帧号不会跨片段。"""
    def __init__(self, nenv, clip, device, gamma, mode, max_step_reward):
        self.starts = torch.tensor(clip.starts, device=device)
        self.lengths = torch.tensor(clip.ends - clip.starts, device=device)
        self.mode = mode
        self.gamma = gamma
        self.return_scale = max_step_reward / (1-gamma)
        self.bin_starts = torch.cat([torch.arange(int(s), int(e), 50, device=device)
                                     for s,e in zip(clip.starts, clip.ends)])
        self.bin_clip = torch.cat([torch.full(((int(e-s)+49)//50,), i, device=device)
                                   for i,(s,e) in enumerate(zip(clip.starts, clip.ends))]).long()
        self.bin_spans = torch.minimum(torch.full_like(self.bin_starts, 50),
                                      self.starts[self.bin_clip]+self.lengths[self.bin_clip]-self.bin_starts)
        self.frame_bins = torch.zeros(len(clip.q), device=device, dtype=torch.long)
        for i,(s,span) in enumerate(zip(self.bin_starts.tolist(),self.bin_spans.tolist())):
            self.frame_bins[s:s+span] = i
        self.value = torch.zeros(len(self.bin_starts), device=device)
        self.memory = torch.ones_like(self.value)
        self.rewards = torch.zeros(nenv, int(self.lengths.max()), device=device)
        self.frames = torch.zeros_like(self.rewards, dtype=torch.long)
        self.count = torch.zeros(nenv, device=device, dtype=torch.long)

    def sample(self, n):
        device = self.starts.device
        clips = torch.randint(len(self.starts), (n,), device=device)
        result = self.starts[clips].clone()
        for i in range(len(self.starts)):
            mask = clips == i
            count = int(mask.sum())
            if not count or self.mode == 'start':
                continue
            if self.mode == 'uniform':
                result[mask] += torch.randint(int(self.lengths[i]), (count,), device=device)
                continue
            bins = (self.bin_clip == i).nonzero().flatten()
            hard = torch.softmax(-25*self.value[bins]/self.return_scale, 0)
            self.memory[bins] = .99*self.memory[bins] + .01*hard
            prob = .5/len(bins)+.4*hard+.1*self.memory[bins]/self.memory[bins].sum()
            chosen = bins[torch.multinomial(prob,count,replacement=True)]
            sampled = self.bin_starts[chosen]+(torch.rand(count,device=device)*self.bin_spans[chosen]).long()
            result[mask] = torch.where(torch.rand(count,device=device)<.2,self.starts[i],sampled)
        return result

    @torch.no_grad()
    def record(self, reward, frame, done):
        ids = torch.arange(len(reward), device=reward.device)
        self.rewards[ids,self.count] = reward
        self.frames[ids,self.count] = frame
        self.count += 1
        ended = ids[done]
        if not len(ended):
            return None
        size = int(self.count[ended].max())
        steps = torch.arange(size,device=reward.device)
        valid = steps[None] < self.count[ended,None]
        rewards = self.rewards[ended,:size]*valid
        powers = self.gamma**steps.double()
        discounted = rewards.double()*powers
        returns = torch.flip(torch.cumsum(torch.flip(discounted,[1]),1),[1])/powers
        bins = self.frame_bins[self.frames[ended,:size]][valid]
        total = torch.zeros_like(self.value).index_add_(0,bins,returns[valid].float())
        counts = torch.zeros_like(self.value).index_add_(0,bins,torch.ones_like(bins).float())
        mask = counts > 0
        self.value[mask] = .95*self.value[mask]+.05*total[mask]/counts[mask]
        metrics = dict(episodes=len(ended),mean_reward=float((rewards.sum(1)/self.count[ended]).mean()),
                       episode_steps=float(self.count[ended].float().mean()))
        self.count[ended] = 0
        return metrics
