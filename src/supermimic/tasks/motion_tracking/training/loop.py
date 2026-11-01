"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import TrainConfig
from ..data.motion_convert import load_g1_model
from ..envs import SuperMimicEnvCfgArgs, make_env
from ..envs.mdp.motion_library import discover_npz
from ..envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from ..imagination.policy_trainer import PolicyTrainer
from supermimic.robots.g1.kinematics import G1Kinematics
from ..policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed
from supermimic.training.schedule import cosine_lr
from ..world_model.heads import AnalyticHeads
from ..world_model.wm_trainer import WorldModelTrainer
from ..world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig): # 训练器初始化
        self.cfg = cfg # 保存训练配置
        cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0) # 计算目标输入维度
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs # 同步锚定观测配置
        self.rank, local_rank, self.world = init_distributed() # 初始化分布式训练
        self.is_main = self.rank == 0 # 判断是否为主进程
        if self.world > 1: # 多进程训练
            cfg.device = f"cuda:{local_rank}" # 指定当前进程的 GPU
            # 论文：双 GPU 学习率乘以根号2，各 GPU 独立维护4096个环境及缓冲区
            # 多 GPU 训练可能增大了有效 Batch Size，所以有时会相应调大学习率
            # 当独立样本数增加 \(N\) 倍时，平均梯度的随机噪声标准差通常约下降到原来的 \(1/\sqrt N\)。
            # 把学习率乘以 \(\sqrt N\)，可以在某些条件下补偿更新噪声的下降。
            cfg.wm_trainer.lr *= self.world**0.5 # 缩放世界模型学习率
            cfg.policy_trainer.lr *= self.world**0.5 # 缩放策略学习率
            cfg.policy_trainer.critic_lr *= self.world**0.5 # 缩放价值网络学习率
        torch.manual_seed(cfg.seed + 1000 * self.rank) # 设置各进程的随机种子
        self.device = cfg.device # 保存计算设备
        self.log_dir = Path(cfg.log_root) / cfg.run_name # 构造实验日志路径
        self.log_dir.mkdir(parents=True, exist_ok=True) # 创建日志目录
        if self.is_main: # 仅主进程保存配置
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str)) # 保存配置为JSON
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None  # 创建TensorBoard记录器

        if cfg.data.library is not None: # 使用预打包动作库
            files = [cfg.data.library] # 获取动作库路径
            cfg.env.compact_library = True # 启用紧凑动作库
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest) # 查找NPZ数据
        if not files: # 检查数据是否为空
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(  # 创建仿真环境
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],  # 动作数据路径
                reward_cfg=cfg.rewards,
                num_envs=cfg.env.num_envs,  # 并行环境数
                future_frames=cfg.env.future_frames,  # 未来参考帧数
                episode_length_s=cfg.env.episode_length_s,  # 回合时长
                action_noise_std=cfg.env.action_noise_std,  # 动作噪声标准差
                obs_noise=cfg.env.obs_noise,  # 观测噪声
                reset_noise=cfg.env.reset_noise,  # 重置扰动
                domain_randomization=cfg.env.domain_randomization,  # 域随机化
                sampling_mode=cfg.env.sampling_mode,  # 采样模式
                replay_capacity=cfg.env.replay_capacity,  # 经验回放容量
                compact_library=cfg.env.compact_library,  # 紧凑动作库模式
                anchor_obs=cfg.env.anchor_obs,  # 锚定观测
                seed=cfg.seed + 1000 * self.rank,  # 环境随机种子
            ),
            device=self.device,  # 仿真设备
        )
        env = self.env # 环境局部引用
        self.command = env.command_manager.get_term("motion") # 获取运动指令模块
        self.lib = self.command.library  # 获取参考动作库
        self.recorder = env.recorder_manager.get_term("replay")  # 获取经验记录模块
        self.buffer = self.recorder.buffer  # 获取经验缓冲区
        robot = env.scene["robot"]  # 获取机器人对象

        fk = G1Kinematics(load_g1_model(), device=self.device)  # 创建G1正向运动学模块
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(), reward_cfg=cfg.rewards)  # 创建解析计算模块
        self.wm = WorldModel(cfg.world_model, heads, self.device)  # 创建世界模型
        self.policy = Policy(cfg.policy).to(self.device)  # 创建策略网络
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)  # 创建价值网络
        for m in (self.wm, self.policy, self.critic):  # 遍历三个模型
            broadcast_params(m)  # 同步分布式模型参数
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)  # 创建世界模型训练器
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)  # 创建策略训练器
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:  # 仅主进程输出参数量
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # 训练环境的回合统计（论文中的“训练回报”）
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)  # 各环境累计奖励
        self.ep_len = torch.zeros(env.num_envs, device=self.device)  # 各环境回合长度
        self.max_step_reward = heads.reward_cfg.max_step_reward  # 单步最大奖励
        self.iteration = 0  # 当前迭代次数
        self.start_iteration = 0  # 预热起点，恢复训练后重新填充缓冲区
        self.num_updates = 0  # 累计更新轮数
        self._utd_acc = 0.0  # 更新次数累加器
        if cfg.resume is not None:  # 检查恢复路径
            self.load(cfg.resume)  # 恢复训练状态

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,  # 当前环境迭代次数
                "num_updates": self.num_updates,  # 累计更新次数
                "wm": self.wm.state_dict(),  # 世界模型状态
                "policy": self.policy.state_dict(),  # 策略网络状态
                "critic": self.critic.state_dict(),  # 价值网络状态
                "wm_opt": self.wm_trainer.opt.state_dict(),  # 世界模型优化器状态
                "policy_opt": self.policy_trainer.opt.state_dict(),  # 策略优化器状态
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),  # 价值网络优化器状态
                "bin_return": self.command.bin_return,  # 采样区间回报统计
                "bin_memory": self.command.bin_memory,  # 采样区间历史记忆
                "config": asdict(self.cfg),  # 训练配置字典
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:  # 加载训练检查点
        ck = torch.load(path, map_location=self.device, weights_only=False)  # 读取检查点到指定设备
        self.wm.load_state_dict(ck["wm"])  # 恢复世界模型
        self.policy.load_state_dict(ck["policy"])  # 恢复策略网络
        self.critic.load_state_dict(ck["critic"])  # 恢复价值网络
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])  # 恢复世界模型优化器
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])  # 恢复策略优化器
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])  # 恢复价值网络优化器
        self.command.bin_return.copy_(ck["bin_return"])  # 恢复区间回报统计
        self.command.bin_memory.copy_(ck["bin_memory"])  # 恢复区间历史记忆
        self.iteration = ck["iteration"]  # 恢复迭代次数
        self.start_iteration = self.iteration  # 重置预热起点
        self.num_updates = ck["num_updates"]  # 恢复累计更新次数
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")  # 输出恢复信息

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:  # 执行训练主循环
        cfg = self.cfg  # 训练总配置
        sch = cfg.schedule  # 训练调度配置
        env = self.env  # 仿真环境
        obs, _ = env.reset()  # 重置环境并获取初始观测
        t_start = time.time()  # 记录训练开始时间
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}  # 各模块累计耗时
        agg: dict[str, float] = {}  # 训练指标累加器
        agg_n = 0  # 指标累计次数
        finished_returns: list[float] = []  # 已结束回合的回报
        finished_lens: list[float] = []  # 已结束回合的长度
        last_log = time.time()  # 上次日志时间

        #  ||环境交互与数据收集
        while self.iteration < sch.total_iterations: # 未达到总迭代次数则继续
            # ---------------- (1) 采集一次并行环境步
            t0 = time.perf_counter()  # 开始计时
            action = self.act(obs)  # 策略生成动作
            obs, rew, term, tout, extras = env.step(action)  # 执行动作并获取环境反馈
            self.ep_reward_sum += rew  # 累加各环境奖励
            self.ep_len += 1  # 更新各环境回合长度
            done = term | tout  # 合并终止和超时标志
            if done.any():  # 存在已结束的环境
                ids = done.nonzero().squeeze(-1)  # 获取结束环境的索引
                lens = self.ep_len[ids]  # 获取对应回合长度
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))  # 计算归一化回报
                finished_returns.extend(norm_ret.tolist())  # 记录回合回报
                finished_lens.extend(lens.tolist())  # 记录回合长度
                self.ep_reward_sum[ids] = 0.0  # 清零结束环境的累计奖励
                self.ep_len[ids] = 0.0  # 清零结束环境的回合长度
            torch.cuda.synchronize()  # 等待GPU计算完成
            timers["sim"] += time.perf_counter() - t0  # 累计仿真耗时
            self.iteration += 1  # 环境迭代次数加一

            # ---------------- (2)+(3) 按照更新与数据比例执行模型优化 || 世界模型与策略更新
            if self.iteration - self.start_iteration >= sch.warmup_iterations:  # 完成预热后开始更新
                self._utd_acc += sch.updates_per_env_step  # 累加更新预算
                while self._utd_acc >= 1.0:  # 更新预算达到一次
                    self._utd_acc -= 1.0  # 消耗一次更新预算
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)  # 计算余弦衰减学习率
                    self.wm_trainer.set_lr(lr)  # 设置世界模型学习率
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)  # 按初始比例设置策略学习率
                    t0 = time.perf_counter()  # 开始世界模型更新计时
                    m_wm = self.wm_trainer.update(self.buffer)  # 使用经验缓冲区训练世界模型 ！！！训练世界模型
                    torch.cuda.synchronize()  # 等待GPU更新完成
                    timers["wm"] += time.perf_counter() - t0  # 累计世界模型训练耗时
                    t0 = time.perf_counter()  # 开始策略更新计时
                    for _ in range(sch.policy_updates_per_wm_update):  # 按配置执行策略更新
                        m_pi = self.policy_trainer.update(self.buffer)  # 更新策略及相关训练组件 ！！！ 执行策略训练
                    torch.cuda.synchronize()  # 等待GPU更新完成
                    timers["policy"] += time.perf_counter() - t0  # 累计策略训练耗时
                    self.num_updates += 1  # 外层更新轮数加一
                    for k, v in m_wm.items():  # 遍历世界模型指标
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v  # 累加世界模型指标
                    for k, v in m_pi.items():  # 遍历策略训练指标
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v  # 累加策略指标
                    agg["lr"] = lr  # 记录当前学习率
                    agg_n += 1  # 指标累计次数加一

            # ---------------- logging / checkpoints
            # 日志与检查点
            if self.iteration % sch.log_interval == 0:  # 达到日志记录间隔
                it = self.iteration  # 当前迭代次数
                elapsed = time.time() - t_start  # 累计训练时间（秒）
                if self.world > 1:  # 多进程训练
                    # 所有进程必须执行相同的集合通信，无条件归约回报总和与回合数量
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world  # 汇总各进程回报
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world  # 汇总完成回合数
                    finished_returns = [tot / cnt] if cnt > 0 else []  # 计算全局平均回报
                    n_finished = int(round(cnt))  # 全局完成回合数
                else:
                    n_finished = len(finished_returns)  # 单进程完成回合数
                if not self.is_main:  # 非主进程
                    agg, agg_n = {}, 0  # 清空累计训练指标
                    finished_returns, finished_lens = [], []  # 清空回合统计
                    timers = {k: 0.0 for k in timers}  # 重置计时器
                elif agg_n > 0:  # 主进程存在更新记录
                    for k, v in agg.items():  # 遍历训练指标
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)  # 写入TensorBoard
                if not self.is_main:  # 非主进程
                    tr = float("nan")  # 回报设为无效值
                elif finished_returns:  # 存在已完成回合
                    tr = sum(finished_returns) / len(finished_returns)  # 计算平均归一化回报
                    self.writer.add_scalar("train/episode_norm_return", tr, it)  # 记录回合回报
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)  # 记录平均回合长度
                else:
                    tr = float("nan")  # 无已完成回合
                with torch.no_grad():  # 禁用梯度计算
                    if self.is_main and self.buffer.size > 2048:  # 主进程且缓冲区数据充足
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)  # 计算8步开环预测误差
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)  # 记录单步预测误差
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)  # 记录第8步预测误差
                    else:
                        err = None  # 不执行误差评估
                if self.is_main:  # 仅主进程记录日志
                    for k, v in timers.items():  # 遍历各模块耗时
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)  # 记录平均每步耗时
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)  # 记录训练小时数
                    self.writer.add_scalar("time/updates", self.num_updates, it)  # 记录累计更新次数
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)  # 记录采样熵
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)  # 记录采样区间平均回报
                    dt_log = time.time() - last_log  # 计算日志间隔耗时
                    last_log = time.time()  # 更新日志时间
                    msg = (  # 构造训练日志
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)  # 立即输出日志
                agg, agg_n = {}, 0  # 清空累计训练指标
                finished_returns, finished_lens = [], []  # 清空回合统计
                timers = {k: 0.0 for k in timers}  # 清空计时器
            if self.iteration % sch.checkpoint_interval == 0:  # 到达检查点保存间隔
                self.save("ckpt_latest.pt")  # 更新最新检查点
                self.save(f"ckpt_{self.iteration:07d}.pt")  # 保存带迭代编号的检查点
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:  # 定期检查时间限制
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)  # 判断是否超时
                if self.world > 1:  # 多进程训练
                    stop = allreduce_mean(stop)  # 所有进程归约停止标志
                if stop > 0:  # 达到停止条件
                    if self.is_main:  # 主进程输出提示
                        print("[INFO] max_hours reached, stopping")  # 输出时间限制提示
                    break  # 退出训练循环
        self.save("ckpt_latest.pt")  # 保存最终检查点
        if self.writer is not None:  # 存在日志记录器
            self.writer.flush()  # 刷新TensorBoard缓存
        finalize()  # 清理分布式训练资源
