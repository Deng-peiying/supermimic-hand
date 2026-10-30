"""记录确定性策略执行轨迹（Policy Rollouts），用于后续渲染。
在每个控制时间步，记录机器人的关节位置（robot qpos）
以及参考动作的关节位置（reference qpos）。

运行命令：
  uv run python scripts/media/record_rollout.py --ckpt logs/e8/ckpt_0100000.pt --npz-dir data/npz/bones_seed \
      --clips cartwheelin_003__A486_M jump_ff_360_R_004__A407 --out media/rollouts.npz

采用与附录 F 基准测试（scripts/eval_benchmark.py）相同的实验协议：

1. 每个动作片段都从第一帧开始执行。
2. 不添加观测噪声或动作噪声。
3. 不添加环境重置噪声。
4. 不进行域随机化（Domain Randomization）。
5. 每个动作片段仅执行一次。
6. 当动作片段播放结束或触发终止条件时，停止记录。

使用 scripts/media/render_rollout.py 渲染记录结果。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import NQ, StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint


@torch.no_grad()
def main(ckpt: Path, clips: list[str], npz_dir: Path = Path("data/npz/bones_seed"), out: Path = Path("media/rollouts.npz"), device: str = "cuda:0") -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    missing = set(clips) - {Path(f).stem for f in files}
    assert not missing, f"clips not found in {npz_dir}: {sorted(missing)}"
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    n = len(files)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=n, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        env_clip = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[env_clip]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[env_clip]  # keep the clip if an env is reset again
        max_r = 0.1  # paper reward upper bound per step (dt * 5)
        rob, ref = [], []
        alive = torch.ones(n, dtype=torch.bool, device=device)
        rew_sum = torch.zeros(n, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)

        def snapshot() -> None:
            rob.append(io.read()[:, :NQ].cpu().numpy())
            ref.append(lib.ref_qpos_qvel(cmd.frame)[0].cpu().numpy())

        snapshot()
        while alive.any():
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            rew_sum += rew * alive
            steps += alive.long()
            done = (term | tout) & alive
            alive &= ~done
            if alive.any():
                snapshot()
            if steps.max() > 20000:
                break
        names = [lib.clips[i].name for i in range(n)]
        lengths = steps.cpu().numpy()  # recorded frames = steps (the terminal/reset state is not recorded)
        T = len(rob)
        robot_qpos, ref_qpos = np.stack(rob, 1), np.stack(ref, 1)
        for i in range(n):  # freeze frames after a clip's end so every row has the same length
            robot_qpos[i, lengths[i]:] = robot_qpos[i, lengths[i] - 1]
            ref_qpos[i, lengths[i]:] = ref_qpos[i, lengths[i] - 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        norm_return = (rew_sum / (max_r * steps.clamp(min=1))).cpu().numpy()
        np.savez_compressed(out, names=np.array(names), robot_qpos=robot_qpos[:, :T], ref_qpos=ref_qpos[:, :T], lengths=lengths, norm_return=norm_return)
        for i, nm in enumerate(names):
            print(f"{nm:48s} steps {int(lengths[i]):5d}/{lib.clips[i].length}  return {float(norm_return[i]):.3f}")
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
