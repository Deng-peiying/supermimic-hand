# SuperMimic-Reproduced

# 中文版

从零复现 **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)，CoRL 2026），目标机器人 Unitree G1，底座 [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp。论文没有公开代码，这里的动作库、可微 FK、世界模型、BPTT 训练器、基于回报的重置采样、多卡训练、Appendix F 评测全部按论文自己实现，不使用 rsl_rl。

开头的 GIF：同一个策略，按论文的观测口径在 BONES-SEED 全量（12.99 万条）上训练，四卡 RTX 5090 共 7.1 h。实体 = 仿真中的策略（MuJoCo Warp，Appendix F 的确定性协议），蓝色半透明 = 参考动作。片段依次为：高跳、趴着起身、原地 360° 旋转、单腿跳、慢跑接 270° 转体跳再趴下爬行、尺蠖爬行。18 段的完整版在 [`media/g1_boneseed_showcase.mp4`](media/g1_boneseed_showcase.mp4)。

## 主要结果

评测协议为论文 Appendix F：每条 clip 从首帧整条确定性播放，无噪声、无域随机化；**归一化回报** = Σ每步奖励 / (0.1 × 步数)，1.0 为完美跟踪；"失败" = 训练终止条件触发。

| 基准 | 本仓库 | 论文 |
|---|---|---|
| **BONES-SEED 全量，论文观测**（4×RTX 5090，7.1 h ≈ 28 GPU·h，10 万次迭代） | **0.930**，**145 / 129,491 失败（0.11%）** | >0.90，25 / 129,514（0.02%），1×H100 3 h；2×H100 9.8 h 为 0 失败 |
| LAFAN1 40 条整条（100–300 s），论文观测（1×5090） | 0.56–0.59，3–9 / 40 失败（三次独立 run） | 0.82，1×H100 1.2 h（Fig. 4a） |
| LAFAN1 40 条整条，+ 锚点相对观测（1×5090，3.3 h） | 0.904，3 / 40 失败 | 不可比（多了观测项） |
| 世界模型 8 步开环归一化误差 | 0.0065–0.0099 | 0.007–0.011 |

其他：4096 环境在一张 5090 上约 2×10⁵ env-steps/s；编译后一次想象 fwd+bwd（批 2048，H=32）0.33 s；四卡数据并行（显式梯度 all-reduce，无 NVLink）效率 91%。

上面的数字建立在三个结论上：

- **纯本体观测在长 clip 上会漂移。** 论文 Table A.1 的策略观测没有状态估计。在整条 100–300 s 的 LAFAN1 上，偏航误差约 9 s 内超过 15°，之后世界系的根部与速度奖励项拿不到，整条回报在三个独立 run 里一致封顶 ~0.58；加上 BeyondMimic 的锚点相对位姿（goal 组追加 9 维）可到 0.904。BONES-SEED 的 clip 很短（中位 5.8 s），同样的设计能到 0.93。因此正式结果以论文观测口径的 BONES-SEED 为准，锚点观测的 run 标为不可比。
- **论文的名字过滤包含杂技类。** 最初的过滤保留了 SONIC 关键词表里的杂技项（cartwheel、handstand、safety roll 等），得到 129,943 条而论文是 129,514 条；这 452 条贡献了 495 个失败中的 350 个（handstand 134/134、cartwheel 108/120）。加回关键词后为 129,491 条（与论文差 23 条）、失败率 0.11%，过滤脚本已同步。
- **剩余差距集中在翻转类。** 145 个失败里 54 条 flip 全部失败（论文 3 h 后 44 条 flip 只失败 1 条）。训练回报在 0.81 平台、学习率尚未衰减（计划 32 万次迭代，10 万次即停），延长训练让余弦衰减生效是最直接的下一步。

正式检查点（10 万次迭代，223 MB）挂在 [GitHub release `e8-boneseed-100k`](https://github.com/CuboiLeo/SuperMimic-Reproduced/releases/tag/e8-boneseed-100k)，可直接用于 `scripts/eval_benchmark.py`、`scripts/play.py` 和录制脚本。

每个实验的配置、曲线、评测、失败构成和踩坑见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 实现内容

论文章节 → 代码（全部 torch）：

| 论文 | 实现 |
|---|---|
| Appendix E 数据：LAFAN1（2.45 h）与 BONES-SEED（14.2 万 CSV → 名字 + 可行性过滤后 12.99 万条 / 264 h） | `scripts/data/csv_to_npz_batch.py`、`filter_bones_seed.py`、`build_library.py`；`src/supermimic/data/`（自己的向量化 slerp：mjlab 的在相邻帧几乎相同时提前返回、姿态滞后一帧） |
| 多 clip 动作库、K=10 未来帧目标、1 s 分桶的回报重置采样（50% 均匀 / 40% softmax(−25 G/Gmax) / 10% 记忆） | `src/supermimic/envs/mdp/motion_library.py`（fp16 紧凑库常驻显存、在线 FK）、`commands.py` |
| Table A.1–A.4 观测、BeyondMimic 8 项奖励、终止（\|Δh\| > 0.25 m 或 tilt > 0.8）、sim context ξ = 摩擦 / 关节偏置 / 躯干质心 | `src/supermimic/envs/mdp/{observations,rewards,terminations}.py`、`sim_context.py`、`state_io.py` |
| 可微 G1 FK（位置与速度），fp64 下与 MuJoCo 一致到 1e-10 | `src/supermimic/kinematics/g1_fk.py` |
| 世界模型：朝向规范化特征上的 MLP 动力学 + 显式积分、学习式终止头、经 FK 的**解析**奖励/观测头；k=8 自回归损失（Eq. 1） | `src/supermimic/world_model/` |
| 想象：H=32 BPTT 策略更新（生存加权 + critic 自举，Eq. 2）、TD(λ=0.95) critic（Eq. 3）、`torch.compile` 的 fwd+bwd | `src/supermimic/imagination/` |
| Algorithm A.1 主循环、UTD 0.4、Table A.5/A.6 超参、余弦学习率、torchrun 数据并行（每 rank 一个仿真器、显式 all-reduce、lr × √卡数） | `src/supermimic/training/loop.py`、`src/supermimic/scripts/train.py` |
| Appendix F 评测、推力测试、漂移诊断、交互回放 | `src/supermimic/eval/benchmark.py`、`scripts/eval_benchmark.py`、`eval_push.py`、`diag_drift.py`、`play.py` |

论文未指明之处（U1–U20，见 [`docs/PAPER_SPEC.md`](docs/PAPER_SPEC.md)）和设计决策（D1–D20，见 [`docs/REPRODUCTION_PLAN.md`](docs/REPRODUCTION_PLAN.md)）都记录了理由；实测后改动的决定在实验日志里。未做：速度接口与蒸馏、PPO 基线、H×critic 消融、ONNX 导出、真机。

## 环境、数据、训练、评测

命令与英文版相同，见上文 Setup / Data / Training / Evaluation and media 四节。要点：

- Python 3.12 + `uv`；固定 torch 2.14.1+cu130、mujoco 3.16、warp 1.18、mjlab `8bb2bc9`；在 RTX 5090（驱动 580）上验证。系统 `PYTHONPATH` 若带有其他 Python 包（例如 ROS），命令前加 `env -u PYTHONPATH`。
- 数据不随仓库分发：LAFAN1 来自 HF `lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/`；BONES-SEED 来自 HF `bones-studio/seed` 的 `g1.tar.gz`，转 50 Hz npz → 过滤 → fp16 紧凑库（6.7 GB，常驻显存）。
- 训练默认超参按 Table A.5/A.6；多卡用 `torchrun --nproc_per_node 4 -m supermimic.scripts.train`，一进程一卡。
- 录制与渲染：`scripts/media/record_rollout.py` 按 Appendix F 协议记录每步的机器人与参考 qpos，`scripts/media/render_rollout.py` 用 MuJoCo 离屏渲染（实体 = 策略，半透明 = 参考），`--segments` 选时间窗、`--titles` 写标题、`--concat` 拼接。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/PAPER_SPEC.md](docs/PAPER_SPEC.md) | 论文全部数值、公式、协议的逐条转录；"未指明"清单 U1–U20 |
| [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md) | 复现计划：目标、环境、设计决策 D1–D20、repo 骨架、分阶段步骤、测试、预算、风险 |
| [docs/REVIEW_LOG.md](docs/REVIEW_LOG.md) | 计划的自审与两轮对抗审查记录 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验日志：每个实验的配置、曲线、评测、偏离计划的决定、踩坑记录 |

## 工程备注

- mjlab 的 `csv_to_npz` 四元数 slerp 在相邻帧几乎相同时会提前返回（姿态滞后一帧），本仓库用自己的向量化 slerp（与 scipy 一致到 1e-9）。
- MuJoCo Warp 的 solver 溢出 printf 是编译期常量，必须在 `put_model` 后、图捕获前关闭，否则早期策略每步触发、仿真慢 10×。
- torchrun 下 MuJoCo Warp 总在第一张可见卡上，`scripts/train.py` 按 rank 设置 `CUDA_VISIBLE_DEVICES`。
- 续训时回放缓冲区的预热相对续训点重新计数（正式 run 曾因误杀续训过一次）。

## 致谢与许可

- 方法：Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath, *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, CoRL 2026, arXiv:2610.28480。本仓库为独立复现，与作者无关。
- mjlab、MuJoCo、MuJoCo Warp 按各自许可原样使用。LAFAN1 动作（Ubisoft La Forge，CC BY-NC-ND 4.0）、其 G1 重定向数据和 BONES-SEED 数据均不随仓库分发；演示视频中的参考动作仅作展示，不在本仓库许可范围内。
- 代码：[MIT](LICENSE)，Copyright (c) 2026 Yuxi Liu。
