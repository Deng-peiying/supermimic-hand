# SuperMimic-Reproduced

多任务框架、目录边界与扩展方法见 [架构说明](docs/ARCHITECTURE.md)。G1 与 Prima1 任务分别位于 `src/supermimic/tasks/motion_tracking/` 和 `src/supermimic/tasks/pen_spin/`，统一入口为 `sm-run`。旧目录转发模块和旧命令入口已移除。

两个任务按相同职责分包；查找修改入口、对照目录和新增任务，请从 [任务开发指南](docs/TASK_DEVELOPMENT.md) 开始。

Prima1 当前已接入完整 SuperMimic 训练机制与4096环境 GPU 接触仿真，配置、数据策略及验证边界见 [转笔训练说明](docs/PRIMA1_SUPERMIMIC_4096.md)。

Prima1 左手单片段无视觉转笔实验见 [HandMimic 使用说明](docs/HANDMIMIC.md)。该入口与下述 G1 复现独立，当前是已跑通的实验代码，尚未学会稳定转笔。

公共神经网络位于 `src/supermimic/algorithms/networks.py` 和 `models.py`；`robots/`
只定义机器人状态、运动学与约束，`tasks/` 负责配置与组装。两个任务不互相导入，默认独立训练权重。

## 新增任务：复制任务包

以 `src/supermimic/tasks/motion_tracking/` 为完整任务模板，复制为
`src/supermimic/tasks/my_task/`（只复制源码，不复制 `__pycache__`）。包内使用相对导入，
重命名后内部引用仍指向新任务。公共网络从 `supermimic.algorithms` 引用。

1. 在 `src/supermimic/tasks/registry.py` 注册
   `TaskSpec("my_task", "你的机器人", "mjlab", "supermimic.tasks.my_task.train:main")`。
2. 修改新任务引用的 `supermimic.robots.*`、机器人资产、关节和跟踪 body 配置。
3. 修改 `policy/nets.py` 的网络组装、观测和目标维度，以及 `world_model/state_spec.py`
   的状态编码、输出解释和积分。同步检查 `envs/state_io.py`、`envs/sim_context.py`
   和 `world_model/heads.py` 的布局、维度与缩放。
4. 修改 `envs/mdp/` 中的观测、动作参考、奖励、终止和 `reward_cfg.py`；
   对应更新想象过程使用的 `world_model/heads.py` 与 `imagination/` 任务逻辑。
5. 用 `sm-run my_task --help` 检查入口，再运行短训练。

任务包保留自己的 WM 状态语义、想象观测/奖励适配与训练组装；公共网络和回报算子无需复制。
当前 WM 多步损失、imagination rollout 和优化器循环仍有任务专用实现，尚未统一为公共训练引擎。
目前模板默认是 G1：同一机器人的新任务主要改 MDP；换成不同状态结构的机器人，不能只改维度。
现有 `scripts/` 下的 G1 数据转换与评测脚本也不会随任务包复制自动变成新任务入口。

待清理的临时产物统一放入 `laji/`，可手动清空其中内容。该目录已加入 Git 忽略列表。

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
| Appendix E 数据：LAFAN1（2.45 h）与 BONES-SEED（14.2 万 CSV → 名字 + 可行性过滤后 12.99 万条 / 264 h） | `scripts/data/csv_to_npz_batch.py`、`filter_bones_seed.py`、`build_library.py`；`src/supermimic/tasks/motion_tracking/data/`（自己的向量化 slerp：mjlab 的在相邻帧几乎相同时提前返回、姿态滞后一帧） |
| 多 clip 动作库、K=10 未来帧目标、1 s 分桶的回报重置采样（50% 均匀 / 40% softmax(−25 G/Gmax) / 10% 记忆） | `src/supermimic/tasks/motion_tracking/envs/mdp/motion_library.py`（fp16 紧凑库常驻显存、在线 FK）、`commands.py` |
| Table A.1–A.4 观测、BeyondMimic 8 项奖励、终止（\|Δh\| > 0.25 m 或 tilt > 0.8）、sim context ξ = 摩擦 / 关节偏置 / 躯干质心 | `src/supermimic/tasks/motion_tracking/envs/mdp/{observations,rewards,terminations}.py`、`sim_context.py`、`state_io.py` |
| 可微 G1 FK（位置与速度），fp64 下与 MuJoCo 一致到 1e-10 | `src/supermimic/robots/g1/kinematics.py` |
| 世界模型：朝向规范化特征上的 MLP 动力学 + 显式积分、学习式终止头、经 FK 的**解析**奖励/观测头；k=8 自回归损失（Eq. 1） | `src/supermimic/tasks/motion_tracking/world_model/` |
| 想象：H=32 BPTT 策略更新（生存加权 + critic 自举，Eq. 2）、TD(λ=0.95) critic（Eq. 3）、`torch.compile` 的 fwd+bwd | `src/supermimic/tasks/motion_tracking/imagination/` |
| Algorithm A.1 主循环、UTD 0.4、Table A.5/A.6 超参、余弦学习率、torchrun 数据并行（每 rank 一个仿真器、显式 all-reduce、lr × √卡数） | `src/supermimic/tasks/motion_tracking/training/loop.py`、`src/supermimic/tasks/motion_tracking/train.py` |
| Appendix F 评测、推力测试、漂移诊断、交互回放 | `src/supermimic/tasks/motion_tracking/eval/benchmark.py`、`scripts/eval_benchmark.py`、`eval_push.py`、`diag_drift.py`、`play.py` |

论文未指明之处（U1–U20，见 [`docs/PAPER_SPEC.md`](docs/PAPER_SPEC.md)）和设计决策（D1–D20，见 [`docs/REPRODUCTION_PLAN.md`](docs/REPRODUCTION_PLAN.md)）都记录了理由；实测后改动的决定在实验日志里。未做：速度接口与蒸馏、PPO 基线、H×critic 消融、ONNX 导出、真机。

## 环境、数据、训练、评测

以下命令在仓库根目录执行，GPU 训练使用 Linux / WSL 环境。

安装依赖并查看任务：

```bash
uv sync
uv run sm-run --list
uv run sm-run g1_tracking --help
uv run sm-run prima1_pen_spin --help
```

G1 单卡训练、多卡训练与评估（需先准备对应 NPZ 数据）：

```bash
uv run sm-run g1_tracking --run-name g1-lafan1 \
  --data.npz-dir data/npz/lafan1 --env.num-envs 4096

uv run torchrun --nproc_per_node 4 -m supermimic.scripts.run g1_tracking \
  --run-name g1-lafan1-4gpu --data.npz-dir data/npz/lafan1

uv run python scripts/eval_benchmark.py \
  --ckpt logs/g1-lafan1/ckpt_latest.pt --npz-dir data/npz/lafan1
```

Prima1 查看奖励配置、hold 阶段训练、同阶段续训与评估：

```bash
uv run sm-run prima1_pen_spin describe --stage hold
uv run sm-run prima1_pen_spin train --stage hold --output runs/hand/hold-001
uv run sm-run prima1_pen_spin train --stage hold \
  --resume runs/hand/hold-001/checkpoint.pt --output runs/hand/hold-001
uv run sm-run prima1_pen_spin evaluate --stage hold \
  --checkpoint runs/hand/hold-001/checkpoint.pt --output runs/hand/hold-eval
```

使用已有虚拟环境时，可用 `PYTHONPATH=src python -m supermimic.scripts.run` 替代
`uv run sm-run`。Prima1 资产准备、short/full 阶段与跨阶段初始化见
[HandMimic 使用说明](docs/HANDMIMIC.md)。旧 Hand 检查点若记录的是旧奖励函数路径，
使用 `--initialize` 开启新实验，不能直接当作新配置 `--resume`。

环境与数据要点：

- Python 3.12 + `uv`；固定 torch 2.14.1+cu130、mujoco 3.16、warp 1.18、mjlab `8bb2bc9`；在 RTX 5090（驱动 580）上验证。系统 `PYTHONPATH` 若带有其他 Python 包（例如 ROS），命令前加 `env -u PYTHONPATH`。
- 数据不随仓库分发：LAFAN1 来自 HF `lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/`；BONES-SEED 来自 HF `bones-studio/seed` 的 `g1.tar.gz`，转 50 Hz npz → 过滤 → fp16 紧凑库（6.7 GB，常驻显存）。
- 训练默认超参按 Table A.5/A.6；多卡用 `torchrun --nproc_per_node 4 -m supermimic.scripts.run g1_tracking`，一进程一卡。
- 录制与渲染：`scripts/media/record_rollout.py` 按 Appendix F 协议记录每步的机器人与参考 qpos，`scripts/media/render_rollout.py` 用 MuJoCo 离屏渲染（实体 = 策略，半透明 = 参考），`--segments` 选时间窗、`--titles` 写标题、`--concat` 拼接。

## 文档

### G1 训练日志字段

例如 `it 12500 | 0.71 h | 4.8 it/s | updates 4984 | train_ret 0.702 (n=3842) | wm nse1 0.0063 nse8 0.0133 | imag_ret 2.475 surv 0.999 | t sim/wm/pi 2.5/6.4/12.0s`：

| 字段 | 含义与判断方式 |
|---|---|
| `it` | 并行环境步进次数；一次迭代让各环境各走一步，不是回合数。 |
| `h` | 本次运行的墙钟小时数；续训重新计时。 |
| `it/s` | 最近日志区间的迭代速度，包含仿真与优化过程。 |
| `updates` | 累计完成的外层更新轮数；每轮包含世界模型更新以及配置数量的策略更新，不等于梯度步数。 |
| `train_ret` | 最近区间结束回合的归一化回报均值：每回合 `sum(reward)/(max_step_reward × steps)`。默认 `max_step_reward=0.1`，1 为完美跟踪尺度；0.702 不是 70.2% 成功率。 |
| `n` | 上述统计使用的结束回合数量，包含失败与时间截断；多卡时汇总各进程。 |
| `wm nse1` | 世界模型训练批次的一步归一化平方状态误差，越低越好；它不是百分比或单一物理单位的误差。 |
| `nse8` | 近期回放样本上，用实际动作连续开环预测至第 8 步的归一化平方误差；默认对应 0.16 秒后。与 nse1 的采样及统计口径不同。 |
| `imag_ret` | 想象时域内按存活权重加权的未折扣奖励和，不含 critic 自举，也未按时域长度归一化。默认 H=32 时，满分参考为 3.2。 |
| `surv` | 想象轨迹末端的平均预测存活权重，沿途存活概率累乘；不是实测成功率。 |
| `t sim/wm/pi` | 最近日志区间累计的仿真、世界模型优化、策略与 critic 优化耗时，单位秒，不是单步耗时。 |

`train_ret` 与 `imag_ret` 的起始状态、时域和归一化方式不同，不能直接比较。
判断训练是否改善应结合归一化回报、回合长度、失败率以及独立整段评估；仅凭几行日志不能判断收敛。

### 文档索引

| 文件 | 内容 |
|---|---|
| [docs/PAPER_SPEC.md](docs/PAPER_SPEC.md) | 论文全部数值、公式、协议的逐条转录；"未指明"清单 U1–U20 |
| [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md) | 复现计划：目标、环境、设计决策 D1–D20、repo 骨架、分阶段步骤、测试、预算、风险 |
| [docs/REVIEW_LOG.md](docs/REVIEW_LOG.md) | 计划的自审与两轮对抗审查记录 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验日志：每个实验的配置、曲线、评测、偏离计划的决定、踩坑记录 |

## 工程备注

- mjlab 的 `csv_to_npz` 四元数 slerp 在相邻帧几乎相同时会提前返回（姿态滞后一帧），本仓库用自己的向量化 slerp（与 scipy 一致到 1e-9）。
- MuJoCo Warp 的 solver 溢出 printf 是编译期常量，必须在 `put_model` 后、图捕获前关闭，否则早期策略每步触发、仿真慢 10×。
- torchrun 下 MuJoCo Warp 总在第一张可见卡上，`src/supermimic/tasks/motion_tracking/train.py` 按 rank 设置 `CUDA_VISIBLE_DEVICES`。
- 续训时回放缓冲区的预热相对续训点重新计数（正式 run 曾因误杀续训过一次）。

## 致谢与许可

- 方法：Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath, *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, CoRL 2026, arXiv:2610.28480。本仓库为独立复现，与作者无关。
- mjlab、MuJoCo、MuJoCo Warp 按各自许可原样使用。LAFAN1 动作（Ubisoft La Forge，CC BY-NC-ND 4.0）、其 G1 重定向数据和 BONES-SEED 数据均不随仓库分发；演示视频中的参考动作仅作展示，不在本仓库许可范围内。
- 代码：[MIT](LICENSE)，Copyright (c) 2026 Yuxi Liu。
