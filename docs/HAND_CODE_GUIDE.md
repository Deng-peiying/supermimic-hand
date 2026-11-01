# Prima1 HandMimic：从配置读到奖励

当前训练机制与4096环境配置见 [完整SuperMimic转笔训练](PRIMA1_SUPERMIMIC_4096.md)。

多任务目录与扩展方法见 [框架说明](ARCHITECTURE.md)。任务实现位于 `tasks/pen_spin/`，旧目录转发源码已移除。
共享网络在 `algorithms/models.py`，任务 `policy/nets.py` 和 `world_model/world_model.py` 分别组装 Actor 与 WM。
与 G1 的逐项目录对照和新增任务步骤见 [任务开发指南](TASK_DEVELOPMENT.md)。

## 先回答：默认继承了什么？

**Prima1 没有继承任何 G1 或 mjlab tracking 奖励。**
`PenSpinEnvCfg` 是独立 dataclass；`pen_spin_rewards()` 返回的 6 项就是全部奖励。
`hold`、`short`、`full` 不再添加或覆盖奖励，只改变参考时钟及阶段长度。

这里参考 mjlab 的组织方式：**一个 term 一个函数；配置明确写 func、weight、params；管理器求和并记录各项**。
训练使用`ParallelHandEnv`（MuJoCo Warp），`HandEnv`（单环境CPU MuJoCo）用于独立交叉评估。
因此 `TensorRewardManager` 是适配真实状态/WM 预测张量的独立管理器，**不是 mjlab 管理器的子类**；
两种环境共用同一组张量奖励。

## 推荐阅读顺序

| 文件（相对仓库） | 只负责什么 | 在哪里改 |
|---|---|---|
| `src/supermimic/tasks/pen_spin/config.py` | 训练超参数与 CLI | 改采集量、batch、更新次数 |
| `src/supermimic/tasks/pen_spin/reward_cfg.py` | 完整奖励列表 | 增删 term、改权重和误差尺度 |
| `src/supermimic/tasks/pen_spin/envs/pen_spin_env_cfg.py` | 终止阈值、阶段时长、奖励组装 | 修改阶段与终止配置 |
| `src/supermimic/tasks/pen_spin/envs/mdp/rewards.py` | 每项奖励的无权重公式 | 修改某一项数学定义 |
| `src/supermimic/core/rewards.py` | 调用函数、乘权重、可选乘 dt、相加 | 通常无需修改 |
| `src/supermimic/tasks/pen_spin/envs/mdp/terminations.py` | 想象轨迹的可微存活权重 | 调整平滑终止公式 |
| `src/supermimic/tasks/pen_spin/envs/env.py` | 真实 MuJoCo reset/step、硬终止 | 修改物理交互 |
| `src/supermimic/tasks/pen_spin/envs/mdp/observations.py` | 仅本体历史 | 修改策略输入历史 |
| `src/supermimic/tasks/pen_spin/policy/nets.py` | Actor 组装 | 修改策略网络选型 |
| `src/supermimic/tasks/pen_spin/world_model/world_model.py` | WM 组装与状态尺度 | 修改动力学模型选型 |
| `src/supermimic/tasks/pen_spin/world_model/wm_trainer.py` | WM 监督更新与多步验证 | 修改 WM 学习流程 |
| `src/supermimic/tasks/pen_spin/imagination/rollout.py` | WM 中推进状态并积累奖励 | 修改想象逻辑 |
| `src/supermimic/tasks/pen_spin/imagination/policy_trainer.py` | 冻结 WM，反传更新 Actor | 修改策略目标 |
| `src/supermimic/tasks/pen_spin/training/loop.py` | 采集 → WM 更新 → Actor 更新 → 真实评估 | 修改训练调度 |
| `src/supermimic/tasks/pen_spin/eval/benchmark.py` | 真实评估与分项指标 | 修改评估协议 |
| `src/supermimic/tasks/pen_spin/deployment/` | 导出与离线推理 | 修改部署接口 |
| `src/supermimic/tasks/pen_spin/train.py` | 命令分派 | 修改启动入口 |

`world_model.heads.mimic_reward` 是只返回总奖励的便捷函数；训练使用奖励管理器取得分项。

## 一项奖励怎么读

```python
"pen_position": TensorRewardTermCfg(
    func=mdp.pen_position_tracking_exp,
    weight=3.0,
    params={"std": 0.025},
)
```

含义：调用 `pen_position_tracking_exp(data, std=0.025)`，得到每个样本的原始值，乘 3 后加入总奖励。
函数原始值为 `exp(-||笔位置 - 参考位置||² / 0.025²)`。
误差为 0 时原始值是 1、加权贡献是 3；偏差 2.5 cm 时贡献是 `3/e`。

`func` 是函数引用，不会在构造配置时执行。`params` 是调用它时传入的关键字参数。
这里的 `std` 是奖励容差尺度，不是观测噪声。

## 全部奖励（没有未列出的默认项）

| 名称 | 原始值 | 权重 | 意义 |
|---|---|---:|---|
| `joint_position` | exp(-关节位置平方误差均值 / 0.15²) | +1 | 手势软跟踪，尺度 0.15 rad |
| `joint_velocity` | exp(-关节速度平方误差均值 / 3²) | +0.2 | 弱速度跟踪，尺度 3 rad/s |
| `pen_position` | exp(-笔中心距离平方 / 0.025²) | +3 | 主要操作目标之一，尺度 2.5 cm |
| `pen_axis` | exp(-(1-轴点积²) / 0.2) | +2 | 主要操作目标之一，忽略无标记笔两端与长轴自旋 |
| `pen_distance` | 笔中心距离平方，单位 m² | -20 | 离得较远时仍提供距离梯度 |
| `action_delta` | 相邻执行目标差除以 0.12 后的平方均值 | -0.02 | 减少目标跳变，不是力矩/电流惩罚 |

惩罚函数也返回正数，**负号来自负权重**。6 项加权后求和，无额外归一化或截断。
完美跟踪且动作不变时，加权和为 6.2；乘控制周期 0.02 后，每步最高正奖励为 0.124。

v4 显式 `scale_rewards_by_dt=True`，真实仿真和想象均只乘一次控制周期。
时间缩放与 G1/Dex 对齐，具体奖励项和权重仍为 Prima1 转笔任务定义。
要改变时间缩放需显式改配置并作为新实验，不能悄悄改后继续同一个 run。

## 这些不是额外奖励

- 10 cm 距离阈值：真实环境的失败/终止条件，只是掉笔的工程代理，不保证真实抓握成功。
- `envs/mdp/terminations.py`中的距离sigmoid仅保留为诊断函数；当前想象训练使用学习终止头。
- 0.99折扣和0.95 GAE lambda：Actor-Critic目标参数，不是reward权重。
- 关节限位、每步 0.12 rad 限幅、±0.15 rad 残差：动作约束，不是自动增加的奖励。
- MuJoCo 接触/重力/力矩限幅：物理计算。当前没有独立的触碰奖励、指腹接触奖励或自碰撞惩罚项。

笔的真值只用于 WM、仿真奖励和评估；Actor 仍只接收关节历史、历史目标、参考关节角与内部时钟。

## 看完整配置与每项贡献

WSL 中无需加载资产就能列出配置：

```bash
cd /mnt/d/Prima1/supermimic-hand
PYTHONPATH=src /root/.venvs/supermimic-hand/bin/python -m supermimic.scripts.run prima1_pen_spin describe --stage short
```

输出中 `inherits_rewards_from: []` 明确表示无父任务奖励。
每个训练/评估目录会保存 `task_config.json`：稳定函数身份、权重、参数、dt 缩放和终止配置。
六项函数身份保留目录迁移前的字符串以兼容已有检查点；实际源码在 `envs/mdp/rewards.py`，不通过字符串动态导入。
`evaluation.json`、`baseline.json` 和训练 `metrics.jsonl` 内的 `evaluation` 包含：

- `reward_raw_mean`：公式原始值的每步均值。
- `reward_weighted_mean`：乘权重及可选 dt 后的每步贡献；各项之和等于 `mean_reward`。
- `reward_term_return`：每项整个回合的加权累计值；各项之和等于 `return`。

修改奖励配置后，`--resume` 和普通检查点评估会拒绝不匹配配置；如需沿用网络，用 `--initialize` 开始新实验。
旧检查点没有记录任务配置，不能验证同任务恢复，也需用 `--initialize`。旧 Actor 权重和离线导出结构没有变化。

## G1 为什么看起来像“继承”

G1 的 `src/supermimic/tasks/motion_tracking/envs/tracking_env_cfg.py` 也不是先加载一个完整 tracking 任务再自动加 reward。
它显式构造 `rewards` 字典，然后交给 `ManagerBasedRlEnvCfg`：

- 6 个运动跟踪函数来自 `mjlab.tasks.tracking.mdp`，本仓库显式指定权重和参数。
- 动作变化、关节限位两个函数来自 `mjlab.envs.mdp`。
- 没有注册自碰撞项，所以没有自碰撞奖励；文件顶部“移除”描述的是相对上游方案的差异，不是运行时隐藏操作。
- G1 世界模型想象还使用本仓库 `envs/mdp/rewards.py` 的可微对应公式；两处配置需保持一致。

Prima1 本次使用同一份张量公式同时服务评估和想象，避免再维护两份容易不一致的奖励。
5090 的 G1 checkout 和训练进程没有同步本次本地修改。

## 本次验证范围

11 项 Hand 测试通过，覆盖旧公式数值和 BPTT 梯度一致性、显式 dt 缩放、无笔观测泄漏、
自由笔物理步进、奖励分项求和、配置变更保护和离线部署接口。
这次是组织结构与可观测性改进，没有增加新的操作奖励，也不代表已经训练出稳定转笔策略。

本地 `runs/hand/reward-refactor-20261101` 完成 hold 短训练及同阶段恢复，
独立评估在 `runs/hand/reward-refactor-eval-20261101`，与训练末尾评估一致。
两轮共 128 条采集转移、20 次 WM 更新和 2 次 Actor 更新；0.36 秒触发距离失效阈值。
只验证训练/恢复/评估链路，没有据此启动长训练或宣称改善。
