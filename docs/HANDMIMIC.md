# Prima1 左手：单片段无视觉转笔实验

本分支是第一轮可执行的模型式模仿实验，不是已经学会转笔的策略。G1 使用 `sm-run g1_tracking`，Prima1 使用 `sm-run prima1_pen_spin`。
基线提交 `1001aa9` 的备注是“添加中文注释”，新增代码在 `hand` 分支。

多任务目录和扩展方式见 [框架说明](ARCHITECTURE.md)，任务实现位于 `tasks/pen_spin/`，旧目录转发源码已移除。

**代码阅读入口：[Prima1 配置与奖励指南](HAND_CODE_GUIDE.md)。**
奖励已按 mjlab 的 `func / weight / params` 风格拆分；没有继承 G1 奖励。
全部 6 项在 `src/supermimic/tasks/pen_spin/reward_cfg.py`，公式在 `tasks/pen_spin/envs/mdp/rewards.py`。
运行 `python -m supermimic.scripts.run prima1_pen_spin describe --stage short` 可查看完整生效配置。

## 1. 先只用这一段

数据：`../datasets/prima1_left_pen_spin/candidates/round3_kinematic_clip001/candidate.npz`。
Windows 完整位置：`D:\Prima1\datasets\prima1_left_pen_spin\candidates\round3_kinematic_clip001\candidate.npz`。

- 来源是 Wuji `01_single_axis/clip_001.npz` 重定向到 **Prima1 左手**的 round3；不是人手数据。
- 369 帧、50 Hz、首末帧跨度 7.36 秒，22 个关节；位姿在固定手掌基座坐标系，四元数 wxyz。
- SHA256：`6833ef6510be13abd904096d8ce592bb112411d2c4ae83b2d887009fa2b8b619`。
- 唯一目标资产：`../Assets-repo/xynova/Prima1/left/prima_left_after_1.1.urdf`，不使用 Studio 旧模型。
- 加载时校验候选、原 URDF、全部 STL、审查版 MJCF 哈希，并检查 `review_status.json` 的哈希绑定训练许可。
- 依据用户“我看你重定向候选里没有问题”及后续实验授权，记录为**允许单片段仿真实验**。
  原元数据中的 `approved:false` 是生成时状态；后续人工验收以 `review_status.json` 为准。
  几何认可不等于动力学可行、训练成功或硬件许可。
- 不再批量转换，也不需要下载新数据集。数据与大网格保持在原目录，不重复提交到代码仓库。
  移机时需要复制整个候选目录和左手资产目录，使用 `--clip`、`--asset` 指定路径。

单片段是最低成本的可行性实验，不能证明对其他放笔位置、笔型号或未知运动的泛化。
也不是一个可直接无限循环的数据：末帧到首帧未验证连续性，部署接口到片段末尾即停止。

## 2. 什么叫“无外部定位”

| 部分 | 允许输入 |
|---|---|
| 部署 Actor | 最近 30 步关节位置、位置差分速度、历史执行目标；固定片段的参考关节角与内部时钟 |
| 世界模型 | 仿真手关节与笔完整状态、执行目标 |
| 仿真奖励与评估 | 笔位置/轴方向、参考动作、手关节状态 |
| 离线部署包 | Actor、参考关节序列和接口说明；不含笔轨迹或世界模型 |

历史覆盖 0.6 秒。速度通过位置差分得到，不要求 SDK 单独提供准确速度反馈；首帧历史以静止状态填充。
这仍是“按已知参考时钟执行”的任务，不是完全无任务时钟的通用自发转笔策略。
无笔观测意味着无法直接判定真机是否掉笔，也无法保证从任意未知放笔姿态启动；先人工放笔到指定姿态。

**本地 Dex 的重要区别**：`src/dex_supermimic/rl/dex_supermimic/pen.py` 的
`PenMDP.observation()` 包含当前笔轴端点；`camera_enabled=False` 时仍返回这一观测。
因此仅关闭相机噪声不满足本实验条件。本分支 Actor 接口完全独立，不传入笔状态。

## 3. 已实现的训练方法

1. 用真实 MuJoCo 接触仿真采集转移，控制周期 20 ms、物理子步 2 ms。
2. 用8步自回归监督学习拟合57维特权状态的显式积分世界模型，同时训练终止头。
3. 冻结WM/Critic参数，从真实回放初态通过32步BPTT更新Actor，含末端价值自举；再用GAE目标拟合Critic。
4. 并行采集记录真实回报，独立CPU评估从首帧比较参考角回放与学习策略。

Actor 是历史 MLP，输出参考关节角上最多 ±0.15 rad 的残差；每控制周期目标变化最多 0.12 rad，
并使用 URDF 关节限位。环境执行同样的限幅，回放保存实际执行目标。
自由笔只在 reset 初始化，step 中不瞬移回参考位置；结束状态与下一次 reset 不拼成训练转移。

现已保留完整SuperMimic训练机制并适配Prima1：Warp并行接触采集、学习终止、Actor/Critic、
GAE自举、序列回放与UTD调度。具体配置和后端验证边界见 [4096环境训练说明](PRIMA1_SUPERMIMIC_4096.md)。
没有学习接触头或在线域适应，不能把仿真训练启动理解为已学会转笔或已完成真机验证。

### 奖励

当前为关节位置跟踪、较弱的关节速度跟踪、笔中心位置、笔长轴方向、动作目标变化惩罚和距离惩罚。
笔位置与长轴方向权重分别为 3 和 2；关节位置为 1、速度为 0.2。
圆柱绕自身长轴的旋转不奖励，两个无标记端点视为等价，避免对物理不可辨别姿态做过度约束。
关节参考只作为软约束，允许策略为了稳定笔而偏离重定向动作。

当前不把重定向文件里的距离接触假设当成真实接触标签，也未加入指腹 FK、穿透或持续电流的学习奖励。
MuJoCo 会计算真实碰撞约束并限制执行器力矩，但这不能代替未来的碰撞/负载评估。
后续应先校准指腹位置与驱动参数，再加入手指相对笔的几何奖励、力矩负载和手指自碰撞代价。
不能因为手势与参考一致就认定物体操作成功。

### 评估含义

固定从第 0 帧、零初速度启动，笔中心距离参考超过 10 cm 记为 `dropped`（工程上的失效代理）。
报告存活时长、平均/最大位置误差、平均轴误差、累计回报与是否无掉笔到达阶段末尾。
`completed_without_drop` 只表示未触发该距离阈值，不自动证明准确完成转笔。
平均每步奖励可能因早停而虚高，必须同时看累计回报、存活时长和轨迹。

训练日志的nse1/nseK来自回放训练窗口，不是独立测试集成绩。还需结合真实仿真首帧跟踪和
额外收集的轨迹判断WM误差，不能仅因想象回报上涨就认定控制改善。

## 4. 在 WSL 里运行

沿用 Linux 虚拟环境，别在 Windows 项目根目录重新创建 `.venv`。以下均在 Ubuntu-22.04 执行：

```bash
cd /mnt/d/Prima1/supermimic-hand
export UV_PROJECT_ENVIRONMENT=/root/.venvs/supermimic-hand
export PYTHONPATH=src
PY=/root/.venvs/supermimic-hand/bin/python

# 首先复查接口和物理步进
$PY -m pytest tests/test_hand.py -q

# 先做一个有界的 hold 实验：固定首帧参考，最长 2 秒
$PY -m supermimic.scripts.run prima1_pen_spin train --stage hold \
  --output runs/hand/hold-001 --iterations 200 --num-envs 64 --device cuda:0

# 查看持笔结果；成功后再进入前 1 秒动态片段
$PY -m supermimic.scripts.run prima1_pen_spin evaluate --stage hold \
  --checkpoint runs/hand/hold-001/checkpoint.pt --output runs/hand/hold-eval

$PY -m supermimic.scripts.run prima1_pen_spin train --stage short \
  --initialize runs/hand/hold-001/checkpoint.pt \
  --output runs/hand/short-001 --iterations 200 --num-envs 64 --device cuda:0

# 前 1 秒稳定后再尝试完整 7.36 秒，不能直接把短片成功外推到全片
$PY -m supermimic.scripts.run prima1_pen_spin train --stage full \
  --initialize runs/hand/short-001/checkpoint.pt \
  --output runs/hand/full-001 --iterations 200 --num-envs 64 --device cuda:0

$PY -m supermimic.scripts.run prima1_pen_spin evaluate --stage full \
  --checkpoint runs/hand/full-001/checkpoint.pt --output runs/hand/full-eval
```

这些命令是顺序实验阶段，**不是建议不看结果一次运行完**。hold 不稳定时先检查可达抓握、接触和临时驱动参数。
`--initialize`加载兼容Actor开启新实验；`--resume`恢复网络、优化器和采样统计，然后重新填充回放。
`--iterations`是总环境迭代上限。输出已有checkpoint时，必须显式resume或换目录。
`--sampling return/uniform/start`选择起点采样方式；正式CPU评估始终从首帧启动。
训练需要CUDA GPU，CPU用于独立物理评估。

每个输出目录有：

- `checkpoint.pt`：训练状态，**仅加载可信本地文件**。
- `metrics.jsonl`：并行真实回报、失败次数、WM误差、Actor/Critic更新指标。
- `evaluation.json/.npz`：独立evaluate命令产生的指标和自由笔轨迹，状态排列见`envs/env.py`。
- `provenance.json`：哈希、人工验收范围及参数校准状态。
- `task_config.json`：完整奖励函数、权重、参数、时间缩放及终止配置；评估还记录各项原始均值、加权均值和累计贡献。
- `actor.ts`、`joint_reference.npz`、`actor_schema.json`：离线推理包，不是可直接上机的 SDK 程序。

新检查点保存任务配置；奖励/终止配置改变后拒绝 `--resume`，应使用 `--initialize` 建立新实验。
重构前的旧检查点缺少此配置记录，也使用 `--initialize`，不能直接作为已验证的同任务续训。

## 5. 本次实测与尚未解决的部分

2026-11-01，WSL Python 3.12 / Torch 2.14.1 / MuJoCo 3.16，CPU 有界测试：

| 实验 | 数据/更新量 | 结果 |
|---|---|---|
| short 零残差参考回放 | 从首帧 | 0.76 秒触发 10 cm 失效阈值 |
| short 最终代码冒烟训练 | 2 轮共 128 转移 | 0.60 秒失效；尚未优于基线 |
| hold 冒烟训练 | 40 转移，10 次 WM 更新、1 次策略更新 | 基线 0.34 秒，策略 0.40 秒失效；不能据此声称改善 |
| full 参考回放 | 从首帧 | 同样在 0.76 秒失效 |

7 项 Hand 接口/动力学/梯度测试通过，Ruff 检查通过；也单独验证了 checkpoint 恢复追加训练。
WSL 原先缺 C 编译器导致 G1 的 Torch/Triton 编译失败；更新 apt 索引并安装 `build-essential` 后，
选取的原 G1 世界模型/动作转换/紧凑动作库测试结果为 3 passed、4 skipped，日志在 `runs/hand/g1-regression.log`。
最终 short 的输出在 `runs/hand/smoke-final`；早期恢复测试在 `runs/hand/smoke`。
上述仅证明端到端代码、恢复和导出能运行，数据量远不足以判断最终学习可行性。
hold 的 4 步世界模型验证位置误差约 14.3 mm，说明模型仍很粗糙；短跑结果不可部署。
原审查视频使用 MuJoCo 3.17，训练环境为项目依赖的 3.16，接触数值差异应纳入后续一致性检查。
笔质量 30 g、长 22 cm、半径 7.5 mm 及 kp=1.5、kv=0.04、力矩限幅 ±0.25 Nm 都仍是临时值。

## 6. 论文与开源依据

- **Lessons from Learning to Spin “Pens”**，CoRL 2024：
  [项目与视频](https://penspin.github.io/)、[论文](https://arxiv.org/abs/2407.18902)、
  [官方代码](https://github.com/HaozhiQi/penspin)。
  项目列出的机构是 UCSD、CMU、UC Berkeley；本次未确认“Google 无视觉转笔”对应的确切论文。
  PenSpin 的部署学生使用本体历史；论文使用 30 步关节位置与历史目标。
  路线是特权 oracle → 仿真学生预训练 → 真机轨迹采集 → 少于 50 条成功轨迹微调，
  **不是仅靠仿真零样本成功迁移**，也不是这里的世界模型 BPTT 算法。
  多种 canonical 初始姿态对 finger gait 很关键，支持后续在单段验证后扩充起始抓握分布。
  官方公开的 `real_data.h5` 是机器人关节/目标轨迹，不是人手关键点转笔数据；当前无需下载。
- **General In-hand Object Rotation with Vision and Touch (RotateIt)**，CoRL 2023：
  [正式论文页](https://proceedings.mlr.press/v229/qi23a.html)。使用视觉、触觉和本体融合，
  可以参考训练设计，但不能拿它作为本任务“仅本体信息”的直接证据。
- **In-Hand Object Rotation via Rapid Motor Adaptation (HORA)**，CoRL 2022：
  [项目与视频](https://haozhi.io/hora/)、[论文](https://arxiv.org/abs/2210.04887)、
  [代码](https://github.com/HaozhiQi/hora)。作者机构为 Berkeley 与 Meta AI。
  使用本体历史进行快速适应，先训练带特权信息的基础策略，再训练仅本体历史的适应模块；
  从稳定抓握开始做物体转动，不能把它与细长笔的跨指连续翻转难度等同。
  它支持“先解决稳定初始抓握，再扩展旋转与物理参数适应”的实验顺序。
- 本地 **Dex-Supermimic** 的 `PenMDP`：参考了残差参考跟踪与物体优先奖励思路；
  本实现没有复制其带笔观测的策略接口或 Wuji 20 关节假设。

## 7. 到 Prima1 真机还缺什么

`supermimic.tasks.pen_spin.deployment.controller.OfflineController` 已提供**只返回目标、不发硬件命令**的推理接口。
要求显式填写每个 URDF 关节对应的 SDK 名称、方向/单位换算和零点；拒绝用猜测的关节索引控制手。
它检查首帧姿态、反馈数值/限位和时间戳，片段结束或反馈异常后停止生成新命令。
若调用本身完全中断，Python 接口无法主动停车，实际集成必须由 SDK/独立线程提供看门狗。
SDK 若有额外限幅，需要将真正执行目标反馈到历史；不能默认为返回值已执行。

上机顺序：

1. 实测自由度及耦合关系、SDK 顺序/方向/单位、关节零点、频率、延迟、目标限幅。
2. 无笔低速动作辨识，校准 kp/阻尼/力矩/摩擦；测量实际笔几何、质量和重心。
3. 在仿真加入基于测量范围的随机化，包括延迟、编码器噪声、摩擦、质量与初始放笔误差；
   做多个种子及不同初始状态评估，不能只报告一个确定性短片。
4. 人工放笔、固定手腕、保护区域内先验证持笔，再做短片；由人监护掉笔和紧急停止。
5. 记录真实 q、目标、时间戳及成功/失败标注，评估是否需要像 PenSpin 一样进行少量真机微调。

当前未接 SDK，未发送硬件动作，未声称完成上述标定、随机化训练或稳定连续转笔。
