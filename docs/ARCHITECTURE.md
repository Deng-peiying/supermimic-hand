# 多任务框架与扩展入口

后续升级：Prima1现已接入完整WM/Actor/Critic训练和Warp并行环境，最新配置见
[转笔训练说明](PRIMA1_SUPERMIMIC_4096.md)。文末首轮重构测试记录是历史记录。

本仓库现在把机器人、任务、算法组件和仿真后端分开。G1 跟踪和 Prima1 转笔是并列任务。
旧目录的转发源码已移除，代码和脚本统一使用以下新路径。

```text
src/supermimic/
  core/
    state.py                 StateLayout、命名状态视图、奖励输入
    rewards.py               奖励项配置、加权聚合、分项结果
    history.py               本体观测历史，训练与部署共用
  algorithms/
    networks.py              公共 MLP、目标编码/动作解码、动力学、价值与终止网络
    objective.py             BPTT 折扣回报、可选价值自举、GAE
    replay_buffer.py         按环境与时间排列的回放缓冲区
    models.py                残差 Actor、残差世界模型
  robots/
    g1/                      G1 状态布局与可微运动学
    prima1/config.py         Prima1 关节顺序、状态布局、动作约束
  backends/mujoco_cpu.py      CPU MuJoCo reset/forward/step
  tasks/
    registry.py              惰性任务注册与入口选择
    motion_tracking/         G1/mjlab 跟踪任务及其训练适配
      reward_cfg.py          仿真与想象共用的奖励参数
      envs/                  环境配置、参考运动、MDP
      world_model/           G1 显式积分与解析 heads
      policy/nets.py         跟踪任务的网络参数、缩放和组装（网络实现在公共层）
      imagination/           G1 想象状态/观测适配
      training/              G1 并行采集与训练调度
    pen_spin/                Prima1 单片段转笔任务
      config.py             实验 TrainConfig 与 CLI 参数
      reward_cfg.py         完整六项奖励配置
      envs/                 环境配置、reset/step、MDP
      data/                 片段验收、参考张量、真实转移采集
      policy/nets.py        本体感知 Actor 组装
      world_model/          残差 WM、监督更新与多步验证
      imagination/          可微 rollout 与 Actor 更新
      training/             主调度与检查点配置校验
      eval/benchmark.py     单片段真实仿真评估
      deployment/           策略导出与离线控制器
      train.py              train/evaluate/describe 入口
  scripts/run.py             统一任务入口
```

`core/` 和 `algorithms/` 不导入机器人、任务或仿真器。依赖方向由架构测试检查。
旋转工具、分布式工具、学习率调度和 mmap IO 继续保留在原来的公共模块。

## 公共网络与机器人配置

扩展单位是完整 task 包：复制 `tasks/motion_tracking/` 并重命名，修改机器人引用、
网络输入输出与 MDP，然后注册新的任务入口。任务内部使用相对导入，避免复制后仍调用原任务。
具体步骤见 README 的“新增任务：复制任务包”。不需要为每个机器人复制公共神经网络。

两个任务现在按相同职责分包；阅读对照、修改入口和扩展步骤见
[任务开发指南](TASK_DEVELOPMENT.md)。两者都有并行采集、学习终止和critic，但状态、
观测、参考与物理接口按任务独立实现。

task 下的 `world_model/` 和 `imagination/` 可以保存状态语义和任务适配，目录存在本身不是耦合问题。
但当前其中也包含多步损失、rollout 和训练循环的具体实现；公共层目前共享网络和回报算子，
不能视为已经完成通用 WM / imagination 训练引擎的抽取。改变状态结构还需修改编码、积分、
仿真读写和解析 heads，不能仅修改网络维度。

公共网络实现位于 `algorithms/networks.py` 和 `algorithms/models.py`，不导入机器人或任务。
`robots/` 保存机器人状态布局、运动学和约束，不定义神经网络。两个任务不能互相导入，
由 `tests/test_shared_networks.py` 检查这一边界。

一个世界模型包含三部分：状态编码 → 网络预测 → 状态更新。
G1 的浮动基座、Prima1 的固定手加自由物体，需要不同的状态布局、坐标系处理和积分方法；
任务还可能添加物体、接触或其他状态，因此世界模型表示由机器人与任务共同确定。
这些差别通过状态适配器以及显式输入/输出维度表达，不要求每个机器人复制 MLP。

- 公共网络：层结构、前向计算、初始化方式；尺寸由构造参数提供。
- 机器人/状态适配：关节数、状态编码、坐标转换、归一化、四元数与积分规则。
- 任务/实验配置：选用哪类模型、隐藏层宽度/深度、参考输入、奖励与终止。

G1 的 `Policy`、`Critic` 任务类只负责按跟踪配置组装公共网络，前向实现继承公共组件。
G1 世界模型调用公共 `DynamicsMLP` 和 `TerminationHead`，再由自己的 `StateSpec` 解释输出。
Prima1使用公共残差Actor；WM使用公共DynamicsMLP和TerminationHead，按自由笔状态显式积分。
共享实现不表示共享权重；各任务仍独立训练，跨机器人权重共享需要额外的统一表示设计。

## 使用方式

```bash
sm-run --list
sm-run prima1_pen_spin describe --stage short
sm-run prima1_pen_spin train --stage hold --output runs/hand/example
sm-run g1_tracking --help
```

没有重新安装项目时，使用 `python -m supermimic.scripts.run` 替代 `sm-run`。
安装后只提供 `sm-run` 命令入口；G1 与 Prima1 均通过任务名选择。
任务注册只保存模块路径，列出任务不会加载 mjlab；运行 Hand 不会导入 G1 训练栈。

## 奖励和状态的边界

`StateLayout` 负责张量字段的位置。任务函数读取 `data.current("joint_pos")`、
`data.desired("object_pos")` 等命名字段，不写固定的 `22:44` 或 `44:47`。
同一组转笔奖励已通过“3 个关节、物体字段排在关节之前”的测试。

`core.RewardInputs` 接受显式 layout；转笔模块内的 `RewardInputs` 提供 Prima1 默认值，
支持省略默认布局的四参数构造方式。新增机器人应显式提供自己的布局。

奖励管理器只调用函数、验证每个样本返回一个标量、乘权重及可选 dt，并返回总值和分项。
`params` 可以包含字符串、布尔值、零或负数；误差尺度必须为正这类约束由具体奖励项验证。
管理器和状态视图都保留梯度，真实仿真与想象轨迹使用同一份转笔奖励公式。

G1 的 mjlab 奖励函数和张量奖励函数仍是两个后端实现，参数统一来自 `TrackingRewardCfg`。
训练入口将 `TrainConfig.rewards` 同时交给仿真配置和解析 heads。
历史 `joint_limit_weight` 覆盖参数继续支持；使用自定义环境手动构造 heads 时，
将 `resolved_reward_cfg(env_args)` 传给 heads。G1 奖励 dt 必须匹配 0.02 秒控制周期。

## 保留的训练语义

- Prima1 六项奖励、权重、容差保持独立；v4 统一乘 dt，并使用未来10帧关节参考。
- Actor 参数名称、张量形状、动作限幅以及导出接口保持兼容。
- 奖励配置记录稳定的函数身份。此次转笔目录迁移保留迁移前的奖励身份字符串，
  实际函数在 `envs/mdp/rewards.py`，字符串不用于动态导入。已有带任务配置的检查点可以恢复；
  更早未记录任务配置的 Hand 检查点仍需用 `--initialize` 开启新实验。
- 终止转移仍计入当前奖励；存活权重抑制其后的奖励。两项任务共用折扣回报算子。
- G1与Hand均使用显式状态积分、学习终止头、critic/GAE和想象末端价值自举。
  两者的采集器、训练调度和状态适配保留在任务层，按需要调用共享组件。
- Prima1 数据加载继续校验既有人工验收资产与片段的哈希。增加机器人并不自动放开数据审核。

## 指标含义

`tracking_failed` 表示笔中心偏离参考超过阈值。`dropped` 保留为旧字段别名，
不应解释成基于接触的真实掉笔检测。
`completed_without_drop` 是旧的“到达时间上限且未触发距离阈值”指标；
新增 `completion_criterion` 明确其含义。
`physical_drop_detected` 和 `spin_success` 当前为 null，表示尚未建立对应判据，
不能把 null 当作成功或失败。此次结构调整没有引入未经验证的接触奖励或转笔成功率。

## 增加任务或机器人

1. 在 `robots/<name>/` 定义关节、动作约束和状态布局；仿真状态和参考必须遵循同一布局。
2. 在 `tasks/<task>/` 定义参考、奖励、终止、观测与训练适配。适用时复用共享网络、回放和目标。
3. 通过 `register_task(TaskSpec(...))` 添加独立名字和惰性入口。
4. 验证奖励数值及梯度、终止与截断、Actor 信息边界、部署动作映射。

测试验证规范模块有独立实现，并确认旧转发源码和旧命令入口已移除。
`tests/test_hand_rewards.py` 冻结原六项公式；`tests/test_framework.py` 检查扩展能力与层间依赖。
此处首轮重构记录早于Warp升级；后续4096训练使用独立远端目录，原G1 checkout未改动。

## 2026-11-01 首轮重构验证记录（移除旧入口之前）

- 最终完整 pytest：42 passed、4 skipped；跳过项均为缺少 LAFAN1 原始 CSV 的转换测试。
  包含本机 G1 GPU 环境、解析奖励、世界模型与训练/恢复测试。
- Hand 六项奖励数值及 BPTT 梯度回归通过；新旧 Actor/WM 的 state_dict 严格加载、
  输出和 WM 梯度与迁移前 Git HEAD 实现逐项完全一致。
- 新共享层、Hand 实现和新增测试的 Ruff 检查通过；G1 迁移实现的 F/E9 检查通过；
  `git diff --check` 通过。
- `runs/hand/framework-smoke-20261101`：新统一入口训练一轮，再用旧 Hand 入口续训一轮，
  每轮采集 32 条转移、2 次 WM 更新、1 次 Actor 更新，完成策略导出。
  `runs/hand/framework-eval-20261101` 独立评估结果与训练末尾评估完全相同。
- 上述短实验在 0.36 秒触发距离阈值；它验证框架链路，不代表转笔成功或性能提升。
- 完整测试日志：`runs/framework-final-tests.log`。这些运行产物保持 Git 忽略。

## 移除旧入口及公共网络抽取验证

- 移除兼容源码、更新全部导入及命令后：43 passed、4 skipped，日志为 `runs/no-compat-tests.log`。
- 抽取 G1 公共网络后：45 passed、4 skipped、1 deselected，日志为 `runs/shared-networks-tests.log`。
  4 个跳过项仍为缺少原始 CSV；本轮未重复运行完整编译训练测试，改用下述短训练验证组装链路。
- Policy、Critic、DynamicsMLP、TerminationHead 的 state_dict 严格加载、输出和输入梯度与抽取前完全一致。
- 本机 G1 GPU 完成 8 次迭代、5 轮更新，并成功恢复检查点；该短训练关闭 torch.compile。
  日志为 `runs/shared-g1-train.log`。新入口的 Hand 短训练也已通过。
- 源码 F/E9、新架构测试的 Ruff 检查以及 `git diff --check` 通过。
  对整个 scripts/tests 的额外检查仍报告 6 处既有未使用局部变量，未在本次导入迁移中改动。
- 所有修改与测试只在本地仓库和本机 WSL 执行，没有同步或修改远端 5090。

## 本地待手动删除的缓存目录

以下旧目录的 Python 源码已经移除，仅余缓存或空子目录。工具自动审批拒绝了缓存目录删除命令，
返回 `blocked by policy`，未给出具体原因；可在本机资源管理器中删除以下完整目录：

- `D:\Prima1\supermimic-hand\src\supermimic\hand`
- `D:\Prima1\supermimic-hand\src\supermimic\envs`
- `D:\Prima1\supermimic-hand\src\supermimic\eval`
- `D:\Prima1\supermimic-hand\src\supermimic\imagination`
- `D:\Prima1\supermimic-hand\src\supermimic\policy`
- `D:\Prima1\supermimic-hand\src\supermimic\world_model`

`data`、`training`、`kinematics` 中仍有公共工具，需保留。
