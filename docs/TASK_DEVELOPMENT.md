# 独立任务的阅读与开发

当前转笔训练能力已升级，见 [完整SuperMimic转笔训练](PRIMA1_SUPERMIMIC_4096.md)。
“原来为什么差很多”和迁移前后数值对照记录描述此前那轮目录调整。

目标是让开发者在两个任务中按同一条路线定位代码：实验配置 → 环境和奖励 → 数据 → 网络 → 更新 → 评估。
G1 与 Prima1 分别拥有任务实现、配置、数据协议和检查点。任务之间禁止互相导入。
公共 `core/`、`algorithms/`、机器人描述及仿真后端是显式依赖，不共享任务配置或模型权重。

## 原来为什么差很多

`motion_tracking` 从完整 G1 复现迁移而来，已有并行采集、世界模型、critic、想象训练和评估模块。
`pen_spin` 从单片段实验原型开始，原 `train.py` 同时负责采集、监督更新、想象、评估、导出，
原 `networks.py` 混合了 Actor、WM、观测历史和奖励接口。它没有调用 G1 任务实现；
两者都使用公共网络层，因此“共享网络”不是目录不一致的必要原因。

本次将转笔按职责拆开，保留 G1 的成熟结构和训练行为。统一模块职责，不要求两个任务使用相同的算法或后端。

## 对照阅读

下表路径相对于各自的任务根目录。

| 职责 | motion_tracking | pen_spin |
|---|---|---|
| 启动入口 | `train.py` | `train.py` |
| 实验配置 | `config.py` | `config.py` |
| 奖励参数与权重 | `reward_cfg.py` | `reward_cfg.py` |
| 环境配置 | `envs/tracking_env_cfg.py` | `envs/pen_spin_env_cfg.py` |
| 仿真状态交互 | `envs/state_io.py`、mjlab 环境 | `envs/parallel_env.py`训练；`envs/env.py`独立CPU评估 |
| 观测/奖励/终止 | `envs/mdp/` | `envs/mdp/` |
| 数据转换/采集 | `data/` | `data/` |
| 策略网络组装 | `policy/nets.py` | `policy/nets.py` |
| 世界模型组装 | `world_model/world_model.py` | `world_model/world_model.py` |
| WM 监督更新 | `world_model/wm_trainer.py` | `world_model/wm_trainer.py` |
| 想象状态推进 | `imagination/rollout.py` | `imagination/rollout.py` |
| 策略梯度更新 | `imagination/policy_trainer.py` | `imagination/policy_trainer.py` |
| 训练主循环 | `training/loop.py` | `training/loop.py` |
| 真实仿真评估 | `eval/benchmark.py` | `eval/benchmark.py` |
| 实机接口 | 尚未实现 | `deployment/export.py`、`controller.py`，仅离线适配 |

两个任务的`world_model/state_spec.py`分别实现自己的物理状态编码与显式积分。
Prima1字段布局仍定义在`robots/prima1/config.py`。
G1 没有实机部署实现，所以不创建空的 `deployment/` 假装支持。

## 每层允许做什么

- `train.py` 解析参数并分派操作；`config.py` 保存实验默认值，不写优化循环。
- `envs/` 管理仿真交互和任务规则；`envs/mdp/` 明确策略能观察什么、奖励和终止怎么算。
- `data/` 负责数据身份、参考转换与采集；禁止把 reset 后的状态拼到上一回合末尾。
- `policy/`、`world_model/` 按任务语义组装公共网络；网络尺寸、状态含义和动作约束显式传入。
- `world_model/wm_trainer.py` 监督更新 WM；`imagination/` 冻结 WM 参数并沿模型反传到 Actor。
- `training/loop.py` 调度上述模块，负责实验日志与检查点；下层不反向导入它或 CLI。
- `eval/` 独立执行真实仿真评估，不依赖训练主循环；`deployment/` 只消费导出契约。

不强制两任务继承一个BaseTask：mjlab管理器环境与转笔直接Warp环境的协议仍然不同。
先统一定位方式和依赖方向，等出现稳定相同的实现再向公共层抽取，避免为消除文件差异而引入大量分支。

## 按修改需求找到入口

1. **新增奖励**：在本任务 `envs/mdp/rewards.py` 写公式，在 `reward_cfg.py` 显式注册权重与参数；
   验证数值、梯度、分项日志和仿真/想象一致性。G1 的两套后端公式需要同时维护。
2. **修改观测**：先检查 `envs/mdp/observations.py`，再改 `policy/nets.py`；Prima1 同时检查导出和离线控制器。
   笔的真值只允许 WM、奖励和评估读取，不加入 Actor 输入。
3. **修改 WM 或策略训练**：分别改 `world_model/wm_trainer.py` 与 `imagination/policy_trainer.py`；
   改更新先后顺序或采集预算才进入 `training/loop.py`。
4. **增加任务**：复制最接近的任务包，改包名、配置与机器人引用，再在 `tasks/registry.py` 注册独立名字。
   包内用相对导入；固定资产路径需随新任务显式配置，复制源码不会自动创造新数据集。
5. **增加机器人**：在 `robots/<name>/` 定义关节、状态布局与约束；逐一核对环境状态、参考、网络输出与部署映射。
   不能只改关节数量。若改了任务语义，使用新的奖励身份和新实验，不能伪装成原实验续训。

## 兼容性与当前能力

CLI 仍为 `sm-run g1_tracking ...` 和 `sm-run prima1_pen_spin train/evaluate/describe ...`。
删除了转笔原有混合职责文件，Python 调用者应使用上表中的新路径。
Actor/WM 的 state_dict、动作限幅、奖励权重、数据哈希验收和导出格式未改变。
转笔六项奖励的历史身份字符串继续写入检查点，源码位置可以移动；字符串不用于 import。
奖励配置值变更仍拒绝 `resume`，避免把两个目标不同的实验接在一起。

转笔只有一条已验收片段；现已支持Warp并行、学习终止和非对称Critic/GAE自举。
不增加未验收数据、真实硬件控制或未经验证的转笔成功判据。G1的5090源码和检查点未改动。

## 验证入口

`tests/test_task_template.py` 检查两个任务复制重命名后使用自己的实现，并约束依赖方向。
`tests/test_hand.py`、`test_hand_rewards.py` 检查本体信息边界、梯度、自由物体、数据身份和导出。
`tests/test_shared_networks.py`、`test_framework.py` 检查任务隔离、公共层边界和奖励配置。

2026-11-01 本地迁移对照：固定种子 42，单轮采集 32 步、WM 更新 2 次、Actor 更新 1 次，
batch 8、想象长度 3。迁移前后的 Actor/WM 所有参数、评估指标、逐帧轨迹、导出 schema 和
任务配置逐项完全一致。迁移前检查点使用新代码继续一轮成功；相关产物在
`runs/task-layout-before`、`runs/task-layout-after`、`runs/task-layout-resume`。
这些是结构回归检查，不构成转笔成功或训练收敛的证据。

完整 pytest：**50 passed、4 skipped**（350.49 秒，包含 G1 GPU 编译训练与恢复）。
4 个跳过项为缺少原始 LAFAN1 CSV 的转换测试。日志：`runs/task-layout-full-tests.log`。
本次修改涉及的转笔源码和测试通过 Ruff，`git diff --check` 通过。

旧 `pen_spin/mdp/` 的 Python 源码已删除；仅余 `__pycache__`。自动审批拒绝了该缓存目录
的删除命令（`blocked by policy`，无具体原因），因此文件树可能仍显示这个旧目录，
它不是新架构中的源码入口。
