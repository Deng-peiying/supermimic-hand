# Prima1 完整 SuperMimic：单片段、4096 环境

2026-11-01：转笔已从有限时域原型升级为完整模型式训练流程。
任务保持独立；公共网络、序列回放、折扣回报、GAE和学习率工具可复用，不导入G1任务。
这是Prima1任务适配，不是论文G1数值结果的复现。

## 先用一条数据

第一轮只用人工验收的 `round3_kinematic_clip001/candidate.npz`：369帧、50Hz、
368次控制转移约7.36秒。4096个环境使用同一片段，不表示有4096条不同示范。
轨迹没有首尾闭合证明，不能自动拼接为无限循环。

先验证单段能否在自由笔接触动力学中从首帧完成。如果还不能完成，优先检查模型误差、
动作可达性、接触与奖励。通过后再增加不同握持、速度和换指方式的数据，逐条完成
重定向、MuJoCo播放、穿模检查、人工验收。数据入口继续校验片段、资产和许可的哈希。

## 训练机制

| 机制 | 实现位置（相对任务根目录） |
|---|---|
| GPU自由笔接触仿真 | `envs/parallel_env.py` |
| 状态编码、显式积分、旋转误差 | `world_model/state_spec.py` |
| 学习动力学与终止头 | `world_model/world_model.py` |
| k步自回归状态损失与终止BCE | `world_model/wm_trainer.py` |
| 不跨reset的序列回放、传感器历史重建 | `data/replay.py` |
| 基于真实回报的区间采样 | `data/sampling.py` |
| Actor与非对称Critic | `policy/nets.py` |
| BPTT、末端价值自举、GAE拟合Critic | `imagination/` |
| UTD预算、余弦学习率、检查点 | `training/loop.py` |
| 独立CPU物理评估 | `eval/benchmark.py` |
| 仅Actor导出及离线控制器 | `deployment/` |

WM预测速度增量及位姿修正，按新速度显式积分；笔四元数使用乘法积分并归一化。
freejoint角速度采用机体系约定，已与MuJoCo `mj_integratePos`对照。
Actor目标为想象回报加末端价值自举，学习终止概率用于累乘存活权重。
Critic使用固定GAE目标更新。WM和Critic参数在Actor反传期间冻结，但状态梯度保持畅通。

物理失败转移保留当步奖励，之后不给奖励；参考末帧不作为物理失败标签训练终止头，
但在想象中准确停止后续奖励与自举，防止重复末帧获奖。hold阶段时间上限按截断处理。

## 信息边界

- Actor：30步历史，每步22个关节位置、位置差分速度、实际执行目标；未来10帧参考关节角（t+1到t+10，0.2秒）和内部相位。
  没有笔真值、接触真值或图像。
- WM：57维特权仿真状态、实际执行动作、摩擦和PD增益随机化上下文。
- Critic：特权状态、本体历史、参考和上下文，仅训练使用。
- 部署：只导出Actor、关节参考和接口说明，不导出WM、Critic、参考笔轨迹。

reset可以将笔放到参考姿态，速度归零；step只写手指控制，绝不覆盖自由笔状态。
默认return采样约20%回合从首帧启动，其余练习片段内区间；随机起点回报不能替代首帧整段评估。

## 第一轮配置

| 项目 | 值 |
|---|---:|
| 并行环境 | 4096 |
| 控制/物理频率 | 50Hz / 500Hz |
| 总环境迭代预算 | 320000 |
| 预热 | 40步 |
| 更新预算UTD | 每环境步0.4轮 |
| 每轮WM | 10 batches × 1024条8步窗口 |
| 每轮Actor | 1 batch × 512起点 × 32想象步 |
| 每轮Critic | 5 epochs × 4 minibatches |
| 折扣/GAE lambda | 0.99 / 0.95 |
| 学习率 | 2e-4，余弦衰减至3% |
| 回放 | 每环境256行，含终止专用行 |
| WM隐藏层 | 512 × 512 × 512 |
| 日志/检查点 | 每100 / 4000步 |

5090实际为24GB型号，环境数与训练batch独立配置。当前关闭Torch编译，Warp物理使用CUDA图。
动作探索噪声标准差0.02rad；摩擦倍率0.8–1.2，PD增益倍率0.9–1.1。
这些是初始实验范围，不表示真机已标定。

## GPU后端验证与限制

CPU/GPU无接触十步对照，关节最大差约2.4e-7rad。启用接触后，0.2秒处某关节约差0.14rad，
笔中心约差1.3mm，接触点数量也会分叉。因此训练后需分别报告GPU首帧结果和CPU交叉评估，
不能宣称长轨迹后端等价。

初次4096预检发现Warp默认EPA horizon容量24不足。`backends/warp_capacity.py`
在创建碰撞图前扩大实际工作区至128，不关闭告警、不改几何或碰撞公式、不修改依赖文件。
该适配使用锁定版本的内部容量接口，升级依赖必须重新验证。
每控制步检查overflow、接触数和约束数，容量不足直接停止训练。
修复后4096预检完成120步、32轮更新，无溢出。

`failure_count`表示笔偏离参考超过10cm；`reference_end_count`表示未触发距离失败且到达末帧。
都不是已经定义的真实转笔成功率。`gpu_peak_gb`只统计Torch，总显存需用`nvidia-smi`读取。

## 运行与恢复

```bash
python -m supermimic.scripts.run prima1_pen_spin train \
  --clip data/prima1/clip001 --asset assets/prima1_left \
  --num-envs 4096 --stage full --output runs/prima1_clip001_4096_20261101_v4
```

新检查点格式`prima1-supermimic-v4`保存Actor、WM、Critic、优化器、采样统计和数据身份。
resume恢复这些状态，重置仿真并重新预热回放，不是逐位确定性的原地续跑。
旧原型检查点不能恢复新WM/优化器，只能通过initialize加载兼容Actor开启新实验。
SIGINT/SIGTERM会在当前迭代结束后保存；带编号检查点和`checkpoint.pt`均原子写入。

```bash
python -m supermimic.scripts.run prima1_pen_spin evaluate \
  --clip data/prima1/clip001 --asset assets/prima1_left --stage full \
  --checkpoint runs/prima1_clip001_4096_20261101_v4/checkpoint.pt \
  --output runs/prima1_cpu_eval
```

CPU评估从首帧确定性执行。真机仍需实测SDK映射、反馈时序、增益/限幅和看门狗；本次是仿真训练。

## 本轮验证记录

- 本地回归：54 passed、4 skipped（缺原始CSV）、1 deselected（未改动的G1长编译训练）。
  日志`runs/hand-supermimic-final-tests.log`。
- 本地8环境完成完整更新、保存和恢复；恢复时Actor/WM/Critic及三个优化器均恢复。
- 5090修复后4096环境预检完成120步、32轮更新，约4.2 it/s，无碰撞/约束溢出。
  预检不用于正式初始化，正式实验从零开始。
- 独立远端源码目录：`/home/robot/supermimic-prima1-20261101`。
  源码、测试、说明、验收数据和资产按SHA256清单传输，清单与依赖版本记录在实验目录。
- 实验名：`prima1_clip001_4096_20261101_v4`。

查看实时日志：

```bash
ssh ubuntu5090 'tail -f /home/robot/supermimic-prima1-20261101/runs/prima1_clip001_4096_20261101_v4/train.log'
```

## 价值学习稳定性修正

v2在最初几百步出现critic负值自举漂移，已停止，不作为训练结果。v3从零开始。
保留WM/BPTT/GAE完整机制，critic采用零输出初始化；Actor末端自举及GAE下一状态价值使用EMA目标critic，tau=0.005，每次策略更新后更新一次。拟合损失按想象存活概率加权，目标网络保留输入梯度但冻结参数。新增目标值均值/绝对最大值日志；检查点包含target_critic，旧v2不允许resume。
这是一项转笔适配的稳定化调整，并非声称逐项复现原始G1超参数。

## v4 正式统一

奖励统一乘控制周期0.02；六项加权和的最高正值6.2对应单步奖励0.124。CPU仿真、Warp、想象、评估共用同一配置。原始分项不乘dt，加权分项乘dt一次。
Actor使用未来10帧手关节参考，末端重复阶段最后一帧，hold重复当前帧；不读取笔参考。残差加在窗口第一帧。部署包接口版本4，训练、评估、采集、离线控制共用窗口函数。v3旧权重不可resume或initialize。
本次对齐未来参考窗口和奖励时间缩放，保留30帧历史、Prima1残差Actor、特权Critic及EMA目标Critic，并非声称网络和G1逐项一致。
验证：未来窗口/末端/hold/未来帧梯度、Actor信息边界、导出推理一致性、奖励缩放、训练及恢复。接触对照整组首次1.01mm超1mm阈值，独立复测通过；未放宽阈值，不宣称后端接触等价。
