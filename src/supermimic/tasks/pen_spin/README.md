# Prima1 左手转笔

当前为完整SuperMimic GPU并行训练，配置和验证边界见
[4096环境训练说明](../../../../docs/PRIMA1_SUPERMIMIC_4096.md)。

从 `config.py`、`reward_cfg.py`、`envs/pen_spin_env_cfg.py` 开始读，再看 `training/loop.py` 的调度。
Actor 输入只有本体历史、关节参考与内部相位；世界模型和奖励可以读取仿真笔真值。

- 训练：`sm-run prima1_pen_spin train --stage short --output runs/hand/example`
- 配置：`sm-run prima1_pen_spin describe --stage short`
- 评估：`sm-run prima1_pen_spin evaluate --stage short --checkpoint runs/hand/example/checkpoint.pt`

`data/clip.py` 只接受已有人工验收与哈希绑定的片段；目录拆分没有放开审核。
`deployment/` 是导出与离线控制器，不连接真机。
完整目录对照与扩展步骤见 [任务开发指南](../../../../docs/TASK_DEVELOPMENT.md)。
