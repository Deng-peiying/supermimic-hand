# G1 动作跟踪

从 `config.py`、`reward_cfg.py`、`envs/tracking_env_cfg.py` 开始读，再看 `training/loop.py` 的调度。
`world_model/` 负责监督学习动力学，`imagination/` 负责在模型内更新策略和 critic。

- 训练：`sm-run g1_tracking --help`
- 交互回放：仓库根目录 `scripts/play.py`
- 批量评估：仓库根目录 `scripts/eval_benchmark.py`

本任务使用 mjlab/MuJoCo Warp 并行环境，与 Prima1 单环境 CPU 后端不同。
任务不依赖 `pen_spin`，公共网络位于 `supermimic/algorithms`。
完整目录对照与扩展步骤见 [任务开发指南](../../../../docs/TASK_DEVELOPMENT.md)。
