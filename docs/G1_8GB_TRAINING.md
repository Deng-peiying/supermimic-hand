# G1 全量 BONES-SEED：8 GB 显卡训练

本地使用 RTX 4060 Laptop 8 GB。保留按仓库 e8 规则筛选后的全部 129,491 条动作，不额外筛选或抽样。
当前清单 SHA256：`11cfa51ffb16fa20ee08a41b78e848ac31c1913179b7c64f270c7e62483a716d`。

## 内存映射动作库

全量参考状态为 47,345,789 × 71 个 FP16 数，约 6.72 GB。
原实现将全部参考状态放到 GPU；只减 batch 无法解决这部分固定显存需求。
新增 `.json` + `.npy` 入口：CPU/操作系统文件缓存保管参考状态，仅把本轮需要的帧传入 GPU。
片段索引和采样统计仍在 GPU，所有动作均参与原来的采样机制。
原 NPZ 和 `.pt` GPU 加载路径保持不变；不修改奖励、观测、参考帧精度或训练算法。
CPU 到 GPU 的同步传输有速度代价。

生成缓存（Ubuntu-22.04）：

```bash
cd /mnt/d/Prima1/supermimic-hand
/root/.venvs/supermimic-hand/bin/python src/supermimic/data/compact_cache.py \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --out /root/.cache/supermimic/bones_seed_filtered_v1 --workers 8
```

构建器限制在途文件数量，逐批写入，不创建全量 FP32 内存副本。
核对帧数、有限数及 FP16 溢出；完成后才发布 `library.json`，其中记录清单与参考文件 SHA256。
输出目录已有缓存时拒绝覆盖。首次构建日志在 `runs/g1_full/build.log`。

缓存保存在 WSL 的 Linux 文件系统，避免训练时逐个读取 Windows 挂载盘上的十几万个 NPZ。
原始转换数据继续保留在 D 盘。

## 正式训练配置

| 参数 | 本次设置 |
|---|---:|
| 数据 | 全部 129,491 条、263.03 小时 |
| 并行环境 | 128 |
| 世界模型 batch | 128 |
| 策略 batch | 128 |
| 每次 WM 更新批数 | 10（仓库默认） |
| Replay 容量 | 1000（仓库默认） |
| 想象窗口 | 32（仓库默认） |
| 网络规模、学习率、噪声、域随机化、采样 | 仓库默认 |
| 总迭代计划 | 320,000（仓库默认） |
| 保存间隔 | 1,000 步 |
| 日志间隔 | 100 步 |

从随机初始化开始，不加载预训练权重，也不从之前的 64 步预检续训。
环境数和 batch 减小后，同样的迭代次数对应的采样量和梯度统计与 README 四卡结果不同，不能预设相同训练时间或成绩。

等价启动命令：

```bash
cd /mnt/d/Prima1/supermimic-hand
export PYTHONPATH=src OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
/root/.venvs/supermimic-hand/bin/python -u -m supermimic.scripts.run g1_tracking \
  --run-name g1_bones_full_4060_20261101 \
  --data.library /root/.cache/supermimic/bones_seed_filtered_v1/library.json \
  --env.num-envs 128 --env.replay-capacity 1000 \
  --wm-trainer.batch-size 128 --policy-trainer.batch-size 128 \
  --schedule.total-iterations 320000 --schedule.log-interval 100 \
  --schedule.checkpoint-interval 1000
```

运行中的任务不要重复启动。实际后台启动信息保存在：

- `logs/g1_bones_full_4060_20261101/launch.json`：PID、完整命令与数据校验摘要。
- `logs/g1_bones_full_4060_20261101/train.log`：训练输出。
- `logs/g1_bones_full_4060_20261101/ckpt_latest.pt`：首次达到 1000 步后生成。
- 同目录 `tb/`：TensorBoard 指标。

查看日志：

```bash
tail -f logs/g1_bones_full_4060_20261101/train.log
```

## 验证

`test_motion_library_mmap.py` 对比 CPU 映射与 GPU compact 路径：原始参考帧逐位一致，
目标特征和 FK 身体位置在 1e-6 浮点误差范围内一致；跨片段末帧截断、二维索引及文件完整性检查通过。
原 compact 对照测试也保留。

另用映射的单条 LAFAN1 做了独立 64 步训练预检，完成 10 次更新并保存 checkpoint。
该预检只检验执行链路，不作为正式训练数据集或学习成功证据。
