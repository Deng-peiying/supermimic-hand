"""恢复实验前校验奖励、终止与阶段配置。"""


def check_task_config(checkpoint, env):
    """防止改了奖励后误当作同一实验续训；旧权重可用 initialize 开始新实验。"""
    if "task_config" not in checkpoint:
        raise ValueError("旧检查点未记录任务配置，不能验证同任务恢复；请使用 --initialize")
    expected = checkpoint["task_config"]
    if expected != env.cfg.describe():
        raise ValueError("任务奖励/终止配置不同，请用 --initialize 开始新实验")
