"""日志显示不把原始单步奖励或训练批次误差冒充G1评估指标。"""

from supermimic.tasks.pen_spin.training.logging import format_progress


def test_readable_progress_preserves_metric_meanings():
    row = {
        "iteration": 10000,
        "hours": 0.6939,
        "iterations_per_s": 4.023,
        "updates": 3984,
        "episodes": 5281,
        "train_reward_mean": 0.043723548407853495,
        "wm": {"nse1": 0.136631, "nseK": 0.415934},
        "policy": {"imag_return": 0.862901, "survival": 0.726338},
    }
    line = format_progress(row)
    assert "train_ret 0.353 (n=5281)" in line
    assert "nse8_fit 0.4159" in line
    assert "imag_ret 0.863 surv 0.726" in line
    assert "t sim/wm/pi" not in line  # 历史日志没有分段计时，不能伪造。
    assert row["train_reward_mean"] == 0.043723548407853495
