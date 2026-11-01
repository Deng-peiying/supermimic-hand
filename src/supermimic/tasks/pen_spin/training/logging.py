"""可读终端摘要；完整原始指标仍保存为JSONL，不改变训练数值。"""


def format_progress(row, max_step_reward=0.124, wm_window=8):
    wm, pi = row.get("wm", {}), row.get("policy", {})
    nan = float("nan")
    normalized = row.get("train_reward_mean", nan) / max_step_reward if row["episodes"] else nan
    message = (
        f"it {row['iteration']} | {row['hours']:.2f} h | "
        f"{row['iterations_per_s']:.1f} it/s | updates {row['updates']} | "
        f"train_ret {normalized:.3f} (n={row['episodes']}) | "
        f"wm nse1 {wm.get('nse1', nan):.4f} nse{wm_window}_fit {wm.get('nseK', nan):.4f} | "
        f"imag_ret {pi.get('imag_return', nan):.3f} surv {pi.get('survival', nan):.3f}"
    )
    if "timing" in row:
        t = row["timing"]
        message += f" | t sim/wm/pi {t['sim']:.1f}/{t['wm']:.1f}/{t['policy']:.1f}s"
    return message
