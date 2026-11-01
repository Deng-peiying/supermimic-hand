"""训练、评估、离线部署共同使用的本体历史；不读取笔状态。"""

from supermimic.core import history as proprioception

HISTORY = 30  # 50 Hz 下的 0.6 秒历史。


def append_history(history, q, previous_q, executed_target, dt=0.02):
    return proprioception.append_history(history, q, previous_q, executed_target, dt)


def initial_history(q):
    return proprioception.initial_history(q, HISTORY)
