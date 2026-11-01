"""显示运行中实验的JSONL为可读单行；不触碰训练进程或修改原始日志。"""

import argparse
import json
import time
from pathlib import Path

from supermimic.tasks.pen_spin.training.logging import format_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((args.path.parent / "task_config.json").read_text())
    run = json.loads((args.path.parent / "config.json").read_text())
    provenance = json.loads((args.path.parent / "provenance.json").read_text())
    scale = provenance["dt"] if cfg["scale_rewards_by_dt"] else 1.0
    maximum = sum(max(0, term["weight"]) for term in cfg["rewards"].values()) * scale
    with args.path.open() as stream:
        while True:
            pos = stream.tell()
            line = stream.readline()
            if line.endswith("\n"):
                print(format_progress(json.loads(line), maximum, run["wm_window"]), flush=True)
            elif args.follow:
                stream.seek(pos)
                time.sleep(0.5)
            else:
                break


if __name__ == "__main__":
    main()
