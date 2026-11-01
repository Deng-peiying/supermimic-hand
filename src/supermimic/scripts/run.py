"""Dispatch to a registered task: sm-run prima1_pen_spin describe."""

import sys

from supermimic.tasks.registry import get_task, list_tasks


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "--list"):
        print("Usage: sm-run TASK [task arguments]\nAvailable tasks:")
        for task in list_tasks():
            print(f"  {task.name}: robot={task.robot}, backend={task.backend}")
        return
    try:
        task = get_task(sys.argv[1])
    except ValueError as error:
        raise SystemExit(str(error)) from None
    argv = sys.argv
    try:
        sys.argv = [argv[0], *argv[2:]]
        task.load()()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    main()
