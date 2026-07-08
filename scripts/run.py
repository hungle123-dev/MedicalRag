"""Run one or more experiment configs.

    python scripts/run.py configs/e0.yaml configs/e1.yaml configs/e2.yaml

Writes experiments/<date>_<arm>/results.json per config.
"""
import sys
import datetime
import os

from medgraphrag.config import load_config
from medgraphrag.runner import run_config, save_results


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: run.py <config.yaml> [config.yaml ...]")
        return 2
    date = datetime.date.today().isoformat()
    for cfg_path in argv:
        cfg = load_config(cfg_path)
        res = run_config(cfg)
        out_dir = os.path.join("experiments", f"{date}_{res.arm_name}")
        out = os.path.join(out_dir, "results.json")
        save_results([res], out, config=cfg)
        print(f"{res.arm_name}\t{res.dataset}\tacc={res.accuracy:.3f}\t-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
