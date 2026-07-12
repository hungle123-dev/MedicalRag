"""Loads .env (KEY=VALUE lines) into os.environ if present. ponytail: tiny
hand-rolled parser, no python-dotenv dependency for 2 keys."""
import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        # .env is the source of truth for THIS project — override any stale
        # shell-profile export (e.g. a different proxy URL).
        os.environ[key.strip()] = val.strip()
