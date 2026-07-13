from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(root: Path) -> None:
    """Load a simple project .env without overriding the process environment."""
    path = root / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
