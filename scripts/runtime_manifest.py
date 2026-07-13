"""Record the runtime used to build local indexes without committing model files."""
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def build_state(name: str) -> dict | None:
    path = ROOT / "indexes" / name / "build_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


manifest = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("torch", "transformers", "faiss-cpu", "rank-bm25")},
            "cuda": {"available": torch.cuda.is_available(), "torch_runtime": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None},
            "medcpt_c0_build": build_state("medcpt"),
            "medcpt_c2_build": build_state("medcpt_c2")}
(ROOT / "data/manifests/runtime.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
