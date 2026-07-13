"""Record the runtime used to build local indexes without committing model files."""
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
manifest = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("torch", "transformers", "faiss-cpu", "rank-bm25")},
            "cuda": {"available": torch.cuda.is_available(), "torch_runtime": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None},
            "medcpt_c0_build": {"documents": 49513, "dimension": 768, "batch_size": 8,
                                "elapsed_seconds": 1459.1, "mean_documents_per_second": 34.4},
            "medcpt_c2_build": {"documents": 80061, "dimension": 768, "batch_size": 8,
                                "resumed_from_checkpoint": True, "final_segment_documents_per_second": 58.9}}
(ROOT / "data/manifests/runtime.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
