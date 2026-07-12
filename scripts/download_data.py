"""Downloads the real datasets this project uses (idempotent).

- MIRAGE benchmark JSON (7663 questions, 5 subtasks) from GitHub.
- MedRAG/textbooks chunked corpus (~200MB) from HuggingFace.

Run once: python scripts/download_data.py
"""
import subprocess
import sys
from pathlib import Path

MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MIRAGE_OUT = Path("data/raw/mirage_benchmark.json")
TEXTBOOKS_DIR = Path("data/raw/medrag_textbooks")


def download_mirage() -> None:
    if MIRAGE_OUT.exists():
        print(f"skip: {MIRAGE_OUT} already exists")
        return
    MIRAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sL", "--max-time", "60", MIRAGE_URL, "-o", str(MIRAGE_OUT)],
                    check=True)
    print(f"downloaded: {MIRAGE_OUT}")


def download_textbooks() -> None:
    if (TEXTBOOKS_DIR / "chunk").exists():
        print(f"skip: {TEXTBOOKS_DIR} already exists")
        return
    from huggingface_hub import snapshot_download
    snapshot_download("MedRAG/textbooks", repo_type="dataset",
                       allow_patterns=["chunk/*.jsonl"],
                       local_dir=str(TEXTBOOKS_DIR))
    print(f"downloaded: {TEXTBOOKS_DIR}")


if __name__ == "__main__":
    download_mirage()
    download_textbooks()
    sys.exit(0)
