import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "raw" / "bioasq" / "corpus.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "indexes" / "bm25_c0.pkl")
    args = parser.parse_args()
    index = BM25Index.build(args.corpus)
    index.save(args.output)
    print(f"wrote {args.output} with {len(index.documents)} documents")


if __name__ == "__main__":
    main()
