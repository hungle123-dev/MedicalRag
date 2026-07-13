import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval import BM25Index


class BM25IndexTest(unittest.TestCase):
    def test_build_save_load_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                "\n".join(json.dumps(row) for row in [
                    {"id": "1", "title": "Asthma", "text": "Bronchoconstriction and beta blockers."},
                    {"id": "2", "title": "Diabetes", "text": "Insulin lowers glucose."},
                    {"id": "3", "title": "Cancer", "text": "Tumor treatment study."},
                    {"id": "4", "title": "Kidney", "text": "Renal function study."},
                ]), encoding="utf-8",
            )
            path = root / "index.pkl"
            BM25Index.build(corpus).save(path)
            self.assertEqual(BM25Index.load(path).search("asthma bronchoconstriction", 1)[0]["id"], "1")


if __name__ == "__main__":
    unittest.main()
