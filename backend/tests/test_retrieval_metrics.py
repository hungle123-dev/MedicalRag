import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.evaluate_bm25 import question_metrics, sentence_chunks
from scripts.analyze_retrieval import paired_bootstrap


class RetrievalMetricsTest(unittest.TestCase):
    def test_metrics_and_sentence_chunks(self):
        metrics = question_metrics(["x", "gold", "y"], {"gold"})
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(len(sentence_chunks("One short sentence. Another sentence.", limit=3)), 2)
        bootstrap = paired_bootstrap([0, 1, 0], [1, 1, 1], seed=1, resamples=100)
        self.assertGreater(bootstrap["mean_delta_right_minus_left"], 0)

    def test_recall_is_fraction_of_all_gold_documents_not_hit_rate(self):
        metrics = question_metrics(["a", "noise"], {"a", "b"})
        self.assertEqual(metrics["recall_at_10"], 0.5)


if __name__ == "__main__":
    unittest.main()
