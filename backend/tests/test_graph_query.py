import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.primekgqa_gate import execute
from scripts.evaluate_primekgqa import norm


class GraphQueryTest(unittest.TestCase):
    def test_executes_node_and_relation_variables(self):
        outgoing = {("1", "associated with"): {"2", "3"}}
        incoming = {("associated with", "2"): {"1"}}
        between = {("1", "2"): {"associated with"}}
        self.assertEqual(execute([("1", "associated with", None)], outgoing, incoming, between), ("node", {"2", "3"}))
        self.assertEqual(execute([("1", None, "2")], outgoing, incoming, between), ("relation", {"associated with"}))
        self.assertEqual(norm("[is in protein-protein interaction with]"), "ppi")


if __name__ == "__main__":
    unittest.main()
