import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.graph import PrimeKGIndex


class PrimeKGIndexTest(unittest.TestCase):
    def test_links_and_retrieves_path(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.db"
            connection = sqlite3.connect(database)
            connection.executescript("""
                CREATE TABLE nodes (id INTEGER PRIMARY KEY, external_id TEXT, type TEXT, name TEXT, name_norm TEXT, source TEXT);
                CREATE TABLE edges (relation TEXT, display_relation TEXT, x INTEGER, y INTEGER);
                INSERT INTO nodes VALUES (1,'a','drug','Propranolol','propranolol','test'),(2,'b','disease','Asthma','asthma','test'),
                  (3,'c','biological_process','metabolic process','metabolic process','test'),
                  (4,'d','biological_process','polyprenol metabolic process','polyprenol metabolic process','test');
                INSERT INTO edges VALUES ('contraindication','contraindication',1,2),
                  ('a','associated with',1,3),('b','associated with',3,4),('c','associated with',4,2);
            """)
            connection.close()
            index = PrimeKGIndex(database)
            seeds = index.link("Can propranolol worsen asthma?")
            self.assertEqual({seed["name"] for seed in seeds}, {"Propranolol", "Asthma"})
            self.assertTrue(index.paths(seeds))
            self.assertTrue(any(path["hop_count"] == 3 for path in index.paths(seeds, limit=20, max_hops=3)))
            nested = index.link("How does polyprenol metabolic process work?")
            self.assertEqual([seed["name"] for seed in nested], ["polyprenol metabolic process"])


if __name__ == "__main__":
    unittest.main()
