import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.generator import MockGenerator
import app.pipelines as pipeline_state


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.generator = pipeline_state._generator
        pipeline_state._generator = MockGenerator()

    def tearDown(self):
        pipeline_state._generator = self.generator

    def test_question_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "jobs.db", root / "artifacts")) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                self.assertEqual(client.get("/api/v1/health").json(), {"status": "ok"})
                pipelines = client.get("/api/v1/pipelines").json()["items"]
                self.assertEqual([item["id"] for item in pipelines], ["B0", "B1", "B2", "B3", "G1", "G2"])
                response = client.post("/api/v1/questions", json={"question": "What treats asthma?", "pipeline_id": "B0"})
                self.assertEqual(response.status_code, 202)
                job = client.get(f"/api/v1/questions/{response.json()['id']}").json()
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["result"]["details"]["pipeline"], "B0")
                self.assertEqual(job["result"]["evidence"], [])
                self.assertEqual(client.post("/api/v1/questions", json={"question": "valid question", "unexpected": 1}).status_code, 422)
                self.assertEqual(client.post("/api/v1/questions", json={"question": "x" * 2001}).status_code, 422)
                self.assertTrue((root / "artifacts" / f"{job['id']}.json").exists())

    def test_rejects_unknown_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "jobs.db", root / "artifacts")) as client:
                response = client.post("/api/v1/questions", json={"question": "A valid question", "pipeline_id": "X"})
                self.assertEqual(response.status_code, 422)

    def test_readiness_honors_runtime_index_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "MEDICAL_RAG_BM25_INDEX": str(root / "missing-bm25.pkl"),
                "MEDICAL_RAG_GRAPH_INDEX": str(root / "missing-graph.sqlite3"),
                "MEDICAL_RAG_MEDCPT_INDEX": str(root / "missing-medcpt"),
                "MEDICAL_RAG_GENERATOR": "mock",
            }
            with patch.dict("os.environ", environment), TestClient(
                    create_app(root / "jobs.db", root / "artifacts")) as client:
                response = client.get("/api/v1/ready")
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["pipelines"]["B1"])


if __name__ == "__main__":
    unittest.main()
