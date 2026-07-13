import tempfile
import unittest
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
                response = client.post("/api/v1/questions", json={"question": "What treats asthma?", "pipeline_id": "B1"})
                self.assertEqual(response.status_code, 202)
                job = client.get(f"/api/v1/questions/{response.json()['id']}").json()
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["result"]["details"]["pipeline"], "B1")
                self.assertTrue(job["result"]["evidence"])
                self.assertEqual(client.post("/api/v1/questions", json={"question": "valid question", "unexpected": 1}).status_code, 422)
                self.assertEqual(client.post("/api/v1/questions", json={"question": "x" * 2001}).status_code, 422)
                self.assertTrue((root / "artifacts" / f"{job['id']}.json").exists())

    def test_rejects_unknown_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "jobs.db", root / "artifacts")) as client:
                response = client.post("/api/v1/questions", json={"question": "A valid question", "pipeline_id": "X"})
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
