import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


class ApiTest(unittest.TestCase):
    def test_question_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "jobs.db", root / "artifacts")) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                pipelines = client.get("/api/v1/pipelines").json()["items"]
                self.assertEqual([item["id"] for item in pipelines], ["B0", "B1", "B2", "B3", "G1", "G2"])
                response = client.post("/api/v1/questions", json={"question": "What treats asthma?", "pipeline_id": "G2"})
                self.assertEqual(response.status_code, 202)
                job = client.get(f"/api/v1/questions/{response.json()['id']}").json()
                self.assertEqual(job["status"], "completed")
                self.assertIn("What treats asthma?", job["result"]["answer"])
                self.assertTrue((root / "artifacts" / f"{job['id']}.json").exists())

    def test_rejects_unknown_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root / "jobs.db", root / "artifacts")) as client:
                response = client.post("/api/v1/questions", json={"question": "A valid question", "pipeline_id": "X"})
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

