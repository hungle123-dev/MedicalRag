from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_and_ready_do_not_expose_secrets() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert "key" not in str(payload).casefold()
