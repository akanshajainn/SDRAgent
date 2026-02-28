from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


def test_e2e_mock_agent_run(tmp_path: Path) -> None:
    original_provider = settings.llm_provider
    original_db_path = settings.db_path
    original_allow_mock_llm = settings.allow_mock_llm

    object.__setattr__(settings, "llm_provider", "mock")
    object.__setattr__(settings, "allow_mock_llm", True)
    object.__setattr__(settings, "db_path", str(tmp_path / "e2e_mock.db"))

    try:
        app = create_app()
        client = TestClient(app)

        run = client.post("/run-agent", json={"domain": "apple.com"})
        assert run.status_code == 200
        payload = run.json()
        assert payload["domain"] == "apple.com"
        assert payload["company_name"]
        assert payload["research"]["summary"]
        assert payload["email"]["subject"]
        assert payload["evaluation"]["overall_score"] > 0

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["evaluations_last_7d"] >= 1

        crm = client.get("/crm/recent?limit=5")
        assert crm.status_code == 200
        assert len(crm.json()) >= 1

        regression = client.get("/eval-regression")
        assert regression.status_code == 200
        assert regression.json()["status"] in {"stable", "regressing"}
    finally:
        object.__setattr__(settings, "llm_provider", original_provider)
        object.__setattr__(settings, "allow_mock_llm", original_allow_mock_llm)
        object.__setattr__(settings, "db_path", original_db_path)
