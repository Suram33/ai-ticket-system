"""Missing data and local model failures are handled gracefully."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ml.predictor import SolutionPredictor

from .conftest import ROOT, FakeLLM, make_services


def test_missing_required_ticket_fields(client):
    r = client.post("/api/tickets", json={"subject": "Hi"})
    assert r.status_code == 422
    assert any(e["loc"][-1] == "customer_name" for e in r.json()["detail"])
    assert client.post("/api/tickets", json={"customer_name": "   ", "subject": "x", "custom_fields": {"contact_channel": "Email"}}).status_code == 422
    assert client.post("/api/tickets", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_ticket_without_description_still_works(client):
    t = client.post("/api/tickets", json={"customer_name": "A B", "subject": "Forgot password", "description": None,
                                          "custom_fields": {"contact_channel": "Email"}})
    assert t.status_code == 201
    assert t.json()["description"] == ""
    assert t.json()["predicted_solution"] == "Reset the password using the account recovery process."


def test_llm_down_falls_back_to_grounded_template(tmp_path, trained_model, ticket_payload):
    services = make_services(tmp_path, trained_model, FakeLLM(fail=True))
    with TestClient(create_app(Settings(), services=services)) as c:
        r = c.post("/api/tickets", json=ticket_payload)
        assert r.status_code == 201
        t = r.json()
        assert t["response_source"] == "fallback"
        assert "Local LLM unavailable" in t["generation_error"]
        assert "Forgot Password option" in t["ai_response"]   # still grounded in the KB
        assert t["ai_response"].startswith("Hi Jane,")
        assert c.get("/api/health").json()["llm"]["reachable"] is False


def test_real_ollama_client_unreachable_is_handled(tmp_path, trained_model, ticket_payload):
    from app.rag.llm import OllamaLLM

    llm = OllamaLLM("http://127.0.0.1:9", "llama3.2:3b", timeout=2)  # nothing listens on port 9
    services = make_services(tmp_path, trained_model, llm)
    with TestClient(create_app(Settings(), services=services)) as c:
        t = c.post("/api/tickets", json=ticket_payload).json()
    assert t["response_source"] == "fallback"
    assert "Cannot connect to Ollama" in t["generation_error"]


def test_ml_model_missing_ticket_still_saved(tmp_path, trained_model, ticket_payload):
    services = make_services(tmp_path, trained_model, FakeLLM())
    broken = SolutionPredictor(tmp_path / "nope.joblib", tmp_path / "missing.csv").load()
    assert broken.available is False
    services["predictor"] = broken
    services["tickets"].predictor = broken
    with TestClient(create_app(Settings(), services=services)) as c:
        r = c.post("/api/tickets", json=ticket_payload)
        assert r.status_code == 201
        t = r.json()
        assert t["predicted_solution"] is None
        assert "ML model unavailable" in t["prediction_error"]
        assert t["ai_response"]  # RAG still runs on the ticket text alone
        assert c.post("/api/predict", json={"subject": "x y z", "description": ""}).status_code == 503
        assert c.get("/api/health").json()["ml"]["available"] is False


def test_corrupt_model_file_is_retrained_or_reported(tmp_path, ticket_payload):
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"not a model")
    p = SolutionPredictor(bad, ROOT / "data" / "sample_tickets.csv").load()
    assert p.available is False and "ML model unavailable" in p.error
    p.retrain()  # recovery path (also exposed as POST /api/ml/retrain)
    assert p.available is True


def test_empty_knowledge_base(tmp_path, trained_model, ticket_payload):
    empty = tmp_path / "empty_kb"
    empty.mkdir()
    services = make_services(tmp_path, trained_model, llm=None, kb_dir=empty)
    with TestClient(create_app(Settings(), services=services)) as c:
        t = c.post("/api/tickets", json=ticket_payload).json()
    assert t["rag_sources"] == []
    assert "support specialist will review" in t["ai_response"]


def test_retrieval_crash_does_not_lose_ticket(tmp_path, trained_model, ticket_payload):
    services = make_services(tmp_path, trained_model, FakeLLM())

    def boom(*a, **k):
        raise RuntimeError("index corrupted")

    services["kb"].search = boom
    with TestClient(create_app(Settings(), services=services)) as c:
        r = c.post("/api/tickets", json=ticket_payload)
        assert r.status_code == 201
        assert "Response generation failed" in r.json()["generation_error"]
        assert r.json()["predicted_solution"]
