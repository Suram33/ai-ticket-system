"""Shared fixtures. Every test gets its own temporary database and model file.

The LLM is replaced by a fake so tests are fast, deterministic and do not need
Ollama running. Retrieval uses the TF-IDF embedder (no downloads needed).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database import Database
from app.main import create_app
from app.ml.predictor import SolutionPredictor
from app.rag.embeddings import TfidfEmbedder
from app.rag.knowledge_base import KnowledgeBase
from app.rag.llm import LLMError
from app.rag.pipeline import ResponseGenerator
from app.tickets import TicketService

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Records the prompt it receives and answers with a canned or computed reply."""

    name = "fake:test-llm"

    def __init__(self, reply=None, fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []

    def status(self) -> dict:
        return {"reachable": not self.fail, "model_available": not self.fail, "detail": None}

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        if self.fail:
            raise LLMError("Cannot connect to Ollama at http://localhost:11434. Is `ollama serve` running?")
        if callable(self.reply):
            return self.reply(prompt)
        return self.reply or "We're sorry for the trouble. Please use the Forgot Password option on the login page.\n\nThe Support Team"


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    SolutionPredictor(path, ROOT / "data" / "sample_tickets.csv").load()  # trains + saves once
    return path


def make_services(tmp_path: Path, model_path: Path, llm=None, kb_dir: Path | None = None, seed_fields: bool = True) -> dict:
    db = Database(tmp_path / "test.db")
    db.init(seed_fields=seed_fields)
    predictor = SolutionPredictor(model_path, ROOT / "data" / "sample_tickets.csv").load()
    kb = KnowledgeBase(kb_dir or ROOT / "data" / "knowledge_base", TfidfEmbedder()).build()
    generator = ResponseGenerator(kb, llm, top_k=3, min_score=0.1)
    return {"db": db, "predictor": predictor, "kb": kb, "llm": llm, "generator": generator,
            "tickets": TicketService(db, predictor, generator)}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(tmp_path, trained_model, fake_llm) -> dict:
    return make_services(tmp_path, trained_model, fake_llm)


@pytest.fixture
def client(services):
    app = create_app(Settings(), services=services)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ticket_payload() -> dict:
    return {
        "customer_name": "Jane Doe",
        "subject": "Unable to log in",
        "description": "I forgot my password and cannot access my account.",
        "priority": "High",
        "custom_fields": {"contact_channel": "Email"},
    }
