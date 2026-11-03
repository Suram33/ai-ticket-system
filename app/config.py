"""Application settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    database_path: Path = field(default_factory=lambda: Path(_env("DATABASE_PATH", str(BASE_DIR / "tickets.db"))))
    training_data_path: Path = field(default_factory=lambda: Path(_env("TRAINING_DATA_PATH", str(BASE_DIR / "data" / "sample_tickets.csv"))))
    model_path: Path = field(default_factory=lambda: Path(_env("MODEL_PATH", str(BASE_DIR / "models" / "solution_model.joblib"))))
    kb_dir: Path = field(default_factory=lambda: Path(_env("KB_DIR", str(BASE_DIR / "data" / "knowledge_base"))))

    # ML
    ml_confidence_threshold: float = field(default_factory=lambda: float(_env("ML_CONFIDENCE_THRESHOLD", "0.20")))

    # RAG / embeddings: "auto" | "sentence-transformers" | "ollama" | "tfidf"
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ollama_embedding_model: str = field(default_factory=lambda: _env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "3")))
    rag_min_score: float = field(default_factory=lambda: float(_env("RAG_MIN_SCORE", "0.10")))

    # LLM (Ollama running locally)
    llm_enabled: bool = field(default_factory=lambda: _env("LLM_ENABLED", "true").lower() == "true")
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2:3b"))
    llm_timeout_seconds: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "120")))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))

    # Workflow
    auto_generate_on_create: bool = field(default_factory=lambda: _env("AUTO_GENERATE_ON_CREATE", "true").lower() == "true")
    seed_default_fields: bool = field(default_factory=lambda: _env("SEED_DEFAULT_FIELDS", "true").lower() == "true")
