"""Local embedding backends for knowledge-base retrieval.

All three options run on the local machine; nothing is sent to a cloud API.

* sentence-transformers  - a small neural embedding model (all-MiniLM-L6-v2,
                           ~90 MB, CPU friendly). Best semantic matching.
* ollama                 - embeddings from a local Ollama server
                           (e.g. `ollama pull nomic-embed-text`).
* tfidf                  - scikit-learn TF-IDF vectors. No download needed;
                           used automatically if the others are unavailable.

"auto" tries sentence-transformers, then Ollama, then TF-IDF.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.vectorizer.transform(texts).toarray())


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = f"sentence-transformers:{model_name.split('/')[-1]}"

    def fit(self, texts: list[str]) -> None:  # pre-trained, nothing to fit
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 30) -> None:
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self.embed(["health check"])  # fail fast if the server/model is missing

    def fit(self, texts: list[str]) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        r = requests.post(self.url, json={"model": self.model, "input": texts}, timeout=self.timeout)
        if r.status_code == 404:
            raise RuntimeError(f"embedding model '{self.model}' not pulled (optional: ollama pull {self.model})")
        r.raise_for_status()
        return _normalize(np.asarray(r.json()["embeddings"], dtype=float))


def create_embedder(backend: str, st_model: str, ollama_url: str, ollama_model: str) -> Embedder:
    backend = (backend or "auto").lower()
    attempts = {
        "sentence-transformers": lambda: SentenceTransformerEmbedder(st_model),
        "ollama": lambda: OllamaEmbedder(ollama_url, ollama_model),
        "tfidf": TfidfEmbedder,
    }
    order = ["sentence-transformers", "ollama", "tfidf"] if backend == "auto" else [backend, "tfidf"]
    for name in order:
        if name not in attempts:
            log.warning("Unknown embedding backend %r, skipping", name)
            continue
        try:
            emb = attempts[name]()
            log.info("Using embedding backend: %s", emb.name)
            return emb
        except Exception as exc:
            log.info("Embedding backend %s unavailable (%s)", name, exc)
    return TfidfEmbedder()
