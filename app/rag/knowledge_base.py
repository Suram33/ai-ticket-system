"""Loads the local knowledge base, splits it into chunks and searches it."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    source: str     # file name
    title: str      # document title
    section: str    # "## " heading
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title} - {self.section}: {self.text}"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {**asdict(self.chunk), "score": round(self.score, 3)}


def chunk_markdown(path: Path) -> list[Chunk]:
    """One chunk per '## ' section. Small, focused chunks retrieve best."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    chunks: list[Chunk] = []
    if len(parts) == 1:  # no sections: whole file is one chunk
        body = re.sub(r"^# .+$", "", text, flags=re.M).strip()
        if body:
            chunks.append(Chunk(f"{path.stem}#0", path.name, title, title, " ".join(body.split())))
        return chunks
    for i in range(1, len(parts), 2):
        section, body = parts[i].strip(), " ".join(parts[i + 1].split())
        if body:
            chunks.append(Chunk(f"{path.stem}#{i // 2}", path.name, title, section, body))
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: Path, embedder: Embedder):
        self.kb_dir = Path(kb_dir)
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self) -> "KnowledgeBase":
        files = sorted(list(self.kb_dir.glob("*.md")) + list(self.kb_dir.glob("*.txt"))) if self.kb_dir.exists() else []
        self.chunks = [c for f in files for c in chunk_markdown(f)]
        if not self.chunks:
            log.warning("Knowledge base at %s is empty", self.kb_dir)
            self.matrix = None
            return self
        texts = [c.full_text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)
        log.info("Indexed %d knowledge-base chunks from %d files", len(self.chunks), len(files))
        return self

    @property
    def documents(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[SearchResult]:
        if self.matrix is None or not query.strip():
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q  # vectors are L2-normalised, so this is cosine similarity
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
