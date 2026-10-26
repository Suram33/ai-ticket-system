"""Client for a locally running LLM served by Ollama (https://ollama.com).

Ollama runs open-weight models (Llama 3.2, Mistral, Qwen, Phi ...) on the
local machine and exposes them on http://localhost:11434. No cloud API.
"""
from __future__ import annotations

import requests


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 120, temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def status(self) -> dict:
        """Is the server up and is the configured model pulled?"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            has_model = self.model in names or f"{self.model}:latest" in names
            return {"reachable": True, "model_available": has_model,
                    "detail": None if has_model else f"Run: ollama pull {self.model}"}
        except Exception as exc:
            return {"reachable": False, "model_available": False, "detail": f"Ollama not reachable at {self.base_url} ({exc.__class__.__name__})"}

    def generate(self, system: str, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(f"Cannot connect to Ollama at {self.base_url}. Is `ollama serve` running?") from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(f"Ollama did not answer within {self.timeout:.0f}s") from exc
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        content = (r.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise LLMError("Ollama returned an empty response")
        return content
