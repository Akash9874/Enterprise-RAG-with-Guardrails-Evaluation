"""Generation provider.

Ollama-backed; the protocol keeps the rest of the codebase independent of it.
"""

from __future__ import annotations

from typing import Any, Protocol

import ollama
import structlog

from rag.config import Settings

log = structlog.get_logger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def is_ready(self) -> bool: ...


class OllamaClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else ollama.Client(host=settings.ollama.host)

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self._settings.models.generator,
            messages=messages,
            options={"temperature": self._settings.ollama.temperature},
        )
        return str(response["message"]["content"])

    def is_ready(self) -> bool:
        try:
            listed = self._client.list()
        except Exception as exc:  # noqa: BLE001 - readiness must never raise
            log.warning("ollama_not_ready", error=str(exc))
            return False
        names = {m.get("model", "") for m in listed.get("models", [])}
        return self._settings.models.generator in names
