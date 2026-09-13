"""Generation provider.

Ollama-backed; the protocol keeps the rest of the codebase independent of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import ollama
import structlog

from rag.config import Settings

log = structlog.get_logger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]: ...

    def is_ready(self) -> bool: ...


class OllamaClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        # Without an explicit timeout a hung generation blocks the request indefinitely,
        # which is a real risk for multi-second CPU inference.
        self._client = (
            client
            if client is not None
            else ollama.Client(host=settings.ollama.host, timeout=settings.ollama.timeout_s)
        )

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = self._messages(prompt, system)

        response = self._client.chat(
            model=self._settings.models.generator,
            messages=messages,
            options={"temperature": self._settings.ollama.temperature},
        )
        return str(response["message"]["content"])

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield content deltas.

        Streaming is not a nicety here: at 12-18 tok/s a 400-token answer takes ~30 s, and
        a first token in 1-2 s is what keeps that tolerable (ADR-002). Output rails run on
        the completed text, so callers must reassemble before enforcing anything.
        """
        stream = self._client.chat(
            model=self._settings.models.generator,
            messages=self._messages(prompt, system),
            options={"temperature": self._settings.ollama.temperature},
            stream=True,
        )
        for part in stream:
            content = part["message"]["content"]
            if content:
                yield str(content)

    def is_ready(self) -> bool:
        try:
            listed = self._client.list()
        except Exception as exc:  # noqa: BLE001 - readiness must never raise
            log.warning("ollama_not_ready", error=str(exc))
            return False
        names = {m.get("model", "") for m in listed.get("models", [])}
        return self._settings.models.generator in names
