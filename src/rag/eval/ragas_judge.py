"""Ragas adapter for Tier C. Importable without the extra; constructing it needs it (ADR-027).

Only this file knows Ragas' API. If a Ragas release changes a signature, this is the one
place that moves — `judged.py` and its tests do not.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

from rag.config import Settings

MISSING_EXTRA = "Tier C needs the judge extra: `uv sync --extra judge` (ADR-027)."


class RagasJudge:
    def __init__(self, settings: Settings) -> None:
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import AnswerRelevancy, Faithfulness
        except ImportError as exc:
            raise RuntimeError(MISSING_EXTRA) from exc

        cfg = settings.judge
        model = cfg.model or settings.models.generator
        self.name = f"{model}@{urlparse(cfg.base_url).netloc}"
        client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env, "ollama")
        )
        llm = llm_factory(model, client=client)
        # Relevancy embeds generated questions; reuse the retrieval embedder locally, never an
        # API. `embedding_factory` is deprecated in ragas 0.4.3 — this is its named replacement.
        embeddings = HuggingFaceEmbeddings(
            model=settings.models.embedder, use_api=False, device="cpu"
        )
        self._faithfulness: Any = Faithfulness(llm=llm)
        self._relevancy: Any = AnswerRelevancy(llm=llm, embeddings=embeddings)

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        result = asyncio.run(
            self._faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
        )
        return float(result.value)

    def answer_relevancy(self, question: str, answer: str) -> float:
        result = asyncio.run(self._relevancy.ascore(user_input=question, response=answer))
        return float(result.value)
