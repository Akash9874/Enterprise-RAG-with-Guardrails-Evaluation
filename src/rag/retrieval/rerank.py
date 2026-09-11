"""Cross-encoder reranking.

ms-marco-MiniLM-L-6-v2 (22M) scores 30 pairs in ~70 ms on this CPU (measured, warm).
bge-reranker-large (560M) costs ~2 s for the same work — see ADR-003. Keep this
stage toggleable: the eval harness measures reranker lift by running with it on
and off over identical fused candidates.
"""

from __future__ import annotations

from typing import Any

from rag.config import Settings
from rag.contracts import Retrieved


class CrossEncoderReranker:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._settings.models.reranker, device="cpu")
        return self._model

    def rerank(self, query: str, items: list[Retrieved]) -> list[tuple[Retrieved, float]]:
        if not items:
            return []
        pairs = [(query, item.chunk.text) for item in items]
        scores = self.model.predict(pairs)
        paired = [(item, float(score)) for item, score in zip(items, scores, strict=True)]
        paired.sort(key=lambda pair: (-pair[1], pair[0].chunk.chunk_id))
        return paired
