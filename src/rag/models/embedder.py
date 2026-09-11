"""Dense (bge-small) and sparse (BM25) embedding.

bge models expect an instruction prefix on QUERIES ONLY. Applying it to documents, or
omitting it on queries, silently degrades recall — see tests.
"""

from __future__ import annotations

from typing import Any

from rag.config import Settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SPARSE_MODEL_NAME = "Qdrant/bm25"

SparseVec = tuple[list[int], list[float]]


class Embedder:
    def __init__(
        self,
        settings: Settings,
        dense_model: Any | None = None,
        sparse_model: Any | None = None,
    ) -> None:
        self._settings = settings
        self._dense = dense_model
        self._sparse = sparse_model

    @property
    def dense(self) -> Any:
        if self._dense is None:
            from sentence_transformers import SentenceTransformer

            self._dense = SentenceTransformer(self._settings.models.embedder, device="cpu")
        return self._dense

    @property
    def sparse(self) -> Any:
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        return self._sparse

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.dense.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.dense.encode(
            [f"{QUERY_PREFIX}{text}"], normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vectors[0]]

    def embed_sparse(self, texts: list[str]) -> list[SparseVec]:
        if not texts:
            return []
        return [
            ([int(i) for i in emb.indices], [float(v) for v in emb.values])
            for emb in self.sparse.embed(texts)
        ]
