"""Collection schema and payload mapping.

Named vectors let one collection hold both representations, so dense and sparse can be
fused server-side in a single round trip (PRD FR-R3).
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from rag.contracts import Chunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def vectors_config(size: int) -> dict[str, models.VectorParams]:
    return {DENSE_VECTOR: models.VectorParams(size=size, distance=models.Distance.COSINE)}


def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    # IDF modifier is what makes these behave as BM25 rather than raw term counts.
    return {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def point_id(chunk_id: str) -> str:
    """Qdrant needs a UUID or unsigned int. Derive one deterministically from chunk_id
    so upserts stay idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump()


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk.model_validate(payload)
