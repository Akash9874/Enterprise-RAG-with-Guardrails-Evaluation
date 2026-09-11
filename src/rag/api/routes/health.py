from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

import rag
from rag.api.deps import get_llm, get_store
from rag.index.qdrant_store import QdrantStore
from rag.models.llm import OllamaClient

router = APIRouter()


@router.get("/health")
def health(
    store: Annotated[QdrantStore, Depends(get_store)],
    llm: Annotated[OllamaClient, Depends(get_llm)],
) -> dict[str, object]:
    """Liveness plus per-dependency readiness.

    Returns 200 even when degraded: a dependency being down is information, not a
    transport error. The `status` field carries the verdict.
    """
    dependencies = {"qdrant": store.is_ready(), "ollama": llm.is_ready()}
    status = "ok" if all(dependencies.values()) else "degraded"
    return {"status": status, "dependencies": dependencies, "version": rag.__version__}
