import pytest

from rag.config import Settings
from rag.contracts import Chunk, make_chunk_id
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder


@pytest.fixture
def store() -> QdrantStore:
    settings = Settings.model_validate({"qdrant": {"collection": "test_upsert"}})
    s = QdrantStore(settings)
    s.ensure_collection(recreate=True)
    return s


@pytest.mark.integration
@pytest.mark.slow
def test_upsert_is_idempotent(store: QdrantStore) -> None:
    text = "Reciprocal Rank Fusion combines ranked lists."
    chunk = Chunk(
        chunk_id=make_chunk_id("a.md", text),
        doc_id="a.md",
        text=text,
        source_path="a.md",
        language="markdown",
    )
    embedder = Embedder(Settings())

    store.upsert_chunks([chunk], embedder)
    store.upsert_chunks([chunk], embedder)

    assert store.count() == 1
