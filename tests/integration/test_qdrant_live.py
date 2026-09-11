import pytest

from rag.config import Settings
from rag.index.qdrant_store import QdrantStore


@pytest.mark.integration
def test_live_qdrant_is_reachable() -> None:
    store = QdrantStore(Settings())
    assert store.is_ready() is True
