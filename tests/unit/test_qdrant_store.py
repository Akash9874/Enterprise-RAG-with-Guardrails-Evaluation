from unittest.mock import MagicMock

from rag.config import Settings
from rag.index.qdrant_store import QdrantStore


def test_is_ready_returns_false_when_the_client_raises() -> None:
    client = MagicMock()
    client.get_collections.side_effect = ConnectionError("refused")
    store = QdrantStore(Settings(), client=client)
    assert store.is_ready() is False


def test_is_ready_returns_true_when_the_client_responds() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    store = QdrantStore(Settings(), client=client)
    assert store.is_ready() is True


def test_collection_exists_matches_the_configured_name() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock(), MagicMock()])
    # MagicMock(name=...) sets the mock's repr, not an attribute -- set it explicitly.
    client.get_collections.return_value.collections[0].name = "other"
    client.get_collections.return_value.collections[1].name = "rag_corpus"
    store = QdrantStore(Settings(), client=client)
    assert store.collection_exists() is True


def test_collection_exists_is_false_when_absent() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock()])
    client.get_collections.return_value.collections[0].name = "something_else"
    store = QdrantStore(Settings(), client=client)
    assert store.collection_exists() is False
