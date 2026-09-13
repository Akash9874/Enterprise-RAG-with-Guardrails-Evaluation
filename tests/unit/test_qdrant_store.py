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


def _point(vector: list[float]) -> MagicMock:
    point = MagicMock()
    point.vector = {"dense": vector}
    return point


def test_dense_centroid_averages_the_indexed_vectors() -> None:
    """The topicality rail compares a query against this (ADR-015)."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock()])
    client.get_collections.return_value.collections[0].name = Settings().qdrant.collection
    client.scroll.side_effect = [
        ([_point([1.0, 0.0]), _point([0.0, 1.0])], None),
    ]
    store = QdrantStore(Settings(), client=client)
    assert store.dense_centroid() == [0.5, 0.5]


def test_dense_centroid_pages_through_every_point() -> None:
    """A single scroll page would silently compute the centroid of the first batch only."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock()])
    client.get_collections.return_value.collections[0].name = Settings().qdrant.collection
    client.scroll.side_effect = [
        ([_point([1.0, 0.0])], "next-page"),
        ([_point([0.0, 1.0])], None),
    ]
    store = QdrantStore(Settings(), client=client)
    assert store.dense_centroid() == [0.5, 0.5]
    assert client.scroll.call_count == 2


def test_iter_payloads_pages_until_offset_is_none() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock()])
    client.get_collections.return_value.collections[0].name = Settings().qdrant.collection
    client.scroll.side_effect = [
        ([MagicMock(payload={"chunk_id": "a"})], "next"),
        ([MagicMock(payload={"chunk_id": "b"})], None),
    ]
    store = QdrantStore(Settings(), client=client)

    assert store.chunk_ids() == {"a", "b"}
    assert client.scroll.call_args.kwargs["with_payload"] == ["chunk_id"]
    assert client.scroll.call_args.kwargs["with_vectors"] is False


def test_corpus_commits_are_the_distinct_stamps_in_the_index() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[MagicMock()])
    client.get_collections.return_value.collections[0].name = Settings().qdrant.collection
    client.scroll.side_effect = [
        (
            [
                MagicMock(payload={"corpus_commit": "abc1234"}),
                MagicMock(payload={"corpus_commit": "abc1234"}),
                MagicMock(payload={}),  # a chunk indexed before commits were stamped
            ],
            None,
        ),
    ]
    store = QdrantStore(Settings(), client=client)

    assert store.corpus_commits() == {"abc1234", "unknown"}
    assert client.scroll.call_args.kwargs["with_payload"] == ["corpus_commit"]


def test_iter_payloads_of_an_absent_collection_is_empty() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    assert list(QdrantStore(Settings(), client=client).iter_payloads()) == []
    client.scroll.assert_not_called()


def test_dense_centroid_of_an_absent_collection_is_none() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    assert QdrantStore(Settings(), client=client).dense_centroid() is None
