from unittest.mock import MagicMock

import pytest

from rag.config import Settings
from rag.models.embedder import QUERY_PREFIX, Embedder


def _embedder() -> tuple[Embedder, MagicMock, MagicMock]:
    dense = MagicMock()
    dense.encode.return_value = [[0.1] * 384]
    sparse = MagicMock()
    sparse.embed.return_value = iter([MagicMock(indices=[1, 5], values=[0.7, 0.3])])
    return Embedder(Settings(), dense_model=dense, sparse_model=sparse), dense, sparse


def test_query_embedding_applies_the_bge_instruction_prefix() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_query("what is rrf")
    sent = dense.encode.call_args.args[0]
    assert sent == [f"{QUERY_PREFIX}what is rrf"]


def test_document_embedding_does_not_apply_the_prefix() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_documents(["rrf fuses ranked lists"])
    sent = dense.encode.call_args.args[0]
    assert sent == ["rrf fuses ranked lists"]
    assert QUERY_PREFIX not in sent[0]


def test_embeddings_are_normalised_for_cosine() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_documents(["text"])
    assert dense.encode.call_args.kwargs["normalize_embeddings"] is True


def test_sparse_embedding_returns_index_value_pairs() -> None:
    embedder, _, _ = _embedder()
    result = embedder.embed_sparse(["rrf fuses ranked lists"])
    assert result == [([1, 5], [0.7, 0.3])]


def test_embedding_an_empty_list_does_not_call_the_model() -> None:
    embedder, dense, _ = _embedder()
    assert embedder.embed_documents([]) == []
    dense.encode.assert_not_called()


@pytest.mark.slow
def test_real_embedder_produces_384_dimensions() -> None:
    embedder = Embedder(Settings())
    vector = embedder.embed_query("what is reciprocal rank fusion")
    assert len(vector) == 384
