from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from rag.api.deps import get_llm, get_store
from rag.api.main import create_app


def _client(qdrant_ready: bool, ollama_ready: bool) -> TestClient:
    app = create_app()
    store = MagicMock()
    store.is_ready.return_value = qdrant_ready
    llm = MagicMock()
    llm.is_ready.return_value = ollama_ready
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def test_health_is_ok_when_all_dependencies_are_ready() -> None:
    response = _client(True, True).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"] == {"qdrant": True, "ollama": True}


def test_health_is_degraded_but_still_200_when_a_dependency_is_down() -> None:
    response = _client(False, True).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_reports_each_dependency_independently() -> None:
    body = _client(True, False).get("/health").json()
    assert body["dependencies"] == {"qdrant": True, "ollama": False}
    assert body["status"] == "degraded"


def test_health_reports_the_package_version() -> None:
    import rag

    assert _client(True, True).get("/health").json()["version"] == rag.__version__
