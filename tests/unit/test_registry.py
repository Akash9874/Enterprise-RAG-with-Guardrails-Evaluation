from typing import Any

import pytest

from rag.models.registry import ModelRegistry


def test_loader_is_not_called_at_registration() -> None:
    calls: list[str] = []

    def loader() -> str:
        calls.append("a")
        return "model-a"

    registry = ModelRegistry(max_resident=2)
    registry.register("a", loader)
    assert calls == []


def test_loader_is_called_once_on_first_get() -> None:
    calls: list[str] = []

    def loader() -> str:
        calls.append("a")
        return "model-a"

    registry = ModelRegistry(max_resident=2)
    registry.register("a", loader)
    assert registry.get("a") == "model-a"
    assert registry.get("a") == "model-a"
    assert calls == ["a"]


def test_lru_evicts_the_least_recently_used_model() -> None:
    registry = ModelRegistry(max_resident=2)
    for name in ("a", "b", "c"):

        def loader(n: str = name) -> str:
            return f"model-{n}"

        registry.register(name, loader)

    registry.get("a")
    registry.get("b")
    registry.get("a")  # "a" is now more recent than "b"
    registry.get("c")  # exceeds max_resident -> evicts "b"

    assert set(registry.resident()) == {"a", "c"}


def test_evicted_model_is_reloaded_on_next_get() -> None:
    calls: list[str] = []

    def loader_a() -> str:
        calls.append("a")
        return "model-a"

    def loader_b() -> str:
        return "model-b"

    registry = ModelRegistry(max_resident=1)
    registry.register("a", loader_a)
    registry.register("b", loader_b)

    registry.get("a")
    registry.get("b")  # evicts "a"
    registry.get("a")  # reloads "a"

    assert calls == ["a", "a"]


def test_getting_an_unregistered_model_raises() -> None:
    registry = ModelRegistry(max_resident=2)
    with pytest.raises(KeyError, match="nope"):
        registry.get("nope")


def test_evict_all_clears_residency_but_keeps_loaders() -> None:
    registry = ModelRegistry(max_resident=2)

    def loader() -> Any:
        return "model-a"

    registry.register("a", loader)
    registry.get("a")
    registry.evict_all()
    assert registry.resident() == []
    assert registry.get("a") == "model-a"
